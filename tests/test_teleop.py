"""Teleoperation recovery through the pinned loop with simulated hardware."""

import contextlib
import io
import math
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from lerobot.scripts import lerobot_teleoperate as upstream
from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig
from lerobot_robot_ur5e.spacemouse import SpaceMouse
from lerobot_robot_ur5e.ur5e import CONNECTED, GRIPPER, JOINTS
from scripts import teleop as command


class TeleopTests(unittest.TestCase):
    @contextlib.contextmanager
    def session(self, home_at=None, duration=None, stop_after_command=True):
        with tempfile.TemporaryDirectory() as directory:
            mouse = SpaceMouse(
                SpaceMouseConfig(
                    axes="bounded",
                    neutral_rotvec=[0.0, 0.0, 0.0],
                    tilt_limit_deg=10,
                    stop_on_escape=True,
                    calibration_dir=Path(directory),
                )
            )
            tcp = np.array([0.0, 0.0, 0.3, math.radians(30), 0.0, 0.0])
            reads, homes, sent = [], [], []

            def get_state():
                reads.append(True)
                if len(reads) == home_at:
                    mouse._home_requested.set()
                return np.zeros(6), 0

            def home():
                homes.append(True)
                tcp[3:] = 0.0

            def send(action):
                sent.append(action)
                if stop_after_command:
                    mouse._stop_requested.set()
                return action

            mouse.hid = SimpleNamespace(get_state=get_state)
            robot = SimpleNamespace(
                name="ur5e",
                action_features={name: float for name in JOINTS + [GRIPPER]},
                measured_joints=lambda: np.zeros(6),
                rtde_r=SimpleNamespace(getActualTCPPose=lambda: tcp.copy()),
                gripper=SimpleNamespace(position=lambda: 0.0),
                go_home=home,
                get_observation=lambda: {},
                send_action=send,
                connect=Mock(),
                disconnect=Mock(),
            )
            cfg = upstream.TeleoperateConfig(
                teleop=SimpleNamespace(),
                robot=SimpleNamespace(),
                fps=10,
                teleop_time_s=duration,
            )
            clock = SimpleNamespace(now=0.0)

            def sleep(seconds):
                clock.now += seconds

            output = io.StringIO()
            with (
                patch.dict(CONNECTED, {mouse.config.robot_ip: robot}, clear=True),
                patch.object(mouse, "connect"),
                patch.object(mouse, "disconnect") as disconnect_mouse,
                patch.object(
                    upstream, "make_teleoperator_from_config", return_value=mouse
                ),
                patch.object(upstream, "make_robot_from_config", return_value=robot),
                patch.object(
                    upstream.time, "perf_counter", side_effect=lambda: clock.now
                ),
                patch.object(upstream.time, "sleep", side_effect=sleep),
                patch("lerobot.utils.cycle_timer.precise_sleep", side_effect=sleep),
                contextlib.redirect_stdout(output),
            ):
                yield SimpleNamespace(
                    cfg=cfg,
                    mouse=mouse,
                    robot=robot,
                    reads=reads,
                    homes=homes,
                    sent=sent,
                    clock=clock,
                    output=output,
                    disconnect_mouse=disconnect_mouse,
                )

    def test_outside_tilt_waits_for_home_and_resumes_without_changing_neutral(self):
        with self.session(home_at=3) as s:
            s.robot.gripper = SimpleNamespace(position=lambda: 1.0)  # closed before h
            s.mouse._gripper_cmd = 1.0
            command.run_teleoperation(s.cfg)
            self.assertEqual(s.homes, [True])
            self.assertEqual(len(s.sent), 1)
            # go_home opened it; teleop must not re-close.
            self.assertEqual(s.sent[0][GRIPPER], 0.0)
            self.assertEqual(s.output.getvalue().count("RECOVERY"), 1)
            self.assertIn("Press h", s.output.getvalue())
            self.assertIn("TELEOP ACTIVE", s.output.getvalue())
            self.assertEqual(s.mouse.config.neutral_rotvec, [0.0, 0.0, 0.0])
            s.robot.disconnect.assert_called_once()
            s.disconnect_mouse.assert_called_once()

    def test_duration_expires_during_recovery_without_sending_a_command(self):
        with self.session(duration=0.15, stop_after_command=False) as s:
            command.run_teleoperation(s.cfg)
            self.assertEqual(s.sent, [])
            self.assertAlmostEqual(s.clock.now, 0.15)
            s.robot.disconnect.assert_called_once()

    def test_escape_or_ctrl_c_during_recovery_disconnects_cleanly(self):
        for ctrl_c in (False, True):
            with self.subTest(ctrl_c=ctrl_c), self.session() as s:
                read = s.mouse.hid.get_state

                def get_state():
                    if s.reads:
                        if ctrl_c:
                            raise KeyboardInterrupt
                        s.mouse._stop_requested.set()
                    return read()

                s.mouse.hid.get_state = get_state
                command.run_teleoperation(s.cfg)
                self.assertEqual(s.sent, [])
                s.robot.disconnect.assert_called_once()
                s.disconnect_mouse.assert_called_once()

    def test_other_errors_propagate_and_upstream_hooks_are_restored(self):
        loop, init = upstream.teleop_loop, upstream.init_visualization
        with self.session() as s:
            s.mouse.hid.get_state = Mock(side_effect=RuntimeError("HID failed"))
            with self.assertRaisesRegex(RuntimeError, "HID failed"):
                command.run_teleoperation(s.cfg)
            s.robot.disconnect.assert_called_once()
            s.disconnect_mouse.assert_called_once()
        self.assertIs(upstream.teleop_loop, loop)
        self.assertIs(upstream.init_visualization, init)

    def test_optional_rerun_recovery_clears_after_homing_and_viewer_closes(self):
        with self.session(home_at=2) as s:
            s.cfg.display_data = True
            viewer = Mock()
            feedback = command.RecordingFeedback(s.cfg)
            rr = feedback.rr = Mock()
            feedback.viewer = viewer
            with (
                patch.object(command, "RecordingFeedback", return_value=feedback),
                patch.object(feedback, "init_viewer"),
                patch.object(upstream, "log_visualization_data"),
            ):
                command.run_teleoperation(s.cfg)
            texts = [call.args[0] for call in rr.TextDocument.call_args_list]
            phases = [text.splitlines()[0] for text in texts]
            recovery = next(i for i, phase in enumerate(phases) if "RECOVERY" in phase)
            self.assertTrue(
                any("TELEOP ACTIVE" in phase for phase in phases[recovery + 1 :])
            )
            self.assertFalse(any("Right" in text for text in texts))
            viewer.close.assert_called_once()

    def test_pixi_teleop_uses_project_entry_point_with_existing_controls(self):
        root = Path(__file__).resolve().parents[1]
        task = tomllib.loads((root / "pixi.toml").read_text())["tasks"]["teleop"]["cmd"]
        self.assertIn("python -m scripts.teleop", task)
        self.assertIn("--teleop.stop_on_escape=true", task)
        self.assertIn("--fps=$CONTROL_FPS", task)


if __name__ == "__main__":
    unittest.main()
