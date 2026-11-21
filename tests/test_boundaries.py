"""Five-face teaching and config persistence, with controller/HID/keyboard stand-ins."""

import contextlib
import io
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import set_boundaries as command


RAW = {
    "UR5E_Z_MIN_M": 0.05,
    "UR5E_Y_MAX_M": 0.4,
    "UR5E_Y_MIN_M": -0.3,
    "UR5E_X_MAX_M": 0.5,
    "UR5E_X_MIN_M": -0.4,
}


class BoundaryTests(unittest.TestCase):
    def test_clearance_is_inward_on_all_five_faces_and_invalid_boxes_are_rejected(self):
        actual = command.inset_bounds(RAW, 0.01)
        expected = {
            "UR5E_Z_MIN_M": 0.06,
            "UR5E_Y_MAX_M": 0.39,
            "UR5E_Y_MIN_M": -0.29,
            "UR5E_X_MAX_M": 0.49,
            "UR5E_X_MIN_M": -0.39,
        }
        self.assertEqual(set(actual), set(expected))  # No ceiling setting.
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key])
        # Captures swapped by base orientation (this station's +Y faces the operator) are sorted.
        swapped = {**RAW, "UR5E_Y_MIN_M": 0.4, "UR5E_Y_MAX_M": -0.3}
        self.assertEqual(
            command.inset_bounds(swapped, 0.01), command.inset_bounds(RAW, 0.01)
        )
        for margin, raw in [
            (-0.01, RAW),
            (float("nan"), RAW),
            (0.5, RAW),
            (0.01, {**RAW, "UR5E_X_MIN_M": 0.495}),
        ]:
            with self.subTest(margin=margin, raw=raw), self.assertRaises(ValueError):
                command.inset_bounds(raw, margin)

    @contextlib.contextmanager
    def teaching(self, answers=("", "y"), bad_pose=False, cancel=False):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config/local/machine.env"
            config.parent.mkdir(parents=True)
            original = (
                '# operator settings\nKEEP="untouched"\nUR5E_Z_MIN_M=0.2 # old floor\n'
            )
            config.write_text(original)
            config.chmod(0o600)
            key = object()
            callbacks = []
            stopped = []

            class Listener:
                def __init__(self, on_release):
                    callbacks.append(on_release)

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    stopped.append(True)

            robot = Mock()
            # Physical order: floor, left (-X), right (+X), front (-Y), back (+Y).
            # Other coordinates differ so a swapped capture axis cannot pass unnoticed.
            poses = [
                [0, 0, 0.05, 0, 0, 0],
                [-0.4, 0.23, 0.5, 0, 0, 0],
                [0.5, -0.14, 0.5, 0, 0, 0],
                [0.17, -0.3, 0.5, 0, 0, 0],
                [-0.26, 0.4, 0.5, 0, 0, 0],
            ]
            if bad_pose:
                poses[0][0] = float("nan")
            robot.rtde_r.getActualTCPPose.side_effect = poses
            robot.send_action.side_effect = lambda _: callbacks[0](key)
            mouse = Mock()
            mouse.get_action.side_effect = (
                [dict(), KeyboardInterrupt()] if cancel else None
            )
            keyboard = SimpleNamespace(
                Listener=Listener, Key=SimpleNamespace(enter=key)
            )
            env = {
                "PIXI_PROJECT_ROOT": directory,
                "UR5E_WORKSPACE_MARGIN_M": ".01",
                "BOUNDARY_LINEAR_SPEED_M_S": ".03",
                "BOUNDARY_ANGULAR_SPEED_RAD_S": ".15",
                "SPACEMOUSE_LINEAR_SPEED_M_S": ".2",
                "SPACEMOUSE_ANGULAR_SPEED_RAD_S": ".7",
                "SPACEMOUSE_AXES": "xyz_yaw",
                "CONTROL_FPS": "15",
            }
            with (
                patch.dict(os.environ, env),
                patch.object(sys, "argv", ["set-boundaries"]),
                patch.dict(sys.modules, {"pynput": SimpleNamespace(keyboard=keyboard)}),
                patch.object(command, "UR5e", return_value=robot) as robot_type,
                patch.object(command, "SpaceMouse", return_value=mouse) as mouse_type,
                patch("builtins.input", side_effect=answers),
                patch.object(command.time, "sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                yield SimpleNamespace(
                    config=config,
                    original=original,
                    robot=robot,
                    robot_type=robot_type,
                    mouse=mouse,
                    mouse_type=mouse_type,
                    stopped=stopped,
                )

    def test_complete_teaching_saves_five_faces_with_backup_and_preserves_private_settings(
        self,
    ):
        with self.teaching() as s:
            command.main()
            text = s.config.read_text()
            self.assertIn('KEEP="untouched"', text)
            values = dict(token.split("=", 1) for token in shlex.split(text, comments=True))
            for key, expected in {
                "UR5E_Z_MIN_M": 0.06,
                "UR5E_Y_MAX_M": 0.39,
                "UR5E_Y_MIN_M": -0.29,
                "UR5E_X_MAX_M": 0.49,
                "UR5E_X_MIN_M": -0.39,
            }.items():
                self.assertAlmostEqual(float(values[key]), expected, places=6)
            self.assertEqual(text.count("UR5E_Z_MIN_M="), 1)
            backups = list(s.config.parent.glob("machine.env.before-boundaries-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), s.original)
            self.assertEqual(s.config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(s.robot.rtde_r.getActualTCPPose.call_count, 5)
            s.mouse.disconnect.assert_called_once()
            s.robot.disconnect.assert_called_once()
            self.assertTrue(s.stopped)
            teaching = s.mouse_type.call_args.args[0]
            robot_cfg = s.robot_type.call_args.args[0]
            self.assertTrue(teaching.stop_on_escape)
            self.assertEqual(
                (
                    robot_cfg.x_min_m,
                    robot_cfg.x_max_m,
                    robot_cfg.y_min_m,
                    robot_cfg.y_max_m,
                    robot_cfg.z_min_m,
                ),
                (None,) * 5,
            )
            self.assertEqual(
                (teaching.max_linear_speed_m_s, teaching.max_angular_speed_rad_s),
                (0.03, 0.15),
            )

    def test_reject_or_cancel_keeps_config_unchanged_and_disconnects(self):
        for cancel in (False, True):
            with (
                self.subTest(cancel=cancel),
                self.teaching(answers=("", "n"), cancel=cancel) as s,
            ):
                command.main()
                self.assertEqual(s.config.read_text(), s.original)
                self.assertEqual(
                    list(s.config.parent.glob("machine.env.before-boundaries-*")), []
                )
                s.robot.disconnect.assert_called_once()
                s.mouse.disconnect.assert_called_once()

    def test_invalid_measurement_exits_without_saving_and_disconnects(self):
        with self.teaching(bad_pose=True) as s:
            with self.assertRaises(ValueError):
                command.main()
            self.assertEqual(s.config.read_text(), s.original)
            s.robot.disconnect.assert_called_once()
            s.mouse.disconnect.assert_called_once()

    def test_concurrent_config_edit_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.env"
            path.write_text("KEEP=new operator value\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                command.save_bounds(
                    path, "KEEP=old value\n", command.inset_bounds(RAW, 0.01)
                )
            self.assertEqual(path.read_text(), "KEEP=new operator value\n")


if __name__ == "__main__":
    unittest.main()
