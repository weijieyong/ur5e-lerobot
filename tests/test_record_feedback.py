"""Recording feedback at the pinned LeRobot orchestration boundary; no hardware."""

import contextlib
import io
import math
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.image_writer import AsyncImageWriter, safe_stop_image_writer
from lerobot.scripts import lerobot_record as upstream
from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig
from lerobot_robot_ur5e.spacemouse import SpaceMouse
from lerobot_robot_ur5e.ur5e import CONNECTED


from scripts import record as recording


class RecordingFeedbackTests(unittest.TestCase):
    @contextlib.contextmanager
    def outside_tilt(self):
        with tempfile.TemporaryDirectory() as directory:
            mouse = SpaceMouse(
                SpaceMouseConfig(
                    axes="bounded",
                    neutral_rotvec=[0.0, 0.0, 0.0],
                    tilt_limit_deg=30,
                    calibration_dir=Path(directory),
                )
            )
            mouse.hid = SimpleNamespace(get_state=lambda: (np.zeros(6), 0))
            pose = np.array([0.0, 0.0, 0.5, math.radians(45), 0.0, 0.0])
            homes = []

            def home():
                homes.append(True)
                pose[3:] = 0.0

            robot = SimpleNamespace(
                measured_joints=lambda: np.zeros(6),
                rtde_r=SimpleNamespace(getActualTCPPose=lambda: pose.copy()),
                gripper=SimpleNamespace(position=lambda: 0.0),
                go_home=home,
            )
            with (
                patch.dict(CONNECTED, {mouse.config.robot_ip: robot}, clear=True),
                patch.object(mouse, "connect"),
                patch.object(
                    upstream, "make_teleoperator_from_config", return_value=mouse
                ),
            ):
                yield mouse, homes, pose

    @contextlib.contextmanager
    def recorder(self, loop, episodes=2):
        # External writer/hardware stand-ins; exercise LeRobot's actual episode state machine.
        dataset = SimpleNamespace(num_episodes=5, finalize=lambda: None)
        dataset.save_episode = lambda: setattr(
            dataset, "num_episodes", dataset.num_episodes + 1
        )
        robot = SimpleNamespace(
            cameras={},
            action_features={},
            observation_features={},
            name="ur5e",
            robot_type="ur5e",
            is_connected=False,
        )
        robot.connect = lambda: None
        teleop = SimpleNamespace(connect=lambda: None, is_connected=False)
        cfg = upstream.RecordConfig(
            robot=SimpleNamespace(),
            teleop=SimpleNamespace(),
            play_sounds=False,
            dataset=upstream.DatasetRecordConfig(
                repo_id="local/test",
                single_task="pick and place",
                num_episodes=episodes,
                reset_time_s=60,
                episode_time_s=60,
                no_stamp=True,
                push_to_hub=False,
            ),
        )
        events = dict(exit_early=False, rerecord_episode=False, stop_recording=False)
        output = io.StringIO()
        dataset.clear_episode_buffer = lambda: None
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(upstream, "make_robot_from_config", return_value=robot)
            )
            stack.enter_context(
                patch.object(
                    upstream, "make_teleoperator_from_config", return_value=teleop
                )
            )
            stack.enter_context(
                patch.object(upstream.LeRobotDataset, "create", return_value=dataset)
            )
            stack.enter_context(
                patch.object(
                    upstream,
                    "VideoEncodingManager",
                    return_value=contextlib.nullcontext(),
                )
            )
            stack.enter_context(
                patch.object(
                    upstream, "init_keyboard_listener", return_value=(None, events)
                )
            )
            stack.enter_context(patch.object(upstream, "record_loop", side_effect=loop))
            stack.enter_context(contextlib.redirect_stdout(output))
            yield cfg, dataset, events, output

    def test_outside_tilt_shows_recovery_and_right_cannot_start_until_home(self):
        phases = []

        def loop(**kwargs):
            phases.append("record" if kwargs.get("dataset") is not None else "ready")
            if len(phases) == 2:
                return
            if len(phases) == 3:
                mouse._home_requested.set()
            kwargs["teleop"].get_action()

        with (
            self.recorder(loop, episodes=1) as (cfg, dataset, events, output),
            self.outside_tilt() as (mouse, homes, pose),
        ):
            feedback = recording.RecordingFeedback(cfg)
            rr = feedback.rr = Mock()
            with patch.object(recording, "RecordingFeedback", return_value=feedback):
                recording.run_recording(cfg)
            self.assertEqual(dataset.num_episodes, 6)
            self.assertEqual(homes, [True])
            self.assertEqual(phases, ["ready", "ready", "ready", "record"])
            self.assertEqual(output.getvalue().count("RECOVERY"), 1)
            self.assertIn("Press h", output.getvalue())
            documents = [call.args[0] for call in rr.TextDocument.call_args_list]
            self.assertTrue(
                any(
                    "RECOVERY" in text and "outside" in text and "Press h" in text
                    for text in documents
                )
            )

    def test_esc_or_ctrl_c_during_tilt_recovery_exits_without_saving(self):
        for ctrl_c in (False, True):
            with self.subTest(ctrl_c=ctrl_c):
                calls = []

                def loop(**kwargs):
                    self.assertIsNone(kwargs.get("dataset"))
                    calls.append(True)
                    if len(calls) == 2:
                        if ctrl_c:
                            signal.raise_signal(signal.SIGINT)
                        else:
                            kwargs["events"]["stop_recording"] = True
                    kwargs["teleop"].get_action()

                original_handler = signal.getsignal(signal.SIGINT)
                with (
                    self.recorder(loop) as (cfg, dataset, events, output),
                    self.outside_tilt(),
                ):
                    recording.run_recording(cfg)
                    self.assertEqual(dataset.num_episodes, 5)
                    self.assertEqual(len(calls), 2)
                    self.assertNotIn("RECORDING 1/2", output.getvalue())
                self.assertEqual(signal.getsignal(signal.SIGINT), original_handler)

    def test_recording_tilt_error_discards_unsaved_attempt_and_returns_to_recovery(
        self,
    ):
        phases, discarded = [], []

        def loop(**kwargs):
            phases.append("record" if kwargs.get("dataset") is not None else "ready")
            if len(phases) in (1, 4):
                mouse._home_requested.set()
            if len(phases) == 2:
                pose[3] = math.radians(45)
                mouse.reset_reference()
            if len(phases) == 4:
                # The native record-loop decorator stops this writer on the tilt error.
                # Retrying must still be able to persist camera frames without hanging.
                self.assertTrue(
                    any(t.is_alive() for t in dataset.writer.image_writer.threads)
                )
                image_path = Path(mouse.config.calibration_dir) / "retry.png"
                dataset.writer.image_writer.save_image(
                    np.zeros((8, 8, 3), dtype=np.uint8), image_path
                )
                dataset.writer.image_writer.wait_until_done()
                self.assertTrue(image_path.is_file())
            kwargs["teleop"].get_action()

        with (
            self.recorder(safe_stop_image_writer(loop), episodes=1) as (
                cfg,
                dataset,
                events,
                output,
            ),
            self.outside_tilt() as (mouse, homes, pose),
        ):
            dataset.clear_episode_buffer = lambda: discarded.append(True)
            writer = dataset.writer = SimpleNamespace(
                image_writer=AsyncImageWriter(0, 1)
            )
            writer.start_image_writer = lambda **kwargs: (
                DatasetWriter.start_image_writer(writer, **kwargs)
            )
            try:
                recording.run_recording(cfg)
            finally:
                writer.image_writer.stop()
            self.assertEqual(phases, ["ready", "record", "ready", "ready", "record"])
            self.assertEqual(dataset.num_episodes, 6)
            self.assertEqual(discarded, [True])
            self.assertEqual(homes, [True, True])
            self.assertIn("This unsaved attempt is discarded", output.getvalue())

    def test_non_tilt_errors_still_propagate(self):
        def loop(**kwargs):
            raise RuntimeError("HID reader failed")

        with self.recorder(loop) as (cfg, dataset, events, output):
            with self.assertRaisesRegex(RuntimeError, "HID reader failed"):
                recording.run_recording(cfg)
            self.assertEqual(dataset.num_episodes, 5)

    def test_stop_saves_before_ready_and_next_start_requires_another_operator_input(
        self,
    ):
        phases = []
        saved_counts = []

        def loop(**kwargs):
            phases.append("record" if kwargs.get("dataset") is not None else "reset")
            saved_counts.append(dataset.num_episodes)
            self.assertEqual(kwargs["control_time_s"], float("inf"))

        with self.recorder(loop) as (cfg, dataset, events, output):
            recording.run_recording(cfg)

        self.assertEqual(phases, ["reset", "record", "reset", "record"])
        self.assertEqual(saved_counts, [5, 5, 6, 6])
        text = output.getvalue()
        for expected in (
            "READY — NOT RECORDING",
            "RECORDING 1/2",
            "dataset episode 5",
            "SAVING",
            "NOT RECORDING",
            "Right",
            "RECORDING 2/2",
            "dataset episode 6",
            "7 saved episodes",
        ):
            self.assertIn(expected, text)
        self.assertEqual(dataset.num_episodes, 7)

    def test_ctrl_c_while_ready_after_save_exits_without_another_episode(self):
        recorded = []

        def loop(**kwargs):
            if kwargs.get("dataset") is not None:
                recorded.append(True)
            elif recorded:
                signal.raise_signal(signal.SIGINT)

        original_handler = signal.getsignal(signal.SIGINT)
        with self.recorder(loop) as (cfg, dataset, events, output):
            try:
                recording.run_recording(cfg)
            except KeyboardInterrupt:
                self.fail(
                    "Ctrl+C interrupted saving instead of requesting a graceful stop"
                )
            self.assertEqual(dataset.num_episodes, 6)
            self.assertTrue(events["stop_recording"])
        self.assertEqual(signal.getsignal(signal.SIGINT), original_handler)

    def test_keys_pressed_during_saving_do_not_start_or_discard_the_next_episode(self):
        def loop(**kwargs):
            if dataset.num_episodes == 6:
                self.assertIsNone(kwargs.get("dataset"))
                self.assertFalse(events["exit_early"])
                self.assertFalse(events["rerecord_episode"])
                events["stop_recording"] = True

        with self.recorder(loop) as (cfg, dataset, events, output):

            def save():
                dataset.num_episodes += 1
                # Keyboard input while the external writer is busy must not carry into READY.
                events["exit_early"] = True
                events["rerecord_episode"] = True

            dataset.save_episode = save
            recording.run_recording(cfg)
            self.assertEqual(dataset.num_episodes, 6)

    def test_quitting_before_start_does_not_record_or_save_an_empty_episode(self):
        def loop(**kwargs):
            self.assertIsNone(kwargs.get("dataset"))
            signal.raise_signal(signal.SIGINT)

        with self.recorder(loop) as (cfg, dataset, events, output):
            recording.run_recording(cfg)
            self.assertEqual(dataset.num_episodes, 5)
            self.assertNotIn("RECORDING 1/2", output.getvalue())

    def test_retry_discards_buffer_without_advancing_the_episode_number(self):
        calls = []

        def loop(**kwargs):
            calls.append("record" if kwargs.get("dataset") is not None else "reset")
            if len(calls) == 2:
                kwargs["events"]["rerecord_episode"] = True
            elif len(calls) == 3:
                self.assertEqual(discarded, [True])
                self.assertEqual(dataset.num_episodes, 5)

        with self.recorder(loop, episodes=1) as (cfg, dataset, events, output):
            discarded = []
            dataset.clear_episode_buffer = lambda: discarded.append(True)
            recording.run_recording(cfg)
            self.assertEqual(calls, ["reset", "record", "reset", "record"])
            self.assertEqual(discarded, [True])
            self.assertEqual(dataset.num_episodes, 6)
            self.assertEqual(output.getvalue().count("RECORDING 1/1"), 2)
            self.assertIn("retry this episode", output.getvalue())

    def test_lerobot_default_layout_keeps_phase_banner_until_viewer_closes(self):
        import rerun as rr
        from rerun.experimental import ViewerClient
        from lerobot.utils import rerun_visualization
        from lerobot.utils.visualization_utils import log_rerun_data

        native = rerun_visualization._build_blueprint
        feedback = recording.RecordingFeedback(SimpleNamespace())
        viewer = Mock(url="rerun+http://127.0.0.1:12345/proxy")
        try:
            with (
                patch.object(recording.socket, "socket") as socket_factory,
                patch.object(ViewerClient, "spawn", return_value=viewer),
                patch.object(rr, "connect_grpc"),
                patch.object(rr, "log"),
                patch.object(rr, "send_blueprint") as send_blueprint,
            ):
                socket_factory.return_value.__enter__.return_value.getsockname.return_value = (
                    "127.0.0.1",
                    12345,
                )
                feedback.init_viewer()
                log_rerun_data(
                    observation={
                        "joint_0.pos": 0.1,
                        "front": np.zeros((8, 8, 3), np.uint8),
                    },
                    action={"joint_0.pos": 0.2},
                )
                # LeRobot still sends (and later grows) its own grid, under the phase/controls strip.
                status, grid = send_blueprint.call_args.args[0].root_container.contents
                self.assertEqual(
                    [view.name for view in status.contents], ["Phase", "Controls"]
                )
                self.assertEqual(
                    [view.name for view in grid.contents],
                    ["observation.front", "observation", "action"],
                )
            feedback.close_viewer()
            viewer.close.assert_called_once()
            self.assertIs(rerun_visualization._build_blueprint, native)
        finally:
            rerun_visualization._build_blueprint = native
            rr.rerun_shutdown()

    def test_owned_viewer_closes_even_when_recording_is_interrupted_or_cleanup_fails(
        self,
    ):
        import rerun as rr
        from rerun.experimental import ViewerClient

        for error in (KeyboardInterrupt(), RuntimeError("writer cleanup failed")):
            with (
                self.subTest(error=type(error).__name__),
                self.recorder(lambda **_: None) as fixture,
            ):
                cfg, _, _, _ = fixture
                cfg.display_data = True
                viewer = Mock(url="rerun+http://127.0.0.1:12345/proxy")

                def fail(cfg):
                    upstream.init_visualization(
                        "rerun", session_name="recording", ip=None, port=None
                    )
                    raise error

                original_loop = upstream.record_loop
                original_init = upstream.init_visualization
                try:
                    with (
                        patch.object(recording.socket, "socket") as socket_factory,
                        patch.object(ViewerClient, "spawn", return_value=viewer),
                        patch.object(rr, "connect_grpc"),
                        patch.object(upstream.record, "__wrapped__", side_effect=fail),
                    ):
                        socket_factory.return_value.__enter__.return_value.getsockname.return_value = (
                            "127.0.0.1",
                            12345,
                        )
                        with self.assertRaises(type(error)):
                            recording.run_recording(cfg)
                    viewer.close.assert_called_once()
                    self.assertIs(upstream.record_loop, original_loop)
                    self.assertIs(upstream.init_visualization, original_init)
                finally:
                    rr.rerun_shutdown()


if __name__ == "__main__":
    unittest.main()
