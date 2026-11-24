"""Offline regression at the LeRobot recording / robot-command boundary."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.feature_utils import hw_to_dataset_features

from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig, UR5eConfig
from lerobot_robot_ur5e.spacemouse import SpaceMouse
from lerobot_robot_ur5e.ur5e import GRIPPER, JOINTS, UR5e


class ScriptedMouse(SpaceMouse):
    def get_action(self):
        return dict(zip(JOINTS + [GRIPPER], [1.0, -1.0, 0, 0, 0, 0, 0.7]))


class MemoryDataset:
    fps = 15
    features = {
        name: {"dtype": "float32", "shape": (7,), "names": JOINTS + [GRIPPER]}
        for name in ("observation.state", "action")
    }

    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)


class RecordingTests(unittest.TestCase):
    def test_recorded_action_matches_the_clipped_target_sent_to_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = UR5e(
                UR5eConfig(
                    cameras={},
                    gripper_type="robotiq",
                    calibration_dir=Path(directory) / "robot",
                    timing_log_dir=str(Path(directory) / "timing"),
                )
            )
            sent = []
            # Stand-ins for external hardware only; use the real plugin and LeRobot loop/processors.
            robot.rtde_r = SimpleNamespace(
                getActualQ=lambda: [0.0] * 6,
                isProtectiveStopped=lambda: False,
                isEmergencyStopped=lambda: False,
                disconnect=lambda: None,
            )
            robot.rtde_c = SimpleNamespace(
                servoJ=lambda *args: sent.append(args[0]),
                isConnected=lambda: False,
                servoStop=lambda: None,
                stopScript=lambda: None,
                disconnect=lambda: None,
            )
            robot.gripper = SimpleNamespace(
                position=lambda: 0.0, grip=lambda _: None, close=lambda: None
            )
            mouse = ScriptedMouse(
                SpaceMouseConfig(calibration_dir=Path(directory) / "mouse")
            )
            dataset = MemoryDataset()
            teleop_processor, action_processor, observation_processor = (
                make_default_processors()
            )
            record_loop(
                robot=robot,
                events={"exit_early": False},
                fps=15,
                teleop_action_processor=teleop_processor,
                robot_action_processor=action_processor,
                robot_observation_processor=observation_processor,
                dataset=dataset,
                teleop=mouse,
                control_time_s=0.001,
                single_task="pick and place",
            )
            self.assertEqual(len(dataset.frames), 1)
            np.testing.assert_allclose(sent[0], [0.05, -0.05, 0, 0, 0, 0])
            np.testing.assert_allclose(
                dataset.frames[0]["action"], [0.05, -0.05, 0, 0, 0, 0, 1]
            )
            np.testing.assert_array_equal(
                dataset.frames[0]["observation.state"], [0] * 7
            )
            robot.disconnect()
            logs = list((Path(directory) / "timing").glob("ur5e-*.jsonl"))
            self.assertEqual(len(logs), 1)
            row = json.loads(logs[0].read_text())
            self.assertTrue(row["clipped"])
            self.assertEqual(row["sent_joints"], sent[0])
            self.assertLessEqual(row["observation_start_s"], row["observation_end_s"])
            self.assertLessEqual(row["observation_end_s"], row["send_start_s"])
            self.assertLessEqual(row["send_start_s"], row["send_end_s"])

    def test_six_joint_recording_skips_gripper_on_connect_buttons_and_home(self):
        ip = "192.0.2.1"
        measured = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"GRIPPER_TYPE": "none"}),
            patch("rtde_receive.RTDEReceiveInterface") as connect_receiver,
            patch("rtde_control.RTDEControlInterface") as connect_controller,
            patch("lerobot_robot_ur5e.ur5e.RobotiqSocket") as robotiq,
            patch("lerobot_robot_ur5e.ur5e.OnRobotRG2") as rg2,
        ):
            robot = UR5e(
                UR5eConfig(
                    ip=ip,
                    cameras={},
                    calibration_dir=Path(directory) / "robot",
                    home_joints_deg=[0.0] * 6,
                    timing_log_dir="",
                )
            )
            mouse = SpaceMouse(
                SpaceMouseConfig(
                    robot_ip=ip, axes="full", calibration_dir=Path(directory) / "mouse"
                )
            )
            self.assertEqual(list(mouse.action_features), JOINTS)
            mouse.hid.get_state = Mock(
                return_value=(np.zeros(6), 1 << mouse.config.close_button)
            )
            receiver = connect_receiver.return_value
            receiver.isProtectiveStopped.return_value = False
            receiver.isEmergencyStopped.return_value = False
            receiver.getActualQ.return_value = measured
            receiver.getActualTCPPose.return_value = [0.0] * 6
            controller = connect_controller.return_value
            robot.connect()
            try:
                self.assertEqual(mouse.action_features, robot.action_features)
                mouse._home_requested.set()
                dataset = MemoryDataset()
                dataset.features = {
                    **hw_to_dataset_features(robot.action_features, "action"),
                    **hw_to_dataset_features(robot.observation_features, "observation"),
                }
                for feature in dataset.features.values():
                    self.assertEqual(feature["names"], JOINTS)
                    self.assertEqual(feature["shape"], (6,))
                teleop, action, observation = make_default_processors()
                record_loop(
                    robot=robot,
                    events={"exit_early": False},
                    fps=15,
                    teleop_action_processor=teleop,
                    robot_action_processor=action,
                    robot_observation_processor=observation,
                    dataset=dataset,
                    teleop=mouse,
                    control_time_s=0.001,
                    single_task="move the arm",
                )
                self.assertEqual(len(dataset.frames), 1)
                np.testing.assert_allclose(dataset.frames[0]["action"], measured)
                np.testing.assert_allclose(
                    dataset.frames[0]["observation.state"], measured
                )
                controller.moveJ.assert_called_once_with([0.0] * 6, 0.3, 0.5)
                self.assertEqual(controller.servoJ.call_args.args[0], measured)
                mouse.hid.get_state.return_value = (
                    np.zeros(6),
                    1 << mouse.config.open_button,
                )
                self.assertEqual(list(mouse.get_action()), JOINTS)
            finally:
                robot.disconnect()
            robotiq.assert_not_called()
            rg2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
