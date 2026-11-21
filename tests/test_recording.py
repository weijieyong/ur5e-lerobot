"""Offline regression at the LeRobot recording / robot-command boundary."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop

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
                    calibration_dir=Path(directory) / "robot",
                    timing_log_dir=str(Path(directory) / "timing"),
                    x_min_m=None,
                    x_max_m=None,
                    y_min_m=None,
                    y_max_m=None,
                    z_min_m=None,
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


if __name__ == "__main__":
    unittest.main()
