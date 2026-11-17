"""Bounded roll/pitch relative to a fixed neutral, with free base-Z yaw."""

import math
import itertools
import queue
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lerobot_robot_ur5e.spacemouse import (
    limit_tilt,
    relative_rpy,
    mat_to_rotvec,
    rotvec_to_mat,
    rot_z,
)
from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig, UR5eConfig
from lerobot_robot_ur5e.spacemouse import SpaceMouse
from lerobot_robot_ur5e.ur5e import CONNECTED, UR5e


def pose(roll, pitch, yaw, neutral):
    # Independent known ZYX construction; arguments are degrees.
    r, p, y = map(math.radians, (roll, pitch, yaw))
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
    )
    Ry = np.array(
        [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
    )
    return rot_z(y) @ Ry @ Rx @ neutral


class OrientationTests(unittest.TestCase):
    def test_bounded_mode_enables_roll_pitch_and_requires_a_fixed_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SpaceMouseConfig(
                axes="bounded",
                neutral_rotvec=[0, math.pi, 0],
                tilt_limit_deg=30,
                axis_map="+0 -1 -2 +3 -4 -5".split(),
                operator_yaw_deg=0,
                max_angular_speed_rad_s=0.5,
                calibration_dir=Path(directory),
            )
            mouse = SpaceMouse(config)
            _, angular = mouse.map_axes(np.array([0, 0, 0, 350, -350, 0]))
            np.testing.assert_allclose(angular, [0.5, 0.5, 0])
        with self.assertRaises(ValueError):
            SpaceMouseConfig(axes="bounded", neutral_rotvec=[])

    def test_clamps_each_tilt_axis_and_preserves_yaw_around_base_z(self):
        neutral = rotvec_to_mat(np.array([0.2, 3.0, -0.3]))
        actual = limit_tilt(pose(45, -50, 120, neutral), neutral, math.radians(30))
        np.testing.assert_allclose(actual, pose(30, -30, 120, neutral), atol=1e-10)

    def test_free_yaw_does_not_tilt_the_neutral_posture(self):
        neutral = rotvec_to_mat(np.array([0.4, 2.9, -0.5]))
        for yaw in (-180, -179.999, -90, 0, 90, 179.999, 180):
            with self.subTest(yaw=yaw):
                requested = pose(0, 0, yaw, neutral)
                np.testing.assert_allclose(
                    limit_tilt(requested, neutral, math.radians(30)),
                    requested,
                    atol=1e-10,
                )

    def test_inside_bounds_is_unchanged_and_reversing_leaves_limit_immediately(self):
        neutral = np.diag([1.0, -1.0, -1.0])
        requested = pose(-12, 23, -175, neutral)
        np.testing.assert_allclose(
            limit_tilt(requested, neutral, math.radians(30)), requested, atol=1e-10
        )
        at_limit = limit_tilt(pose(40, 0, 0, neutral), neutral, math.radians(30))
        reversed_request = rotvec_to_mat(np.array([math.radians(-1), 0, 0])) @ at_limit
        np.testing.assert_allclose(
            limit_tilt(reversed_request, neutral, math.radians(30)),
            pose(29, 0, 0, neutral),
            atol=1e-10,
        )

    def test_live_input_path_limits_ik_requests_and_rejects_an_outside_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot = UR5e(
                UR5eConfig(
                    cameras={}, calibration_dir=root / "robot", timing_log_dir=""
                )
            )
            measured = np.array([0.0, 0.0, 0.5, 0.0, math.pi, 0.0])
            robot.rtde_r = SimpleNamespace(
                getActualQ=lambda: [0.0] * 6, getActualTCPPose=lambda: measured.copy()
            )
            requests = []

            def ik(target, seed):
                requests.append(target)
                return [0.0] * 6

            robot.rtde_c = SimpleNamespace(
                getInverseKinematics=ik, isConnected=lambda: False
            )
            robot.gripper = SimpleNamespace(position=lambda: 0.0)
            packets = queue.Queue()

            def read(*args):
                try:
                    return packets.get(timeout=0.01)
                except queue.Empty:
                    return []

            device = SimpleNamespace(
                open_path=lambda _: None, read=read, close=lambda: None
            )
            backend = SimpleNamespace(
                enumerate=lambda *args: [{"path": b"fixture"}], device=lambda: device
            )

            class Listener:
                def __init__(self, on_press):
                    pass

                def start(self):
                    pass

                def stop(self):
                    pass

            keyboard = SimpleNamespace(
                Key=SimpleNamespace(esc=object()), Listener=Listener
            )
            config = SpaceMouseConfig(
                axes="bounded",
                neutral_rotvec=[0, math.pi, 0],
                tilt_limit_deg=30,
                robot_ip="192.0.2.2",
                axis_map="+0 -1 -2 +3 -4 -5".split(),
                operator_yaw_deg=0,
                max_angular_speed_rad_s=0.5,
                calibration_dir=root / "mouse",
            )
            mouse = SpaceMouse(config)
            with (
                patch.dict(CONNECTED, {config.robot_ip: robot}),
                patch.dict(
                    "sys.modules",
                    {"hidraw": backend, "pynput": SimpleNamespace(keyboard=keyboard)},
                ),
            ):
                try:
                    mouse.connect()
                    report = b"\x02" + b"".join(
                        value.to_bytes(2, "little", signed=True)
                        for value in (350, 0, 0)
                    )
                    packets.put(report)
                    deadline = time.monotonic() + 1
                    while mouse.hid.get_state()[0][3] != 350:
                        if time.monotonic() > deadline:
                            self.fail("Fixture HID report was not read")
                        time.sleep(0.001)
                    clock = itertools.count(0.0, 0.1)
                    with patch(
                        "lerobot_robot_ur5e.spacemouse.time.monotonic",
                        side_effect=lambda: next(clock),
                    ):
                        mouse.get_action()
                        self.assertEqual(requests, [])  # no startup orientation snap
                        for _ in range(20):
                            mouse.get_action()
                    neutral = rotvec_to_mat(np.array(config.neutral_rotvec))
                    angles = [
                        np.degrees(
                            relative_rpy(rotvec_to_mat(np.array(target[3:])), neutral)
                        )
                        for target in requests
                    ]
                    self.assertAlmostEqual(angles[-1][0], 30, places=4)
                    self.assertTrue(
                        all(
                            max(abs(roll), abs(pitch)) <= 30.0001
                            for roll, pitch, _ in angles
                        )
                    )
                    mouse.disconnect()
                    measured[3:] = mat_to_rotvec(pose(45, 0, 0, neutral))
                    mouse.connect()
                    count = len(requests)
                    with self.assertRaisesRegex(RuntimeError, "outside"):
                        mouse.get_action()
                    self.assertEqual(len(requests), count)
                finally:
                    mouse.disconnect()
