"""TCP workspace limits exercised through SpaceMouse's actual target/IK path."""

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig
from lerobot_robot_ur5e.spacemouse import SpaceMouse
from lerobot_robot_ur5e.ur5e import CONNECTED


class WorkspaceTests(unittest.TestCase):
    def request(self, measured, axes):
        with tempfile.TemporaryDirectory() as directory:
            cfg = SpaceMouseConfig(
                axes="full",
                x_min_m=-0.5,
                x_max_m=0.5,
                y_min_m=-0.3,
                y_max_m=0.3,
                z_min_m=0.1,
                axis_map="+0 +1 +2 +3 +4 +5".split(),
                operator_yaw_deg=0,
                max_linear_speed_m_s=1,
                max_lead_m=1,
                calibration_dir=Path(directory),
            )
            mouse = SpaceMouse(cfg)
            mouse.hid = SimpleNamespace(
                get_state=lambda: (np.array(axes + [0, 0, 0]), 0)
            )
            targets = []
            robot = SimpleNamespace(
                measured_joints=lambda: np.zeros(6),
                rtde_r=SimpleNamespace(
                    getActualTCPPose=lambda: list(measured) + [0, 0, 0]
                ),
                rtde_c=SimpleNamespace(
                    getInverseKinematics=lambda p, q: targets.append(p) or [0.0] * 6
                ),
                gripper=SimpleNamespace(position=lambda: 0.0),
            )
            with (
                patch.dict(CONNECTED, {cfg.robot_ip: robot}),
                patch(
                    "lerobot_robot_ur5e.spacemouse.time.monotonic",
                    side_effect=[0.0, 0.1, 0.2],
                ),
            ):
                mouse.get_action()
                self.assertEqual(targets, [])
                mouse.get_action()
                mouse.get_action()
            return np.array(targets)[:, :3]

    def test_each_side_and_floor_clamps_but_top_is_unbounded(self):
        cases = [
            ([0.49, 0, 0.5], [350, 0, 0], [0.5, 0, 0.5]),
            ([-0.49, 0, 0.5], [-350, 0, 0], [-0.5, 0, 0.5]),
            ([0, 0.29, 0.5], [0, 350, 0], [0, 0.3, 0.5]),
            ([0, -0.29, 0.5], [0, -350, 0], [0, -0.3, 0.5]),
            ([0, 0, 0.11], [0, 0, -350], [0, 0, 0.1]),
        ]
        for measured, axes, expected in cases:
            with self.subTest(measured=measured):
                np.testing.assert_allclose(
                    self.request(measured, axes), [expected, expected]
                )
        np.testing.assert_allclose(
            self.request([0, 0, 2], [0, 0, 350]), [[0, 0, 2.1], [0, 0, 2.2]]
        )

    def test_can_slide_along_wall_and_reverse_inward(self):
        np.testing.assert_allclose(
            self.request([0.5, 0, 0.5], [350, 350, 0]),
            [[0.5, 0.1, 0.5], [0.5, 0.2, 0.5]],
        )
        np.testing.assert_allclose(
            self.request([0.5, 0, 0.5], [-350, 0, 0]), [[0.4, 0, 0.5], [0.3, 0, 0.5]]
        )

    def test_outside_start_can_recover_inward_without_snapping_or_moving_further_out(
        self,
    ):
        np.testing.assert_allclose(
            self.request([0.8, 0, 0.5], [-350, 0, 0]), [[0.7, 0, 0.5], [0.6, 0, 0.5]]
        )
        np.testing.assert_allclose(
            self.request([0.8, 0, 0.5], [350, 0, 0]), [[0.8, 0, 0.5], [0.8, 0, 0.5]]
        )
        np.testing.assert_allclose(
            self.request([0, 0, 0.05], [0, 0, 350]), [[0, 0, 0.15], [0, 0, 0.25]]
        )

    def test_configuration_rejects_invalid_bounds_and_loads_environment(self):
        for kwargs in (
            {"x_min_m": math.nan},
            {"y_max_m": math.inf},
            {"x_min_m": 1, "x_max_m": 0},
            {"y_min_m": 0, "y_max_m": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpaceMouseConfig(**kwargs)
        with patch.dict(
            "os.environ",
            {
                "UR5E_X_MIN_M": "-.2",
                "UR5E_X_MAX_M": ".6",
                "UR5E_Y_MIN_M": "",
                "UR5E_Y_MAX_M": "",
            },
        ):
            cfg = SpaceMouseConfig()
            self.assertEqual(
                (cfg.x_min_m, cfg.x_max_m, cfg.y_min_m, cfg.y_max_m),
                (-0.2, 0.6, None, None),
            )


if __name__ == "__main__":
    unittest.main()
