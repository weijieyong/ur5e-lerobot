"""TCP workspace limits across SpaceMouse pre-IK clamping and UR5e.send_action()."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig, UR5eConfig
from lerobot_robot_ur5e.spacemouse import SpaceMouse
from lerobot_robot_ur5e.ur5e import CONNECTED, GRIPPER, JOINTS, UR5e


class WorkspaceTests(unittest.TestCase):
    def request(self, measured, axes):
        with tempfile.TemporaryDirectory() as directory:
            robot_cfg = UR5eConfig(
                cameras={},
                x_min_m=-0.5,
                x_max_m=0.5,
                y_min_m=-0.3,
                y_max_m=0.3,
                z_min_m=0.1,
                calibration_dir=Path(directory) / "robot",
            )
            cfg = SpaceMouseConfig(
                axes="full",
                axis_map="+0 +1 +2 +3 +4 +5".split(),
                operator_yaw_deg=0,
                max_linear_speed_m_s=1,
                max_lead_m=1,
                calibration_dir=Path(directory) / "mouse",
            )
            mouse = SpaceMouse(cfg)
            mouse.hid = SimpleNamespace(
                get_state=lambda: (np.array(axes + [0, 0, 0]), 0)
            )
            targets = []
            robot = SimpleNamespace(
                config=robot_cfg,
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
                mouse.get_action()  # Startup holds measured pose.
                self.assertEqual(targets, [])
                mouse.get_action()
                mouse.get_action()  # No accumulation of outward position at a wall.
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
            {"max_ik_jump_rad": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                UR5eConfig(cameras={}, **kwargs)
        with patch.dict(
            "os.environ",
            {
                "UR5E_X_MIN_M": "-.2",
                "UR5E_X_MAX_M": ".6",
                "UR5E_Y_MIN_M": "",
                "UR5E_Y_MAX_M": "",
                "UR5E_Z_MIN_M": "0.12",
            },
        ):
            cfg = UR5eConfig(cameras={})
            self.assertEqual(
                (cfg.x_min_m, cfg.x_max_m, cfg.y_min_m, cfg.y_max_m, cfg.z_min_m),
                (-0.2, 0.6, None, None, 0.12),
            )


class UR5eWorkspaceEnforcementTests(unittest.TestCase):
    """Exercise FK -> clip -> IK workspace clamping inside UR5e.send_action()."""

    def make_robot(
        self,
        *,
        q_meas=None,
        tcp_meas=None,
        fk=None,
        ik=None,
        max_joint_step_rad=0.05,
        max_ik_jump_rad=0.5,
        timing_dir="",
        **bounds,
    ):
        # Simple linear kinematics by default: TCP xyz == first 3 joint values, rotvec == last 3.
        q_meas = list(q_meas if q_meas is not None else [0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
        tcp_meas = list(tcp_meas if tcp_meas is not None else q_meas)
        default_bounds = {
            "x_min_m": -0.5,
            "x_max_m": 0.5,
            "y_min_m": -0.3,
            "y_max_m": 0.3,
            "z_min_m": 0.1,
        }
        default_bounds.update(bounds)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        robot = UR5e(
            UR5eConfig(
                ip="192.0.2.1",
                cameras={},
                calibration_dir=Path(directory.name) / "robot",
                timing_log_dir=timing_dir,
                max_joint_step_rad=max_joint_step_rad,
                max_ik_jump_rad=max_ik_jump_rad,
                **default_bounds,
            )
        )
        calls = SimpleNamespace(sent=[], fk=[], ik=[], tcp=[], grips=[])

        def default_fk(q):
            calls.fk.append(list(q))
            return list(q)

        def default_ik(pose, qnear):
            calls.ik.append((list(pose), list(qnear)))
            return list(pose)

        def get_tcp():
            calls.tcp.append(True)
            return list(tcp_meas)

        robot.rtde_r = SimpleNamespace(
            getActualQ=lambda: list(q_meas),
            getActualTCPPose=get_tcp,
            isProtectiveStopped=lambda: False,
            isEmergencyStopped=lambda: False,
            disconnect=lambda: None,
        )
        robot.rtde_c = SimpleNamespace(
            servoJ=lambda *args: calls.sent.append(args[0]) or True,
            getForwardKinematics=fk or default_fk,
            getInverseKinematics=ik or default_ik,
            isConnected=lambda: False,
            servoStop=lambda: None,
            stopScript=lambda: None,
            disconnect=lambda: None,
        )
        robot.gripper = SimpleNamespace(
            position=lambda: 0.0,
            grip=lambda closed: calls.grips.append(closed),
            close=lambda: None,
        )
        self.addCleanup(robot.disconnect)
        return robot, calls

    @staticmethod
    def action(joints, gripper=0.0):
        return {**dict(zip(JOINTS, joints)), GRIPPER: gripper}

    def test_in_bounds_fast_path_calls_fk_once_without_tcp_or_ik(self):
        robot, calls = self.make_robot(q_meas=[0.0, 0.0, 0.2, 0.1, 0.2, 0.3])
        act = self.action([0.02, -0.01, 0.18, 0.1, 0.2, 0.3], gripper=0.0)
        robot.send_action(act)
        np.testing.assert_allclose(calls.sent[0], [0.02, -0.01, 0.18, 0.1, 0.2, 0.3])
        self.assertEqual(len(calls.fk), 1)
        self.assertEqual(calls.tcp, [])
        self.assertEqual(calls.ik, [])

    def test_out_of_bounds_clamps_and_slides_preserving_orientation_and_gripper(self):
        with tempfile.TemporaryDirectory() as timing_dir:
            robot, calls = self.make_robot(
                q_meas=[0.48, 0.0, 0.11, 0.1, 0.2, 0.3],
                timing_dir=timing_dir,
            )
            # Push past +X and below floor Z while moving in +Y and closing gripper.
            act = self.action([0.52, 0.03, 0.08, 0.1, 0.2, 0.3], gripper=1.0)
            with self.assertLogs("lerobot_robot_ur5e.ur5e", level="WARNING"):
                robot.send_action(act)
            expected = [0.5, 0.03, 0.1, 0.1, 0.2, 0.3]
            np.testing.assert_allclose(calls.sent[0], expected)
            # Action dict is updated in-place with the actually sent clamped joints.
            np.testing.assert_allclose([act[j] for j in JOINTS], expected)
            self.assertEqual(act[GRIPPER], 1.0)
            self.assertEqual(calls.grips, [True])
            robot.disconnect()
            log_file = next(Path(timing_dir).glob("ur5e-*.jsonl"))
            row = json.loads(log_file.read_text())
            self.assertTrue(row["clipped"])
            self.assertTrue(row["workspace_clamped"])
            self.assertFalse(row["workspace_held"])
            self.assertEqual(row["workspace_faces"], ["x_max", "z_min"])

    def test_outside_start_allows_inward_recovery_and_blocks_further_outward(self):
        # Currently at z=0.05 (below floor 0.10): moving upward to 0.08 is allowed without IK.
        robot, calls = self.make_robot(q_meas=[0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
        robot.send_action(self.action([0.0, 0.0, 0.08, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(calls.sent[0], [0.0, 0.0, 0.08, 0.0, 0.0, 0.0])
        self.assertEqual(calls.ik, [])

        # Pushing further down to 0.02 is clamped at current z=0.05 (no snap to 0.10).
        with self.assertLogs("lerobot_robot_ur5e.ur5e", level="WARNING"):
            robot.send_action(self.action([0.01, 0.0, 0.02, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(calls.sent[1], [0.01, 0.0, 0.05, 0.0, 0.0, 0.0])

    def test_ik_exceeding_step_cap_clips_and_rechecks_fk_before_falling_back_to_hold(
        self,
    ):
        # Case 1: IK nudges joint_0 past max_joint_step_rad (0.01), clipped result stays in bounds.
        robot, calls = self.make_robot(
            q_meas=[0.0, 0.0, 0.105, 0.0, 0.0, 0.0],
            max_joint_step_rad=0.01,
            ik=lambda pose, qnear: [0.02, 0.0, 0.1, 0.0, 0.0, 0.0],
        )
        with self.assertLogs("lerobot_robot_ur5e.ur5e", level="WARNING"):
            robot.send_action(self.action([0.005, 0.0, 0.095, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(calls.sent[0], [0.01, 0.0, 0.1, 0.0, 0.0, 0.0])

        # Case 2: Re-clipped IK bends FK back below floor -> falls back to holding q_meas.
        q_meas = [0.0, 0.0, 0.105, 0.0, 0.0, 0.0]

        def fk_violates_on_reclip(q):
            # Initial target is below floor; re-clipped target at joint_0=0.01 also dips below floor.
            if abs(q[0] - 0.01) < 1e-9:
                return [0.01, 0.0, 0.09, 0.0, 0.0, 0.0]
            return list(q)

        robot2, calls2 = self.make_robot(
            q_meas=q_meas,
            max_joint_step_rad=0.01,
            fk=fk_violates_on_reclip,
            ik=lambda pose, qnear: [0.02, 0.0, 0.1, 0.0, 0.0, 0.0],
        )
        with self.assertLogs("lerobot_robot_ur5e.ur5e", level="WARNING"):
            robot2.send_action(
                self.action([0.005, 0.0, 0.095, 0.0, 0.0, 0.0], gripper=1.0)
            )
        np.testing.assert_allclose(calls2.sent[0], q_meas)
        self.assertEqual(calls2.grips, [True])

    def test_ik_failure_or_large_jump_holds_current_joints_and_runs_gripper(self):
        q_meas = [0.0, 0.0, 0.11, 0.0, 0.0, 0.0]
        for label, ik_fn in (
            ("empty", lambda pose, qnear: []),
            ("jump", lambda pose, qnear: [1.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        ):
            with self.subTest(label=label):
                robot, calls = self.make_robot(q_meas=q_meas, ik=ik_fn)
                with self.assertLogs("lerobot_robot_ur5e.ur5e", level="WARNING"):
                    robot.send_action(
                        self.action([0.0, 0.0, 0.08, 0.0, 0.0, 0.0], gripper=1.0)
                    )
                np.testing.assert_allclose(calls.sent[0], q_meas)
                self.assertEqual(calls.grips, [True])

    def test_invalid_fk_or_tcp_fails_closed_before_servoj(self):
        robot, calls = self.make_robot(
            fk=lambda q: [float("nan")] * 6,
        )
        with self.assertRaisesRegex(RuntimeError, "forward kinematics"):
            robot.send_action(self.action([0.0, 0.0, 0.2, 0.0, 0.0, 0.0]))
        self.assertEqual(calls.sent, [])

        robot2, calls2 = self.make_robot(
            q_meas=[0.0, 0.0, 0.11, 0.0, 0.0, 0.0],
            tcp_meas=[0.0, 0.0, float("nan"), 0.0, 0.0, 0.0],
        )
        with self.assertRaisesRegex(RuntimeError, "TCP pose"):
            robot2.send_action(self.action([0.0, 0.0, 0.08, 0.0, 0.0, 0.0]))
        self.assertEqual(calls2.sent, [])

    def test_workspace_warning_is_rate_limited(self):
        robot, _ = self.make_robot(q_meas=[0.0, 0.0, 0.11, 0.0, 0.0, 0.0])
        with (
            patch("lerobot_robot_ur5e.ur5e.time.monotonic", side_effect=[10.0, 10.5, 12.5]),
            self.assertLogs("lerobot_robot_ur5e.ur5e", level="WARNING") as logs,
        ):
            for _ in range(3):
                robot.send_action(self.action([0.0, 0.0, 0.08, 0.0, 0.0, 0.0]))
        self.assertEqual(len(logs.output), 2)

    def test_go_home_is_exempt_from_workspace_bounds(self):
        robot, _ = self.make_robot(q_meas=[0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
        robot.config.home_joints_deg = [0.0] * 6
        moved = []
        robot.rtde_c.moveJ = lambda *args: moved.append(args[0]) or True
        robot.rtde_c.getForwardKinematics = lambda q: self.fail("go_home called FK")
        robot.go_home()
        self.assertEqual(len(moved), 1)


if __name__ == "__main__":
    unittest.main()
