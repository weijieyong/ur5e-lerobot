"""Scripted physics teacher teleoperator for UR5e MuJoCo data collection ONLY.

Uses damped least-squares (DLS) Jacobian IK on a private MuJoCo MjData copy so it never
mutates simulator physics state. Exists strictly for collecting teacher evidence;
never runs during simulation rollout and never counts as learned policy success.
"""

from __future__ import annotations

import mujoco
import numpy as np

from lerobot.teleoperators import Teleoperator

from ..ur5e import GRIPPER, JOINTS
from .config_sim import UR5eSimTeacherConfig
from .env import ACTIVE_SIM_ENVS, UR5ePickPlaceEnv

# Desired gripper pinch orientation: local +Z pointing straight down (-Z world),
# jaw opening axis aligned cleanly with world Y so jaws straddle the cube symmetrically.
TARGET_PINCH_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

PHASES = (
    "APPROACH_ABOVE_CUBE",
    "DESCEND_TO_CUBE",
    "CLOSE_GRIPPER",
    "LIFT_CUBE",
    "MOVE_TO_TRAY",
    "LOWER_TO_TRAY",
    "OPEN_GRIPPER",
    "RETREAT_AND_SETTLE",
)


class UR5eSimTeacher(Teleoperator):
    """Deterministic scripted physics teacher driving `SimRobot(env=ur5e_sim)`."""

    config_class = UR5eSimTeacherConfig
    name = "ur5e_sim_teacher"

    def __init__(self, config: UR5eSimTeacherConfig) -> None:
        super().__init__(config)
        self.config = config
        self._connected = False
        self._env: UR5ePickPlaceEnv | None = None
        self._ik_data: mujoco.MjData | None = None
        self._episode_count: int = -1
        self._phase_idx: int = 0
        self._phase_steps: int = 0
        self._hold_qpos: np.ndarray | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys((*JOINTS, GRIPPER), float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
        self._env = None
        self._ik_data = None

    def _resolve_env(self) -> UR5ePickPlaceEnv:
        env = ACTIVE_SIM_ENVS.get(self.config.env_id)
        if env is None:
            raise RuntimeError(
                f"No active UR5ePickPlaceEnv registered for env_id={self.config.env_id!r}"
            )
        if env is not self._env:
            self._env = env
            self._ik_data = mujoco.MjData(env.model)
            self._episode_count = -1
        return env

    def _solve_ik(
        self,
        env: UR5ePickPlaceEnv,
        q_seed: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        iters: int = 24,
    ) -> np.ndarray:
        """Compute damped least-squares 6-DoF IK for the pinch site on a private MjData."""
        assert self._ik_data is not None
        ik_data = self._ik_data
        model = env.model
        ik_data.qpos[:] = env.data.qpos
        ik_data.qpos[env._arm_qpos_adr] = q_seed

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        damping = 0.04
        arm_dofs = env._arm_dof_adr

        for _ in range(iters):
            mujoco.mj_kinematics(model, ik_data)
            mujoco.mj_comPos(model, ik_data)
            cur_pos = ik_data.site_xpos[env._pinch_site_id]
            cur_rot = ik_data.site_xmat[env._pinch_site_id].reshape(3, 3)

            pos_err = target_pos - cur_pos
            rot_err = 0.5 * (
                np.cross(cur_rot[:, 0], target_rot[:, 0])
                + np.cross(cur_rot[:, 1], target_rot[:, 1])
                + np.cross(cur_rot[:, 2], target_rot[:, 2])
            )
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                break

            mujoco.mj_jacSite(model, ik_data, jacp, jacr, env._pinch_site_id)
            jac = np.vstack([jacp[:, arm_dofs], jacr[:, arm_dofs]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + (damping**2) * np.eye(6), err)
            dq = np.clip(dq, -0.25, 0.25)
            q_next = np.clip(
                ik_data.qpos[env._arm_qpos_adr] + dq,
                env._arm_ctrl_low,
                env._arm_ctrl_high,
            )
            ik_data.qpos[env._arm_qpos_adr] = q_next

        return ik_data.qpos[env._arm_qpos_adr].copy()

    def _step_toward(self, q_cur: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        step = self.config.max_joint_step_rad
        delta = q_goal - q_cur
        max_abs = float(np.max(np.abs(delta)))
        if max_abs > step:
            delta = delta * (step / max_abs)
        return q_cur + delta

    def get_action(self) -> dict[str, float]:
        env = self._resolve_env()
        state = env.teacher_state()

        # Reset phase state machine automatically on every new episode.
        if state["episode_count"] != self._episode_count:
            self._episode_count = state["episode_count"]
            self._phase_idx = 0
            self._phase_steps = 0
            self._hold_qpos = state["qpos_arm"].copy()

        q_cur = state["qpos_arm"]
        pinch_pos = state["pinch_pos"]
        cube_pos = state["cube_pos"]
        tray_xy = state["tray_center_xy"]
        tray_top_z = float(state["tray_top_z"])
        cube_half = float(state["cube_half_size"])

        phase = PHASES[self._phase_idx]
        self._phase_steps += 1
        gripper_cmd = 0.0

        above_cube = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.12])
        grasp_cube = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.002])
        lift_cube = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.14])
        above_tray = np.array([tray_xy[0], tray_xy[1], tray_top_z + cube_half + 0.15])
        place_tray = np.array([tray_xy[0], tray_xy[1], tray_top_z + cube_half + 0.018])
        retreat_tray = np.array([tray_xy[0], tray_xy[1], tray_top_z + cube_half + 0.14])

        if phase == "APPROACH_ABOVE_CUBE":
            target_pos = above_cube
            gripper_cmd = 0.0
            if np.linalg.norm(pinch_pos - target_pos) < 0.012 or self._phase_steps >= 40:
                self._advance_phase(q_cur)
        elif phase == "DESCEND_TO_CUBE":
            target_pos = grasp_cube
            gripper_cmd = 0.0
            if np.linalg.norm(pinch_pos - target_pos) < 0.008 or self._phase_steps >= 35:
                self._advance_phase(q_cur)
        elif phase == "CLOSE_GRIPPER":
            target_pos = None
            gripper_cmd = 1.0
            if self._phase_steps >= 14:
                self._advance_phase(q_cur)
        elif phase == "LIFT_CUBE":
            target_pos = np.array(
                [grasp_cube[0], grasp_cube[1], max(lift_cube[2], 0.18)]
            )
            gripper_cmd = 1.0
            if pinch_pos[2] >= 0.165 or self._phase_steps >= 35:
                self._advance_phase(q_cur)
        elif phase == "MOVE_TO_TRAY":
            target_pos = above_tray
            gripper_cmd = 1.0
            if np.linalg.norm(pinch_pos - target_pos) < 0.015 or self._phase_steps >= 55:
                self._advance_phase(q_cur)
        elif phase == "LOWER_TO_TRAY":
            target_pos = place_tray
            gripper_cmd = 1.0
            if np.linalg.norm(pinch_pos - target_pos) < 0.010 or self._phase_steps >= 35:
                self._advance_phase(q_cur)
        elif phase == "OPEN_GRIPPER":
            target_pos = None
            gripper_cmd = 0.0
            if self._phase_steps >= 10:
                self._advance_phase(q_cur)
        else:  # RETREAT_AND_SETTLE
            target_pos = retreat_tray
            gripper_cmd = 0.0

        if target_pos is None:
            q_cmd = self._hold_qpos if self._hold_qpos is not None else q_cur
        else:
            q_goal = self._solve_ik(env, q_cur, target_pos, TARGET_PINCH_ROT)
            q_cmd = self._step_toward(q_cur, q_goal)

        action = {name: float(val) for name, val in zip(JOINTS, q_cmd, strict=True)}
        action[GRIPPER] = float(gripper_cmd)
        return action

    def _advance_phase(self, q_cur: np.ndarray) -> None:
        if self._phase_idx < len(PHASES) - 1:
            self._phase_idx += 1
            self._phase_steps = 0
            self._hold_qpos = q_cur.copy()
