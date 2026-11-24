"""Minimal Gymnasium MuJoCo pick-and-place environment for UR5e + Robotiq 2F-85."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .config_sim import (
    SIM_DEFAULT_FPS,
    SIM_DEFAULT_HEIGHT,
    SIM_DEFAULT_WIDTH,
    SIM_TASK_DESCRIPTION,
    SIM_TASK_ID,
)
from .scene import (
    ARM_ACTUATOR_NAMES,
    ARM_JOINT_NAMES,
    CUBE_HALF_SIZE,
    CUBE_START_POS,
    GRIPPER_ACTUATOR_NAME,
    GRIPPER_DRIVER_JOINTS,
    GRIPPER_DRIVER_MAX_RAD,
    GRIPPER_PREFIX,
    HOME_QPOS,
    PINCH_SITE_NAME,
    TABLE_TOP_Z,
    TRAY_CENTER_XY,
    TRAY_INNER_HALF_XY,
    TRAY_TOP_Z,
    build_pick_place_model,
)

# Active in-process simulation envs keyed by env_id so the scripted collection teacher
# can read simulator ground truth without exposing it in learned-policy observations.
ACTIVE_SIM_ENVS: dict[str, "UR5ePickPlaceEnv"] = {}

LIFT_CLEARANCE_M = 0.035
SETTLE_LINEAR_SPEED_M_S = 0.025
SETTLE_ANGULAR_SPEED_RAD_S = 0.25
TRAY_REST_Z_TOLERANCE_M = 0.012


@dataclass(frozen=True)
class EpisodePhysicsStatus:
    """Explicit physical manipulation milestones (teacher/evaluation only, never in obs)."""

    left_pad_contact: bool
    right_pad_contact: bool
    robot_contact: bool
    table_contact: bool
    tray_contact: bool
    cube_pos: tuple[float, float, float]
    cube_lin_speed: float
    cube_ang_speed: float
    grasped_lifted: bool
    ever_grasped_lifted: bool
    in_tray_resting: bool
    settled_steps: int
    is_success: bool


class UR5ePickPlaceEnv(gym.Env):
    """Fixed-layout UR5e + Robotiq 2F-85 tabletop red-cube-to-blue-tray pick-and-place."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": SIM_DEFAULT_FPS}

    def __init__(
        self,
        *,
        fps: int = SIM_DEFAULT_FPS,
        observation_width: int = SIM_DEFAULT_WIDTH,
        observation_height: int = SIM_DEFAULT_HEIGHT,
        terminate_on_success: bool = True,
        settle_steps: int = 10,
        env_id: str = "default",
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__()
        self.fps = int(fps)
        self.observation_width = int(observation_width)
        self.observation_height = int(observation_height)
        self.terminate_on_success = bool(terminate_on_success)
        self.settle_steps = int(settle_steps)
        self.env_id = str(env_id)
        self.render_mode = render_mode
        self.task = SIM_TASK_ID
        self.task_description = SIM_TASK_DESCRIPTION

        self.model = build_pick_place_model(
            width=self.observation_width, height=self.observation_height
        )
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(
            self.model, height=self.observation_height, width=self.observation_width
        )

        self._substeps = max(1, int(round(1.0 / (self.fps * self.model.opt.timestep))))

        # Resolve model indices once for fast, deterministic stepping and contact inspection.
        self._arm_jnt_ids = np.array(
            [self._require_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._arm_qpos_adr = self.model.jnt_qposadr[self._arm_jnt_ids]
        self._arm_dof_adr = self.model.jnt_dofadr[self._arm_jnt_ids]
        self._arm_act_ids = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in ARM_ACTUATOR_NAMES
            ],
            dtype=np.int32,
        )
        self._arm_ctrl_low = self.model.actuator_ctrlrange[self._arm_act_ids, 0].copy()
        self._arm_ctrl_high = self.model.actuator_ctrlrange[self._arm_act_ids, 1].copy()

        self._gripper_act_id = self._require_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
        )
        self._gripper_driver_qpos_adr = np.array(
            [
                self.model.jnt_qposadr[self._require_id(mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in GRIPPER_DRIVER_JOINTS
            ],
            dtype=np.int32,
        )
        self._pinch_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE, PINCH_SITE_NAME
        )
        self._cube_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self._cube_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "red_cube_geom")
        self._cube_jnt_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT, "red_cube_freejoint"
        )
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_jnt_id])
        self._cube_dof_adr = int(self.model.jnt_dofadr[self._cube_jnt_id])
        self._table_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self._tray_floor_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM, "tray_floor"
        )
        self._tray_geom_ids = {
            gid
            for gid in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith(
                "tray_"
            )
        }
        self._left_pad_geom_ids = {
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, f"{GRIPPER_PREFIX}left_pad1"),
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, f"{GRIPPER_PREFIX}left_pad2"),
        }
        self._right_pad_geom_ids = {
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, f"{GRIPPER_PREFIX}right_pad1"),
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, f"{GRIPPER_PREFIX}right_pad2"),
        }
        self._robot_geom_ids = {
            gid
            for gid in range(self.model.ngeom)
            if gid != self._cube_geom_id
            and gid != self._table_geom_id
            and gid not in self._tray_geom_ids
            and (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
            != "room_floor"
        }

        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "pixels": spaces.Dict(
                    {
                        "front": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.observation_height, self.observation_width, 3),
                            dtype=np.uint8,
                        ),
                        "wrist": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.observation_height, self.observation_width, 3),
                            dtype=np.uint8,
                        ),
                    }
                ),
            }
        )
        action_low = np.concatenate([self._arm_ctrl_low, [0.0]]).astype(np.float32)
        action_high = np.concatenate([self._arm_ctrl_high, [1.0]]).astype(np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, shape=(7,), dtype=np.float32
        )

        self.episode_count = 0
        self._step_count = 0
        self._last_gripper_closed = False
        self._ever_grasped_lifted = False
        self._settled_steps = 0
        self._last_status: EpisodePhysicsStatus | None = None

        ACTIVE_SIM_ENVS[self.env_id] = self

    def _require_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise RuntimeError(f"MuJoCo model is missing required {obj_type}: {name!r}")
        return int(obj_id)

    def _measured_arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self._arm_qpos_adr].copy()

    def _measured_gripper_closure(self) -> float:
        drivers = self.data.qpos[self._gripper_driver_qpos_adr]
        return float(np.clip(np.mean(drivers) / GRIPPER_DRIVER_MAX_RAD, 0.0, 1.0))

    def _render_camera(self, name: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=name)
        frame = self._renderer.render()
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _get_obs(self) -> dict[str, Any]:
        # Learned-policy observations strictly exclude ground-truth object/tray poses & contacts.
        state = np.empty(7, dtype=np.float32)
        state[:6] = self._measured_arm_qpos()
        state[6] = self._measured_gripper_closure()
        return {
            "agent_pos": state,
            "pixels": {
                "front": self._render_camera("front"),
                "wrist": self._render_camera("wrist"),
            },
        }

    def _evaluate_physics_status(self) -> EpisodePhysicsStatus:
        left_pad_contact = False
        right_pad_contact = False
        robot_contact = False
        table_contact = False
        tray_contact = False

        for idx in range(self.data.ncon):
            con = self.data.contact[idx]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == self._cube_geom_id:
                other = g2
            elif g2 == self._cube_geom_id:
                other = g1
            else:
                continue
            if other in self._left_pad_geom_ids:
                left_pad_contact = True
            if other in self._right_pad_geom_ids:
                right_pad_contact = True
            if other in self._robot_geom_ids:
                robot_contact = True
            if other == self._table_geom_id:
                table_contact = True
            if other in self._tray_geom_ids:
                tray_contact = True

        cube_pos = self.data.xpos[self._cube_body_id].copy()
        cube_vel = self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6]
        cube_lin_speed = float(np.linalg.norm(cube_vel[:3]))
        cube_ang_speed = float(np.linalg.norm(cube_vel[3:6]))

        lifted_height = TABLE_TOP_Z + CUBE_HALF_SIZE + LIFT_CLEARANCE_M
        grasped_lifted = bool(
            left_pad_contact
            and right_pad_contact
            and not table_contact
            and not tray_contact
            and cube_pos[2] >= lifted_height
        )
        if grasped_lifted:
            self._ever_grasped_lifted = True

        rel_xy = np.abs(cube_pos[:2] - TRAY_CENTER_XY)
        max_xy = TRAY_INNER_HALF_XY - CUBE_HALF_SIZE * 0.5
        expected_tray_z = TRAY_TOP_Z + CUBE_HALF_SIZE
        in_tray_resting = bool(
            not self._last_gripper_closed
            and not robot_contact
            and tray_contact
            and np.all(rel_xy <= max_xy)
            and abs(float(cube_pos[2]) - expected_tray_z) <= TRAY_REST_Z_TOLERANCE_M
            and cube_lin_speed <= SETTLE_LINEAR_SPEED_M_S
            and cube_ang_speed <= SETTLE_ANGULAR_SPEED_RAD_S
        )

        if self._ever_grasped_lifted and in_tray_resting:
            self._settled_steps += 1
        else:
            self._settled_steps = 0

        is_success = self._settled_steps >= self.settle_steps
        return EpisodePhysicsStatus(
            left_pad_contact=left_pad_contact,
            right_pad_contact=right_pad_contact,
            robot_contact=robot_contact,
            table_contact=table_contact,
            tray_contact=tray_contact,
            cube_pos=(float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])),
            cube_lin_speed=cube_lin_speed,
            cube_ang_speed=cube_ang_speed,
            grasped_lifted=grasped_lifted,
            ever_grasped_lifted=self._ever_grasped_lifted,
            in_tray_resting=in_tray_resting,
            settled_steps=self._settled_steps,
            is_success=is_success,
        )

    def _status_info(self, status: EpisodePhysicsStatus) -> dict[str, Any]:
        return {
            "is_success": status.is_success,
            "metrics": {
                "cube_z": status.cube_pos[2],
                "grasped_lifted": float(status.grasped_lifted),
                "ever_grasped_lifted": float(status.ever_grasped_lifted),
                "in_tray_resting": float(status.in_tray_resting),
                "settled_steps": float(status.settled_steps),
                "left_pad_contact": float(status.left_pad_contact),
                "right_pad_contact": float(status.right_pad_contact),
            },
        }

    def teacher_state(self) -> dict[str, Any]:
        """Simulator truth exposed ONLY to the scripted physics teacher (never in obs)."""
        return {
            "episode_count": self.episode_count,
            "step_count": self._step_count,
            "qpos_arm": self._measured_arm_qpos(),
            "gripper_closure": self._measured_gripper_closure(),
            "pinch_pos": self.data.site_xpos[self._pinch_site_id].copy(),
            "pinch_xmat": self.data.site_xmat[self._pinch_site_id].reshape(3, 3).copy(),
            "cube_pos": self.data.xpos[self._cube_body_id].copy(),
            "tray_center_xy": TRAY_CENTER_XY.copy(),
            "tray_top_z": TRAY_TOP_Z,
            "cube_half_size": CUBE_HALF_SIZE,
            "status": self._last_status,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Fixed starting layout (resets happen strictly between episodes).
        self.data.qpos[self._arm_qpos_adr] = HOME_QPOS
        self.data.ctrl[self._arm_act_ids] = HOME_QPOS
        self.data.ctrl[self._gripper_act_id] = 0.0
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = CUBE_START_POS
        self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Let the initial pose and resting cube settle cleanly for 0.1 s before step 0.
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.episode_count += 1
        self._step_count = 0
        self._last_gripper_closed = False
        self._ever_grasped_lifted = False
        self._settled_steps = 0
        self._last_status = self._evaluate_physics_status()
        return self._get_obs(), self._status_info(self._last_status)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_arr.shape != (7,) or not np.isfinite(action_arr).all():
            raise ValueError(f"Action must contain 7 finite values, got {action_arr!r}")

        arm_target = np.clip(action_arr[:6], self._arm_ctrl_low, self._arm_ctrl_high)
        gripper_closed = bool(action_arr[6] >= 0.5)
        self._last_gripper_closed = gripper_closed

        self.data.ctrl[self._arm_act_ids] = arm_target
        self.data.ctrl[self._gripper_act_id] = 255.0 if gripper_closed else 0.0

        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        status = self._evaluate_physics_status()
        self._last_status = status

        terminated = bool(self.terminate_on_success and status.is_success)
        # Episode step truncation is owned by Gymnasium TimeLimit / LocalVectorEnv.
        truncated = False
        reward = 1.0 if status.is_success else 0.0
        return self._get_obs(), reward, terminated, truncated, self._status_info(status)

    def render(self) -> np.ndarray:
        return self._render_camera("front")

    def close(self) -> None:
        if ACTIVE_SIM_ENVS.get(self.env_id) is self:
            ACTIVE_SIM_ENVS.pop(self.env_id, None)
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None
