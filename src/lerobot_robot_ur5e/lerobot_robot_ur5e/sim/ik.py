"""Pinch-site inverse kinematics for scripted MuJoCo motion."""

import mujoco
import numpy as np

from .scene import ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, PINCH_SITE_NAME

# Local +Z points down and the jaw opening axis aligns with world Y.
DOWNWARD_PINCH_ROT = np.diag([1.0, -1.0, -1.0])


class PinchIK:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.arm_qpos = np.array([model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES])
        self.arm_dofs = np.array([model.joint(name).dofadr[0] for name in ARM_JOINT_NAMES])
        self.arm_actuators = np.array([model.actuator(name).id for name in ARM_ACTUATOR_NAMES])
        self.site_id = model.site(PINCH_SITE_NAME).id
        self.ctrl_low, self.ctrl_high = model.actuator_ctrlrange[self.arm_actuators].T

    def solve(
        self, qpos: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray
    ) -> np.ndarray:
        """Solve on private MuJoCo data seeded with the current joint positions."""
        model = self.model
        data = self.data
        data.qpos[:] = qpos
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        regularizer = 0.04**2 * np.eye(6)

        for _ in range(24):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            cur_pos = data.site_xpos[self.site_id]
            cur_rot = data.site_xmat[self.site_id].reshape(3, 3)
            pos_err = target_pos - cur_pos
            rot_err = 0.5 * np.cross(cur_rot.T, target_rot.T).sum(axis=0)
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                break

            mujoco.mj_jacSite(model, data, jacp, jacr, self.site_id)
            jac = np.vstack([jacp[:, self.arm_dofs], jacr[:, self.arm_dofs]])
            err = np.concatenate([pos_err, rot_err])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + regularizer, err)
            data.qpos[self.arm_qpos] = np.clip(
                data.qpos[self.arm_qpos] + np.clip(dq, -0.25, 0.25),
                self.ctrl_low,
                self.ctrl_high,
            )

        return data.qpos[self.arm_qpos].copy()
