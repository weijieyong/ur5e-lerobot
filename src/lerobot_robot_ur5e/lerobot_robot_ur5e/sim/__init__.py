"""UR5e + Robotiq 2F-85 MuJoCo simulation package (Gymnasium + LeRobot integration).

Importing this package registers `lerobot_robot_ur5e/UR5ePickPlace-v0` with
Gymnasium lazily (via string entry_point) without importing `mujoco` until
an environment or teacher is actually instantiated.
"""

import gymnasium as gym

from .config_sim import (
    SIM_DEFAULT_FPS,
    SIM_DEFAULT_HEIGHT,
    SIM_DEFAULT_WIDTH,
    SIM_STATE_ACTION_NAMES,
    SIM_TASK_DESCRIPTION,
    SIM_TASK_ID,
    UR5eSimEnvConfig,
    UR5eSimTeacherConfig,
)

GYM_ENV_ID = f"lerobot_robot_ur5e/{SIM_TASK_ID}"

if GYM_ENV_ID not in gym.registry:
    gym.register(
        id=GYM_ENV_ID,
        entry_point="lerobot_robot_ur5e.sim.env:UR5ePickPlaceEnv",
        max_episode_steps=240,
    )


def __getattr__(name: str):
    if name in ("ACTIVE_SIM_ENVS", "EpisodePhysicsStatus", "UR5ePickPlaceEnv"):
        from . import env

        return getattr(env, name)
    if name == "UR5eSimTeacher":
        from . import teacher

        return teacher.UR5eSimTeacher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACTIVE_SIM_ENVS",
    "EpisodePhysicsStatus",
    "GYM_ENV_ID",
    "SIM_DEFAULT_FPS",
    "SIM_DEFAULT_HEIGHT",
    "SIM_DEFAULT_WIDTH",
    "SIM_STATE_ACTION_NAMES",
    "SIM_TASK_DESCRIPTION",
    "SIM_TASK_ID",
    "UR5ePickPlaceEnv",
    "UR5eSimEnvConfig",
    "UR5eSimTeacher",
    "UR5eSimTeacherConfig",
]
