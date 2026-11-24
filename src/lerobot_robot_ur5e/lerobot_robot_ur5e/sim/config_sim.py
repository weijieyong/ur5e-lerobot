"""LeRobot EnvConfig and TeleoperatorConfig registrations for UR5e MuJoCo simulation.

Kept free of `mujoco` imports so hardware-only workflows and tests can import
`lerobot_robot_ur5e` without loading simulator binaries.
"""

from dataclasses import dataclass
from typing import Any

from lerobot.envs.configs import EnvConfig, _gym_settings
from lerobot.teleoperators import TeleoperatorConfig

from ..ur5e import GRIPPER, JOINTS

SIM_STATE_ACTION_NAMES: tuple[str, ...] = (*JOINTS, GRIPPER)
SIM_TASK_ID = "UR5ePickPlace-v0"
SIM_TASK_DESCRIPTION = "Pick up the red cube and place it on the blue tray"
SIM_DEFAULT_FPS = 15
SIM_DEFAULT_WIDTH = 640
SIM_DEFAULT_HEIGHT = 480


@EnvConfig.register_subclass("ur5e_sim")
@dataclass
class UR5eSimEnvConfig(EnvConfig):
    """Local MuJoCo UR5e + Robotiq 2F-85 fixed-layout pick-and-place environment."""

    task: str = SIM_TASK_ID
    fps: int = SIM_DEFAULT_FPS
    episode_length: int = 240
    observation_width: int = SIM_DEFAULT_WIDTH
    observation_height: int = SIM_DEFAULT_HEIGHT
    terminate_on_success: bool = True
    settle_steps: int = 10
    env_id: str = "default"

    def __post_init__(self) -> None:
        if self.fps != SIM_DEFAULT_FPS:
            raise ValueError(
                f"UR5e simulation contract requires fps={SIM_DEFAULT_FPS}, got {self.fps}"
            )
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.observation_width <= 0 or self.observation_height <= 0:
            raise ValueError("camera observation dimensions must be positive")
        if self.settle_steps <= 0:
            raise ValueError("settle_steps must be positive")

    @property
    def package_name(self) -> str:
        return "lerobot_robot_ur5e"

    @property
    def gym_kwargs(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "max_episode_steps": self.episode_length,
            "observation_width": self.observation_width,
            "observation_height": self.observation_height,
            "terminate_on_success": self.terminate_on_success,
            "settle_steps": self.settle_steps,
            "env_id": self.env_id,
        }

    def sim_settings(self) -> dict[str, Any]:
        return {
            **_gym_settings(
                self,
                "position",
                SIM_STATE_ACTION_NAMES,
                SIM_STATE_ACTION_NAMES,
            ),
            "robot_type": "ur5e_sim",
        }


@TeleoperatorConfig.register_subclass("ur5e_sim_teacher")
@dataclass
class UR5eSimTeacherConfig(TeleoperatorConfig):
    """Scripted physics teacher for simulation data collection only (never rollout)."""

    env_id: str = "default"
    max_joint_step_rad: float = 0.05

    def __post_init__(self) -> None:
        if self.max_joint_step_rad <= 0:
            raise ValueError("max_joint_step_rad must be positive")
