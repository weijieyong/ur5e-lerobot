"""LeRobot plugin: UR5e + Robotiq 2F-85 / OnRobot RG2 robot (`--robot.type=ur5e`),
SpaceMouse teleoperator (`--teleop.type=spacemouse`), and local MuJoCo simulation
configs (`--robot.env.type=ur5e_sim`, `--teleop.type=ur5e_sim_teacher`).
Auto-imported by LeRobot CLIs because the package name starts with `lerobot_robot_`.
"""

from .config_ur5e import SpaceMouseConfig, UR5eConfig
from .sim.config_sim import UR5eSimEnvConfig, UR5eSimTeacherConfig
from .spacemouse import SpaceMouse
from .ur5e import UR5e

__all__ = [
    "UR5e",
    "UR5eConfig",
    "SpaceMouse",
    "SpaceMouseConfig",
    "UR5eSimEnvConfig",
    "UR5eSimTeacherConfig",
]
