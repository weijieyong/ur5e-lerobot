"""LeRobot plugin: UR5e + Robotiq 2F-85 robot (`--robot.type=ur5e`) and SpaceMouse teleoperator
(`--teleop.type=spacemouse`). Auto-imported by LeRobot CLIs because the package name starts with
`lerobot_robot_`."""

from .config_ur5e import SpaceMouseConfig, UR5eConfig
from .spacemouse import SpaceMouse
from .ur5e import UR5e

__all__ = ["UR5e", "UR5eConfig", "SpaceMouse", "SpaceMouseConfig"]
