"""Configs for the UR5e + gripper (Robotiq 2F-85 or OnRobot RG2) robot and the SpaceMouse teleoperator.

Machine-specific values default to environment variables, which `pixi run` loads
from config/local/machine.env (see config/machine.env.example). Any field can
still be overridden on the command line, e.g. `--robot.ip=...`.
"""

import math
import os
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def require_ip(ip: str) -> str:
    """Return the controller address, or fail before any connection attempt when it is unset."""
    if not ip:
        raise ValueError(
            "UR5E_IP is not set: add the controller address to config/local/machine.env "
            "or pass --robot.ip / --teleop.robot_ip"
        )
    return ip


def _env_floats(name: str) -> list[float]:
    raw = _env(name)
    return [float(v) for v in raw.replace(",", " ").split()] if raw else []


def _env_int(name: str) -> int | None:
    raw = _env(name)
    return int(raw) if raw else None


def _env_gripper_force() -> int | None:
    for name in ("GRIPPER_FORCE", "ROLLOUT_GRIPPER_FORCE"):
        value = _env_int(name)
        if value is not None:
            return value
    return None


DEFAULT_GRIPPER_FORCE = {"robotiq": 64, "onrobot_rg2": 128}


def default_gripper_type() -> str:
    return _env("GRIPPER_TYPE") or "robotiq"


def _default_cameras() -> dict[str, CameraConfig]:
    w, h, fps = (
        int(_env("CAMERA_WIDTH", "640")),
        int(_env("CAMERA_HEIGHT", "480")),
        int(_env("CAMERA_FPS", "30")),
    )
    cams: dict[str, CameraConfig] = {}
    for key, var in (
        ("front", "FRONT_CAMERA_SERIAL"),
        ("wrist", "WRIST_CAMERA_SERIAL"),
    ):
        if _env(var):
            prefix = key.upper()
            cams[key] = RealSenseCameraConfig(
                serial_number_or_name=_env(var),
                fps=fps,
                width=w,
                height=h,
                exposure=_env_int(f"{prefix}_CAMERA_EXPOSURE"),
                gain=_env_int(f"{prefix}_CAMERA_GAIN"),
                white_balance=_env_int(f"{prefix}_CAMERA_WHITE_BALANCE"),
            )
    return cams


@RobotConfig.register_subclass("ur5e")
@dataclass
class UR5eConfig(RobotConfig):
    ip: str = field(default_factory=lambda: _env("UR5E_IP"))
    cameras: dict[str, CameraConfig] = field(default_factory=_default_cameras)
    camera_max_age_ms: int = field(
        default_factory=lambda: int(_env("CAMERA_MAX_AGE_MS", "100"))
    )
    timing_log_dir: str = field(default_factory=lambda: _env("TIMING_LOG_DIR"))

    home_joints_deg: list[float] = field(
        default_factory=lambda: _env_floats("UR5E_HOME_JOINTS_DEG")
    )
    home_speed_rad_s: float = 0.3

    control_period_s: float = field(
        default_factory=lambda: 1 / float(_env("CONTROL_FPS", "15"))
    )
    servo_lookahead_s: float = 0.1
    servo_gain: float = 300.0

    max_joint_step_rad: float = 0.05

    gripper_type: str = field(default_factory=default_gripper_type)
    gripper_port: int | None = None
    gripper_speed: int = 128
    gripper_force: int | None = field(default_factory=_env_gripper_force)
    gripper_close_width_mm: float = field(
        default_factory=lambda: float(_env("GRIPPER_CLOSE_WIDTH_MM", "8"))
    )

    @property
    def gripper_enabled(self) -> bool:
        return self.gripper_type != "none"

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "control_period_s",
            "max_joint_step_rad",
            "home_speed_rad_s",
            "camera_max_age_ms",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.home_joints_deg and (
            len(self.home_joints_deg) != 6
            or not all(map(math.isfinite, self.home_joints_deg))
        ):
            raise ValueError(
                "Home pose must contain six finite joint angles in degrees"
            )
        if not self.gripper_enabled:
            return
        if self.gripper_type not in DEFAULT_GRIPPER_FORCE:
            raise ValueError("GRIPPER_TYPE must be robotiq, onrobot_rg2 or none")
        if self.gripper_force is None:
            self.gripper_force = DEFAULT_GRIPPER_FORCE[self.gripper_type]
        if not all(
            0 <= value <= 255 for value in (self.gripper_speed, self.gripper_force)
        ):
            raise ValueError("Gripper speed and force must be in [0, 255]")
        if (
            not math.isfinite(self.gripper_close_width_mm)
            or self.gripper_close_width_mm <= 0
        ):
            raise ValueError("GRIPPER_CLOSE_WIDTH_MM must be finite and above 0 mm")


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseConfig(TeleoperatorConfig):
    robot_ip: str = field(default_factory=lambda: _env("UR5E_IP"))

    axes: str = field(default_factory=lambda: _env("SPACEMOUSE_AXES", "xyz_yaw"))
    neutral_rotvec: list[float] = field(
        default_factory=lambda: _env_floats("SPACEMOUSE_NEUTRAL_ROTVEC")
    )
    tilt_limit_deg: float = field(
        default_factory=lambda: float(_env("SPACEMOUSE_TILT_LIMIT_DEG", "10"))
    )

    max_linear_speed_m_s: float = field(
        default_factory=lambda: float(_env("SPACEMOUSE_LINEAR_SPEED_M_S", "0.10"))
    )
    max_angular_speed_rad_s: float = field(
        default_factory=lambda: float(_env("SPACEMOUSE_ANGULAR_SPEED_RAD_S", "0.5"))
    )
    deadband: float = 0.1
    full_scale: float = 350.0

    operator_yaw_deg: float = field(
        default_factory=lambda: float(_env("SPACEMOUSE_OPERATOR_YAW_DEG", "0"))
    )
    axis_map: list[str] = field(
        default_factory=lambda: _env("SPACEMOUSE_AXIS_MAP", "+0 -1 -2 +3 -4 -5").split()
    )

    max_lead_m: float = 0.02
    x_min_m: float | None = field(
        default_factory=lambda: (
            float(_env("UR5E_X_MIN_M")) if _env("UR5E_X_MIN_M") else None
        )
    )
    x_max_m: float | None = field(
        default_factory=lambda: (
            float(_env("UR5E_X_MAX_M")) if _env("UR5E_X_MAX_M") else None
        )
    )
    y_min_m: float | None = field(
        default_factory=lambda: (
            float(_env("UR5E_Y_MIN_M")) if _env("UR5E_Y_MIN_M") else None
        )
    )
    y_max_m: float | None = field(
        default_factory=lambda: (
            float(_env("UR5E_Y_MAX_M")) if _env("UR5E_Y_MAX_M") else None
        )
    )
    z_min_m: float | None = field(
        default_factory=lambda: (
            float(_env("UR5E_Z_MIN_M")) if _env("UR5E_Z_MIN_M") else None
        )
    )
    max_ik_jump_rad: float = 0.5

    close_button: int = field(
        default_factory=lambda: int(_env("SPACEMOUSE_CLOSE_BUTTON", "0"))
    )
    open_button: int = field(
        default_factory=lambda: int(_env("SPACEMOUSE_OPEN_BUTTON", "1"))
    )
    home_key: str = "h"
    stop_on_escape: bool = False

    def __post_init__(self):
        valid = len(self.axis_map) == 6 and all(
            len(token) == 2 and token[0] in "+-" and token[1] in "012345"
            for token in self.axis_map
        )
        if not valid or len({token[1] for token in self.axis_map}) != 6:
            raise ValueError(
                "SPACEMOUSE_AXIS_MAP must be six signed, distinct channels, e.g. +0 -1 -2 +3 -4 -5"
            )
        if self.axes not in ("xyz_yaw", "bounded", "full"):
            raise ValueError("SPACEMOUSE_AXES must be xyz_yaw, bounded or full")
        if self.neutral_rotvec and (
            len(self.neutral_rotvec) != 3
            or not all(map(math.isfinite, self.neutral_rotvec))
        ):
            raise ValueError(
                "SPACEMOUSE_NEUTRAL_ROTVEC must contain three finite rotation-vector radians"
            )
        if self.axes == "bounded" and not self.neutral_rotvec:
            raise ValueError(
                "Bounded mode needs a neutral orientation; run pixi run set-neutral first"
            )
        if not math.isfinite(self.tilt_limit_deg) or not 0 < self.tilt_limit_deg < 90:
            raise ValueError(
                "SPACEMOUSE_TILT_LIMIT_DEG must be between 0 and 90 degrees (exclusive)"
            )
        if (
            not (0 <= self.close_button <= 7 and 0 <= self.open_button <= 7)
            or self.close_button == self.open_button
        ):
            raise ValueError(
                "SpaceMouse open/close buttons must be distinct indices in [0, 7]"
            )
        for name in (
            "max_linear_speed_m_s",
            "max_angular_speed_rad_s",
            "full_scale",
            "max_lead_m",
            "max_ik_jump_rad",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.deadband < 1 or not math.isfinite(self.operator_yaw_deg):
            raise ValueError("Invalid SpaceMouse deadband or operator yaw")
        for name in ("x_min_m", "x_max_m", "y_min_m", "y_max_m", "z_min_m"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite or empty")
        for axis in ("x", "y"):
            low, high = getattr(self, f"{axis}_min_m"), getattr(self, f"{axis}_max_m")
            if low is not None and high is not None and low >= high:
                raise ValueError(f"{axis}_min_m must be less than {axis}_max_m")
