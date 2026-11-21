"""SpaceMouse teleoperator that outputs UR5e joint targets.

Puck motion -> TCP velocity (operator frame) -> integrated target TCP pose -> UR controller IK
-> absolute joint targets. lerobot-record stores exactly this output as the action.
"""

import logging
import math
import threading
import time

import numpy as np
from lerobot.teleoperators import Teleoperator

from .config_ur5e import SpaceMouseConfig
from .ur5e import CONNECTED, GRIPPER, JOINTS

log = logging.getLogger(__name__)


class TiltLimitError(RuntimeError):
    """Operator recovery is needed before bounded teleoperation can continue."""


VENDOR_3DCONNEXION = 0x256F


# --- rotation helpers (UR poses use axis-angle rotation vectors) --------------------------------
def rotvec_to_mat(r: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = math.acos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    if math.pi - theta < 1e-6:  # near 180 deg: take axis from the symmetric part
        k = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, None))
        i = int(np.argmax(k))
        k = R[:, i] + np.eye(3)[:, i]
        return theta * k / np.linalg.norm(k)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * w / (2 * math.sin(theta))


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def relative_rpy(R: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    """ZYX angles in radians where R = Rz(yaw) Ry(pitch) Rx(roll) neutral.

    R and neutral rotate TCP coordinates into the robot base. Yaw is about base Z.
    Bounded mode stays away from the pitch +/-90 degree singularity.
    """
    delta = R @ neutral.T
    return np.array(
        [
            math.atan2(delta[2, 1], delta[2, 2]),
            math.atan2(-delta[2, 0], math.hypot(delta[0, 0], delta[1, 0])),
            math.atan2(delta[1, 0], delta[0, 0]),
        ]
    )


def limit_tilt(R: np.ndarray, neutral: np.ndarray, limit_rad: float) -> np.ndarray:
    """Clamp each relative roll/pitch component, preserving free base-Z yaw."""
    roll, pitch, yaw = relative_rpy(R, neutral)
    roll, pitch = np.clip([roll, pitch], -limit_rad, limit_rad)
    return (
        rot_z(yaw)
        @ rotvec_to_mat(np.array([0, pitch, 0]))
        @ rotvec_to_mat(np.array([roll, 0, 0]))
        @ neutral
    )


# --- HID reader ----------------------------------------------------------------------------------
def _hid():
    """hidapi's hidraw backend (uses the /dev/hidraw* udev permissions); the default `hid` module is libusb."""
    try:
        import hidraw as hid
    except ImportError:
        import hid
    return hid


class SpaceMouseHID:
    """Background reader for 3Dconnexion devices over hidraw (hidapi). Keeps the latest state."""

    def __init__(self):
        self.axes = np.zeros(6)  # raw x, y, z, roll, pitch, yaw
        self.buttons = 0
        self._dev = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.product = ""
        self._error: Exception | None = None

    @staticmethod
    def find() -> dict | None:
        hid = _hid()

        devs = [
            d
            for d in hid.enumerate(VENDOR_3DCONNEXION)
            if d.get("usage_page") == 1 and d.get("usage") == 8
        ]
        devs = devs or hid.enumerate(VENDOR_3DCONNEXION)
        return devs[0] if devs else None

    def open(self) -> None:
        hid = _hid()
        info = self.find()
        if info is None:
            raise ConnectionError(
                "No 3Dconnexion device found (check USB and the udev rule in config/udev/)"
            )
        self._dev = hid.device()
        try:
            self._dev.open_path(info["path"])
        except OSError as error:
            raise PermissionError(
                f"Found {info.get('product_string') or 'a 3Dconnexion device'} but cannot open it ({error}). "
                "Install the udev rule: sudo cp config/udev/99-3dconnexion.rules /etc/udev/rules.d/ && "
                "sudo udevadm control --reload-rules && sudo udevadm trigger, then replug it."
            ) from error
        self.product = info.get("product_string", "")
        self._stop.clear()
        self._error = None
        self.axes[:] = 0
        self.buttons = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            self._read_reports()
        except Exception as error:
            self._error = error
            self.axes[:] = 0
            self.buttons = 0

    def _read_reports(self) -> None:
        while not self._stop.is_set():
            data = self._dev.read(64, 50)
            if not data:
                continue
            rid = data[0]
            vals = [
                int.from_bytes(bytes(data[i : i + 2]), "little", signed=True)
                for i in range(1, len(data) - 1, 2)
            ]
            if rid == 1 and len(vals) >= 6:  # newer devices: all six axes in one report
                self.axes[:] = vals[:6]
            elif rid == 1:
                self.axes[:3] = vals[:3]
            elif rid == 2:
                self.axes[3:] = vals[:3]
            elif rid == 3:
                self.buttons = data[1]

    def get_state(self) -> tuple[np.ndarray, int]:
        if self._error is not None:
            raise ConnectionError(
                "SpaceMouse input failed; stopping teleoperation"
            ) from self._error
        if self._thread is None or not self._thread.is_alive():
            raise ConnectionError("SpaceMouse reader is not running")
        return self.axes.copy(), self.buttons

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._dev is not None:
            self._dev.close()
        self._dev = self._thread = None


class SpaceMouse(Teleoperator):
    config_class = SpaceMouseConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseConfig):
        super().__init__(config)
        self.config = config
        self.hid = SpaceMouseHID()
        self._connected = False
        self._target_pose: np.ndarray | None = (
            None  # [x, y, z, rx, ry, rz] in robot base frame
        )
        self._q_target: np.ndarray | None = None
        self._gripper_cmd: float | None = None
        self._last_t: float | None = None
        self._home_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._key_listener = None

    @property
    def action_features(self) -> dict:
        return {**{j: float for j in JOINTS}, GRIPPER: float}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        self._stop_requested.clear()
        self.reset_reference()
        try:
            self.hid.open()
            self._start_home_key()
        except BaseException:
            self.disconnect()
            raise
        self._connected = True
        log.info(
            "SpaceMouse connected (%s), axes=%s", self.hid.product, self.config.axes
        )

    def reset_reference(self) -> None:
        """Hold measured pose/latest gripper command after policy; drop queued home input."""
        self._target_pose = self._q_target = self._last_t = self._gripper_cmd = None
        closed = getattr(CONNECTED.get(self.config.robot_ip), "_gripper_closed", None)
        if closed is not None:
            # An object can stop closing below POS=127; keep the close intent, not that threshold.
            self._gripper_cmd = float(closed)
        self._home_requested.clear()

    def _start_home_key(self) -> None:
        try:
            from pynput import keyboard
        except Exception as error:
            raise RuntimeError(
                "Keyboard unavailable; run teleop/record in a terminal on the X11 desktop"
            ) from error

        def on_press(key):
            if key == keyboard.Key.esc:
                self._stop_requested.set()
            if getattr(key, "char", None) == self.config.home_key:
                self._home_requested.set()

        self._key_listener = keyboard.Listener(on_press=on_press)
        self._key_listener.start()

    def disconnect(self) -> None:
        try:
            self.hid.close()
        except Exception:
            log.exception("SpaceMouse HID cleanup failed")
        if self._key_listener is not None:
            try:
                self._key_listener.stop()
            except Exception:
                log.exception("SpaceMouse keyboard cleanup failed")
            self._key_listener = None
        self._connected = False  # let LeRobot's following robot.disconnect() run even after an input cleanup error

    def send_feedback(self, feedback: dict) -> None:
        pass

    def check_tilt_range(self, tcp_pose=None) -> None:
        """Read/validate current tilt without moving the robot or changing neutral."""
        c = self.config
        if c.axes != "bounded":
            return
        if tcp_pose is None:
            robot = CONNECTED.get(c.robot_ip)
            if robot is None:
                raise RuntimeError(f"No connected UR5e at {c.robot_ip} in this process")
            tcp_pose = robot.rtde_r.getActualTCPPose()
        pose = np.asarray(tcp_pose, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise RuntimeError("UR5e returned an invalid TCP pose")
        neutral = rotvec_to_mat(np.asarray(c.neutral_rotvec))
        roll, pitch, _ = relative_rpy(rotvec_to_mat(pose[3:]), neutral)
        if max(abs(roll), abs(pitch)) > math.radians(c.tilt_limit_deg) + 1e-6:
            raise TiltLimitError(
                "Current TCP orientation is outside the bounded tilt range. "
                "Return home with h (clear path), or restore the intended posture on the pendant."
            )

    # --- core ---------------------------------------------------------------------------------
    def map_axes(self, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert a raw HID snapshot to the configured base-frame velocity; no robot access."""
        c = self.config
        raw = np.array([float(m[0] + "1") * axes[int(m[1:])] for m in c.axis_map])
        u = np.clip(raw / c.full_scale, -1, 1)
        u = np.where(
            np.abs(u) < c.deadband,
            0.0,
            (np.abs(u) - c.deadband) / (1 - c.deadband) * np.sign(u),
        )
        if c.axes == "xyz_yaw":
            u[3] = u[4] = 0.0
        R_op = rot_z(math.radians(c.operator_yaw_deg))
        return R_op @ (u[:3] * c.max_linear_speed_m_s), R_op @ (
            u[3:] * c.max_angular_speed_rad_s
        )

    def get_action(self) -> dict:
        if self.config.stop_on_escape and self._stop_requested.is_set():
            raise KeyboardInterrupt
        axes, buttons = self.hid.get_state()
        robot = CONNECTED.get(self.config.robot_ip)
        if robot is None:
            raise RuntimeError(
                f"No connected UR5e at {self.config.robot_ip} in this process"
            )
        c = self.config

        if self._home_requested.is_set():
            self._home_requested.clear()
            robot.go_home()
            self._gripper_cmd = (
                0.0  # go_home opened the gripper; don't re-close a latched command
            )
            self._target_pose = None
            self._last_t = None
            axes, buttons = self.hid.get_state()
            if self.config.stop_on_escape and self._stop_requested.is_set():
                raise KeyboardInterrupt

        q_meas = robot.measured_joints()
        tcp_meas = np.asarray(robot.rtde_r.getActualTCPPose())
        neutral = (
            rotvec_to_mat(np.asarray(c.neutral_rotvec)) if c.axes == "bounded" else None
        )
        tilt_limit = math.radians(c.tilt_limit_deg)
        if self._target_pose is None and neutral is not None:
            self.check_tilt_range(tcp_meas)
        now = time.monotonic()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.2)
        self._last_t = now

        # Gripper: buttons latch a command; start from the measured state.
        if self._gripper_cmd is None:
            self._gripper_cmd = float(robot.gripper.position() > 0.5)
        if buttons & (1 << c.close_button):
            self._gripper_cmd = 1.0
        elif buttons & (1 << c.open_button):
            self._gripper_cmd = 0.0

        v, w = self.map_axes(axes)
        if self._target_pose is None or (not v.any() and not w.any()):
            # Idle (or first call): re-anchor to where the arm actually is, so nothing winds up.
            self._target_pose = tcp_meas.copy()
            self._q_target = q_meas
        else:
            pos = self._target_pose[:3] + v * dt
            lead = pos - tcp_meas[:3]
            if np.linalg.norm(lead) > c.max_lead_m:
                pos = tcp_meas[:3] + lead / np.linalg.norm(lead) * c.max_lead_m
            rc = getattr(robot, "config", None)
            lower = np.array(
                [
                    v if v is not None else -np.inf
                    for v in (
                        getattr(rc, "x_min_m", None),
                        getattr(rc, "y_min_m", None),
                        getattr(rc, "z_min_m", None),
                    )
                ]
            )
            upper = np.array(
                [
                    v if v is not None else np.inf
                    for v in (
                        getattr(rc, "x_max_m", None),
                        getattr(rc, "y_max_m", None),
                        None,
                    )
                ]
            )
            # Outside a saved boundary, allow gradual inward recovery without a snap.
            # Within the box, clamp outward motion while allowing sliding along each face.
            pos = np.clip(
                pos, np.minimum(lower, tcp_meas[:3]), np.maximum(upper, tcp_meas[:3])
            )
            R = rotvec_to_mat(w * dt) @ rotvec_to_mat(
                self._target_pose[3:]
            )  # rotate about base axes
            if neutral is not None:
                R = limit_tilt(R, neutral, tilt_limit)
            candidate = np.concatenate([pos, mat_to_rotvec(R)])

            q_ik = np.asarray(
                robot.rtde_c.getInverseKinematics(candidate.tolist(), q_meas.tolist())
            )
            if (
                q_ik.shape == (6,)
                and np.max(np.abs(q_ik - q_meas)) <= c.max_ik_jump_rad
            ):
                self._target_pose, self._q_target = candidate, q_ik
            else:
                log.warning("IK rejected (no solution or jump too large); holding")

        return {
            **{j: float(v) for j, v in zip(JOINTS, self._q_target)},
            GRIPPER: self._gripper_cmd,
        }
