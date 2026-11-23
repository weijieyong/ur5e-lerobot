"""UR5e arm (ur_rtde, servoJ) + gripper + RealSense cameras as a LeRobot robot.

The gripper is selected by GRIPPER_TYPE: a Robotiq 2F-85 (Robotiq URCap socket) or an
OnRobot RG2 (OnRobot URCap XML-RPC); both are reached through the UR controller's IP.

Observation: joint_0..5.pos (rad, measured), gripper.pos (0 = open, 1 = closed, measured), camera images.
Action:      joint_0..5.pos (rad, absolute targets), gripper.pos (command: < 0.5 open, >= 0.5 closed).
"""

import json
import logging
import math
import socket
import time
import xmlrpc.client
from pathlib import Path

import numpy as np
from lerobot.robots import Robot

from .cameras import make_cameras
from .config_ur5e import UR5eConfig, require_ip

log = logging.getLogger(__name__)

JOINTS = [f"joint_{i}.pos" for i in range(6)]
GRIPPER = "gripper.pos"
# Before homing: wait for gripper position (0 open .. 1 closed) to reach this, or give up without moving.
HOME_OPEN_POSITION = 0.05
HOME_OPEN_TIMEOUT_S = 3.0

# Connected robots by IP, so the SpaceMouse teleoperator in the same process can use the
# robot's UR connection for IK and homing (only one control connection to the UR is possible).
CONNECTED: dict[str, "UR5e"] = {}


# Every gripper client provides: activate() (make ready; may move the fingers), check() (read-only
# status summary, raises if not ready), position() (0 = open .. 1 = closed), grip(closed), close().


class RobotiqSocket:
    """Minimal client for the Robotiq URCap gripper socket (ASCII 'SET'/'GET' protocol)."""

    def __init__(self, host: str, port: int = 63352, speed: int = 128, force: int = 64):
        self.speed, self.force = speed, force
        self.sock = socket.create_connection((host, port), timeout=2.0)

    def _cmd(self, msg: str) -> str:
        self.sock.sendall(msg.encode() + b"\n")
        return self.sock.recv(1024).decode().strip()

    def get(self, var: str) -> int:
        reply = self._cmd(f"GET {var}")  # e.g. "POS 3"
        fields = reply.split()
        if len(fields) != 2 or fields[0] != var:
            raise RuntimeError(f"Unexpected gripper reply to GET {var}: {reply!r}")
        return int(fields[1])

    def set(self, **values: int) -> None:
        body = " ".join(f"{k} {v}" for k, v in values.items())
        if self._cmd(f"SET {body}") != "ack":
            raise RuntimeError(f"gripper did not ack 'SET {body}'")

    def activate(self) -> None:
        if self.get("ACT") == 1 and self.get("STA") == 3:
            return
        log.info("Activating gripper (it will open and close once)")
        self.set(ACT=1)
        t0 = time.monotonic()
        while not (self.get("ACT") == 1 and self.get("STA") == 3):
            if time.monotonic() - t0 > 10:
                raise TimeoutError("gripper activation timed out")
            time.sleep(0.05)
        self.set(GTO=1)

    def check(self) -> str:
        status = {v: self.get(v) for v in ("ACT", "STA", "POS", "OBJ", "FLT")}
        if status["ACT"] != 1 or status["STA"] != 3 or status["FLT"] != 0:
            raise RuntimeError(
                f"Robotiq {status}: activate the gripper / clear its fault on the pendant"
            )
        return f"Robotiq {status}"

    def position(self) -> float:
        return self.get("POS") / 255.0

    def grip(self, closed: bool) -> None:
        self.set(POS=255 if closed else 0, SPE=self.speed, FOR=self.force, GTO=1)

    def close(self) -> None:
        self.sock.close()


class _TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 2.0
        return connection


class OnRobotRG2:
    """OnRobot RG2 on the UR tool flange, via the OnRobot URCap's XML-RPC server on the controller.

    Calls take ~1 ms; rg_grip returns without waiting for the fingers (measured 2026-11-12).
    """

    STROKE_MM = 110.0
    MAX_FORCE_N = 40.0
    SAFETY_FLAGS = (
        "s1_pushed",
        "s1_triggered",
        "s2_pushed",
        "s2_triggered",
        "safety_failed",
    )

    def __init__(
        self,
        host: str,
        port: int = 41414,
        force: int = 64,
        close_width_mm: float = 8.0,
        device: int = 0,
    ):
        # The trailing slash matters: without it xmlrpc.client posts to /RPC2 and the URCap returns 404.
        self.url = f"http://{host}:{port}/"
        self.device = device
        self.force_n = force / 255 * self.MAX_FORCE_N
        # Closing to 0 mm on nothing trips the S2 safety switch and latches the gripper until power-cycled.
        self.close_width_mm = close_width_mm
        self.open_width_mm: float | None = None
        self._proxy = xmlrpc.client.ServerProxy(self.url, transport=_TimeoutTransport())

    def check(self) -> str:
        p, d = self._proxy, self.device
        width, offset = float(p.rg_get_width(d)), float(p.rg_get_fingertip_offset(d))
        faults = [name for name in self.SAFETY_FLAGS if getattr(p, f"rg_get_{name}")(d)]
        summary = f"RG2 width={width:.1f} mm fingertip_offset={offset:.1f} mm safety_faults={faults or 'none'}"
        if width < 0 or offset < 0:
            raise RuntimeError(
                f"{summary}: gripper not responding; power on the arm so the tool flange powers it"
            )
        if faults:
            raise RuntimeError(
                f"{summary}: safety latched; power-cycle the tool from the OnRobot toolbar on the pendant"
            )
        self.open_width_mm = self.STROKE_MM - 2 * offset
        if not 0 < self.close_width_mm < self.open_width_mm:
            raise RuntimeError(
                f"RG2 close width {self.close_width_mm} mm must be in (0, {self.open_width_mm:.1f}) mm"
            )
        return summary

    def activate(self) -> None:
        self.check()  # the RG2 needs no activation motion; refuse to start unpowered or latched

    def position(self) -> float:
        width = float(self._proxy.rg_get_width(self.device))
        if width < 0:
            raise RuntimeError(
                "RG2 stopped responding (width reads -999); check tool power on the pendant"
            )
        return min(max(1.0 - width / self.open_width_mm, 0.0), 1.0)

    def grip(self, closed: bool) -> None:
        # The URCap ignores rg_grip while the fingers move; stop first so a reversal takes effect at once.
        if self._proxy.rg_get_busy(self.device):
            self._proxy.rg_stop(self.device)
        self._proxy.rg_grip(
            self.device,
            self.close_width_mm if closed else self.open_width_mm,
            self.force_n,
        )

    def close(self) -> None:
        pass  # each XML-RPC call opens its own HTTP connection


def make_gripper(config: UR5eConfig) -> RobotiqSocket | OnRobotRG2:
    """Connect to the configured gripper without moving it (activate() may move it)."""
    if config.gripper_type == "onrobot_rg2":
        return OnRobotRG2(
            require_ip(config.ip),
            config.gripper_port or 41414,
            config.gripper_force,
            config.gripper_close_width_mm,
        )
    return RobotiqSocket(
        require_ip(config.ip),
        config.gripper_port or 63352,
        config.gripper_speed,
        config.gripper_force,
    )


class UR5e(Robot):
    config_class = UR5eConfig
    name = "ur5e"
    supports_command_hold = True

    def __init__(self, config: UR5eConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras(config.cameras)
        self.rtde_c = None
        self.rtde_r = None
        self.gripper: RobotiqSocket | OnRobotRG2 | None = None
        self._gripper_closed: bool | None = None
        self._servo_active = False
        self._observation_timing = {}
        self._timing_file = None

    # --- features -------------------------------------------------------------------------
    @property
    def _state_ft(self) -> dict[str, type]:
        return {**{j: float for j in JOINTS}, GRIPPER: float}

    @property
    def observation_features(self) -> dict:
        cams = {k: (c.height, c.width, 3) for k, c in self.config.cameras.items()}
        return {**self._state_ft, **cams}

    @property
    def action_features(self) -> dict:
        return self._state_ft

    # --- lifecycle ------------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self.rtde_c is not None and self.rtde_c.isConnected()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        import rtde_control
        import rtde_receive

        if self.is_connected:
            return
        ip = require_ip(self.config.ip)
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
            self.check_safety()
            # Uploads the ur_rtde control script; the pendant must be in Remote Control mode.
            self.rtde_c = rtde_control.RTDEControlInterface(ip)
            self.gripper = make_gripper(self.config)
            self.gripper.activate()
            for cam in self.cameras.values():
                cam.connect()
        except BaseException:
            self.disconnect()
            raise
        self._gripper_closed = None
        self._servo_active = False
        CONNECTED[ip] = self
        log.info("UR5e connected at %s", ip)

    def disconnect(self) -> None:
        CONNECTED.pop(self.config.ip, None)
        cleanup = []
        if self.rtde_c is not None:
            cleanup.extend(
                (
                    ("servoStop", self.rtde_c.servoStop),
                    ("stopScript", self.rtde_c.stopScript),
                    ("control disconnect", self.rtde_c.disconnect),
                )
            )
        if self.rtde_r is not None:
            cleanup.append(("receive disconnect", self.rtde_r.disconnect))
        if self.gripper is not None:
            cleanup.append(("gripper close", self.gripper.close))
        for cam in self.cameras.values():
            if cam.is_connected:
                cleanup.append(("camera disconnect", cam.disconnect))
        if self._timing_file is not None:
            cleanup.append(("timing log close", self._timing_file.close))
        for name, operation in cleanup:
            try:
                if operation() is False:
                    log.error("UR5e cleanup %s returned failure", name)
            except Exception:
                log.exception(
                    "UR5e cleanup %s failed; continuing remaining cleanup", name
                )
        self._servo_active = False
        self.rtde_c = self.rtde_r = self.gripper = self._timing_file = None

    # --- io ---------------------------------------------------------------------------------
    def check_safety(self) -> None:
        if self.rtde_r.isProtectiveStopped() or self.rtde_r.isEmergencyStopped():
            raise RuntimeError(
                "UR5e is protective- or emergency-stopped; clear it on the pendant first"
            )

    def measured_joints(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualQ())

    def get_observation(self) -> dict:
        start = time.perf_counter()
        q = self.measured_joints()
        state_read_end = time.perf_counter()
        obs = {j: float(v) for j, v in zip(JOINTS, q)}
        obs[GRIPPER] = self.gripper.position()
        camera_ready = {}
        for key, cam in self.cameras.items():
            obs[key] = cam.read_latest(max_age_ms=self.config.camera_max_age_ms)
            if self.config.timing_log_dir:
                # LeRobot 0.7.0 timestamps the buffer after host postprocessing, not at exposure.
                # If a newer frame arrived after our read, don't associate its timestamp with this image.
                with cam.frame_lock:
                    camera_ready[key] = (
                        cam.latest_timestamp
                        if cam.latest_color_frame is obs[key]
                        else None
                    )
        self._observation_timing = {
            "observation_start_s": start,
            "state_read_end_s": state_read_end,
            "observation_end_s": time.perf_counter(),
            "camera_ready_s": camera_ready,
        }
        return obs

    def _write_timing(self, event: dict) -> None:
        if not self.config.timing_log_dir:
            return
        if self._timing_file is None:
            directory = Path(self.config.timing_log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"ur5e-{time.time_ns()}.jsonl"
            self._timing_file = path.open("x", buffering=1)
            log.info("Robot timing log: %s", path)
        self._timing_file.write(json.dumps(event) + "\n")

    def send_action(self, action: dict) -> dict:
        start = time.perf_counter()
        q_req = np.array([action[j] for j in JOINTS], dtype=float)
        if not np.isfinite(q_req).all() or not math.isfinite(float(action[GRIPPER])):
            raise ValueError("Joint targets and gripper command must be finite")
        self.check_safety()
        q_meas = self.measured_joints()
        if q_meas.shape != (6,) or not np.isfinite(q_meas).all():
            raise ValueError("UR5e returned invalid measured joint angles")
        step = self.config.max_joint_step_rad
        q_cmd = q_meas + np.clip(q_req - q_meas, -step, step)
        accepted = self.rtde_c.servoJ(
            q_cmd.tolist(), 0.0, 0.0, self.config.control_period_s,
            self.config.servo_lookahead_s, self.config.servo_gain,
        )  # fmt: skip
        if accepted is False:
            raise RuntimeError(
                "UR5e rejected servoJ; check remote control and the pendant safety state"
            )
        self._servo_active = True

        closed = action[GRIPPER] >= 0.5
        if closed != self._gripper_closed:
            self.gripper.grip(closed)
            self._gripper_closed = closed

        # LeRobot 0.7.0's record_loop still ignores send_action's return value and saves
        # action_values. Its default identity processors share this dict with action_values;
        # update it in place to record the clipped target and binary gripper command.
        action.update(
            {**{j: float(v) for j, v in zip(JOINTS, q_cmd)}, GRIPPER: float(closed)}
        )
        self._write_timing(
            {
                "event": "sample",
                **self._observation_timing,
                "send_start_s": start,
                "send_end_s": time.perf_counter(),
                "control_period_s": self.config.control_period_s,
                "clipped": bool(np.any(q_req != q_cmd)),
                "requested_joints": q_req.tolist(),
                "sent_joints": q_cmd.tolist(),
            }
        )
        return action

    def hold(self) -> None:
        """Stop active servoJ streaming cleanly and verify robot safety."""
        self.check_safety()
        if not self._servo_active:
            return
        start = time.perf_counter()
        if self.rtde_c.servoStop() is False:
            raise RuntimeError(
                "UR5e rejected servoStop; check remote control and the pendant safety state"
            )
        self._servo_active = False
        self._write_timing(
            {"event": "hold", "start_s": start, "end_s": time.perf_counter()}
        )

    def go_home(self) -> None:
        """Open the gripper where the arm is, then a blocking, slow joint move to the configured home pose."""
        if len(self.config.home_joints_deg) != 6:
            log.warning(
                "No home pose configured (UR5E_HOME_JOINTS_DEG); ignoring home request"
            )
            return
        self.check_safety()
        start = time.perf_counter()
        self.rtde_c.servoStop()
        self._servo_active = False
        # Release any held object before moving so every trial starts at home with the gripper open.
        self.gripper.grip(False)
        self._gripper_closed = False
        deadline = time.monotonic() + HOME_OPEN_TIMEOUT_S
        while self.gripper.position() > HOME_OPEN_POSITION:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Gripper did not open before homing; check it on the pendant"
                )
            time.sleep(0.02)
        home = [math.radians(d) for d in self.config.home_joints_deg]
        if self.rtde_c.moveJ(home, self.config.home_speed_rad_s, 0.5) is False:
            raise RuntimeError("UR5e rejected home move; check the pendant")
        self._write_timing(
            {"event": "home", "start_s": start, "end_s": time.perf_counter()}
        )
