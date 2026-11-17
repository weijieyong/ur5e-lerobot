"""Read-only hardware check: never uploads a control script, never moves the arm or gripper.

Prints robot joints/TCP/safety state, gripper status and round-trip latency, camera frame
shape and rate, then streams SpaceMouse axes/buttons so you can check signs and buttons.

    pixi run hw-check [--spacemouse-seconds 10]
"""

import argparse
from contextlib import ExitStack
import math
import os
import socket
import time

import numpy as np

from lerobot_robot_ur5e.config_ur5e import require_ip


def check_robot(ip: str) -> None:
    import rtde_receive

    r = rtde_receive.RTDEReceiveInterface(ip)
    try:
        q = r.getActualQ()
        print(
            f"[robot] {ip} safety_mode={r.getSafetyMode()} protective_stop={r.isProtectiveStopped()}"
        )
        print(
            "[robot] joints (deg), base -> wrist 3:",
            " ".join(f"{math.degrees(v):.1f}" for v in q),
        )
        tcp = np.asarray(r.getActualTCPPose())
        print("[robot] TCP pose [m, rotvec]:", np.round(tcp, 4).tolist())
        from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig
        from lerobot_robot_ur5e.spacemouse import relative_rpy, rotvec_to_mat

        config = SpaceMouseConfig()
        if config.axes == "bounded":
            angles = relative_rpy(
                rotvec_to_mat(tcp[3:]), rotvec_to_mat(np.asarray(config.neutral_rotvec))
            )
            print(
                f"[robot] neutral-relative roll/pitch/yaw (deg): {np.round(np.degrees(angles), 2).tolist()} "
                f"tilt limits=+/-{config.tilt_limit_deg} deg; base-Z yaw free"
            )
        if r.isProtectiveStopped() or r.isEmergencyStopped():
            raise RuntimeError(
                "Clear the protective/e-stop on the pendant before teleoperation"
            )
    finally:
        r.disconnect()
    # Dashboard query only: never power on, release brakes, load or play a program.
    with socket.create_connection((ip, 29999), timeout=2.0) as dashboard:
        with dashboard.makefile("rb") as stream:
            stream.readline()  # greeting
            dashboard.sendall(b"is in remote control\n")
            remote = stream.readline().decode().strip().lower()
    print(f"[robot] remote control: {remote}")
    if remote != "true":
        raise RuntimeError("Select Remote Control on the pendant before teleoperation")


def check_gripper() -> None:
    from lerobot_robot_ur5e.config_ur5e import UR5eConfig
    from lerobot_robot_ur5e.ur5e import make_gripper

    g = make_gripper(
        UR5eConfig(cameras={})
    )  # never activate(): that may move the fingers
    try:
        summary = g.check()
        t0 = time.perf_counter()
        for _ in range(20):
            position = g.position()
        ms = (time.perf_counter() - t0) / 20 * 1000
        print(
            f"[gripper] {summary} position={position:.3f}  read round trip: {ms:.1f} ms"
        )
    finally:
        g.close()


def check_cameras(seconds: float = 2.0, preview: bool = False) -> None:
    from lerobot_robot_ur5e.cameras import make_cameras
    from lerobot_robot_ur5e.config_ur5e import _default_cameras

    cams = make_cameras(_default_cameras())
    if set(cams) != {"front", "wrist"}:
        raise ValueError("Configure both FRONT_CAMERA_SERIAL and WRIST_CAMERA_SERIAL")
    with ExitStack() as cleanup:
        if preview:
            if not os.environ.get("DISPLAY"):
                raise RuntimeError("Camera preview needs a terminal on the X11 desktop")
            import cv2

            cleanup.callback(cv2.destroyAllWindows)
        for key, cam in cams.items():
            cleanup.callback(
                lambda cam=cam: cam.disconnect() if cam.is_connected else None
            )
            cam.connect()
            print(
                f"[camera {key}] configured serial={cam.config.serial_number_or_name} "
                f"exposure={cam.config.exposure} gain={cam.config.gain} white_balance={cam.config.white_balance}"
            )
        counts = dict.fromkeys(cams, 0)
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            for key, cam in cams.items():
                img = cam.async_read()
                counts[key] += 1
                if preview:
                    cv2.imshow(key, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if preview and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        elapsed = time.perf_counter() - start
        for key in cams:
            print(
                f"[camera {key}] sampled_frames={counts[key]} sample_rate={counts[key] / elapsed:.1f} Hz "
                "(sequential reads; preview can reduce sampling rate)"
            )


def check_spacemouse(seconds: float) -> None:
    from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig
    from lerobot_robot_ur5e.spacemouse import SpaceMouse

    m = SpaceMouse(SpaceMouseConfig())
    try:
        m.hid.open()
        print(
            f"[spacemouse] {m.hid.product} map={m.config.axis_map} axes={m.config.axes} yaw={m.config.operator_yaw_deg} deg"
        )
        print(
            "Move one axis at a time. v=(base x,y,z) m/s; w=(base x,y,z) rad/s. No robot connection or motion."
        )
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            raw, buttons = m.hid.get_state()
            v, w = m.map_axes(raw)
            print(
                f"\r  raw={raw.astype(int).tolist()} v={np.round(v, 3)} w={np.round(w, 3)} "
                f"close={bool(buttons & (1 << m.config.close_button))} open={bool(buttons & (1 << m.config.open_button))}    ",
                end="",
                flush=True,
            )
            time.sleep(0.05)
        print()
    finally:
        m.disconnect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("robot", "gripper", "cameras", "spacemouse"))
    ap.add_argument("--spacemouse-seconds", type=float, default=10)
    ap.add_argument("--camera-seconds", type=float, default=2)
    ap.add_argument(
        "--preview", action="store_true", help="Show RGB camera windows (X11 desktop)"
    )
    args = ap.parse_args()
    if not all(
        math.isfinite(value) and value > 0
        for value in (args.spacemouse_seconds, args.camera_seconds)
    ):
        ap.error("Check durations must be finite and positive")
    failures = 0
    for name, fn in (
        ("robot", lambda: check_robot(require_ip(os.environ.get("UR5E_IP", "")))),
        ("gripper", check_gripper),
        ("cameras", lambda: check_cameras(args.camera_seconds, args.preview)),
        ("spacemouse", lambda: check_spacemouse(args.spacemouse_seconds)),
    ):
        if args.only and name != args.only:
            continue
        try:
            fn()
        except Exception as e:  # keep going so one missing device doesn't hide the rest
            failures += 1
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    raise SystemExit(1 if failures else 0)
