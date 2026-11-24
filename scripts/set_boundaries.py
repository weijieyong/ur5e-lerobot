"""Teach five TCP workspace faces with the SpaceMouse; leave the ceiling unbounded."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time

import numpy as np

from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig, UR5eConfig
from lerobot_robot_ur5e.spacemouse import SpaceMouse, TiltLimitError
from lerobot_robot_ur5e.ur5e import UR5e


FACES = (
    ("Floor / base Z minimum", "UR5E_Z_MIN_M", 2, 1),
    ("Left side (base X)", "UR5E_X_MIN_M", 0, 1),
    ("Right side (base X)", "UR5E_X_MAX_M", 0, -1),
    ("Front side (base Y)", "UR5E_Y_MIN_M", 1, 1),
    ("Back side (base Y)", "UR5E_Y_MAX_M", 1, -1),
)


def inset_bounds(captured, margin):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("UR5E_WORKSPACE_MARGIN_M must be finite and nonnegative")
    captured = dict(captured)
    for axis in ("X", "Y"):
        low, high = f"UR5E_{axis}_MIN_M", f"UR5E_{axis}_MAX_M"
        if captured[low] > captured[high]:
            captured[low], captured[high] = captured[high], captured[low]
    bounds = {key: round(captured[key] + sign * margin, 6) for _, key, _, sign in FACES}
    if not all(math.isfinite(value) for value in bounds.values()):
        raise ValueError("TCP boundary measurements must be finite")
    for axis in ("X", "Y"):
        if bounds[f"UR5E_{axis}_MIN_M"] >= bounds[f"UR5E_{axis}_MAX_M"]:
            raise ValueError(
                f"{axis} bounds are too narrow for the clearance; config not saved"
            )
    return bounds


def capture_bounds(robot, mouse, fps):
    from pynput import keyboard

    capture = threading.Event()

    def on_release(key):
        if key == keyboard.Key.enter:
            capture.set()

    captured = {}
    recovering = False
    with keyboard.Listener(on_release=on_release):
        for label, key, axis, _ in FACES:
            print(
                f"\nMove TCP to: {label}. Release the puck, then Enter to capture. Esc cancels.",
                flush=True,
            )
            while True:
                tick = time.monotonic()
                try:
                    action = mouse.get_action()
                except TiltLimitError:
                    capture.clear()
                    if not recovering:
                        print(
                            "Tilt outside range: press h with a clear path or restore posture on the pendant.",
                            flush=True,
                        )
                        recovering = True
                    time.sleep(1 / fps)
                    continue
                recovering = False
                if capture.is_set():
                    capture.clear()
                    pose = np.asarray(robot.rtde_r.getActualTCPPose(), dtype=float)
                    if pose.shape != (6,) or not np.isfinite(pose).all():
                        raise ValueError(
                            "Robot returned an invalid TCP pose; config not saved"
                        )
                    captured[key] = float(pose[axis])
                    print(f"Captured base {'XYZ'[axis]}={pose[axis]:.6f} m", flush=True)
                    break
                robot.send_action(action)
                time.sleep(max(0, 1 / fps - (time.monotonic() - tick)))
    return captured


def save_bounds(config, original, bounds):
    if config.read_text() != original:
        raise ValueError("machine.env changed during teaching; config not overwritten")
    updated = original
    for key, value in bounds.items():
        line = f"{key}={value:.6f}"
        updated, count = re.subn(
            rf"(?m)^[ \t]*(?:export[ \t]+)?{key}=.*$", lambda _: line, updated
        )
        if not count:
            updated = updated.rstrip("\n") + "\n" + line + "\n"
    backup = config.with_name(f"{config.name}.before-boundaries-{time.time_ns()}")
    with tempfile.NamedTemporaryFile(
        mode="w", dir=config.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(updated)
    try:
        temporary.chmod(config.stat().st_mode & 0o777)
        shutil.copy2(config, backup)
        if config.read_text() != original:
            raise ValueError(
                "machine.env changed during teaching; config not overwritten"
            )
        temporary.replace(config)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(
        os.environ.get("PIXI_PROJECT_ROOT", Path(__file__).resolve().parents[1])
    )
    config = root / "config/local/machine.env"
    original = config.read_text()
    margin = float(os.environ.get("UR5E_WORKSPACE_MARGIN_M", ".01"))
    linear = float(os.environ.get("BOUNDARY_LINEAR_SPEED_M_S", ".03"))
    angular = float(os.environ.get("BOUNDARY_ANGULAR_SPEED_RAD_S", ".15"))
    fps = float(os.environ.get("CONTROL_FPS", "15"))
    if not all(math.isfinite(v) and v > 0 for v in (linear, angular, fps)):
        raise ValueError("Teaching speeds and CONTROL_FPS must be finite and positive")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("UR5E_WORKSPACE_MARGIN_M must be finite and nonnegative")
    teaching = SpaceMouseConfig(
        x_min_m=None,
        x_max_m=None,
        y_min_m=None,
        y_max_m=None,
        z_min_m=None,
        stop_on_escape=True,
    )
    teaching.max_linear_speed_m_s = min(teaching.max_linear_speed_m_s, linear)
    teaching.max_angular_speed_rad_s = min(teaching.max_angular_speed_rad_s, angular)
    print(
        "TEACHING TCP BOUNDARIES — base axes, metres; no ceiling.\n"
        "Stop other robot-control commands first. Previous floor/side guards are disabled here.\n"
        "Tilt and joint guards remain active; h is an explicit joint home move.\n"
        f"SpaceMouse speed caps: {teaching.max_linear_speed_m_s:g} m/s, "
        f"{teaching.max_angular_speed_rad_s:g} rad/s. Clearance: {margin:g} m."
    )
    try:
        if input("Enter to enable SpaceMouse teaching; Ctrl+C to cancel: ").strip():
            return
        robot = UR5e(UR5eConfig(cameras={}, timing_log_dir=""))
        mouse = SpaceMouse(teaching)
        try:
            robot.connect()
            mouse.connect()
            captured = capture_bounds(robot, mouse, fps)
        finally:
            try:
                mouse.disconnect()
            finally:
                robot.disconnect()
        bounds = inset_bounds(captured, margin)
        print("\nProposed bounds (clearance already applied; ceiling unbounded):")
        for key, value in bounds.items():
            print(f"  {key}={value:.6f}")
        if input("Save these bounds? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Not saved.")
            return
        backup = save_bounds(config, original, bounds)
        print(
            f"Saved {config}; backup: {backup}. Restart teleop/record to load the bounds."
        )
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled; config not saved.")


if __name__ == "__main__":
    main()
