"""Save home, floor or neutral TCP orientation using read-only RTDE queries."""

import argparse
import math
import os
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", choices=("home", "floor", "neutral"))
    args = parser.parse_args()
    root = Path(
        os.environ.get("PIXI_PROJECT_ROOT", Path(__file__).resolve().parents[1])
    )
    config = root / "config/local/machine.env"
    if not config.is_file():
        parser.error("Run pixi run setup first.")
    margin = float(os.environ.get("UR5E_FLOOR_MARGIN_M", "0.01"))
    if not math.isfinite(margin) or margin < 0:
        parser.error("UR5E_FLOOR_MARGIN_M must be finite and nonnegative.")

    import rtde_receive

    ip = os.environ.get("UR5E_IP", "")
    if not ip:
        parser.error("UR5E_IP is not set; add it to config/local/machine.env")
    receiver = rtde_receive.RTDEReceiveInterface(ip)
    try:
        values = (
            receiver.getActualQ()
            if args.reference == "home"
            else receiver.getActualTCPPose()
        )
    finally:
        receiver.disconnect()
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        parser.error(
            "Robot returned an invalid joint/TCP measurement; config was not changed."
        )

    if args.reference == "home":
        key = "UR5E_HOME_JOINTS_DEG"
        value = '"' + " ".join(f"{math.degrees(angle):.6f}" for angle in values) + '"'
    elif args.reference == "neutral":
        key = "SPACEMOUSE_NEUTRAL_ROTVEC"
        value = '"' + " ".join(f"{angle:.12g}" for angle in values[3:]) + '"'
    else:
        key, value = "UR5E_Z_MIN_M", f"{values[2] + margin:.6f}"
        print(f"Measured TCP z={values[2]:.6f} m; added clearance={margin:.6f} m")
    line = f"{key}={value}"
    original = config.read_text()
    updated, count = re.subn(
        rf"(?m)^[ \t]*(?:export[ \t]+)?{key}=.*$", lambda _: line, original
    )
    if not count:
        updated = original.rstrip("\n") + "\n" + line + "\n"
    config.write_text(updated)
    print(
        f"Saved {line} in {config}; no motion commanded. Restart teleop/record to load it."
    )


if __name__ == "__main__":
    main()
