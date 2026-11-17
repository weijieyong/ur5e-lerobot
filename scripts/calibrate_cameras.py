"""Calibrate manual RealSense controls against a user-selected white-paper patch.

Run `pixi run calibrate-cameras` after closing hw-check/record/teleop.
Select a clean patch INSIDE each sheet, then press Enter. Esc cancels.
Review the recommendations, then answer y/n to update machine.env (default n).
The final tuned images are shown before the confirmation prompt.
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import tempfile

import numpy as np


CONTROLS = ("exposure", "gain", "white_balance")


def measure(image, roi):
    x, y, w, h = roi
    if (
        x < 0
        or y < 0
        or min(w, h) < 10
        or x + w > image.shape[1]
        or y + h > image.shape[0]
    ):
        raise ValueError("Select a patch at least 10x10 pixels, fully inside the image")
    pixels = image[y : y + h, x : x + w].reshape(-1, 3).astype(float)
    rgb = np.median(pixels, axis=0)[::-1]
    return {
        "rgb": rgb.tolist(),
        "brightness": float(np.median(rgb)),
        "clipped_fraction": float(np.mean(np.any(pixels >= 250, axis=1))),
        "colour_error": float((rgb.max() - rgb.min()) / max(rgb.mean(), 1)),
    }


def read_config(text):
    values = {}
    for line in text.splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens and tokens[0] == "export":
            tokens.pop(0)
        if tokens and "=" in tokens[0]:
            key, value = tokens[0].split("=", 1)
            values[key] = value
    return values


def confirm_and_save(path, original, settings):
    print("Suggested camera settings:")
    for key, value in settings.items():
        print(f"{key}={value}")
    while True:
        try:
            answer = (
                input(f"Overwrite camera settings in {path}? [y/N] ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("", "n", "no"):
            print("machine.env was not changed.")
            return
        if answer in ("y", "yes"):
            break
        print("Please answer y or n.")

    path = path.resolve(strict=True)
    if path.read_text() != original:
        raise RuntimeError(
            "Config changed during calibration; refusing to overwrite it. Run calibration again"
        )
    updated = original
    for key, value in settings.items():
        pattern = rf"(?m)^([ \t]*(?:export[ \t]+)?{re.escape(key)}[ \t]*=[ \t]*)[^#\r\n]*?([ \t]*(?:#[^\r\n]*)?)$"
        updated, count = re.subn(
            pattern, lambda match: f"{match[1]}{value}{match[2]}", updated
        )
        if not count:
            updated += (
                "" if not updated or updated.endswith("\n") else "\n"
            ) + f"{key}={value}\n"
    mode = path.stat().st_mode & 0o777
    backup = path.with_name(
        path.name + ".bak." + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    with backup.open("x") as stream:
        os.chmod(backup, mode)
        stream.write(original)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(updated)
        os.chmod(temporary, mode)
        if path.read_text() != original:
            raise RuntimeError("Config changed before saving; refusing to overwrite it")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(
        f"Updated {path}. Backup saved to {backup}. Restart hw-check/record to load the new settings."
    )


def show_tuned_preview(output):
    import cv2

    images = [
        cv2.imread(str(output / f"{key}-after.png")) for key in ("front", "wrist")
    ]
    if any(image is None for image in images):
        raise OSError(f"Could not load final tuned images from {output}")
    title = "Final tuned cameras: FRONT | WRIST - press any key to continue to y/n"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        cv2.imshow(title, np.hstack(images))
        cv2.resizeWindow(title, 1600, 600)
        print(
            "Review the final tuned images. Press any key in the preview to continue to the y/n choice.",
            flush=True,
        )
        cv2.waitKey(0)
    finally:
        cv2.destroyWindow(title)


class Camera:
    """Own one RGB stream and restore its original sensor controls on exit."""

    def __init__(self, rs, serial, width, height, fps):
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.original = {}
        self.ranges = {}
        self.sensor = None
        self.started = False
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.disable_all_streams()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    def open(self):
        profile = self.pipeline.start(self.config)
        self.started = True
        device = profile.get_device()
        self.name = device.get_info(self.rs.camera_info.name)
        self.sensor = (
            device.first_depth_sensor()
            if "D405" in self.name
            else device.first_color_sensor()
        )
        for name in ("enable_auto_exposure", "enable_auto_white_balance", *CONTROLS):
            option = getattr(self.rs.option, name)
            if self.sensor.supports(option):
                self.original[name] = self.sensor.get_option(option)
        for name in CONTROLS:
            option = getattr(self.rs.option, name)
            if name not in self.original or self.sensor.is_option_read_only(option):
                raise RuntimeError(f"{self.name} does not support writable {name}")
            limits = self.sensor.get_option_range(option)
            self.ranges[name] = (limits.min, limits.max, limits.step or 1)
        for name in ("enable_auto_exposure", "enable_auto_white_balance"):
            if name in self.original:
                self.sensor.set_option(getattr(self.rs.option, name), 0)

    def set(self, name, value):
        low, high, step = self.ranges[name]
        value = low + round((max(low, min(high, value)) - low) / step) * step
        value = max(low, min(high, value))
        self.sensor.set_option(getattr(self.rs.option, name), value)
        return self.get(name)

    def get(self, name):
        return int(round(self.sensor.get_option(getattr(self.rs.option, name))))

    def frame(self):
        frame = self.pipeline.wait_for_frames(5000).get_color_frame()
        if not frame:
            raise RuntimeError("Camera returned no colour frame")
        return np.asanyarray(frame.get_data()).copy()

    def sample(self):
        # Firmware controls and queued frames need time to catch up after a write.
        for _ in range(6):
            self.frame()
        return np.median([self.frame() for _ in range(5)], axis=0).astype(np.uint8)

    def close(self):
        try:
            if self.sensor is not None:
                for name in (
                    *CONTROLS,
                    "enable_auto_exposure",
                    "enable_auto_white_balance",
                ):
                    if name in self.original:
                        try:
                            self.sensor.set_option(
                                getattr(self.rs.option, name), self.original[name]
                            )
                        except RuntimeError as error:
                            print(f"Warning: could not restore {name}: {error}")
        finally:
            if self.started:
                self.pipeline.stop()


def tune_brightness(camera, roi, target, max_exposure):
    camera.set("gain", camera.ranges["gain"][0])
    for name, ceiling in (
        ("exposure", max_exposure),
        ("gain", camera.ranges["gain"][1]),
    ):
        low, high, step = camera.ranges[name]
        high = low + math.floor((min(high, ceiling) - low) / step) * step
        best = None
        for _ in range(18):
            if low > high:
                break
            value = camera.set(name, low + math.floor((high - low) / (2 * step)) * step)
            stats = measure(camera.sample(), roi)
            brightness = stats["brightness"]
            score = abs(brightness - target) + 255 * stats["clipped_fraction"]
            if best is None or score < best[0]:
                best = (score, value)
            if abs(brightness - target) <= 2 and stats["clipped_fraction"] <= 0.01:
                break
            if brightness < target and stats["clipped_fraction"] <= 0.01:
                low = value + step
            else:
                high = value - step
        if best is None:
            raise ValueError(
                f"Exposure ceiling is below the supported minimum for {camera.name}"
            )
        camera.set(name, best[1])
        stats = measure(camera.sample(), roi)
        if stats["brightness"] >= target - 5 or name == "gain":
            break
        camera.set(name, ceiling)
        ceiling_stats = measure(camera.sample(), roi)
        if (
            ceiling_stats["clipped_fraction"] > 0.01
            or ceiling_stats["brightness"] >= target - 5
        ):
            camera.set(name, best[1])
            break


def tune_white_balance(camera, roi):
    low, high, step = camera.ranges["white_balance"]
    best = None
    for spacing in (max(step, 400), max(step, 100), step):
        start, stop = (
            (low, high)
            if best is None
            else (max(low, best[1] - 400), min(high, best[1] + 400))
        )
        if spacing == step:
            start, stop = max(low, best[1] - 100), min(high, best[1] + 100)
        candidates = sorted({start, stop, *np.arange(start, stop + 1, spacing)})
        for value in candidates:
            value = camera.set("white_balance", value)
            stats = measure(camera.sample(), roi)
            rgb = np.asarray(stats["rgb"])
            if rgb.min() < 15 or stats["clipped_fraction"] > 0.01:
                continue
            score = abs(math.log(rgb[0] / rgb[2]))
            if best is None or score < best[0]:
                best = (score, value)
        if best is None:
            raise RuntimeError(
                "Paper is too dark or clipped for white balance; select an evenly lit patch"
            )
        camera.set("white_balance", best[1])


def calibrate(camera, key, values, args, output):
    import cv2

    prefix = key.upper() + "_CAMERA_"
    for name in CONTROLS:
        if values.get(prefix + name.upper()):
            camera.set(name, int(values[prefix + name.upper()]))
    before = camera.sample()
    cv2.imwrite(str(output / f"{key}-before.png"), before)
    roi = getattr(args, key + "_roi")
    if roi is None:
        title = f"{key}: drag INSIDE white paper, Enter to accept, Esc to cancel"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 960, 720)
        roi = cv2.selectROI(title, before, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
    roi = tuple(map(int, roi))
    initial = measure(before, roi)
    max_exposure = getattr(args, key + "_max_exposure")
    if max_exposure is None:
        max_exposure = float(values.get(prefix + "EXPOSURE") or camera.get("exposure"))
    if not camera.ranges["exposure"][0] <= max_exposure <= camera.ranges["exposure"][1]:
        raise ValueError(
            f"{key} exposure ceiling outside supported range {camera.ranges['exposure']}"
        )
    print(
        f"[{key}] {camera.name}; ROI={roi}; native ranges={camera.ranges}", flush=True
    )
    print(
        f"[{key}] tuning exposure/gain (exposure ceiling {max_exposure:g})...",
        flush=True,
    )
    # A midtone patch leaves room to compare colour temperatures without clipping.
    tune_brightness(camera, roi, min(args.target, 160), max_exposure)
    print(f"[{key}] searching white balance...", flush=True)
    tune_white_balance(camera, roi)
    tune_brightness(camera, roi, args.target, max_exposure)
    after = camera.sample()
    final = measure(after, roi)
    settings = {prefix + name.upper(): camera.get(name) for name in CONTROLS}
    issues = []
    if abs(final["brightness"] - args.target) > 10:
        issues.append(
            "brightness target not reached; adjust exposure ceiling or lighting"
        )
    if final["clipped_fraction"] > 0.01:
        issues.append("paper patch has clipped highlights")
    if final["colour_error"] > 0.10:
        issues.append(
            "remaining colour cast exceeds 10%; Kelvin white balance cannot correct all tints"
        )
    x, y, w, h = roi
    cv2.rectangle(after, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(str(output / f"{key}-after.png"), after)
    print(f"[{key}] before={initial}; after={final}", flush=True)
    for key_name, value in settings.items():
        print(f"{key_name}={value}", flush=True)
    return {
        "settings": settings,
        "roi": roi,
        "before": initial,
        "after": final,
        "issues": issues,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=root / "config/local/machine.env")
    parser.add_argument(
        "--headless", action="store_true", help="No GUI; requires both ROI arguments"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=220,
        help="White paper brightness on 0-255 scale (default 220)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "outputs/camera-calibration"
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
    )
    for key in ("front", "wrist"):
        parser.add_argument(
            f"--{key}-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H")
        )
        parser.add_argument(
            f"--{key}-max-exposure",
            type=float,
            help="Ceiling in native sensor units; defaults to machine.env exposure",
        )
    args = parser.parse_args()
    if not math.isfinite(args.target) or not 100 <= args.target <= 235:
        parser.error("--target must be finite and between 100 and 235")
    for key in ("front", "wrist"):
        ceiling = getattr(args, key + "_max_exposure")
        if ceiling is not None and (not math.isfinite(ceiling) or ceiling <= 0):
            parser.error("Exposure ceilings must be finite and positive")
    if args.headless and (args.front_roi is None or args.wrist_roi is None):
        parser.error("--headless requires --front-roi X Y W H and --wrist-roi X Y W H")
    if (args.front_roi is None or args.wrist_roi is None) and not os.environ.get(
        "DISPLAY"
    ):
        parser.error(
            "ROI selection needs a desktop DISPLAY; use --headless with both ROI arguments"
        )
    try:
        original = args.env.read_text()
        values = read_config(original)
        serials = [
            values.get(key.upper() + "_CAMERA_SERIAL") for key in ("front", "wrist")
        ]
        if not all(serials) or len(set(serials)) != 2:
            raise ValueError(
                "Configure two distinct FRONT_CAMERA_SERIAL and WRIST_CAMERA_SERIAL values"
            )
        width, height, fps = (
            int(values.get(key, default))
            for key, default in (
                ("CAMERA_WIDTH", 640),
                ("CAMERA_HEIGHT", 480),
                ("CAMERA_FPS", 30),
            )
        )
        import pyrealsense2 as rs

        context = rs.context()
        connected = {
            device.get_info(rs.camera_info.serial_number)
            for device in context.query_devices()
        }
        if not set(serials) <= connected:
            raise RuntimeError(
                f"Configured cameras not found. Visible serials: {sorted(connected)}"
            )
        args.output.mkdir(parents=True, exist_ok=True)
        results = {}
        for key, serial in zip(("front", "wrist"), serials):
            camera = Camera(rs, serial, width, height, fps)
            try:
                camera.open()
                results[key] = calibrate(camera, key, values, args, args.output)
            finally:
                camera.close()
        (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Diagnostics saved in {args.output}")
        if not args.headless and os.environ.get("DISPLAY"):
            show_tuned_preview(args.output)
        else:
            print(
                f"Final tuned images: {args.output / 'front-after.png'} and {args.output / 'wrist-after.png'}"
            )
        settings = {
            key: value
            for result in results.values()
            for key, value in result["settings"].items()
        }
        confirm_and_save(args.env, original, settings)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(
            1,
            f"Calibration failed: {error}\nClose camera previews/recording and run from a local desktop terminal with USB access.\n",
        )
    except KeyboardInterrupt:
        parser.exit(130, "Calibration cancelled.\n")


if __name__ == "__main__":
    main()
