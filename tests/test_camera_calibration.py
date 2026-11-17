"""Offline checks for white-paper measurements and config preservation."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import calibrate_cameras as calibration


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/calibrate_cameras.py"


class CalibrationTests(unittest.TestCase):
    def test_white_patch_measurement_and_clipping(self):
        image = np.full((40, 40, 3), (205, 220, 235), dtype=np.uint8)
        stats = calibration.measure(image, (5, 5, 20, 20))
        self.assertEqual(stats["rgb"], [235.0, 220.0, 205.0])
        self.assertEqual(stats["brightness"], 220)
        self.assertEqual(stats["clipped_fraction"], 0)
        image[:] = 255
        self.assertEqual(
            calibration.measure(image, (0, 0, 40, 40))["clipped_fraction"], 1
        )

    def test_roi_must_be_inside_image_and_large_enough(self):
        for roi in ((0, 0, 0, 20), (-1, 0, 20, 20), (30, 30, 20, 20)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                calibration.measure(np.zeros((40, 40, 3), dtype=np.uint8), roi)

    def test_config_parser_preserves_serial_and_handles_comments(self):
        values = calibration.read_config(
            "export FRONT_CAMERA_SERIAL=00123 # camera\nCAMERA_WIDTH=640\n"
        )
        self.assertEqual(values["FRONT_CAMERA_SERIAL"], "00123")
        self.assertEqual(values["CAMERA_WIDTH"], "640")

    def test_calibration_does_not_write_machine_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.env"
            original = (
                'KEEP="as is"\nFRONT_CAMERA_SERIAL=00123\nWRIST_CAMERA_SERIAL=00456\n'
            )
            path.write_text(original)
            before = path.stat().st_mtime_ns
            devices = [
                SimpleNamespace(get_info=lambda _, serial=serial: serial)
                for serial in ("00123", "00456")
            ]
            rs = SimpleNamespace(
                context=lambda: SimpleNamespace(query_devices=lambda: devices),
                camera_info=SimpleNamespace(serial_number=0),
            )
            argv = [
                str(SCRIPT),
                "--env",
                str(path),
                "--headless",
                "--front-roi",
                "0",
                "0",
                "20",
                "20",
                "--wrist-roi",
                "0",
                "0",
                "20",
                "20",
                "--output",
                str(Path(directory) / "results"),
            ]
            with (
                patch.dict(sys.modules, {"pyrealsense2": rs}),
                patch.object(sys, "argv", argv),
                patch.object(calibration, "Camera"),
                patch.object(
                    calibration,
                    "calibrate",
                    return_value={"settings": {}, "issues": []},
                ),
                patch("builtins.input", return_value="n"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                calibration.main()
            self.assertEqual(path.read_text(), original)
            self.assertEqual(path.stat().st_mtime_ns, before)
            self.assertTrue((Path(directory) / "results/report.json").is_file())

    def test_confirmation_defaults_to_no_and_accepts_only_y_or_yes(self):
        for answer in ("n", "", "no"):
            with (
                self.subTest(answer=answer),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "machine.env"
                original = "FRONT_CAMERA_GAIN=20\n"
                path.write_text(original)
                with (
                    patch("builtins.input", return_value=answer),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    calibration.confirm_and_save(
                        path, original, {"FRONT_CAMERA_GAIN": 16}
                    )
                self.assertEqual(path.read_text(), original)
                self.assertEqual(list(path.parent.iterdir()), [path])
        for answer in ("y", " YES "):
            with (
                self.subTest(answer=answer),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "machine.env"
                original = '# keep this\nexport FRONT_CAMERA_GAIN=20 # camera gain\nOTHER="as is"\n'
                path.write_text(original)
                path.chmod(0o600)
                with (
                    patch("builtins.input", return_value=answer),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    calibration.confirm_and_save(
                        path,
                        original,
                        {"FRONT_CAMERA_GAIN": 16, "WRIST_CAMERA_GAIN": 17},
                    )
                self.assertEqual(
                    path.read_text(),
                    '# keep this\nexport FRONT_CAMERA_GAIN=16 # camera gain\nOTHER="as is"\nWRIST_CAMERA_GAIN=17\n',
                )
                backups = list(path.parent.glob("machine.env.bak.*"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), original)
                self.assertEqual(backups[0].stat().st_mode & 0o777, 0o600)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_quality_issues_do_not_block_saving_after_cameras_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.env"
            original = (
                'KEEP="as is"\nFRONT_CAMERA_SERIAL=00123\nWRIST_CAMERA_SERIAL=00456\n'
            )
            path.write_text(original)
            devices = [
                SimpleNamespace(get_info=lambda _, serial=serial: serial)
                for serial in ("00123", "00456")
            ]
            rs = SimpleNamespace(
                context=lambda: SimpleNamespace(query_devices=lambda: devices),
                camera_info=SimpleNamespace(serial_number=0),
            )
            settings = {
                f"{key}_CAMERA_{name}": value
                for key in ("FRONT", "WRIST")
                for name, value in (
                    ("EXPOSURE", 100),
                    ("GAIN", 16),
                    ("WHITE_BALANCE", 4000),
                )
            }
            results = [
                {
                    "settings": {
                        key: value
                        for key, value in settings.items()
                        if key.startswith(prefix)
                    },
                    "issues": [
                        "brightness target not reached",
                        "paper patch has clipped highlights",
                        "remaining colour cast exceeds 10%",
                    ],
                }
                for prefix in ("FRONT", "WRIST")
            ]
            argv = [
                str(SCRIPT),
                "--env",
                str(path),
                "--front-roi",
                "0",
                "0",
                "20",
                "20",
                "--wrist-roi",
                "0",
                "0",
                "20",
                "20",
                "--output",
                str(Path(directory) / "results"),
            ]
            with (
                patch.dict(sys.modules, {"pyrealsense2": rs}),
                patch.object(sys, "argv", argv),
                patch.object(calibration, "Camera") as camera,
                patch.object(calibration, "calibrate", side_effect=results),
                patch.dict(calibration.os.environ, {"DISPLAY": ":test"}),
                patch.object(calibration, "show_tuned_preview") as preview,
                contextlib.redirect_stdout(io.StringIO()),
            ):

                def confirm(_):
                    self.assertEqual(camera.return_value.close.call_count, 2)
                    preview.assert_called_once_with(Path(directory) / "results")
                    return "y"

                with patch("builtins.input", side_effect=confirm) as prompt:
                    calibration.main()
                prompt.assert_called_once()
            updated = calibration.read_config(path.read_text())
            for key, value in settings.items():
                self.assertEqual(updated[key], str(value))
            self.assertIn(original, path.read_text())
            self.assertEqual(
                next(path.parent.glob("machine.env.bak.*")).read_text(), original
            )

    def test_preview_shows_both_final_images_and_waits_for_review(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            front = np.full((20, 30, 3), 100, dtype=np.uint8)
            wrist = np.full((20, 30, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(output / "front-after.png"), front)
            cv2.imwrite(str(output / "wrist-after.png"), wrist)
            with (
                patch.object(cv2, "namedWindow"),
                patch.object(cv2, "imshow") as show,
                patch.object(cv2, "resizeWindow"),
                patch.object(cv2, "waitKey", return_value=13) as wait,
                patch.object(cv2, "destroyWindow") as close,
            ):
                calibration.show_tuned_preview(output)
            np.testing.assert_array_equal(
                show.call_args.args[1], np.hstack((front, wrist))
            )
            wait.assert_called_once_with(0)
            close.assert_called_once_with(show.call_args.args[0])

    def test_eof_and_interrupt_do_not_save(self):
        for error in (EOFError, KeyboardInterrupt):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "machine.env"
                original = "FRONT_CAMERA_GAIN=20\n"
                path.write_text(original)
                with (
                    patch("builtins.input", side_effect=error),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    calibration.confirm_and_save(
                        path, original, {"FRONT_CAMERA_GAIN": 16}
                    )
                self.assertEqual(path.read_text(), original)

    def test_changed_config_is_not_overwritten_after_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.env"
            original = "FRONT_CAMERA_GAIN=20\n"
            path.write_text("FRONT_CAMERA_GAIN=30\n")
            with (
                patch("builtins.input", return_value="y"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    calibration.confirm_and_save(
                        path, original, {"FRONT_CAMERA_GAIN": 16}
                    )
            self.assertEqual(path.read_text(), "FRONT_CAMERA_GAIN=30\n")

    def test_invalid_answer_reprompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.env"
            original = "FRONT_CAMERA_GAIN=20\n"
            path.write_text(original)
            with (
                patch("builtins.input", side_effect=["maybe", "n"]) as prompt,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                calibration.confirm_and_save(path, original, {"FRONT_CAMERA_GAIN": 16})
            self.assertEqual(prompt.call_count, 2)
            self.assertEqual(path.read_text(), original)

    def test_brightness_prefers_low_gain_and_uses_gain_only_at_ceiling(self):
        camera = SyntheticCamera()
        calibration.tune_brightness(camera, (0, 0, 20, 20), 220, 600)
        self.assertEqual(camera.get("gain"), 16)
        self.assertAlmostEqual(
            calibration.measure(camera.sample(), (0, 0, 20, 20))["brightness"],
            220,
            delta=3,
        )
        calibration.tune_brightness(camera, (0, 0, 20, 20), 220, 100)
        self.assertEqual(camera.get("exposure"), 100)
        self.assertGreater(camera.get("gain"), 16)
        self.assertAlmostEqual(
            calibration.measure(camera.sample(), (0, 0, 20, 20))["brightness"],
            220,
            delta=3,
        )

    def test_white_balance_uses_paper_to_remove_red_blue_cast(self):
        camera = SyntheticCamera()
        calibration.tune_white_balance(camera, (0, 0, 20, 20))
        self.assertAlmostEqual(camera.get("white_balance"), 4400, delta=100)
        self.assertLess(
            calibration.measure(camera.sample(), (0, 0, 20, 20))["colour_error"], 0.03
        )

    def test_brightness_does_not_force_ceiling_when_colour_channel_clips(self):
        camera = SyntheticCamera()
        camera.values["white_balance"] = 6500
        calibration.tune_brightness(camera, (0, 0, 20, 20), 220, 1000)
        stats = calibration.measure(camera.sample(), (0, 0, 20, 20))
        self.assertEqual(stats["clipped_fraction"], 0)
        self.assertEqual(camera.get("gain"), 16)


class SyntheticCamera:
    name = "synthetic camera"
    ranges = {
        "exposure": (1, 1000, 1),
        "gain": (16, 128, 1),
        "white_balance": (2800, 6500, 100),
    }

    def __init__(self):
        self.values = {"exposure": 300, "gain": 16, "white_balance": 3600}

    def get(self, name):
        return self.values[name]

    def set(self, name, value):
        low, high, step = self.ranges[name]
        self.values[name] = int(
            max(low, min(high, low + round((value - low) / step) * step))
        )
        return self.values[name]

    def sample(self):
        brightness = self.get("exposure") * self.get("gain") / 32
        cast = (self.get("white_balance") - 4400) / 10000
        bgr = np.clip(
            [brightness * (1 - cast), brightness, brightness * (1 + cast)], 0, 255
        ).astype(np.uint8)
        return np.tile(bgr, (20, 20, 1))


if __name__ == "__main__":
    unittest.main()
