"""Read-only reference capture: units and preservation of the machine config."""

import shlex
import contextlib
import io
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ReferenceTests(unittest.TestCase):
    def test_home_and_floor_capture_preserve_other_settings(self):
        script = Path(__file__).resolve().parents[1] / "scripts/set_robot_reference.py"
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config/local/machine.env"
            config.parent.mkdir(parents=True)
            config.write_text(
                '# private settings\nKEEP="untouched"\nUR5E_Z_MIN_M=0.1\n'
            )
            receiver = SimpleNamespace(
                getActualQ=lambda: [0, math.pi / 2, -math.pi / 2, math.pi, 0, -math.pi],
                getActualTCPPose=lambda: [0, 0, 0.234, 0.2, -0.3, 3.0],
                disconnect=lambda: None,
            )
            module = SimpleNamespace(RTDEReceiveInterface=lambda _: receiver)
            with (
                patch.dict(
                    os.environ,
                    {
                        "PIXI_PROJECT_ROOT": directory,
                        "UR5E_FLOOR_MARGIN_M": "0.01",
                        "UR5E_IP": "192.0.2.1",
                    },
                ),
                patch.dict(sys.modules, {"rtde_receive": module}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                for reference in ("home", "floor", "neutral"):
                    with patch.object(sys, "argv", [str(script), reference]):
                        runpy.run_module(
                            "scripts.set_robot_reference", run_name="__main__"
                        )
            text = config.read_text()
            values = dict(token.split("=", 1) for token in shlex.split(text, comments=True))
            self.assertIn('KEEP="untouched"', text)
            self.assertEqual(
                [float(value) for value in values["UR5E_HOME_JOINTS_DEG"].split()],
                [0, 90, -90, 180, 0, -180],
            )
            self.assertAlmostEqual(float(values["UR5E_Z_MIN_M"]), 0.244, places=6)
            self.assertEqual(
                [float(value) for value in values["SPACEMOUSE_NEUTRAL_ROTVEC"].split()],
                [0.2, -0.3, 3.0],
            )
            self.assertEqual(text.count("UR5E_Z_MIN_M="), 1)

    def test_invalid_reference_measurement_does_not_change_config(self):
        script = Path(__file__).resolve().parents[1] / "scripts/set_robot_reference.py"
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config/local/machine.env"
            config.parent.mkdir(parents=True)
            original = "UR5E_Z_MIN_M=0.1\n"
            config.write_text(original)
            receiver = SimpleNamespace(
                getActualTCPPose=lambda: [0, 0, float("nan"), 0, 0, 0],
                disconnect=lambda: None,
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "PIXI_PROJECT_ROOT": directory,
                        "UR5E_FLOOR_MARGIN_M": "0.01",
                        "UR5E_IP": "192.0.2.1",
                    },
                ),
                patch.dict(
                    sys.modules,
                    {
                        "rtde_receive": SimpleNamespace(
                            RTDEReceiveInterface=lambda _: receiver
                        )
                    },
                ),
                patch.object(sys, "argv", [str(script), "floor"]),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    runpy.run_module("scripts.set_robot_reference", run_name="__main__")
            self.assertEqual(config.read_text(), original)
