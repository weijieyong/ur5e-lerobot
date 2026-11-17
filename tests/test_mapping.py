"""Known SpaceMouse frame/sign examples and disconnected-input behavior."""

import contextlib
import io
import runpy
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig
from lerobot_robot_ur5e.spacemouse import SpaceMouse, SpaceMouseHID


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def mouse(self, **kwargs):
        settings = dict(
            axis_map="+0 -1 -2 +3 -4 -5".split(), operator_yaw_deg=0, axes="xyz_yaw"
        )
        settings.update(kwargs)
        return SpaceMouse(
            SpaceMouseConfig(calibration_dir=Path(self.directory.name), **settings)
        )

    def test_known_full_deflection_signs_and_operator_rotation(self):
        mouse = self.mouse(max_linear_speed_m_s=0.1, max_angular_speed_rad_s=0.5)
        velocity, angular = mouse.map_axes(np.array([350, 0, 0, 0, 0, 350]))
        np.testing.assert_allclose(velocity, [0.1, 0, 0])
        np.testing.assert_allclose(angular, [0, 0, -0.5])
        velocity, _ = mouse.map_axes(np.array([0, 350, 350, 0, 0, 0]))
        np.testing.assert_allclose(velocity, [0, -0.1, -0.1])
        rotated = self.mouse(operator_yaw_deg=90, max_linear_speed_m_s=0.1)
        velocity, _ = rotated.map_axes(np.array([350, 0, 0, 0, 0, 0]))
        np.testing.assert_allclose(velocity, [0, 0.1, 0], atol=1e-12)

    def test_full_rotation_mapping_and_invalid_channel_map(self):
        mouse = self.mouse(axes="full", max_angular_speed_rad_s=0.5)
        _, angular = mouse.map_axes(np.array([0, 0, 0, 350, 0, 0]))
        np.testing.assert_allclose(angular, [0.5, 0, 0])
        _, angular = mouse.map_axes(np.array([0, 0, 0, 0, 350, 0]))
        np.testing.assert_allclose(angular, [0, -0.5, 0])
        with self.assertRaises(ValueError):
            SpaceMouseConfig(axis_map=["+0"] * 6)

    def test_unopenable_device_points_to_the_udev_rule(self):
        def denied(_):
            raise OSError("open failed")

        device = SimpleNamespace(open_path=denied, close=lambda: None)
        backend = SimpleNamespace(
            enumerate=lambda *args: [
                {"path": b"fixture", "product_string": "SpaceMouse"}
            ],
            device=lambda: device,
        )
        with patch.dict("sys.modules", {"hidraw": backend}):
            with self.assertRaisesRegex(
                PermissionError, "Found SpaceMouse but cannot open.*udev rule"
            ):
                SpaceMouseHID().open()

    def test_reader_failure_is_reported_and_can_be_reopened(self):
        def failed_read(*args):
            raise OSError("fixture: USB disconnected")

        device = SimpleNamespace(
            open_path=lambda _: None, read=failed_read, close=lambda: None
        )
        backend = SimpleNamespace(
            enumerate=lambda *args: [{"path": b"fixture"}], device=lambda: device
        )
        with patch.dict("sys.modules", {"hidraw": backend}):
            reader = SpaceMouseHID()
            try:
                reader.open()
                deadline = time.monotonic() + 1
                while True:
                    try:
                        reader.get_state()
                    except ConnectionError:
                        break
                    if time.monotonic() > deadline:
                        self.fail("Disconnected input was not reported")
                    time.sleep(0.001)
                reader.close()
                device.read = lambda *args: threading.Event().wait(0.01) or []
                reader.open()
                axes, buttons = reader.get_state()
                np.testing.assert_array_equal(axes, [0] * 6)
                self.assertEqual(buttons, 0)
            finally:
                reader.close()

    def test_escape_stops_teleop_before_robot_lookup(self):
        listeners = []

        class Listener:
            def __init__(self, on_press):
                self.on_press = on_press
                listeners.append(self)

            def start(self):
                pass

            def stop(self):
                pass

        esc = object()
        keyboard = SimpleNamespace(Key=SimpleNamespace(esc=esc), Listener=Listener)
        device = SimpleNamespace(
            open_path=lambda _: None,
            read=lambda *args: threading.Event().wait(0.01) or [],
            close=lambda: None,
        )
        backend = SimpleNamespace(
            enumerate=lambda *args: [{"path": b"fixture"}], device=lambda: device
        )
        with patch.dict(
            "sys.modules",
            {"hidraw": backend, "pynput": SimpleNamespace(keyboard=keyboard)},
        ):
            mouse = self.mouse(stop_on_escape=True)
            try:
                mouse.connect()
                listeners[0].on_press(esc)
                with self.assertRaises(KeyboardInterrupt):
                    mouse.get_action()
            finally:
                mouse.disconnect()

    def test_mapping_check_never_opens_a_robot_connection(self):
        script = Path(__file__).resolve().parents[1] / "scripts/hw_check.py"
        device = SimpleNamespace(
            open_path=lambda _: None,
            read=lambda *args: threading.Event().wait(0.01) or [],
            close=lambda: None,
        )
        backend = SimpleNamespace(
            enumerate=lambda *args: [{"path": b"fixture"}], device=lambda: device
        )
        blocked = AssertionError("Mapping check must not connect to robot/gripper")
        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"hidraw": backend}),
            patch("rtde_receive.RTDEReceiveInterface", side_effect=blocked),
            patch("rtde_control.RTDEControlInterface", side_effect=blocked),
            patch("socket.create_connection", side_effect=blocked),
            patch.object(
                sys,
                "argv",
                [str(script), "--only", "spacemouse", "--spacemouse-seconds", "0.01"],
            ),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(SystemExit) as exit_status:
                runpy.run_module("scripts.hw_check", run_name="__main__")
        self.assertEqual(exit_status.exception.code, 0)
        self.assertIn("No robot connection or motion", output.getvalue())
