"""Hardware-boundary regressions, using SDK/socket stand-ins; no hardware access."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop

from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig, UR5eConfig
from lerobot_robot_ur5e.ur5e import GRIPPER, JOINTS, OnRobotRG2, RobotiqSocket, UR5e
from test_recording import MemoryDataset, ScriptedMouse


class FakeOnRobotURCap:
    """Stand-in for the OnRobot URCap XML-RPC server; every proxy shares this state."""

    def __init__(self, width=101.6, offset=4.2, faults=()):
        self.width, self.offset, self.faults = width, offset, set(faults)
        self.grips, self.busy, self.stops = [], False, 0

    def proxy(self, url, transport=None):
        flags = {
            f"rg_get_{name}": (lambda _, name=name: name in self.faults)
            for name in OnRobotRG2.SAFETY_FLAGS
        }
        return SimpleNamespace(
            rg_get_width=lambda _: self.width,
            rg_get_fingertip_offset=lambda _: self.offset,
            rg_grip=lambda *args: self.grips.append(args),
            rg_get_busy=lambda _: self.busy,
            rg_stop=lambda _: setattr(self, "stops", self.stops + 1),
            **flags,
        )


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.robot = UR5e(
            UR5eConfig(
                ip="192.0.2.1",
                cameras={},
                calibration_dir=root / "robot",
                timing_log_dir="",
                gripper_type="robotiq",
                x_min_m=None,
                x_max_m=None,
                y_min_m=None,
                y_max_m=None,
                z_min_m=None,
            )
        )
        self.sent = []
        self.robot.rtde_r = SimpleNamespace(
            getActualQ=lambda: [0.0] * 6,
            isProtectiveStopped=lambda: False,
            isEmergencyStopped=lambda: False,
            disconnect=lambda: None,
        )
        self.robot.rtde_c = SimpleNamespace(
            servoJ=lambda *args: self.sent.append(args[0]),
            isConnected=lambda: False,
            servoStop=lambda: None,
            stopScript=lambda: None,
            disconnect=lambda: None,
        )
        self.robot.gripper = SimpleNamespace(
            position=lambda: 0.0, grip=lambda _: None, close=lambda: None
        )
        self.mouse = ScriptedMouse(SpaceMouseConfig(calibration_dir=root / "mouse"))
        self.addCleanup(self.robot.disconnect)

    def test_rejected_controller_command_is_not_saved_as_a_demonstration_frame(self):
        self.robot.rtde_c.servoJ = lambda *args: False
        dataset = MemoryDataset()
        teleop, action, observation = make_default_processors()
        with self.assertRaisesRegex(RuntimeError, "servoJ"):
            record_loop(
                robot=self.robot,
                events={"exit_early": False},
                fps=15,
                teleop_action_processor=teleop,
                robot_action_processor=action,
                robot_observation_processor=observation,
                dataset=dataset,
                teleop=self.mouse,
                control_time_s=0.001,
                single_task="pick and place",
            )
        self.assertEqual(dataset.frames, [])

    def test_nonfinite_action_is_rejected_before_any_controller_command(self):
        for key in (JOINTS[0], GRIPPER):
            with self.subTest(key=key):
                action = {**dict.fromkeys(JOINTS, 0.0), GRIPPER: 0.0, key: float("nan")}
                with self.assertRaises(ValueError):
                    self.robot.send_action(action)
        self.assertEqual(self.sent, [])

    def test_disconnect_attempts_all_cleanup_even_if_servo_stop_fails(self):
        calls = []

        def failed_stop():
            calls.append("servo_stop")
            raise RuntimeError("fixture: stop failed")

        self.robot.rtde_c.servoStop = failed_stop
        self.robot.rtde_c.stopScript = lambda: calls.append("stop_script")
        self.robot.rtde_c.disconnect = lambda: calls.append("control_disconnect")
        self.robot.rtde_r.disconnect = lambda: calls.append("receive_disconnect")
        self.robot.gripper.close = lambda: calls.append("gripper_close")
        with self.assertLogs("lerobot_robot_ur5e.ur5e", level="ERROR"):
            self.robot.disconnect()
        self.assertEqual(
            set(calls),
            {
                "servo_stop",
                "stop_script",
                "control_disconnect",
                "receive_disconnect",
                "gripper_close",
            },
        )
        self.assertFalse(self.robot.is_connected)

    def test_failed_camera_connection_stops_the_uploaded_control_script(self):
        control, receiver = self.robot.rtde_c, self.robot.rtde_r
        calls = []
        control.stopScript = lambda: calls.append("stop_script")
        control.disconnect = lambda: calls.append("control_disconnect")
        receiver.disconnect = lambda: calls.append("receive_disconnect")

        def failed_camera():
            raise RuntimeError("fixture: camera missing")

        self.robot.cameras = {
            "front": SimpleNamespace(connect=failed_camera, is_connected=False)
        }
        socket = SimpleNamespace(
            sendall=lambda _: None,
            recv=lambda _: b"ACT 1",
            close=lambda: calls.append("socket_close"),
        )
        replies = iter((b"ACT 1", b"STA 3"))
        socket.recv = lambda _: next(replies)
        with (
            patch("rtde_control.RTDEControlInterface", return_value=control),
            patch("rtde_receive.RTDEReceiveInterface", return_value=receiver),
            patch("socket.create_connection", return_value=socket),
        ):
            with self.assertRaisesRegex(RuntimeError, "camera missing"):
                self.robot.connect()
        self.assertEqual(
            set(calls),
            {"stop_script", "control_disconnect", "receive_disconnect", "socket_close"},
        )

    def test_robotiq_set_command_and_failed_ack(self):
        wire = []
        replies = iter((b"ack", b"bad ack"))
        socket = SimpleNamespace(
            sendall=wire.append, recv=lambda _: next(replies), close=lambda: None
        )
        with patch("socket.create_connection", return_value=socket):
            gripper = RobotiqSocket("192.0.2.1", 63352)
            gripper.set(POS=255, SPE=128, FOR=64, GTO=1)
            self.assertEqual(wire[0], b"SET POS 255 SPE 128 FOR 64 GTO 1\n")
            with self.assertRaisesRegex(RuntimeError, "ack"):
                gripper.set(POS=0)
            gripper.close()

    def test_robotiq_get_rejects_a_reply_for_the_wrong_variable(self):
        socket = SimpleNamespace(
            sendall=lambda _: None, recv=lambda _: b"STA 3", close=lambda: None
        )
        with patch("socket.create_connection", return_value=socket):
            gripper = RobotiqSocket("192.0.2.1", 63352)
            with self.assertRaisesRegex(RuntimeError, "Unexpected gripper reply"):
                gripper.get("POS")
            gripper.close()

    def test_rg2_position_spans_open_to_closed(self):
        urcap = FakeOnRobotURCap()
        with patch("xmlrpc.client.ServerProxy", side_effect=urcap.proxy):
            gripper = OnRobotRG2("192.0.2.1")
            self.addCleanup(gripper.close)
            gripper.activate()
            urcap.width = 110.0 - 2 * urcap.offset  # fingers fully open
            self.assertEqual(gripper.position(), 0.0)
            urcap.width = 0.0  # fingers fully closed
            self.assertEqual(gripper.position(), 1.0)

    def test_rg2_close_never_targets_zero_width(self):
        urcap = FakeOnRobotURCap()
        with patch("xmlrpc.client.ServerProxy", side_effect=urcap.proxy):
            gripper = OnRobotRG2("192.0.2.1", close_width_mm=8.0)
            gripper.activate()
            gripper.grip(True)
        self.assertEqual(
            urcap.grips[0][1], 8.0
        )  # 0 mm on nothing latches the RG2 safety switch
        with self.assertRaisesRegex(ValueError, "above 0 mm"):
            UR5eConfig(
                cameras={}, gripper_type="onrobot_rg2", gripper_close_width_mm=0.0
            )

    def test_rg2_stops_a_moving_gripper_before_reversing(self):
        urcap = FakeOnRobotURCap()
        with patch("xmlrpc.client.ServerProxy", side_effect=urcap.proxy):
            gripper = OnRobotRG2("192.0.2.1")
            gripper.activate()
            gripper.grip(False)
            self.assertEqual(urcap.stops, 0)  # idle: no stop needed
            urcap.busy = True  # still opening when close is pressed
            gripper.grip(True)
        self.assertEqual(urcap.stops, 1)
        self.assertEqual(urcap.grips[-1][1], gripper.close_width_mm)

    def test_rg2_refuses_unpowered_latched_or_unclosable_gripper(self):
        for urcap, close_width_mm, message in (
            (FakeOnRobotURCap(width=-999.0, offset=-1.0), 8.0, "not responding"),
            (FakeOnRobotURCap(faults={"s2_triggered"}), 8.0, "safety latched"),
            (
                FakeOnRobotURCap(),
                101.6,
                "close width",
            ),  # at the open width "closed" would not close
        ):
            with (
                self.subTest(message=message),
                patch("xmlrpc.client.ServerProxy", side_effect=urcap.proxy),
            ):
                gripper = OnRobotRG2("192.0.2.1", close_width_mm=close_width_mm)
                self.addCleanup(gripper.close)
                with self.assertRaisesRegex(RuntimeError, message):
                    gripper.activate()

    def test_gripper_force_defaults_by_type_and_honours_overrides(self):
        def force(gripper_type, **env):
            clean = {
                k: v
                for k, v in os.environ.items()
                if k not in ("GRIPPER_FORCE", "ROLLOUT_GRIPPER_FORCE")
            }
            with patch.dict("os.environ", {**clean, **env}, clear=True):
                return UR5eConfig(cameras={}, gripper_type=gripper_type).gripper_force

        self.assertEqual(force("onrobot_rg2"), 128)  # about 20 N
        self.assertEqual(force("robotiq"), 64)
        self.assertEqual(
            force("onrobot_rg2", GRIPPER_FORCE=""), 128
        )  # empty means unset, not a crash
        self.assertEqual(force("onrobot_rg2", GRIPPER_FORCE="255"), 255)
        self.assertEqual(force("onrobot_rg2", GRIPPER_FORCE="0"), 0)
        self.assertEqual(
            force("onrobot_rg2", ROLLOUT_GRIPPER_FORCE="200"), 200
        )  # former name kept
        self.assertEqual(
            force("onrobot_rg2", GRIPPER_FORCE="90", ROLLOUT_GRIPPER_FORCE="200"), 90
        )
        with self.assertRaisesRegex(ValueError, r"\[0, 255\]"):
            force("onrobot_rg2", GRIPPER_FORCE="300")

    def test_empty_gripper_type_falls_back_to_robotiq(self):
        with patch.dict("os.environ", {"GRIPPER_TYPE": ""}):
            self.assertEqual(UR5eConfig(cameras={}).gripper_type, "robotiq")

    def test_home_opens_gripper_and_waits_before_moving(self):
        calls, positions = [], iter([1.0, 0.6, 0.01])
        self.robot.config.home_joints_deg = [0] * 6
        self.robot._gripper_closed = True
        self.robot.rtde_c.servoStop = lambda: calls.append("stop")
        self.robot.rtde_c.moveJ = lambda *args: calls.append("move")
        self.robot.gripper = SimpleNamespace(
            grip=lambda closed: calls.append(("grip", closed)),
            position=lambda: next(positions),
            close=lambda: None,
        )
        with patch("lerobot_robot_ur5e.ur5e.time.sleep"):
            self.robot.go_home()
        self.assertEqual(calls, ["stop", ("grip", False), "move"])
        self.assertIs(self.robot._gripper_closed, False)

    def test_home_does_not_move_if_gripper_never_opens(self):
        clock = iter(range(100))
        self.robot.config.home_joints_deg = [0] * 6
        self.robot.rtde_c.moveJ = lambda *args: self.fail(
            "Arm moved with the gripper still closed"
        )
        self.robot.gripper = SimpleNamespace(
            grip=lambda _: None, position=lambda: 1.0, close=lambda: None
        )
        with (
            patch("lerobot_robot_ur5e.ur5e.time.sleep"),
            patch(
                "lerobot_robot_ur5e.ur5e.time.monotonic",
                side_effect=lambda: next(clock),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not open"):
                self.robot.go_home()

    def test_rejected_home_move_is_reported(self):
        self.robot.config.home_joints_deg = [0] * 6
        self.robot.rtde_c.moveJ = lambda *args: False
        with self.assertRaisesRegex(RuntimeError, "home move"):
            self.robot.go_home()
