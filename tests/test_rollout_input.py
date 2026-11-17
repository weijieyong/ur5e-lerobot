"""Real PTY bytes through the rollout listener; never captures a real keyboard."""

import contextlib
import io
import os
import pty
import select
import termios
import threading
import time
import unittest
from unittest.mock import patch

from scripts.rollout_input import RolloutInput, RolloutTerminal, preflight


class InputTests(unittest.TestCase):
    def control(self):
        messages = []
        control = RolloutInput("previous", messages.append)
        control.ready()
        return control, RolloutTerminal(control), messages

    def test_optional_edit_cancel_empty_invalid_and_codepoint_backspace(self):
        for draft in (
            b"\r",
            b"\x1b",
            b"bad\x01\r",
            b"bad\xff\r",
            b"a" * 1025 + b"\r",
            b"\x1b[99~\r",
        ):
            control, listener, messages = self.control()
            listener.feed(b"t" + draft)
            if listener.escape == b"\x1b":
                listener.escape.clear()
                control.key("esc")
            self.assertEqual(control.current, "previous")
            self.assertEqual(control.phase, "READY")
            self.assertFalse(any(control.events.values()))
        control, listener, _ = self.control()
        for value in "t抓é".encode():
            listener.feed(bytes([value]))
        listener.feed(b"\x7f!\r")
        self.assertEqual(control.current, "抓!")
        self.assertEqual(control.phase, "READY")
        listener.feed(b"\r")
        self.assertEqual(control.phase, "READY")

    def test_editor_command_letters_are_literal_and_arrows_never_command(self):
        control, listener, _ = self.control()
        listener.feed(b"th n r q\r")
        self.assertEqual(control.current, "h n r q")
        self.assertFalse(control.home)
        self.assertFalse(any(control.events.values()))
        listener.feed(b"t\x1b[C\x1b[D\r")
        self.assertEqual(control.current, "h n r q")
        self.assertEqual(control.phase, "READY")

    def test_split_paste_drops_entire_frame_in_every_phase(self):
        for phase in (
            "READY",
            "EDITING",
            "START_PENDING",
            "HOME_PENDING",
            "BUSY",
            "RUNNING",
            "CLOSED",
        ):
            control, listener, messages = self.control()
            control.phase = phase
            frame = b"\x1b[200~h\nn\rrq\x1b[C\xff\x1b[201~"
            for value in frame:
                listener.feed(bytes([value]))
            self.assertEqual(control.phase, phase)
            self.assertFalse(any(control.events.values()))
            self.assertFalse(control.home)
            self.assertEqual(control.current, "previous")
            self.assertTrue(
                any("paste" in m.lower() or "REJECTED" in m for m in messages)
            )
            if phase == "EDITING":
                listener.feed(b"new\r")
                self.assertEqual(control.current, "previous")
            # The next genuinely separate command is retained.
            control.ready()
            listener.feed(b"\x1bOC")
            self.assertEqual(control.phase, "START_PENDING")

    def test_admission_both_orders_and_busy_callbacks_not_deferred(self):
        control, _, _ = self.control()
        control.key("t")
        control.key("n")
        self.assertEqual(control.draft, "n")
        self.assertEqual(control.phase, "EDITING")
        control.key("esc")
        control.key("n")
        control.key("t")
        control.key("h")
        self.assertEqual(control.phase, "START_PENDING")
        self.assertFalse(control.home)
        control.phase = "BUSY"
        with control.lock:
            thread = threading.Thread(
                target=lambda: [control.key(k) for k in ("t", "h", "n")]
            )
            thread.start()
            thread.join(0.5)
            self.assertFalse(thread.is_alive())
        control.ready()
        self.assertEqual(control.phase, "READY")
        self.assertFalse(control.home)

    @contextlib.contextmanager
    def terminal(self):
        master, slave = pty.openpty()
        stream = os.fdopen(os.dup(slave), "r")
        attrs = termios.tcgetattr(slave)
        control, listener, messages = self.control()
        try:
            with patch("sys.stdin", stream):
                listener.start()
                yield master, slave, control, listener, messages
                listener.stop()
            self.assertEqual(termios.tcgetattr(slave), attrs)
            output = b""
            while select.select([master], [], [], 0)[0]:
                output += os.read(master, 4096)
            self.assertIn(b"\x1b[?2004h", output)
            self.assertIn(b"\x1b[?2004l", output)
        finally:
            listener.stop()
            stream.close()
            os.close(master)
            os.close(slave)

    def wait_for(self, predicate):
        until = time.monotonic() + 2
        while not predicate() and time.monotonic() < until:
            time.sleep(0.005)
        self.assertTrue(predicate())

    def test_pty_actual_path_split_utf8_paste_and_escape(self):
        with self.terminal() as (master, _, control, _, _):
            os.write(master, b"t")
            self.wait_for(lambda: control.phase == "EDITING")
            for value in "抓é".encode():
                os.write(master, bytes([value]))
                time.sleep(0.002)
            os.write(master, b"\r")
            self.wait_for(lambda: control.current == "抓é")
            os.write(master, b"\x1b[20")
            time.sleep(0.08)  # CSI marker split longer than native escape timeout.
            os.write(master, b"0~n\rhqr\x1b[20")
            time.sleep(0.08)
            os.write(master, b"1~")
            time.sleep(0.08)
            self.assertEqual(control.phase, "READY")
            self.assertFalse(any(control.events.values()))
            os.write(master, b"tabc\x1b")
            self.wait_for(lambda: control.phase == "READY")
            self.assertEqual(control.current, "抓é")

    def test_late_marker_suffix_never_becomes_commands_after_escape_timeout(self):
        with self.terminal() as (master, _, control, _, _):
            os.write(master, b"tdraft")
            self.wait_for(lambda: control.phase == "EDITING")
            os.write(master, b"\x1b")
            self.wait_for(lambda: control.phase == "READY")
            os.write(master, b"[200~hnrq\r\x1b[201~")
            time.sleep(0.1)
            self.assertEqual(control.phase, "READY")
            self.assertFalse(control.home)
            self.assertFalse(any(control.events.values()))
            self.assertEqual(control.current, "previous")

    def test_partial_start_error_restores_termios_and_bracket_mode(self):
        master, slave = pty.openpty()
        attrs = termios.tcgetattr(slave)
        control, listener, _ = self.control()
        try:
            with (
                os.fdopen(os.dup(slave), "r") as stream,
                patch("sys.stdin", stream),
                patch(
                    "threading.Thread.start", side_effect=RuntimeError("thread failed")
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "thread failed"):
                    listener.start()
            self.assertEqual(termios.tcgetattr(slave), attrs)
            output = os.read(master, 4096)
            self.assertIn(b"\x1b[?2004h", output)
            self.assertIn(b"\x1b[?2004l", output)
        finally:
            listener.stop()
            os.close(master)
            os.close(slave)

    def test_pty_reader_error_and_interrupt_restore_terminal(self):
        for error in (OSError("reader failed"), KeyboardInterrupt()):
            with self.terminal() as (master, _, control, listener, _):
                with patch.object(listener, "feed", side_effect=error):
                    os.write(master, b"x")
                    self.wait_for(lambda: not listener._running)
                self.assertEqual(control.phase, "CLOSED")
                with self.assertRaisesRegex(RuntimeError, "terminal input failed"):
                    control.check()

    def test_eof_and_read_failure_close_and_surface(self):
        for error in (None, OSError("read failed")):
            control, listener, _ = self.control()
            listener._fd = 99
            listener._running = True
            with (
                patch(
                    "scripts.rollout_input.select.select", return_value=([99], [], [])
                ),
                patch(
                    "scripts.rollout_input.os.read", return_value=b"", side_effect=error
                ),
            ):
                listener._run()
            self.assertFalse(listener._running)
            self.assertEqual(control.phase, "CLOSED")
            self.assertTrue(control.events["stop_recording"])
            with self.assertRaisesRegex(RuntimeError, "terminal input failed"):
                control.check()

    def test_preflight_rejects_non_tty(self):
        with patch("sys.stdin", io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "POSIX TTY"):
                preflight()


if __name__ == "__main__":
    unittest.main()
