"""Terminal-only rollout admission and a deliberately small typing-only editor."""

import atexit
import codecs
import contextlib
import os
import select
import sys
import threading

from lerobot.utils.keyboard_input import TerminalKeyListener, apply_recording_control


def preflight():
    """Fail before context construction; never fall back to global capture."""
    try:
        import fcntl
        import termios

        if not sys.stdin.isatty() or not os.isatty(sys.stdin.fileno()):
            raise ValueError("stdin is not a terminal")
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDWR:
            raise ValueError(
                "stdin TTY must be readable and writable for paste framing"
            )
    except (ImportError, OSError, ValueError) as error:
        raise RuntimeError(
            f"Rollout requires a focused POSIX TTY with usable termios: {error}"
        ) from error


class RolloutInput:
    def __init__(self, task, show, preview=None):
        self.current = task
        self.show = show
        self.preview = preview or show
        self.lock = threading.Lock()
        self.phase = "BUSY"
        self.home = False
        self.error = None
        self.events = dict(
            exit_early=False, rerecord_episode=False, stop_recording=False
        )
        self.draft = ""
        self.poisoned = False

    def ready(self):
        with self.lock:
            self.events["exit_early"] = self.events["rerecord_episode"] = False
            self.home = False
            self.phase = "READY"
            self.show(
                f"READY — POLICY PAUSED\nCurrent task: {self.current}\n"
                "n/Right: START | t: edit (optional) | h: home | q/Esc: exit"
            )

    def fail(self, error):
        self.error = error
        apply_recording_control("esc", self.events)
        self.phase = "CLOSED"

    def check(self):
        if self.error is not None:
            raise RuntimeError(
                f"Rollout terminal input failed: {self.error}"
            ) from self.error

    def key(self, key):
        if not self.lock.acquire(blocking=False):
            if key in ("q", "esc") and self.phase != "EDITING":
                apply_recording_control("esc", self.events)
            elif key in ("r", "left", "n", "right") and self.phase in (
                "START_PENDING",
                "RUNNING",
            ):
                apply_recording_control(
                    "left" if key in ("r", "left") else "right", self.events
                )
            return
        try:
            if self.phase == "EDITING":
                if key == "esc":
                    self.phase = "READY"
                elif key == "enter":
                    if self.draft.strip() and not self.poisoned:
                        self.current = self.draft.strip()
                    self.phase = "READY"
                elif key == "backspace":
                    self.draft = self.draft[:-1]
                elif key in ("invalid", "paste", "up", "down", "left", "right", "tab"):
                    self.poisoned = True
                elif key == "space" or (len(key) == 1 and key.isprintable()):
                    if len(self.draft) < 1024:
                        self.draft += " " if key == "space" else key
                    else:
                        self.poisoned = True
                else:
                    self.poisoned = True
                if self.phase == "EDITING":
                    self.preview(
                        f"\r\x1b[2KDraft: {self.draft}"
                        + (
                            " [REJECTED — Esc/Enter keeps previous task]"
                            if self.poisoned
                            else ""
                        )
                    )
                else:
                    self.preview("\n")
                    self.show(f"READY — Current task: {self.current}")
                return
            if key == "paste":
                self.show(
                    "Bracketed paste rejected; typing only. Current task unchanged."
                )
            if key in ("q", "esc"):
                apply_recording_control("esc", self.events)
            elif self.phase == "READY":
                if key == "t":
                    self.phase, self.draft, self.poisoned = "EDITING", "", False
                    self.preview("\r\x1b[2KDraft: ")
                elif key in ("n", "right"):
                    self.phase = "START_PENDING"
                elif key == "h":
                    self.home, self.phase = True, "HOME_PENDING"
            elif self.phase in ("RUNNING", "START_PENDING"):
                if key in ("n", "right", "r", "left"):
                    apply_recording_control(
                        "left" if key in ("r", "left") else "right", self.events
                    )
        finally:
            self.lock.release()


class RolloutTerminal(TerminalKeyListener):
    """Reuse native terminal lifecycle, replacing its lossy byte/escape parser.

    CSI sequences stay buffered across reads. Only a lone Escape uses a timeout;
    CSI/paste parsing after the introducer has no inter-byte deadline. A delayed
    first Escape is ambiguous with a bare Escape (see operating guide).
    """

    def __init__(self, control):
        super().__init__(control.key)
        self.control = control
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.escape = bytearray()
        self.expired_escape = False
        self.paste = False
        self.paste_tail = bytearray()
        self.bracketed = False

    def start(self):
        preflight()
        try:
            self._fd = sys.stdin.fileno()
            self.bracketed = True
            atexit.register(self.stop)
            os.write(self._fd, b"\x1b[?2004h")
            super().start()
        except BaseException:
            with contextlib.suppress(Exception):
                self.stop()
            raise

    def stop(self):
        if self._thread is threading.current_thread() or (
            self._thread is not None and self._thread.ident is None
        ):
            self._thread = None
        try:
            super().stop()
        finally:
            if self.bracketed:
                self.bracketed = False
                os.write(self._fd, b"\x1b[?2004l")
            atexit.unregister(self.stop)

    def feed(self, data):
        for value in data:
            if self.expired_escape:
                self.expired_escape = False
                if value in (ord("["), ord("O")):
                    self.escape.append(27)
            if self.paste:
                self.paste_tail.append(value)
                self.paste_tail = self.paste_tail[-6:]
                if self.paste_tail == b"\x1b[201~":
                    self.paste = False
                    self.paste_tail.clear()
                continue
            if self.escape:
                self.escape.append(value)
                sequence = bytes(self.escape)
                if sequence == b"\x1b[200~":
                    self.escape.clear()
                    self.paste = True
                    self._on_key("paste")
                elif len(sequence) == 2 and value not in (ord("["), ord("O")):
                    self.escape.clear()
                    self._on_key("invalid")
                elif len(sequence) >= 3 and 0x40 <= value <= 0x7E:
                    self.escape.clear()
                    arrows = {
                        b"\x1b[C": "right",
                        b"\x1b[D": "left",
                        b"\x1b[A": "up",
                        b"\x1b[B": "down",
                        b"\x1bOC": "right",
                        b"\x1bOD": "left",
                        b"\x1bOA": "up",
                        b"\x1bOB": "down",
                    }
                    self._on_key(arrows.get(sequence, "invalid"))
                elif len(sequence) > 64:
                    raise ValueError("Unsupported overlong terminal escape sequence")
                continue
            if value == 27:
                if self.decoder.getstate()[0]:
                    self.decoder.reset()
                    self._on_key("invalid")
                self.escape.append(value)
                continue
            try:
                char = self.decoder.decode(bytes([value]))
            except UnicodeDecodeError:
                self.decoder.reset()
                self._on_key("invalid")
                continue
            if not char:
                continue
            key = {
                "\r": "enter",
                "\n": "enter",
                "\x7f": "backspace",
                "\x08": "backspace",
                " ": "space",
            }.get(char, char if char.isprintable() else "invalid")
            self._on_key(key)

    def _run(self):
        try:
            while self._running:
                ready, _, _ = select.select([self._fd], [], [], 0.05)
                if not ready:
                    if self.escape == b"\x1b":
                        self.escape.clear()
                        self.expired_escape = True
                        self._on_key("esc")
                    continue
                data = os.read(self._fd, 1)
                if not data:
                    raise EOFError("terminal closed")
                self.feed(data)
        except BaseException as error:
            self.control.fail(error)
        finally:
            try:
                self.stop()
            except Exception as error:
                self.control.fail(error)
