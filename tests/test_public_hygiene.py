"""Shared source carries no machine-private details: home paths, lab IPs or device serials."""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".pixi",
    ".cache",
    "__pycache__",
    "data",
    "outputs",
    "hf",
    "local",
    ".claude",
}
SKIP_FILES = {"pixi.lock"}
PRIVATE = {
    "absolute user path": re.compile(
        r"(?<![\w/.-])/(home|media|Users)/[A-Za-z0-9_.-]+/"
    ),
    "private network IP": re.compile(
        r"\b(10\.\d{1,3}|192\.168|172\.(1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"
    ),
    "device serial": re.compile(r"(?<![0-9A-Za-z])\d{12}(?![0-9A-Za-z])"),
}


def shared_files():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if (
            path.is_file()
            and not SKIP_DIRS & set(rel.parts)
            and rel.name not in SKIP_FILES
        ):
            try:
                yield rel, path.read_text()
            except UnicodeDecodeError:
                continue


class PublicHygieneTests(unittest.TestCase):
    def test_no_machine_private_details(self):
        hits = [
            f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {label} {m.group(0)!r}"
            for rel, text in shared_files()
            for label, rx in PRIVATE.items()
            for m in rx.finditer(text)
        ]
        self.assertEqual(
            hits, [], "Machine-private details belong in config/local/machine.env"
        )


if __name__ == "__main__":
    unittest.main()
