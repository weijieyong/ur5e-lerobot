"""Comment ratchet: what counts as a comment, and that the baseline only moves down."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_comments as cc

PYTHON = '''#!/usr/bin/env python
"""Docstring with a # hash."""
import os  # noqa: F401
x = "not # a comment"
# own line
y = 1  # inline
z = (
    2
)  # fmt: skip
'''

SHELL = """#!/usr/bin/env sh
# own line
echo "quoted # hash" 'single # hash'
n=${#items}
url=http://host/#anchor
KEY=value  # inline
"""


class ScannerTests(unittest.TestCase):
    def test_python_skips_pragmas_shebang_strings_and_docstrings(self):
        found = cc.python_comments(PYTHON)
        self.assertEqual(
            [(c.line, c.text, c.own_line) for c in found],
            [(5, "# own line", True), (6, "# inline", False)],
        )

    def test_hash_files_skip_quotes_parameter_length_and_anchors(self):
        found = cc.hash_comments(SHELL)
        self.assertEqual(
            [(c.line, c.text, c.own_line) for c in found],
            [(2, "# own line", True), (6, "# inline", False)],
        )

    def test_only_known_languages_are_scanned(self):
        for name in ("a.py", "a.sh", "pixi.toml", "x.rules", "machine.env.example"):
            self.assertIsNotNone(cc.scanner(Path(name)), name)
        for name in ("README.md", "recipe.json", "pixi.lock"):
            self.assertIsNone(cc.scanner(Path(name)), name)
        self.assertIsNotNone(cc.scanner(Path("docker/smoke/Dockerfile")))
        self.assertIsNotNone(cc.scanner(Path(".githooks/pre-commit")))


def comments(count):
    return [cc.Comment(i + 1, 0, f"# c{i}", True) for i in range(count)]


class RatchetTests(unittest.TestCase):
    def test_new_or_extra_comments_fail_with_locations(self):
        errors = cc.check({"a.py": comments(2), "b.py": comments(1)}, {"a.py": 1})
        self.assertTrue(any(e.startswith("a.py: 2 comments") for e in errors))
        self.assertTrue(any(e.startswith("b.py: 1 comments") for e in errors))
        self.assertIn("  b.py:1:1: # c0", errors)

    def test_reduction_fails_until_baseline_is_lowered(self):
        errors = cc.check({"a.py": comments(1)}, {"a.py": 3, "gone.py": 1})
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(cc.UPDATE_TASK in e for e in errors))
        self.assertEqual(cc.check({"a.py": comments(1)}, {"a.py": 1}), [])

    def test_update_lowers_and_drops_but_never_raises(self):
        counts, raised = cc.tightened({"a.py": comments(1)}, {"a.py": 3, "gone.py": 2})
        self.assertEqual((counts, raised), ({"a.py": 1}, []))
        counts, raised = cc.tightened({"a.py": comments(4)}, {"a.py": 3})
        self.assertEqual((counts, raised), (None, ["a.py"]))

    def test_scan_includes_untracked_files_and_respects_gitignore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".gitignore").write_text("ignored/\n")
            (root / "ignored").mkdir()
            (root / "ignored/a.py").write_text("# hidden\n")
            (root / "new.py").write_text("x = 1  # new\n")
            (root / "clean.py").write_text("x = 1\n")
            found = cc.scan(root)
        self.assertEqual(list(found), ["new.py"])

    def test_main_update_writes_lowered_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.json"
            baseline.write_text(json.dumps({"a.py": 5}))
            with (
                patch.object(cc, "BASELINE", baseline),
                patch.object(cc, "scan", return_value={"a.py": comments(2)}),
                patch("builtins.print"),
            ):
                self.assertEqual(cc.main(["--update"]), 0)
                self.assertEqual(json.loads(baseline.read_text()), {"a.py": 2})
                self.assertEqual(cc.main([]), 0)


if __name__ == "__main__":
    unittest.main()
