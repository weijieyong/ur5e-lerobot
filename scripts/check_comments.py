"""Comment ratchet: no file may hold more comments than config/comment-baseline.json allows.

Rationale belongs in docs/adr/; see docs/adr/0001-no-code-comments.md.
"""

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config/comment-baseline.json"
UPDATE_TASK = "pixi run lint-comments-update"
PRAGMA = re.compile(
    r"#\s*(noqa\b|type:|fmt:|pragma:|ruff:|pyright:|mypy:|-\*-\s*coding)"
)
HASH_SUFFIXES = {".sh", ".toml", ".rules", ".dockerignore", ".example", ".cfg"}
HASH_NAMES = {"Dockerfile", "pre-commit"}


@dataclass(frozen=True)
class Comment:
    line: int
    column: int
    text: str
    own_line: bool


def python_comments(text: str) -> list[Comment]:
    lines = text.splitlines()
    found = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        row, column = token.start
        if (row == 1 and token.string.startswith("#!")) or PRAGMA.match(token.string):
            continue
        own_line = not lines[row - 1][:column].strip()
        found.append(Comment(row, column, token.string, own_line))
    return found


def hash_comments(text: str) -> list[Comment]:
    found = []
    for row, line in enumerate(text.splitlines(), start=1):
        if row == 1 and line.startswith("#!"):
            continue
        quote = None
        for column, char in enumerate(line):
            if quote:
                if char == quote and line[column - 1] != "\\":
                    quote = None
            elif char in "'\"":
                quote = char
            elif char == "#" and (column == 0 or line[column - 1] in " \t"):
                own_line = not line[:column].strip()
                found.append(Comment(row, column, line[column:], own_line))
                break
    return found


def scanner(path: Path):
    if path.suffix == ".py":
        return python_comments
    if path.suffix in HASH_SUFFIXES or path.name in HASH_NAMES:
        return hash_comments
    return None


def source_files(root: Path = ROOT) -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.decode()
    return sorted(
        Path(name)
        for name in listed.split("\0")
        if name and scanner(Path(name)) and (root / name).is_file()
    )


def scan(root: Path = ROOT) -> dict[str, list[Comment]]:
    found = {}
    for path in source_files(root):
        comments = scanner(path)((root / path).read_text())
        if comments:
            found[path.as_posix()] = comments
    return found


def load_baseline(path: Path | None = None) -> dict[str, int]:
    path = path or BASELINE
    return json.loads(path.read_text()) if path.is_file() else {}


def check(found: dict[str, list[Comment]], baseline: dict[str, int]) -> list[str]:
    errors = []
    for path, comments in found.items():
        allowed = baseline.get(path, 0)
        if len(comments) > allowed:
            errors.append(
                f"{path}: {len(comments)} comments, baseline allows {allowed}. "
                "Move rationale to docs/adr/, file an issue for a workaround, or delete it:"
            )
            errors.extend(
                f"  {path}:{c.line}:{c.column + 1}: {c.text}" for c in comments
            )
    for path, allowed in sorted(baseline.items()):
        actual = len(found.get(path, ()))
        if actual < allowed:
            errors.append(
                f"{path}: {actual} comments, baseline still allows {allowed}. "
                f"Lock in the reduction with `{UPDATE_TASK}`."
            )
    return errors


def tightened(found: dict[str, list[Comment]], baseline: dict[str, int]):
    raised = sorted(p for p, c in found.items() if len(c) > baseline.get(p, 0))
    if raised:
        return None, raised
    counts = {p: len(found[p]) for p in sorted(found)}
    return counts, []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="lower the baseline to current counts; never raises it",
    )
    args = parser.parse_args(argv)
    found = scan()
    baseline = load_baseline()
    if args.update:
        counts, raised = tightened(found, baseline)
        if raised:
            print(
                "Refusing to raise the comment baseline for: " + ", ".join(raised),
                file=sys.stderr,
            )
            return 1
        BASELINE.write_text(json.dumps(counts, indent=2) + "\n")
        print(
            f"Comment baseline: {sum(counts.values())} comments in {len(counts)} files"
        )
        return 0
    errors = check(found, baseline)
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(f"Comment ratchet: {sum(baseline.values())} comments remain in baseline")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
