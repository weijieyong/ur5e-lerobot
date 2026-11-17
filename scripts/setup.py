"""Create the local config, or append missing defaults without changing existing values."""

import re
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    template = (root / "config/machine.env.example").read_text()
    local = root / "config/local/machine.env"
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        with local.open("x") as stream:
            stream.write(template)
        print(f"Created {local}")
    else:
        assignment = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=")
        original = local.read_text()
        existing = {
            match[1]
            for line in original.splitlines()
            if (match := assignment.match(line))
        }
        additions = []
        for block in template.split("\n\n"):
            missing = [
                line
                for line in block.splitlines()
                if (match := assignment.match(line)) and match[1] not in existing
            ]
            if missing:
                additions.append(
                    "\n".join(
                        line
                        for line in block.splitlines()
                        if not (match := assignment.match(line))
                        or match[1] not in existing
                    )
                )
        if additions:
            with local.open("a") as stream:
                stream.write(
                    "\n\n# Missing defaults added by setup; existing values preserved.\n"
                    + "\n\n".join(additions)
                    + "\n"
                )
            print(f"Added missing defaults to {local}; existing values preserved")
        else:
            print(f"Config already up to date: {local}")
    if (root / ".git").exists():
        subprocess.run(
            ["git", "config", "core.hooksPath", ".githooks"], cwd=root, check=True
        )
        print("Enabled .githooks/pre-commit (lint + offline tests)")
    print(
        "Edit this file: check IPs/serials, home pose, TCP floor height and camera colour settings before motion."
    )


if __name__ == "__main__":
    main()
