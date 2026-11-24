# 0001: No code comments

Status: accepted (2026-11-24)

## Context

An audit of every comment in the repository found four kinds:

- **Tautology**: restates the code beside it.
- **Stale**: describes code, runs or projects that no longer exist.
- **Excuse**: justifies a workaround, so the workaround is never fixed and gets copied.
- **Decision record**: a real, non-obvious decision, but invisible to anyone not reading that exact line.

Only the last kind carries value, and it is easier to find, review and keep current in one place.

## Decision

- Code carries no comments. Tool pragmas (`# noqa`, `# fmt:`, `# type:`, `# pragma:`, `# ruff:`,
  `# pyright:`, `# mypy:`), encoding lines and shebangs are not comments.
- Decision records live in `docs/adr/`. Workarounds are tracked as GitHub issues, not explained inline.
- `scripts/check_comments.py` runs in `pixi run lint`. It scans Python (tokenizer) and `#`-comment
  files (shell, TOML, Dockerfile, udev rules, `.env.example`), and fails when a file holds more
  comments than `config/comment-baseline.json` allows.
- The baseline is a ratchet. When a file drops below its allowance, lint fails until
  `pixi run lint-comments-update` lowers it; that command refuses to raise any entry. Raising one
  means hand-editing the JSON, which a reviewer sees.
- `/delete-comment` (`.claude/skills/delete-comment/`) repeats the audit.

## Consequences

- `config/machine.env.example` keeps its comments for now: they are the operator's documentation and
  `scripts/setup.py` copies them into each machine's config. They sit in the baseline until that
  documentation moves to `docs/operating-guide.md`.
- Docstrings are not comments and are not linted.
- Reading a module now means reading its ADR too; `docs/adr/README.md` maps modules to ADRs.
