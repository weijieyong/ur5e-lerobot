---
name: delete-comment
description: Audit code comments, delete them, move decision records into docs/adr/, file GitHub issues for workarounds, and lower the comment ratchet. Use when asked to delete, audit or burn down comments, or when `pixi run lint` reports the comment ratchet.
disable-model-invocation: true
argument-hint: "[path ...]  (default: every file in config/comment-baseline.json plus any lint offenders)"
---

# /delete-comment

Policy: [docs/adr/0001-no-code-comments.md](../../../docs/adr/0001-no-code-comments.md). Code carries
no comments. Pragmas (`# noqa`, `# fmt:`, `# type:`, `# pragma:`, `# ruff:`, `# pyright:`, `# mypy:`),
shebangs and docstrings are not comments.

## 1. Find the comments

```bash
pixi run lint                              # offenders above the baseline, with file:line
pixi run python -c "from scripts.check_comments import scan
for p, cs in scan().items():
    [print(f'{p}:{c.line}: {c.text}') for c in cs]"
```

Restrict to `$ARGUMENTS` when paths are given. Read the code around every comment before you classify
it. A comment's category depends on the code, not the wording.

## 2. Classify each comment

| Category | Test | Action |
| --- | --- | --- |
| Tautology | The code beside it already says this: names, units in the identifier, a test's assertion, section dividers. | Delete. |
| Stale | It names a symbol, run, machine, project or behaviour that no longer exists (grep to confirm). | Delete. |
| Excuse | It explains *why a workaround is acceptable*: monkeypatching or private upstream APIs, string-matching log text, relying on aliasing or mutation, swallowing `SystemExit`, legacy aliases, "upstream does X so we do Y". | Delete **and** file an issue. |
| Decision record | A non-obvious, still-true decision a maintainer must not undo: safety rules, hardware quirks, invariants, ordering, units, why an obvious alternative fails. | Move into an ADR, then delete. |

When in doubt between Excuse and Decision record, ask: would we remove this code if upstream or our
design improved? Yes means Excuse. A comment with several parts can fall in several categories; split it.

Keep the classification as run evidence in `outputs/comment-audit-<date>/classification.tsv` (`file`,
`line`, `category`, `destination`, `comment`). That directory is git-ignored.

## 3. Act

- **Decision record**: add it to the ADR that governs the module (see the table in
  `docs/adr/README.md`). Write it as a decision, naming the symbol, not a line number. Create
  `docs/adr/NNNN-<slug>.md` (next number; Status / Context / Decision / Consequences) only when no
  existing ADR fits, and add it to the README table.
- **Excuse**: one issue per workaround, not per comment:
  `gh issue create --title "<what is hacky>" --body-file <file>`. The body states the location
  (file and symbol), the deleted comment verbatim, why it is fragile, and a suggested fix. Do not
  file a duplicate: run `gh issue list --search "<symbol>"` first.
- **Delete** the comment. Remove an own-line comment's whole line. Strip an inline comment and its
  trailing whitespace. Then reformat only what the deletion touched. Some files are not ruff-format
  clean, so use `ruff format` on files that were clean before, and `ruff format --range a:b` (or a hand
  edit) elsewhere. Never commit unrelated reformatting.
- Where a decision record or ADR says "operator documentation" (setup steps, config meaning), move it
  to `docs/operating-guide.md` or `docs/training.md` instead of an ADR.

## 4. Lock in the ratchet

```bash
pixi run lint-comments-update   # lowers config/comment-baseline.json; refuses to raise it
pixi run lint && pixi run test  # both must pass (offline, no hardware)
```

Never raise a baseline entry to make lint pass. If a comment truly must stay, hand-edit the JSON in a
separate commit and say why in the PR.

## 5. Report

Give counts per category, the ADRs touched, the issues filed (with links), the comment lines removed
(`git diff --shortstat`) and the remaining baseline total.
