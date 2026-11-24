# 0004: The operator gates every recording and rollout phase

Status: accepted (2026-11-24)

## Context

Upstream LeRobot runs episodes and resets on timers. On a physical robot, a supervising operator
must decide when each episode starts, is saved or is discarded, and the robot must never start or
home on its own.

## Decision

### Recording (`scripts/record.py`)

- The operator controls every phase boundary. Upstream episode and reset timers are disabled.
- READY lies outside the recording cadence window: the loop timer restarts when recording starts.
- Next/retry inputs queued while saving are dropped. READY needs fresh operator input.
- The first Ctrl+C behaves like Esc, so the buffered episode is still written. A second Ctrl+C
  interrupts a stuck control call.
- Recording opens its own Rerun viewer on a free port and never attaches to an existing viewer. The
  viewer keeps LeRobot's default grid (and lets it grow) and adds only a phase/controls strip on top.

### Rollout (`scripts/rollout.py`)

- Only the main control thread commands motion; failures propagate.
- Homing never starts an episode. Measured joints are checked against home on START, within
  `ROLLOUT_START_TOLERANCE_DEG`.
- START admission, task latch, engine reset and resume form one boundary under the control lock.
  Requests that arrive during the reset are rechecked before resuming.
- While paused, observations stay live (safety and tilt checks run) and no action is sent.
- A terminal-restoration failure during teardown must not skip native dataset and hardware teardown.
- Joint vectors must hold six finite scalars. Nested arrays are rejected, because `float()` accepts
  one-element arrays.
- Rollout settings are printed before model loading and hardware setup, because native logging
  starts later.
- `--policy.path` is resolved with the native `parser.get_path_arg` (CLI precedence and extracted
  YAML/JSON paths) and checked locally *before* native `__post_init__`, which would fall back to the
  Hub. A cached non-PI05 config must not substitute for a missing local deployment.
- RTC deep-copies the preprocessor; that copy must not duplicate the policy on the GPU.

### Rollout keyboard (`scripts/rollout_input.py`)

- Terminal framing is enabled before the reader can consume any input.
- A key callback never blocks on the admission lock. Blocking would replay the key in READY. Input
  errors are published even while the main thread holds the lock.
- Stop requests (`q`, Esc) still reach a START in progress, except while editing the task.
- After a lone-Esc timeout, a late escape sequence is still consumed whole. Its suffix must never
  become home/start/save/quit keys.

### Policy loading (`scripts/policy_loading.py`)

- Upstream `PI05Policy.from_pretrained` catches load failures and can return random weights. The base
  and adapter are loaded directly instead, with strict key checks.

## Consequences

Upstream lifecycle changes must be reviewed against this list. The places where we override upstream
hooks to enforce it are tracked as issues; see `docs/adr/0001-no-code-comments.md`.
