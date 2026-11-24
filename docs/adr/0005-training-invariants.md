# 0005: Training runs only the supported, verified route

Status: accepted (2026-11-24)

## Context

LoRA training of pi0.5 is launched through `scripts/train_lora.py` onto shared GPUs. A plausible run
with a silently wrong config, base or GPU set wastes days, so every run proves its configuration.
See `docs/training.md` for usage.

## Decision

### Plan and validation (`scripts/train_lora.py`)

- `REQUIRED_FLAGS` are invariants of the supported route, whatever recipe is named.
- Recipes carry no dataset inputs. The caller supplies `--dataset-root` and `--repo-id`.
- `SMOKE_OVERRIDES` define a mechanics-only GPU gate: native eval and save twice on both ranks. A
  smoke run is not a learning result.
- Native validation parses the real `TrainPipelineConfig`. `device_requested` is read from the
  flags, because the CPU-only parse may report `cpu` while ranks use `cuda`.
- `--dry-run` validates every launch-resolvable topology (1..`dp_replicate` GPUs) on CPU.
- A real launch selects idle eligible GPUs *before* building the effective native plan.

### GPU admission

- Selected GPU UUIDs are revalidated immediately before `torchrun`. Admission never waits or retries.
- Once admitted, the global admission lock is released, so a run on another GPU can launch while this
  one trains. Per-GPU reservations stay held until the child finishes or is stopped.

### Worker (`scripts/train_lora_entry.py`)

- Topology is recomputed from the original preset and the declared GPU selection, never taken from
  `run.json`'s effective flags.
- Two action layouts are supported: six arm joints, optionally followed by the gripper.
- Train-only statistics select columns, which avoids image/video decoding and respects the native
  episode filter.
- The original base must be loaded exactly once, strictly and cleanly, with positive evidence.
- `init_adapter` warm-starts from a previous run's LoRA weights, which must use the same LoRA config.
- The worker receives the native (scaled) scheduler first, then rebuilds a fresh one on the same
  optimizer with `cfg.steps`, which keeps the decay horizon.

## Consequences

A new recipe can change only what `REQUIRED_FLAGS` leaves open. Topology or scheduler changes need
matching updates to `tests/test_train_lora.py`.
