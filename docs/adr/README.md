# Architecture decision records

Code in this repository carries no comments ([ADR 0001](0001-no-code-comments.md)). A non-obvious
decision is recorded here instead, under the module it governs. Read the ADR for a file before
changing behaviour it describes.

| ADR | Decision | Governs |
| --- | --- | --- |
| [0001](0001-no-code-comments.md) | No code comments; a ratchet keeps the count falling | `scripts/check_comments.py`, `config/comment-baseline.json` |
| [0002](0002-hardware-access-safety.md) | Hardware access never moves or powers anything unasked | `ur5e.py`, `hw_check.py`, gripper fields of `config_ur5e.py` |
| [0003](0003-teleoperation-motion-limits.md) | Motion limits and SpaceMouse control semantics | `spacemouse.py`, `config_ur5e.py`, `set_boundaries.py` |
| [0004](0004-operator-gated-episodes.md) | The operator, not a timer, gates every recording and rollout phase | `record.py`, `rollout.py`, `rollout_input.py`, `policy_loading.py` |
| [0005](0005-training-invariants.md) | Training runs only the supported, verified route | `train_lora.py`, `train_lora_entry.py` |
| [0006](0006-camera-calibration.md) | Camera calibration waits for firmware and tunes on a midtone | `calibrate_cameras.py` |
| [0007](0007-test-doubles-and-oracles.md) | Tests fake hardware only and use independent oracles | `tests/` |
| [0008](0008-pinned-environment.md) | The Pixi environment is isolated from the host | `pixi.toml`, `scripts/activate.sh`, `docker/smoke/` |

New ADRs take the next number. Keep each entry to: context, decision, consequences.
