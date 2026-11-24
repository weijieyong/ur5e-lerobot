# Simulation Scope: Minimal UR5e Pick-and-Place Demo

> **Documentation-only phase — finite local GPU pilot completed; P1, P3, and P4 passed; P2 partially verified; L1 executed and failed (`0/5` learned successes).**
> Scripted-teacher collection is accepted as the minimal route; the completed pilot ran `10` teacher episodes through `1,000`-update expert-only pi0.5 LoRA training (`config/training/sim-pilot.recipe.json`), strict fresh-process reload, and `5` teacher-free RTC rollout attempts (`0/5` verified successes; fixed layout only, no learned-success, reliability, or generalization claim; evidence: `outputs/simulation-learning-ram40-1574807730323984446/`). Manual operator and GUI checks remain incomplete. Prior pilot authority is spent; no new execution, downloads/installations, remote access, physical motion, or git staging/commit/push/publication is authorized now.

**Baseline:** `56faae027fc0c1532c9323da362963a6d1bc3be1`, including six-joint/no-gripper support and pi0.5 VLM + action expert LoRA. The executable checkpoint has independent source review and retained offline/runtime evidence. Documentation review and teacher success do not establish learned-policy performance. Detailed review provenance and raw evidence remain local under ignored `outputs/`.

## Authority Boundaries & Open Owner Decisions

Keep the four authority layers strictly separated:

1. **Current authority (active — documentation-only):** Scope is limited to maintaining status documentation here. No new training, rollout, model execution, downloads/installations, remote access, physical motion, or git staging/commit/push/publication is authorized without fresh owner approval.
2. **Completed pilot authority (historical — spent):** Prior owner approvals covered the bounded CPU simulation slice (`mujoco-python`, pinned Menagerie assets, P1/P2 checks) and one finite local GPU pilot under a `40 GiB` RAM cap sharing only existing persistent services, using the `1,000` optimizer-update / `60`-minute training and `5`-evaluation / `10`-minute limits. That budget is consumed and grants no new execution authority; cached offline models and token auth are not ongoing blockers.
3. **Future run and publication budgets (OPEN — NOT authorized now):** Any subsequent training, tuning, rollout, manual check, or git commit/push/publication requires explicit new owner authorization with fresh compute, storage, and time limits.
4. **Rollout inference mode and physical parity (OPEN — UNRESOLVED):** Existing physical UR5e rollout uses **RTC** (`pixi.toml`, `scripts/rollout.py`). The historical synchronous default (`--inference.type=sync`) is **not accepted**. Simulation rollout targets the RTC-compatible path (`--inference.type=rtc`), never silently falls back to synchronous inference, and claims no tested RTC parity. The completed pilot measured a `12.45 Hz` software loop against the `15 Hz` nominal target, with no claimed causal explanation of the `0/5` outcome. Keep RTC tuning out of scope until the owner resolves required parity.

## Ordered Next Steps

1. **Minimal CPU simulation slice & P1/P2 verification (P1 passed; P2 partially verified):** Implemented the minimal MuJoCo env, pinned Menagerie UR5e + Robotiq 2F-85 assets, scripted physics teacher, thin simulation rollout glue, and offline tests. **P1** passed and the accepted scripted collection/curation critical path completed; manual operator and GUI checks under **P2** remain incomplete and are not prerequisites to the completed pilot.
2. **Finite local GPU pilot — P3, P4, and L1 (completed under spent pilot authority):** Executed the `1,000`-update finite training run (passing **P3** directly without a separate `--smoke` run), verified strict fresh-process reload (**P4** passed), and ran `5` teacher-free RTC rollout attempts (**L1** executed and failed at `0/5`; details in Acceptance Gates and `outputs/simulation-learning-ram40-1574807730323984446/`).
3. **Owner decisions (current gate — documentation-only now):** Obtain explicit owner decisions on committing the verified checkpoint delta, resolving RTC physical parity (Open Decision #4), and authorizing any follow-on run budget or remaining manual operator/GUI checks.
   - *Done when:* the owner records explicit authorization for the chosen next action.
4. **Follow-on learning run & re-evaluation of L1 (blocked on Step 3; NOT authorized now):** Only under fresh owner run authority, run any follow-on training budget, repeat **P4** on any new checkpoint, and re-run closed-loop learned simulation rollout gate **L1** (`>= 1/5` verified learned successes with no teacher).

## Task and Learner Journey

- **Task:** UR5e + Robotiq 2F-85 pick-and-place — *"Pick up the red cube and place it on the blue tray"* on a table, using one fixed starting layout for the initial demo.
- **Physical manipulation & episode integrity:** Success requires genuine MuJoCo rigid-body contacts for grasp, lift off the table, carry, release, and rest inside the tray — verified against both saved video and an explicit simulator success predicate. Fixed-layout resets happen only between episodes; mid-episode physics uses no teleporting, welding, or fake feedback.
- **Learner journey (end-to-end):**
  1. **See** live rendered simulation collection (`front` and `wrist` RGB views, 7-D robot state, and action).
  2. **Curate** native LeRobot v3.0 datasets via native inspection, visualization, and filtered copies (`delete_episodes`; raw recordings stay immutable).
  3. **Train** with the existing pi0.5 LoRA route (`scripts/train_lora.py`, `config/training/*.recipe.json`, including `sim-pilot.recipe.json`) — exercised once in the completed pilot; new runs require fresh owner authority.
  4. **Reload** the saved checkpoint and unchanged saved processors in a fresh process through the strict fail-closed loader (`scripts/policy_loading.py`) — verified in the completed pilot.
  5. **Roll out** closed-loop in simulation with the learned policy only, reviewing saved simulation-rollout videos — exercised in the completed pilot (`0/5` successes); new runs require fresh owner authority.
- **Teacher separation & glossary boundary:** The scripted physics teacher exists for data collection only and is reported strictly as *teacher evidence*; it never runs during simulation rollout and never counts as learned success. Physical `GLOSSARY.md` terms (**Demonstration** and **Rollout**) remain reserved for the real-robot workflow.

## Reuse vs. Minimal New Code

- **Reuse unchanged:**
  - LeRobot 0.7.0 native `SimRobot`, `LocalVectorEnv` + local Gymnasium integration, native `lerobot-record`, native `LeRobotDataset` v3.0 writer, and native dataset tools (`lerobot-edit-dataset`, `lerobot-dataset-viz`).
  - Existing physical hardware plugin (`UR5e` including six-joint/no-gripper support, `SpaceMouse`, `scripts/record.py`, `scripts/rollout.py`), strict fail-closed LoRA loader (`scripts/policy_loading.py`), and native episodic rollout engine (`EpisodicStrategy`, `RolloutContext`, plus shared `RolloutConfig` and `bind_relative_anchor` from `scripts/rollout.py`).
- **Reused training route with scoped additions:**
  - Existing pi0.5 LoRA training route (`scripts/train_lora.py`, `scripts/train_lora_entry.py`, `config/training/*.recipe.json` including VLM + expert LoRA) reused with scoped GPU admission, resource/telemetry, and `TMPDIR` guard fixes plus `config/training/sim-pilot.recipe.json`; no replacement trainer.
- **Minimal new task-specific code only:**
  - Pinned Menagerie UR5e + Robotiq 2F-85 assets (with upstream licenses and `PROVENANCE.md`) and composed MuJoCo scene (`table`, red cube, blue tray, `front` and `wrist` cameras).
  - Small Gymnasium environment (`UR5ePickPlaceEnv`) and registered LeRobot `EnvConfig` (`ur5e_sim`).
  - Scripted physics teacher teleoperator (`ur5e_sim_teacher`, collection only).
  - Thin simulation rollout entry glue (`scripts/sim_rollout.py`) and explicit isolated Pixi tasks.
- **Scope boundary:** Stay inside the pinned LeRobot/Pixi stack; build no replacement recorder, trainer, dataset export/conversion pipeline, dashboard, or framework.

## Feature, Control, and Data Contract

- **Robot state (`observation.state`, `float32[7]`):** six measured joint angles in radians (`joint_0.pos` .. `joint_5.pos`, base to wrist 3) followed by measured `gripper.pos` (`0` open, `1` closed).
- **Action (`action`, `float32[7]`):** six absolute joint targets (`joint_0.pos` .. `joint_5.pos`) followed by `gripper.pos` (`0` open, `1` closed; `< 0.5` open, `>= 0.5` closed).
- **Visual observations:** `observation.images.front` and `observation.images.wrist` (`480x640x3` `uint8` RGB rendered from current MuJoCo physics state, matching `config/machine.env.example` camera dimensions).
- **Rate:** `15 Hz` control and dataset `fps`.
- **Observation hygiene:** Learned-policy observations contain strictly `observation.state`, `observation.images.front`, and `observation.images.wrist`. Ground-truth object/tray poses, contacts, and the success predicate stay out of learned observations; only the scripted teacher may read simulator truth.
- **Native format & isolation:**
  - Write directly via native `LeRobotDataset` v3.0 with no intermediate format or conversion.
  - Keep simulation dataset repo IDs, roots, and checkpoints explicit and isolated from real-robot `config/local/machine.env` defaults and hardware convenience tasks (`info`, `view`, `rollout-info`, `rollout-view`).
  - Keep Hub push disabled and offline local loading enforced on every command.

## Hardware vs. Simulation Rollout Lifecycle

- Keep physical hardware wrappers (`scripts/record.py`, `scripts/rollout.py`) and their guards (`home_joints_deg`, `measured_joints()`, `go_home()`, `ROLLOUT_START_TOLERANCE_DEG`, SpaceMouse tilt recovery) unchanged.
- Simulation rollout (`scripts/sim_rollout.py`) reuses native `EpisodicStrategy`, `RolloutContext`, `scripts/rollout.py`'s `RolloutConfig` and `bind_relative_anchor`, and `scripts/policy_loading.py`'s strict loader, adding only thin `SimRobot` episode reset/end lifecycle handling (`SimRobot.start_episode` and `episode_ended`) and refusing `--teleop`.
- Rollout inference mode (RTC vs. sync) and parity remain governed by Open Owner Decision #4 above.

## Acceptance Gates

> Raw run logs, models, datasets, and receipts belong under ignored `outputs/<topic>-<date>/` (completed pilot evidence: `outputs/simulation-learning-ram40-1574807730323984446/`); tracked docs carry only brief maintained status and result summaries. Gates **P3**, **P4**, and **L1** ran in the completed pilot; no new execution is authorized in the current documentation-only phase. Offline `pixi run lint` and `pixi run test` supplement — never replace — live physics and native writer execution.

### Pipeline-Execution Gates (Plumbing — Necessary, Insufficient for Learning)

- **P1. Environment — PASSED:** Live simulator `reset`/`step` runs; `front` and `wrist` render non-blank RGB that changes with motion; env descriptor matches the 15 Hz / 7-D / 2-camera contract; genuine physical grasp, lift, carry, release, and rest in tray pass both the explicit success predicate and video review.
- **P2. Native recording & curation — PARTIALLY VERIFIED:** Native `lerobot-record` lifecycle (start, stop/save, discard, exit matching `num_episodes`) writes a 15 Hz v3.0 dataset with valid schema and playable front/wrist videos; native `lerobot-edit-dataset` inspection and filtered copy (`delete_episodes`) verify while leaving the original intact. The accepted scripted collection/curation route verified `10` deterministic fixed-layout episodes (`2,170` frames; train `0–7` / dev `8–9`, train-only stats); manual operator and GUI checks remain incomplete and are not prerequisites to the completed pilot.
- **P3. Existing training mechanics — PASSED (via 1,000-update finite run):** Verified dataloader, image decoding, train-only normalization fitting, optimizer updates, and saved checkpoint integrity directly on the `1,000`-update finite run (`config/training/sim-pilot.recipe.json`, batch `1`, expert-only pi0.5 LoRA with `80` trainable tensors and VLM frozen; all `80` adapter tensors changed from step `250` to `1000`; no separate `--smoke` run was executed).
- **P4. Fresh-process strict reload — PASSED:** Fresh process reloaded the saved base model and adapter through `scripts/policy_loading.py` with zero missing/unexpected keys, passed native `validate_checkpoint`, and verified all `180` saved processor normalization tensors identical.

### Useful Learned-Outcome Gate (Learning — EXECUTED / FAILED)

- **L1. Closed-loop learned simulation rollout — EXECUTED / FAILED (`0/5`):** Reloaded finite-trained learned policy runs closed-loop with **no teacher** for **5 attempts** from matched fixed resets; report every started attempt, cross-check predicate successes against saved video; pass requires `>= 1/5` verified learned successes (fixed layout only — no learned-success, reliability, generalization, safety, or sim-to-real claim). Completed pilot result: `5` teacher-free policy-only RTC attempts started and recorded (`0` discarded, `1,540` frames, `12.45 Hz` measured software loop vs. `15 Hz` nominal target, no claimed causal explanation of failure) with `0/5` verified successes cross-checked against saved videos.

## Source-of-Truth Pointers

Look up live commands, defaults, schemas, and recipes in source instead of caching them here:

| Topic | Authoritative Source |
|---|---|
| Dependencies & Pixi task definitions | `pixi.toml`, `pixi.lock` |
| Canonical hardware terms | `GLOSSARY.md` |
| Physical UR5e feature names, control mapping, homing | `src/lerobot_robot_ur5e/lerobot_robot_ur5e/ur5e.py`, `config_ur5e.py` |
| Simulation env, scene, teacher, `EnvConfig`, asset provenance | `src/lerobot_robot_ur5e/lerobot_robot_ur5e/sim/` |
| Live hardware operations & physical RTC rollout | `docs/operating-guide.md`, `scripts/record.py`, `scripts/rollout.py` |
| Thin simulation rollout lifecycle glue | `scripts/sim_rollout.py` |
| Dataset filtering, LoRA recipes, training guards, deployment | `docs/training.md`, `config/training/*.recipe.json`, `scripts/train_lora.py`, `scripts/train_lora_entry.py` |
| Strict fail-closed LoRA policy loader | `scripts/policy_loading.py` |
| Native LeRobot 0.7.0 sim, recording, and episodic rollout | `lerobot.robots.sim_robot`, `lerobot.sim.integrations.gym`, `lerobot.scripts.lerobot_record`, `lerobot.rollout.strategies.episodic` |
