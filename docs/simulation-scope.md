# Simulation Scope: Minimal UR5e Pick-and-Place Demo

> **Implementation checkpoint — scripted-teacher collection accepted as the minimal route.**
> CPU physics, rendering and native v3.0 recording have been exercised; manual operator checks remain incomplete. Training, fresh trained-checkpoint reload and learned-policy rollout are not yet verified or authorized to run.

**Baseline:** `56faae027fc0c1532c9323da362963a6d1bc3be1`, including six-joint/no-gripper support and pi0.5 VLM + action expert LoRA. The executable checkpoint has independent source review and retained offline/runtime evidence. Documentation review and teacher success do not establish learned-policy performance. Detailed review provenance and raw evidence remain local under ignored `outputs/`.

## Authority Boundaries & Open Owner Decisions

Keep the four authority layers strictly separated:

1. **Latest user start authority (active):** Supersedes the documentation-only HOLD for bounded implementation of the minimal UR5e + Robotiq 2F-85 simulation slice, routine project-local Pixi dependency locking/install (`mujoco-python`), pinned public Menagerie assets with upstream licenses/provenance, and CPU simulator verification for plumbing gates **P1** and **P2**.
2. **Operational bounds:** Worktree write isolation, cycle-time/download/evidence/CPU runtime caps, no GPU compute, no model weight loading/inference, no remote host access/probes, and no real hardware motion. A separate local checkpoint commit is authorized; push/publication and further runtime budgets are not.
3. **Still-unset owner training budgets (OPEN — P3/P4/L1 NOT authorized):** Training machine, GPU, storage, time, and finite run budgets remain **UNSET**. No remote machine may be accessed or probed, and no weights/training/model execution (`train-check` if model-accessing, `--smoke`, finite training, or learned rollout) may run until separate owner authorization resolves them.
4. **Rollout inference mode and physical parity (OPEN — UNRESOLVED):** Existing physical UR5e rollout uses **RTC** (`pixi.toml`, `scripts/rollout.py`). The historical synchronous default (`--inference.type=sync`) is **not accepted**. Simulation rollout targets the RTC-compatible path (`--inference.type=rtc`), never silently falls back to synchronous inference, and claims no tested RTC parity. Keep RTC tuning out of scope until the owner resolves required parity.

## Ordered Next Steps

1. **Minimal CPU simulation slice & P1/P2 verification (authorized now):** Implement the minimal MuJoCo env, pinned Menagerie UR5e + Robotiq 2F-85 assets, scripted physics teacher, thin simulation rollout glue, and offline tests; execute **P1** (live reset/step/contact/render grasp-lift-release-settle) and **P2** (native `lerobot-record` save/discard/exit + native `lerobot-edit-dataset` info/filtered copy) with evidence under ignored `outputs/`.
   - *Done when:* P1 and P2 runtime evidence is recorded under ignored `outputs/`, offline `pixi run lint` and `pixi run test` pass, and fresh independent review inspects the changed source.
2. **Owner resolution of remaining open decisions:** Obtain explicit owner resolutions for rollout RTC/parity acceptance and training compute/storage/time/finite budgets.
   - *Done when:* both remaining open owner decisions are explicitly recorded by the owner.
3. **Training mechanics & strict reload gates P3–P4 (blocked on Step 2; NOT authorized now):** Run `train-check`, GPU `--smoke`, and fresh-process strict reload (`scripts/policy_loading.py` + `validate_checkpoint` + unchanged saved normalizer stats).
4. **Finite learning run & closed-loop learned rollout gate L1 (blocked on separate owner run authority; NOT authorized now):** Run the authorized finite training budget, repeat P4 on the finite checkpoint, and run closed-loop learned simulation rollout gate L1 (`>= 1/5` verified learned successes with no teacher).

## Task and Learner Journey

- **Task:** UR5e + Robotiq 2F-85 pick-and-place — *"Pick up the red cube and place it on the blue tray"* on a table, using one fixed starting layout for the initial demo.
- **Physical manipulation & episode integrity:** Success requires genuine MuJoCo rigid-body contacts for grasp, lift off the table, carry, release, and rest inside the tray — verified against both saved video and an explicit simulator success predicate. Fixed-layout resets happen only between episodes; mid-episode physics uses no teleporting, welding, or fake feedback.
- **Learner journey (end-to-end):**
  1. **See** live rendered simulation collection (`front` and `wrist` RGB views, 7-D robot state, and action).
  2. **Curate** native LeRobot v3.0 datasets via native inspection, visualization, and filtered copies (`delete_episodes`; raw recordings stay immutable).
  3. **Train** (later phase, not authorized now) with the existing pi0.5 LoRA route (`scripts/train_lora.py`, `config/training/*.recipe.json`).
  4. **Reload** (later phase, not authorized now) the saved checkpoint and unchanged saved processors in a fresh process through the strict fail-closed loader (`scripts/policy_loading.py`).
  5. **Roll out** (later phase, not authorized now) closed-loop in simulation with the learned policy only, reviewing saved simulation-rollout videos.
- **Teacher separation & glossary boundary:** The scripted physics teacher exists for data collection only and is reported strictly as *teacher evidence*; it never runs during simulation rollout and never counts as learned success. Physical `GLOSSARY.md` terms (**Demonstration** and **Rollout**) remain reserved for the real-robot workflow.

## Reuse vs. Minimal New Code

- **Reuse unchanged:**
  - LeRobot 0.7.0 native `SimRobot`, `LocalVectorEnv` + local Gymnasium integration, native `lerobot-record`, native `LeRobotDataset` v3.0 writer, and native dataset tools (`lerobot-edit-dataset`, `lerobot-dataset-viz`).
  - Existing physical hardware plugin (`UR5e` including six-joint/no-gripper support, `SpaceMouse`, `scripts/record.py`, `scripts/rollout.py`), training route (`scripts/train_lora.py`, `scripts/train_lora_entry.py`, `config/training/*.recipe.json` including VLM + expert LoRA), strict fail-closed LoRA loader (`scripts/policy_loading.py`), and native episodic rollout engine (`EpisodicStrategy`, `RolloutContext`, plus shared `RolloutConfig` and `bind_relative_anchor` from `scripts/rollout.py`).
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

> Run evidence belongs under ignored `outputs/<topic>-<date>/`, never in tracked docs. Gates **P3**, **P4**, and **L1** are **UNEXECUTED and NOT AUTHORIZED** in this phase. Offline `pixi run lint` and `pixi run test` supplement — never replace — live physics and native writer execution.

### Pipeline-Execution Gates (Plumbing — Necessary, Insufficient for Learning)

- **P1. Environment (authorized in this phase):** Live simulator `reset`/`step` runs; `front` and `wrist` render non-blank RGB that changes with motion; env descriptor matches the 15 Hz / 7-D / 2-camera contract; genuine physical grasp, lift, carry, release, and rest in tray pass both the explicit success predicate and video review.
- **P2. Native recording & curation (authorized in this phase):** Native `lerobot-record` lifecycle (start, stop/save, discard, exit matching `num_episodes`) writes a 15 Hz v3.0 dataset with valid schema and playable front/wrist videos; native `lerobot-edit-dataset` inspection and filtered copy (`delete_episodes`) verify while leaving the original intact.
- **P3. Existing training mechanics (`--smoke`) — UNEXECUTED / NOT AUTHORIZED:** `train-check` followed by GPU `--smoke` verifying dataloader, image decoding, train-only normalization fitting, optimizer updates, and `check_saved_checkpoint`.
- **P4. Fresh-process strict reload — UNEXECUTED / NOT AUTHORIZED:** Fresh process reloads the saved adapter through `scripts/policy_loading.py` with zero missing/unexpected keys, passes native `validate_checkpoint`, and preserves saved normalizer tensors unchanged.

### Useful Learned-Outcome Gate (Learning — UNEXECUTED / NOT AUTHORIZED)

- **L1. Closed-loop learned simulation rollout — UNEXECUTED / NOT AUTHORIZED:** Reloaded finite-trained learned policy runs closed-loop with **no teacher** for **5 attempts** from matched fixed resets; report every started attempt, cross-check predicate successes against saved video; pass requires `>= 1/5` verified learned successes (fixed layout only — no reliability, generalization, safety, or sim-to-real claim).

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
