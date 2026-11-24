# Simulation Guide: UR5e Pick-and-Place in MuJoCo

Fixed-layout MuJoCo simulation of the UR5e + Robotiq 2F-85 pick-and-place task (`"Pick up the red cube and place it on the blue tray"`) in the combined Pixi environment (LeRobot 0.7.0, `mujoco-python` 3.16.0).

## Status and limits

- **Implemented:** MuJoCo env (`ur5e_sim` / `lerobot_robot_ur5e/UR5ePickPlace-v0`) from unmodified pinned Menagerie models, collection-only scripted teacher (`ur5e_sim_teacher`), native v3.0 recording (`sim-record`), shared pi0.5 LoRA training, strict checkpoint reload, and teacher-free RTC rollout (`sim-rollout`).
- **Tested:**
  - `pixi run lint` and `pixi run test` (208 offline tests) pass.
  - A 1-episode CPU recording smoke wrote 217 rows at 15 FPS and two decoded 640x480 RGB videos (`front`, `wrist`) in v3.0 format.
  - Physical entry imports (`scripts/record.py`, `scripts/rollout.py`) and plugin registration did not load `mujoco` or `lerobot_robot_ur5e.sim.env` (no hardware connected).
  - Historical pilot: 10 identical teacher episodes (2170 frames), 8/2 whole-episode split with train-only stats, 1000 expert-only pi0.5 LoRA updates (`sim-pilot.recipe.json`, batch 1), strict fresh-process reload, and 5 teacher-free RTC rollouts scoring **0/5** (1540 frames; 12.45 Hz loop vs. 15 Hz nominal).
- **Unqualified and not verified:**
  - **Mechanics only:** tested one fixed layout (single cube, tray, and arm start pose), not learned success, reliability, or generalization.
  - **No sim-to-real:** MuJoCo does not qualify physical motion, contacts, or safety guards; follow [operating-guide.md](operating-guide.md) on hardware.
  - **RTC parity untested:** scheduling parity with physical UR5e RTC rollout is unverified; the 12.45 Hz loop does not establish whether timing affected 0/5.
  - **Interactive checks incomplete:** manual recording keys and live GUI/Rerun preview paths in simulation are not fully verified.
  - **MuJoCo 3.16.0 pin:** pinned/tested conda-forge version selected initially, not a hard minimum or LeRobot requirement; no comparative qualification or 3.16-specific rationale was recorded, and vendored Menagerie README minima are not project-wide compatibility claims.

## Task, data, and control contract

- **Physics & success:** Tabletop with a 5 cm red cube, blue tray, and fixed `front`/`wrist` cameras. Resets happen only between episodes; mid-episode motion uses MuJoCo rigid-body contacts with no teleporting, welding, or fake grasp attachment. `EpisodePhysicsStatus` requires lifting the cube off the table/tray while contacting both gripper pads, then releasing it to settle inside the tray with no robot contact; cross-check successes against saved videos.
- **Teacher & terminology:** Only `ur5e_sim_teacher` reads simulator truth (`UR5ePickPlaceEnv.teacher_state()`). Learned observations omit object/tray poses, contacts, and success flags. `scripts/sim_rollout.py` refuses `--teleop`, never runs a teacher, and teacher episodes never count as learned success. [GLOSSARY.md](../GLOSSARY.md) terms (**Demonstration**, **Rollout**) remain reserved for hardware.
- **Features (15 Hz nominal):** `observation.state` (`float32[7]`) holds six measured joints in radians (`joint_0.pos`..`joint_5.pos`, base to wrist 3) + measured `gripper.pos` (`0` open, `1` closed). `action` (`float32[7]`) holds six absolute joint targets in radians + `gripper.pos` (`< 0.5` open, `>= 0.5` closed). `observation.images.front` and `observation.images.wrist` are `480x640x3` `uint8` (640x480 RGB).
- **Local v3.0 isolation:** Writes native `LeRobotDataset` v3.0 directly. Keep simulation roots, repo IDs, and checkpoints separate from `config/local/machine.env` defaults and hardware tasks (`info`, `view`, `rollout-info`, `rollout-view`). Both sim tasks set `--dataset.push_to_hub=false` and `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

## Usage

Commands live in [pixi.toml](../pixi.toml); examples use project-relative paths.

**1. Record (`sim-record`)** — args: `dataset_root`, `repo_id`, `episodes` (default `2`), `extra`.

```sh
pixi run sim-record data/local/sim_pick_cube local/sim_pick_cube 10
```

For inspection or filtered copies (`lerobot-edit-dataset` with `--push_to_hub=false`, raw data intact), see [Filter saved episodes](training.md#filter-saved-episodes).

**2. Train (`train-check` / `train`)** — reuses the cache, `--smoke` option, and train-only normalization in [training.md](training.md). Both seven-value (simulation and gripper hardware) and six-joint (`GRIPPER_TYPE=none`) datasets work on the shared trainer:

```sh
pixi run train-check data/local/sim_pick_cube local/sim_pick_cube -- --recipe config/training/sim-pilot.recipe.json
pixi run train data/local/sim_pick_cube local/sim_pick_cube -- --recipe config/training/sim-pilot.recipe.json
```

**3. Roll out (`sim-rollout`)** — args: `policy_path`, `dataset_root`, `repo_id`, `episodes` (default `5`), `extra`. Replace `REPLACE_WITH_RUN` and step `001000` with your saved `pretrained_model` path:

```sh
pixi run sim-rollout \
  outputs/training-runs/REPLACE_WITH_RUN/native/checkpoints/001000/pretrained_model \
  data/local/sim_rollout_eval local/sim_rollout_eval 5
```

`scripts/sim_rollout.py` enforces `--robot.type=sim`, `--strategy.type=episodic`, `--interactive=false`, `--dataset.push_to_hub=false`, and `--inference.type=rtc` (no `--teleop`, no sync fallback) and loads adapters via `scripts/policy_loading.py`. View episodes with `lerobot-dataset-viz`:

```sh
pixi run lerobot-dataset-viz --repo-id local/sim_rollout_eval --root ./data/local/sim_rollout_eval --episode-index 0
```

## Hardware boundary and source map

Physical drivers and guards (`home_joints_deg`, `measured_joints()`, `go_home()`, `ROLLOUT_START_TOLERANCE_DEG`, SpaceMouse workspace/tilt guards) are unchanged. Simulation shares the single Pixi environment, the training launcher (GPU-admission allowlist, cgroup/CUDA telemetry, `TMPDIR` socket-length checks), and plugin registration (`lerobot_robot_ur5e/__init__.py` imports `sim/config_sim.py` and `sim/__init__.py`, loading `gymnasium` and registering the Gym ID lazily without importing `mujoco` or `sim/env.py`).

- **Tasks & lock:** [pixi.toml](../pixi.toml), [pixi.lock](../pixi.lock)
- **Sim & assets:** `src/lerobot_robot_ur5e/lerobot_robot_ur5e/sim/` (`config_sim.py`, `scene.py`, `env.py`, `teacher.py`), [PROVENANCE.md](../src/lerobot_robot_ur5e/lerobot_robot_ur5e/sim/assets/PROVENANCE.md)
- **Rollout & loader:** [sim_rollout.py](../scripts/sim_rollout.py), [policy_loading.py](../scripts/policy_loading.py)
- **Training:** [training.md](training.md), [train_lora.py](../scripts/train_lora.py), [train_lora_entry.py](../scripts/train_lora_entry.py), `config/training/*.recipe.json`
- **Hardware & docs:** [operating-guide.md](operating-guide.md), [GLOSSARY.md](../GLOSSARY.md), `src/lerobot_robot_ur5e/lerobot_robot_ur5e/ur5e.py`, `config_ur5e.py`, [record.py](../scripts/record.py), [rollout.py](../scripts/rollout.py)
- **LeRobot 0.7.0:** `lerobot.robots.sim_robot`, `lerobot.sim.integrations.gym`, `lerobot.scripts.lerobot_record`, `lerobot.rollout.strategies.episodic`
