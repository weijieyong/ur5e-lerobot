# AGENTS.md

UR5e + Robotiq 2F-85 or OnRobot RG2 pi0.5 pick-and-place on LeRobot, managed by Pixi.
See `pixi.toml` for declared dependencies and `pixi.lock` for resolved versions.

## Done

A change is done when `pixi run lint` and `pixi run test` both pass. Both are offline and touch no hardware.

## Commands drive real hardware

Every command is a Pixi task in `pixi.toml`. Each description states whether the task moves the robot, queries it read-only, or needs no hardware. Read the description before running a task; run one that moves the robot only when the user asks for that run.

## Where things live

- `README.md`: entry point; `docs/operating-guide.md` owns setup/live operations and `docs/training.md` owns automatic normalization, training and deployment.
- `docs/simulation-scope.md`: MuJoCo simulation guide (task, data/control contract, commands, limits); read before changing `sim/`, `scripts/sim_rollout.py` or the `sim-*` Pixi tasks.
- `GLOSSARY.md`: task, safety, data and machine terms; use these names in code and docs.
- `src/lerobot_robot_ur5e/`: the LeRobot plugin (UR5e robot, SpaceMouse teleop, cameras, MuJoCo sim).
- `scripts/`: entry points, run as `python -m scripts.<name>`. `teleop`, `rollout` and the tests import shared recording feedback from `scripts.record`.
- `config/local/machine.env`: private per-machine settings (git-ignored). `pixi run setup` merges defaults from `config/machine.env.example` and preserves existing values.

## Data stays local

Datasets and checkpoints stay on local machines: every LeRobot command runs with Hub push disabled.

## Run evidence

Write investigation and run evidence to `outputs/<topic>-<date>/` (git-ignored), never to tracked docs.
