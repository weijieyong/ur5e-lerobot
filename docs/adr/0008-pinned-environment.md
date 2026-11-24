# 0008: The Pixi environment is isolated from the host

Status: accepted (2026-11-24)

## Decision

- `pyrealsense2` comes from the PyPI wheel, not conda-forge. The conda package ships no dist metadata,
  so LeRobot reports it missing.
- `scripts/activate.sh` keeps the pinned Python/native stack independent of any sourced ROS, Conda or
  MuJoCo environment, while preserving desktop/session variables (`DISPLAY`, `XAUTHORITY`, ...) for
  hardware UI. Pixi merges activation changes into the parent environment, so `unset` alone does not
  remove inherited entries; the script exports empty overrides instead.
- `docker/smoke/` copies only source inputs into the image, never host environments, credentials,
  models or data. Each `docker run` starts without a project environment or machine configuration.
- `scripts/test_clean_env.sh` passes `-t` to Docker only when stdout is a terminal, so Pixi can draw
  progress interactively without corrupting piped or logged output.
