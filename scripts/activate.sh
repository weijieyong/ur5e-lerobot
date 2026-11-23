# Sourced by `pixi run` / `pixi shell` (see [activation] in pixi.toml).
# Machine-specific settings live in config/local/machine.env (git-ignored);
# copy config/machine.env.example there and edit it.
if [ -f "$PIXI_PROJECT_ROOT/config/local/machine.env" ]; then
  set -a; . "$PIXI_PROJECT_ROOT/config/local/machine.env"; set +a
fi
export ROLLOUT_ENFORCE_WORKSPACE="${ROLLOUT_ENFORCE_WORKSPACE:-true}"

# Keep this pinned Python/native stack independent of sourced ROS, Conda or MuJoCo.
# Preserve desktop/session variables (DISPLAY, XAUTHORITY, etc.) for hardware UI.
# Pixi merges activation changes into the parent environment: unset alone does
# not remove inherited entries, so export empty overrides instead.
export PYTHONPATH= PYTHONHOME= LD_LIBRARY_PATH=
export PYTHONNOUSERSITE=1
