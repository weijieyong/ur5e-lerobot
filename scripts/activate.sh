if [ -f "$PIXI_PROJECT_ROOT/config/local/machine.env" ]; then
  set -a; . "$PIXI_PROJECT_ROOT/config/local/machine.env"; set +a
fi

export PYTHONPATH= PYTHONHOME= LD_LIBRARY_PATH=
export PYTHONNOUSERSITE=1
