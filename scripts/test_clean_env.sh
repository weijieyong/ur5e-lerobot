#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image=pai-ws-pixi-smoke:local

docker build --file "$repo_root/docker/smoke/Dockerfile" --tag "$image" "$repo_root"
tty_flag=()
[ -t 1 ] && tty_flag=(-t)
exec docker run --rm "${tty_flag[@]}" --runtime=nvidia --gpus all "$image"
