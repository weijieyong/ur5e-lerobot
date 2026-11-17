#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image=pai-ws-pixi-smoke:local

docker build --file "$repo_root/docker/smoke/Dockerfile" --tag "$image" "$repo_root"
# A terminal (-t) lets Pixi draw install progress; omit it when output is piped or logged.
tty_flag=()
[ -t 1 ] && tty_flag=(-t)
exec docker run --rm "${tty_flag[@]}" --runtime=nvidia --gpus all "$image"
