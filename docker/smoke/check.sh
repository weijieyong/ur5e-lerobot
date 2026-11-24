#!/usr/bin/env bash
set -euo pipefail

step() { printf '\n==> %s\n' "$*"; }

step "1/7 Check clean container"
test ! -e .pixi
test ! -e config/local
sha256sum pixi.toml pixi.lock > /tmp/pai-inputs.sha256
pixi --version
nvidia-smi
step "2/7 Install locked environment (downloads several GB; can take many minutes)"
pixi install --locked
step "3/7 Create example machine config"
pixi run --locked setup

step "4/7 CUDA float32/bfloat16 check"
pixi run --locked python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable inside the container"
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}", flush=True)
torch.set_float32_matmul_precision("highest")
x = torch.arange(1024, dtype=torch.float32).reshape(32, 32) / 1024
expected = x @ x.T
gpu = x.cuda()
torch.testing.assert_close((gpu @ gpu.T).cpu(), expected)
assert torch.cuda.is_bf16_supported(), "pi0.5 requires bfloat16 support"
bf16 = gpu.to(torch.bfloat16)
torch.testing.assert_close((bf16 @ bf16.T).float().cpu(), expected, rtol=0.01, atol=0.01)
torch.cuda.synchronize()
print("CUDA float32 and bfloat16 matrix calculations passed", flush=True)
PY

step "5/7 Correctness lint"
pixi run --locked lint
step "6/7 Offline tests"
pixi run --locked test
step "7/7 Check pixi.toml and pixi.lock unchanged"
sha256sum --check /tmp/pai-inputs.sha256
echo "Clean-container Pixi/GPU smoke test passed"
