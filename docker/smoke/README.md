# Clean Pixi/GPU smoke test

From the repository root, with Docker Engine, an NVIDIA driver and NVIDIA
Container Toolkit already configured on the host:

```sh
pixi run smoke-docker
```

The task wraps `./scripts/test_clean_env.sh`, which builds the image and runs the
check; no Compose setup is needed. The script also runs without Pixi, including by
absolute path from another directory. Docker must
be accessible to your user, and the NVIDIA runtime must already be configured.

The image contains Ubuntu 24.04, CA certificates, curl, Pixi 0.83.0 and the
repository's source inputs. Each run downloads and installs the committed
lockfile into a fresh container, creates example machine configuration, checks
CUDA float32/bfloat16 calculations and runs `pixi run --locked lint` with the
repository's `ruff.toml`, then `pixi run --locked test`. It also checks that neither
the manifest nor lockfile changed. A successful run ends
with `Clean-container Pixi/GPU smoke test passed` and exit code 0. Each step prints
a `==> n/7` line; step 2 (the locked install) is the long one.

There are no host directory/cache mounts, USB devices or desktop display
connections. Private config, datasets, models and credentials are excluded
from the build context. Internet access and sufficient Docker disk space are
needed for each fresh installation; package downloads are intentionally uncached
between runs. The wrapper rebuilds the image with the current source on every run.

This checks clean Linux userspace installation and GPU computation using the
host's existing kernel/driver. It does not provision those host prerequisites,
load a policy, train a model, or validate physical hardware or GUI operation.

Tested with LeRobot 0.7.0 in the Ubuntu 24.04 container on an RTX 5080.

Reference: [NVIDIA Container Toolkit sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html).
