# ur5e-lerobot

UR5e pick-and-place policies on LeRobot, currently with pi0.5.

UR5e with a Robotiq 2F-85 or OnRobot RG2 gripper, RealSense D455 or D435i (front, external) and D405
(wrist) cameras and a 3Dconnexion SpaceMouse. LeRobot 0.7.0, locked with Pixi: setup/calibration → SpaceMouse collection →
fresh-base LoRA training with automatic train-only stats → supervised RTC rollout.
Data, checkpoints and machine configuration stay local; Hub uploads are disabled.

## Prerequisites

**[Pixi](https://pixi.sh).** Pixi installs everything else (Python, PyTorch, CUDA libraries, LeRobot) from
`pixi.lock`; no system Python, Conda, ROS or CUDA toolkit is needed. Pixi 0.83.0 is the
tested version ([installation guide](https://pixi.sh/latest/installation/)):

```sh
curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=0.83.0 bash
pixi run --locked setup   # installs the environment and creates config/local/machine.env
pixi run --locked test    # offline; no hardware
```

Tested on Ubuntu 22.04 (robot PC and training machine: full workflow). `pixi run smoke-docker`
additionally checks a clean install and the offline tests in an Ubuntu 24.04 container.

**Robot PC (teleop, recording, inference).** x86-64 with an X11 desktop and an NVIDIA
driver supporting CUDA 12.9.

| | Recommended | Notes |
| --- | --- | --- |
| GPU | 24 GB VRAM (e.g. RTX 4090) | Rollout peaks at about 8.9 GiB, so a 10 GB RTX 3080 works with little headroom. |
| Disk | 50 GB+ free | Environment about 13 GB, plus model weights, datasets and outputs. |
| Hardware | UR5e in Remote Control, gripper URCap, D455 or D435i + D405 on USB 3, SpaceMouse | See the [operating guide](docs/operating-guide.md). |

**Training machine.** Can be the robot PC if it has the GPU memory.

| | Recommended | Notes |
| --- | --- | --- |
| GPU | Minimum 1× RTX 6000 Ada (48 GB); tested on 2× | The reference recipe trains batch 4 per GPU on two GPUs, or batch 8 on one. Run `pixi run train-check` before a long run. |
| Disk | 100 GB+ free | Base model, datasets and checkpoints. |

> [!IMPORTANT]
> Training needs the PaliGemma tokenizer, which is gated: use a Hugging Face account that has
> accepted the Gemma license. See [model downloads](docs/training.md#model-downloads).

Terms such as READY, floor, workspace and action chunk are defined in the [glossary](GLOSSARY.md).

## Workflow

> [!WARNING]
> These commands move a real robot. Run them only with an operator at the robot, a clear
> workspace and the pendant e-stop in reach. Example config values are not safe defaults:
> measure IP, devices, home, TCP, floor and neutral for your setup first.

1. Follow the [operating guide](docs/operating-guide.md) for prerequisites, measured
   configuration, camera checks, teleoperation and collection.
2. Follow [training and deployment](docs/training.md) to train a finalized dataset with the reference schedule and copy complete assets for rollout.
3. Return to the operating guide for supervised rollout and outcome recording.

Offline tests establish software behavior only.
Clean installation, GPU training/save/strict reload and supervised physical
qualification remain separate acceptance gates for each deployment.

`pixi run lint` and `pixi run test` are offline checks; neither touches hardware.

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).
Models, tokenizers and datasets are not covered: pi0.5 and PaliGemma weights carry
their own licenses, so review their upstream terms before redistributing them.
