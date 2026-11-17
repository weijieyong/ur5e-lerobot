# ur5e-lerobot

UR5e pick-and-place policies on LeRobot, currently with pi0.5.

UR5e with a Robotiq 2F-85 or OnRobot RG2 gripper, RealSense D455 (front) and D405 (wrist)
cameras and a 3Dconnexion SpaceMouse. LeRobot 0.7.0, locked with Pixi: setup/calibration → SpaceMouse collection →
fresh-base LoRA training with automatic train-only stats → supervised RTC rollout.
Data, checkpoints and machine configuration stay local; Hub uploads are disabled.

1. Follow the [operating guide](docs/operating-guide.md) for prerequisites, measured
   configuration, camera checks, teleoperation and collection. Motion commands require
   an operator, clear workspace and pendant stop access.
2. Follow [training and deployment](docs/training.md) to train a finalized dataset with the reference schedule and copy complete assets for rollout.
3. Return to the operating guide for supervised rollout and outcome recording.

Operation requires measured IP/device/home/TCP/floor/neutral configuration;
no example pose is a safe default. Offline tests establish software behavior only.
Clean installation, GPU training/save/strict reload and supervised physical
qualification remain separate acceptance gates for each deployment.

`pixi run lint` and `pixi run test` are offline checks; neither touches hardware.

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).
Models, tokenizers and datasets are not covered: pi0.5 and PaliGemma weights carry
their own licenses, so review their upstream terms before redistributing them.
