"""Onboarding commands against synthetic local datasets; no model or hardware."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class OnboardingTests(unittest.TestCase):
    def test_inspection_tasks_route_ids_and_nondefault_roots(self):
        tasks = tomllib.loads((ROOT / "pixi.toml").read_text())["tasks"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.py"
            capture.write_text("import json,sys; print(json.dumps(sys.argv[1:]))")
            env = dict(
                os.environ,
                DATASET_REPO_ID="local/demonstrations",
                ROLLOUT_DATASET_REPO_ID="local/rollout_session",
                HF_LEROBOT_HOME=str(root / "custom data"),
            )
            for task, expected in (
                ("info", "local/demonstrations"),
                ("view", "local/demonstrations"),
                ("rollout-info", "local/rollout_session"),
                ("rollout-view", "local/rollout_session"),
            ):
                command = tasks[task]["cmd"].replace("{{ episode }}", "2")
                command = (
                    shlex.quote(sys.executable)
                    + " "
                    + shlex.quote(str(capture))
                    + " "
                    + command.split(" ", 1)[1]
                )
                result = subprocess.run(
                    command,
                    shell=True,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                args = json.loads(result.stdout)
                self.assertTrue(expected in args or f"--repo_id={expected}" in args)
                destination = str(root / "custom data" / expected)
                self.assertTrue(destination in args or f"--root={destination}" in args)

    def test_native_filter_copy_preserves_source_and_metadata(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.scripts.lerobot_edit_dataset import (
            DeleteEpisodesConfig,
            EditDatasetConfig,
            handle_delete_episodes,
        )

        with (
            TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}),
            patch("socket.socket.connect", side_effect=AssertionError("network")),
        ):
            root = Path(tmp)
            source, dest = root / "raw", root / "filtered"
            import datasets.config

            cache = patch.object(datasets.config, "HF_DATASETS_CACHE", root / "cache")
            cache.start()
            self.addCleanup(cache.stop)
            writer = LeRobotDataset.create(
                "local/raw",
                root=source,
                fps=15,
                robot_type="SYNTHETIC",
                features={
                    key: {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["a", "b", "c", "d", "e", "f", "gripper.pos"],
                    }
                    for key in ("observation.state", "action")
                },
                use_videos=False,
            )
            for episode in range(3):
                for frame in range(2):
                    writer.add_frame(
                        {
                            "observation.state": np.full(7, episode, dtype=np.float32),
                            "action": np.full(7, episode + 1, dtype=np.float32),
                            "task": "synthetic task",
                        }
                    )
                writer.save_episode()
            writer.finalize()

            def hashes():
                return {
                    p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.rglob("*")
                    if p.is_file()
                }

            before = hashes()
            handle_delete_episodes(
                EditDatasetConfig(
                    repo_id="local/raw",
                    root=str(source),
                    new_repo_id="local/filtered",
                    new_root=str(dest),
                    push_to_hub=False,
                    operation=DeleteEpisodesConfig(episode_indices=[1]),
                )
            )
            self.assertEqual(hashes(), before)
            filtered = LeRobotDataset("local/filtered", root=dest)
            self.assertEqual(filtered.meta.total_episodes, 2)
            self.assertEqual(filtered.meta.fps, 15)
            torch.testing.assert_close(
                filtered[2]["observation.state"], torch.full((7,), 2.0)
            )

if __name__ == "__main__":
    unittest.main()
