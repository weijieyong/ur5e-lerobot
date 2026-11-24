"""Synthetic native recorder-shaped VIDEO -> automatic stats -> saved CPU processors."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from scripts import train_lora as tl
from scripts.train_lora_entry import fit_training_datasets


def recorder_features(use_video, height=16, width=16):
    """Use actual robot properties + native recorder conversion, without robot construction/I/O."""
    from types import SimpleNamespace

    from lerobot_robot_ur5e.ur5e import UR5e
    from lerobot.utils.feature_utils import hw_to_dataset_features

    robot = object.__new__(UR5e)
    robot.config = SimpleNamespace(
        cameras={
            key: SimpleNamespace(height=height, width=width)
            for key in ("front", "wrist")
        }
    )
    return {
        **hw_to_dataset_features(robot.action_features, "action", use_video=use_video),
        **hw_to_dataset_features(
            robot.observation_features, "observation", use_video=use_video
        ),
    }


class TrainingStatsTests(unittest.TestCase):
    def setUp(self):
        from lerobot.configs import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw-video"
        self.features = recorder_features(use_video=True, height=32, width=48)
        writer = LeRobotDataset.create(
            "local/synthetic_video",
            root=self.raw,
            fps=15,
            features=self.features,
            robot_type="SYNTHETIC",
            use_videos=True,
            video_backend="pyav",
            rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=18),
            encoder_threads=1,
            image_writer_threads=1,
        )
        for ep in range(3):
            for t in range(52):
                writer.add_frame(
                    {
                        "observation.state": np.asarray(
                            [(1000000 if ep == 2 else ep * 100) + t] * 6 + [t % 2],
                            dtype=np.float32,
                        ),
                        "action": np.asarray(
                            [(1000000 if ep == 2 else ep * 100) + t + 2] * 6 + [t % 2],
                            dtype=np.float32,
                        ),
                        "observation.images.front": np.full(
                            (32, 48, 3), 40 + ep * 40, dtype=np.uint8
                        ),
                        "observation.images.wrist": np.full(
                            (32, 48, 3), 180 - ep * 20, dtype=np.uint8
                        ),
                        "task": "synthetic video task",
                    }
                )
            writer.save_episode(parallel_encoding=False)
        writer.finalize()

    def config(self):
        preset = tl.bind_dataset(
            tl.load_preset("training-2gpu"), "raw-video", "local/synthetic_video"
        )
        cfg = tl.validate_native(
            tl.build_native_argv(preset, self.root, self.root / "native")
        )
        cfg.dataset.video_backend = "pyav"
        return cfg

    def test_native_video_split_fit_and_source_preservation(self):
        original = {p: tl.sha256(p) for p in self.raw.rglob("*") if p.is_file()}
        train, dev = fit_training_datasets(self.config())
        self.assertEqual(train.episodes, [0, 1])
        self.assertEqual(dev.episodes, [2])
        self.assertEqual(
            tuple(train.meta.features["observation.images.front"]["shape"]), (32, 48, 3)
        )
        self.assertNotIn("observation.images.front", train.hf_dataset.column_names)
        self.assertEqual(tuple(train[0]["observation.images.front"].shape), (3, 32, 48))
        self.assertAlmostEqual(
            float(train[0]["observation.images.front"].float().mean()), 40, delta=3
        )
        self.assertAlmostEqual(
            float(dev[0]["observation.images.wrist"].float().mean()), 140, delta=3
        )
        self.assertTrue(train[51]["action_is_pad"].any())
        stats = train.meta.stats
        self.assertEqual(stats["observation.state"]["count"].tolist(), [104])
        self.assertEqual(stats["action"]["count"].tolist(), [300])
        states = np.asarray(
            [[ep * 100 + t] * 6 + [t % 2] for ep in range(2) for t in range(52)]
        )
        for stat, fn in (
            ("min", np.min),
            ("max", np.max),
            ("mean", np.mean),
            ("std", np.std),
        ):
            np.testing.assert_allclose(
                stats["observation.state"][stat], fn(states, axis=0), atol=1e-7
            )
        from lerobot.utils.constants import IMAGENET_STATS

        for key in train.meta.camera_keys:
            for field, value in IMAGENET_STATS.items():
                np.testing.assert_allclose(stats[key][field], value)

        # Independent complete-window enumeration: no cross-episode or padded starts.
        values = np.asarray(
            [
                [offset + 2] * 6 + [(start + offset) % 2]
                for ep in range(2)
                for start in range(3)
                for offset in range(50)
            ]
        )
        for stat, fn in (
            ("min", np.min),
            ("max", np.max),
            ("mean", np.mean),
            ("std", np.std),
        ):
            np.testing.assert_allclose(
                stats["action"][stat], fn(values, axis=0), atol=1e-7
            )
        for q in (1, 99):
            np.testing.assert_allclose(
                stats["action"][f"q{q:02d}"],
                np.quantile(values, q / 100, axis=0, method="inverted_cdf"),
                atol=0.011,
            )
        self.assertIn("observation.images.front", stats)
        self.assertGreater(dev.meta.stats["observation.state"]["max"][0], 1e6)
        for path, digest in original.items():
            self.assertEqual(tl.sha256(path), digest)

    def test_config_drives_chunk_and_mask_and_rejects_invalid_arrays(self):
        from datasets import Dataset
        from types import SimpleNamespace
        from lerobot.datasets.factory import make_train_eval_datasets

        cfg = self.config()
        cfg.policy.chunk_size = cfg.policy.n_action_steps = 10
        train, dev = fit_training_datasets(cfg)
        self.assertEqual(int(train.meta.stats["action"]["count"][0]), 860)
        cfg.policy.relative_exclude_joints = []
        with self.assertRaisesRegex(tl.TrainingConfigError, "gripper absolute"):
            fit_training_datasets(cfg)
        cfg.policy.relative_exclude_joints = ["gripper"]
        train, dev = make_train_eval_datasets(cfg)
        original = train.hf_dataset.select_columns(
            ["observation.state", "action", "episode_index"]
        ).with_format("numpy")[:]
        for kind in ("nonfinite", "shape", "short"):
            rows = {k: np.array(v, copy=True) for k, v in original.items()}
            if kind == "nonfinite":
                rows["action"][0, 0] = np.nan
            elif kind == "shape":
                rows["action"] = rows["action"][:, :6]
            else:
                rows = {k: v[:3] for k, v in rows.items()}
            bad_train = SimpleNamespace(
                hf_dataset=Dataset.from_dict(rows),
                meta=train.meta,
                episodes=train.episodes,
            )
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(RuntimeError, "nonfinite|invalid|usable"),
            ):
                fit_training_datasets(cfg, lambda _: (bad_train, dev))
        train.meta.features["action"]["names"] = list(
            reversed(train.meta.features["action"]["names"])
        )
        with self.assertRaisesRegex(tl.TrainingConfigError, "matching six joints"):
            fit_training_datasets(cfg, lambda _: (train, dev))
        for mode in ("resume", "streaming"):
            cfg = self.config()
            if mode == "resume":
                cfg.resume = True
            else:
                cfg.dataset.streaming = True
            with self.assertRaisesRegex(tl.TrainingConfigError, "unsupported"):
                fit_training_datasets(cfg)

    def test_native_pi05_processors_save_reload_and_roundtrip(self):
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        from lerobot.configs import FeatureType
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        from lerobot.processor.factory import load_pretrained_policy_processors
        from lerobot.processor.rename_processor import rename_stats
        from lerobot.utils.feature_utils import dataset_to_policy_features

        cfg = self.config()
        train, _ = fit_training_datasets(cfg)
        tokenizer = self.root / "tiny-tokenizer"
        PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                models.WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")
            ),
            unk_token="[UNK]",
            pad_token="[PAD]",
        ).save_pretrained(tokenizer)
        policy = cfg.policy
        policy.device = "cpu"
        policy.text_tokenizer_name = str(tokenizer)
        features = dataset_to_policy_features(train.meta.features)
        policy.input_features = {
            k: v for k, v in features.items() if v.type != FeatureType.ACTION
        }
        policy.output_features = {
            k: v for k, v in features.items() if v.type == FeatureType.ACTION
        }
        policy.action_feature_names = train.meta.features["action"]["names"]
        pre, post = make_pi05_pre_post_processors(
            policy, rename_stats(train.meta.stats, cfg.rename_map)
        )
        state = torch.tensor([10.0] * 6 + [0.4])
        action = torch.tensor(
            [[offset + 12.0] * 6 + [offset % 2] for offset in range(50)]
        )
        sample = {
            "observation.state": state,
            "action": action,
            "task": "synthetic video task",
        }
        normalized = pre(sample)["action"]
        relative = action.clone()
        relative[:, :6] -= state[:6]
        q01 = torch.tensor(train.meta.stats["action"]["q01"], dtype=torch.float32)
        q99 = torch.tensor(train.meta.stats["action"]["q99"], dtype=torch.float32)
        expected = 2 * (relative - q01) / (q99 - q01) - 1
        torch.testing.assert_close(normalized.squeeze(0), expected)

        torch.testing.assert_close(post(normalized), action, atol=1e-5, rtol=1e-5)
        saved = self.root / "processors"
        pre.save_pretrained(saved)
        post.save_pretrained(saved)
        self.assertTrue(list(saved.glob("*.safetensors")))
        restored_pre, restored_post = load_pretrained_policy_processors(saved)
        result = restored_pre(sample)["action"]
        torch.testing.assert_close(result, normalized)
        torch.testing.assert_close(restored_post(result), action, atol=1e-5, rtol=1e-5)
