"""Exercise strict base and adapter loading with real tiny Torch/PEFT weights."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file

from scripts.policy_loading import load_local_pi05_adapter, load_rollout_policy


class TinyPolicy(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {"model_type": "custom"}
        self.layer = torch.nn.Linear(2, 2)

    def _convert_openpi_state_dict(self, weights):
        return weights


class PolicyLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.base = root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text('{"model_type": "custom"}')
        self.adapter = root / "adapter"
        policy = TinyPolicy()
        policy.name_or_path = str(self.base)
        self.base_weights = {k: v.clone() for k, v in policy.state_dict().items()}
        save_file(self.base_weights, self.base / "model.safetensors")
        peft_config = LoraConfig(
            r=1,
            lora_alpha=1,
            target_modules=["layer"],
            base_model_name_or_path=str(self.base),
        )
        model = get_peft_model(policy, peft_config)
        with torch.no_grad():
            model.base_model.model.layer.lora_B.default.weight.fill_(0.25)
        self.adapter_weights = get_peft_model_state_dict(model)
        model.save_pretrained(self.adapter)
        self.config = SimpleNamespace(
            type="pi05", use_peft=True, pretrained_path=str(self.adapter)
        )
        self.policy_patch = patch("scripts.policy_loading.PI05Policy", TinyPolicy)
        self.policy_patch.start()
        self.addCleanup(self.policy_patch.stop)

    def test_actual_base_and_adapter_tensors_are_loaded(self):
        model = load_local_pi05_adapter(self.config)
        layer = model.base_model.model.layer.base_layer
        torch.testing.assert_close(
            layer.weight, self.base_weights["layer.weight"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer.bias, self.base_weights["layer.bias"], rtol=0, atol=0
        )
        for key, value in get_peft_model_state_dict(model).items():
            torch.testing.assert_close(value, self.adapter_weights[key], rtol=0, atol=0)
        self.assertFalse(model.training)

    def test_missing_base_is_rejected_before_model_construction(self):
        (self.base / "model.safetensors").unlink()
        with patch("scripts.policy_loading.PI05Policy") as constructor:
            with self.assertRaisesRegex(FileNotFoundError, "base weights are missing"):
                load_local_pi05_adapter(self.config)
            constructor.assert_not_called()

    def test_incomplete_base_cannot_return_random_model(self):
        save_file(
            {"layer.weight": self.base_weights["layer.weight"]},
            self.base / "model.safetensors",
        )
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            load_local_pi05_adapter(self.config)

    def test_corrupt_base_is_rejected(self):
        (self.base / "model.safetensors").write_text("incomplete transfer")
        with self.assertRaises(Exception):
            load_local_pi05_adapter(self.config)

    def test_incomplete_adapter_is_rejected(self):
        weights = load_file(self.adapter / "adapter_model.safetensors")
        weights.pop(next(iter(weights)))
        save_file(weights, self.adapter / "adapter_model.safetensors")
        with self.assertRaisesRegex(RuntimeError, "adapter state mismatch"):
            load_local_pi05_adapter(self.config)

    def test_unexpected_adapter_key_is_rejected(self):
        weights = load_file(self.adapter / "adapter_model.safetensors")
        weights["unrecognized.weight"] = torch.ones(1)
        save_file(weights, self.adapter / "adapter_model.safetensors")
        with self.assertRaisesRegex(RuntimeError, "adapter state mismatch"):
            load_local_pi05_adapter(self.config)

    def test_missing_adapter_is_rejected(self):
        (self.adapter / "adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "adapter weights are missing"):
            load_local_pi05_adapter(self.config)

    def test_other_policy_loading_is_preserved(self):
        fallback = Mock(return_value=object())
        self.config.type = "other"
        self.assertIs(load_rollout_policy(self.config, fallback), fallback.return_value)
        fallback.assert_called_once_with(self.config)

    def test_missing_local_checkpoint_never_reaches_upstream_loader(self):
        self.config.pretrained_path = str(
            Path(self.directory.name) / "missing-checkpoint"
        )
        fallback = Mock()
        with self.assertRaisesRegex(
            FileNotFoundError, "checkpoint directory is missing"
        ):
            load_rollout_policy(self.config, fallback)
        fallback.assert_not_called()

    def test_missing_relative_checkpoint_never_reaches_upstream(self):
        fallback = Mock()
        for path in (
            "outputs/missing/pretrained_model",
            "./outputs/missing",
            "outputs/missing",
        ):
            with self.subTest(path=path):
                self.config.pretrained_path = path
                with self.assertRaisesRegex(
                    FileNotFoundError, "checkpoint directory is missing"
                ):
                    load_rollout_policy(self.config, fallback)
        fallback.assert_not_called()

    def test_valid_relative_checkpoint_loads_real_tiny_weights(self):
        self.config.pretrained_path = os.path.relpath(self.adapter)
        fallback = Mock()
        model = load_rollout_policy(self.config, fallback)
        torch.testing.assert_close(
            model.base_model.model.layer.base_layer.weight,
            self.base_weights["layer.weight"],
            rtol=0,
            atol=0,
        )
        fallback.assert_not_called()

    def test_missing_local_adapter_config_never_reaches_upstream_loader(self):
        (self.adapter / "adapter_config.json").unlink()
        fallback = Mock()
        with self.assertRaisesRegex(
            FileNotFoundError, "adapter configuration is missing"
        ):
            load_rollout_policy(self.config, fallback)
        fallback.assert_not_called()

    def test_remote_identifier_preserves_upstream_loading(self):
        for identifier in ("organization/remote-policy", "remote-policy"):
            self.config.pretrained_path = identifier
            fallback = Mock(return_value=object())
            self.assertIs(
                load_rollout_policy(self.config, fallback), fallback.return_value
            )
            fallback.assert_called_once_with(self.config)

    def test_missing_path_objects_are_local_even_without_dot_prefix(self):
        for value in (Path("./namespace/missing"), Path("missing")):
            with self.subTest(value=value):
                self.config.pretrained_path = value
                fallback = Mock()
                with self.assertRaisesRegex(
                    FileNotFoundError, "checkpoint directory is missing"
                ):
                    load_rollout_policy(self.config, fallback)
                fallback.assert_not_called()

    def test_valid_path_object_loads_local_weights(self):
        self.config.pretrained_path = self.adapter
        fallback = Mock()
        load_rollout_policy(self.config, fallback)
        fallback.assert_not_called()

    def test_rollout_uses_strict_loader_and_restores_it_after_failure(self):
        from lerobot.inference import RTCInferenceConfig
        from scripts import rollout

        self.config.robot = SimpleNamespace(
            max_joint_step_rad=0.05, home_joints_deg=[0.0] * 6, z_min_m=0.1
        )
        self.config.teleop = None
        self.config.interactive = False
        self.config.strategy = SimpleNamespace(type="episodic")
        self.config.inference = RTCInferenceConfig()
        self.config.fps = 15
        original_loader = rollout.rollout_context._load_pretrained_policy
        (self.base / "model.safetensors").unlink()

        def build_context(config):
            return rollout.rollout_context._load_pretrained_policy(config)

        with (
            patch.object(rollout, "preflight"),
            patch.object(rollout, "RecordingFeedback"),
            patch.object(
                rollout.upstream.rollout, "__wrapped__", side_effect=build_context
            ),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "base weights are missing"):
                rollout.run_rollout(self.config)
        self.assertIs(rollout.rollout_context._load_pretrained_policy, original_loader)

    def test_rollout_binds_relative_anchor_without_a_policy_reference_and_restores_it(
        self,
    ):
        from lerobot.inference import RTCInferenceConfig
        from scripts import rollout

        self.config.robot = SimpleNamespace(
            max_joint_step_rad=0.05, home_joints_deg=[0.0] * 6, z_min_m=0.1
        )
        self.config.teleop = None
        self.config.interactive = False
        self.config.strategy = SimpleNamespace(type="episodic")
        self.config.inference = RTCInferenceConfig()
        self.config.fps = 15
        original_bind = rollout.rollout_context.bind_relative_anchor
        seen = []

        def build_context(config):
            seen.append(rollout.rollout_context.bind_relative_anchor)
            raise RuntimeError("stop before hardware")

        with (
            patch.object(rollout, "preflight"),
            patch.object(rollout, "RecordingFeedback"),
            patch.object(
                rollout.upstream.rollout, "__wrapped__", side_effect=build_context
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before hardware"):
                rollout.run_rollout(self.config)
        self.assertIs(
            seen[0], rollout.bind_relative_anchor
        )  # installed while the context builds
        self.assertIs(rollout.rollout_context.bind_relative_anchor, original_bind)


if __name__ == "__main__":
    unittest.main()
