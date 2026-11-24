"""CPU-only regression tests for optional LoRA adapter warm-start initialization."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from scripts import train_lora as tl
from scripts.train_lora_entry import (
    load_init_adapter,
    prepare_optimizer_and_scheduler,
)


class _Block(torch.nn.Module):
    def __init__(self, *names, dim=8):
        super().__init__()
        for name in names:
            setattr(self, name, torch.nn.Linear(dim, dim, bias=False))


class _TinyPolicy(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.paligemma = _Block("proj", dim=dim)
        self.gemma_expert = _Block("q_proj", "v_proj", "k_proj", dim=dim)


def _build_peft_policy(
    *,
    r=4,
    lora_alpha=8,
    target_modules=("q_proj", "v_proj"),
    use_rslora=False,
    init_lora_weights=True,
    seed=0,
):
    torch.manual_seed(seed)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        use_rslora=use_rslora,
        init_lora_weights=init_lora_weights,
    )
    return get_peft_model(_TinyPolicy(), config)


def _fill_distinct_lora(model, offset=1.0):
    with torch.no_grad():
        for idx, (_, param) in enumerate(model.named_parameters()):
            if param.requires_grad:
                values = (
                    torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape)
                    * 0.01
                    + offset
                    + idx
                )
                param.copy_(values)


def _save_adapter(adapter_dir, *, offset=1.0, fill=True, **model_kwargs):
    source = _build_peft_policy(**model_kwargs)
    if fill:
        _fill_distinct_lora(source, offset=offset)
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    source.save_pretrained(str(adapter_dir))
    return source


class AdapterInitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.guards = {"trainable_lora_tensors": 4}

    def _make_cfg(self):
        scheduler_cfg = CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=1000,
            num_decay_steps=30000,
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        )
        return SimpleNamespace(scheduler=scheduler_cfg, steps=15000)

    def _make_optimizer(self, cfg, policy):
        params = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.scheduler.peak_lr)
        return optimizer, cfg.scheduler.build(optimizer, cfg.steps)

    def test_valid_adapter_restores_lora_preserves_frozen_base_and_portable_path(self):
        rel = "runs/prev/pretrained_model"
        adapter_dir = self.root / rel
        source = _save_adapter(adapter_dir, offset=2.0, seed=10)
        cfg_path = adapter_dir / "adapter_config.json"
        cfg_data = json.loads(cfg_path.read_text())
        cfg_data["base_model_name_or_path"] = "/other-host/hf/hub/pi05_base"
        cfg_path.write_text(json.dumps(cfg_data))

        target = _build_peft_policy(seed=20)
        _fill_distinct_lora(target, offset=50.0)
        base_before = {
            n: p.detach().clone()
            for n, p in target.named_parameters()
            if not p.requires_grad
        }
        source_lora = {
            n: p.detach().clone()
            for n, p in source.named_parameters()
            if p.requires_grad
        }
        receipt = {}
        optimizer, _ = prepare_optimizer_and_scheduler(
            self._make_cfg(),
            target,
            {**self.guards, "init_adapter": rel},
            receipt,
            self._make_optimizer,
            root=self.root,
        )

        target_lora = {n: p for n, p in target.named_parameters() if p.requires_grad}
        self.assertEqual(set(target_lora), set(source_lora))
        for name, param in target_lora.items():
            self.assertTrue(torch.equal(param.detach(), source_lora[name]), name)
        target_base = {
            n: p for n, p in target.named_parameters() if not p.requires_grad
        }
        self.assertEqual(set(target_base), set(base_before))
        for name, param in target_base.items():
            self.assertTrue(torch.equal(param.detach(), base_before[name]), name)
        self.assertEqual(receipt["init_adapter"], rel)
        self.assertEqual(receipt["trainable_lora_tensors"], 4)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.state, {})
        opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        self.assertEqual(opt_ids, {id(p) for p in target_lora.values()})

    def test_without_init_adapter_keeps_fresh_lora_and_omits_receipt(self):
        target = _build_peft_policy(seed=30)
        before = {n: p.detach().clone() for n, p in target.named_parameters()}
        receipt = {}
        optimizer, _ = prepare_optimizer_and_scheduler(
            self._make_cfg(),
            target,
            self.guards,
            receipt,
            self._make_optimizer,
            root=self.root,
        )
        self.assertNotIn("init_adapter", receipt)
        self.assertEqual(receipt["trainable_lora_tensors"], 4)
        self.assertEqual(optimizer.state, {})
        for name, param in target.named_parameters():
            self.assertTrue(torch.equal(param.detach(), before[name]), name)

    def test_rejects_key_mismatch(self):
        with self.subTest(case="smuggled_base_weight"):
            adapter_dir = self.root / "smuggled"
            _save_adapter(adapter_dir, offset=5.0, seed=4)
            weights_path = adapter_dir / "adapter_model.safetensors"
            tensors = load_file(str(weights_path))
            del tensors[sorted(tensors)[-1]]
            tensors["base_model.model.paligemma.proj.weight"] = torch.full((8, 8), 99.0)
            save_file(tensors, str(weights_path))

            target = _build_peft_policy(seed=99)
            _fill_distinct_lora(target, offset=77.0)
            before = {n: p.detach().clone() for n, p in target.named_parameters()}
            with self.assertRaises(tl.TrainingConfigError):
                load_init_adapter(target, adapter_dir, self.guards)
            for name, param in target.named_parameters():
                self.assertTrue(torch.equal(param.detach(), before[name]), name)

        with self.subTest(case="partial_adapter"):
            adapter_dir = self.root / "partial"
            _save_adapter(adapter_dir, offset=8.0, seed=7)
            weights_path = adapter_dir / "adapter_model.safetensors"
            tensors = load_file(str(weights_path))
            del tensors[sorted(tensors)[-1]]
            save_file(tensors, str(weights_path))
            with self.assertRaises(tl.TrainingConfigError):
                load_init_adapter(_build_peft_policy(seed=99), adapter_dir, self.guards)

        with self.subTest(case="mixed_vlm_and_expert"):
            adapter_dir = self.root / "mixed"
            _save_adapter(
                adapter_dir, offset=7.0, seed=6, target_modules=("proj", "q_proj")
            )
            with self.assertRaises(tl.TrainingConfigError):
                load_init_adapter(_build_peft_policy(seed=99), adapter_dir, self.guards)

    def test_rejects_scaling_mismatch(self):
        for label, kwargs in (
            ("alpha_mismatch", {"lora_alpha": 16}),
            ("rslora_mismatch", {"use_rslora": True}),
        ):
            with self.subTest(case=label):
                adapter_dir = self.root / label
                _save_adapter(adapter_dir, offset=3.0, seed=2, **kwargs)
                with self.assertRaises(tl.TrainingConfigError):
                    load_init_adapter(
                        _build_peft_policy(seed=99), adapter_dir, self.guards
                    )

    def test_rejects_real_pissa_adapter(self):
        adapter_dir = self.root / "real_pissa"
        _save_adapter(adapter_dir, fill=False, seed=12, init_lora_weights="pissa")
        saved_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        self.assertEqual(saved_cfg["init_lora_weights"], "pissa")
        self.assertEqual(
            len(load_file(str(adapter_dir / "adapter_model.safetensors"))), 4
        )
        with self.assertRaises(tl.TrainingConfigError):
            load_init_adapter(_build_peft_policy(seed=99), adapter_dir, self.guards)


if __name__ == "__main__":
    unittest.main()
