"""Offline checks of the supported LoRA training route (CPU only; no model, GPU or training)."""

import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from scripts import train_lora as tl
from scripts.train_lora_entry import load_spec

BASELINE = tl.bind_dataset(
    tl.load_preset("training-2gpu"), "data/synthetic", "local/synthetic"
)
# Mathematical schedule variants used by these tests.
BASELINE["native_flags"]["steps"] = 50000
LOWLR = copy.deepcopy(BASELINE)
LOWLR["native_flags"].update(
    {"policy.optimizer_lr": 2.5e-5, "policy.scheduler_decay_lr": 2.5e-6, "steps": 15000}
)


TWO = [{"index": 0, "uuid": "GPU-a"}, {"index": 1, "uuid": "GPU-b"}]
GPU0, GPU1 = [TWO[0]], [TWO[1]]
TOPOLOGY_FLAGS = {
    "batch_size",
    "parallelism.dp_replicate",
    "accelerator.gradient_accumulation.steps",
}


def mutated(preset, **changes):
    result = copy.deepcopy(preset)
    for dotted, value in changes.items():
        section, key = dotted.split("__")
        result[section][key.replace("_DOT_", ".")] = value
    return result


def fake_assets(root, preset):
    """Small on-disk stand-ins at the preset's relative paths; returns a matching preset."""
    flags, guards = preset["native_flags"], copy.deepcopy(preset["guards"])
    data = root / flags["dataset.root"]
    (data / "meta").mkdir(parents=True)
    (data / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 5,
                "total_frames": 260,
                "fps": 15,
            }
        )
    )
    base = root / flags["policy.pretrained_path"]
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"weights")
    guards.update(base_bytes=7, base_sha256=hashlib.sha256(b"weights").hexdigest())
    return {**copy.deepcopy(preset), "guards": guards}


class PresetToArgvTests(unittest.TestCase):
    def test_each_preset_value_reaches_native_argv_with_root_relative_paths(self):
        for preset in (BASELINE, LOWLR):
            with TemporaryDirectory() as tmp:
                argv = tl.build_native_argv(
                    preset, Path(tmp), Path(tmp) / "run" / "native"
                )
                flags = tl.parse_argv(argv)
                expected = dict(preset["native_flags"])
                for key in tl.PATH_FLAGS:
                    expected[key] = str(Path(tmp) / preset["native_flags"][key])
                expected["output_dir"] = str(Path(tmp) / "run" / "native")
                self.assertEqual(flags, expected)

    def test_smoke_overrides_only_budget_and_intervals(self):
        normal = tl.parse_argv(tl.build_native_argv(LOWLR, tl.ROOT, "/x/native"))
        smoke = tl.parse_argv(
            tl.build_native_argv(LOWLR, tl.ROOT, "/x/native", smoke=True)
        )
        self.assertEqual(
            {k for k in normal if normal[k] != smoke[k]}, set(tl.SMOKE_OVERRIDES)
        )
        self.assertEqual(smoke["policy.scheduler_decay_steps"], 30000)


class ScheduleTests(unittest.TestCase):
    def scheduler(self, flags):
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=flags["policy.scheduler_warmup_steps"],
            num_decay_steps=flags["policy.scheduler_decay_steps"],
            peak_lr=flags["policy.optimizer_lr"],
            decay_lr=flags["policy.scheduler_decay_lr"],
        )

    def lr(self, scheduler, step):
        return scheduler.base_lrs[0] * scheduler.lr_lambdas[0](step)

    def test_short_stop_budget_keeps_the_decay_horizon(self):
        flags = LOWLR["native_flags"]
        cfg = self.scheduler(flags)
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.zeros(1))], lr=cfg.peak_lr
        )
        native = cfg.build(
            optimizer, flags["steps"]
        )  # what plain lerobot-train would do
        self.assertAlmostEqual(
            self.lr(native, 15000), 2.5e-6
        )  # compressed: floor already at the stop
        kept = tl.build_scheduler(cfg, optimizer, flags["steps"])
        self.assertTrue(
            math.isclose(self.lr(kept, 15000), 2.5e-6 + 2.25e-5 / 2, rel_tol=1e-12)
        )
        self.assertTrue(math.isclose(self.lr(kept, 30000), 2.5e-6, rel_tol=1e-12))
        for step in (0, 500, 1000, 7000, 15000, 29999):
            self.assertTrue(
                math.isclose(
                    self.lr(kept, step), tl.expected_lr(flags, step), rel_tol=1e-12
                )
            )

    def test_rebuilding_after_the_native_factory_drives_the_optimizer_lr(self):
        # The worker receives the native (scaled) scheduler first, then rebuilds on the same optimizer.
        flags = LOWLR["native_flags"]
        cfg = self.scheduler(flags)
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.zeros(1))], lr=cfg.peak_lr
        )
        cfg.build(optimizer, flags["steps"])
        kept = tl.build_scheduler(cfg, optimizer, flags["steps"])
        self.assertEqual(kept.base_lrs, [2.5e-5])
        for step in range(1, 1201):
            optimizer.step()
            kept.step()
            if step in (1, 999, 1000, 1200):
                lr = optimizer.param_groups[0]["lr"]
                self.assertTrue(
                    math.isclose(lr, tl.expected_lr(flags, step), rel_tol=1e-9),
                    (step, lr),
                )

    def test_long_stop_budget_is_exactly_native(self):
        flags = BASELINE["native_flags"]
        cfg = self.scheduler(flags)
        samples = tl.schedule_samples(
            cfg, flags["steps"], [0, 1000, 15000, 30000, 50000]
        )
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.zeros(1))], lr=cfg.peak_lr
        )
        native = cfg.build(optimizer, flags["steps"])
        for step, value in samples.items():
            self.assertEqual(value, self.lr(native, step))
        self.assertTrue(
            math.isclose(samples[15000], 5.5e-5)
        )  # matches the accepted 15k checkpoint


class RecipeGuardTests(unittest.TestCase):
    def assertRejected(self, preset, text):
        with self.assertRaisesRegex(tl.TrainingConfigError, text):
            tl.check_recipe(preset)

    def test_unsafe_or_inconsistent_recipes_are_rejected(self):
        cases = [
            ({"native_flags__resume": True}, "resume"),
            ({"native_flags__policy_DOT_push_to_hub": True}, "push_to_hub"),
            ({"native_flags__save_checkpoint_to_hub": True}, "save_checkpoint_to_hub"),
            (
                {"native_flags__policy_DOT_train_expert_only": "false"},
                "train_expert_only",
            ),
            (
                {"native_flags__policy_DOT_train_expert_only": False},
                "target_modules",
            ),
            (
                {"native_flags__policy_DOT_relative_exclude_joints": []},
                "relative_exclude_joints",
            ),
            (
                {"native_flags__dataset_DOT_image_transforms_DOT_enable": "true"},
                "image_transforms",
            ),
            ({"native_flags__max_eval_samples": -1}, "max_eval_samples"),
            ({"native_flags__max_eval_samples": True}, "max_eval_samples"),
            ({"native_flags__max_eval_samples": 1.5}, "max_eval_samples"),
            ({"native_flags__batch_size": 2}, "batch topology"),
            (
                {"native_flags__accelerator_DOT_gradient_accumulation_DOT_steps": 2},
                "accumulation",
            ),
            (
                {
                    "native_flags__parallelism_DOT_dp_replicate": 4,
                    "native_flags__batch_size": 2,
                },
                "guards.gpus",
            ),
            ({"native_flags__policy_DOT_scheduler_decay_lr": 1e-4}, "floor"),
            ({"native_flags__policy_DOT_optimizer_lr": 0.01}, "peak"),
            ({"native_flags__policy_DOT_scheduler_warmup_steps": 30000}, "warmup"),
            ({"native_flags__steps": 500}, "stop budget"),
            ({"native_flags__eval_steps": -1}, "eval_steps"),
            ({"native_flags__save_freq": 0}, "save_freq"),
            ({"native_flags__save_freq": True}, "save_freq"),
            ({"native_flags__output_dir": "/tmp/x"}, "output_dir"),
            (
                {"native_flags__dataset_DOT_root": "/abs/elsewhere/dataset"},
                "project-relative",
            ),
            (
                {"native_flags__dataset_DOT_root": "../elsewhere/dataset"},
                "project-relative",
            ),
            (
                {
                    "native_flags__policy_DOT_pretrained_path": "outputs/some-run/checkpoints/015000/pretrained_model"
                },
                "original-base",
            ),
            ({"native_flags__dataset_DOT_eval_split": 0.0}, "eval_split"),
            ({"guards__global_batch": 16}, "batch topology"),
        ]
        for changes, text in cases:
            with self.subTest(changes=changes):
                self.assertRejected(mutated(LOWLR, **changes), text)

    def test_preset_names_and_files(self):
        with self.assertRaisesRegex(tl.TrainingConfigError, "invalid preset"):
            tl.load_preset("../test-recipe")
        with self.assertRaisesRegex(tl.TrainingConfigError, "available"):
            tl.load_preset("lora-missing")
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "other.recipe.json").write_text(
                json.dumps({**BASELINE, "schema": 2})
            )
            with self.assertRaisesRegex(tl.TrainingConfigError, "schema"):
                tl.load_preset("other", tmp)


class NativeValidationTests(unittest.TestCase):
    def test_existing_output_is_refused_by_native_validation(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "native"
            output.mkdir()
            with self.assertRaisesRegex(tl.TrainingConfigError, "already exists"):
                tl.validate_native(tl.build_native_argv(LOWLR, tl.ROOT, output))

    def test_unknown_native_flag_is_rejected(self):
        argv = tl.build_native_argv(LOWLR, tl.ROOT, "/nonexistent/native") + [
            "--policy.not_a_field=1"
        ]
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(tl.TrainingConfigError),
        ):
            tl.validate_native(argv)

    def test_parsed_value_disagreeing_with_preset_is_rejected(self):
        argv = tl.build_native_argv(LOWLR, tl.ROOT, "/nonexistent/native")
        cfg = tl.validate_native(argv)
        flags = tl.parse_argv(argv)
        flags["policy.scheduler_decay_steps"] = 15000
        with self.assertRaisesRegex(tl.TrainingConfigError, "decay_horizon_steps"):
            tl.effective_settings(cfg, flags)


class PlanAndAssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "project"
        self.preset = fake_assets(self.root, LOWLR)
        self.presets = Path(self.tmp.name) / "presets"
        self.presets.mkdir()
        (self.presets / "test-recipe.recipe.json").write_text(json.dumps(self.preset))

    def test_plan_validates_natively_without_creating_run_directories(self):
        now = dt.datetime(2026, 11, 11, 7, 0, tzinfo=dt.timezone.utc)
        spec = tl.plan(
            "test-recipe",
            self.root,
            self.presets,
            now=now,
            hash_base=True,
            selection=TWO,
        )
        self.assertFalse((self.root / tl.RUNS_SUBDIR).exists())
        self.assertEqual(
            spec["run_dir"],
            str(self.root / tl.RUNS_SUBDIR / "test-recipe-20261111T070000.000000Z"),
        )
        self.assertEqual(
            spec["effective"]["base"],
            str(self.root / self.preset["native_flags"]["policy.pretrained_path"]),
        )
        self.assertEqual(
            spec["torchrun"][-2:], ["--run", spec["run_dir"] + "/run.json"]
        )
        self.assertIn("--nproc_per_node=2", spec["torchrun"])
        self.assertTrue(math.isclose(spec["lr_at_step"]["15000"], 1.375e-5))
        later = tl.plan(
            "test-recipe",
            self.root,
            self.presets,
            now=now + dt.timedelta(microseconds=1),
        )
        self.assertNotEqual(later["run_dir"], spec["run_dir"])

    def test_existing_run_directory_is_never_reused(self):
        now = dt.datetime(2026, 11, 11, 7, 0, tzinfo=dt.timezone.utc)
        spec = tl.plan("test-recipe", self.root, self.presets, now=now, selection=TWO)
        run, _ = tl.write_run_record(spec)
        with self.assertRaisesRegex(tl.TrainingConfigError, "reuse"):
            tl.plan("test-recipe", self.root, self.presets, now=now)
        with self.assertRaises(FileExistsError):
            tl.write_run_record(spec)
        self.assertTrue((run / "run.json").is_file())

    def test_worker_recomputes_the_launcher_argv_and_refuses_drift(self):
        spec = tl.plan("test-recipe", self.root, self.presets, selection=TWO)
        run, _ = tl.write_run_record(spec)
        loaded, preset, loaded_run, argv = load_spec(
            run / "run.json", self.root, self.presets
        )
        self.assertEqual(argv, spec["native_argv"])
        self.assertEqual(loaded_run, run)
        edited = mutated(self.preset, native_flags__steps=20000)
        (self.presets / "test-recipe.recipe.json").write_text(json.dumps(edited))
        with self.assertRaisesRegex(tl.TrainingConfigError, "preset file changed"):
            load_spec(run / "run.json", self.root, self.presets)

    def test_training_rejects_mismatched_or_missing_collection_fps(self):
        info_path = (
            self.root / self.preset["native_flags"]["dataset.root"] / "meta/info.json"
        )
        info = json.loads(info_path.read_text())
        for fps in (30, 0, None, "15", True):
            with self.subTest(fps=fps):
                info["fps"] = fps
                info_path.write_text(json.dumps(info))
                with self.assertRaisesRegex(tl.TrainingConfigError, "15 Hz"):
                    tl.check_assets(self.preset, self.root)

    def test_unapproved_assets_are_rejected(self):
        tl.check_assets(self.preset, self.root, hash_base=True)
        base = self.root / self.preset["native_flags"]["policy.pretrained_path"]
        meta = self.root / self.preset["native_flags"]["dataset.root"] / "meta"
        cases = [
            (lambda: (base / "adapter_config.json").write_text("{}"), "original base"),
            (lambda: (base / "model.safetensors").write_bytes(b"other"), "size"),
            (lambda: (base / "model.safetensors").write_bytes(b"WEIGHTS"), "SHA256"),
            (
                lambda: (meta / "info.json").write_text(
                    '{"codebase_version": "v3.0", "total_episodes": 1}'
                ),
                "episodes",
            ),
            (
                lambda: (meta / "info.json").write_text('{"codebase_version": "v2.1"}'),
                "v3",
            ),
        ]
        for damage, text in cases:
            with self.subTest(text=text):
                snapshot = {
                    p: p.read_bytes()
                    for p in (*base.iterdir(), *meta.iterdir())
                    if p.is_file()
                }
                damage()
                with self.assertRaisesRegex(tl.TrainingConfigError, text):
                    tl.check_assets(self.preset, self.root, hash_base=True)
                (base / "adapter_config.json").unlink(missing_ok=True)
                for path, data in snapshot.items():
                    path.write_bytes(data)

    def test_init_adapter_must_be_relative_and_present(self):
        preset = mutated(self.preset, guards__init_adapter="runs/old/pretrained_model")
        with self.assertRaisesRegex(tl.TrainingConfigError, "init adapter missing"):
            tl.check_assets(preset, self.root)
        adapter = self.root / "runs/old/pretrained_model"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"x")
        with self.assertRaisesRegex(tl.TrainingConfigError, "init adapter config missing"):
            tl.check_assets(preset, self.root)
        cfg = {
            "peft_type": "LORA",
            "r": preset["native_flags"]["peft.r"],
            "lora_alpha": preset["native_flags"]["peft.lora_alpha"],
        }
        (adapter / "adapter_config.json").write_text(
            json.dumps({**cfg, "init_lora_weights": "pissa"})
        )
        with self.assertRaisesRegex(tl.TrainingConfigError, "init_lora_weights"):
            tl.check_assets(preset, self.root)
        (adapter / "adapter_config.json").write_text(json.dumps(cfg))
        self.assertIn("init_adapter", tl.check_assets(preset, self.root))
        with self.assertRaisesRegex(tl.TrainingConfigError, "project-relative"):
            tl.check_recipe(mutated(preset, guards__init_adapter="/abs/path"))

    def test_trainable_set_guard(self):
        names = [f"model.gemma_expert.layers.{i}.lora_A.weight" for i in range(80)]
        tl.check_trainable(names, self.preset["guards"])
        with self.assertRaises(tl.TrainingConfigError):
            tl.check_trainable(
                names[:-1] + ["model.paligemma.x.lora_A.weight"], self.preset["guards"]
            )
        with self.assertRaises(tl.TrainingConfigError):
            tl.check_trainable(names[:79], self.preset["guards"])

    def test_vlm_recipe_requires_lora_in_vlm_and_expert(self):
        recipe = tl.check_recipe(tl.load_preset("training-vlm-expert"))
        guards = recipe["guards"]
        vlm = [f"m.paligemma.layers.{i}.lora_A.weight" for i in range(180)]
        expert = [f"m.gemma_expert.layers.{i}.lora_A.weight" for i in range(80)]
        tl.check_trainable(vlm + expert, guards, expert_only=False)
        for bad in (
            [n.replace("paligemma", "other") for n in vlm] + expert,
            vlm + [n.replace("gemma_expert", "other") for n in expert],
            ["m.paligemma.dense.weight"] + vlm[1:] + expert,
            vlm + expert[1:],
        ):
            with self.assertRaises(tl.TrainingConfigError):
                tl.check_trainable(bad, guards, expert_only=False)
        with self.assertRaisesRegex(tl.TrainingConfigError, "frozen VLM"):
            tl.check_trainable(vlm + expert, guards)

    def test_vlm_targets_reach_native_config(self):
        preset = mutated(
            self.preset,
            native_flags__policy_DOT_train_expert_only=False,
            native_flags__peft_DOT_target_modules=r".*\.paligemma\..*\.q_proj",
        )
        argv = tl.build_native_argv(preset, self.root, self.root / "out-vlm")
        flags = tl.parse_argv(argv)
        effective = tl.effective_settings(tl.validate_native(argv), flags)
        self.assertFalse(effective["train_expert_only"])
        self.assertEqual(
            effective["lora_target_modules"], r".*\.paligemma\..*\.q_proj"
        )


class CheckpointTests(unittest.TestCase):
    def make_checkpoint(self, root, flags, step, lr):
        model, state = root / "pretrained_model", root / "training_state"
        (model / "tokenizer").mkdir(parents=True)
        state.mkdir()
        for name in tl.CHECKPOINT_FILES:
            (model / name).write_text("x")
        (model / "train_config.json").write_text(
            json.dumps(
                {
                    "steps": flags["steps"],
                    "batch_size": flags["batch_size"],
                    "resume": False,
                    "seed": flags["seed"],
                    "scheduler": {
                        "peak_lr": flags["policy.optimizer_lr"],
                        "decay_lr": flags["policy.scheduler_decay_lr"],
                        "num_warmup_steps": flags["policy.scheduler_warmup_steps"],
                        "num_decay_steps": flags["policy.scheduler_decay_steps"],
                    },
                }
            )
        )
        (model / "adapter_config.json").write_text(
            json.dumps({"r": flags["peft.r"], "lora_alpha": flags["peft.lora_alpha"]})
        )
        (state / "training_step.json").write_text(json.dumps({"step": step}))
        (state / "scheduler_state.json").write_text(json.dumps({"_last_lr": [lr]}))
        return root

    def test_saved_checkpoint_must_carry_processors_tokenizer_and_unscaled_lr(self):
        flags = LOWLR["native_flags"]
        with TemporaryDirectory() as tmp:
            ok = self.make_checkpoint(
                Path(tmp) / "a", flags, 15000, tl.expected_lr(flags, 15000)
            )
            self.assertEqual(tl.check_saved_checkpoint(ok, 15000, flags)["step"], 15000)
            compressed = self.make_checkpoint(Path(tmp) / "b", flags, 15000, 2.5e-6)
            with self.assertRaisesRegex(tl.TrainingConfigError, "preset schedule"):
                tl.check_saved_checkpoint(compressed, 15000, flags)
            (ok / "pretrained_model" / "tokenizer" / "tokenizer.json").unlink()
            with self.assertRaisesRegex(tl.TrainingConfigError, "tokenizer"):
                tl.check_saved_checkpoint(ok, 15000, flags)

    def test_expected_saves(self):
        self.assertEqual(
            tl.expected_save_steps(15000, 2500), [2500, 5000, 7500, 10000, 12500, 15000]
        )
        self.assertEqual(tl.expected_save_steps(4, 2), [2, 4])

    def test_noninterval_final_checkpoint_is_still_verified(self):
        flags = {**LOWLR["native_flags"], "steps": 31000}
        saves = tl.expected_save_steps(31000, 2500)
        self.assertEqual(saves, [*range(2500, 30001, 2500), 31000])
        with TemporaryDirectory() as tmp:
            final = self.make_checkpoint(
                Path(tmp) / "final", flags, saves[-1], tl.expected_lr(flags, saves[-1])
            )
            self.assertEqual(
                tl.check_saved_checkpoint(final, 31000, flags)["step"], 31000
            )
            (final / "pretrained_model" / "adapter_model.safetensors").unlink()
            with self.assertRaisesRegex(tl.TrainingConfigError, "adapter_model"):
                tl.check_saved_checkpoint(final, 31000, flags)


class RecipeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "project"
        self.presets = self.root / "config/training"
        self.presets.mkdir(parents=True)
        bound = tl.bind_dataset(tl.load_preset(), "data/synthetic", "local/synthetic")
        self.preset = fake_assets(self.root, bound)
        # Source recipes do not contain dataset inputs: the caller must supply both.
        del self.preset["native_flags"]["dataset.root"]
        del self.preset["native_flags"]["dataset.repo_id"]
        self.default = self.presets / "training.recipe.json"
        self.default.write_text(json.dumps(self.preset))

    def plan(self, recipe=None, **kwargs):
        return tl.plan(
            recipe,
            root=self.root,
            preset_dir=self.presets,
            dataset_root="data/synthetic",
            repo_id="local/synthetic",
            **kwargs,
        )

    def copy_recipe(self, filename="training-45k.recipe.json", flags=None):
        preset = copy.deepcopy(self.preset)
        preset["native_flags"].update({"steps": 45000, **(flags or {})})
        path = self.presets / filename
        path.write_text(json.dumps(preset))
        return path

    def test_supplied_recipes_and_default_path(self):
        default = tl.load_preset()
        for recipe, ranks, batch in (
            (default, 1, 8),
            (tl.load_preset("training-2gpu"), 2, 4),
            (tl.load_preset("training-warm-start"), 1, 8),
        ):
            expected = copy.deepcopy(default)
            expected.update(name=recipe["name"], summary=recipe["summary"])
            expected["native_flags"].update(
                {"parallelism.dp_replicate": ranks, "batch_size": batch}
            )
            if recipe["name"] == "training-warm-start":
                expected["native_flags"].update(
                    {
                        "policy.optimizer_lr": 5e-5,
                        "policy.scheduler_decay_lr": 5e-6,
                        "policy.scheduler_warmup_steps": 500,
                    }
                )
                expected["guards"]["init_adapter"] = (
                    "outputs/training-runs/REPLACE_WITH_RUN/native/checkpoints/030000/pretrained_model"
                )
            self.assertEqual(recipe, expected)
        self.assertEqual(
            tl.resolve_recipe("config/training/training.recipe.json"),
            tl.resolve_recipe(),
        )
        spec = self.plan()
        self.assertEqual(Path(spec["preset_file"]), self.default)
        self.assertEqual(spec["preset"], "training")
        self.assertEqual(spec["effective"]["stop_budget_steps"], 30000)
        self.assertEqual(spec["effective"]["world_size"], 1)
        self.assertEqual(spec["effective"]["per_rank_batch"], 8)
        with self.assertRaisesRegex(tl.TrainingConfigError, "requested ranks"):
            self.plan(selection=TWO)
        self.assertFalse((self.root / tl.RUNS_SUBDIR).exists())
        selected = self.plan(selection=GPU0)
        run, _ = tl.write_run_record(selected)
        record = json.loads((run / "run.json").read_text())
        record["gpu_selection"] = TWO
        (run / "run.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(tl.TrainingConfigError, "requested ranks"):
            load_spec(run / "run.json", self.root, self.presets)

    def test_copied_recipe_keeps_internal_name_and_is_revalidated_by_worker(self):
        path = self.copy_recipe()
        self.assertEqual(tl.load_preset(path)["name"], "training")
        spec = self.plan(path, selection=GPU0)
        self.assertEqual(spec["preset"], "training-45k")
        self.assertEqual(spec["effective"]["stop_budget_steps"], 45000)
        run, _ = tl.write_run_record(spec)
        _, preset, _, argv = load_spec(run / "run.json", self.root, self.presets)
        self.assertEqual(preset["name"], "training")
        self.assertEqual(argv, spec["native_argv"])
        self.copy_recipe(flags={"steps": 31000})
        with self.assertRaisesRegex(tl.TrainingConfigError, "preset file changed"):
            load_spec(run / "run.json", self.root, self.presets)

    def test_experiment_settings_reach_native_config_and_worker(self):
        path = self.copy_recipe(
            flags={
                "steps": 31000,
                "eval_steps": 1000,
                "save_freq": 2500,
                "max_eval_samples": 100,
                "dataset.image_transforms.enable": True,
            },
        )
        spec = self.plan(path, selection=GPU0)
        effective = spec["effective"]
        self.assertEqual(effective["stop_budget_steps"], 31000)
        self.assertEqual(effective["eval_steps"], 1000)
        self.assertEqual(effective["save_freq"], 2500)
        self.assertEqual(effective["max_eval_samples"], 100)
        self.assertIs(effective["image_augmentation"], True)
        run, _ = tl.write_run_record(spec)
        _, _, _, argv = load_spec(run / "run.json", self.root, self.presets)
        self.assertEqual(argv, spec["native_argv"])

    def test_recipe_paths_and_internal_names_cannot_escape_run_directory(self):
        self.preset["name"] = "../../outside"
        for filename, label in (
            ("training-30k.recipe.json", "training-30k"),
            ("..odd name_! .recipe.json", "odd-name"),
            ("....recipe.json", "recipe"),
        ):
            with self.subTest(filename=filename):
                path = self.copy_recipe(filename)
                spec = self.plan(path)
                run = Path(spec["run_dir"])
                self.assertEqual(spec["preset"], label)
                self.assertEqual(run.parent, self.root / tl.RUNS_SUBDIR)
                self.assertTrue(run.name.startswith(f"{label}-"))
        with self.assertRaisesRegex(tl.TrainingConfigError, "invalid run label"):
            tl.planned_run_dir(self.root, "../outside")

    def test_cli_default_and_recipe_smoke_use_same_selector_and_idle_gpu_callback(self):
        path = self.copy_recipe()
        dual = self.copy_recipe(
            "training-2gpu.recipe.json",
            {"parallelism.dp_replicate": 2, "batch_size": 4, "steps": 30000},
        )
        for options, expected, steps, selection in (
            ([], self.default, 30000, GPU0),
            (["--recipe", str(path), "--smoke"], path, 4, GPU0),
            (["--recipe", str(dual)], dual, 30000, TWO),
            (["--recipe", str(dual), "--smoke"], dual, 4, TWO),
        ):
            original = tl.plan
            with (
                self.subTest(options=options),
                patch.object(
                    tl,
                    "plan",
                    side_effect=lambda recipe, **kwargs: original(
                        recipe, root=self.root, preset_dir=self.presets, **kwargs
                    ),
                ),
                patch.object(tl, "launch", return_value=0) as launch,
                patch.object(tl, "query_gpus") as query,
            ):
                query.return_value = tl.gpu_inventory("0, GPU-a, 15, 0\n1, GPU-b, 15, 0\n", "")
                self.assertEqual(
                    tl.main(
                        [
                            "--dataset-root",
                            "data/synthetic",
                            "--repo-id",
                            "local/synthetic",
                            *options,
                        ]
                    ),
                    0,
                )
                spec = launch.call_args.args[0]
                self.assertEqual(Path(spec["preset_file"]), expected)
                self.assertEqual(spec["effective"]["stop_budget_steps"], steps)
                self.assertEqual(spec["gpu_selection"], selection)
                self.assertEqual(spec["effective"]["global_batch"], 8)
                if options:
                    self.assertEqual(
                        spec["_preset"]["native_flags"]["steps"],
                        30000 if expected == dual else 45000,
                    )
                    run, _ = tl.write_run_record(spec)
                    self.assertEqual(
                        load_spec(run / "run.json", self.root, self.presets)[3],
                        spec["native_argv"],
                    )

    def test_cli_check_defaults_and_selects_without_gpu_access(self):
        path = self.copy_recipe()
        for options, expected, steps in (
            ([], self.default, 30000),
            (["--recipe", str(path)], path, 45000),
        ):
            original = tl.plan
            with (
                self.subTest(options=options),
                patch.object(
                    tl,
                    "plan",
                    side_effect=lambda recipe, **kwargs: original(
                        recipe, root=self.root, preset_dir=self.presets, **kwargs
                    ),
                ),
                patch.object(tl, "query_gpus", side_effect=AssertionError("GPU query")),
                patch.dict("os.environ"),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(
                    tl.main(
                        [
                            "--dataset-root",
                            "data/synthetic",
                            "--repo-id",
                            "local/synthetic",
                            "--dry-run",
                            *options,
                        ]
                    ),
                    0,
                )
                report = json.loads(stdout.getvalue())
                self.assertEqual(Path(report["preset_file"]), expected)
                self.assertEqual(report["effective"]["stop_budget_steps"], steps)
                self.assertFalse(report["training_started"])
                self.assertEqual(set(report["topology_candidates"]), {"1"})

    def test_cli_still_requires_both_dataset_arguments(self):
        for args in (
            [],
            ["--dataset-root", "data/synthetic"],
            ["--repo-id", "local/synthetic"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    tl.main(args)
                self.assertEqual(error.exception.code, 2)


class TopologyTests(unittest.TestCase):
    def test_topology_is_derived_from_selected_count_with_fixed_global_batch(self):
        for preset in (BASELINE, LOWLR):
            self.assertEqual(
                tl.resolve_topology(preset, 2),
                {
                    "batch_size": 4,
                    "parallelism.dp_replicate": 2,
                    "accelerator.gradient_accumulation.steps": 1,
                },
            )
            self.assertEqual(
                tl.resolve_topology(preset, 1),
                {
                    "batch_size": 8,
                    "parallelism.dp_replicate": 1,
                    "accelerator.gradient_accumulation.steps": 1,
                },
            )
            for bad in (0, 3, True):
                with self.assertRaises(tl.TrainingConfigError):
                    tl.resolve_topology(preset, bad)

    def test_unsupported_mappings_are_rejected(self):
        odd = mutated(LOWLR, guards__global_batch=9)
        with self.assertRaisesRegex(tl.TrainingConfigError, "divisible"):
            tl.resolve_topology(odd, 2)
        accum = mutated(
            LOWLR, native_flags__accelerator_DOT_gradient_accumulation_DOT_steps=2
        )
        with self.assertRaisesRegex(tl.TrainingConfigError, "accumulation 1"):
            tl.resolve_topology(accum, 1)

    def test_only_topology_flags_change_and_preset_bytes_are_untouched(self):
        before = {
            n: (tl.PRESET_DIR / f"{n}.recipe.json").read_bytes()
            for n in ("training", "training-2gpu", "training-warm-start")
        }
        nominal = tl.parse_argv(tl.build_native_argv(LOWLR, tl.ROOT, "/x/native"))
        for count in (1, 2):
            flags = tl.parse_argv(
                tl.build_native_argv(LOWLR, tl.ROOT, "/x/native", gpu_count=count)
            )
            self.assertEqual(
                {k: v for k, v in flags.items() if k not in TOPOLOGY_FLAGS},
                {k: v for k, v in nominal.items() if k not in TOPOLOGY_FLAGS},
            )
            self.assertEqual(
                flags["batch_size"]
                * flags["parallelism.dp_replicate"]
                * flags["accelerator.gradient_accumulation.steps"],
                8,
            )
            self.assertEqual(flags["accelerator.gradient_accumulation.steps"], 1)
        self.assertEqual(
            tl.build_native_argv(LOWLR, tl.ROOT, "/x/native", gpu_count=2),
            tl.build_native_argv(LOWLR, tl.ROOT, "/x/native"),
        )
        self.assertEqual(
            before,
            {n: (tl.PRESET_DIR / f"{n}.recipe.json").read_bytes() for n in before},
        )
        self.assertEqual(
            tl.load_preset("training-2gpu")["native_flags"]["policy.optimizer_lr"], 1e-4
        )

    def test_selection_must_be_eligible_unique_and_well_formed(self):
        for bad in (
            None,
            [],
            [{"index": 2, "uuid": "GPU-c"}],
            [TWO[1], TWO[0]],
            [TWO[0], TWO[0]],
            [{"index": 0, "uuid": "0"}],
            [{"index": 0, "uuid": "GPU-a", "batch_size": 8}],
            [{"index": 0, "uuid": "GPU-a"}, {"index": 1, "uuid": "GPU-a"}],
        ):
            with self.subTest(bad=bad), self.assertRaises(tl.TrainingConfigError):
                tl.check_selection(LOWLR, bad)

    def test_native_cpu_validation_of_one_and_two_gpu_plans(self):
        for count, per_rank in ((1, 8), (2, 4)):
            with self.subTest(count=count), TemporaryDirectory() as tmp:
                argv = tl.build_native_argv(
                    LOWLR, tl.ROOT, Path(tmp) / "native", gpu_count=count
                )
                effective = tl.effective_settings(
                    tl.validate_native(argv), tl.parse_argv(argv)
                )
                self.assertEqual(
                    (
                        effective["per_rank_batch"],
                        effective["world_size"],
                        effective["grad_accumulation"],
                        effective["global_batch"],
                        effective["stop_budget_steps"],
                        effective["decay_horizon_steps"],
                        effective["warmup_steps"],
                        effective["peak_lr"],
                        effective["seed"],
                        effective["lora_r"],
                    ),
                    (
                        per_rank,
                        count,
                        1,
                        8,
                        15000,
                        30000,
                        LOWLR["native_flags"]["policy.scheduler_warmup_steps"],
                        LOWLR["native_flags"]["policy.optimizer_lr"],
                        LOWLR["native_flags"]["seed"],
                        LOWLR["native_flags"]["peft.r"],
                    ),
                )
                self.assertEqual(list(Path(tmp).iterdir()), [])


class SelectedPlanReplayTests(unittest.TestCase):
    setUp = PlanAndAssetTests.setUp

    def test_one_and_two_gpu_plans_are_recorded_and_replayed_by_ranks(self):
        for selection, per_rank in ((GPU1, 8), (GPU0, 8), (TWO, 4)):
            with self.subTest(selection=selection):
                spec = tl.plan(
                    "test-recipe", self.root, self.presets, selection=selection
                )
                self.assertIn(f"--nproc_per_node={len(selection)}", spec["torchrun"])
                self.assertEqual(
                    spec["topology"]["selected_uuids"], [g["uuid"] for g in selection]
                )
                self.assertEqual(
                    spec["topology"]["effective"],
                    {
                        "per_rank_batch": per_rank,
                        "world_size": len(selection),
                        "grad_accumulation": 1,
                        "global_batch": 8,
                    },
                )
                run, record = tl.write_run_record(spec)
                self.assertEqual(record["preset_content"], self.preset)
                self.assertEqual(record["gpu_selection"], selection)
                _, _, _, argv = load_spec(run / "run.json", self.root, self.presets)
                self.assertEqual(argv, spec["native_argv"])

    def test_rank_replay_detects_tampered_selection_argv_or_world(self):
        spec = tl.plan("test-recipe", self.root, self.presets, selection=GPU1)
        run, _ = tl.write_run_record(spec)
        original = json.loads((run / "run.json").read_text())

        def tampered(edit, pattern):
            record = copy.deepcopy(original)
            edit(record)
            (run / "run.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(tl.TrainingConfigError, pattern):
                load_spec(run / "run.json", self.root, self.presets)

        tampered(
            lambda r: r.update(gpu_selection=TWO), "native argv"
        )  # selection without matching argv
        tampered(
            lambda r: r.update(gpu_selection=[{"index": 2, "uuid": "GPU-c"}]),
            "eligible",
        )
        tampered(lambda r: r.update(gpu_selection=None), "selection")
        tampered(
            lambda r: r.update(
                native_argv=[
                    a.replace("--batch_size=8", "--batch_size=4")
                    for a in r["native_argv"]
                ]
            ),
            "native argv",
        )
        tampered(
            lambda r: r.update(
                torchrun=[
                    a.replace("--nproc_per_node=1", "--nproc_per_node=2")
                    for a in r["torchrun"]
                ]
            ),
            "world size",
        )
        (run / "run.json").write_text(json.dumps(original))
        load_spec(run / "run.json", self.root, self.presets)


class LaunchSeamTests(unittest.TestCase):
    GPUS = "0, GPU-a, 15, 0\n1, GPU-b, 15, 0\n2, GPU-c, 30000, 99\n"

    def select(self, apps="", gpus=None, limit=None):
        return tl.select_idle_gpus(
            tl.gpu_inventory(self.GPUS if gpus is None else gpus, apps), [0, 1], limit=limit
        )

    def test_idle_selection_prefers_both_and_falls_back_to_the_idle_one(self):
        self.assertEqual(self.select(), TWO)
        self.assertEqual(self.select("GPU-a, 4242\n"), GPU1)  # GPU0 busy -> GPU1 only
        self.assertEqual(self.select("GPU-b, 4242\n"), GPU0)  # GPU1 busy -> GPU0 only
        self.assertEqual(self.select(limit=1), GPU0)
        self.assertEqual(self.select("GPU-a, 4242\n", limit=1), GPU1)
        self.assertEqual(self.select("GPU-b, 4242\n", limit=1), GPU0)
        self.assertEqual(self.select(limit=2), TWO)
        with self.assertRaisesRegex(
            tl.TrainingConfigError, "no idle eligible GPU.*not waiting"
        ):
            self.select("GPU-a, 1\nGPU-b, 2\n")

    def test_single_gpu_host_selects_the_one_eligible_gpu(self):
        self.assertEqual(self.select("", "0, GPU-a, 15, 0\n"), GPU0)
        self.assertEqual(self.select("", "1, GPU-b, 15, 0\n"), GPU1)
        with self.assertRaisesRegex(tl.TrainingConfigError, "no idle eligible GPU"):
            self.select("GPU-a, 1\n", "0, GPU-a, 15, 0\n")

    def test_any_compute_process_is_busy_even_at_low_utilization_and_memory(self):
        gpus = "0, GPU-a, 0, 0\n1, GPU-b, 0, 0\n"
        self.assertEqual(self.select("GPU-a, 99\n", gpus), GPU1)
        busy_mem = "0, GPU-a, 9000, 0\n1, GPU-b, 15, 0\n"
        self.assertEqual(self.select("", busy_mem), GPU1)
        busy_util = "0, GPU-a, 15, 3\n1, GPU-b, 15, 0\n"
        self.assertEqual(self.select("", busy_util), GPU1)

    def test_malformed_inventory_fails_closed(self):
        for gpus, apps in (
            ("0, GPU-a, 15\n", ""),
            ("x, GPU-a, 15, 0\n", ""),
            ("0, a, 15, 0\n", ""),
            ("0, GPU-a, 15, 0\n0, GPU-b, 15, 0\n", ""),
            ("0, GPU-a, 15, 0\n1, GPU-a, 15, 0\n", ""),
            ("0, GPU-a, N/A, 0\n", ""),
            ("", ""),
            (self.GPUS, "GPU-z, 1\n"),
            (self.GPUS, "GPU-a\n"),
            ("2, GPU-c, 0, 0\n", ""),
        ):
            with (
                self.subTest(gpus=gpus, apps=apps),
                self.assertRaises(tl.TrainingConfigError),
            ):
                self.select(apps, gpus)

    def test_revalidation_requires_same_uuid_and_still_idle(self):
        inventory = tl.gpu_inventory(self.GPUS, "")
        self.assertEqual(
            [r["uuid"] for r in tl.check_idle_gpus(inventory, TWO)], ["GPU-a", "GPU-b"]
        )
        remapped = tl.gpu_inventory("0, GPU-b, 15, 0\n1, GPU-a, 15, 0\n", "")
        with self.assertRaisesRegex(tl.TrainingConfigError, "identity changed"):
            tl.check_idle_gpus(remapped, GPU1)
        with self.assertRaisesRegex(tl.TrainingConfigError, "compute processes"):
            tl.check_idle_gpus(tl.gpu_inventory(self.GPUS, "GPU-b, 1234\n"), TWO)
        with self.assertRaisesRegex(tl.TrainingConfigError, "not idle"):
            tl.check_idle_gpus(inventory, [{"index": 2, "uuid": "GPU-c"}])
        with self.assertRaisesRegex(tl.TrainingConfigError, "identity changed"):
            tl.check_idle_gpus(inventory, [{"index": 3, "uuid": "GPU-d"}])

    def test_cli_passes_idle_gpu_selection_callback_to_plan(self):
        with (
            patch.object(tl, "plan") as plan,
            patch.object(tl, "launch", return_value=0) as launch,
            patch.object(
                tl, "query_gpus", return_value=tl.gpu_inventory(self.GPUS, "GPU-a, 4242\n")
            ),
            patch.dict("os.environ"),
        ):
            self.assertEqual(
                tl.main(
                    [
                        "test-recipe",
                        "--dataset-root",
                        "data/synthetic",
                        "--repo-id",
                        "local/synthetic",
                    ]
                ),
                0,
            )
            select = plan.call_args.kwargs["select"]
            self.assertEqual(select(LOWLR), GPU1)
            launch.assert_called_once()

    def test_child_environment_maps_selected_physical_uuids(self):
        env = tl.child_environment(Path("/p"), GPU1, {"CUDA_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-b")
        self.assertEqual(
            tl.child_environment(Path("/p"), TWO, {})["CUDA_VISIBLE_DEVICES"],
            "GPU-a,GPU-b",
        )

    def test_child_environment_is_local_only_and_pinned(self):
        env = tl.child_environment(
            Path("/p"),
            TWO,
            {
                "PYTHONPATH": "/x",
                "ACCELERATE_MIXED_PRECISION": "fp16",
                "HF_HUB_OFFLINE": "0",
                "HF_TOKEN_PATH": "/h/.cache/huggingface/token",
                "HOME": "/h",
            },
        )
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("ACCELERATE_MIXED_PRECISION", env)
        self.assertEqual(
            (
                env["HF_HUB_OFFLINE"],
                env["CUDA_VISIBLE_DEVICES"],
                env["HF_HOME"],
                env["HF_HUB_CACHE"],
                env["HF_TOKEN_PATH"],
                env["HOME"],
            ),
            (
                "1",
                "GPU-a,GPU-b",
                "/p/hf",
                "/p/hf/hub",
                "/h/.cache/huggingface/token",
                "/h",
            ),
        )

    def test_child_environment_honors_inherited_tmpdir_and_preserves_default(self):
        long_root = Path("/" + "/".join(["deep-worktree-segment"] * 5))
        default_env = tl.child_environment(long_root, GPU0, {"HOME": "/h"})
        self.assertEqual(default_env["TMPDIR"], str(long_root / ".tmp" / "training"))
        empty_env = tl.child_environment(long_root, GPU0, {"TMPDIR": ""})
        self.assertEqual(empty_env["TMPDIR"], str(long_root / ".tmp" / "training"))
        custom_env = tl.child_environment(long_root, GPU0, {"TMPDIR": "/tmp/short-ipc"})
        self.assertEqual(custom_env["TMPDIR"], "/tmp/short-ipc")

    def test_launch_rejects_too_long_tmpdir_and_creates_explicit_short_tmpdir(self):
        with TemporaryDirectory() as short_base:
            long_root = (
                Path(short_base)
                / ("worktree_" + "x" * 40)
                / ("checkpoint_" + "y" * 40)
            )
            presets = Path(short_base) / "presets"
            presets.mkdir()
            preset = fake_assets(long_root, LOWLR)
            (presets / "test-recipe.recipe.json").write_text(json.dumps(preset))
            spec = tl.plan("test-recipe", long_root, presets, selection=GPU0)

            base_env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TMPDIR")
            }
            with (
                patch.dict(os.environ, base_env, clear=True),
                patch.object(
                    tl.sys, "prefix", str(long_root / ".pixi" / "envs" / "default")
                ),
                patch.object(
                    tl.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=100 * 1024**3),
                ),
                patch.object(
                    tl, "query_gpus", return_value=tl.gpu_inventory("0, GPU-a, 15, 0\n", "")
                ),
                patch.object(tl.subprocess, "Popen") as popen,
            ):
                with self.assertRaisesRegex(tl.TrainingConfigError, "AF_UNIX"):
                    tl.launch(spec, root=long_root)
                popen.assert_not_called()

            # Explicit short TMPDIR is created before Popen and passed unchanged to child.
            short_tmp = Path(short_base) / "ipc"
            self.assertFalse(short_tmp.exists())
            seen_env = {}

            def fake_popen(cmd, **kwargs):
                seen_env.update(kwargs["env"])
                self.assertTrue(short_tmp.is_dir())
                run_json = Path(cmd[cmd.index("--run") + 1])
                (run_json.parent / "rank-0.json").write_text(
                    json.dumps({"status": "success", "rank": 0})
                )
                return SimpleNamespace(pid=30000, poll=lambda: 0, wait=lambda timeout=None: 0)

            with (
                patch.dict(
                    os.environ, {**base_env, "TMPDIR": str(short_tmp)}, clear=True
                ),
                patch.object(
                    tl.sys, "prefix", str(long_root / ".pixi" / "envs" / "default")
                ),
                patch.object(
                    tl.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=100 * 1024**3),
                ),
                patch.object(
                    tl, "query_gpus", return_value=tl.gpu_inventory("0, GPU-a, 15, 0\n", "")
                ),
                patch.object(tl.subprocess, "Popen", side_effect=fake_popen),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc = tl.launch(spec, root=long_root)
            self.assertEqual(rc, 0)
            self.assertEqual(seen_env["TMPDIR"], str(short_tmp))

    def test_spawn_torch_storage_fd_sharing_succeeds_with_short_tmpdir_under_long_root(self):
        import subprocess
        import sys

        with TemporaryDirectory() as short_base:
            long_root = Path(short_base) / ("deep_" + "z" * 80)
            short_tmp = Path(short_base) / "s"
            short_tmp.mkdir()
            env = tl.child_environment(
                long_root, GPU0, {**os.environ, "TMPDIR": str(short_tmp)}
            )
            env["CUDA_VISIBLE_DEVICES"] = ""
            # Spawn worker shares a torch.Tensor across multiprocessing Queue (triggers DupFd/AF_UNIX).
            script = Path(short_base) / "spawn_probe.py"
            script.write_text(
                "import multiprocessing as mp, torch\n"
                "def worker(q, done):\n"
                "    q.put(torch.arange(16, dtype=torch.float32))\n"
                "    done.wait(timeout=15)\n"
                "if __name__ == '__main__':\n"
                "    ctx = mp.get_context('spawn')\n"
                "    q, done = ctx.Queue(), ctx.Event()\n"
                "    p = ctx.Process(target=worker, args=(q, done))\n"
                "    p.start()\n"
                "    t = q.get(timeout=15)\n"
                "    done.set()\n"
                "    p.join(timeout=15)\n"
                "    assert p.exitcode == 0 and int(t.sum().item()) == 120\n"
            )
            res = subprocess.run(
                [sys.executable, str(script)],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(res.returncode, 0, res.stderr)

    def test_cli_rejects_unknown_preset_before_any_work(self):
        with patch.dict("os.environ"), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(
                tl.main(
                    [
                        "lora-missing",
                        "--dataset-root",
                        "data/synthetic",
                        "--repo-id",
                        "local/synthetic",
                        "--dry-run",
                    ]
                ),
                2,
            )
        self.assertIn("REJECTED", err.getvalue())


class GpuAdmissionTests(unittest.TestCase):
    IDLE = "0, GPU-a, 15, 0\n1, GPU-b, 15, 0\n"
    BASE_NOW = dt.datetime(2026, 11, 23, 9, 0, tzinfo=dt.timezone.utc)

    def setUp(self):
        PlanAndAssetTests.setUp(self)
        self._seq = 0

    def _plan(self, selection):
        self._seq += 1
        return tl.plan(
            "test-recipe",
            self.root,
            self.presets,
            now=self.BASE_NOW + dt.timedelta(seconds=self._seq),
            selection=selection,
        )

    def _make_proc(self, cmd, env, pid, on_wait=None):
        class _FakeProc:
            def __init__(self):
                self.cmd, self.env, self.pid, self._rc = list(cmd), dict(env), pid, None

            def poll(self):
                return self._rc

            def wait(self, timeout=None):
                if on_wait is not None:
                    on_wait()
                run_json = Path(self.cmd[self.cmd.index("--run") + 1])
                world = json.loads(run_json.read_text())["effective"]["world_size"]
                for rank in range(world):
                    (run_json.parent / f"rank-{rank}.json").write_text(
                        json.dumps({"status": "success", "rank": rank})
                    )
                self._rc = 0
                return 0

        return _FakeProc()

    @contextlib.contextmanager
    def _harness(self, popen):
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        }
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.object(
                tl.sys, "prefix", str(self.root / ".pixi" / "envs" / "default")
            ),
            patch.object(
                tl.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=100 * 1024**3),
            ),
            patch.object(tl, "stop_owned_group"),
            patch.object(
                tl, "query_gpus", side_effect=lambda: tl.gpu_inventory(self.IDLE, "")
            ),
            patch.object(tl.subprocess, "Popen", side_effect=popen),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            yield

    def test_gpu_reservation_blocks_overlap_allows_disjoint_and_releases(self):
        spawned, nested_err = [], []
        spec_gpu1 = self._plan(GPU1)
        spec_same = self._plan(GPU1)
        spec_both = self._plan(TWO)
        spec_gpu0 = self._plan(GPU0)
        spec_after = self._plan(GPU1)

        def during_run1():
            try:
                # NVML still reports both GPUs idle; GPU1 is reserved by run 1.
                with self.assertRaises(tl.TrainingConfigError):
                    tl.launch(spec_same, root=self.root)
                # Partial acquisition of GPU0 followed by collision on GPU1 must release GPU0.
                with self.assertRaises(tl.TrainingConfigError):
                    tl.launch(spec_both, root=self.root)
                self.assertEqual(len(spawned), 1)
                # Disjoint GPU0 launches concurrently while run 1 holds GPU1.
                self.assertEqual(tl.launch(spec_gpu0, root=self.root), 0)
                self.assertEqual(len(spawned), 2)
                self.assertEqual(spawned[1].env["CUDA_VISIBLE_DEVICES"], "GPU-a")
            except BaseException as exc:
                nested_err.append(exc)
                raise

        def fake_popen(cmd, **kwargs):
            idx = len(spawned)
            proc = self._make_proc(
                cmd,
                kwargs["env"],
                10000 + idx,
                on_wait=during_run1 if idx == 0 else None,
            )
            spawned.append(proc)
            return proc

        with self._harness(fake_popen):
            rc1 = tl.launch(spec_gpu1, root=self.root)
            if nested_err:
                raise nested_err[0]
            self.assertEqual(rc1, 0)
            self.assertEqual(tl.launch(spec_after, root=self.root), 0)
        self.assertEqual(
            [p.env["CUDA_VISIBLE_DEVICES"] for p in spawned],
            ["GPU-b", "GPU-a", "GPU-b"],
        )

    def test_gpu_lock_released_after_popen_error(self):
        calls = 0

        def flaky_popen(cmd, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("spawn failed")
            return self._make_proc(cmd, kwargs["env"], 20000)

        spec1, spec2 = self._plan(GPU0), self._plan(GPU0)
        with self._harness(flaky_popen):
            self.assertEqual(tl.launch(spec1, root=self.root), 1)
            self.assertEqual(tl.launch(spec2, root=self.root), 0)
        self.assertEqual(calls, 2)
        self.assertEqual(
            json.loads((Path(spec1["run_dir"]) / "status.json").read_text())["status"],
            "failed",
        )


class RunScopedGpuAllowlistTests(unittest.TestCase):
    GPU_CSV = "0, GPU-a, 1427, 0\n"
    APPS_CSV = "GPU-a, 41001, 264\nGPU-a, 41002, 432\n"
    LIVE_PROCS = {
        41001: {
            "pid": 41001,
            "starttime": 3533,
            "exe": "/usr/libexec/remote-desktop-daemon",
            "uid": 1000,
            "cgroup": "0::/user.slice/app.slice/remote-desktop.service",
        },
        41002: {
            "pid": 41002,
            "starttime": 47238672,
            "exe": "/opt/service/bin/python3.14",
            "uid": 1000,
            "cgroup": "0::/user.slice/app.slice/background-gateway.service",
        },
    }

    def allowlist(self):
        return {
            "schema": 1,
            "gpu_uuid": "GPU-a",
            "max_background_memory_mib": 1920,
            "max_noncompute_memory_mib": 1024,
            "max_background_util_pct": 5,
            "allowed_processes": [
                {**self.LIVE_PROCS[41001], "max_memory_mib": 384},
                {**self.LIVE_PROCS[41002], "max_memory_mib": 640},
            ],
        }

    def read_proc(self, overrides=None):
        table = copy.deepcopy(self.LIVE_PROCS)
        if overrides:
            table.update(overrides)
        return lambda pid: table.get(pid)

    def test_default_strict_idle_rejects_shared_background_gpu(self):
        inv = tl.gpu_inventory(self.GPU_CSV, self.APPS_CSV)
        self.assertFalse(tl.is_idle(inv[0]))
        with self.assertRaisesRegex(tl.TrainingConfigError, "no idle eligible GPU"):
            tl.select_idle_gpus(inv, [0], limit=1)
        with self.assertRaisesRegex(tl.TrainingConfigError, "compute processes"):
            tl.check_idle_gpus(inv, GPU0)

    def test_exact_identity_and_vram_bounds_admit_selected_gpu(self):
        inv = tl.gpu_inventory(self.GPU_CSV, self.APPS_CSV)
        allow = self.allowlist()
        selected = tl.select_idle_gpus(
            inv, [0], limit=1, allowlist=allow, read_proc=self.read_proc()
        )
        self.assertEqual(selected, GPU0)
        admitted = tl.check_idle_gpus(
            inv, selected, allowlist=allow, read_proc=self.read_proc()
        )
        self.assertEqual(admitted[0]["admission_mode"], "run_scoped_allowlist")
        self.assertEqual(admitted[0]["compute_memory_mib"], 696)
        self.assertEqual(admitted[0]["noncompute_memory_mib"], 731)
        self.assertEqual([p["pid"] for p in admitted[0]["verified_processes"]], [41001, 41002])

    def test_rejects_unknown_pid_reuse_identity_mismatch_vram_and_util_drift(self):
        allow = self.allowlist()
        # Unknown extra compute process
        unknown_inv = tl.gpu_inventory(
            self.GPU_CSV, self.APPS_CSV + "GPU-a, 99999, 100\n"
        )
        with self.assertRaisesRegex(tl.TrainingConfigError, "unapproved compute process"):
            tl.check_idle_gpus(unknown_inv, GPU0, allowlist=allow, read_proc=self.read_proc())

        # PID reuse (starttime changed), exe changed, uid changed, cgroup changed, vanished /proc
        base_inv = tl.gpu_inventory(self.GPU_CSV, self.APPS_CSV)
        for overrides, pattern in (
            ({41002: {**self.LIVE_PROCS[41002], "starttime": 99999999}}, "starttime"),
            ({41002: {**self.LIVE_PROCS[41002], "exe": "/usr/bin/python3"}}, "exe"),
            ({41002: {**self.LIVE_PROCS[41002], "uid": 0}}, "uid"),
            ({41002: {**self.LIVE_PROCS[41002], "cgroup": "0::/other.service"}}, "cgroup"),
            ({41002: None}, "cannot verify live /proc"),
        ):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(tl.TrainingConfigError, pattern):
                    tl.check_idle_gpus(
                        base_inv, GPU0, allowlist=allow, read_proc=self.read_proc(overrides)
                    )

        # Per-PID VRAM spike, total background VRAM spike, non-compute VRAM drift, util spike
        for g_csv, a_csv, pattern in (
            (self.GPU_CSV, "GPU-a, 41001, 264\nGPU-a, 41002, 900\n", "max_memory_mib"),
            ("0, GPU-a, 2100, 0\n", self.APPS_CSV, "total ceiling"),
            ("0, GPU-a, 1850, 0\n", self.APPS_CSV, "non-compute VRAM"),
            ("0, GPU-a, 1427, 12\n", self.APPS_CSV, "util 12%"),
            (self.GPU_CSV, "GPU-a, 41001\nGPU-a, 41002\n", "missing per-process VRAM"),
        ):
            with self.subTest(pattern=pattern):
                inv = tl.gpu_inventory(g_csv, a_csv)
                with self.assertRaisesRegex(tl.TrainingConfigError, pattern):
                    tl.check_idle_gpus(inv, GPU0, allowlist=allow, read_proc=self.read_proc())

        # Allowlist applies ONLY to its named GPU UUID; another GPU stays strict idle
        other_inv = tl.gpu_inventory("0, GPU-b, 1427, 0\n", "GPU-b, 41001, 264\n")
        with self.assertRaisesRegex(tl.TrainingConfigError, "compute processes"):
            tl.check_idle_gpus(
                other_inv, [{"index": 0, "uuid": "GPU-b"}], allowlist=allow, read_proc=self.read_proc()
            )

    def test_launch_pre_popen_recheck_blocks_drift_after_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            presets = Path(tmp) / "presets"
            presets.mkdir()
            preset = fake_assets(root, LOWLR)
            (presets / "test-recipe.recipe.json").write_text(json.dumps(preset))
            spec = tl.plan("test-recipe", root, presets, selection=GPU0)
            allow = self.allowlist()
            clean_env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
            }
            # Unknown process appears right before Popen -> launch must reject and never spawn.
            drifted_inv = tl.gpu_inventory(
                self.GPU_CSV, self.APPS_CSV + "GPU-a, 77777, 500\n"
            )
            with (
                patch.dict(os.environ, clean_env, clear=True),
                patch.object(tl.sys, "prefix", str(root / ".pixi" / "envs" / "default")),
                patch.object(
                    tl.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)
                ),
                patch.object(tl, "query_gpus", return_value=drifted_inv),
                patch.object(tl, "read_proc_identity", side_effect=self.read_proc()),
                patch.object(tl.subprocess, "Popen") as popen,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                rc = tl.launch(spec, root=root, allowlist=allow)
            self.assertEqual(rc, 1)
            popen.assert_not_called()
            status = json.loads((Path(spec["run_dir"]) / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("unapproved compute process", status["error"])

    def test_cli_dry_run_validates_allowlist_without_gpu_query(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            presets = root / "config/training"
            presets.mkdir(parents=True)
            bound = tl.bind_dataset(tl.load_preset(), "data/synthetic", "local/synthetic")
            preset = fake_assets(root, bound)
            del preset["native_flags"]["dataset.root"]
            del preset["native_flags"]["dataset.repo_id"]
            (presets / "training.recipe.json").write_text(json.dumps(preset))
            allow_path = Path(tmp) / "allow.json"
            allow_path.write_text(json.dumps(self.allowlist()))
            original = tl.plan
            with (
                patch.object(
                    tl,
                    "plan",
                    side_effect=lambda recipe, **kwargs: original(
                        recipe, root=root, preset_dir=presets, **kwargs
                    ),
                ),
                patch.object(tl, "query_gpus", side_effect=AssertionError("GPU query")),
                patch.dict("os.environ"),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                rc = tl.main(
                    [
                        "--dataset-root",
                        "data/synthetic",
                        "--repo-id",
                        "local/synthetic",
                        "--gpu-allowlist",
                        str(allow_path),
                        "--dry-run",
                    ]
                )
            self.assertEqual(rc, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["gpu_allowlist"]["gpu_uuid"], "GPU-a")
            self.assertEqual(report["gpu_allowlist"]["allowed_pids"], [41001, 41002])


class ExampleConfigTests(unittest.TestCase):
    def test_example_config_has_no_training_knobs_outside_presets(self):
        import re

        example = (tl.ROOT / "config" / "machine.env.example").read_text()
        keys = set(re.findall(r"^((?:TRAIN|PI05)_[A-Z_]+)=", example, re.M))
        self.assertEqual(keys, set())


if __name__ == "__main__":
    unittest.main()
