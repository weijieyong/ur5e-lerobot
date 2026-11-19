"""Offline checks of the supported LoRA training route (CPU only; no model, GPU or training)."""

import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from scripts import train_lora as tl
from scripts.train_lora_entry import load_spec

BASELINE = tl.bind_dataset(
    tl.load_preset("reference-15k"), "data/synthetic", "local/synthetic"
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
                {"native_flags__policy_DOT_train_expert_only": False},
                "train_expert_only",
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
                    "native_flags__parallelism_DOT_dp_replicate": 1,
                    "native_flags__batch_size": 8,
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
            tl.load_preset("../reference-15k")
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
        (self.presets / "reference-15k.recipe.json").write_text(json.dumps(self.preset))

    def test_plan_validates_natively_without_creating_run_directories(self):
        now = dt.datetime(2026, 11, 11, 7, 0, tzinfo=dt.timezone.utc)
        spec = tl.plan(
            "reference-15k",
            self.root,
            self.presets,
            now=now,
            hash_base=True,
            selection=TWO,
        )
        self.assertFalse((self.root / tl.RUNS_SUBDIR).exists())
        self.assertEqual(
            spec["run_dir"],
            str(self.root / tl.RUNS_SUBDIR / "reference-15k-20261111T070000.000000Z"),
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
            "reference-15k",
            self.root,
            self.presets,
            now=now + dt.timedelta(microseconds=1),
        )
        self.assertNotEqual(later["run_dir"], spec["run_dir"])

    def test_existing_run_directory_is_never_reused(self):
        now = dt.datetime(2026, 11, 11, 7, 0, tzinfo=dt.timezone.utc)
        spec = tl.plan("reference-15k", self.root, self.presets, now=now, selection=TWO)
        run, _ = tl.write_run_record(spec)
        with self.assertRaisesRegex(tl.TrainingConfigError, "reuse"):
            tl.plan("reference-15k", self.root, self.presets, now=now)
        with self.assertRaises(FileExistsError):
            tl.write_run_record(spec)
        self.assertTrue((run / "run.json").is_file())

    def test_worker_recomputes_the_launcher_argv_and_refuses_drift(self):
        spec = tl.plan("reference-15k", self.root, self.presets, selection=TWO)
        run, _ = tl.write_run_record(spec)
        loaded, preset, loaded_run, argv = load_spec(
            run / "run.json", self.root, self.presets
        )
        self.assertEqual(argv, spec["native_argv"])
        self.assertEqual(loaded_run, run)
        edited = mutated(self.preset, native_flags__steps=20000)
        (self.presets / "reference-15k.recipe.json").write_text(json.dumps(edited))
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

    def test_trainable_set_guard(self):
        names = [f"model.gemma_expert.layers.{i}.lora_A.weight" for i in range(80)]
        tl.check_trainable(names, self.preset["guards"])
        with self.assertRaises(tl.TrainingConfigError):
            tl.check_trainable(
                names[:-1] + ["model.paligemma.x.lora_A.weight"], self.preset["guards"]
            )
        with self.assertRaises(tl.TrainingConfigError):
            tl.check_trainable(names[:79], self.preset["guards"])


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

    def copy_recipe(self, filename="training-30k.recipe.json", flags=None):
        preset = copy.deepcopy(self.preset)
        preset["native_flags"].update({"steps": 30000, **(flags or {})})
        path = self.presets / filename
        path.write_text(json.dumps(preset))
        return path

    def test_default_has_reference_settings_and_resolves_project_relative_path(self):
        default = tl.load_preset()
        reference = tl.load_preset("reference-15k")
        self.assertEqual(default["name"], "training")
        self.assertEqual({**default, "name": reference["name"]}, reference)
        self.assertEqual(
            tl.resolve_recipe("config/training/training.recipe.json"),
            tl.resolve_recipe(),
        )
        spec = self.plan()
        self.assertEqual(Path(spec["preset_file"]), self.default)
        self.assertEqual(spec["preset"], "training")
        self.assertEqual(spec["effective"]["stop_budget_steps"], 15000)
        self.assertFalse((self.root / tl.RUNS_SUBDIR).exists())

    def test_copied_recipe_keeps_internal_name_and_is_revalidated_by_worker(self):
        path = self.copy_recipe()
        self.assertEqual(tl.load_preset(path)["name"], "training")
        spec = self.plan(path, selection=TWO)
        self.assertEqual(spec["preset"], "training-30k")
        self.assertEqual(spec["effective"]["stop_budget_steps"], 30000)
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
        for options, expected, steps in (
            ([], self.default, 15000),
            (["--recipe", str(path), "--smoke"], path, 4),
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
                query.return_value = tl.gpu_inventory("0, GPU-a, 15, 0\n", "")
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
                self.assertEqual(spec["gpu_selection"], GPU0)
                if options:
                    self.assertEqual(spec["_preset"]["native_flags"]["steps"], 30000)
                    run, _ = tl.write_run_record(spec)
                    self.assertEqual(
                        load_spec(run / "run.json", self.root, self.presets)[3],
                        spec["native_argv"],
                    )

    def test_cli_check_defaults_and_selects_without_gpu_access(self):
        path = self.copy_recipe()
        for options, expected, steps in (
            ([], self.default, 15000),
            (["--recipe", str(path)], path, 30000),
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
            for n in ("reference-15k",)
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
            tl.load_preset("reference-15k")["native_flags"]["policy.optimizer_lr"], 1e-4
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
                    "reference-15k", self.root, self.presets, selection=selection
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
        spec = tl.plan("reference-15k", self.root, self.presets, selection=GPU1)
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

    def select(self, apps="", gpus=None):
        return tl.select_idle_gpus(
            tl.gpu_inventory(self.GPUS if gpus is None else gpus, apps), [0, 1]
        )

    def test_idle_selection_prefers_both_and_falls_back_to_the_idle_one(self):
        self.assertEqual(self.select(), TWO)
        self.assertEqual(self.select("GPU-a, 4242\n"), GPU1)  # GPU0 busy -> GPU1 only
        self.assertEqual(self.select("GPU-b, 4242\n"), GPU0)  # GPU1 busy -> GPU0 only
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
                        "reference-15k",
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
                env["HOME"],
            ),
            ("1", "GPU-a,GPU-b", "/p/hf", "/h"),
        )

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


class ExampleConfigTests(unittest.TestCase):
    def test_example_config_has_no_training_knobs_outside_presets(self):
        import re

        example = (tl.ROOT / "config" / "machine.env.example").read_text()
        keys = set(re.findall(r"^((?:TRAIN|PI05)_[A-Z_]+)=", example, re.M))
        self.assertEqual(keys, set())


if __name__ == "__main__":
    unittest.main()
