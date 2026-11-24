"""Fresh-base pi0.5 LoRA training with automatic train-only normalization fitting.

python -m scripts.train_lora --dataset-root data/local/pick --repo-id local/pick --dry-run

The selected recipe owns training settings; dataset inputs are explicit. Finalized raw
recordings stay unchanged. Each run starts fresh; no resume or Hub transfer.
Actual idle GPU topology is resolved at launch with global batch preserved.
"""

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "config" / "training"
RUNS_SUBDIR = Path("outputs") / "training-runs"
PATH_FLAGS = ("dataset.root", "policy.pretrained_path")

REQUIRED_FLAGS = {
    "policy.type": "pi05",
    "policy.device": "cuda",
    "policy.dtype": "bfloat16",
    "policy.use_relative_actions": True,
    "policy.relative_exclude_joints": ["gripper"],
    "policy.compile_model": False,
    "policy.push_to_hub": False,
    "peft.method_type": "LORA",
    "parallelism.dp_shard": 1,
    "accelerator.mixed_precision": "bf16",
    "dataset.streaming": False,
    "resume": False,
    "save_checkpoint": True,
    "save_checkpoint_to_hub": False,
    "trackio.enable": False,
    "env_eval_freq": 0,
    "accelerator.gradient_accumulation.steps": 1,
}
SCHEDULE_FLAGS = (
    "policy.optimizer_lr",
    "policy.scheduler_decay_lr",
    "policy.scheduler_warmup_steps",
    "policy.scheduler_decay_steps",
    "steps",
)
SMOKE_OVERRIDES = {
    "steps": 4,
    "eval_steps": 2,
    "save_freq": 2,
    "max_eval_samples": 32,
    "log_freq": 1,
}
CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "train_config.json",
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
HUB_OFFLINE = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


class TrainingConfigError(RuntimeError):
    """A preset, asset or runtime check failed; nothing was started."""


def require(ok, message):
    if not ok:
        raise TrainingConfigError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, mode="w"):
    with Path(path).open(mode) as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resolve_recipe(recipe=None, preset_dir=PRESET_DIR):
    """Resolve the default, a legacy preset name, or a project-relative/absolute recipe file."""
    selector = "training" if recipe is None else str(recipe)
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", selector):
        path = Path(preset_dir) / f"{selector}.recipe.json"
    else:
        require(
            selector.endswith(".recipe.json"), f"invalid preset or recipe {selector!r}"
        )
        path = Path(selector)
        if not path.is_absolute():
            path = ROOT / path
    available = sorted(p.name for p in Path(preset_dir).glob("*.recipe.json"))
    require(
        path.is_file(), f"unknown preset/recipe {selector!r}; available: {available}"
    )
    return path.resolve()


def recipe_label(path):
    """Use only the filename, never recipe metadata or directories, in generated output paths."""
    name = Path(path).name.removesuffix(".json").removesuffix(".recipe")
    return re.sub(r"[^a-zA-Z0-9-]+", "-", name).strip("-")[:80] or "recipe"


def load_preset(recipe=None, preset_dir=PRESET_DIR):
    path = resolve_recipe(recipe, preset_dir)
    preset = json.loads(path.read_text())
    require(
        preset.get("schema") == 1,
        f"{path}: schema mismatch",
    )
    check_recipe(preset)
    return preset


def check_recipe(preset):
    """Reject unsafe or inconsistent recipes before anything touches assets or GPUs."""
    flags, guards = preset.get("native_flags"), preset.get("guards")
    require(
        isinstance(flags, dict) and isinstance(guards, dict),
        "preset needs native_flags and guards",
    )
    require(
        "output_dir" not in flags,
        "output_dir is generated per run; remove it from the preset",
    )
    for key, value in REQUIRED_FLAGS.items():
        require(
            key in flags and flags[key] == value and type(flags[key]) is type(value),
            f"{key} must be {value!r} on the supported route, got {flags.get(key)!r}",
        )
    expert_only = flags.get("policy.train_expert_only")
    require(type(expert_only) is bool, "policy.train_expert_only must be a boolean")
    targets = flags.get("peft.target_modules")
    require(
        expert_only or (isinstance(targets, str) and targets),
        "peft.target_modules must select VLM layers when train_expert_only is false "
        "(native default targets are expert-only)",
    )
    for key in PATH_FLAGS:
        if key != "dataset.root" or key in flags:
            relative_path(flags.get(key))
    ints = (
        "steps",
        "batch_size",
        "seed",
        "save_freq",
        "eval_steps",
        "max_eval_samples",
        "log_freq",
        "num_workers",
        "peft.r",
        "parallelism.dp_replicate",
        "accelerator.gradient_accumulation.steps",
        "policy.scheduler_warmup_steps",
        "policy.scheduler_decay_steps",
        "policy.chunk_size",
        "policy.n_action_steps",
    )
    for key in ints:
        require(
            type(flags.get(key)) is int and flags[key] >= 0,
            f"{key} must be a non-negative integer",
        )
    peak, floor = (
        flags.get("policy.optimizer_lr"),
        flags.get("policy.scheduler_decay_lr"),
    )
    require(
        all(isinstance(v, (int, float)) and math.isfinite(v) for v in (peak, floor)),
        "LR must be finite",
    )
    require(
        0 < floor < peak <= 1e-3, f"need 0 < floor ({floor}) < peak ({peak}) <= 1e-3"
    )
    warmup, decay, steps = (
        flags["policy.scheduler_warmup_steps"],
        flags["policy.scheduler_decay_steps"],
        flags["steps"],
    )
    require(
        0 < warmup < decay, "warmup must be positive and shorter than the decay horizon"
    )
    require(warmup < steps, "stop budget must extend past warmup")
    require(flags["save_freq"] > 0, "save_freq must be positive")
    require(
        type(flags.get("dataset.image_transforms.enable")) is bool,
        "dataset.image_transforms.enable must be a boolean",
    )
    require(
        flags["peft.r"] > 0
        and isinstance(flags.get("peft.lora_alpha"), (int, float))
        and flags["peft.lora_alpha"] > 0,
        "LoRA rank/alpha must be positive",
    )
    require(
        0 < flags["policy.n_action_steps"] <= flags["policy.chunk_size"],
        "n_action_steps must fit the chunk",
    )
    world, accum, per_rank = (
        flags["parallelism.dp_replicate"],
        flags["accelerator.gradient_accumulation.steps"],
        flags["batch_size"],
    )
    require(
        world >= 1 and accum >= 1 and per_rank >= 1,
        "batch topology values must be positive",
    )
    require(
        per_rank * world * accum == guards.get("global_batch"),
        f"batch topology {per_rank} x {world} ranks x {accum} accumulation != global_batch "
        f"{guards.get('global_batch')}",
    )
    gpus = guards.get("gpus")
    require(
        isinstance(gpus, list)
        and len(gpus) >= world
        and len(set(gpus)) == len(gpus)
        and all(type(g) is int and g >= 0 for g in gpus),
        "guards.gpus must list distinct eligible GPUs for the requested ranks",
    )
    split = flags.get("dataset.eval_split")
    require(
        isinstance(split, float) and 0 < split < 1,
        "dataset.eval_split must be in (0, 1)",
    )
    for key in (
        "lerobot_version",
        "native_train_sha256",
        "base_revision",
        "base_sha256",
    ):
        require(
            isinstance(guards.get(key), str) and guards[key],
            f"guards.{key} is required",
        )
    require(
        Path(flags["policy.pretrained_path"]).name == guards["base_revision"],
        "base path must be the approved original-base snapshot revision",
    )
    if "init_adapter" in guards:
        relative_path(guards["init_adapter"])
    return preset


def relative_path(value):
    require(isinstance(value, str) and value, "path flag must be a non-empty string")
    path = Path(value)
    require(
        not path.is_absolute() and ".." not in path.parts,
        f"preset paths must be project-relative, got {value!r}",
    )
    return path


def format_flag(key, value):
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    return f"--{key}={text}"


def resolve_topology(preset, gpu_count):
    """Execution topology for gpu_count selected GPUs; the global batch and accumulation are fixed.

    Gradient accumulation is never used to absorb a missing GPU: native 0.7 counts optimizer
    and scheduler steps per microbatch, which would change the update/sample budget.
    """
    flags, guards = preset["native_flags"], preset["guards"]
    global_batch, accum = (
        guards["global_batch"],
        flags["accelerator.gradient_accumulation.steps"],
    )
    require(
        type(gpu_count) is int
        and 1 <= gpu_count <= min(len(guards["gpus"]), flags["parallelism.dp_replicate"]),
        f"{gpu_count} GPUs exceeds the recipe's requested ranks or eligible GPUs",
    )
    require(accum == 1, "topology derivation requires gradient accumulation 1")
    require(
        global_batch % gpu_count == 0,
        f"global batch {global_batch} not divisible over {gpu_count} GPUs",
    )
    return {
        "batch_size": global_batch // gpu_count,
        "parallelism.dp_replicate": gpu_count,
        "accelerator.gradient_accumulation.steps": accum,
    }


def check_selection(preset, selection):
    """A launch-time GPU selection: unique eligible indices with distinct physical UUIDs."""
    eligible = preset["guards"]["gpus"]
    require(
        isinstance(selection, list) and selection,
        "GPU selection must be a non-empty list",
    )
    for gpu in selection:
        require(
            isinstance(gpu, dict)
            and set(gpu) == {"index", "uuid"}
            and type(gpu["index"]) is int
            and isinstance(gpu["uuid"], str)
            and gpu["uuid"].startswith("GPU-"),
            f"malformed GPU selection {gpu!r}",
        )
        require(
            gpu["index"] in eligible,
            f"GPU {gpu['index']} is not in the preset's eligible GPUs {eligible}",
        )
    indices, uuids = [g["index"] for g in selection], [g["uuid"] for g in selection]
    require(
        indices == sorted(set(indices)) and len(set(uuids)) == len(uuids),
        "duplicate/unsorted GPU selection",
    )
    return selection


def build_native_argv(preset, root, native_output, smoke=False, gpu_count=None):
    """The single argv builder used by --dry-run, run.json and every rank.

    gpu_count None keeps the preset's nominal topology (dry-run display); otherwise the
    launch-time topology from resolve_topology replaces only the three topology flags.
    """
    flags = dict(preset["native_flags"])
    if gpu_count is not None:
        flags.update(resolve_topology(preset, gpu_count))
    for key in PATH_FLAGS:
        flags[key] = str(Path(root) / relative_path(flags[key]))
    if smoke:
        flags.update(SMOKE_OVERRIDES)
    flags["output_dir"] = str(native_output)
    return [format_flag(key, value) for key, value in flags.items()]


def parse_argv(argv):
    """Inverse of build_native_argv, for receipts and tests."""
    flags = {}
    for item in argv:
        require(
            item.startswith("--") and "=" in item, f"malformed native flag {item!r}"
        )
        key, value = item[2:].split("=", 1)
        require(key not in flags, f"duplicate native flag {key}")
        try:
            flags[key] = json.loads(value)
        except json.JSONDecodeError:
            flags[key] = value
    return flags


def planned_run_dir(root, name, smoke=False, now=None):
    require(
        re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]*", name or "") is not None,
        f"invalid run label {name!r}",
    )
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    run = Path(root) / RUNS_SUBDIR / f"{name}-{stamp}{'-smoke' if smoke else ''}"
    require(not run.exists(), f"refusing to reuse existing run directory {run}")
    return run


def build_scheduler(scheduler_cfg, optimizer, stop_steps):
    """Native cosine schedule with the decay horizon kept independent of the stop budget.

    LeRobot 0.7.0 compresses warmup and decay into --steps whenever steps < decay steps.
    Building with max(stop, decay) keeps the preset's warmup/decay endpoint; when the
    stop budget is at least the decay horizon this is exactly the native call.
    """
    require(
        type(scheduler_cfg).__name__ == "CosineDecayWithWarmupSchedulerConfig",
        f"unexpected native scheduler {type(scheduler_cfg).__name__}",
    )
    return scheduler_cfg.build(
        optimizer, max(stop_steps, scheduler_cfg.num_decay_steps)
    )


def expected_lr(flags, step):
    peak, floor = flags["policy.optimizer_lr"], flags["policy.scheduler_decay_lr"]
    warmup, decay = (
        flags["policy.scheduler_warmup_steps"],
        flags["policy.scheduler_decay_steps"],
    )
    if step < warmup:
        return peak * ((1 / (warmup + 1) - 1) * (1 - step / warmup) + 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(step, decay) / decay))
    return floor + (peak - floor) * cosine


def schedule_samples(scheduler_cfg, stop_steps, steps):
    """LR the native LambdaLR yields at given steps, on a CPU dummy parameter."""
    import torch

    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.zeros(1))], lr=scheduler_cfg.peak_lr
    )
    scheduler = build_scheduler(scheduler_cfg, optimizer, stop_steps)
    return {
        step: scheduler.base_lrs[0] * scheduler.lr_lambdas[0](step) for step in steps
    }


def check_runtime(guards):
    from importlib.metadata import version
    import importlib.util

    require(
        version("lerobot") == guards["lerobot_version"],
        f"native LeRobot {guards['lerobot_version']} required",
    )
    source = Path(importlib.util.find_spec("lerobot.scripts.lerobot_train").origin)
    require(
        sha256(source) == guards["native_train_sha256"],
        f"native trainer source changed: {source}",
    )
    return {"lerobot": version("lerobot"), "native_train": str(source)}


STANDARD_INIT_LORA_WEIGHTS = ("gaussian", "orthogonal")


def check_init_lora_weights(value):
    """Allow only ordinary non-base-mutating LoRA initializers on a pristine base model."""
    ok = isinstance(value, bool) or (
        isinstance(value, str) and value in STANDARD_INIT_LORA_WEIGHTS
    )
    require(
        ok,
        f"init adapter init_lora_weights={value!r} is incompatible with a pristine base model",
    )


def check_init_adapter_config(adapter_dir, r, lora_alpha):
    """Check that an optional init_adapter uses compatible standard LoRA settings."""
    path = Path(adapter_dir) / "adapter_config.json"
    require(path.is_file(), f"init adapter config missing: {path}")
    try:
        cfg = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise TrainingConfigError(f"invalid init adapter config {path}: {exc}") from None
    require(isinstance(cfg, dict), f"invalid init adapter config {path}")
    alpha = cfg.get("lora_alpha")
    require(
        cfg.get("peft_type") == "LORA"
        and type(cfg.get("r")) is int
        and cfg["r"] == r
        and type(alpha) in (int, float)
        and math.isfinite(alpha)
        and math.isclose(float(alpha), float(lora_alpha), rel_tol=0.0, abs_tol=1e-12),
        f"init adapter config incompatible with LoRA r={r}, alpha={lora_alpha}",
    )
    require(
        not cfg.get("use_rslora") and not cfg.get("alpha_pattern"),
        "init adapter scaling (use_rslora/alpha_pattern) is incompatible with the supported LoRA route",
    )
    check_init_lora_weights(cfg.get("init_lora_weights", True))
    return cfg


def check_assets(preset, root, hash_base=False):
    """Approved native dataset/stats and original base; cheap unless hash_base."""
    flags, guards = preset["native_flags"], preset["guards"]
    data = Path(root) / flags["dataset.root"]
    info_path = data / "meta" / "info.json"
    require(info_path.is_file(), f"dataset metadata missing: {info_path}")
    info = json.loads(info_path.read_text())
    require(
        str(info.get("codebase_version", "")).startswith("v3"),
        "native LeRobot v3 dataset required",
    )
    require(
        type(info.get("total_episodes")) is int
        and info["total_episodes"] >= 2
        and info.get("total_frames", 0) > 0,
        "dataset needs finalized episodes and frames",
    )
    require(
        type(info.get("fps")) in (int, float) and info["fps"] == 15,
        "training requires a 15 Hz dataset; inspect meta/info.json fps",
    )
    base = Path(root) / flags["policy.pretrained_path"]
    weights = base / "model.safetensors"
    require(
        not (base / "adapter_config.json").exists(),
        "original base required, not an adapter checkpoint",
    )
    require(
        (base / "config.json").is_file() and weights.is_file(),
        f"base snapshot incomplete: {base}",
    )
    require(weights.stat().st_size == guards["base_bytes"], "base model size mismatch")
    summary = {
        "dataset": str(data),
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "base": str(base),
        "base_bytes": guards["base_bytes"],
        "base_sha256": "deferred to launch",
    }
    if "init_adapter" in guards:
        adapter_dir = Path(root) / guards["init_adapter"]
        adapter = adapter_dir / "adapter_model.safetensors"
        require(adapter.is_file(), f"init adapter missing: {adapter}")
        check_init_adapter_config(adapter_dir, flags["peft.r"], flags["peft.lora_alpha"])
        summary["init_adapter"] = str(adapter)
    if hash_base:
        require(sha256(weights) == guards["base_sha256"], "base model SHA256 mismatch")
        summary["base_sha256"] = guards["base_sha256"]
    return summary


def check_trainable(names, guards, expert_only=True):
    """Exactly the expected LoRA tensors; VLM adapters only when the recipe opts in."""
    names = list(names)
    require(
        len(names) == guards["trainable_lora_tensors"]
        and all("lora_" in n for n in names),
        f"expected exactly {guards['trainable_lora_tensors']} LoRA-only tensors, got {len(names)}",
    )
    vlm = any(".paligemma." in n for n in names)
    if expert_only:
        require(not vlm, "expected expert-side LoRA tensors and a frozen VLM")
    else:
        require(
            vlm and any(".gemma_expert." in n for n in names),
            "expected LoRA tensors in both the VLM and the action expert",
        )


def expected_save_steps(steps, save_freq):
    return sorted(set(range(save_freq, steps + 1, save_freq)) | {steps})


def check_saved_checkpoint(checkpoint_dir, step, flags):
    """Native save completed with processors, tokenizer, adapter and the unscaled schedule."""
    checkpoint_dir = Path(checkpoint_dir)
    model, state = (
        checkpoint_dir / "pretrained_model",
        checkpoint_dir / "training_state",
    )
    for name in CHECKPOINT_FILES:
        require(
            (model / name).is_file() and (model / name).stat().st_size > 0,
            f"missing saved {name}",
        )
    require(
        json.loads((state / "training_step.json").read_text())["step"] == step,
        "saved step mismatch",
    )
    saved = json.loads((model / "train_config.json").read_text())
    require(
        saved["steps"] == flags["steps"]
        and saved["batch_size"] == flags["batch_size"]
        and saved["resume"] is False
        and saved["seed"] == flags["seed"],
        "saved native configuration mismatch",
    )
    scheduler = saved["scheduler"]
    require(
        (
            scheduler["peak_lr"],
            scheduler["decay_lr"],
            scheduler["num_warmup_steps"],
            scheduler["num_decay_steps"],
        )
        == (
            flags["policy.optimizer_lr"],
            flags["policy.scheduler_decay_lr"],
            flags["policy.scheduler_warmup_steps"],
            flags["policy.scheduler_decay_steps"],
        ),
        "saved scheduler differs from the preset",
    )
    adapter = json.loads((model / "adapter_config.json").read_text())
    require(
        (adapter["r"], adapter["lora_alpha"])
        == (flags["peft.r"], flags["peft.lora_alpha"]),
        "saved LoRA rank/alpha mismatch",
    )
    last_lr = json.loads((state / "scheduler_state.json").read_text())["_last_lr"][0]
    require(
        math.isclose(last_lr, expected_lr(flags, step), rel_tol=1e-6),
        f"saved LR {last_lr} at step {step} is not the preset schedule {expected_lr(flags, step)}",
    )
    return {"step": step, "path": str(checkpoint_dir), "lr": last_lr}


def validate_native(argv):
    """Parse and validate the actual native TrainPipelineConfig without starting training."""
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401 (registers pi05)

    @parser.wrap()
    def parse(cfg: TrainPipelineConfig):
        cfg.validate()
        return cfg

    saved = sys.argv
    sys.argv = ["lerobot-train", *argv]
    try:
        return parse()
    except SystemExit as exc:
        raise TrainingConfigError(
            f"native configuration rejected (exit {exc.code})"
        ) from None
    except (ValueError, TypeError, FileExistsError, NotImplementedError) as exc:
        raise TrainingConfigError(
            f"native validation failed: {type(exc).__name__}: {exc}"
        ) from None
    finally:
        sys.argv = saved


def effective_settings(cfg, flags):
    """Settings as parsed by the native config; fail if any differ from the preset."""
    world, accum = (
        cfg.parallelism.dp_replicate,
        cfg.accelerator.gradient_accumulation.steps,
    )
    effective = {
        "peak_lr": cfg.scheduler.peak_lr,
        "floor_lr": cfg.scheduler.decay_lr,
        "optimizer_lr": cfg.optimizer.lr,
        "warmup_steps": cfg.scheduler.num_warmup_steps,
        "decay_horizon_steps": cfg.scheduler.num_decay_steps,
        "stop_budget_steps": cfg.steps,
        "per_rank_batch": cfg.batch_size,
        "world_size": world,
        "grad_accumulation": accum,
        "global_batch": cfg.batch_size * world * accum,
        "seed": cfg.seed,
        "eval_steps": cfg.eval_steps,
        "save_freq": cfg.save_freq,
        "max_eval_samples": cfg.max_eval_samples,
        "eval_split": cfg.dataset.eval_split,
        "dataset_repo_id": cfg.dataset.repo_id,
        "dataset_root": str(cfg.dataset.root),
        "base": str(cfg.policy.pretrained_path),
        "output_dir": str(cfg.output_dir),
        "resume": cfg.resume,
        "lora_r": cfg.peft.r,
        "lora_alpha": cfg.peft.lora_alpha,
        "peft_method": str(cfg.peft.method_type),
        "train_expert_only": cfg.policy.train_expert_only,
        "freeze_vision_encoder": cfg.policy.freeze_vision_encoder,
        "lora_target_modules": cfg.peft.target_modules,
        "use_relative_actions": cfg.policy.use_relative_actions,
        "relative_exclude_joints": list(cfg.policy.relative_exclude_joints),
        "chunk_size": cfg.policy.chunk_size,
        "dtype": str(cfg.policy.dtype),
        "device_requested": flags["policy.device"],
        "mixed_precision": cfg.accelerator.mixed_precision,
        "gradient_checkpointing": cfg.policy.gradient_checkpointing,
        "image_augmentation": cfg.dataset.image_transforms.enable,
        "policy_push_to_hub": cfg.policy.push_to_hub,
        "save_checkpoint_to_hub": cfg.save_checkpoint_to_hub,
    }
    expected = {
        "peak_lr": flags["policy.optimizer_lr"],
        "optimizer_lr": flags["policy.optimizer_lr"],
        "floor_lr": flags["policy.scheduler_decay_lr"],
        "warmup_steps": flags["policy.scheduler_warmup_steps"],
        "decay_horizon_steps": flags["policy.scheduler_decay_steps"],
        "stop_budget_steps": flags["steps"],
        "per_rank_batch": flags["batch_size"],
        "world_size": flags["parallelism.dp_replicate"],
        "grad_accumulation": flags["accelerator.gradient_accumulation.steps"],
        "seed": flags["seed"],
        "eval_steps": flags["eval_steps"],
        "save_freq": flags["save_freq"],
        "max_eval_samples": flags["max_eval_samples"],
        "resume": False,
        "lora_r": flags["peft.r"],
        "lora_alpha": flags["peft.lora_alpha"],
        "train_expert_only": flags["policy.train_expert_only"],
        "freeze_vision_encoder": flags.get("policy.freeze_vision_encoder", False),
        "lora_target_modules": flags.get("peft.target_modules"),
        "dataset_root": flags["dataset.root"],
        "base": flags["policy.pretrained_path"],
        "output_dir": flags["output_dir"],
        "policy_push_to_hub": False,
        "save_checkpoint_to_hub": False,
        "image_augmentation": flags["dataset.image_transforms.enable"],
    }
    for key, value in expected.items():
        require(
            effective[key] == value,
            f"native config {key}={effective[key]!r} differs from preset {value!r}",
        )
    require(not Path(cfg.output_dir).exists(), "native output directory already exists")
    return effective


def gpu_inventory(gpu_csv, apps_csv):
    """Parse nvidia-smi CSV (index,uuid,memory MiB,util %) and (gpu_uuid,pid); fail closed."""
    inventory = {}
    try:
        for line in filter(str.strip, gpu_csv.splitlines()):
            index, uuid, memory, util = (c.strip() for c in line.split(","))
            require(
                uuid.startswith("GPU-")
                and uuid not in {g["uuid"] for g in inventory.values()}
                and int(index) not in inventory,
                f"malformed GPU inventory row {line!r}",
            )
            inventory[int(index)] = {
                "index": int(index),
                "uuid": uuid,
                "memory_mib": int(memory),
                "util_pct": int(util),
                "pids": [],
            }
        by_uuid = {g["uuid"]: g for g in inventory.values()}
        for line in filter(str.strip, apps_csv.splitlines()):
            uuid, pid = (c.strip() for c in line.split(","))
            require(uuid in by_uuid, f"compute process on unknown GPU {line!r}")
            by_uuid[uuid]["pids"].append(int(pid))
    except ValueError as exc:
        raise TrainingConfigError(f"malformed nvidia-smi output: {exc}") from None
    require(inventory, "nvidia-smi reported no GPUs")
    return inventory


def is_idle(gpu):
    """Any compute process makes a GPU busy regardless of utilization; foreign jobs are never touched."""
    return not gpu["pids"] and gpu["memory_mib"] < 512 and gpu["util_pct"] == 0


def select_idle_gpus(inventory, eligible, limit=None):
    """Select up to limit idle eligible GPUs in index order; refuse if none."""
    present = [g for g in sorted(eligible) if g in inventory]
    require(present, f"none of the eligible GPUs {sorted(eligible)} is present")
    idle = [
        {"index": g, "uuid": inventory[g]["uuid"]}
        for g in present
        if is_idle(inventory[g])
    ]
    require(
        idle,
        "no idle eligible GPU: "
        + "; ".join(
            f"GPU {g}: pids={inventory[g]['pids']} mem={inventory[g]['memory_mib']}MiB util={inventory[g]['util_pct']}%"
            for g in present
        )
        + " (not starting, not waiting)",
    )
    return idle[:limit]


def check_idle_gpus(inventory, selection):
    """Revalidate the selection: same physical UUID at each index, still idle."""
    rows = []
    for gpu in selection:
        row = inventory.get(gpu["index"])
        require(
            row is not None and row["uuid"] == gpu["uuid"],
            f"GPU {gpu['index']} identity changed since selection",
        )
        require(
            not row["pids"],
            f"selected GPU {gpu['index']} already has compute processes: {row['pids']}",
        )
        require(is_idle(row), f"selected GPU {gpu['index']} is not idle: {row}")
        rows.append(row)
    return rows


def query_gpus():
    def query(*args):
        return subprocess.check_output(
            ["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True
        )

    return gpu_inventory(
        query("--query-gpu=index,uuid,memory.used,utilization.gpu"),
        query("--query-compute-apps=gpu_uuid,pid"),
    )


def child_environment(root, selection, base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    for key in list(env):
        if key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH") or key.startswith(
            "ACCELERATE_"
        ):
            env.pop(key)
    env.update(
        HUB_OFFLINE,
        CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in selection),
        HF_HOME=str(Path(root) / "hf"),
        HF_HUB_CACHE=str(Path(root) / "hf" / "hub"),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        TMPDIR=str(Path(root) / ".tmp" / "training"),
        OMP_NUM_THREADS="1",
    )
    return env


def torchrun_command(world, run_json):
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={world}",
        "--max_restarts=0",
        "-m",
        "scripts.train_lora_entry",
        "--run",
        str(run_json),
    ]


def bind_dataset(preset, dataset_root=None, repo_id=None):
    preset = json.loads(json.dumps(preset))
    flags = preset["native_flags"]
    if dataset_root is not None:
        flags["dataset.root"] = str(relative_path(str(dataset_root)))
    if repo_id is not None:
        flags["dataset.repo_id"] = repo_id
    require(
        flags.get("dataset.root") and flags.get("dataset.repo_id"),
        "explicit --dataset-root and --repo-id are required",
    )
    check_recipe(preset)
    return preset


def plan(
    name=None,
    root=ROOT,
    preset_dir=PRESET_DIR,
    smoke=False,
    now=None,
    hash_base=False,
    selection=None,
    select=None,
    dataset_root=None,
    repo_id=None,
):
    """Everything a run needs, validated, without creating any directory.

    selection: launch-time idle GPUs ([{index, uuid}]); select(preset) returns one after the
    preset is loaded. None for both is the nominal preset topology (dry-run display only).
    """
    source_preset = resolve_recipe(name, preset_dir)
    preset = bind_dataset(load_preset(source_preset, preset_dir), dataset_root, repo_id)
    if select is not None:
        selection = select(preset)
    if selection is not None:
        check_selection(preset, selection)
    label = recipe_label(source_preset)
    run = planned_run_dir(root, label, smoke, now)
    argv = build_native_argv(
        preset,
        root,
        run / "native",
        smoke,
        None if selection is None else len(selection),
    )
    runtime = check_runtime(preset["guards"])
    assets = check_assets(preset, root, hash_base)
    flags = parse_argv(argv)
    cfg = validate_native(argv)
    effective = effective_settings(cfg, flags)
    lrs = schedule_samples(
        cfg.scheduler,
        flags["steps"],
        sorted(
            {
                flags["policy.scheduler_warmup_steps"],
                flags["steps"],
                flags["policy.scheduler_decay_steps"],
            }
        ),
    )
    for step, lr in lrs.items():
        require(
            math.isclose(lr, expected_lr(flags, step), rel_tol=1e-9),
            f"schedule mismatch at step {step}",
        )
    return {
        "preset": label,
        "preset_file": str(source_preset),
        "preset_sha256": sha256(source_preset),
        "root": str(root),
        "dataset_inputs": {
            "dataset_root": preset["native_flags"]["dataset.root"],
            "repo_id": preset["native_flags"]["dataset.repo_id"],
        },
        "python": sys.executable,
        "mode": "smoke" if smoke else "train",
        "run_dir": str(run),
        "effective": effective,
        "lr_at_step": {str(k): v for k, v in lrs.items()},
        "assets": assets,
        "runtime": runtime,
        "native_argv": argv,
        "torchrun": torchrun_command(effective["world_size"], run / "run.json"),
        "gpu_selection": selection,
        "topology": topology_record(preset, effective, selection),
        "_preset": preset,
    }


def topology_record(preset, effective, selection):
    flags = preset["native_flags"]
    return {
        "status": "nominal preset topology; actual topology auto-resolved at launch"
        if selection is None
        else "resolved at launch from idle eligible GPUs",
        "eligible_gpus": preset["guards"]["gpus"],
        "selected_indices": None
        if selection is None
        else [g["index"] for g in selection],
        "selected_uuids": None if selection is None else [g["uuid"] for g in selection],
        "nominal": {
            "per_rank_batch": flags["batch_size"],
            "world_size": flags["parallelism.dp_replicate"],
            "grad_accumulation": flags["accelerator.gradient_accumulation.steps"],
        },
        "effective": {
            k: effective[k]
            for k in (
                "per_rank_batch",
                "world_size",
                "grad_accumulation",
                "global_batch",
            )
        },
    }


def stop_owned_group(proc):
    """Stop only the process group this launcher created (TERM, then KILL after 30 s)."""
    import signal

    for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def write_run_record(spec):
    """Create the fresh run directory and run.json that every rank re-verifies."""
    run = Path(spec["run_dir"])
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir(exist_ok=False)
    base = Path(spec["assets"]["base"]) / "model.safetensors"
    record = {k: v for k, v in spec.items() if k != "_preset"}
    record.update(
        preset_content=spec["_preset"],
        smoke=spec["mode"] == "smoke",
        base_mtime_ns=base.stat().st_mtime_ns,
    )
    write_json(run / "run.json", record, "x")
    return run, record


@contextlib.contextmanager
def reserve_gpus(runs, selection):
    """Hold nonblocking per-UUID flock reservations for a run's lifetime; never wait or unlink."""
    import fcntl

    lock_dir = Path(runs) / ".gpu-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        for uuid in sorted({g["uuid"] for g in selection}):
            require(
                isinstance(uuid, str) and re.fullmatch(r"GPU-[A-Za-z0-9_-]+", uuid),
                f"malformed GPU UUID {uuid!r}",
            )
            handle = stack.enter_context((lock_dir / f"{uuid}.lock").open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TrainingConfigError(
                    f"selected GPU {uuid} is already reserved by an active launch "
                    "(not starting, not waiting)"
                ) from None
            stack.callback(fcntl.flock, handle, fcntl.LOCK_UN)
        yield


def launch(spec, root=ROOT):
    """Operator-only: admission, fresh run directory, native training on the selected GPUs."""
    import fcntl

    require(
        Path(sys.prefix).resolve()
        == (Path(root) / ".pixi" / "envs" / "default").resolve(),
        "use the project's installed Pixi environment",
    )
    require(
        not any(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")),
        "nested torchrun forbidden",
    )
    require(spec["gpu_selection"], "launch requires a launch-time GPU selection")
    runs = Path(root) / RUNS_SUBDIR
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / ".launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with reserve_gpus(runs, spec["gpu_selection"]):
            require(
                shutil.disk_usage(runs).free >= 10 * 1024**3,
                "at least 10 GiB free required",
            )
            check_assets(spec["_preset"], root, hash_base=True)
            run, record = write_run_record(spec)
            (Path(root) / ".tmp" / "training").mkdir(parents=True, exist_ok=True)
            status = {
                "status": "preflight",
                "controller_pid": os.getpid(),
                "command": spec["torchrun"],
            }
            proc = None
            try:
                print(run, flush=True)
                with (run / "train.out").open("x") as log:
                    status["gpu_admission"] = check_idle_gpus(
                        query_gpus(), spec["gpu_selection"]
                    )
                    proc = subprocess.Popen(
                        spec["torchrun"],
                        cwd=root,
                        env=child_environment(root, spec["gpu_selection"]),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    status.update(status="running", torchrun_pid=proc.pid)
                    write_json(run / "status.json", status)
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    status["returncode"] = proc.wait(
                        timeout=900 if record["smoke"] else None
                    )
                require(
                    status["returncode"] == 0,
                    f"torchrun exited {status['returncode']}; inspect train.out",
                )
                for rank in range(spec["effective"]["world_size"]):
                    receipt = json.loads((run / f"rank-{rank}.json").read_text())
                    require(
                        receipt["status"] == "success", f"rank {rank} did not complete"
                    )
                status["status"] = "success"
                return 0
            except BaseException as exc:
                if proc is not None and proc.poll() is None:
                    stop_owned_group(proc)
                status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"FAILED: {exc}; evidence: {run}", file=sys.stderr)
                return 1
            finally:
                write_json(run / "status.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "preset", nargs="?", help="recipe name, e.g. training-2gpu"
    )
    parser.add_argument(
        "--recipe",
        help="recipe JSON path (default: config/training/training.recipe.json)",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="finalized local recording, relative to project root",
    )
    parser.add_argument(
        "--repo-id", required=True, help="local dataset identity, e.g. local/pick-tray"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only native configuration check; creates nothing",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="4-update GPU mechanics gate with eval/save at 2 and 4",
    )
    args = parser.parse_args(argv)
    if args.preset is not None and args.recipe is not None:
        parser.error("choose either a legacy preset name or --recipe, not both")
    recipe = args.recipe if args.recipe is not None else args.preset
    try:
        if args.dry_run:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ.update(HUB_OFFLINE)
        if args.dry_run:
            spec = plan(recipe, dataset_root=args.dataset_root, repo_id=args.repo_id)
            preset = spec["_preset"]
            spec["topology_candidates"] = {}
            for count in range(
                1, preset["native_flags"]["parallelism.dp_replicate"] + 1
            ):
                try:
                    argv = build_native_argv(
                        preset, ROOT, Path(spec["run_dir"]) / "native", False, count
                    )
                    effective = effective_settings(
                        validate_native(argv), parse_argv(argv)
                    )
                    spec["topology_candidates"][str(count)] = topology_record(
                        preset, effective, None
                    )["effective"]
                except TrainingConfigError as exc:
                    spec["topology_candidates"][str(count)] = f"unsupported: {exc}"
        else:
            spec = plan(
                recipe,
                smoke=args.smoke,
                dataset_root=args.dataset_root,
                repo_id=args.repo_id,
                select=lambda preset: select_idle_gpus(
                    query_gpus(),
                    preset["guards"]["gpus"],
                    limit=preset["native_flags"]["parallelism.dp_replicate"],
                ),
            )
    except TrainingConfigError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        report = {k: v for k, v in spec.items() if k != "_preset"}
        report.update(
            validated="native TrainPipelineConfig parsed and validated on CPU",
            training_started=False,
            run_dir_created=False,
            gpu_admission="at launch only: idle eligible GPUs selected, topology derived from count",
        )
        print(json.dumps(report, indent=2))
        return 0
    import signal

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    return launch(spec)


if __name__ == "__main__":
    sys.exit(main())
