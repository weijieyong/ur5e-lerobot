"""Per-rank entry started by scripts.train_lora: fail-closed hooks around native lerobot_train.

Not an operator command. Each rank recomputes the native argv from the preset snapshot and
the declared launch-time GPU selection in run.json with the same builder as the launcher,
and refuses to start if anything (argv, world size, visible GPU UUIDs) changed.
"""

import argparse
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import traceback

from scripts.train_lora import (
    PRESET_DIR,
    ROOT,
    TrainingConfigError,
    build_native_argv,
    build_scheduler,
    check_init_adapter_config,
    check_recipe,
    check_saved_checkpoint,
    check_selection,
    torchrun_command,
    check_trainable,
    expected_save_steps,
    bind_dataset,
    load_preset,
    parse_argv,
    require,
    resolve_recipe,
    sha256,
    write_json,
)

LOADER_FAILURES = (
    "Returning model without loading pretrained weights",
    "Warning: Could not load state dict",
    "Could not load state dict from remote files",
    "Missing keys when loading state dict",
    "Unexpected keys when loading state dict",
)


def load_spec(run_json, root=ROOT, preset_dir=PRESET_DIR):
    spec = json.loads(Path(run_json).read_text())
    run = Path(spec["run_dir"])
    require(
        Path(run_json).resolve() == (run / "run.json").resolve(),
        "run identity mismatch",
    )
    preset = spec["preset_content"]
    source_preset = resolve_recipe(spec["preset_file"], preset_dir)
    require(
        sha256(source_preset) == spec["preset_sha256"]
        and bind_dataset(
            load_preset(source_preset, preset_dir), **spec["dataset_inputs"]
        )
        == preset,
        "preset file changed since launch",
    )
    check_recipe(preset)
    # Topology is recomputed from the original preset and the declared GPU selection, never
    # taken from run.json's effective flags.
    selection = check_selection(preset, spec.get("gpu_selection"))
    argv = build_native_argv(
        preset, root, run / "native", spec["smoke"], len(selection)
    )
    require(argv == spec["native_argv"], "native argv differs from the launcher's")
    require(
        spec["torchrun"][1:] == torchrun_command(len(selection), run / "run.json")[1:],
        "torchrun world size differs from the GPU selection",
    )
    return spec, preset, run, argv


class Tee(io.TextIOBase):
    def __init__(self, stream, capture):
        self.stream, self.capture = stream, capture

    def write(self, text):
        self.capture.write(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()


def check_normalization(policy_cfg):
    mapping = {
        k: getattr(v, "value", v) for k, v in policy_cfg.normalization_mapping.items()
    }
    require(
        mapping.get("STATE") == "QUANTILES" and mapping.get("ACTION") == "QUANTILES",
        "native PI05 STATE/ACTION normalization must be QUANTILES",
    )
    return mapping


def fit_training_datasets(cfg, make_datasets=None):
    """Fit actual native training rows before policy/processors; never modify recording files."""
    import numpy as np
    from lerobot.datasets.compute_stats import (
        RunningQuantileStats,
        compute_relative_action_stats,
    )
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.processor import RelativeActionsProcessorStep

    require(
        not cfg.resume and not cfg.dataset.streaming,
        "resume and streaming are unsupported",
    )
    require(0 < cfg.dataset.eval_split < 1, "a held-out development split is required")
    train, dev = (make_datasets or make_train_eval_datasets)(cfg)
    check_normalization(cfg.policy)
    require(train.meta.total_tasks == 1, "only single-task recordings are supported")
    arm = [f"joint_{i}.pos" for i in range(6)]
    # Two supported layouts: six arm joints, optionally followed by the gripper.
    has_gripper = train.meta.features["action"].get("names") != arm
    names = arm + ["gripper.pos"] if has_gripper else arm
    dim = len(names)
    for key in ("observation.state", "action"):
        feature = train.meta.features[key]
        require(
            feature.get("names") == names
            and tuple(feature["shape"]) == (dim,)
            and feature["dtype"] == "float32",
            f"invalid {key}: expected matching six joints, optionally + gripper.pos",
        )
    policy = cfg.policy
    mask = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=policy.relative_exclude_joints, action_names=names
    )._build_mask(dim)
    require(
        policy.use_relative_actions and mask == [True] * 6 + [False] * has_gripper,
        "relative actions must subtract six arm joints and keep gripper absolute",
    )
    # Selecting columns avoids image/video decoding and respects the native episode filter.
    rows = train.hf_dataset.select_columns(
        ["observation.state", "action", "episode_index"]
    ).with_format("numpy")
    arrays = {
        key: np.asarray(rows[key], dtype=np.float32)
        for key in ("observation.state", "action")
    }
    for key, values in arrays.items():
        require(
            values.ndim == 2 and values.shape[1] == dim and np.isfinite(values).all(),
            f"invalid/nonfinite training {key}",
        )
        require(
            not has_gripper or np.all((values[:, -1] >= 0) & (values[:, -1] <= 1)),
            "gripper values must be in [0,1]",
        )
    arrays["episode_index"] = np.asarray(rows["episode_index"])
    state = RunningQuantileStats()
    try:
        state.update(arrays["observation.state"])
        fitted = {
            "observation.state": state.get_statistics(),
            "action": compute_relative_action_stats(
                arrays,
                train.meta.features,
                policy.chunk_size,
                policy.relative_exclude_joints,
                num_workers=0,
            ),
        }
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"cannot fit training normalization (need usable within-episode chunks): {exc}"
        ) from exc
    require(
        all(
            np.isfinite(value).all()
            for stats in fitted.values()
            for value in stats.values()
        ),
        "nonfinite fitted normalization",
    )
    train.meta.stats = {**train.meta.stats, **fitted}
    return train, dev


def load_init_adapter(policy, adapter_dir, guards):
    """Validate and load optional warm-start LoRA weights into a freshly wrapped PEFT policy."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    weights_path = adapter_dir / "adapter_model.safetensors"
    require(weights_path.is_file(), f"init adapter missing: {weights_path}")
    adapter_name = getattr(policy, "active_adapter", "default")
    require(
        isinstance(adapter_name, str)
        and adapter_name in getattr(policy, "peft_config", {}),
        f"policy has no active PEFT adapter config {adapter_name!r}",
    )
    target_cfg = policy.peft_config[adapter_name]
    check_init_adapter_config(adapter_dir, target_cfg.r, target_cfg.lora_alpha)
    expected = get_peft_model_state_dict(policy, adapter_name=adapter_name)
    require(
        len(expected) == guards["trainable_lora_tensors"],
        f"policy adapter tensor count {len(expected)} != expected {guards['trainable_lora_tensors']}",
    )
    try:
        init = load_file(str(weights_path))
    except Exception as exc:
        raise TrainingConfigError(
            f"cannot load init adapter weights {weights_path}: {exc}"
        ) from None
    missing = sorted(set(expected) - set(init))
    extra = sorted(set(init) - set(expected))
    require(
        not missing and not extra,
        f"init adapter key mismatch (missing={missing[:5]}, extra={extra[:5]})",
    )
    for key, ref in expected.items():
        require(
            tuple(init[key].shape) == tuple(ref.shape),
            f"init adapter shape mismatch for {key!r}: got {tuple(init[key].shape)}, expected {tuple(ref.shape)}",
        )
    loaded = set_peft_model_state_dict(policy, init, adapter_name=adapter_name)
    require(not loaded.unexpected_keys, "init adapter keys do not match the LoRA")
    return loaded


def prepare_optimizer_and_scheduler(
    cfg, policy, guards, receipt, make_optimizer, root=ROOT, expert_only=True
):
    """Optionally warm-start LoRA weights, verify trainability, and build fresh optimizer/scheduler."""
    import torch

    if "init_adapter" in guards:  # previous run's LoRA weights, same LoRA config
        load_init_adapter(policy, Path(root) / guards["init_adapter"], guards)
    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    check_trainable(trainable, guards, expert_only)
    require(
        all(torch.isfinite(p).all().item() for p in trainable.values()),
        "nonfinite initial adapter",
    )
    optimizer, _scaled = make_optimizer(cfg, policy)
    require(
        isinstance(optimizer, torch.optim.AdamW) and not optimizer.state,
        "fresh native AdamW required",
    )
    actual = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    }
    require(
        actual == {id(p) for p in trainable.values()},
        "optimizer trainable set mismatch",
    )
    scheduler = build_scheduler(
        cfg.scheduler, optimizer, cfg.steps
    )  # keep decay horizon
    if "init_adapter" in guards:
        receipt["init_adapter"] = guards["init_adapter"]
    receipt.update(
        trainable_lora_tensors=len(trainable), trainable_names=list(trainable)
    )
    return optimizer, scheduler


def _cgroup_limits():
    """Read-only snapshot of this rank's cgroup and ancestor memory limits; never mutates caps."""
    try:
        line = Path("/proc/self/cgroup").read_text().strip()
    except OSError:
        return None
    rel = line.split("::", 1)[1].lstrip("/") if "::" in line else ""
    cg_root = Path("/sys/fs/cgroup")
    ancestors = []
    cur = cg_root / rel
    while True:
        entry = {"path": str(cur)}
        for name in ("memory.max", "memory.high", "memory.current", "memory.swap.max"):
            p = cur / name
            if p.is_file():
                try:
                    entry[name] = p.read_text().strip()
                except OSError:
                    pass
        ancestors.append(entry)
        if cur == cg_root or cur.parent == cur:
            break
        cur = cur.parent
    return {"self": line, "ancestors": ancestors}


def _cuda_memory_stats(rank):
    """Best-effort torch CUDA memory bytes for this rank (allocated/reserved/peaks)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(rank)),
            "reserved_bytes": int(torch.cuda.memory_reserved(rank)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(rank)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(rank)),
        }
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    rank = int(os.environ["RANK"])
    receipt = {"status": "starting", "rank": rank, "pid": os.getpid(), "saves": []}
    run = Path(json.loads(args.run.read_text())["run_dir"])
    try:
        spec, preset, run, native_argv = load_spec(args.run)
        flags, guards = parse_argv(native_argv), preset["guards"]
        world = flags["parallelism.dp_replicate"]
        require(
            int(os.environ["WORLD_SIZE"]) == world
            and int(os.environ["LOCAL_RANK"]) == rank,
            f"{world} local DDP ranks required",
        )
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES")
            == ",".join(g["uuid"] for g in spec["gpu_selection"]),
            "CUDA_VISIBLE_DEVICES differs from the selected GPU UUIDs",
        )
        from importlib.metadata import version

        import torch
        from safetensors.torch import load_file
        from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401 (registers pi05)
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        import lerobot.scripts.lerobot_train as native
        import numpy as np

        require(
            version("lerobot") == guards["lerobot_version"],
            "native LeRobot version changed",
        )
        require(
            sha256(inspect.getsourcefile(native)) == guards["native_train_sha256"],
            "native trainer changed",
        )
        base = Path(flags["policy.pretrained_path"])
        weights = base / "model.safetensors"
        require(
            weights.stat().st_size == guards["base_bytes"]
            and weights.stat().st_mtime_ns == spec["base_mtime_ns"],
            "base changed since admission",
        )
        require(not (base / "adapter_config.json").exists(), "adapter base forbidden")
        require(
            torch.cuda.device_count() == world, f"exactly {world} visible GPUs required"
        )
        torch.cuda.set_device(rank)
        require(torch.cuda.is_bf16_supported(), "BF16 required")
        receipt.update(
            gpu=torch.cuda.get_device_name(rank),
            native_argv=native_argv,
            cgroup=_cgroup_limits(),
        )
        write_json(run / f"rank-{rank}.startup.json", receipt)

        # Original base: exactly one strict, clean native load with positive evidence.
        clean_loads = 0
        original_load = PI05Policy.load_state_dict

        def strict_load(policy, values, strict=True, **kwargs):
            nonlocal clean_loads
            require(strict is True, "non-strict base load forbidden")
            result = original_load(policy, values, strict=strict, **kwargs)
            require(
                not result.missing_keys and not result.unexpected_keys,
                "base key mismatch",
            )
            clean_loads += 1
            return result

        PI05Policy.load_state_dict = strict_load
        original_policy = native.make_policy

        def checked_policy(*a, **kw):
            capture = io.StringIO()
            with (
                contextlib.redirect_stdout(Tee(sys.stdout, capture)),
                contextlib.redirect_stderr(Tee(sys.stderr, capture)),
            ):
                policy = original_policy(*a, **kw)
            text = capture.getvalue()
            (run / f"rank-{rank}.loader.log").write_text(text)
            require(
                not any(s in text for s in LOADER_FAILURES),
                "pretrained loader failed open",
            )
            require(
                "All keys loaded successfully!" in text and clean_loads == 1,
                "strict native all-keys base load evidence missing",
            )
            check_normalization(policy.config)
            receipt["strict_clean_loads"] = clean_loads
            return policy

        native.make_policy = checked_policy
        original_datasets = native.make_train_eval_datasets

        def checked_datasets(cfg):
            train, dev = fit_training_datasets(cfg, original_datasets)
            if rank == 0:
                write_json(
                    run / "normalization.json",
                    {
                        "train_ids": list(train.episodes),
                        "dev_ids": None if dev is None else list(dev.episodes),
                        "stats": {
                            k: {
                                s: np.asarray(v).tolist()
                                for s, v in train.meta.stats[k].items()
                            }
                            for k in ("observation.state", "action")
                        },
                    },
                )
            return train, dev

        native.make_train_eval_datasets = checked_datasets
        original_optimizer = native.make_optimizer_and_scheduler

        def checked_optimizer(cfg, policy):
            optimizer, scheduler = prepare_optimizer_and_scheduler(
                cfg,
                policy,
                guards,
                receipt,
                original_optimizer,
                root=ROOT,
                expert_only=flags["policy.train_expert_only"],
            )
            write_json(run / f"rank-{rank}.model.json", receipt)
            return optimizer, scheduler

        native.make_optimizer_and_scheduler = checked_optimizer
        original_save = native.save_checkpoint
        save_signature = inspect.signature(original_save)

        def checked_save(*a, **kw):
            bound = save_signature.bind(*a, **kw)
            result = original_save(*a, **kw)
            bound.arguments["accelerator"].wait_for_everyone()
            step = bound.arguments["step"]
            path = Path(bound.arguments["checkpoint_dir"])
            saved = check_saved_checkpoint(path, step, flags)
            if spec["smoke"] or step == flags["steps"]:
                tensors = load_file(
                    str(path / "pretrained_model" / "adapter_model.safetensors")
                )
                require(
                    len(tensors) == guards["trainable_lora_tensors"]
                    and all(torch.isfinite(t).all().item() for t in tensors.values()),
                    "invalid/nonfinite saved adapter",
                )
            saved["cuda_memory"] = _cuda_memory_stats(rank)
            receipt["saves"].append(saved)
            receipt["cuda_memory"] = saved["cuda_memory"]
            write_json(run / f"rank-{rank}.progress.json", receipt)
            return result

        native.save_checkpoint = checked_save
        sys.argv = ["lerobot-train", *native_argv]
        native.train()
        require(
            [s["step"] for s in receipt["saves"]]
            == expected_save_steps(flags["steps"], flags["save_freq"]),
            "native checkpoint intervals incomplete",
        )
        require(
            receipt.get("strict_clean_loads") == 1
            and receipt.get("trainable_lora_tensors")
            == guards["trainable_lora_tensors"],
            "startup proofs incomplete",
        )
        receipt["cuda_memory"] = _cuda_memory_stats(rank)
        receipt["status"] = "success"
        write_json(run / f"rank-{rank}.json", receipt)
        return 0
    except BaseException as exc:
        receipt.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            cuda_memory=_cuda_memory_stats(rank),
        )
        write_json(run / f"rank-{rank}.failure.json", receipt)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
