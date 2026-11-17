"""Fail-closed loading for local PI05 LoRA deployment checkpoints."""

from pathlib import Path

from huggingface_hub.utils import HFValidationError, validate_repo_id
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def local_checkpoint_path(value: str | Path) -> Path | None:
    """Validate local directories; only strings may designate upstream Hub IDs.

    Reserve outputs/ for deployments. Other ambiguous strings retain upstream
    behavior; prefix them with ./ to request local loading. Path objects always
    mean local paths, since Path removes an explicit ./ prefix.
    """
    checkpoint = Path(value)
    hub_id = False
    if isinstance(value, str):
        try:
            validate_repo_id(value)
            hub_id = not value.startswith("outputs/")
        except HFValidationError:
            pass
    if not checkpoint.exists() and hub_id:
        return None
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory is missing: {checkpoint}")
    return checkpoint


def load_local_pi05_adapter(config):
    checkpoint = Path(config.pretrained_path)
    adapter_config = PeftConfig.from_pretrained(checkpoint, local_files_only=True)
    base_file = Path(adapter_config.base_model_name_or_path) / "model.safetensors"
    if not base_file.is_file():
        raise FileNotFoundError(f"PI05 base weights are missing: {base_file}")
    adapter_file = checkpoint / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"PI05 adapter weights are missing: {adapter_file}")

    # Upstream PI05.from_pretrained catches failures and can return random weights.
    base_weights = load_file(str(base_file))
    policy = PI05Policy(config)
    policy.load_state_dict(policy._convert_openpi_state_dict(base_weights), strict=True)
    del base_weights
    model = PeftModel(policy, adapter_config)
    result = model.load_adapter(
        checkpoint, adapter_name="default", local_files_only=True
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"PI05 adapter state mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    model.eval()
    return model


def load_rollout_policy(config, fallback):
    if config.type == "pi05" and config.use_peft and config.pretrained_path is not None:
        checkpoint = local_checkpoint_path(config.pretrained_path)
        if checkpoint is not None:
            if not (checkpoint / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"PI05 adapter configuration is missing: {checkpoint / 'adapter_config.json'}"
                )
            return load_local_pi05_adapter(config)
    return fallback(config)
