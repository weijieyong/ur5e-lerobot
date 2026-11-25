# Training and deployment

Use the [operating guide](operating-guide.md) to record and review demonstrations.
Copy the complete native v3 dataset (`meta`, `data`, `videos`) to the training host,
checking transfer hashes. Keep original bytes immutable. The training launcher
never downloads anything; fetch the base model and tokenizer once, below.

## Model downloads

`HF_HOME` in `config/local/machine.env` controls the download/auth cache and defaults
to `$PIXI_PROJECT_ROOT/hf`. The launcher always reads `$PIXI_PROJECT_ROOT/hf/hub`;
changing `HF_HOME` does not relocate the recipe's base snapshot or training cache.
Download them on the training machine; the robot PC uses the deployment copies
described below.

| Asset | Repo | Access |
| --- | --- | --- |
| pi0.5 base weights (14.5 GB) | [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | Public |
| PaliGemma tokenizer | [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) | Gated |

> [!IMPORTANT]
> Before downloading the tokenizer, open its repo page, accept the Gemma license and wait
> for approval. Without it the download fails with `Access denied`.

For the first download, set `HF_HUB_OFFLINE=0` in `config/local/machine.env`; it applies to
every `pixi run`, so a `HF_HUB_OFFLINE=0 pixi run ...` prefix has no effect.

Log in with a Hugging Face [access token](https://huggingface.co/settings/tokens) (read scope).
With the default `HF_HOME`, the token is saved under the project's git-ignored `hf/` folder:

```sh
pixi run hf auth login
```

From the project root, download both assets explicitly into the training cache:

```sh
pixi run hf download lerobot/pi05_base --cache-dir ./hf/hub --revision 22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0
pixi run hf download google/paligemma-3b-pt-224 --cache-dir ./hf/hub \
  config.json tokenizer.json tokenizer.model tokenizer_config.json \
  special_tokens_map.json added_tokens.json
```

> [!TIP]
> Set `HF_HUB_OFFLINE=1` again afterwards. Training and rollout load only local files
> either way, so forgetting only affects LeRobot commands you run by hand.

The revision is the snapshot `config/training/training.recipe.json` expects;
`pixi run train-check` confirms the base snapshot is in place.

## Filter saved episodes

Inspect the original dataset, then write a filtered copy with LeRobot 0.7.0's
`delete_episodes` operation. Example: reject episode 1 from `local/new-recording`.
Substitute your dataset identity, root and rejected indices. Start with a fresh output
path; keep the source unchanged. Filtering renumbers the retained episodes.

```sh
pixi run lerobot-edit-dataset --repo_id=local/new-recording --root=./data/local/new-recording --operation.type=info --operation.show_features=true --push_to_hub=false
pixi run lerobot-dataset-viz --repo-id local/new-recording --root ./data/local/new-recording --episode-index 1
# Destination must not exist. This writes a copy, leaving the original intact.
test ! -e ./data/local/new-recording-filtered && pixi run lerobot-edit-dataset \
  --repo_id=local/new-recording --root=./data/local/new-recording \
  --new_repo_id=local/new-recording-filtered --new_root=./data/local/new-recording-filtered \
  --operation.type=delete_episodes --operation.episode_indices='[1]' --push_to_hub=false
pixi run lerobot-edit-dataset --repo_id=local/new-recording-filtered --root=./data/local/new-recording-filtered --operation.type=info --operation.show_features=true --push_to_hub=false
pixi run lerobot-dataset-viz --repo-id local/new-recording-filtered --root ./data/local/new-recording-filtered --episode-index 0
```

Record rejected original indices and reasons with the copy. Keep at least two usable
episodes for the train/development split. `info`/`view` select the configured demonstration
dataset; the explicit arguments above select the filtered copy regardless of config.
Keep `HF_HUB_OFFLINE=1` during inspection/filtering.

## Train directly from finalized recordings

Train from the reviewed copy and keep it immutable for the run. No separate stats file
is required. The trainer fits normalization from the native training split before
policy and processor construction. Raw actions stay absolute.

```sh
pixi run train-check data/local/new-recording-filtered local/new-recording-filtered
pixi run train data/local/new-recording-filtered local/new-recording-filtered --smoke
pixi run train data/local/new-recording-filtered local/new-recording-filtered
```

Equivalent offline config check:

```sh
pixi run python -m scripts.train_lora --dataset-root data/local/new-recording-filtered --repo-id local/new-recording-filtered --dry-run
```

Dataset root is project-relative. Both dataset inputs are required. Training and
`train-check` default to `config/training/training.recipe.json`.

For an experiment, copy the default recipe and edit its `native_flags`, for example
set `steps` to 45000. The filename selects the recipe; its internal `name` can stay
`training`. Use Pixi's `--` separator to forward recipe options:

```sh
cp config/training/training.recipe.json config/training/training-45k.recipe.json
# Edit native_flags.steps in the copy before checking or training.
pixi run train-check data/local/new-recording-filtered local/new-recording-filtered -- --recipe config/training/training-45k.recipe.json
pixi run train data/local/new-recording-filtered local/new-recording-filtered -- --recipe config/training/training-45k.recipe.json --smoke
pixi run train data/local/new-recording-filtered local/new-recording-filtered -- --recipe config/training/training-45k.recipe.json
```

The default uses one idle GPU for 30000 updates, with batch 8. To request two GPUs
for the same budget, select the supplied recipe:

```sh
pixi run train data/local/new-recording-filtered local/new-recording-filtered -- --recipe config/training/training-2gpu.recipe.json
```

Direct module invocation accepts the same `--recipe` option, without the Pixi
separator. Recipe paths are project-relative or absolute. Run-directory labels
come from the selected filename, not its internal `name`.

Evaluation (`eval_steps`) and saving (`save_freq`) can use separate intervals.
Set `eval_steps=0` to disable development evaluation; the held-out split is still
reserved. `max_eval_samples=0` evaluates the full development split; a positive
value caps it.
Set `dataset.image_transforms.enable=true` to enable native image augmentation.
The stop budget need not be a multiple of `save_freq`; the final checkpoint is
still required and verified. `--smoke` overrides the budget and eval/save intervals
for its four-update mechanics check, not a learning result.

The supported route is local, single-task training from the original base, optionally
warm-starting its LoRA weights. Streaming and optimizer-state resume are not supported.
Native splitting holds out the last ceil(episodes × 0.2) episodes **per task**,
not a random frame sample. Both training and development episodes must exist.
Use six-joint radians, optionally followed by normalized `gripper.pos`, in matching
state/action order, with recorder camera identities and the collection/control rate
of 15 Hz. `train-check` and training startup reject a missing or non-15 `fps` in the
dataset's `meta/info.json`. This checks declared rate, not measured capture timing.

Each rank fits independently from actual `train.hf_dataset` numeric columns.
State quantiles use all training states; action quantiles use native complete
within-episode windows of the policy's configured chunk size. Future six-arm
commands subtract the window's initial measured state; gripper stays absolute
when present. Padded tail windows are excluded from fitting. Missing/nonfinite arrays, wrong
joint order, wrong gripper masking or no usable chunks fail clearly. Native camera/
ImageNet statistics are preserved. No image/video decoding is needed for fitting.
Native stat-key renaming and processor construction then use these fitted values.
The native saved checkpoint carries normalizers into rollout; rollout needs no refit.

Fitting repeats at startup on each rank and uses CPU memory for training arrays
and native chunk batches. Cost is unmeasured; no cache/broadcast protocol is used.
Rank0 writes `normalization.json` in the fresh run directory for inspection only;
it is never a required input or proof. Keep finalized dataset files unchanged
throughout a run. `train-check` checks assets/native CPU configuration, not fitted
statistics or a model load; fitting occurs when training starts.

Default settings:

- Base revision: `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`.
- Learning rate: peak 1e-4, floor 1e-5, warmup 1000 steps, decay horizon 30000 steps.
- Stop: 30000 steps.
- Batch: global batch 8, gradient accumulation 1.
- Adapter: expert-side LoRA with rank 32 and alpha 64. The VLM is frozen.
- Precision and data: BF16, no augmentation, seed 1000.
- Full development evaluation and save every 2500 steps.

These settings do not guarantee results on new data.

The supplied recipes allow physical GPU indices 0 and 1. `guards.gpus` lists eligible cards;
`parallelism.dp_replicate` caps how many to use. The default selects one idle card,
preferring GPU 0 and falling back to GPU 1. The two-GPU recipe selects both when
idle, with batch 4 per GPU; if only one is idle, it uses batch 8 on that card. The
global batch stays 8 in either case.

The trainer queries `nvidia-smi` and revalidates selected UUIDs before starting.
If no eligible GPU is idle, it refuses to start. It does not wait or touch other
jobs. Version, source, base hash, strict loading, optimizer, LoRA tensor, schedule
and save guards stay in force.
Run the operator-authorized GPU smoke/save and strict reload checks before a long
run. Every run has a new `outputs/training-runs/` directory.

## Plot training results

From the project root, replace `RUN_DIR` with a directory name under
`outputs/training-runs/`.

```sh
pixi run plot-training outputs/training-runs/RUN_DIR \
  -o "outputs/training-report-$(date +%F)/report.html"
```

To compare runs, add more run directory paths before `-o`. Open the HTML report in
a browser to inspect training and development loss, learning rate, and saved
checkpoints. The task reads local logs and checkpoint metadata and runs offline.

## Warm-start from a saved adapter

Use `config/training/training-warm-start.recipe.json` to start from a previous
run's LoRA weights. Replace `guards.init_adapter` with the project-relative path
to that checkpoint's `pretrained_model` directory (replace `REPLACE_WITH_RUN` and
the checkpoint number). It must contain `adapter_config.json` and
`adapter_model.safetensors`, and match the original base, LoRA targets, rank 32
and alpha 64. Keep the original checkpoint unchanged.

```sh
pixi run train-check data/local/new-recording-filtered local/new-recording-filtered -- --recipe config/training/training-warm-start.recipe.json
pixi run train data/local/new-recording-filtered local/new-recording-filtered -- --recipe config/training/training-warm-start.recipe.json --smoke
pixi run train data/local/new-recording-filtered local/new-recording-filtered -- --recipe config/training/training-warm-start.recipe.json
```

This is **not resume**: only adapter weights are restored. The optimizer and
learning-rate schedule restart, normalization is fitted on the current training
split, and `steps` counts new updates. The sample uses a lower peak learning rate
of `5e-5`, decay learning rate `5e-6`, and 500 warmup steps. Other settings match
the single-GPU default. These are starting points for refinement, not a measured
optimum; adjust the learning rate and update budget for your experiment.
`train-check` validates configuration and adapter metadata, not the full tensor
load; the worker checks adapter keys and shapes before optimization.

## Deploy for rollout

On a single machine (training machine and robot PC in the same checkout), point
`ROLLOUT_POLICY_PATH` directly to the checkpoint's `pretrained_model` directory.
No copies or JSON edits are needed: `adapter_config.json` already points to the
local `hf/hub/` base snapshot, and `tokenizer/` resolves relative to the
checkpoint.

When moving a checkpoint to a separate robot PC (or after moving the base snapshot):

- Copy the full `pretrained_model` directory: adapter config and weights,
  `config.json`, `train_config.json`, both processor JSONs, normalizer
  `.safetensors`, and `tokenizer/`.
- Copy the original base snapshot once, including its config files and
  `model.safetensors`. Dereference Hugging Face cache symlinks (for example
  `rsync -aL`) if copying a standalone snapshot directory. Verify
  `model.safetensors` against `guards.base_sha256` in the recipe, keep a SHA256
  manifest of copied files, and leave the original checkpoint and base files
  untouched.
- In the copy only, set `adapter_config.json` → `base_model_name_or_path` to
  the destination base directory. Leave `config.json`, tokenizer processor
  paths, `train_config.json` (training provenance only), and tensor files
  unchanged: `pixi run rollout` overrides `pretrained_path` in memory, and
  relative tokenizer artifacts resolve against the checkpoint. Record the JSON
  change and verify all other bytes match.

Set `ROLLOUT_POLICY_PATH` in `config/local/machine.env` and keep
`HF_HUB_OFFLINE=1`. Use an absolute path, a `./` prefix, or a path under
`outputs/` (such as `outputs/deploy/pretrained_model`). `outputs/` is reserved
for local checkpoints, and missing local paths fail before any Hub lookup.
Strings that match a Hub ID (`namespace/name` or a bare repo name) fall back to
upstream loading; prefix them with `./` to force local loading.

Before supervised rollouts on the robot PC, run a strict reload and
recorded-input RTC check with the real weights. Metadata checks and synthetic
tests do not verify loading or RTC behavior.
