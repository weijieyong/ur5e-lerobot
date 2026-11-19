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
come from the selected filename, not its internal `name`. The legacy positional
`reference-15k` module argument still selects the reference recipe.

Evaluation (`eval_steps`) and saving (`save_freq`) can use separate intervals.
Set `eval_steps=0` to disable development evaluation; the held-out split is still
reserved. `max_eval_samples=0` evaluates the full development split; a positive
value caps it.
Set `dataset.image_transforms.enable=true` to enable native image augmentation.
The stop budget need not be a multiple of `save_freq`; the final checkpoint is
still required and verified. `--smoke` overrides the budget and eval/save intervals
for its four-update mechanics check, not a learning result.

The supported route is local, single-task, fresh-base training: no streaming or
resume. Native splitting holds out the last ceil(episodes × 0.2) episodes **per task**,
not a random frame sample. Both training and development episodes must exist.
Use matching six-joint radians plus normalized `gripper.pos` state/action order,
recorder camera identities and the collection/control rate of 15 Hz. `train-check` and
training startup reject a missing or non-15 `fps` in the dataset's `meta/info.json`.
This checks declared rate, not measured capture timing.

Each rank fits independently from actual `train.hf_dataset` numeric columns.
State quantiles use all training states; action quantiles use native complete
within-episode windows of the policy's configured chunk size. Future six-arm
commands subtract the window's initial measured state; gripper stays absolute.
Padded tail windows are excluded from fitting. Missing/nonfinite arrays, wrong
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

The original `reference-15k.recipe.json` remains available as a 15000-step reference.
Neither recipe guarantees results on new data.

Both recipes allow physical GPU indices 0 and 1. `guards.gpus` lists eligible cards;
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

## Deployment copy across hosts

Copy the complete selected checkpoint's `pretrained_model` directory, including
adapter tensors/config, policy/train configs, both processor JSONs and normalizer
tensors, and tokenizer directory. Separately copy the approved original base
snapshot including config and immutable `model.safetensors`; verify its SHA256
against the reference recipe. Retain a manifest of every original and destination file hash.
Never edit the original checkpoint or original base bytes.

For a deployment copy only, explicitly rewrite local path metadata to the destination:
`adapter_config.json`'s `base_model_name_or_path` → approved base snapshot;
`config.json`'s `pretrained_path` → deployment checkpoint itself, and its
`text_tokenizer_name` → copied tokenizer directory; tokenizer processor
step's `config.tokenizer_name` → copied tokenizer directory. Inspect the actual
processor steps to find that tokenizer field; do not replace unrelated text or
rewrite tensor files. If saved `train_config.json` carries policy/base/dataset
paths, retain it as training provenance; it is not an instruction to resume.
Record the exact allowlisted JSON differences and verify all other bytes unchanged.
Ensure base config/processor assets referenced by the copied metadata are present
locally and compatible, never substituted from another run.

Set private `ROLLOUT_POLICY_PATH` to the copied directory and keep assets offline.
Use absolute paths, `./` paths or nested local paths such as
`outputs/deploy/pretrained_model`; `outputs/` is reserved for local deployments.
Missing local paths fail before Hub config resolution. Other valid Hub ID strings
(`namespace/name` or bare repo names) retain upstream loading; prefix with `./` to
mean local.

Perform a strict real-model reload and recorded-input RTC qualification on the destination
before supervised robot use. Metadata inspection or synthetic tests do not establish
successful reload or RTC qualification.
