# Training and deployment

Use the [operating guide](operating-guide.md) to record and review demonstrations.
Copy the complete native v3 dataset (`meta`, `data`, `videos`) to the training host,
checking transfer hashes. Keep original bytes immutable. Obtain approved base and
tokenizer assets separately; the training launcher neither downloads nor provisions them.
Keep `HF_HUB_OFFLINE=1`; an empty cache cannot load a policy.

## Train directly from finalized recordings

Review/reject unusable episodes using native dataset tools, finalize the recording,
and keep it immutable for the run. No separate preparation, dataset copy or stats
file is required. The trainer fits normalization automatically from the native
training split before policy and processor construction. Raw actions stay absolute.

Example commands below use an illustrative newly recorded dataset identity; they
are instructions, not claims that this real-data run was executed:

```sh
pixi run train-check data/local/new-recording local/new-recording
pixi run train data/local/new-recording local/new-recording --smoke
pixi run train data/local/new-recording local/new-recording
```

Equivalent Python CLI:

```sh
python -m scripts.train_lora reference-15k --dataset-root data/local/new-recording --repo-id local/new-recording --dry-run
```

Dataset root is project-relative. Both dataset inputs are explicit; the one
`config/training/reference-15k.recipe.json` owns training settings.

The supported route is local, single-task, fresh-base training: no streaming or
resume. Native splitting holds out the last ceil(episodes ×0.2) episodes **per task**,
not a random frame sample. Both training and development episodes must exist.
Use matching six-joint radians plus normalized `gripper.pos` state/action order,
recorder camera identities and the collection/control rate15Hz.

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

Reference settings: original base revision
`22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`, peak1e-4, floor1e-5,
warmup1000, decay horizon30000, stop15000, global batch8, accumulation1,
rank32/alpha64 expert-side LoRA, frozen VLM, BF16, no augmentation, seed1000.
Full development evaluation and save occur every2500 steps. This is a reference
schedule through15k, not a bit-identical replay or guarantee for new data.

The reference training host must expose eligible physical GPU indices0 and1.
Admission rejects a missing eligible index, selects only idle eligible devices and
revalidates their UUIDs before launch. One idle GPU on that two-device host uses
batch8; this does not support/qualify a one-physical-GPU host. No idle GPU means
refusal without waiting or touching other jobs. Version/source/base hash, strict
loading, optimizer, LoRA tensor, schedule and save guards remain intact.
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
Perform a strict real-model load and recorded-input RTC check on the destination
before supervised robot use. The runtime loader remains strict and unchanged;
metadata inspection or synthetic tests do not establish successful reload.
An arbitrary path-string substitution is not sufficient deployment verification.
