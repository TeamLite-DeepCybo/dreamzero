# Training DreamZero on DeepCybo Lite

How to post-train the DreamZero-14B world-action model on our robot's data,
on this branch (`deepcybo-reconstructed`). This is the branch and recipe the
batch-16 LoRA runs were trained with. For serving and evaluating the
resulting checkpoints, switch to the `deepcybo-inference` branch
(`docs/INFERENCE.md` there); for what the branches are and how they relate,
see `docs/BRANCHES.md`.

## What you are training

LoRA post-training (rank 4, alpha 4, targets `q,k,v,o,ffn.0,ffn.2`) plus
the action/state projector heads, starting from the pretrained
`GEAR-Dreams/DreamZero-AgiBot` checkpoint. The model jointly predicts 24
actions (0.8 s at 30 Hz) and 2 latent video frames per block. DeepCybo Lite
is a bilateral robot: 16-D state/action (2×7 joints + 2 grippers), three
cameras (head + both wrists), and it reuses AgiBot's projector slot
(embodiment index 26 — same bilateral head/left/right layout).

## Environment

Python 3.11. Follow the upstream README install, then apply the pins in
`local-overrides.txt` at the repo root (torch 2.9.0+cu128, transformers
4.57.3 — the upstream 4.51.3 pin hits a deepspeed circular import with
deepspeed ≥ 0.17):

```bash
uv pip install -e . --override local-overrides.txt
```

## Weights and data to fetch

```bash
huggingface-cli download GEAR-Dreams/DreamZero-AgiBot   --local-dir ./checkpoints/DreamZero-AgiBot   # 43 GiB base
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P     --local-dir ./checkpoints/Wan2.1-I2V-14B-480P # T5/CLIP/VAE components
huggingface-cli download google/umt5-xxl                --local-dir ./checkpoints/umt5-xxl
```

All three are also mirrored on the internal A6000 machine (below), which
is faster than HuggingFace from the office LAN.

**Dataset**: the boat-sorting teleop set (150 episodes / 46,668 rows,
30 Hz, three cameras) in LeRobot/GEAR format. Two conversions exist with **different
layouts** — see `docs/BRANCHES.md` before mixing anything:

- `lite_boat_full_deepcybo` — the conversion this branch's config
  (`groot/vla/configs/data/dreamzero/deepcybo_lite_relative.yaml`) was
  trained on. Lives on the B300 training machine at
  `/hdfs/share-data-1/datasets/dreamzero/lite_boat_full_deepcybo`.
- `deepcybo_lite_bilateral_gear` — the other conversion (different
  state/action order and camera key names), stored on the internal machines.
  Usable only with a matching modality config.

To convert new teleop data, use `scripts/data/convert_lerobot_to_gear.py`
and check the resulting `meta/modality.json` slice order against the table
in `docs/BRANCHES.md`.

## Launching on the B300 training machine (primary path)

The batch-16 runs were launched on the 2×B300 machine via its experiment
launcher, using the dev image `exp_lite_dreamzero_posttrain:0.1.0`:

```bash
export HF_TOKEN="<YOUR_HF_TOKEN>"
export WANDB_KEY="<YOUR_WANDB_KEY>"
hf auth login --token "$HF_TOKEN" --add-to-git-credential || true

bash /usr/local/bin/setup-files/launch-dreamzero-exp.sh \
  --task <your_task_name> \
  --data dreamzero/deepcybo_lite_relative \
  --embodiment deepcybo_lite \
  --data-root /hdfs/share-data-1/datasets/dreamzero/lite_boat_full_deepcybo \
  --arch lora \
  --base-ckpt ./checkpoints/DreamZero-AgiBot \
  --max-steps 5000 \
  --lr 1e-4 \
  --batch-size 8 \
  --gpus 2 \
  --wandb-key "$WANDB_KEY" \
  --extra save_steps=500 \
  --extra save_lora_only=false \
  --extra training_args.save_only_model=true \
  --detach
```

- `--detach` runs in the background; logs land in
  `/hdfs/share-data-1/projects/dreamzero/runs/<task_name>/train.log`.
- Add `--dry-run` first to print the exact `torchrun` command before
  launching.
- The launcher records `--wandb-key` in the task's
  `experiment_record.json`.

## The raw training command (any machine)

Equivalent `torchrun` invocation, for machines without the launcher (the
exact resolved config of the batch-16 runs is archived on the storage
machine at `~/dreamzero_eval/checkpoints/b300_experiment_cfg/conf.yaml`):

```bash
export HYDRA_FULL_ERROR=1
torchrun --nproc_per_node <NUM_GPUS> --standalone groot/vla/experiment/experiment.py \
    data=dreamzero/deepcybo_lite_relative \
    train_architecture=lora \
    num_frames=33 action_horizon=24 num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 num_action_per_block=24 num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.weight_decay=1e-5 \
    training_args.warmup_ratio=0.05 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_overlap.json" \
    per_device_train_batch_size=8 \
    max_steps=5000 save_steps=500 \
    ++training_args.save_only_model=true \
    output_dir=<OUTPUT_DIR>
```

Notes that matter (each learned the hard way):

- **`learning_rate=1e-4`, not the upstream 1e-5.** At 1e-5 our runs
  plateaued without ever beating the hold-position baseline; 1e-4 was the
  binding fix (first checkpoint to beat baseline came 500 steps in).
- **`++training_args.save_only_model=true`** saves ~200 MB LoRA+projector
  packages instead of ~87 GB of deepspeed optimizer state per save.
- Batch size: the batch-16 runs used `per_device_train_batch_size=8` on
  2 GPUs. A single 96 GB GPU fits batch 1 + grad-accum with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (that was the v3
  series' setup, ~29 s/step on an RTX Pro 6000).
- A saved checkpoint directory contains `model.safetensors` (LoRA +
  projector only), `config.json`, and `experiment_cfg/` — all three are
  needed to load it later. This branch's config sets `save_lora_only: false`
  in the saved experiment config, which changes which loader the eval stack
  routes through; the inference branch handles both paths (see the
  loader-routing gotcha in `docs/INFERENCE.md`).

## Evaluating a checkpoint

Use the `deepcybo-inference` branch: `scripts/open_loop_deepcybo.py` for
our-layout datasets, `scripts/open_loop_b300native.py` for checkpoints
trained with this branch's data layout. The honest metric is the HORIZON-24
MSE ratio vs the hold-position baseline on a 100-sample dense grid
(first-step MSE is misleading: holding still is nearly optimal at 1 step).
Results feed the eval dashboard (`dashboard/` on that branch).

## Existing checkpoints and where everything lives

Internal machines (addresses and credentials: ask the team; the machine map
is in `docs/BRANCHES.md`):

- **The A6000 machine** (48 GB) — storage/eval box. Long-term home of all
  artifacts, under `~/dreamzero_eval/`:
  - `checkpoints/DreamZero-AgiBot/`, `checkpoints/umt5-xxl-tokenizer/` —
    base model + tokenizer mirrors
  - `checkpoints/b300_b16_{2500,3000,5000}_lora_ckpt/` — the batch-16 series
    trained with this branch's recipe (best: step 5000)
  - `checkpoints/v3-checkpoint-{500,1000,1500,2000,2500}/` — the v3 series
    (single-GPU, our data layout; best: step 2500)
  - `checkpoints/checkpoint-{500,1000,1500,2000}/` — the older v2 series
    (LR 1e-5; kept for the record, superseded)
  - `checkpoints/b300_experiment_cfg/` — exact resolved training config of
    the batch-16 runs
  - `checkpoints/b300_code/` — the exact eval script + model source files
    used on the training machine, plus base-model MD5s (for cross-machine
    result verification)
  - `checkpoints/b300_native_smoke/` — reference eval outputs from the
    training machine
  - `datasets/` — both dataset conversions; `results/` — eval runs
- **The B300 training machine** — training outputs land in
  `/hdfs/share-data-1/projects/dreamzero/runs/<task_name>/` (a
  `checkpoint-<step>/` directory per save, plus `train.log` and
  `experiment_record.json`). The published batch-16 series came from the
  `v3_b300_batch8_eff16_v3` run there; steps 2500/3000/5000 are mirrored to
  the A6000 machine as the `b300_b16_*` packages above.
- **The Pro 6000 machine** (96 GB) — inference box: working copies of
  checkpoints/datasets, the serving stack, and fast evals. The base model
  and the serving venv live in `/dev/shm` there (RAM-backed — gone after a
  reboot; re-copy from the A6000 machine).

Measured quality reference (teacher-forced dense grid, ratio vs hold
baseline — higher is better): `b300_b16_5000` 1.75× at default inference
settings, up to 2.15× with the inference branch's 3-step denoise mask;
`v3-checkpoint-2500` 1.80× with the full inference stack. Nothing has been
validated on the physical robot yet.
