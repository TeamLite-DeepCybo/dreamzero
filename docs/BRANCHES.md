# This fork: what the branches are and where things live

This is TeamLite-DeepCybo's fork of [NVIDIA DreamZero](https://github.com/dreamzero0/dreamzero)
(Apache 2.0), adapted for the **DeepCybo Lite** robot: bilateral arms with
2×7 joints + 2 grippers, three cameras (head + both wrists), 30 Hz control.
Our task data is the colored-boat sorting set (~150 teleop episodes).

Upstream docs (README, `docs/DATASET_TO_GEAR_AND_TRAIN.md`) still apply for
general DreamZero background. Everything DeepCybo-specific is on the two
branches below.

## The two branches

### `deepcybo-reconstructed` — training

The newer upstream base (commit `ab790c19`, Apr 2026) plus the DeepCybo Lite
embodiment and the training recipe used for the batch-16 LoRA runs on the
B300 training machine. Contains the dataset converter, embodiment configs,
deepspeed configs, and training fixes. **Start here to train.**
See `docs/TRAINING.md` on that branch.

### `deepcybo-inference` — serving & evaluation

An older upstream base plus everything needed to run finetuned checkpoints:
the HTTP serving stack for the robot, open-loop eval + latency benchmarks,
env-gated inference speedups (custom denoise masks, CFG-off, torch.compile),
and the eval dashboard (`dashboard/`). **Start here to serve or evaluate.**
See `docs/INFERENCE.md` on that branch.

The bases differ, which is why these are two branches rather than one: the
inference stack predates the training branch's upstream bump. The inference
patches are env-gated no-ops by default and are expected to port onto the
newer base; that unification has not been done yet. The inference branch
loads and evaluates checkpoints from **both** training lineages (use
`open_loop_deepcybo.py` for our data layout, `open_loop_b300native.py` for
the B300 layout) — all published batch-16 eval numbers were produced by it.
If you write a new loading entry point, read the loader-routing gotcha in
`docs/INFERENCE.md` first.

## Data conventions — read this before touching checkpoints

Two dataset conversions of the same teleop data exist, with different
layouts. Confusing them silently scrambles arms and grippers:

| | ours (`deepcybo_lite_bilateral_gear`) | B300 conversion (`lite_boat_full_deepcybo`) |
|---|---|---|
| state/action 16-D order | `[L joints ×7, R joints ×7, L grip, R grip]` | `[L joints ×7, L grip, R joints ×7, R grip]` |
| camera keys | `image_head`, `image_wrist_left`, `image_wrist_right` | `top/left/right_camera-images-rgb` |
| task string | `tasks.jsonl` via `task_index` | `annotation.task` column in the parquet |
| `frame_index` | starts at 2 (post-trim) | starts at 0 |

The permutation between the layouts:
`REORDER = list(range(0,7)) + [14] + list(range(7,14)) + [15]` maps ours →
theirs; `argsort(REORDER)` maps back. A checkpoint must be fed data in the
layout it was trained on.

## Machines and artifacts

The machines this project runs on (internal network):

| machine | role |
|---|---|
| `192.168.100.124` (A6000 48 GB) | **storage/eval box** — long-term home of all checkpoints, datasets, eval results |
| `192.168.100.196` (RTX Pro 6000 96 GB) | **inference box** — serving, fast evals, benchmarks |
| B300 training machine | batch-16 training runs; its artifacts are mirrored to the storage box |

On the **storage box**, everything lives under `~/dreamzero_eval/`:

- `checkpoints/DreamZero-AgiBot/` — the 43 GiB pretrained base (mirror of
  the public HuggingFace `GEAR-Dreams/DreamZero-AgiBot`)
- `checkpoints/umt5-xxl-tokenizer/`
- `checkpoints/v3-checkpoint-{500,1000,1500,2000,2500}/` — our v3 LoRA
  series (trained on the inference box, LR 1e-4, our data layout)
- `checkpoints/checkpoint-{500,1000,1500,2000}/` — the older v2 series
  (LR 1e-5; superseded by v3)
- `checkpoints/b300_b16_{2500,3000,5000}_lora_ckpt/` — the batch-16 series
  from the B300 machine (B300 data layout)
- `checkpoints/b300_experiment_cfg/` — the exact resolved training config of
  the batch-16 runs
- `checkpoints/b300_code/` — the exact eval script + model files the B300
  runs used, plus base-model checksums
- `checkpoints/b300_native_smoke/` — reference eval outputs from the B300
  machine (for cross-machine result verification)
- `datasets/deepcybo_lite_bilateral_gear/` — our GEAR-format dataset
- `datasets/b300_dataset_copy/` — a one-episode sample of the B300-format
  dataset (full-dataset meta included)
- `results/<run>_dense/chunks.npz` — eval runs; the dashboard pulls from here

Best checkpoints as of Sep 2026, by teacher-forced dense-grid ratio
(higher = better; 1.0 = hold-position baseline):
`b300_b16_5000` ≈ 2.15× (with the 3-step denoise mask, 1.75× at default
settings) and `v3-checkpoint-2500` ≈ 1.80×. Latency on the inference box
reaches 0.71 s per 0.8 s action chunk (sub-realtime) with the 2-step mask —
all measured offline; nothing validated on the physical robot yet.
