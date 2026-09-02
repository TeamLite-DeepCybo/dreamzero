# DreamZero inference & serving — DeepCybo Lite

This branch carries the inference/serving stack for running the finetuned
DreamZero-14B world-action model on the DeepCybo Lite robot (bilateral 2×7
joints + 2 grippers, 3 cameras). Everything model-side is **env-gated**: with
no `DZ_*` variables set, behavior is byte-identical to upstream, so this code
is safe to keep on a training branch.

Headline (1× RTX PRO 6000 Blackwell 96GB, single GPU):

| config | warm s/chunk | vs 0.8 s budget | dense quality* |
|---|---|---|---|
| upstream default (8-step mask, CFG 5) | 4.16 | 5.2× too slow | 1.75× |
| dyncache + no-CFG + per-block compile | 0.99–1.22 (mean 1.07) | 1.34× | 1.82× |
| `DZ_DIT_MASK=0,1,2` + no-CFG + compile | 0.82–1.06 (mean 0.90) | 1.1× | **2.15×** |
| `DZ_DIT_MASK=0,1` + no-CFG + compile | 0.64–0.89 (mean **0.71**) | **sub-realtime** | 2.11× |

\* **How quality is measured** (offline, no robot): we pick a test point
every 460 rows of the teleop dataset — 91 points spread across all 150
episodes. At each point the model gets the real camera frames and joint
state up to that moment and predicts the next 24 actions (0.8 s at 30 Hz);
MSE is computed against what the human operator actually did over those 24
steps. "Hold" is a do-nothing baseline that keeps the current pose for the
whole 24 steps; its MSE on the same 91 points is 0.0352. The quality score
is hold-MSE ÷ model-MSE — 1.0 means no better than freezing, 2.0 means half
the error of freezing. Every number in this document uses the same 91
points and the same baseline, so they compare directly. Checkpoint: b300
step-5000 throughout this table.

A chunk is 24 actions at 30 Hz = 0.8 s of robot motion. Fewer, earlier
denoise steps measurably **improve** action MSE on this action head (the
sweep is monotonic: mask {0,1,2} 2.15 > {0,1,5} 2.06 > {0,1,8} 1.97 >
{0,1,12} 1.85 > dyncache 1.82 > 8-step 1.75). The dreamed video gets the
same truncated schedule as the actions, so dream quality and action quality
can diverge under aggressive masks.

> **Status: not yet validated on a physical robot.** Every number and
> serving behavior in this document comes from offline benchmarks and paced
> teacher-forced HTTP replay on the RTX Pro 6000. No config has run against
> the real RoboDriver/robot yet.

## What you need

- **Base model**: `GEAR-Dreams/DreamZero-AgiBot` (~43 GiB full checkpoint:
  DiT + T5 + CLIP + VAE). With `skip_component_loading` the raw
  Wan2.1-I2V-14B repo is *not* needed.
- **Finetuned checkpoint**: a LoRA checkpoint directory (`model.safetensors`
  + `config.json` + `experiment_cfg/`).
- **Tokenizer**: `umt5-xxl` directory.
- **Env**: torch ≥ 2.9 + cu12.8, transformers 4.51.3, flash-attn 2, peft,
  diffusers, dm-tree, ftfy, pandas, opencv, tianshou. torch 2.8 works but is
  ~10% slower and torchao FP8 needs ≤0.12 there.
- ~46 GiB VRAM for weights + up to ~30 GiB KV cache headroom. CFG off
  (`DZ_CFG_SCALE=1.0`) halves KV growth; a 48 GB card can serve short
  sessions only (use `--max_session_chunks`).

## Quickstart: HTTP serving

```bash
export PYTHONPATH=/path/to/dreamzero
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache   # persist compile cache!
export DZ_CFG_SCALE=1.0            # drop the uncond branch (2x compute + KV)
export DZ_DIT_MASK=0,1,2           # 3-step denoise (see flag table)
export DZ_COMPILE_BLOCKS=perblock  # compile the 40 DiT blocks
export DZ_NO_ENCODER_COMPILE=1 DZ_NO_SCHED_COMPILE=1

python scripts/serve_deepcybo_lite_dreamzero_http.py \
  --model_path  /path/to/lora-checkpoint \
  --base_path   /path/to/DreamZero-AgiBot \
  --tokenizer_path /path/to/umt5-xxl-tokenizer \
  --port 9090 --async_pipeline --ground_split --smooth_taps 3
```

`scripts/run_http_server_pro.sh` is the launcher used on our RTX Pro 6000
(machine-specific paths). First start with a cold inductor cache spends
2–8 min compiling; a persistent `TORCHINDUCTOR_CACHE_DIR` makes restarts
fast.

The server implements the same endpoints and JSON schema as the pi0.5 HTTP
server (openpi fork `feat/deepcybo-lite-pipeline`), so by design the only
robot-side change is RoboDriver's `server_url`. This drop-in claim has been
validated against our own test client only — not against the real
RoboDriver.

### Protocol

- `GET /api/v1/health` → status, model/session info, latency counters.
- `POST /api/v1/infer` → request:
  `{"observation": {"state": [16 floats], "images": {"image_head": b64jpeg,
  "image_wrist_left": b64jpeg, "image_wrist_right": b64jpeg}},
  "prompt": "...", "request_id": "...", "metadata": {"session_id": "..."}}`
  Response: `action_chunk` of shape (8, 128) — first 16 dims are joint
  targets, zero-padded to 128 per the pi contract.
- `POST /api/v1/reset` → force a fresh episode (clears KV cache).

State/action layout (16D): `[left 7 joints, right 7 joints, left gripper,
right gripper]`.

### How serving works

- **Frame buffering**: the robot posts one frame per request; the server
  keeps a 4-deep deque per camera. The model runs every 3rd request
  (24 predicted / 8 executed per request) with the 4 buffered frames — at
  the 8-action request cadence they land on the official `[-23,-16,-8,0]`
  replan spacing. Cold call uses 1 frame (warms the KV cache).
- **`--async_pipeline`**: chunk t+1 is predicted while chunk t executes,
  anchored on the commanded end state (one chunk of observation staleness).
  A single-worker executor serializes GPU work.
- **`--ground_split`** (needs async): mid-cycle, a background "ground" task
  VAE-encodes the fresh real frames and writes the KV cache (2 blocks back,
  because the pipeline runs a chunk ahead); the boundary call then runs in
  "predict" mode and is denoise-only. This also fixes a grounding-span
  misalignment plain async had (executed-vs-GT 0.0245 → 0.0085).
- **`--smooth_taps 3`**: binomial ¼-½-¼ smoothing along the 24-action chunk.
  In teacher-forced eval it improved both MSE and jerk vs no smoothing.
- **Session resets**: explicit `/reset`, `session_id` change, prompt change,
  idle gap (`--idle_reset_s`), OOM auto-retry, optional
  `--max_session_chunks`. The model also self-resets its KV window every
  ~4 chunks (checkpoint trained at 33 frames ⇒ 9 latent slots); the
  re-anchor chunk costs ~0.2 s extra — the pipeline buffer absorbs it.

### Server CLI flags

| flag | default | meaning |
|---|---|---|
| `--execute_horizon` | 8 | actions returned per request; model runs every 24/N requests |
| `--async_pipeline` | off | one-chunk-ahead prefetch; adds one chunk of observation staleness. Untested on a real robot. |
| `--ground_split` | off | with async: mid-cycle KV grounding, denoise-only boundaries. Untested on a real robot. |
| `--smooth_taps` | 3 | 0/3/5-tap binomial chunk smoothing |
| `--max_session_chunks` | 0 (∞) | force reset after N model calls (VRAM guard for 48 GB cards) |
| `--idle_reset_s` | 120 | reset session after idle gap |
| `--default_prompt` | boat task | prompt used when the request has none |
| `--early_prefetch` | off | measured worse than the default trigger in teacher-forced replay (fires later in the cycle, stalls 0.63–1.26 s); kept for reference |

## Environment flags (model-side)

All read in `wan_flow_matching_action_tf.py` (plus the scheduler/attention
modules). Unset ⇒ upstream behavior.

### Denoise schedule (the big speed/quality lever)

| var | values | effect |
|---|---|---|
| `NUM_DIT_STEPS` | 5/6/7/8 (default 8) | preset static masks over the 16 UniPC steps; anything else = all 16 |
| `DZ_DIT_MASK` | e.g. `0,1,2` | **custom static mask**: comma list of step indices to compute; skipped steps coast on the last velocity. Must include 0; keep 1 too (the multistep solver needs history). Overrides `NUM_DIT_STEPS`. |
| `DYNAMIC_CACHE_SCHEDULE` | `true` | adaptive skipping by cosine similarity of consecutive velocities (~4 computed steps). Mutually exclusive with static masks (it takes priority). |
| `DZ_DYN_THRESH` | `0.95,0.93;4,2` | dyncache thresholds;countdowns override |
| `DZ_DYN_FORCE_LAST` | int N | force-compute the last N steps under dyncache. Measured (N=3): reverts exactly to the static-mask numbers at higher cost. |

Measured (standard 91-point eval, b300 step-5000): `DZ_DIT_MASK=0,1,2`
scored the best ratio measured so far (2.15×) at 0.90 s/chunk; `0,1` ran
sub-realtime (0.71 s) at 2.11×. In the sweep, quality degraded monotonically
as the last computed step moved later. None of these masks has run on a
real robot yet.

### CFG

| var | effect |
|---|---|
| `DZ_CFG_SCALE` | overrides `cfg_scale`. `1.0` cleanly skips the uncond branch: ~2× less DiT compute *and* half the KV memory. Quality-neutral in teacher-forced eval on this task. Upstream crashes at 1.0 without this patch. |

### Compilation

| var | effect |
|---|---|
| `DZ_COMPILE_BLOCKS` | `perblock` compiles each of the 40 DiT blocks (`fullgraph=False`) — the reliable win (1.12→0.98 s). Any other value compiles whole `model.forward` with that mode string. |
| `DZ_PERBLOCK_MODE` | torch.compile mode for per-block (default `default`) |
| `DZ_NO_ENCODER_COMPILE=1` | skip T5/CLIP/VAE `torch.compile` (they're off the hot path; skipping avoids warmup/compat issues) |
| `DZ_NO_SCHED_COMPILE=1` | skip the flow-scheduler `@torch.compile` decorators |
| `DZ_DYNAMO_OFF=1` | kill switch: disable dynamo entirely (eval scripts only) |
| `TORCHINDUCTOR_CACHE_DIR` | set to persistent disk — compile warmup is 2–8 min per config otherwise |

### Quantization / attention (measured, kept for the record)

| var | effect |
|---|---|
| `DZ_QUANT=fp8` | torchao FP8 dynamic per-row on DiT linears. Measured **slower than bf16 on sm_120** (2.02 s vs 0.98 s/chunk — per-row act-quant overhead exceeds the FP8 GEMM gain at this batch size). |
| `DZ_ATTN=sage` | SageAttention 2 path (falls through to FA2 unless the call shape is compatible). No speedup measured on this stack; FA2 is the default. |

### Loading

| var | effect |
|---|---|
| `DZ_SKIP_LORA_FP32=1` | skip the fp32 upcast of LoRA params at injection (needed to fit 48 GB cards) |
| `DZ_RENAME_REPLACE=1` | with `--video_key_rename`, delete the original key after renaming |

## Open-loop eval & benchmarking

`scripts/open_loop_deepcybo.py` — teacher-forced eval on a GEAR dataset:

```bash
python scripts/open_loop_deepcybo.py \
  --model_path ... --base_path ... --dataset_path ... --tokenizer_path ... \
  --num_samples 100 --stride 460 --start_idx 40 \
  --output_dir results/myrun --dump_dream_dir results/myrun/dream_seq
```

`--num_samples 100 --stride 460` means "one test point every 460 dataset
rows, up to 100 of them" (points too close to an episode's end are skipped,
which is how the standard grid ends up at 91). At each point the model
predicts the next 24 actions from the real frames; the script prints
`HORIZON-24 MSE model / hold-baseline / ratio` — the model's 24-step
prediction error, the error of simply not moving for those 24 steps, and
baseline ÷ model. Score the full 24 steps, not just the first: teleop data
is slow enough that "don't move" is nearly optimal at 1 step, so first-step
MSE can't distinguish models. Also saves `chunks.npz` (pred/gt/hold/idx)
for the dashboard; `--dump_dream_dir` writes the dreamed frames.

**AR latency bench** (`--bench_ar N --bench_sessions 2`): per session, one
cold 1-frame call then N warm 4-frame calls on the real KV cache; session 2
is steady state (session 1 absorbs compile warmup). This is the serving
number — per-chunk cost is periodic with KV depth (cheapest right after the
window reset, ~+0.2 s on the re-anchor chunk).

`scripts/open_loop_b300native.py` — same harness with the B300 dataset
conventions: state/action layout `[L7, LG, R7, RG]` (a fixed permutation of
ours), camera keys `top/left/right_camera-images-rgb`, `annotation.task`
embedded in the parquet, 0-based `frame_index`. Use it to eval checkpoints
trained on B300-converted data. Its `chunks.npz` is in *their* joint order —
permute with `argsort(REORDER)` before comparing against ours.

`scripts/test_dreamzero_http.py` — paced teacher-forced HTTP client: replays
dataset frames against a live server at the real request cadence
(`--interval 0.267`) and diffs returned actions against an eval-run
`chunks.npz`. This is the serving quality gate; run it after any config
change.

## Gotchas (each of these cost us real debugging time)

1. **Loader routing** (`sim_policy.py`): a checkpoint whose training config
   has `save_lora_only: false` is loaded via `VLA.from_pretrained`, NOT
   `VLA.load_lora`. Both entry points are patched to the low-memory loader
   in these scripts — if you write a new entry point, patch both, or a
   LoRA-only `model.safetensors` gets loaded as the entire 16.5B model and
   silently produces garbage (0.02× instead of 2.4×).
2. **Execute all 24 actions per chunk.** The official pattern; KV-cache
   bookkeeping requires it. The server's 8-action responses are slices of
   the same 24-chunk, replanned every 3rd request — do not turn it into an
   8-action receding horizon with cold calls.
3. **JPEG quality**: the HTTP contract ships JPEGs; a q92 round-trip alone
   adds ~1.3e-03 action MSE vs PNG (training data was H.264, so compressed
   input is in-distribution). Lower quality settings were not measured.
4. **cfg 5.0 + 21 latent slots ≈ 30 GiB KV.** 880 tokens/latent frame ×
   40 layers × K,V × bf16 = 720 MiB/frame/cache, ×2 with CFG on. A 48 GB
   card OOMs mid-session; use `DZ_CFG_SCALE=1.0` + `--max_session_chunks`.
5. **Dream quality vs action quality diverge under truncated schedules.**
   Masks improved action MSE in our sweeps, but the dreamed video gets the
   same 2–3 denoise steps, and long-session KV grounding uses those dreamed
   latents. The effect of degraded dreams on long-session behavior is
   unmeasured; dream frames for each config are in the eval dashboard.
