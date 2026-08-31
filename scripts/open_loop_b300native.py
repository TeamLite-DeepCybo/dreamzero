#!/usr/bin/env python3
"""Offline open-loop evaluation for DreamZero on deepcybo_lite data.

Adapted from scripts/open_loop_yam.py with three key differences:
1. deepcybo_lite camera keys and contiguous 16-dim state/action slices.
2. Video frame lookup uses the parquet `frame_index` column (our videos
   contain the full raw frames; parquet rows are static-trimmed).
3. Custom low-memory model loader: our LoRA was post-trained on top of
   DreamZero-AgiBot weights, so the base must come from the local AgiBot
   checkpoint (NOT raw Wan2.1, which stock load_lora would download).
   Params are initialized on the meta device and base shards are streamed
   from disk directly to GPU (host RAM here is only 31GB).

Usage:
    python scripts/open_loop_deepcybo.py \
        --model_path ~/dreamzero_eval/checkpoints/checkpoint-500 \
        --base_path ~/dreamzero_eval/checkpoints/DreamZero-AgiBot \
        --dataset_path ~/dreamzero_eval/datasets/deepcybo_lite_bilateral_gear \
        --num_samples 100
"""

import torch._dynamo
import os
torch._dynamo.config.disable = os.environ.get("DZ_DYNAMO_OFF") == "1"  # was unconditionally True
_dyn = torch._dynamo.config
for _attr, _val in [("cache_size_limit", 1000), ("recompile_limit", 800),
                    ("accumulated_cache_size_limit", 1000),
                    ("accumulated_recompile_limit", 2000)]:
    if hasattr(_dyn, _attr):
        setattr(_dyn, _attr, _val)

import os
os.environ.setdefault("CUDA_HOME", os.path.expanduser("~/dreamzero_eval/fake_cuda"))
import deepspeed  # noqa: F401 -- break transformers circular import
import argparse
import glob
import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from tianshou.data import Batch
import rerun as rr

from groot.vla.data.schema import EmbodimentTag

# ---------------------------------------------------------------------------
# deepcybo_lite layout (from meta/modality.json)
# ---------------------------------------------------------------------------

VIDEO_CAMERAS = {
    "video.top_camera-images-rgb":   "observation.images.top_camera-images-rgb",
    "video.left_camera-images-rgb":  "observation.images.left_camera-images-rgb",
    "video.right_camera-images-rgb": "observation.images.right_camera-images-rgb",
}

STATE_SLICES = {
    "state.left_joint_pos":    (0, 7),
    "state.left_gripper_pos":  (7, 8),
    "state.right_joint_pos":   (8, 15),
    "state.right_gripper_pos": (15, 16),
}

ACTION_SLICES = {
    "action.left_joint_pos":    (0, 7),
    "action.left_gripper_pos":  (7, 8),
    "action.right_joint_pos":   (8, 15),
    "action.right_gripper_pos": (15, 16),
}

# Must match modality_config_deepcybo_lite concat order
ACTION_KEY_ORDER = [
    "action.left_joint_pos",
    "action.left_gripper_pos",
    "action.right_joint_pos",
    "action.right_gripper_pos",
]


# ---------------------------------------------------------------------------
# Low-memory model loader (meta init + stream shards to GPU)
# ---------------------------------------------------------------------------

def load_model_lowmem(ckpt_dir: str, base_dir: str, device: str):
    """Build VLA on meta device, stream AgiBot base shards to GPU, then
    overlay the LoRA checkpoint. Returns model with LoRA layers present
    (merging is done by GrootSimPolicy afterwards)."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig

    cfg_dict = json.load(open(os.path.join(ckpt_dir, "config.json")))
    config = VLAConfig(**cfg_dict)
    # skip_component_loading stays True (no HF downloads);
    # defer_lora_injection stays True (we inject after base load).

    print("[loader] instantiating model on meta device (bf16)...")
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with init_empty_weights(include_buffers=False):
        model = VLA(config)
    torch.set_default_dtype(prev_dtype)

    idx = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    print(f"[loader] streaming {len(shards)} base shards to {device}...")
    for i, shard in enumerate(shards):
        sd = load_file(os.path.join(base_dir, shard))
        sd = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in sd.items()}
        model.load_state_dict(sd, strict=False, assign=True)
        del sd
        torch.cuda.empty_cache()
        print(f"[loader]   shard {i+1}/{len(shards)} done "
              f"(GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB)")

    metas = [n for n, p in model.named_parameters() if p.is_meta]
    if metas:
        print(f"[loader] WARNING: {len(metas)} params still on meta after base load "
              f"(first: {metas[:5]}) - expected only LoRA-related entries")

    print("[loader] injecting LoRA layers...")
    model.action_head.inject_lora_after_loading()

    print("[loader] loading LoRA checkpoint overlay...")
    sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    if any(".base_layer." in k for k in sd):
        sd = {k.replace(".base_layer.", "."): v for k, v in sd.items()}
    sd = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    loaded = len(sd)
    print(f"[loader] LoRA overlay: {loaded} tensors loaded, "
          f"{len(unexpected)} unexpected")
    if unexpected:
        print(f"[loader]   unexpected (first 5): {unexpected[:5]}")
    del sd

    metas = [n for n, p in model.named_parameters() if p.is_meta]
    if metas:
        raise RuntimeError(f"{len(metas)} params still on meta: {metas[:10]}")

    # Buffers were materialized on CPU by include_buffers=False; move them.
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    print(f"[loader] done. GPU allocated: {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    return model


# ---------------------------------------------------------------------------
# Dataset reader (GEAR/LeRobot chunked format, trimmed parquets)
# ---------------------------------------------------------------------------

class DeepcyboDataset:
    def __init__(self, dataset_path: str):
        self.root = dataset_path

        data_dir = os.path.join(dataset_path, "data")
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "**", "episode_*.parquet"), recursive=True))
        if not parquet_files:
            raise FileNotFoundError(f"No episode_*.parquet found under {data_dir}")

        self.episodes = []
        self.cum_lengths = [0]
        for pf in parquet_files:
            table = pq.read_table(pf)
            self.episodes.append(table)
            self.cum_lengths.append(self.cum_lengths[-1] + table.num_rows)
        self.total_rows = self.cum_lengths[-1]

        # task_index -> task string
        self.tasks = {}
        tasks_path = os.path.join(dataset_path, "meta", "tasks.jsonl")
        with open(tasks_path) as f:
            for line in f:
                t = json.loads(line)
                self.tasks[t["task_index"]] = t["task"]

        videos_root = os.path.join(dataset_path, "videos")
        self.video_dirs = {}
        for server_key, folder_name in VIDEO_CAMERAS.items():
            candidates = sorted(glob.glob(os.path.join(videos_root, "**", folder_name), recursive=True))
            if candidates:
                self.video_dirs[server_key] = candidates[0]

        print(f"DeepcyboDataset: {len(self.episodes)} episodes, "
              f"{self.total_rows} rows, {len(self.video_dirs)} cameras, "
              f"{len(self.tasks)} tasks")

    def __len__(self):
        return self.total_rows

    def _locate(self, idx):
        for ep in range(len(self.episodes)):
            if idx < self.cum_lengths[ep + 1]:
                return ep, idx - self.cum_lengths[ep]
        raise IndexError(f"Index {idx} out of range ({self.total_rows})")

    def get_state(self, idx) -> np.ndarray:
        ep, row = self._locate(idx)
        return np.array(self.episodes[ep].column("observation.state")[row].as_py(), dtype=np.float64)

    def get_action(self, idx) -> np.ndarray:
        ep, row = self._locate(idx)
        return np.array(self.episodes[ep].column("action")[row].as_py(), dtype=np.float64)

    def get_task(self, idx) -> str:
        ep, row = self._locate(idx)
        try:
            if "annotation.task" in self.episodes[ep].column_names:
                v = self.episodes[ep].column("annotation.task")[row].as_py()
                if isinstance(v, list):
                    v = v[0]
                if v:
                    return str(v)
        except Exception:
            pass
        ti = self.episodes[ep].column("task_index")[row].as_py()
        if isinstance(ti, list):
            ti = ti[0]
        return self.tasks.get(int(ti), "")

    def get_frame(self, idx, server_key) -> np.ndarray:
        """Read the video frame matching this parquet row.

        Our videos contain ALL raw frames; parquet rows are trimmed, so the
        video position comes from the `frame_index` column, not the row number.
        """
        ep, row = self._locate(idx)
        fi = self.episodes[ep].column("frame_index")[row].as_py()
        if isinstance(fi, list):
            fi = fi[0]
        mp4 = os.path.join(self.video_dirs[server_key], f"episode_{ep:06d}.mp4")
        cap = cv2.VideoCapture(mp4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {fi} from {mp4}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

VIDEO_KEY_RENAME = {}  # our key -> alternate key, set from --video_key_rename


def _apply_video_renames(obs):
    replace = os.environ.get("DZ_RENAME_REPLACE") == "1"
    for old, new in VIDEO_KEY_RENAME.items():
        if old in obs:
            obs[new] = obs[old]
            if replace:
                del obs[old]
    return obs


def build_obs(dataset: DeepcyboDataset, idx: int, prompt: str) -> dict:
    obs = {}
    for server_key in dataset.video_dirs:
        frame = dataset.get_frame(idx, server_key)
        obs[server_key] = frame[np.newaxis, ...].astype(np.uint8)  # (1, H, W, C)

    state = dataset.get_state(idx)
    for key, (start, end) in STATE_SLICES.items():
        obs[key] = state[start:end].reshape(1, -1).astype(np.float64)

    obs["annotation.task"] = prompt
    return _apply_video_renames(obs)


def get_gt_action_dict(dataset: DeepcyboDataset, idx: int) -> dict:
    action_flat = dataset.get_action(idx)
    gt = {}
    for key in ACTION_KEY_ORDER:
        s, e = ACTION_SLICES[key]
        gt[key] = action_flat[s:e]
    return gt


# ---------------------------------------------------------------------------
# Plotting (same as YAM version)
# ---------------------------------------------------------------------------

def save_plots(all_preds, all_gts, key_names, output_dir):
    pred_flat = np.concatenate([all_preds[k] for k in key_names], axis=-1)
    gt_flat = np.concatenate([all_gts[k] for k in key_names], axis=-1)
    D = pred_flat.shape[1]
    mse_dim = np.mean((pred_flat - gt_flat) ** 2, axis=0)

    ncols = 4
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    overall_mse = float(np.mean(mse_dim))
    fig.suptitle(f"All action dims  (overall MSE={overall_mse:.6f})", fontsize=14)
    for d in range(D):
        ax = axes[d // ncols][d % ncols]
        ax.plot(gt_flat[:, d], label="gt", alpha=0.7, lw=0.8)
        ax.plot(pred_flat[:, d], label="pred", alpha=0.7, lw=0.8)
        ax.set_title(f"dim {d} (MSE={mse_dim[d]:.4f})", fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)
        if d == 0: ax.legend(fontsize=7)
    for d in range(D, nrows * ncols):
        axes[d // ncols][d % ncols].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "all_action_dims.png"), dpi=200)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, len(key_names), figsize=(5 * len(key_names), 4), squeeze=False)
    for i, k in enumerate(key_names):
        ax = axes2[0][i]
        p, g = all_preds[k], all_gts[k]
        for d in range(p.shape[1]):
            ax.plot(g[:, d], '--', alpha=0.5, lw=0.8)
            ax.plot(p[:, d], alpha=0.7, lw=0.8)
        key_mse = float(np.mean((p - g) ** 2))
        ax.set_title(f"{k}\nMSE={key_mse:.6f}", fontsize=9)
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)
    fig2.suptitle("Per-key pred (solid) vs gt (dashed)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(os.path.join(output_dir, "per_key_summary.png"), dpi=200)
    plt.close(fig2)

    return mse_dim, overall_mse


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def build_obs_ar(dataset, indices, prompt):
    """Multi-frame observation for warm KV-cache AR calls."""
    obs = {}
    for server_key in dataset.video_dirs:
        frames = np.stack([dataset.get_frame(i, server_key) for i in indices])
        obs[server_key] = frames.astype(np.uint8)  # (T, H, W, C)
    state = dataset.get_state(indices[-1])
    for key, (start, end) in STATE_SLICES.items():
        obs[key] = state[start:end].reshape(1, -1).astype(np.float64)
    obs["annotation.task"] = prompt
    return _apply_video_renames(obs)


def bench_ar(policy, dataset, args):
    """AR serving latency bench: per session, 1-frame cold call (resets KV cache),
    then N warm 4-frame calls appending to the cache. Session 0 absorbs
    torch.compile warmup; session 1 numbers are steady state."""
    import json as _json
    H = 24
    offsets = [int(x) for x in args.bench_offsets.split(",")]
    n_chunks = args.bench_ar
    idx0 = args.start_idx
    ep, row = dataset._locate(idx0)
    ep_len = dataset.cum_lengths[ep + 1] - dataset.cum_lengths[ep]
    need = 23 + n_chunks * H + 1
    assert row + need <= ep_len, f"episode too short: row {row} + {need} > {ep_len}"
    prompt = dataset.get_task(idx0) or args.prompt
    head = policy.trained_model.action_head
    if os.environ.get("DZ_CFG_SCALE"):
        head.cfg_scale = float(os.environ["DZ_CFG_SCALE"])
        print(f"[bench] cfg_scale overridden to {head.cfg_scale}")
    print(f"[bench] dit mask={head.dit_step_mask} dynamic={head.dynamic_cache_schedule} "
          f"cfg={head.cfg_scale}")

    results = {"offsets": offsets, "n_chunks": n_chunks, "sessions": []}
    for session in range(args.bench_sessions):
        stats = []
        obs = build_obs_ar(dataset, [idx0], prompt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            policy.lazy_joint_forward_causal(Batch(obs=obs))
        torch.cuda.synchronize()
        cold = time.perf_counter() - t0
        print(f"[bench s{session}] cold (1 frame): {cold:.2f}s  "
              f"start_frame={head.current_start_frame}", flush=True)
        stats.append(["cold", cold])
        anchor = idx0 + 23
        for k in range(n_chunks):
            indices = [max(anchor + o, idx0) for o in offsets]
            obs = build_obs_ar(dataset, indices, prompt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                policy.lazy_joint_forward_causal(Batch(obs=obs))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(f"[bench s{session}] warm chunk {k}: {dt:.2f}s  frames={indices}  "
                  f"start_frame={head.current_start_frame}  "
                  f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
            stats.append([f"warm{k}", dt])
            anchor += H
        results["sessions"].append(stats)

    warm2 = [t for name, t in results["sessions"][-1] if name.startswith("warm")]
    cold2 = [t for name, t in results["sessions"][-1] if name == "cold"][0]
    print("\n=== AR LATENCY (last session = steady state) ===")
    print(f"cold first-call: {cold2:.2f}s")
    print(f"warm per-chunk : mean {np.mean(warm2):.2f}s  min {np.min(warm2):.2f}s  "
          f"max {np.max(warm2):.2f}s  (n={len(warm2)})")
    print(f"chunk = {H} actions @30Hz = {H/30:.2f}s of motion -> "
          f"realtime factor {np.mean(warm2)/(H/30):.1f}x slower than realtime")
    out = os.path.join(args.output_dir, "bench_ar.json")
    with open(out, "w") as f:
        _json.dump(results, f, indent=2)
    print(f"saved {out}")


def evaluate(args):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    # Patch VLA.load_lora so GrootSimPolicy uses our low-memory
    # AgiBot-base loader instead of downloading raw Wan2.1.
    from groot.vla.model.dreamzero import base_vla as _bv

    base_dir = os.path.expanduser(args.base_path)
    device = args.device

    def _patched_load_lora(cls, model_path):
        return load_model_lowmem(os.path.expanduser(model_path), base_dir, device)

    _bv.VLA.load_lora = classmethod(_patched_load_lora)

    def _patched_from_pretrained(cls, model_path, config=None, **kwargs):
        return load_model_lowmem(os.path.expanduser(model_path), base_dir, device)

    _bv.VLA.from_pretrained = classmethod(_patched_from_pretrained)

    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    print(f"Loading model from {args.model_path} (base: {base_dir}) ...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.DEEPCYBO_LITE,
        model_path=os.path.expanduser(args.model_path),
        device=device,
        tokenizer_path_override=os.path.expanduser(args.tokenizer_path) if args.tokenizer_path else None,
    )
    print("Model loaded.")

    # --- Evict T5 to CPU with prompt-embedding cache (48GB VRAM cannot hold
    # the full model plus inference workspace; only ~10 unique prompts). ---
    ah = policy.trained_model.action_head
    if os.environ.get("DZ_CFG_SCALE"):
        ah.cfg_scale = float(os.environ["DZ_CFG_SCALE"])
        print(f"[eval] cfg_scale overridden to {ah.cfg_scale}")
    ah.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    _te_cache = {}
    def _cpu_cached_encode(input_ids, attention_mask):
        key = (input_ids.detach().cpu().numpy().tobytes(),
               attention_mask.detach().cpu().numpy().tobytes())
        if key not in _te_cache:
            ids, mask = input_ids.detach().cpu(), attention_mask.detach().cpu()
            t0 = time.perf_counter()
            emb = ah.text_encoder(ids, mask)
            emb = emb.clone().to(dtype=torch.bfloat16)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for _i, _v in enumerate(seq_lens):
                emb[:, _v:] = 0
            _te_cache[key] = emb.to("cuda:0")
            print(f"[t5-cpu] encoded prompt on CPU in {time.perf_counter()-t0:.1f}s "
                  f"(cache size {len(_te_cache)})")
        return _te_cache[key]
    ah.encode_prompt = _cpu_cached_encode
    policy.trained_model._dz_infer_device = torch.device("cuda:0")
    # HF PreTrainedModel.device derives from the FIRST parameter, which is now
    # the CPU-resident T5 -- prepare_input would move all obs to CPU. Pin it.
    from groot.vla.model.dreamzero import base_vla as _bvla
    _bvla.VLA.device = property(lambda self: torch.device("cuda:0"))

    # --- Debug spy: capture unapply intermediates for the first samples ---
    _spy_out = os.environ.get("DZ_UNAPPLY_SPY")
    if _spy_out:
        _orig_unapply = policy.unapply
        _captured = []
        def _unapply_spy(batch, obs=None, **kw):
            res = _orig_unapply(batch, obs=obs, **kw)
            if len(_captured) < 3:
                unnorm = policy.eval_transform.unapply(dict(action=batch.normalized_action.cpu()))
                def _np(x):
                    if hasattr(x, "cpu"): return np.asarray(x.cpu())
                    return np.asarray(x)
                _captured.append(dict(
                    normalized=_np(batch.normalized_action),
                    unnorm={k: _np(v) for k, v in unnorm.items()},
                    states={str(k): _np(obs[k]) for k in obs.keys() if str(k).startswith("state.")},
                    final={str(k): _np(v) for k, v in res.act.items()},
                ))
                if len(_captured) == 3:
                    import pickle
                    with open(_spy_out, "wb") as f:
                        pickle.dump(_captured, f)
                    print(f"[spy] dumped 3 unapply captures to {_spy_out}")
            return res
        policy.unapply = _unapply_spy
    print(f"[t5-cpu] T5 evicted; GPU allocated now "
          f"{torch.cuda.memory_allocated()/2**30:.1f} GiB")

    dataset = DeepcyboDataset(os.path.expanduser(args.dataset_path))
    os.makedirs(args.output_dir, exist_ok=True)
    if getattr(args, "bench_ar", 0):
        bench_ar(policy, dataset, args)
        return
    rrd_path = os.path.join(args.output_dir, "eval.rrd")
    rr.init("dreamzero_openloop")
    rr.save(rrd_path)
    import rerun.blueprint as rrb
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="camera/image_head", name="head"),
            rrb.Spatial2DView(origin="camera/image_wrist_left", name="wrist_left"),
            rrb.Spatial2DView(origin="camera/image_wrist_right", name="wrist_right"),
        ),
        rrb.Horizontal(
            rrb.Spatial2DView(origin="dream/predicted", name="dreamed future (all views)"),
            rrb.Spatial2DView(origin="dream/actual_head", name="actual future (head cam)"),
            rrb.TextDocumentView(origin="task", name="task"),
        ),
        rrb.Horizontal(
            rrb.TimeSeriesView(origin="actions/left_joint_pos", name="left joints"),
            rrb.TimeSeriesView(origin="actions/right_joint_pos", name="right joints"),
            rrb.TimeSeriesView(origin="actions/left_gripper_pos", name="left gripper"),
            rrb.TimeSeriesView(origin="actions/right_gripper_pos", name="right gripper"),
        ),
        row_shares=[3, 3, 2],
    )))

    num = min(args.num_samples, len(dataset))
    preds_per_key = {k: [] for k in ACTION_KEY_ORDER}
    gts_per_key = {k: [] for k in ACTION_KEY_ORDER}
    times = []
    horizon_model, horizon_hold = [], []
    chunk_dump = {"pred": [], "gt": [], "hold": [], "idx": []}

    print(f"\nEvaluating {num} samples (start={args.start_idx}, stride={args.stride}) ...")
    print("-" * 60)

    for i in range(num):
        idx = args.start_idx + i * args.stride
        if idx >= len(dataset):
            break

        prompt = dataset.get_task(idx) or args.prompt
        obs = build_obs(dataset, idx, prompt)

        t0 = time.perf_counter()
        with torch.inference_mode():
            result, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        gt = get_gt_action_dict(dataset, idx)

        rr.set_time_sequence("step", i * 100)
        rr.log("task", rr.TextDocument(prompt))
        for server_key in dataset.video_dirs:
            cam = server_key.split(".")[-1]
            _img = obs.get(server_key)
            if _img is None:
                _img = obs.get(VIDEO_KEY_RENAME.get(server_key, server_key))
            rr.log(f"camera/{cam}", rr.Image(_img[0]).compress(jpeg_quality=75))
        for k in ACTION_KEY_ORDER:
            if k in result.act:
                pv = result.act[k]
                if isinstance(pv, torch.Tensor):
                    pv = pv.cpu().numpy()
                pv = np.atleast_1d(pv[0]).flatten()
                gv = gt[k]
                kname = k.replace("action.", "")
                for d in range(len(gv)):
                    rr.log(f"actions/{kname}/d{d}/pred", rr.Scalar(float(pv[d])))
                    rr.log(f"actions/{kname}/d{d}/gt", rr.Scalar(float(gv[d])))

        # --- Dreamed video: decode predicted latents and log next to GT future ---
        try:
            vp = video_pred
            if torch.is_tensor(vp) and vp.dim() == 5:
                lat = vp.detach()
                if lat.shape[1] != 16 and lat.shape[2] == 16:
                    lat = lat.transpose(1, 2)  # [B,T,C,H,W] -> [B,C,T,H,W]
                with torch.inference_mode():
                    dec = policy.trained_model.action_head.vae.decode(
                        lat.to("cuda:0", torch.bfloat16),
                        tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
                vid = (dec[0].float().clamp(-1, 1).add(1).mul(127.5)
                       .byte().permute(1, 2, 3, 0).cpu().numpy())  # T,H,W,C
                if i == 0:
                    print(f"  [dream] latents {tuple(vp.shape)} -> video {vid.shape}")
                if args.dump_dream_dir:
                    _dd = os.path.expanduser(args.dump_dream_dir)
                    os.makedirs(_dd, exist_ok=True)
                    _T = vid.shape[0]
                    for _ta in range(24):
                        _j = min(int(round(_ta * (_T - 1) / 23)), _T - 1)
                        cv2.imwrite(os.path.join(_dd, f"idx{idx}_t{_ta}.jpg"),
                                    cv2.cvtColor(vid[_j], cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                ep0, row0 = dataset._locate(idx)
                ep_len = dataset.cum_lengths[ep0 + 1] - dataset.cum_lengths[ep0]
                for j in range(vid.shape[0]):
                    rr.set_time_sequence("step", i * 100 + j)
                    rr.log("dream/predicted", rr.Image(vid[j]).compress(jpeg_quality=70))
                    if row0 + j < ep_len:
                        gtf = dataset.get_frame(idx + j, "video.image_head")
                        rr.log("dream/actual_head", rr.Image(gtf).compress(jpeg_quality=70))
                rr.set_time_sequence("step", i * 100)
                del dec, vid
                torch.cuda.empty_cache()
        except Exception as e:
            if i == 0:
                print(f"  [dream] decode failed: {e}")

        for k in ACTION_KEY_ORDER:
            if k in result.act:
                pred_val = result.act[k]
                if isinstance(pred_val, torch.Tensor):
                    pred_val = pred_val.cpu().numpy()
                pred_val = np.atleast_1d(pred_val[0]).flatten()
                preds_per_key[k].append(pred_val)
                gts_per_key[k].append(gt[k])

        # Full-horizon scoring: model chunk vs GT next-24 vs hold-position
        ep_h, row_h = dataset._locate(idx)
        ep_len_h = dataset.cum_lengths[ep_h + 1] - dataset.cum_lengths[ep_h]
        H = 24
        if row_h + H <= ep_len_h:
            gt_chunk = np.stack([dataset.get_action(idx + t) for t in range(H)])  # (24,16)
            pred_parts = []
            for k in ACTION_KEY_ORDER:
                pv = result.act[k]
                if isinstance(pv, torch.Tensor):
                    pv = pv.cpu().numpy()
                pred_parts.append(np.asarray(pv).reshape(H, -1))
            pred_chunk = np.concatenate(pred_parts, axis=-1)  # (24,16) in key order
            gt_reordered = np.concatenate(
                [gt_chunk[:, slice(*ACTION_SLICES[k])] for k in ACTION_KEY_ORDER], axis=-1)
            state_now = dataset.get_state(idx)
            hold_chunk = np.concatenate(
                [np.tile(state_now[slice(*ACTION_SLICES[k])], (H, 1)) for k in ACTION_KEY_ORDER], axis=-1)
            horizon_model.append(np.mean((pred_chunk - gt_reordered) ** 2))
            horizon_hold.append(np.mean((hold_chunk - gt_reordered) ** 2))
            chunk_dump["pred"].append(pred_chunk)
            chunk_dump["gt"].append(gt_reordered)
            chunk_dump["hold"].append(hold_chunk)
            chunk_dump["idx"].append(idx)

        if i % args.log_every == 0:
            if i == 0:
                print(f"  Action keys in output: {list(result.act.keys())}")
                for k in ACTION_KEY_ORDER:
                    if k in result.act:
                        v = result.act[k]
                        shape = v.shape if hasattr(v, 'shape') else "?"
                        print(f"    {k}: pred_shape={shape}, gt_shape={gt[k].shape}")
            print(f"  [{i:>5d}/{num}] idx={idx} infer={elapsed:.3f}s prompt={prompt!r:.60}")

    valid_keys = [k for k in ACTION_KEY_ORDER if len(preds_per_key[k]) > 0]
    if not valid_keys:
        print("No predictions!"); return

    stacked_preds = {k: np.stack(preds_per_key[k]) for k in valid_keys}
    stacked_gts = {k: np.stack(gts_per_key[k]) for k in valid_keys}

    pred_all = np.concatenate([stacked_preds[k] for k in valid_keys], axis=-1)
    gt_all = np.concatenate([stacked_gts[k] for k in valid_keys], axis=-1)
    overall_mse = float(np.mean((pred_all - gt_all) ** 2))

    print(f"\n{'='*60}")
    print(f"Overall MSE: {overall_mse:.6f}  |  Avg inference time: {np.mean(times):.4f}s")
    if horizon_model:
        hm, hh = float(np.mean(horizon_model)), float(np.mean(horizon_hold))
        print(f"HORIZON-24 MSE  model: {hm:.6f}   hold-baseline: {hh:.6f}   ratio: {hh/max(hm,1e-9):.2f}x")
        np.savez(os.path.join(args.output_dir, "chunks.npz"),
                 pred=np.stack(chunk_dump["pred"]), gt=np.stack(chunk_dump["gt"]),
                 hold=np.stack(chunk_dump["hold"]), idx=np.array(chunk_dump["idx"]))
        print(f"chunks saved: {len(chunk_dump['idx'])} samples")
    for k in valid_keys:
        k_mse = float(np.mean((stacked_preds[k] - stacked_gts[k]) ** 2))
        print(f"  {k}: MSE={k_mse:.6f}")
    print(f"{'='*60}")

    mse_dim, _ = save_plots(stacked_preds, stacked_gts, valid_keys, args.output_dir)

    with open(os.path.join(args.output_dir, "mse.txt"), "w") as f:
        f.write(f"overall_mse,{overall_mse}\n")
        for k in valid_keys:
            k_mse = float(np.mean((stacked_preds[k] - stacked_gts[k]) ** 2))
            f.write(f"{k},{k_mse}\n")
        for d, v in enumerate(mse_dim):
            f.write(f"dim_{d},{v}\n")

    print(f"Results saved to {os.path.abspath(args.output_dir)}/")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_path", required=True,
                   help="DreamZero-AgiBot full checkpoint dir (base weights for the LoRA)")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt", default="Put the boat into the matching color box.")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--stride", type=int, default=8,
                   help="Sample every Nth row (action horizon is 24 at 30fps)")
    p.add_argument("--output_dir", default="results_deepcybo")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--video_key_rename", default=None,
                   help="Comma-separated old=new pairs; obs carries BOTH keys "
                        "(for ckpts trained with different video key names)")
    p.add_argument("--dump_dream_dir", default=None,
                   help="Save decoded dream frames as idx{N}_t{0..23}.jpg here")
    p.add_argument("--bench_ar", type=int, default=0,
                   help="Run warm KV-cache AR latency bench with N chunks instead of eval")
    p.add_argument("--bench_offsets", default="-23,-16,-8,0",
                   help="Frame offsets relative to chunk anchor for warm calls")
    p.add_argument("--bench_sessions", type=int, default=2)
    main_args = p.parse_args()
    if main_args.video_key_rename:
        for pair in main_args.video_key_rename.split(","):
            old, new = pair.split("=")
            VIDEO_KEY_RENAME[old.strip()] = new.strip()
        print(f"[eval] video key renames: {VIDEO_KEY_RENAME}")
    evaluate(main_args)


if __name__ == "__main__":
    main()
