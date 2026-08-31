"""HTTP JSON serving for DreamZero on DeepCybo Lite.

Drop-in replacement for the pi0.5 HTTP server (openpi fork, feat/deepcybo-lite-pipeline
scripts/serve_deepcybo_lite_http.py): same endpoints, same request/response schema, so
RoboDriver / InferenceClient only needs a server_url change.

  GET  /api/v1/health
  POST /api/v1/infer     obs = 16D state + base64 JPEGs (image_head, image_wrist_left,
                         image_wrist_right); response action_chunk (N, 128), first 16
                         dims are real joint targets in JOINT_NAMES order.
  POST /api/v1/reset     (additive) force a fresh episode / clear the KV cache.

DreamZero specifics handled inside:
  - Causal AR session: cold call = 1 frame (warms KV cache), then the model runs every
    24 executed actions using the last 4 buffered frames (~8-step spacing, matching the
    official [-23,-16,-8,0] replan schedule).
  - The 24-action prediction is served in EXECUTE_HORIZON-step slices so the robot
    streams frames back at the cadence the frame schedule needs. Only every third
    request pays model latency.
  - Session resets: explicit /reset, metadata.session_id change, prompt change,
    idle gap, --max-session-chunks (VRAM guard for 48GB cards), OOM auto-recovery.
    The model additionally self-resets at 21 KV slots (~10 chunks).
  - Relative->absolute action conversion happens in policy.unapply using the request
    state (same verified path as the open-loop eval).

Usage (A6000 eval machine):
  PYTHONPATH=~/dreamzero_eval/dreamzero CUDA_HOME=~/dreamzero_eval/fake_cuda \
  DZ_SKIP_LORA_FP32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/serve_deepcybo_lite_dreamzero_http.py \
      --model_path .../v3-checkpoint-500 --base_path .../DreamZero-AgiBot \
      --tokenizer_path .../umt5-xxl-tokenizer --port 9090 --max_session_chunks 2
"""

import argparse
import base64
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import open_loop_deepcybo as ol  # noqa: E402  (imports deepspeed/groot in safe order)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from tianshou.data import Batch  # noqa: E402

from groot.vla.data.schema import EmbodimentTag  # noqa: E402

STATE_DIM = 16
HTTP_ACTION_DIM = 128            # response padding, matches pi0.5 server contract
ROBOT_ACTION_DIM = 16
MODEL_CHUNK = 24                 # actions per model call (finetune action horizon)
FRAMES_PER_REPLAN = 4            # frames fed to warm model calls
IMAGE_KEYS = {                   # http key -> model obs key
    "image_head": "video.image_head",
    "image_wrist_left": "video.image_wrist_left",
    "image_wrist_right": "video.image_wrist_right",
}
MODEL_VERSION = "dreamzero-14b-deepcybo-lite"


def _decode_image_b64(b64_str):
    if not b64_str:
        return None
    buf = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _flatten_action(act_dict):
    """Concatenate policy output dict -> (T, 16) in modality order."""
    parts = []
    for k in ol.ACTION_KEY_ORDER:
        v = act_dict[k]
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]
        parts.append(v)
    return np.concatenate(parts, axis=-1)


class DreamZeroHttpPolicy:
    """Owns the model and the (single) AR session. Thread-safe via a lock."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self._exec = ThreadPoolExecutor(max_workers=1)   # serializes GPU work
        self.prefetch = None                             # Future for next chunk
        self.ground_fut = None                           # Future for pre-grounding
        self.model_loaded = False
        self.start_time = time.time()
        self.requests_processed = 0
        self.model_calls_total = 0
        self.last_model_ms = None
        self._load(args)
        self._reset_session("startup")

    # ------------------------------------------------------------------ load
    def _load(self, args):
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29511")
            dist.init_process_group(backend="gloo", world_size=1, rank=0)

        from groot.vla.model.dreamzero import base_vla as _bv

        base_dir = os.path.expanduser(args.base_path)
        device = args.device

        def _patched_load_lora(cls, model_path):
            return ol.load_model_lowmem(os.path.expanduser(model_path), base_dir, device)

        _bv.VLA.load_lora = classmethod(_patched_load_lora)

        def _patched_from_pretrained(cls, model_path, config=None, **kwargs):
            return ol.load_model_lowmem(os.path.expanduser(model_path), base_dir, device)

        _bv.VLA.from_pretrained = classmethod(_patched_from_pretrained)

        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

        print(f"[serve] loading {args.model_path} (base: {base_dir}) ...", flush=True)
        self.policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag.DEEPCYBO_LITE,
            model_path=os.path.expanduser(args.model_path),
            device=device,
            tokenizer_path_override=os.path.expanduser(args.tokenizer_path)
            if args.tokenizer_path else None,
        )

        # T5 -> CPU with prompt cache; pin VLA.device (same as open_loop eval).
        ah = self.policy.trained_model.action_head
        if os.environ.get("DZ_CFG_SCALE"):
            ah.cfg_scale = float(os.environ["DZ_CFG_SCALE"])
            print(f"[serve] cfg_scale overridden to {ah.cfg_scale}", flush=True)
        print(f"[serve] dit mask={ah.dit_step_mask} dynamic={ah.dynamic_cache_schedule} "
              f"force_last={getattr(ah, 'dyn_force_last', 0)} cfg={ah.cfg_scale}", flush=True)
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
                print(f"[t5-cpu] encoded prompt in {time.perf_counter()-t0:.1f}s "
                      f"(cache {len(_te_cache)})", flush=True)
            return _te_cache[key]

        ah.encode_prompt = _cpu_cached_encode
        self.policy.trained_model._dz_infer_device = torch.device("cuda:0")
        from groot.vla.model.dreamzero import base_vla as _bvla
        _bvla.VLA.device = property(lambda self: torch.device("cuda:0"))

        self.action_head = ah
        self.model_loaded = True
        print(f"[serve] model ready; GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB",
              flush=True)

    # --------------------------------------------------------------- session
    def _reset_session(self, reason):
        if getattr(self, "prefetch", None) is not None:
            try:
                self.prefetch.result(timeout=120)
            except Exception:
                pass
            self.prefetch = None
        if getattr(self, "ground_fut", None) is not None:
            try:
                self.ground_fut.result(timeout=120)
            except Exception:
                pass
            self.ground_fut = None
        if self.model_loaded:
            _ah = self.action_head
            _ah._dz_pregrounded = False
            _ah._dz_ref_latents = None
        self.frames = {k: deque(maxlen=FRAMES_PER_REPLAN) for k in IMAGE_KEYS.values()}
        self.pending = []            # remaining actions from the current prediction
        self.session_chunks = 0      # model calls in this session
        self.session_id = None
        self.prompt = None
        self.last_request_time = None
        if self.model_loaded:
            ah = self.action_head
            ah.current_start_frame = 0
            ah.kv_cache1 = ah.kv_cache_neg = None
            ah.crossattn_cache = ah.crossattn_cache_neg = None
            torch.cuda.empty_cache()
        print(f"[serve] session reset ({reason})", flush=True)

    def _maybe_auto_reset(self, session_id, prompt):
        now = time.time()
        if (self.last_request_time is not None
                and now - self.last_request_time > self.args.idle_reset_s):
            self._reset_session(f"idle {now - self.last_request_time:.0f}s")
        elif session_id is not None and self.session_id is not None \
                and session_id != self.session_id:
            self._reset_session(f"session_id {self.session_id} -> {session_id}")
        elif self.prompt is not None and prompt != self.prompt:
            self._reset_session("prompt changed")
        self.session_id = session_id if session_id is not None else self.session_id
        self.prompt = prompt
        self.last_request_time = now

    # --------------------------------------------------------------- request
    def infer(self, req):
        obs_in = req.get("observation", {})
        state = np.asarray(obs_in.get("state", []), dtype=np.float64)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"observation.state must be length {STATE_DIM}, "
                             f"got shape {state.shape}")
        prompt = req.get("prompt") or self.args.default_prompt
        meta = req.get("metadata") or {}
        session_id = meta.get("session_id")

        images = obs_in.get("images", {})
        head = _decode_image_b64(images.get("image_head"))
        if head is None:
            raise ValueError("image_head is required (valid base64 JPEG)")
        decoded = {"video.image_head": head}
        for http_key in ("image_wrist_left", "image_wrist_right"):
            img = _decode_image_b64(images.get(http_key))
            if img is None:
                img = np.zeros_like(head)
            decoded[IMAGE_KEYS[http_key]] = img

        with self.lock:
            self._maybe_auto_reset(session_id, prompt)
            for k, img in decoded.items():
                self.frames[k].append(img)

            model_ran = False
            t0 = time.perf_counter()
            if len(self.pending) < self.args.execute_horizon:
                if self.args.async_pipeline and self.prefetch is not None:
                    fut, self.prefetch = self.prefetch, None
                    try:
                        actions = _flatten_action(fut.result().act)
                        self.pending.extend(list(actions))
                        self.session_chunks += 1
                        model_ran = True
                    except Exception as e:  # noqa: BLE001 - fall back to sync
                        print(f"[serve] prefetch failed ({type(e).__name__}: {e}); "
                              "predicting synchronously", flush=True)
                if len(self.pending) < self.args.execute_horizon:
                    self._predict_chunk(state, prompt)
                    model_ran = True
            _trigger = (len(self.pending) == self.args.execute_horizon
                        if self.args.early_prefetch
                        else len(self.pending) > MODEL_CHUNK - self.args.execute_horizon)
            if (self.args.async_pipeline and self.session_chunks > 0
                    and self.prefetch is None
                    and min(len(f) for f in self.frames.values()) >= 2
                    and _trigger):
                if self.ground_fut is not None:
                    try:
                        self.ground_fut.result(timeout=0.001)
                    except Exception as e:  # noqa: BLE001
                        print(f"[serve] ground task issue: {type(e).__name__}: {e}",
                              flush=True)
                    self.ground_fut = None
                anchor = np.asarray(self.pending[-1], dtype=np.float64)
                self.prefetch = self._exec.submit(
                    self._model_infer, self._prefetch_obs(anchor, prompt),
                    "predict" if self.args.ground_split else None)
            if (self.args.async_pipeline and self.args.ground_split
                    and self.session_chunks > 0 and self.ground_fut is None
                    and self.prefetch is not None
                    and len(self.pending) - self.args.execute_horizon
                        == self.args.execute_horizon):
                self.ground_fut = self._exec.submit(
                    self._model_infer, self._prefetch_obs(state, prompt), "ground")
            n = min(self.args.execute_horizon, len(self.pending))
            chunk = np.stack(self.pending[:n])          # (n, 16)
            self.pending = self.pending[n:]
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            self.requests_processed += 1
            if model_ran:
                self.model_calls_total += 1
                self.last_model_ms = elapsed_ms

        padded = np.zeros((n, HTTP_ACTION_DIM), dtype=np.float64)
        padded[:, :ROBOT_ACTION_DIM] = chunk
        return padded, {
            "fps": meta.get("fps", 30),
            "inference_time_ms": round(elapsed_ms, 1),
            "model_call": model_ran,
            "session_chunks": self.session_chunks,
            "kv_start_frame": int(self.action_head.current_start_frame),
            "model_version": MODEL_VERSION,
            "timestamp_unix": time.time(),
        }

    def _model_infer(self, obs, mode=None):
        ah = self.action_head
        ah._dz_mode = mode
        try:
            with torch.inference_mode():
                result, _ = self.policy.lazy_joint_forward_causal(Batch(obs=obs))
        finally:
            ah._dz_mode = None
        return result

    def _prefetch_obs(self, anchor_state, prompt):
        obs = {}
        for k in self.frames:
            obs[k] = np.stack(list(self.frames[k])).astype(np.uint8)
        for key, (s, e) in ol.STATE_SLICES.items():
            obs[key] = anchor_state[s:e].reshape(1, -1).astype(np.float64)
        obs["annotation.task"] = prompt
        return obs

    def _predict_chunk(self, state, prompt):
        cold = (self.session_chunks == 0
                or (self.args.max_session_chunks
                    and self.session_chunks >= self.args.max_session_chunks))
        if cold and self.session_chunks > 0:
            newest = {k: self.frames[k][-1] for k in self.frames}
            self._reset_session(f"max_session_chunks {self.args.max_session_chunks}")
            for k, img in newest.items():
                self.frames[k].append(img)
            self.prompt = prompt
            self.last_request_time = time.time()

        if self.session_chunks == 0:
            frame_sel = {k: [self.frames[k][-1]] for k in self.frames}
        else:
            frame_sel = {k: list(self.frames[k]) for k in self.frames}

        obs = {}
        for k, fr in frame_sel.items():
            obs[k] = np.stack(fr).astype(np.uint8)      # (T, H, W, C)
        for key, (s, e) in ol.STATE_SLICES.items():
            obs[key] = state[s:e].reshape(1, -1).astype(np.float64)
        obs["annotation.task"] = prompt

        try:
            result = self._exec.submit(self._model_infer, obs).result()
        except torch.OutOfMemoryError:
            print("[serve] OOM during model call -> reset session, retry cold",
                  flush=True)
            newest = {k: self.frames[k][-1] for k in self.frames}
            self._reset_session("oom")
            for k, img in newest.items():
                self.frames[k].append(img)
            self.prompt = prompt
            self.last_request_time = time.time()
            obs_cold = dict(obs)
            for k in frame_sel:
                obs_cold[k] = obs[k][-1:]
            result = self._exec.submit(self._model_infer, obs_cold).result()

        actions = _flatten_action(result.act)           # (24, 16)
        assert actions.shape == (MODEL_CHUNK, ROBOT_ACTION_DIM), actions.shape
        if self.args.smooth_taps == 3:
            pad = np.concatenate([actions[:1], actions, actions[-1:]], axis=0)
            actions = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
        elif self.args.smooth_taps == 5:
            pad = np.concatenate([actions[:1]] * 2 + [actions] + [actions[-1:]] * 2, axis=0)
            actions = (pad[:-4] + 4 * pad[1:-3] + 6 * pad[2:-2]
                       + 4 * pad[3:-1] + pad[4:]) / 16
        self.pending.extend(list(actions))
        self.session_chunks += 1

    def health(self):
        return {
            "status": "ok" if self.model_loaded else "loading",
            "model_loaded": self.model_loaded,
            "gpu_available": torch.cuda.is_available(),
            "mock_mode": False,
            "uptime_s": round(time.time() - self.start_time, 1),
            "requests_processed": self.requests_processed,
            "model_calls_total": self.model_calls_total,
            "last_model_ms": self.last_model_ms,
            "checkpoint": self.args.model_path,
            "action_horizon": MODEL_CHUNK,
            "execute_horizon": self.args.execute_horizon,
            "robot_action_dim": ROBOT_ACTION_DIM,
            "http_action_dim": HTTP_ACTION_DIM,
            "kv_start_frame": int(self.action_head.current_start_frame)
            if self.model_loaded else None,
            "note": "DreamZero causal AR: model runs every "
                    f"{MODEL_CHUNK // self.args.execute_horizon} requests; "
                    "session resets on idle gap, session_id change, or /api/v1/reset",
        }


POLICY = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *a):
        pass  # request logging handled explicitly

    def do_GET(self):
        if self.path == "/api/v1/health":
            self._send(200, POLICY.health())
        else:
            self._send(404, {"status": "error", "error_message": "unknown endpoint"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length)) if length else {}
        except json.JSONDecodeError as e:
            self._send(400, {"status": "error", "error_code": "bad_json",
                             "error_message": str(e)})
            return

        if self.path == "/api/v1/reset":
            with POLICY.lock:
                POLICY._reset_session("explicit /api/v1/reset")
            self._send(200, {"status": "ok"})
            return
        if self.path != "/api/v1/infer":
            self._send(404, {"status": "error", "error_message": "unknown endpoint"})
            return

        rid = req.get("request_id", "")
        try:
            chunk, meta = POLICY.infer(req)
            self._send(200, {
                "request_id": rid,
                "status": "ok",
                "action_chunk": chunk.tolist(),
                "chunk_size": int(chunk.shape[0]),
                "action_dim": HTTP_ACTION_DIM,
                "metadata": meta,
                "error_code": None,
                "error_message": None,
            })
            print(f"[serve] infer {rid}: {meta['inference_time_ms']}ms "
                  f"model_call={meta['model_call']} "
                  f"kv_start={meta['kv_start_frame']}", flush=True)
        except Exception as e:  # noqa: BLE001 - report to client, keep serving
            import traceback
            traceback.print_exc()
            self._send(500, {
                "request_id": rid,
                "status": "error",
                "action_chunk": None,
                "chunk_size": 0,
                "action_dim": HTTP_ACTION_DIM,
                "metadata": {},
                "error_code": type(e).__name__,
                "error_message": str(e),
            })


def main():
    global POLICY
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--execute_horizon", type=int, default=8,
                   help="Actions returned per request; model runs every "
                        "24/execute_horizon requests")
    p.add_argument("--max_session_chunks", type=int, default=0,
                   help="Force a session reset after N model calls "
                        "(VRAM guard for 48GB cards; 0 = unlimited)")
    p.add_argument("--idle_reset_s", type=float, default=120.0)
    p.add_argument("--ground_split", action="store_true",
                   help="With async: pre-ground the KV cache mid-cycle so the "
                        "boundary prediction is denoise-only (~0.73s < 0.8s)")
    p.add_argument("--early_prefetch", action="store_true",
                   help="Launch the next-chunk prediction one request early "
                        "(frame window shifted -8 steps: constant ~267ms obs lag, "
                        "zero boundary stall when compute < 1.07s)")
    p.add_argument("--async_pipeline", action="store_true",
                   help="Predict chunk t+1 while chunk t executes (one-chunk "
                        "observation staleness; anchored on commanded end state)")
    p.add_argument("--smooth_taps", type=int, default=3, choices=[0, 3, 5],
                   help="Binomial smoothing of the 24-action chunk along time "
                        "(0 = off); measured to improve both MSE and jerk")
    p.add_argument("--default_prompt",
                   default="Put the boat into the matching color box.")
    args = p.parse_args()
    assert MODEL_CHUNK % args.execute_horizon == 0, \
        "execute_horizon must divide 24"

    POLICY = DreamZeroHttpPolicy(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
