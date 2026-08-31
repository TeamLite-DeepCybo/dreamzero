"""Teacher-forced HTTP test for serve_deepcybo_lite_dreamzero_http.py.

Replays ground-truth dataset frames/states through the HTTP stack exactly the way
RoboDriver would (one observation per executed 8-step slice) and:
  1. reports latency for model-call vs buffered requests,
  2. validates the first 24 returned actions against the open-loop eval's
     chunks.npz prediction for the same checkpoint + start index (the cold-call
     context is identical, so actions should match to float tolerance).

Run on the serving machine (CPU only):
  PYTHONPATH=~/dreamzero_eval/dreamzero CUDA_VISIBLE_DEVICES="" \
  python scripts/test_dreamzero_http.py --start_idx 40 --n_requests 9 \
      --chunks_npz ~/dreamzero_eval/results/v3_500_dense/chunks.npz
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import open_loop_deepcybo as ol


def http_json(method, url, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


IMG_FORMAT = ".jpg"


def encode_jpeg(img_rgb):
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if IMG_FORMAT == ".jpg" else []
    ok, buf = cv2.imencode(IMG_FORMAT, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           params)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--dataset_path",
                   default=os.path.expanduser(
                       "~/dreamzero_eval/datasets/deepcybo_lite_bilateral_gear"))
    p.add_argument("--start_idx", type=int, default=40)
    p.add_argument("--n_requests", type=int, default=9)
    p.add_argument("--execute_horizon", type=int, default=8)
    p.add_argument("--interval", type=float, default=0.0,
                   help="Sleep between requests (emulate robot execution pacing)")
    p.add_argument("--chunks_npz", default=None,
                   help="open-loop eval chunks.npz to validate the first 24 actions")
    p.add_argument("--img_format", default=".jpg", choices=[".jpg", ".png"],
                   help=".png = lossless, to isolate JPEG round-trip effects")
    args = p.parse_args()
    global IMG_FORMAT
    IMG_FORMAT = args.img_format

    base = f"http://{args.host}:{args.port}/api/v1"
    print("waiting for server health ...")
    while True:
        try:
            h = http_json("GET", base + "/health", timeout=10)
            if h.get("model_loaded"):
                break
            print("  server up, model still loading ...")
        except Exception as e:
            print(f"  not up yet ({type(e).__name__})")
        time.sleep(20)
    print("health:", json.dumps(h, indent=2)[:400])

    ds = ol.DeepcyboDataset(args.dataset_path)
    prompt = ds.get_task(args.start_idx) or "Put the boat into the matching color box."
    print(f"prompt: {prompt!r}")

    http_json("POST", base + "/reset", {})

    executed = []
    timings = []
    for k in range(args.n_requests):
        idx = args.start_idx + k * args.execute_horizon
        images = {}
        for http_key, mk in (("image_head", "video.image_head"),
                             ("image_wrist_left", "video.image_wrist_left"),
                             ("image_wrist_right", "video.image_wrist_right")):
            images[http_key] = encode_jpeg(ds.get_frame(idx, mk))
        req = {
            "request_id": f"tf-{k:03d}",
            "robot_type": "deepcybo_lite",
            "prompt": prompt,
            "observation": {"state": ds.get_state(idx).tolist(), "images": images},
            "metadata": {"fps": 30, "state_dim": 16, "action_dim": 128,
                         "session_id": "tf-test"},
        }
        if k and args.interval:
            time.sleep(args.interval)
        t0 = time.perf_counter()
        resp = http_json("POST", base + "/infer", req)
        wall = time.perf_counter() - t0
        assert resp["status"] == "ok", resp
        chunk = np.asarray(resp["action_chunk"])[:, :16]
        executed.append(chunk)
        m = resp["metadata"]
        timings.append((k, wall, m["inference_time_ms"] / 1e3, m["model_call"],
                        m["kv_start_frame"]))
        print(f"req {k}: wall {wall:6.2f}s  server {m['inference_time_ms']/1e3:6.2f}s  "
              f"model_call={m['model_call']}  kv_start={m['kv_start_frame']}  "
              f"chunk {chunk.shape}")

    model_walls = [w for _, w, _, mc, _ in timings if mc]
    buf_walls = [w for _, w, _, mc, _ in timings if not mc]
    print("\n=== LATENCY ===")
    print(f"model-call requests : n={len(model_walls)}  "
          f"mean {np.mean(model_walls):.2f}s" if model_walls else "no model calls")
    print(f"buffered  requests  : n={len(buf_walls)}  "
          f"mean {np.mean(buf_walls)*1e3:.0f}ms" if buf_walls else "")

    if args.chunks_npz:
        d = np.load(os.path.expanduser(args.chunks_npz))
        idxs = list(d["idx"])
        if args.start_idx in idxs:
            ref = d["pred"][idxs.index(args.start_idx)]        # (24, 16)
            got = np.concatenate(executed[:3], axis=0)[:24]
            mse = float(np.mean((got - ref) ** 2))
            mad = float(np.max(np.abs(got - ref)))
            print("\n=== VALIDATION vs open-loop eval (same ckpt, cold context) ===")
            print(f"first-24-action MSE vs chunks.npz: {mse:.2e}   max|diff|: {mad:.2e}")
            print("MATCH" if mad < 1e-3 else
                  "MISMATCH -- check prompt/frame/state alignment")
        else:
            print(f"start_idx {args.start_idx} not in {args.chunks_npz} idx list")

    # GT continuity check: executed actions vs dataset ground truth
    gts = []
    for k in range(len(executed)):
        for t in range(executed[k].shape[0]):
            gts.append(ds.get_action(args.start_idx + k * args.execute_horizon + t))
    gts = np.stack(gts)
    ex = np.concatenate(executed, axis=0)
    print(f"\nexecuted-vs-GT MSE over {ex.shape[0]} steps: "
          f"{float(np.mean((ex - gts[:ex.shape[0]]) ** 2)):.5f}")


if __name__ == "__main__":
    main()
