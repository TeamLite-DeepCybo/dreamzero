#!/usr/bin/env python3
"""Convert chunks.npz files + samples_meta.json into dashboard JSON."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

meta = json.load(open(os.path.join(DATA, "dashboard_data", "samples_meta.json")))
meta_by_idx = {m["idx"]: m for m in meta}

runs = {}
gt_canon = hold_canon = idx_canon = None
for f in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
    name = os.path.basename(f)[:-4]
    d = np.load(f)
    pred, gt, hold, idxs = d["pred"], d["gt"], d["hold"], d["idx"]
    if gt_canon is None:
        gt_canon, hold_canon, idx_canon = gt, hold, idxs
    runs[name] = pred

def r4(a):
    return np.round(np.asarray(a, dtype=np.float64), 4).tolist()

# shared samples file
samples = []
for i, idx in enumerate(idx_canon):
    m = meta_by_idx.get(int(idx), {})
    h_t = ((hold_canon[i] - gt_canon[i]) ** 2).mean(axis=-1)  # (24,)
    samples.append({
        "i": i, "idx": int(idx),
        "episode": m.get("episode"), "prompt": m.get("prompt", ""),
        "frame": ("data/dashboard_data/" + m["frame"]) if m.get("frame") else None,
        "gt": r4(gt_canon[i]), "hold_state": r4(hold_canon[i][0]),
        "hold_mse": float(np.round(h_t.mean(), 6)),
        "hold_mse_t": r4(h_t),
    })
json.dump(samples, open(os.path.join(DATA, "samples.json"), "w"))

# per-run file: preds + per-sample metrics
out_runs = {}
for name, pred in runs.items():
    per_sample = []
    for i in range(len(idx_canon)):
        m_t = ((pred[i] - gt_canon[i]) ** 2).mean(axis=-1)  # (24,)
        h_t = ((hold_canon[i] - gt_canon[i]) ** 2).mean(axis=-1)
        per_sample.append({
            "pred": r4(pred[i]),
            "mse": float(np.round(m_t.mean(), 6)),
            "mse_t": r4(m_t),
            "ratio": float(np.round(h_t.mean() / max(m_t.mean(), 1e-9), 3)),
        })
    overall_m = np.mean([s["mse"] for s in per_sample])
    overall_h = np.mean([s["hold_mse"] for s in samples])
    out_runs[name] = {
        "samples": per_sample,
        "overall_mse": float(np.round(overall_m, 6)),
        "overall_ratio": float(np.round(overall_h / overall_m, 3)),
    }
json.dump(out_runs, open(os.path.join(DATA, "runs.json"), "w"))
print(f"built: {len(samples)} samples, runs: {list(out_runs.keys())}")
