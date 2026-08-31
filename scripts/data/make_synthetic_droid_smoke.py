#!/usr/bin/env python
"""Generate a tiny synthetic DROID-style LeRobot v2 dataset for smoke testing.

Creates a few synthetic episodes (parquet + mp4 videos) that mimic the
GEAR-Dreams/DreamZero-DROID-Data layout for the ``oxe_droid`` embodiment:
3 camera views, 8-dim state/action (7 joint + 1 gripper), task annotations.

Usage:
    python scripts/data/make_synthetic_droid_smoke.py \
        --output data/synthetic_droid \
        --episodes 3 --frames 60

Then convert to GEAR format (annotation columns are auto-detected; pass no --task-key):
    python scripts/data/convert_lerobot_to_gear.py \
        --dataset-path data/synthetic_droid \
        --embodiment-tag oxe_droid \
        --state-keys '{"joint_position": [0, 7], "gripper_position": [7, 8]}' \
        --action-keys '{"joint_position": [0, 7], "gripper_position": [7, 8]}' \
        --relative-action-keys joint_position
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd


CAMS = [
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
]
LANG_COLS = [
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
]
TASKS = [
    ("pick up the red block and place it in the bowl", "pick up the red block", "place the red block in the bowl"),
    ("move the cup forward", "move the cup forward", "push the cup forward"),
    ("put the marker in the blue box", "pick up the marker", "put the marker in the blue box"),
    ("pick up the apple and put it in the basket", "pick up the apple", "put the apple in the basket"),
    ("slide the plate to the left", "slide the plate", "move the plate left"),
]


def make_frame(cam_idx: int, t: int, h: int, w: int) -> np.ndarray:
    """Synthetic frame: moving color blobs, different per camera."""
    rng = np.random.default_rng(cam_idx * 1000 + t)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., cam_idx % 3] = 60 + 20 * np.sin(t / 8 + cam_idx)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = int(w * (0.2 + 0.6 * (0.5 + 0.5 * np.sin(t / 12 + cam_idx))))
    cy = int(h * (0.3 + 0.4 * (0.5 + 0.5 * np.cos(t / 10 + cam_idx))))
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (h * 0.08) ** 2
    frame[mask] = [200, 40, 40] if cam_idx != 1 else [40, 200, 40]
    return frame


def write_video(path: Path, cam_idx: int, frames: int, h: int, w: int, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for t in range(frames):
        writer.append_data(make_frame(cam_idx, t, h, w))
    writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/synthetic_droid"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for ep in range(args.episodes):
        n = args.frames
        obs = rng.uniform(-0.5, 0.5, size=(n, 8))
        action = np.concatenate(
            [np.sin(np.linspace(0, 2 * np.pi, n))[:, None] * 0.4] * 7
            + [np.cos(np.linspace(0, np.pi, n))[:, None] * 0.2],
            axis=1,
        )
        df = pd.DataFrame(
            {
                "episode_index": np.full(n, ep, dtype=np.int64),
                "frame_index": np.arange(n, dtype=np.int64),
                "timestamp": np.arange(n, dtype=np.float64) / args.fps,
                "observation.state": [obs[i] for i in range(n)],
                "action": [action[i] for i in range(n)],
            }
        )
        ep_tasks = TASKS[ep % len(TASKS)]
        for col, task in zip(LANG_COLS, ep_tasks):
            df[col] = [task] * n
        parquet_path = out / f"data/chunk-000/episode_{ep:06d}.parquet"
        df.to_parquet(parquet_path)

        for cam_idx, cam in enumerate(CAMS):
            write_video(
                out / f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4",
                cam_idx,
                n,
                args.height,
                args.width,
                args.fps,
            )

    features = {
        "observation.state": {"dtype": "float64", "shape": [8]},
        "action": {"dtype": "float64", "shape": [8]},
    }
    for col in LANG_COLS:
        features[col] = {"dtype": "string", "shape": [1]}
    for cam in CAMS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [args.height, args.width, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": args.fps,
                "video.height": args.height,
                "video.width": args.width,
            },
        }
    info = {
        "total_episodes": args.episodes,
        "fps": args.fps,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "robot_type": "franka",
    }
    (out / "meta").mkdir(exist_ok=True)
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Synthetic dataset written to {out} ({args.episodes} episodes x {args.frames} frames)")


if __name__ == "__main__":
    main()
