#!/usr/bin/env python
"""Build a DreamZero-ready LeRobot v2 smoke dataset from DeepCybo Lite boat data.

Source data (OpenPI converted, image frames). On this dev machine it lives at:
  <storage-mount>/projects/openpi-pi05/datasets/pick_and_place_colored_boat/runs/
      ep150_20260720/converted_prompted_smoke/local/deepcybo_lite_bilateral
  (<storage-mount> = /hdfs/share-data-1 or /share-data-1, see
   /root/dev-machine-environment.md §13 "存储映射机制")

The Lite robot is bilateral: 16-dim state/action
  [L_joints(7), R_joints(7), L_gripper(1), R_gripper(1)]
and 3 cameras (image_head / image_wrist_left / image_wrist_right).
It is relabeled onto the existing DreamZero ``yam`` embodiment (bimanual:
left/right joint_pos + gripper_pos, 3 cameras) so no code change is needed.

Output is a standard LeRobot v2 dataset with mp4 videos:
  output/
    data/chunk-000/episode_*.parquet        (state/action reordered to yam order + annotation.task)
    videos/chunk-000/observation.images.<cam>/episode_*.mp4
    meta/info.json

Usage:
  python scripts/data/make_lite_boat_smoke_dataset.py \
      --source <converted_prompted_smoke>/local/deepcybo_lite_bilateral \
      --output <storage-mount>/datasets/dreamzero/lite_boat_smoke

Then convert to GEAR format (yam is already registered):
  python scripts/data/convert_lerobot_to_gear.py --dataset-path <output> \
      --embodiment-tag yam \
      --state-keys '{"left_joint_pos":[0,7],"left_gripper_pos":[7,8],"right_joint_pos":[8,15],"right_gripper_pos":[15,16]}' \
      --action-keys '{"left_joint_pos":[0,7],"left_gripper_pos":[7,8],"right_joint_pos":[8,15],"right_gripper_pos":[15,16]}' \
      --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
      --task-key annotation.task
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


CAMS = [
    ("image_head", "top_camera-images-rgb"),
    ("image_wrist_left", "left_camera-images-rgb"),
    ("image_wrist_right", "right_camera-images-rgb"),
]
# Lite order: L_joints(7), R_joints(7), L_gripper(1), R_gripper(1)
# yam order:  left_joint_pos(7), left_gripper_pos(1), right_joint_pos(7), right_gripper_pos(1)
REORDER = list(range(0, 7)) + [14] + list(range(7, 14)) + [15]


def encode_video(img_dir: Path, vid_out: Path, fps: int, threads: int = 2) -> bool:
    """Encode jpg frames -> mp4 (atomic: write .part then rename).
    Returns False if the output already exists (skipped)."""
    if vid_out.exists() and vid_out.stat().st_size > 0:
        return False
    vid_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = vid_out.with_suffix(".mp4.part")
    if tmp.exists():
        tmp.unlink()  # 清理上次失败残留
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str(img_dir / "frame_%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-threads", str(threads),
        "-f", "mp4",  # 输出为 .part 临时文件，需显式指定容器格式
        str(tmp),
    ]
    last_err = ""
    for attempt in range(2):  # NFS 瞬时故障时重试一次
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                os.replace(tmp, vid_out)
                return True
            last_err = r.stderr.strip() or r.stdout.strip()
        except subprocess.SubprocessError as e:
            last_err = str(e)
        if tmp.exists():
            tmp.unlink()
    raise RuntimeError(f"ffmpeg 编码失败: {vid_out}\n{last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="deepcybo_lite_bilateral dir (with data/ images/ meta/)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=5, help="First N episodes to include")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1,
                    help="并行编码进程数（默认 1，串行）")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已生成且非空的 mp4（断点续传）")
    ap.add_argument("--ffmpeg-threads", type=int, default=2,
                    help="每个 ffmpeg 进程的线程数（默认 2；总占用 ≈ workers×ffmpeg-threads 核）")
    args = ap.parse_args()

    src = args.source
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    tasks = {}
    for line in (src / "meta" / "tasks.jsonl").read_text().splitlines():
        t = json.loads(line)
        tasks[t["task_index"]] = t["task"]

    # 1) Rebuild parquet for the first N episodes (yam ordering + annotation.task)
    encode_tasks = []
    for ep in range(args.episodes):
        src_pq = src / f"data/chunk-000/episode_{ep:06d}.parquet"
        if not src_pq.exists():
            print(f"skip missing {src_pq}")
            continue
        df = pd.read_parquet(src_pq)
        n = len(df)
        state = np.stack(df["observation.state"].values)[:, REORDER].astype(np.float32)
        action = np.stack(df["action"].values)[:, REORDER].astype(np.float32)
        task_idx = int(df["task_index"].iloc[0])
        new_df = pd.DataFrame(
            {
                "episode_index": np.full(n, ep, dtype=np.int64),
                "frame_index": np.arange(n, dtype=np.int64),
                "timestamp": np.arange(n, dtype=np.float64) / args.fps,
                "observation.state": [state[i] for i in range(n)],
                "action": [action[i] for i in range(n)],
                "annotation.task": [tasks.get(task_idx, "")] * n,
            }
        )
        new_df.to_parquet(out / f"data/chunk-000/episode_{ep:06d}.parquet")

        # Collect video encode tasks (yam video key names)
        for src_cam, dst_cam in CAMS:
            img_dir = src / "images" / f"observation.images.{src_cam}" / f"episode_{ep:06d}"
            if not img_dir.exists():
                print(f"skip missing images {img_dir}")
                continue
            vid_out = (
                out / "videos" / "chunk-000" / f"observation.images.{dst_cam}"
                / f"episode_{ep:06d}.mp4"
            )
            encode_tasks.append((img_dir, vid_out))

    # 2) Encode videos (parallel with resume support)
    total = len(encode_tasks)
    done = sum(1 for _, v in encode_tasks if v.exists() and v.stat().st_size > 0)
    print(f"video tasks: {total}, 已完成: {done}, 待编码: {total - done}, workers={args.workers}")
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(
                ex.map(
                    lambda t: encode_video(t[0], t[1], args.fps, args.ffmpeg_threads),
                    encode_tasks,
                )
            )
        print(f"encoded {sum(results)} videos（其余为断点续传跳过）")
    else:
        n = 0
        for img_dir, vid_out in encode_tasks:
            if encode_video(img_dir, vid_out, args.fps, args.ffmpeg_threads):
                n += 1
                print("encoded", vid_out)
        print(f"encoded {n} videos")

    # 3) Write info.json
    features = {
        "observation.state": {"dtype": "float32", "shape": [16]},
        "action": {"dtype": "float32", "shape": [16]},
        "annotation.task": {"dtype": "string", "shape": [1]},
    }
    for src_cam, dst_cam in CAMS:
        features[f"observation.images.{dst_cam}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": args.fps},
        }
    info = {
        "total_episodes": args.episodes,
        "fps": args.fps,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "robot_type": "deepcybo_lite",
    }
    (out / "meta").mkdir(exist_ok=True)
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"done: {out} ({args.episodes} episodes)")


if __name__ == "__main__":
    main()
