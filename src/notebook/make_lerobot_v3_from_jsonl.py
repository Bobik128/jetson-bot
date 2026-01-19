#!/usr/bin/env python3
"""
Convert your current episode.jsonl + jpg frames dataset into a local LeRobotDataset v3.0-style dataset.

Input (your current format):
  root_in/
    <session_id>/
      ep000/
        episode.jsonl
        frames/
          front_000000.jpg
          side_000000.jpg
          ...
      ep001/
        ...

Output (LeRobot v3 style):
  root_out/<repo_id_sanitized>/
    data/chunk-000/file-000.parquet
    videos/front/chunk-000/file-000.mp4
    videos/side/chunk-000/file-000.mp4
    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet

Actions (6 outputs):
  [base_v, base_w, arm_2, arm_3, arm_4, arm_6]

Observation state (6):
  [dyaw_deg, ax_g, ay_g, az_g, tel_v, tel_w]

Notes:
- This script writes a single parquet shard and a single video shard per camera (chunk-000/file-000).
- That is enough for LeRobot tools/policies to work locally for small/medium datasets.
- Requires: pip install pyarrow pandas opencv-python numpy
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# -------------------------
# Utilities
# -------------------------

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def natural_sort_key(s: str):
    # sort "ep2" before "ep10"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def sanitize_repo_id(name: str) -> str:
    # for a local folder name
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "lerobot_dataset"

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] bad json {path}:{i} -> {e}")
    return out

def bgr_from_rgb_jpg(path: str) -> Optional[np.ndarray]:
    # imread returns BGR
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def compute_stats(arr_2d: np.ndarray) -> Dict[str, List[float]]:
    # arr_2d: (N, D)
    if arr_2d.size == 0:
        return {"mean": [], "std": [], "min": [], "max": []}
    mean = arr_2d.mean(axis=0)
    std = arr_2d.std(axis=0, ddof=0)
    mn = arr_2d.min(axis=0)
    mx = arr_2d.max(axis=0)
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "min": mn.astype(np.float32).tolist(),
        "max": mx.astype(np.float32).tolist(),
    }


# -------------------------
# Data model
# -------------------------

@dataclass
class EpisodeMeta:
    episode_index: int
    task_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    data_chunk_index: int
    data_file_index: int
    # per camera video meta
    video_front_chunk_index: int
    video_front_file_index: int
    video_front_from_index: int
    video_front_to_index: int
    video_side_chunk_index: int
    video_side_file_index: int
    video_side_from_index: int
    video_side_to_index: int


# -------------------------
# Main conversion
# -------------------------

def convert(
    root_in: str,
    root_out: str,
    repo_id: str,
    fps: float,
    video_codec: str,
    video_pix_fmt: str,  # informational only
    overwrite: bool,
) -> str:
    """
    Returns output dataset root path.
    """
    repo_folder = sanitize_repo_id(repo_id)
    out_root = os.path.join(root_out, repo_folder)

    if os.path.exists(out_root) and overwrite:
        # light "overwrite": we only ensure dirs and overwrite known files; we do NOT delete unknown files
        pass
    else:
        ensure_dir(out_root)

    # Paths
    data_dir = os.path.join(out_root, "data", "chunk-000")
    vid_front_dir = os.path.join(out_root, "videos", "front", "chunk-000")
    vid_side_dir = os.path.join(out_root, "videos", "side", "chunk-000")
    meta_dir = os.path.join(out_root, "meta")
    meta_ep_dir = os.path.join(meta_dir, "episodes", "chunk-000")

    ensure_dir(data_dir)
    ensure_dir(vid_front_dir)
    ensure_dir(vid_side_dir)
    ensure_dir(meta_dir)
    ensure_dir(meta_ep_dir)

    # Discover sessions/episodes
    # You have: root_in/<session_id>/epXXX/episode.jsonl
    # We'll scan recursively for ep*/episode.jsonl
    episode_jsonl_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_in):
        if "episode.jsonl" in filenames:
            episode_jsonl_paths.append(os.path.join(dirpath, "episode.jsonl"))

    episode_jsonl_paths.sort(key=natural_sort_key)
    if not episode_jsonl_paths:
        raise RuntimeError(f"No episode.jsonl found under: {root_in}")

    # Writers for videos (single shard per cam)
    front_mp4_path = os.path.join(vid_front_dir, "file-000.mp4")
    side_mp4_path = os.path.join(vid_side_dir, "file-000.mp4")

    vw_front = None
    vw_side = None
    front_size = None
    side_size = None

    # Accumulate frame rows for parquet
    rows: List[Dict[str, Any]] = []
    episodes_meta: List[EpisodeMeta] = []

    global_index = 0
    total_episodes = 0
    total_frames = 0

    # Single task setup (extend later if you want)
    task_index = 0

    for ep_path in episode_jsonl_paths:
        ep_dir = os.path.dirname(ep_path)
        frames_dir = os.path.join(ep_dir, "frames")
        records = read_jsonl(ep_path)
        if not records:
            continue

        # Ensure the episode has synchronized images (you always log both, but keep robust)
        valid = []
        for r in records:
            if r.get("img_front") is None and r.get("img_side") is None:
                continue
            valid.append(r)
        records = valid
        if not records:
            continue

        ep_len = len(records)

        # Episode offsets (videos and dataset are aligned here: 1 frame row == 1 video frame)
        ep_dataset_from = global_index
        ep_dataset_to = global_index + ep_len

        # Read first frames to init video writers
        # Your jsonl stores rel paths like "frames/front_000000.jpg"
        def resolve_img(rel: Optional[str]) -> Optional[str]:
            if rel is None:
                return None
            return os.path.join(ep_dir, rel)

        # Prepare video writers using first available frame sizes
        if vw_front is None:
            # find first front frame that exists
            for r in records:
                p = resolve_img(r.get("img_front"))
                if p and os.path.exists(p):
                    img = bgr_from_rgb_jpg(p)
                    if img is not None:
                        h, w = img.shape[:2]
                        front_size = (w, h)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        vw_front = cv2.VideoWriter(front_mp4_path, fourcc, fps, front_size)
                        break
            if vw_front is None:
                # allow "no front camera"
                pass

        if vw_side is None:
            for r in records:
                p = resolve_img(r.get("img_side"))
                if p and os.path.exists(p):
                    img = bgr_from_rgb_jpg(p)
                    if img is not None:
                        h, w = img.shape[:2]
                        side_size = (w, h)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        vw_side = cv2.VideoWriter(side_mp4_path, fourcc, fps, side_size)
                        break
            if vw_side is None:
                pass

        # Per-episode video offsets (since we use one big mp4 per camera, indexes are global)
        ep_front_from = ep_dataset_from
        ep_front_to = ep_dataset_to
        ep_side_from = ep_dataset_from
        ep_side_to = ep_dataset_to

        # Build frame rows, write video frames
        for fi, r in enumerate(records):
            ts = float(r.get("t", 0.0))

            # --- 6D action ---
            base_v = float(r.get("v", 0.0))
            base_w = float(r.get("w", 0.0))

            u_arm = r.get("u_arm", None)
            if isinstance(u_arm, dict):
                # keys stored as strings in your logger
                arm_2 = clamp01(float(u_arm.get("2", u_arm.get(2, 0.5))))
                arm_3 = clamp01(float(u_arm.get("3", u_arm.get(3, 0.5))))
                arm_4 = clamp01(float(u_arm.get("4", u_arm.get(4, 0.5))))
                arm_6 = clamp01(float(u_arm.get("6", u_arm.get(6, 0.5))))
            else:
                arm_2 = arm_3 = arm_4 = arm_6 = 0.5

            action6 = np.array([base_v, base_w, arm_2, arm_3, arm_4, arm_6], dtype=np.float32)

            # --- 6D observation.state ---
            dyaw = float(r.get("dyaw_deg", 0.0))
            ax = float(r.get("ax_g", 0.0))
            ay = float(r.get("ay_g", 0.0))
            az = float(r.get("az_g", 0.0))
            tel_v = float(r.get("tel_v", 0.0))
            tel_w = float(r.get("tel_w", 0.0))
            obs_state6 = np.array([dyaw, ax, ay, az, tel_v, tel_w], dtype=np.float32)

            # done/reward
            done = (fi == (ep_len - 1))
            reward = 0.0

            # Frame table row
            rows.append({
                "action": action6,
                "observation.state": obs_state6,
                "timestamp": float(ts),
                "episode_index": int(total_episodes),
                "frame_index": int(fi),
                "index": int(global_index),
                "task_index": int(task_index),
                "next.reward": float(reward),
                "next.done": bool(done),
            })

            # Write video frames
            # If a camera is missing, write a black frame of correct size (if writer exists)
            if vw_front is not None:
                p = resolve_img(r.get("img_front"))
                if p and os.path.exists(p):
                    img = bgr_from_rgb_jpg(p)
                else:
                    img = None
                if img is None:
                    img = np.zeros((front_size[1], front_size[0], 3), dtype=np.uint8)
                elif (img.shape[1], img.shape[0]) != front_size:
                    img = cv2.resize(img, front_size, interpolation=cv2.INTER_AREA)
                vw_front.write(img)

            if vw_side is not None:
                p = resolve_img(r.get("img_side"))
                if p and os.path.exists(p):
                    img = bgr_from_rgb_jpg(p)
                else:
                    img = None
                if img is None:
                    img = np.zeros((side_size[1], side_size[0], 3), dtype=np.uint8)
                elif (img.shape[1], img.shape[0]) != side_size:
                    img = cv2.resize(img, side_size, interpolation=cv2.INTER_AREA)
                vw_side.write(img)

            global_index += 1
            total_frames += 1

        # Episode metadata row
        episodes_meta.append(EpisodeMeta(
            episode_index=total_episodes,
            task_index=task_index,
            length=ep_len,
            dataset_from_index=ep_dataset_from,
            dataset_to_index=ep_dataset_to,
            data_chunk_index=0,
            data_file_index=0,
            video_front_chunk_index=0,
            video_front_file_index=0,
            video_front_from_index=ep_front_from,
            video_front_to_index=ep_front_to,
            video_side_chunk_index=0,
            video_side_file_index=0,
            video_side_from_index=ep_side_from,
            video_side_to_index=ep_side_to,
        ))

        total_episodes += 1

    # Close video writers
    if vw_front is not None:
        vw_front.release()
    if vw_side is not None:
        vw_side.release()

    if total_episodes == 0 or total_frames == 0:
        raise RuntimeError("No usable episodes/frames were converted. Check input paths and episode.jsonl contents.")

    # -------------------------
    # Write data parquet
    # -------------------------
    data_parquet_path = os.path.join(data_dir, "file-000.parquet")

    # Convert rows to Arrow with fixed-size list columns
    # action: fixed_size_list(float32, 6)
    # observation.state: fixed_size_list(float32, 6)
    action_arr = np.stack([r["action"] for r in rows], axis=0).astype(np.float32)         # (N,6)
    obs_arr = np.stack([r["observation.state"] for r in rows], axis=0).astype(np.float32)# (N,6)

    table = pa.table({
        "action": pa.FixedSizeListArray.from_arrays(
            pa.array(action_arr.reshape(-1), type=pa.float32()), 6
        ),
        "observation.state": pa.FixedSizeListArray.from_arrays(
            pa.array(obs_arr.reshape(-1), type=pa.float32()), 6
        ),

        # scalars
        "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.float32()),
        "frame_index": pa.array([r["frame_index"] for r in rows], type=pa.int64()),
        "episode_index": pa.array([r["episode_index"] for r in rows], type=pa.int64()),
        "index": pa.array([r["index"] for r in rows], type=pa.int64()),
        "task_index": pa.array([r["task_index"] for r in rows], type=pa.int64()),
        "next.reward": pa.array([r["next.reward"] for r in rows], type=pa.float32()),
        "next.done": pa.array([r["next.done"] for r in rows], type=pa.bool_()),
    })

    pq.write_table(table, data_parquet_path)

    # -------------------------
    # Write meta/episodes parquet
    # -------------------------
    meta_episodes_path = os.path.join(meta_ep_dir, "file-000.parquet")
    ep_df = pd.DataFrame([{
        "episode_index": e.episode_index,
        "task_index": e.task_index,
        "length": e.length,
        "dataset_from_index": e.dataset_from_index,
        "dataset_to_index": e.dataset_to_index,
        "data/chunk_index": e.data_chunk_index,
        "data/file_index": e.data_file_index,
        "videos/front/chunk_index": e.video_front_chunk_index,
        "videos/front/file_index": e.video_front_file_index,
        "videos/front/video_from_index": e.video_front_from_index,
        "videos/front/video_to_index": e.video_front_to_index,
        "videos/side/chunk_index": e.video_side_chunk_index,
        "videos/side/file_index": e.video_side_file_index,
        "videos/side/video_from_index": e.video_side_from_index,
        "videos/side/video_to_index": e.video_side_to_index,
    } for e in episodes_meta])
    pq.write_table(pa.Table.from_pandas(ep_df, preserve_index=False), meta_episodes_path)

    # -------------------------
    # Write meta/tasks.parquet
    # -------------------------
    # Minimal single-task dataset. You can add multiple tasks later.
    tasks_path = os.path.join(meta_dir, "tasks.parquet")
    tasks_df = pd.DataFrame([{"task_index": 0, "task": "default"}])
    pq.write_table(pa.Table.from_pandas(tasks_df, preserve_index=False), tasks_path)

    # -------------------------
    # Write meta/stats.json
    # -------------------------
    stats_path = os.path.join(meta_dir, "stats.json")
    stats = {
        "action": compute_stats(action_arr),
        "observation.state": compute_stats(obs_arr),
        # keep others optional; LeRobot uses stats for normalization pipelines
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # -------------------------
    # Write meta/info.json (matches LeRobot v3 example structure)
    # -------------------------
    info_path = os.path.join(meta_dir, "info.json")

    # Video feature info (largely informational; loader uses the mp4 + offsets)
    # We store codec/pix_fmt fields to mirror the example. Actual codec is mp4v via OpenCV.
    front_h = int(front_size[1]) if front_size else 0
    front_w = int(front_size[0]) if front_size else 0
    side_h = int(side_size[1]) if side_size else 0
    side_w = int(side_size[0]) if side_size else 0

    info = {
        "codebase_version": "v3.0",
        "robot_type": None,
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "chunks_size": 1000,  # informational, used by some tooling
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": float(fps),
        "splits": {
            "train": f"0:{total_episodes}",
        },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": {
                    "base_v": 0,
                    "base_w": 1,
                    "arm_2": 2,
                    "arm_3": 3,
                    "arm_4": 4,
                    "arm_6": 5,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": {
                    "dyaw_deg": 0,
                    "ax_g": 1,
                    "ay_g": 2,
                    "az_g": 3,
                    "tel_v": 4,
                    "tel_w": 5,
                },
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": [3, front_h, front_w],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": front_h,
                    "video.width": front_w,
                    "video.codec": video_codec,
                    "video.pix_fmt": video_pix_fmt,
                    "video.is_depth_map": False,
                    "video.fps": float(fps),
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.images.side": {
                "dtype": "video",
                "shape": [3, side_h, side_w],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": side_h,
                    "video.width": side_w,
                    "video.codec": video_codec,
                    "video.pix_fmt": video_pix_fmt,
                    "video.is_depth_map": False,
                    "video.fps": float(fps),
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32"},
            "frame_index": {"dtype": "int64"},
            "episode_index": {"dtype": "int64"},
            "index": {"dtype": "int64"},
            "task_index": {"dtype": "int64"},
            "next.reward": {"dtype": "float32"},
            "next.done": {"dtype": "bool"},
        },
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("[OK] Wrote LeRobot v3 dataset to:", out_root)
    print("  frames:", total_frames, "episodes:", total_episodes)
    print("  data:", data_parquet_path)
    print("  videos:", front_mp4_path, "and", side_mp4_path)
    print("  meta:", info_path)
    return out_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-in", required=True, help="Input root, e.g. ../data/training")
    ap.add_argument("--root-out", required=True, help="Output root folder to create dataset in")
    ap.add_argument("--repo-id", default="my_robot_dataset", help="Name of the dataset folder (local).")
    ap.add_argument("--fps", type=float, default=30.0, help="FPS to write mp4 videos (should match your capture_fps).")
    ap.add_argument("--video-codec", default="mp4v", help="Info field only; OpenCV uses mp4v here.")
    ap.add_argument("--video-pix-fmt", default="yuv420p", help="Info field only.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite known outputs (does not delete extra files).")
    args = ap.parse_args()

    convert(
        root_in=args.root_in,
        root_out=args.root_out,
        repo_id=args.repo_id,
        fps=args.fps,
        video_codec=args.video_codec,
        video_pix_fmt=args.video_pix_fmt,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
