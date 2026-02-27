#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import argparse
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_jetsonbot import JetsonBotClient, JetsonBotClientConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say


def load_episode_frames(dataset: LeRobotDataset, episode_idx: int):
    """
    Dataset v3 stores chunked episodes; safest is to filter by episode_index.
    """
    # Filter dataset to only include frames from the specified episode
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
    if len(episode_frames) == 0:
        raise ValueError(
            f"No frames found for episode_index={episode_idx}. "
            f"Available episode_index values may differ from what you expect."
        )
    return episode_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help='HF dataset repo, e.g. "Bobik553/jetson-bot-3"')
    ap.add_argument("--episode", type=int, default=0, help="episode_index to replay (default: 0)")
    ap.add_argument("--remote_ip", default="100.82.250.91", help="JetsonBot remote IP")
    ap.add_argument("--robot_id", default="jetson-bot", help="Robot id string")
    ap.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier (1.0 = real-time)")
    args = ap.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be > 0")

    # Initialize robot
    robot_config = JetsonBotClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    robot = JetsonBotClient(robot_config)

    # Load dataset (pulls from HF Hub)
    dataset = LeRobotDataset(args.repo_id)
    episode_frames = load_episode_frames(dataset, args.episode)

    # Pull only ACTION column(s)
    actions = episode_frames.select_columns(ACTION)

    # Action names are the canonical schema for replay
    action_names = dataset.features[ACTION]["names"]

    # Connect to robot
    robot.connect()

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("Starting replay loop...")
        log_say(f"Replaying repo={args.repo_id} episode={args.episode} frames={len(episode_frames)}")

        dt_target = (1.0 / float(dataset.fps)) / args.speed

        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()

            # Build action dict from the dataset row
            row = actions[idx][ACTION]  # list/array of floats
            action = {name: float(row[i]) for i, name in enumerate(action_names)}

            # Send to robot
            robot.send_action(action)

            precise_sleep(max(dt_target - (time.perf_counter() - t0), 0.0))

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()