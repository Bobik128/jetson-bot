#!/usr/bin/env python3

from pathlib import Path
import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(
        description="Push an already-recorded local LeRobot dataset to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hub repo ID, e.g. Bobik553/jetson-bot_blue-block-on-box_train-16",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the local dataset directory containing data/, meta/, videos/",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/push the dataset repo as private.",
    )

    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    info_json = root / "meta" / "info.json"
    if not info_json.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found:\n  {info_json}\n"
            "Expected --root to point directly to the dataset folder."
        )

    print(f"Loading local dataset:")
    print(f"  repo_id: {args.repo_id}")
    print(f"  root:    {root}")

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
    )

    print("Pushing dataset to Hugging Face Hub...")
    dataset.push_to_hub(private=args.private)

    print("Done — dataset pushed successfully.")


if __name__ == "__main__":
    main()