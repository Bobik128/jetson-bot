#!/usr/bin/env python

import re
import torch
from huggingface_hub import HfApi

from lerobot_robot_jetsonbot.jetsonbot_client import JetsonBotClient
from lerobot_robot_jetsonbot.config_jetsonbot_client import JetsonBotClientConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop

from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


NUM_EPISODES = 60
FPS = 30
EPISODE_TIME_SEC = 180
RESET_TIME_SEC = 20

TASK_DESCRIPTION = "blue_brick_moving"
HF_MODEL_ID = "Bobik553/jetson-bot_policy-smolvla-blue_on_red"
HF_EVAL_DATASET_BASE_ID = "Bobik553/jetson-bot_blue-block-on-box_eval-help"


def next_available_repo_id(base_repo_id: str) -> str:
    api = HfApi()

    try:
        api.repo_info(repo_id=base_repo_id, repo_type="dataset")
        base_exists = True
    except Exception:
        base_exists = False

    if not base_exists:
        return base_repo_id

    owner, _ = base_repo_id.split("/", 1)
    existing = []
    try:
        for ds in api.list_datasets(author=owner):
            if ds.id == base_repo_id:
                existing.append(0)
                continue

            m = re.fullmatch(re.escape(base_repo_id) + r"-(\d+)", ds.id)
            if m:
                existing.append(int(m.group(1)))
    except Exception:
        existing.append(0)

    n = max(existing) + 1 if existing else 1
    return f"{base_repo_id}-{n}"


def main():
    robot_config = JetsonBotClientConfig(
        remote_ip="127.0.0.1",
        id="jetson-bot",
    )

    robot = JetsonBotClient(robot_config)

    print("Loading policy on CPU...")
    policy = SmolVLAPolicy.from_pretrained(HF_MODEL_ID, device="cpu")
    print("Policy loaded on CPU")

    print("Moving policy to CUDA...")
    policy.to("cuda")
    policy.eval()
    print("Policy moved to CUDA")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    eval_repo_id = next_available_repo_id(HF_EVAL_DATASET_BASE_ID)
    print("Using eval dataset repo_id:", eval_repo_id)

    dataset = LeRobotDataset.create(
        repo_id=eval_repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=1,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        HF_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )

    robot.connect()

    events = {
        "stop_recording": False,
        "exit_early": False,
        "rerecord_episode": False,
        "paused": False,
    }

    init_rerun(session_name="jetsonbot_evaluate_smolvla")

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("Starting evaluate loop...")
        recorded_episodes = 0

        with torch.inference_mode():
            while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
                log_say(
                    f"Running SmolVLA inference, recording eval episode "
                    f"{recorded_episodes + 1} of {NUM_EPISODES}"
                )

                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=False,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    vals_to_add_while_policy=["motor_linear.vel", "motor_angular.vel"],
                )

                if (
                    not events["stop_recording"]
                    and not events["exit_early"]
                    and ((recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"])
                ):
                    log_say("Reset the environment")

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=FPS,
                        control_time_s=RESET_TIME_SEC,
                        single_task=TASK_DESCRIPTION,
                        display_data=False,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop evaluation")

        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as e:
            print(f"robot.disconnect() failed: {e}")

        try:
            dataset.finalize()
        except Exception as e:
            print(f"dataset.finalize() failed: {e}")

        try:
            dataset.push_to_hub()
        except Exception as e:
            print(f"dataset.push_to_hub() failed: {e}")


if __name__ == "__main__":
    main()