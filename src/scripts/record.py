# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re 
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot_robot_jetsonbot import JetsonBotClient, JetsonBotClientConfig
from lerobot_teleoperator_dualsense import SOLeaderPlusDualsenseConfig, SOLeaderPlusDualsense, DualsenseTeleopConfig, MappedTeleop, DualsenseEpisodeButtons, DualsenseEpisodeListener
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


NUM_EPISODES = 60
FPS = 30
EPISODE_TIME_SEC = 90
RESET_TIME_SEC = 20
TASK_DESCRIPTION = "grab blue cube and put it on red plate"
HF_REPO_ID = "Bobik553/jetson-bot_blue_block-on-red-box_NEO"

def next_available_repo_id(base_repo_id: str) -> str:
    """
    Returns base_repo_id if it doesn't exist on HF Hub, otherwise appends -N with the
    next available integer suffix.
    Example: Bobik553/jetson-bot -> Bobik553/jetson-bot-1, -2, ...
    """
    api = HfApi()

    # If base doesn't exist, use it
    try:
        api.repo_info(repo_id=base_repo_id, repo_type="dataset")
        base_exists = True
    except Exception:
        base_exists = False

    if not base_exists:
        return base_repo_id

    # Otherwise list datasets for the user/org and find existing suffixes
    owner, name = base_repo_id.split("/", 1)
    existing = []
    try:
        # Returns DatasetInfo objects
        for ds in api.list_datasets(author=owner):
            if ds.id == base_repo_id:
                existing.append(0)
                continue
            # match e.g. "Bobik553/jetson-bot-12"
            m = re.fullmatch(re.escape(base_repo_id) + r"-(\d+)", ds.id)
            if m:
                existing.append(int(m.group(1)))
    except Exception:
        # If listing fails for some reason, fall back to brute-force probing
        existing.append(0)

    n = 1
    if existing:
        n = max(existing) + 1

    return f"{base_repo_id}-{n}"

def main():
    # Create the robot and teleoperator configurations
    # robot_config = JetsonBotClientConfig(remote_ip="100.82.250.91", id="jetson-bot")
    robot_config = JetsonBotClientConfig(remote_ip="10.102.180.119", id="jetson-bot")
    teleop_config = SOLeaderPlusDualsenseConfig(
        so=SO101LeaderConfig(port="/dev/ttyACM0", use_degrees=False, id="the_leader"),
        ds=DualsenseTeleopConfig(joystick_index=0, axis_forward=1, axis_turn=0, axis_turbo=2),
        allow_partial=False,
    )

    # Initialize the robot and teleoperator
    robot = JetsonBotClient(robot_config)
    teleop = SOLeaderPlusDualsense(teleop_config)
    mapped_teleop = MappedTeleop(teleop)

    # TODO(Steven): Update this example to use pipelines
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # Set unique repo name
    repo_id = next_available_repo_id(HF_REPO_ID)
    print("Using dataset repo_id:", repo_id)

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    mapped_teleop.connect()

    ds_js = teleop.ds.joystick

    if ds_js is None:
        raise RuntimeError("Could not access DualSense joystick instance (ds_js is None). Expose it from your teleop wrapper.")
    
    btns = DualsenseEpisodeButtons(
        btn_pause_toggle=9,      # Options
        btn_exit_early=1,        # Circle
        btn_rerecord_episode=2,  # triangle
        btn_stop_recording=8,    # Create
        poll_hz=60.0,
        debounce_sec=0.20,
        pause_key="paused",
    )
    events = {}

    controller_listener = DualsenseEpisodeListener(ds_js, events, btns)
    controller_listener.start()
    listener = controller_listener

    init_rerun(session_name="jetsonbot_record")

    try:
        if not robot.is_connected or not mapped_teleop.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting record loop...")
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes}")

            # Main record loop
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                teleop=mapped_teleop,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and not events["exit_early"] and (
                (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=mapped_teleop,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
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

            # Save episode
            dataset.save_episode()
            recorded_episodes += 1
    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        mapped_teleop.disconnect()
        if listener is not None:
            listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()