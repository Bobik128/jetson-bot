#!/usr/bin/env python

import re
from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop

from lerobot_robot_jetsonbot import JetsonBotClient, JetsonBotClientConfig
from lerobot_teleoperator_dualsense import (
    SOLeaderPlusDualsenseConfig,
    SOLeaderPlusDualsense,
    DualsenseTeleopConfig,
    MappedTeleop,
    DualsenseEpisodeButtons,
    DualsenseEpisodeListener,
)
from lerobot.teleoperators.so_leader import SO101LeaderConfig

from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


NUM_EPISODES = 60
FPS = 30
EPISODE_TIME_SEC = 180
RESET_TIME_SEC = 20

TASK_DESCRIPTION = "blue_brick_moving"

# IMPORTANT:
# This should point to a SmolVLA checkpoint, not an ACT checkpoint.
# In practice, use your own fine-tuned SmolVLA model.
HF_MODEL_ID = "Bobik553/jetson-bot_policy-smolvla-blue_on_red"

# Use a separate repo for eval rollouts.
HF_EVAL_DATASET_BASE_ID = "Bobik553/jetson-bot_blue-block-on-box_eval-help"


def next_available_repo_id(base_repo_id: str) -> str:
    """
    Returns base_repo_id if it doesn't exist on HF Hub, otherwise appends -N with the
    next available integer suffix.
    Example:
        Bobik553/jetson-bot_eval -> Bobik553/jetson-bot_eval-1, -2, ...
    """
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

    n = 1
    if existing:
        n = max(existing) + 1

    return f"{base_repo_id}-{n}"


def main():
    # -------------------------------------------------------------------------
    # Robot + teleop config
    # -------------------------------------------------------------------------
    robot_config = JetsonBotClientConfig(
        remote_ip="10.98.56.119",
        id="jetson-bot",
    )

    teleop_config = SOLeaderPlusDualsenseConfig(
        so=SO101LeaderConfig(
            port="/dev/ttyACM0",
            use_degrees=False,
            id="the_leader",
        ),
        ds=DualsenseTeleopConfig(
            joystick_index=0,
            axis_forward=1,
            axis_turn=0,
            axis_turbo=2,
        ),
        allow_partial=False,
    )

    robot = JetsonBotClient(robot_config)
    teleop = SOLeaderPlusDualsense(teleop_config)
    mapped_teleop = MappedTeleop(teleop)

    # -------------------------------------------------------------------------
    # Policy: SmolVLA
    # -------------------------------------------------------------------------
    policy = SmolVLAPolicy.from_pretrained(HF_MODEL_ID)

    # -------------------------------------------------------------------------
    # Processors
    # -------------------------------------------------------------------------
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # -------------------------------------------------------------------------
    # Dataset feature spec must match this robot
    # -------------------------------------------------------------------------
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # -------------------------------------------------------------------------
    # Eval dataset repo
    # -------------------------------------------------------------------------
    eval_repo_id = next_available_repo_id(HF_EVAL_DATASET_BASE_ID)
    print("Using eval dataset repo_id:", eval_repo_id)

    dataset = LeRobotDataset.create(
        repo_id=eval_repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # -------------------------------------------------------------------------
    # Policy pre/post processors
    #
    # SmolVLA docs/example use:
    # make_pre_post_processors(policy.config, model_id, ...)
    # -------------------------------------------------------------------------
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        HF_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )

    # -------------------------------------------------------------------------
    # Connect robot + teleop
    # -------------------------------------------------------------------------
    robot.connect()
    mapped_teleop.connect()

    # -------------------------------------------------------------------------
    # DualSense episode controls
    # -------------------------------------------------------------------------
    ds_js = teleop.ds.joystick
    if ds_js is None:
        raise RuntimeError(
            "Could not access DualSense joystick instance (ds_js is None). "
            "Expose it from your teleop wrapper."
        )

    btns = DualsenseEpisodeButtons(
        btn_pause_toggle=9,      # Options
        btn_exit_early=1,        # Circle
        btn_rerecord_episode=2,  # Triangle
        btn_stop_recording=8,    # Create
        poll_hz=60.0,
        debounce_sec=0.20,
        pause_key="paused",
    )
    events = {
        "stop_recording": False,
        "exit_early": False,
        "rerecord_episode": False,
        "paused": False,
    }

    controller_listener = DualsenseEpisodeListener(ds_js, events, btns)
    controller_listener.start()
    listener = controller_listener

    init_rerun(session_name="jetsonbot_evaluate_smolvla")

    try:
        if not robot.is_connected or not mapped_teleop.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting evaluate loop...")
        recorded_episodes = 0

        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(
                f"Running SmolVLA inference, recording eval episode "
                f"{recorded_episodes + 1} of {NUM_EPISODES}"
            )

            # -------------------------------------------------------------
            # Main evaluation rollout: policy drives the robot
            # -------------------------------------------------------------
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                teleop=mapped_teleop,  # optional/manual override path
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,  # important for SmolVLA
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                vals_to_add_while_policy=["motor_linear.vel", "motor_angular.vel"],
            )

            # -------------------------------------------------------------
            # Reset: use teleop so record_loop still has an action source
            # -------------------------------------------------------------
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

            dataset.save_episode()
            recorded_episodes += 1

    finally:
        log_say("Stop evaluation")
        robot.disconnect()
        mapped_teleop.disconnect()

        if listener is not None:
            listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()