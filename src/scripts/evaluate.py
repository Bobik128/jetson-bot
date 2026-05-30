#!/usr/bin/env python
"""
Unified LeRobot evaluation script for JetsonBot.

This replaces the separate evaluate*.py variants with CLI switches for:
  - ACT vs SmolVLA policy
  - notebook/client vs Jetson/host robot endpoint
  - teleop / no-teleop operation
  - model, dataset, timing, FPS, task, remote IP, robot ID

Examples:
  # ACT from notebook/client to remote Jetson, with dummy DualSense teleop controls
  python evaluate_unified.py \
    --policy act \
    --where client \
    --teleop dummy \
    --hf-model-id Bobik553/jetson-bot_policy-blue_cubes_in_red-NEO-1 \
    --hf-eval-dataset-base-id Bobik553/jetson-bot_blue-block-on-box_eval-help-NEO-PC

  # ACT directly on Jetson/host
  python evaluate_unified.py \
    --policy act \
    --where jetson \
    --teleop dummy \
    --hf-model-id Bobik553/jetson-bot_policy-blue_cubes_in_red-NEO-1 \
    --hf-eval-dataset-base-id Bobik553/jetson-bot_blue-block-on-box_eval-help-NEO

  # SmolVLA directly on Jetson, no teleop
  python evaluate_unified.py \
    --policy smolvla \
    --where jetson \
    --teleop none \
    --hf-model-id Bobik553/jetson-bot_policy-smolvla-blue_on_red \
    --hf-eval-dataset-base-id Bobik553/jetson-bot_blue-block-on-box_eval-help
"""

from __future__ import annotations

import argparse
import re
from typing import Any, Optional

from huggingface_hub import HfApi

from lerobot_robot_jetsonbot.jetsonbot_client import JetsonBotClient
from lerobot_robot_jetsonbot.config_jetsonbot_client import JetsonBotClientConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Teleop imports are only used when --teleop is not "none".
from lerobot_teleoperator_dualsense import (
    SOLeaderPlusDualsenseConfig,
    SOLeaderPlusDualsense,
    DualsenseTeleopConfig,
    MappedTeleop,
    DualsenseEpisodeButtons,
    DualsenseEpisodeListener,
)
from lerobot_teleoperator_dualsense.dummy_leader_plus_dualsense import DummySOLeaderPlusDualsense
from lerobot_teleoperator_dualsense.config_dummy_teleop_combo import DummySOLeaderPlusDualsenseConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig


CLIENT_DEFAULT_REMOTE_IP = "10.98.56.119"
JETSON_DEFAULT_REMOTE_IP = "127.0.0.1"
DEFAULT_ROBOT_ID = "jetson-bot"

DEFAULT_NUM_EPISODES = 60
DEFAULT_FPS = 30
DEFAULT_EPISODE_TIME_SEC = 180
DEFAULT_RESET_TIME_SEC = 20
DEFAULT_TASK_DESCRIPTION = "blue_brick_moving"

DEFAULT_ACT_MODEL_ID = "Bobik553/jetson-bot_policy-blue_cubes_in_red-NEO-1"
DEFAULT_SMOLVLA_MODEL_ID = "Bobik553/jetson-bot_policy-smolvla-blue_on_red"

DEFAULT_ACT_CLIENT_EVAL_DATASET_BASE_ID = "Bobik553/jetson-bot_blue-block-on-box_eval-help-NEO-PC"
DEFAULT_ACT_JETSON_EVAL_DATASET_BASE_ID = "Bobik553/jetson-bot_blue-block-on-box_eval-help-NEO"
DEFAULT_SMOLVLA_EVAL_DATASET_BASE_ID = "Bobik553/jetson-bot_blue-block-on-box_eval-help"


class BoolOrAuto(argparse.Action):
    """argparse action accepting true/false/auto strings."""

    def __call__(self, parser, namespace, values, option_string=None):
        value = values.lower()
        if value not in {"true", "false", "auto"}:
            parser.error(f"{option_string} must be one of: true, false, auto")
        setattr(namespace, self.dest, value)


def next_available_repo_id(base_repo_id: str) -> str:
    """
    Return base_repo_id if it does not exist on HF Hub, otherwise append -N with
    the next available integer suffix.
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

            match = re.fullmatch(re.escape(base_repo_id) + r"-(\d+)", ds.id)
            if match:
                existing.append(int(match.group(1)))
    except Exception:
        existing.append(0)

    n = max(existing) + 1 if existing else 1
    return f"{base_repo_id}-{n}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified JetsonBot LeRobot evaluation script."
    )

    parser.add_argument("--policy", choices=["act", "smolvla"], default="act")
    parser.add_argument(
        "--where",
        choices=["client", "jetson"],
        default="jetson",
        help="client = notebook/PC connecting to remote Jetson; jetson = running directly on host.",
    )
    parser.add_argument(
        "--teleop",
        choices=["dummy", "so101", "none"],
        default="dummy",
        help="dummy = DualSense-only dummy leader; so101 = SO101 + DualSense; none = no teleop/reset controls.",
    )

    parser.add_argument("--hf-model-id", default=None)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--episode-time-sec", type=float, default=DEFAULT_EPISODE_TIME_SEC)
    parser.add_argument("--reset-time-sec", type=float, default=DEFAULT_RESET_TIME_SEC)
    parser.add_argument("--task-description", default=DEFAULT_TASK_DESCRIPTION)
    parser.add_argument("--hf-eval-dataset-base-id", default=None)
    parser.add_argument("--remote-ip", default=None)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)

    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=None,
        help="Default: 4 with teleop, 1 for no-teleop Jetson-style runs.",
    )
    parser.add_argument(
        "--display-data",
        action=BoolOrAuto,
        default="auto",
        metavar="true|false|auto",
        help="Default auto: true with teleop, false without teleop.",
    )
    parser.add_argument(
        "--rerun-session-name",
        default=None,
        help="Default is based on robot id and policy.",
    )

    # Policy/device knobs.
    parser.add_argument(
        "--device",
        default=None,
        help="Optional policy device passed to from_pretrained where supported, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--load-on-cpu-then-cuda",
        action="store_true",
        help="Load policy on CPU, then move to CUDA. Useful for SmolVLA on Jetson.",
    )
    parser.add_argument(
        "--no-inference-mode",
        action="store_true",
        help="Disable torch.inference_mode() wrapper. By default it is used for no-teleop SmolVLA runs.",
    )

    # Dummy / DualSense options.
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--axis-forward", type=int, default=1)
    parser.add_argument("--axis-turn", type=int, default=0)
    parser.add_argument("--axis-turbo", type=int, default=2)

    # SO101 options, used only with --teleop so101.
    parser.add_argument("--so101-port", default="/dev/ttyACM0")
    parser.add_argument("--so101-id", default="the_leader")
    parser.add_argument("--so101-use-degrees", action="store_true")
    parser.add_argument("--allow-partial-teleop", action="store_true")

    # Episode controller button mapping.
    parser.add_argument("--btn-pause-toggle", type=int, default=9)      # Options
    parser.add_argument("--btn-exit-early", type=int, default=1)        # Circle
    parser.add_argument("--btn-rerecord-episode", type=int, default=2)  # Triangle
    parser.add_argument("--btn-stop-recording", type=int, default=8)    # Create
    parser.add_argument("--button-poll-hz", type=float, default=60.0)
    parser.add_argument("--button-debounce-sec", type=float, default=0.20)

    parser.add_argument(
        "--no-push-to-hub",
        action="store_true",
        help="Finalize locally but do not push the eval dataset to HF Hub.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Create eval dataset without videos.",
    )

    args = parser.parse_args()

    if args.hf_model_id is None:
        args.hf_model_id = DEFAULT_SMOLVLA_MODEL_ID if args.policy == "smolvla" else DEFAULT_ACT_MODEL_ID

    if args.hf_eval_dataset_base_id is None:
        if args.policy == "smolvla":
            args.hf_eval_dataset_base_id = DEFAULT_SMOLVLA_EVAL_DATASET_BASE_ID
        elif args.where == "client":
            args.hf_eval_dataset_base_id = DEFAULT_ACT_CLIENT_EVAL_DATASET_BASE_ID
        else:
            args.hf_eval_dataset_base_id = DEFAULT_ACT_JETSON_EVAL_DATASET_BASE_ID

    if args.remote_ip is None:
        args.remote_ip = CLIENT_DEFAULT_REMOTE_IP if args.where == "client" else JETSON_DEFAULT_REMOTE_IP

    if args.image_writer_threads is None:
        args.image_writer_threads = 1 if args.teleop == "none" else 4

    if args.display_data == "auto":
        args.display_data = args.teleop != "none"
    else:
        args.display_data = args.display_data == "true"

    if args.rerun_session_name is None:
        suffix = "noteleop" if args.teleop == "none" else args.teleop
        args.rerun_session_name = f"{args.robot_id}_evaluate_{args.policy}_{suffix}"

    return args


def load_policy(args: argparse.Namespace) -> Any:
    if args.policy == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        if args.device is not None:
            return ACTPolicy.from_pretrained(args.hf_model_id, device=args.device)
        return ACTPolicy.from_pretrained(args.hf_model_id)

    if args.policy == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if args.load_on_cpu_then_cuda:
            print("Loading SmolVLA policy on CPU...")
            policy = SmolVLAPolicy.from_pretrained(args.hf_model_id, device="cpu")
            print("Moving SmolVLA policy to CUDA...")
            policy.to("cuda")
            policy.eval()
            return policy

        if args.device is not None:
            return SmolVLAPolicy.from_pretrained(args.hf_model_id, device=args.device)
        return SmolVLAPolicy.from_pretrained(args.hf_model_id)

    raise ValueError(f"Unsupported policy: {args.policy}")


def make_policy_processors(policy: Any, dataset: LeRobotDataset, args: argparse.Namespace):
    """
    ACT and SmolVLA use slightly different calling conventions in the uploaded scripts.
    """
    if args.policy == "act":
        policy_cfg = policy
        pretrained_path = args.hf_model_id
        return make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=pretrained_path,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)}
            },
        )

    return make_pre_post_processors(
        policy.config,
        args.hf_model_id,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )


def make_mapped_teleop(args: argparse.Namespace) -> tuple[Optional[MappedTeleop], Optional[Any]]:
    if args.teleop == "none":
        return None, None

    ds_cfg = DualsenseTeleopConfig(
        joystick_index=args.joystick_index,
        axis_forward=args.axis_forward,
        axis_turn=args.axis_turn,
        axis_turbo=args.axis_turbo,
    )

    if args.teleop == "dummy":
        teleop_config = DummySOLeaderPlusDualsenseConfig(ds=ds_cfg)
        teleop = DummySOLeaderPlusDualsense(teleop_config)
        return MappedTeleop(teleop), teleop

    if args.teleop == "so101":
        teleop_config = SOLeaderPlusDualsenseConfig(
            so=SO101LeaderConfig(
                port=args.so101_port,
                use_degrees=args.so101_use_degrees,
                id=args.so101_id,
            ),
            ds=ds_cfg,
            allow_partial=args.allow_partial_teleop,
        )
        teleop = SOLeaderPlusDualsense(teleop_config)
        return MappedTeleop(teleop), teleop

    raise ValueError(f"Unsupported teleop mode: {args.teleop}")


def make_events() -> dict[str, bool]:
    return {
        "stop_recording": False,
        "exit_early": False,
        "rerecord_episode": False,
        "paused": False,
    }


def start_dualsense_listener_if_available(
    teleop: Optional[Any], events: dict[str, bool], args: argparse.Namespace
) -> Optional[DualsenseEpisodeListener]:
    if teleop is None:
        return None

    ds_obj = getattr(teleop, "ds", None)
    ds_js = getattr(ds_obj, "joystick", None)
    if ds_js is None:
        raise RuntimeError(
            "Could not access DualSense joystick instance (teleop.ds.joystick is None). "
            "Check joystick_index or expose it from your teleop wrapper."
        )

    buttons = DualsenseEpisodeButtons(
        btn_pause_toggle=args.btn_pause_toggle,
        btn_exit_early=args.btn_exit_early,
        btn_rerecord_episode=args.btn_rerecord_episode,
        btn_stop_recording=args.btn_stop_recording,
        poll_hz=args.button_poll_hz,
        debounce_sec=args.button_debounce_sec,
        pause_key="paused",
    )
    listener = DualsenseEpisodeListener(ds_js, events, buttons)
    listener.start()
    return listener


def call_record_loop(
    *,
    robot: JetsonBotClient,
    events: dict[str, bool],
    args: argparse.Namespace,
    teleop: Optional[MappedTeleop],
    dataset: Optional[LeRobotDataset] = None,
    policy: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    postprocessor: Optional[Any] = None,
    control_time_s: float,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
):
    kwargs: dict[str, Any] = {
        "robot": robot,
        "events": events,
        "fps": args.fps,
        "control_time_s": control_time_s,
        "single_task": args.task_description,
        "display_data": args.display_data,
        "teleop_action_processor": teleop_action_processor,
        "robot_action_processor": robot_action_processor,
        "robot_observation_processor": robot_observation_processor,
    }

    if teleop is not None:
        kwargs["teleop"] = teleop

    if dataset is not None:
        kwargs["dataset"] = dataset

    if policy is not None:
        kwargs.update(
            {
                "policy": policy,
                "preprocessor": preprocessor,
                "postprocessor": postprocessor,
                "vals_to_add_while_policy": ["motor_linear.vel", "motor_angular.vel"],
            }
        )

    record_loop(**kwargs)


def run_eval_loop(
    *,
    robot: JetsonBotClient,
    mapped_teleop: Optional[MappedTeleop],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: LeRobotDataset,
    events: dict[str, bool],
    args: argparse.Namespace,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
):
    recorded_episodes = 0

    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        policy_name = "SmolVLA" if args.policy == "smolvla" else "ACT"
        log_say(
            f"Running {policy_name} inference, recording eval episode "
            f"{recorded_episodes + 1} of {args.num_episodes}"
        )

        call_record_loop(
            robot=robot,
            events=events,
            args=args,
            teleop=mapped_teleop,
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            control_time_s=args.episode_time_sec,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        if (
            not events["stop_recording"]
            and not events["exit_early"]
            and ((recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"])
        ):
            log_say("Reset the environment")
            call_record_loop(
                robot=robot,
                events=events,
                args=args,
                teleop=mapped_teleop,
                control_time_s=args.reset_time_sec,
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


def main() -> None:
    args = parse_args()

    robot_config = JetsonBotClientConfig(
        remote_ip=args.remote_ip,
        id=args.robot_id,
    )
    robot = JetsonBotClient(robot_config)

    mapped_teleop, raw_teleop = make_mapped_teleop(args)
    policy = load_policy(args)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    eval_repo_id = next_available_repo_id(args.hf_eval_dataset_base_id)
    print("Using eval dataset repo_id:", eval_repo_id)
    print("Robot remote_ip:", args.remote_ip)
    print("Robot id:", args.robot_id)
    print("Policy type:", args.policy)
    print("Teleop mode:", args.teleop)

    dataset = LeRobotDataset.create(
        repo_id=eval_repo_id,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=not args.no_videos,
        image_writer_threads=args.image_writer_threads,
    )

    preprocessor, postprocessor = make_policy_processors(policy, dataset, args)

    listener = None

    try:
        robot.connect()
        if mapped_teleop is not None:
            mapped_teleop.connect()

        events = make_events()
        listener = start_dualsense_listener_if_available(raw_teleop, events, args)

        init_rerun(session_name=args.rerun_session_name)

        if not robot.is_connected:
            raise ValueError("Robot is not connected!")
        if mapped_teleop is not None and not mapped_teleop.is_connected:
            raise ValueError("Teleop is not connected!")

        print("Starting evaluate loop...")

        use_inference_mode = (
            args.policy == "smolvla" and args.teleop == "none" and not args.no_inference_mode
        )

        if use_inference_mode:
            import torch

            with torch.inference_mode():
                run_eval_loop(
                    robot=robot,
                    mapped_teleop=mapped_teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    events=events,
                    args=args,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )
        else:
            run_eval_loop(
                robot=robot,
                mapped_teleop=mapped_teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                events=events,
                args=args,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

    finally:
        log_say("Stop evaluation")

        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as exc:
            print(f"robot.disconnect() failed: {exc}")

        try:
            if mapped_teleop is not None and mapped_teleop.is_connected:
                mapped_teleop.disconnect()
        except Exception as exc:
            print(f"mapped_teleop.disconnect() failed: {exc}")

        if listener is not None:
            try:
                listener.stop()
            except Exception as exc:
                print(f"listener.stop() failed: {exc}")

        try:
            dataset.finalize()
        except Exception as exc:
            print(f"dataset.finalize() failed: {exc}")

        if not args.no_push_to_hub:
            try:
                dataset.push_to_hub()
            except Exception as exc:
                print(f"dataset.push_to_hub() failed: {exc}")


if __name__ == "__main__":
    main()
