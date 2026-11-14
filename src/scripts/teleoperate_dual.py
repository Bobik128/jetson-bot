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

import time
import pygame
import zmq
from lerobot_robot_jetsonbot.jetsonbot_client import JetsonBotClient
from lerobot_robot_jetsonbot.config_jetsonbot_client import JetsonBotClientConfig
from lerobot_teleoperator_dualsense.dummy_leader_plus_dualsense import DummySOLeaderPlusDualsense
from lerobot_teleoperator_dualsense.config_dummy_teleop_combo import DummySOLeaderPlusDualsenseConfig
from lerobot_teleoperator_dualsense import SOLeaderPlusDualsenseConfig, SOLeaderPlusDualsense, DualsenseTeleopConfig, MappedTeleop
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    # Create the robot and teleoperator configurations
    # robot_config = JetsonBotClientConfig(remote_ip="100.82.250.91", id="jetson-bot")

    context = zmq.Context.instance()
    sim_pub = context.socket(zmq.PUB)
    sim_pub.setsockopt(zmq.SNDHWM, 1)
    sim_pub.bind("tcp://127.0.0.1:5560")

    robot_config = JetsonBotClientConfig(remote_ip="192.168.1.60", id="jetson-bot")
    teleop_config = SOLeaderPlusDualsenseConfig(
        so=SO101LeaderConfig(port="/dev/ttyACM0", use_degrees=False, id="the_leader"),
        ds=DualsenseTeleopConfig(joystick_index=0, axis_forward=1, axis_turn=0, axis_turbo=2),
        allow_partial=False,
    )
    # teleop_config = DummySOLeaderPlusDualsenseConfig(
    #     ds=DualsenseTeleopConfig(joystick_index=0, axis_forward=1, axis_turn=0, axis_turbo=2),
    # )

    # Initialize the robot and teleoperator
    print("Initializing robot client")
    robot = JetsonBotClient(robot_config)
    print("Initializing teleop")
    teleop = SOLeaderPlusDualsense(teleop_config)
    # teleop = DummySOLeaderPlusDualsense(teleop_config)
    mapped_teleop = MappedTeleop(teleop)

    # Connect to the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`

    print("Connecting robot")
    robot.connect()

    print("Connecting teleop")
    mapped_teleop.connect()

    # Init rerun viewer
    init_rerun(session_name="jetsonbot_teleop")

    if not mapped_teleop.is_connected:
        raise ValueError("Teleop is not connected!")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")
        
    print("Starting dual teleop loop...")

    while True:
        t0 = time.perf_counter()

        # Receive observations, including both camera feeds
        observation = robot.get_observation()

        # Read leader arm + PS4 controller
        action = mapped_teleop.get_action()

        # Send commands to the physical robot
        robot.send_action(action)

        # Keep the original camera and state visualization
        log_rerun_data(
            observation=observation,
            action=action,
        )

        # Publish commands to Isaac Lab without blocking
        try:
            sim_pub.send_json(
                {
                    "timestamp_ns": time.monotonic_ns(),
                    "action": {
                        k: float(v)
                        for k, v in action.items()
                    },
                },
                flags=zmq.DONTWAIT,
            )
        except zmq.Again:
            pass  # Skip this simulation update

        precise_sleep(
            max(1.0 / FPS - (time.perf_counter() - t0), 0.0)
        )

if __name__ == "__main__":
    main()