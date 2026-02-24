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

from lerobot_robot_jetsonbot import JetsonBotClient, JetsonBotClientConfig
from lerobot_teleoperator_dualsense import SOLeaderPlusDualsenseConfig, SOLeaderPlusDualsense, DualsenseTeleopConfig, MappedTeleop
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    # Create the robot and teleoperator configurations
    robot_config = JetsonBotClientConfig(remote_ip="100.82.250.91", id="jetson-bot")
    # teleop_arm_config = SO101LeaderConfig(port="/dev/ttyACM0", id="the_leader")
    # dualsense_config = DualsenseTeleopConfig()
    teleop_config = SOLeaderPlusDualsenseConfig(
        so=SO101LeaderConfig(port="/dev/ttyACM0", use_degrees=False, id="the_leader"),
        ds=DualsenseTeleopConfig(joystick_index=0, axis_forward=1, axis_turn=0, axis_turbo=2),
        allow_partial=False,
    )

    # Initialize the robot and teleoperator
    robot = JetsonBotClient(robot_config)
    teleop = SOLeaderPlusDualsense(teleop_config)
    mapped_teleop = MappedTeleop(teleop)

    # Connect to the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    mapped_teleop.connect()

    # Init rerun viewer
    init_rerun(session_name="jetsonbot_teleop")

    if not robot.is_connected or not mapped_teleop.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        observation = robot.get_observation()

    # By default outputs:
    #   - motor_linear.vel  (m/s)
    #   - motor_angular.vel (rad/s)

        action = mapped_teleop.get_action()

        # base_action = dualsense.get_action()

        # action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action


        # Send action to robot
        _ = robot.send_action(action)

        # Visualize
        log_rerun_data(observation=observation, action=action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()