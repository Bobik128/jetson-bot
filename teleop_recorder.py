# teleop_recorder.py
import pygame
import time
import json
import numpy as np

class TeleopRecorder:
    def __init__(self, robot_interface, out_path="demo_log.jsonl"):
        self.iface = robot_interface
        self.out = open(out_path, "w")

        pygame.init()
        pygame.joystick.init()
        self.js = pygame.joystick.Joystick(0)
        self.js.init()

    def _read_joystick(self):
        pygame.event.pump()
        forward_axis = -self.js.get_axis(1)  # left stick Y
        turn_axis    =  self.js.get_axis(0)  # left stick X
        # scale to safe speeds for your robot
        return {
            "linear_vel":  0.3 * forward_axis,
            "angular_vel": 1.0 * turn_axis,
        }

    def record_episode(self, duration_sec=60):
        obs = self.iface.get_observation()
        t_end = time.time() + duration_sec

        while time.time() < t_end:
            action = self._read_joystick()
            new_obs, info = self.iface.step(action)

            sample = {
                "timestamp": time.time(),
                "obs": {
                    "image_front": obs["image_front"].tolist(),  # uint8 -> list for JSON
                    "imu_yaw": obs["imu_yaw"],
                },
                "action": action,
                "info": info,
            }
            self.out.write(json.dumps(sample) + "\n")

            obs = new_obs

        self.out.flush()
