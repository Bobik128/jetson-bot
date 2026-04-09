#!/usr/bin/env python
import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import draccus
import zmq

from .config_jetsonbot import JetsonBotConfig, JetsonBotHostConfig
from .jetsonbot import JetsonBot


@dataclass
class JetsonBotServerConfig:
    """Configuration for the JetsonBot host script."""
    robot: JetsonBotConfig = field(default_factory=JetsonBotConfig)
    host: JetsonBotHostConfig = field(default_factory=JetsonBotHostConfig)


class JetsonBotHost:
    def __init__(self, config: JetsonBotHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


@draccus.wrap()
def main(cfg: JetsonBotServerConfig):
    logging.info("Configuring JetsonBot")
    robot = JetsonBot(cfg.robot)

    logging.info("Connecting JetsonBot")
    robot.connect()

    logging.info("Starting HostAgent")
    host = JetsonBotHost(cfg.host)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")

    try:
        start = time.perf_counter()
        duration = 0.0

        while duration < host.connection_time_s:
            loop_start_time = time.time()
            msg = None

            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))

                allowed = set(robot.action_features.keys())
                data = {k: v for k, v in data.items() if k in allowed}
                _action_sent = robot.send_action(data)

                last_cmd_time = time.time()
                watchdog_active = False

            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")

            except Exception:
                logging.exception("Message handling failed; raw msg=%r", msg)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000.0) and not watchdog_active:
                logging.warning(
                    "Command not received for more than %d milliseconds. Stopping the base.",
                    host.watchdog_timeout_ms,
                )
                watchdog_active = True
                robot.stop_base()

            try:
                last_observation = robot.get_observation()
            except Exception:
                logging.exception("robot.get_observation() failed")
                last_observation = {}

            # Encode image arrays to base64 JPEG
            for cam_key in robot.cameras:
                img = last_observation.get(cam_key)
                if img is None:
                    last_observation[cam_key] = ""
                    continue

                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    ret, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8") if ret else ""
                except Exception:
                    logging.exception("Failed to encode camera frame for key=%s", cam_key)
                    last_observation[cam_key] = ""

            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")
            except Exception:
                logging.exception("Failed to send observation")

            elapsed = time.time() - loop_start_time
            time.sleep(max(1.0 / host.max_loop_freq_hz - elapsed, 0.0))
            duration = time.perf_counter() - start

        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")

    finally:
        print("Shutting down JetsonBot Host.")
        try:
            if getattr(robot, "is_connected", False):
                robot.disconnect()
        except Exception:
            logging.exception("robot.disconnect() failed")
        try:
            host.disconnect()
        except Exception:
            logging.exception("host.disconnect() failed")

    logging.info("Finished JetsonBot cleanly")


if __name__ == "__main__":
    main()