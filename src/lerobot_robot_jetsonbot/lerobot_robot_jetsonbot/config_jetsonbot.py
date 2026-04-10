from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

from .cameras.configuration_jetson_gst import JetsonGstCameraConfig


def jetsonbot_cameras_config():
    return {
        "front": JetsonGstCameraConfig(
            sensor_id=0,
            width=256,
            height=144,
            capture_width=960,
            capture_height=540,
            fps=30,
            base_dir="/tmp/jetsonbot_cam",
        ),
        "wrist": JetsonGstCameraConfig(
            sensor_id=1,
            width=256,
            height=144,
            capture_width=960,
            capture_height=540,
            fps=30,
            base_dir="/tmp/jetsonbot_cam",
        ),
    }


@RobotConfig.register_subclass("jetsonbot")
@dataclass
class JetsonBotConfig(RobotConfig):
    port: str = "/dev/ttyACM0"

    esp_port: str = "/dev/ttyTHS1"
    esp_baud: int = 115200
    esp_timeout: float = 0.1

    max_v_mps: float = 0.3
    max_w_radps: float = 2.0

    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict = field(default_factory=jetsonbot_cameras_config)


@dataclass
class JetsonBotHostConfig:
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connection_time_s: int = 300000
    watchdog_timeout_ms: int = 5000
    max_loop_freq_hz: int = 30


# @RobotConfig.register_subclass("jetsonbot_client")
# @dataclass
# class JetsonBotClientConfig(RobotConfig):
#     remote_ip: str
#     port_zmq_cmd: int = 5555
#     port_zmq_observations: int = 5556

#     teleop_keys: dict[str, str] = field(
#         default_factory=lambda: {
#             "forward": "w",
#             "backward": "s",
#             "left": "a",
#             "right": "d",
#             "speed_up": "r",
#             "speed_down": "f",
#             "quit": "q",
#         }
#     )

#     polling_timeout_ms: int = 15
#     connect_timeout_s: int = 30