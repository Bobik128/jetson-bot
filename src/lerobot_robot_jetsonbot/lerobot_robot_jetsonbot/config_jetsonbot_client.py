from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig


@dataclass
class ClientCameraSpec:
    width: int
    height: int
    fps: int


@RobotConfig.register_subclass("jetsonbot_client")
@dataclass
class JetsonBotClientConfig(RobotConfig):
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "speed_up": "r",
            "speed_down": "f",
            "quit": "q",
        }
    )

    # client-side observation image keys only
    cameras: dict[str, ClientCameraSpec] = field(
        default_factory=lambda: {
            "front": ClientCameraSpec(width=256, height=144, fps=10),
            # "wrist": ClientCameraSpec(width=256, height=144, fps=30),
        }
    )

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 120