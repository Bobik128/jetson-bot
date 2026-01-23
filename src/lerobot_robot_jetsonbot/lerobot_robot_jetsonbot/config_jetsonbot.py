from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from lerobot.robots.config import RobotConfig

def jetsonbot_cameras_config() -> dict[str, CameraConfig]:
    # Capture at a supported sensor mode, then scale to 256x144 for LeRobot.
    front = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, width=256, height=144, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1 sync=false"
    )

    wrist = (
        "nvarguscamerasrc sensor-id=1 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, width=256, height=144, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1 sync=false"
    )

    return {
        "front": OpenCVCameraConfig(index_or_path=front, fps=30, width=256, height=144),
        "wrist": OpenCVCameraConfig(index_or_path=wrist, fps=30, width=256, height=144),
    }

@RobotConfig.register_subclass("jetsonbot")
@dataclass
class JetsonBotConfig(RobotConfig):
    port: str = "/dev/ttyACM0"  # port to connect to the bus

    esp_port: str = "/dev/ttyTHS1" # port to connect to the esp32
    esp_baud: int = 115200  # baud rate for esp32 communication
    esp_timeout: float = 0.1  # timeout for esp communication in seconds

    max_v_mps: float = 0.3  # maximum linear velocity in meters per second
    max_w_radps: float = 2.0  # maximum angular velocity in radians

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=jetsonbot_cameras_config)


@dataclass
class JetsonBotHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 30

    # Watchdog: stop the robot if no command is received for over 0.5 seconds.
    watchdog_timeout_ms: int = 500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("jetsonbot_client")
@dataclass
class JetsonBotClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=jetsonbot_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5