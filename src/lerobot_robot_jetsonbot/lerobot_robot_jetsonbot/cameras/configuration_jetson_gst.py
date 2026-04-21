from __future__ import annotations

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("jetson_gst")
@dataclass(kw_only=True)
class JetsonGstCameraConfig(CameraConfig):
    sensor_id: int = 0
    base_dir: str = "/tmp/jetsonbot_cam"

    capture_width: int = 640
    capture_height: int = 480

    # This is what Argus/GStreamer is asked for. Argus may still choose the closest supported mode.
    fps: int = 30

    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    color_mode: ColorMode = ColorMode.RGB

    # optional startup stabilization
    warmup_s: float = 0.3
    startup_timeout_s: float = 1.5
    read_timeout_s: float = 1.0