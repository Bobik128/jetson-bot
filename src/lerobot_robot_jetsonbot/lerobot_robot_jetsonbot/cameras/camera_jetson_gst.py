from __future__ import annotations

import cv2

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.utils import get_cv2_rotation

from ..gst_cam import GstCam
from .configuration_jetson_gst import JetsonGstCameraConfig


class JetsonGstCamera(Camera):
    config_class = JetsonGstCameraConfig
    name = "jetson_gst"

    def __init__(self, config: JetsonGstCameraConfig):
        super().__init__(config)
        self.config = config
        self.camera = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.camera is not None and self.camera.alive

    def connect(self) -> None:
        if self.is_connected:
            return

        self.camera = GstCam(
            base_dir=self.config.base_dir,
            frame_size=(self.config.width, self.config.height),
            sensor_id=self.config.sensor_id,
            capture_width=self.config.capture_width,
            capture_height=self.config.capture_height,
            capture_fps=self.config.fps,
            warmup_s=self.config.warmup_s,
            startup_timeout_s=self.config.startup_timeout_s,
            read_timeout_s=self.config.read_timeout_s,
            use_rgb=(self.config.color_mode == ColorMode.RGB),
        )

        if not self.camera.alive:
            self.camera = None
            raise ConnectionError(
                f"Failed to start JetsonGstCamera(sensor_id={self.config.sensor_id})"
            )

        self._is_connected = True

    def read(self):
        if not self.is_connected:
            raise ConnectionError(
                f"JetsonGstCamera(sensor_id={self.config.sensor_id}) is not connected"
            )

        frame = self.camera.get_frame_rgb(timeout_s=self.config.read_timeout_s)

        if self.config.rotation != Cv2Rotation.NO_ROTATION:
            cv_rot = get_cv2_rotation(self.config.rotation)
            if cv_rot is not None:
                frame = cv2.rotate(frame, cv_rot)

        return frame

    def async_read(self):
        if not self.is_connected:
            raise ConnectionError(
                f"JetsonGstCamera(sensor_id={self.config.sensor_id}) is not connected"
            )

        frame = self.camera.get_latest_frame_rgb(max_age_s=max(0.5, self.config.read_timeout_s))

        if self.config.rotation != Cv2Rotation.NO_ROTATION:
            cv_rot = get_cv2_rotation(self.config.rotation)
            if cv_rot is not None:
                frame = cv2.rotate(frame, cv_rot)

        return frame

    def disconnect(self) -> None:
        if self.camera is not None:
            try:
                self.camera.release()
            finally:
                self.camera = None
                self._is_connected = False

    @classmethod
    def find_cameras(cls):
        return []