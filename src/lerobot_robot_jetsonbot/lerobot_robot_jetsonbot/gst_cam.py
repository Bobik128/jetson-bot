#!/usr/bin/env python3
import os
import sys
import time

try:
    import gi
except ModuleNotFoundError:
    sys.path.append("/usr/lib/python3/dist-packages")
    import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
import numpy as np


_GST_INITIALIZED = False


def _ensure_gst_init():
    global _GST_INITIALIZED
    if not _GST_INITIALIZED:
        Gst.init(None)
        _GST_INITIALIZED = True


class GstCam:
    def __init__(
        self,
        base_dir,
        frame_size=(128, 128),
        jpeg_quality=90,
        sensor_id=0,
        capture_width=640,
        capture_height=480,
        capture_fps=30,
    ):
        _ensure_gst_init()

        self.base_dir = base_dir
        self.out_w, self.out_h = frame_size
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self.debug_dump_done = False
        self.alive = False
        self.sensor_id = sensor_id
        self.pipeline = None
        self.appsink = None

        pipeline_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={capture_fps}/1 ! "
            "nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.out_w}, height={self.out_h}, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false"
        )

        print(f"[GstCam sensor {sensor_id}] pipeline: {pipeline_str}")

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            print(f"[GstCam sensor {sensor_id}] ERROR: failed to parse pipeline: {e}")
            return

        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsink is None:
            print(f"[GstCam sensor {sensor_id}] ERROR: no appsink")
            self._shutdown_pipeline()
            return

        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print(f"[GstCam sensor {sensor_id}] ERROR: failed to go PLAYING")
            self._shutdown_pipeline()
            return

        time.sleep(0.3)

        # Prove startup by pulling one frame
        sample = self.appsink.emit("try-pull-sample", int(1.5e9))
        if sample is None:
            print(f"[GstCam sensor {sensor_id}] ERROR: no startup frame")
            self._shutdown_pipeline()
            return

        # We got a frame; keep camera alive
        self.alive = True
        print(f"[GstCam sensor {sensor_id}] READY")

    def _shutdown_pipeline(self):
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsink = None
        self.alive = False

    def get_frame_rgb(self, timeout_s=1.0):
        if not self.alive or self.pipeline is None or self.appsink is None:
            raise RuntimeError(f"Camera {self.sensor_id} not alive")

        sample = self.appsink.emit("try-pull-sample", int(timeout_s * 1e9))
        if sample is None:
            raise RuntimeError(f"Camera {self.sensor_id} timeout: no frame")

        buf = sample.get_buffer()
        caps = sample.get_caps()
        caps_struct = caps.get_structure(0)
        width = caps_struct.get_value("width")
        height = caps_struct.get_value("height")

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError(f"Camera {self.sensor_id}: failed to map buffer")

        try:
            frame_bgr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3))
        finally:
            buf.unmap(mapinfo)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if not self.debug_dump_done:
            dbg_dir = os.path.join(self.base_dir, "debug_cam")
            os.makedirs(dbg_dir, exist_ok=True)

            raw_bgr_path = os.path.join(dbg_dir, f"sample_bgr_cam{self.sensor_id}.jpg")
            rgb_preview_path = os.path.join(dbg_dir, f"sample_rgb_cam{self.sensor_id}.jpg")

            cv2.imwrite(raw_bgr_path, frame_bgr, self.encode_param)
            cv2.imwrite(rgb_preview_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), self.encode_param)

            print(f"[GstCam sensor {self.sensor_id} DEBUG] wrote {raw_bgr_path} and {rgb_preview_path}")
            print("  frame_bgr.shape:", frame_bgr.shape, "dtype:", frame_bgr.dtype)
            print("  first pixel BGR:", frame_bgr[0, 0])

            self.debug_dump_done = True

        return frame_rgb

    def save_frame(self, frame_rgb, out_path):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, frame_bgr, self.encode_param)

    def release(self):
        self._shutdown_pipeline()
        print(f"[GstCam sensor {self.sensor_id}] released")