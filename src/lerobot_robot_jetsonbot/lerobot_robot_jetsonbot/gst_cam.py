from __future__ import annotations

import sys
import time
from threading import Event, Lock, Thread

try:
    import gi
except ModuleNotFoundError:
    for p in (
        "/usr/lib/python3/dist-packages",
        "/usr/lib64/python3.12/site-packages",
        "/usr/lib64/python3.13/site-packages",
        "/usr/lib/python3.12/site-packages",
        "/usr/lib/python3.13/site-packages",
    ):
        if p not in sys.path:
            sys.path.append(p)
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
    """
    Jetson CSI camera via nvarguscamerasrc + appsink.

    Background thread continuously pulls frames and stores only the latest one.
    Public API:
        - get_frame_rgb(timeout_s=...)
        - get_latest_frame_rgb(max_age_s=...)
        - release()
    """

    def __init__(
        self,
        base_dir: str | None = None,
        frame_size=(128, 128),
        sensor_id: int = 0,
        capture_width: int = 640,
        capture_height: int = 480,
        capture_fps: int = 30,
        warmup_s: float = 0.3,
        startup_timeout_s: float = 1.5,
        read_timeout_s: float = 0.25,
        use_rgb: bool = True,
    ):
        _ensure_gst_init()

        self.base_dir = base_dir
        self.out_w, self.out_h = frame_size
        self.sensor_id = sensor_id
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.capture_fps = capture_fps
        self.read_timeout_s = read_timeout_s
        self.use_rgb = use_rgb

        self.pipeline = None
        self.appsink = None
        self.alive = False

        self._thread: Thread | None = None
        self._stop_event = Event()
        self._new_frame_event = Event()
        self._frame_lock = Lock()

        self._latest_frame = None
        self._latest_timestamp = None
        self._consecutive_failures = 0

        # Keep output directly at network/input resolution to avoid another resize later.
        # Keep BGR in appsink because OpenCV receives it naturally.
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
            f"framerate={capture_fps}/1, format=NV12 ! "
            "nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.out_w}, height={self.out_h}, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink name=appsink emit-signals=false max-buffers=1 drop=true sync=false"
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

        # Explicitly reinforce queue behavior.
        self.appsink.set_property("emit-signals", False)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print(f"[GstCam sensor {sensor_id}] ERROR: failed to go PLAYING")
            self._shutdown_pipeline()
            return

        time.sleep(warmup_s)

        # Startup validation: require one frame before claiming success.
        sample = self.appsink.emit("try-pull-sample", int(startup_timeout_s * 1e9))
        if sample is None:
            print(f"[GstCam sensor {sensor_id}] ERROR: no startup frame")
            self._shutdown_pipeline()
            return

        frame = self._sample_to_ndarray(sample)
        if frame is None:
            print(f"[GstCam sensor {sensor_id}] ERROR: failed to decode startup frame")
            self._shutdown_pipeline()
            return

        with self._frame_lock:
            self._latest_frame = frame
            self._latest_timestamp = time.perf_counter()

        self.alive = True
        self._thread = Thread(target=self._reader_loop, name=f"gstcam-{sensor_id}", daemon=True)
        self._thread.start()

        print(f"[GstCam sensor {sensor_id}] READY")

    def _sample_to_ndarray(self, sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()
        caps_struct = caps.get_structure(0)

        width = caps_struct.get_value("width")
        height = caps_struct.get_value("height")

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            # Make a real owned copy so frame stays valid after unmap.
            frame_bgr = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=mapinfo.data,
            ).copy()
        finally:
            buf.unmap(mapinfo)

        if self.use_rgb:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame = frame_bgr

        return frame

    def _reader_loop(self):
        timeout_ns = int(self.read_timeout_s * 1e9)

        while not self._stop_event.is_set():
            if not self.alive or self.appsink is None:
                break

            try:
                sample = self.appsink.emit("try-pull-sample", timeout_ns)
                if sample is None:
                    self._consecutive_failures += 1
                    continue

                frame = self._sample_to_ndarray(sample)
                if frame is None:
                    self._consecutive_failures += 1
                    continue

                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_timestamp = time.perf_counter()

                self._new_frame_event.set()
                self._consecutive_failures = 0

            except Exception as e:
                self._consecutive_failures += 1
                if self._consecutive_failures <= 10:
                    print(f"[GstCam sensor {self.sensor_id}] reader warning: {e}")
                else:
                    print(f"[GstCam sensor {self.sensor_id}] reader fatal: {e}")
                    break

        self.alive = False

    def get_frame_rgb(self, timeout_s: float = 1.0):
        """
        Wait for a reasonably fresh frame.
        Compatible with your current JetsonGstCamera.read().
        """
        deadline = time.perf_counter() + timeout_s

        while time.perf_counter() < deadline:
            with self._frame_lock:
                frame = self._latest_frame
                ts = self._latest_timestamp

            if frame is not None and ts is not None:
                age = time.perf_counter() - ts
                # accept frame if it's fresh enough
                if age <= timeout_s:
                    return frame

            self._new_frame_event.wait(timeout=0.01)
            self._new_frame_event.clear()

        raise RuntimeError(f"Camera {self.sensor_id} timeout: no fresh frame")

    def get_latest_frame_rgb(self, max_age_s: float = 0.5):
        """
        Non-blocking-ish peek of the latest frame.
        Better fit for observation loops.
        """
        with self._frame_lock:
            frame = self._latest_frame
            ts = self._latest_timestamp

        if frame is None or ts is None:
            raise RuntimeError(f"Camera {self.sensor_id}: no frame available yet")

        age = time.perf_counter() - ts
        if age > max_age_s:
            raise RuntimeError(
                f"Camera {self.sensor_id}: latest frame too old ({age * 1e3:.1f} ms)"
            )

        return frame

    def _shutdown_pipeline(self):
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass

        self.pipeline = None
        self.appsink = None
        self.alive = False

    def release(self):
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._thread = None
        self._shutdown_pipeline()

        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = None

        print(f"[GstCam sensor {self.sensor_id}] released")