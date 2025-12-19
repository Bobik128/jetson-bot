# sensors.py
import cv2

class Sensors:
    def __init__(self, bus, cam_index=0, out_size=(128,128)):
        self.bus = bus
        self.cap = cv2.VideoCapture(cam_index)
        self.out_w, self.out_h = out_size

    def get_image_front(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        frame_bgr = cv2.resize(
            frame_bgr,
            (self.out_w, self.out_h),
            interpolation=cv2.INTER_AREA
        )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb  # np.uint8 [H,W,3]

    def get_yaw(self):
        state = self.bus.get_state_snapshot()
        return float(state.get("yaw", 0.0))

    def close(self):
        self.cap.release()
