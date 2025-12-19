# quick_cam_preview.py
import cv2, time, os
from datetime import datetime

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR   = os.path.join("camtest_" + SESSION_ID)
os.makedirs(BASE_DIR, exist_ok=True)

from collect_demos import Cam, FRAME_SIZE, JPEG_QUALITY  # if in same folder

cam = Cam(
    frame_size=FRAME_SIZE,
    jpeg_quality=JPEG_QUALITY,
    sensor_id=1,          # flip to 1 for IMX477
    capture_width=640,
    capture_height=480,
    capture_fps=30
)

for i in range(10):
    frame_rgb = cam.get_frame_rgb()
    out_path = os.path.join(BASE_DIR, f"frame_{i:02d}.jpg")
    cam.save_frame(frame_rgb, out_path)
    print("saved", out_path)
    time.sleep(0.1)

cam.release()
print("done, check", BASE_DIR)
