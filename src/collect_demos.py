#!/usr/bin/env python3
import os
import cv2
import time
import json
import argparse
from datetime import datetime

import serial
import pygame

from gst_cam import GstCam

from shared.constants import FRAME_SIZE, JPEG_QUALITY, MAX_V, MAX_W
from shared.esp32_link import ESP32Link
from shared.imu_mpu6050 import MPU6050GyroYaw


########################################
# Recording / control config
########################################

SEND_HZ     = 30.0
DT          = 1.0 / SEND_HZ

DEADZONE    = 0.20
EXPO        = 0.25

USE_TURBO   = True
TURBO_AXIS  = 4
TURBO_MIN   = 0.6
TURBO_GAIN  = 1.75

BTN_TOGGLE_REC  = 0   # A
BTN_NEW_EPISODE = 1   # B


def expo_curve(x, expo=0.25):
    return (1 - expo) * x + expo * (x**3)

def apply_deadzone(x, dz=0.2):
    if abs(x) < dz:
        return 0.0
    return (abs(x) - dz) / (1.0 - dz) * (1 if x > 0 else -1)

def clamp(x, lo, hi):
    return hi if x > hi else lo if x < lo else x

def init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("No gamepad found.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Gamepad connected: {js.get_name()}")
    return js

def read_gamepad_axes(js):
    pygame.event.pump()

    lx = js.get_axis(1)  # left stick vertical
    rx = js.get_axis(2)  # right stick horizontal

    lx = -lx
    rx = rx

    lx = apply_deadzone(lx, DEADZONE)
    rx = apply_deadzone(rx, DEADZONE)

    lx = expo_curve(lx, EXPO)
    rx = expo_curve(rx, EXPO)

    turbo_val = 0.0
    if USE_TURBO:
        try:
            turbo_val = (js.get_axis(TURBO_AXIS) + 1) * 0.5
            turbo_val = clamp(turbo_val, 0.0, 1.0)
        except Exception:
            turbo_val = 0.0

    return lx, rx, turbo_val


class EpisodeLoggerDual:
    """
    Each episode:
      data/<SESSION>/epXXX/
        frames/
        episode.jsonl

    Required schema fields per step:
      img_front, img_side, dyaw_deg, ax_g, ay_g, az_g, v, w
    """

    def __init__(self, base_dir, session_id):
        self.base_dir = base_dir
        self.session_id = session_id
        self.episode_id = 0
        self.step_idx = 0

        self.ep_dir = None
        self.frames_dir = None
        self.meta_path = None
        self.meta_file = None

        self._open_new_files_for_episode()

    def _open_new_files_for_episode(self):
        self.ep_dir = os.path.join(self.base_dir, f"ep{self.episode_id:03d}")
        self.frames_dir = os.path.join(self.ep_dir, "frames")
        self.meta_path = os.path.join(self.ep_dir, "episode.jsonl")
        os.makedirs(self.frames_dir, exist_ok=True)

        if self.meta_file is not None:
            try:
                self.meta_file.close()
            except Exception:
                pass

        self.meta_file = open(self.meta_path, "a", buffering=1)
        print(f"Logging to: {self.meta_path}")

    def start_new_episode(self):
        self.episode_id += 1
        self.step_idx = 0
        print(f"--- New episode started: episode_id={self.episode_id} ---")
        self._open_new_files_for_episode()

    def log_step(self, frame_front_rgb, frame_side_rgb, dyaw_deg, ax_g, ay_g, az_g, v, w):
        ts = time.time()

        front_name = f"front_{self.step_idx:06d}.jpg"
        side_name  = f"side_{self.step_idx:06d}.jpg"

        front_rel = os.path.join("frames", front_name)
        side_rel  = os.path.join("frames", side_name)

        front_path = os.path.join(self.frames_dir, front_name)
        side_path  = os.path.join(self.frames_dir, side_name)

        cv2.imwrite(front_path, cv2.cvtColor(frame_front_rgb, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        cv2.imwrite(side_path,  cv2.cvtColor(frame_side_rgb,  cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        record = {
            "session": self.session_id,
            "episode": self.episode_id,
            "step": self.step_idx,
            "t": ts,
            "img_front": front_rel,
            "img_side": side_rel,
            "dyaw_deg": float(dyaw_deg),
            "ax_g": float(ax_g),
            "ay_g": float(ay_g),
            "az_g": float(az_g),
            "v": float(v),
            "w": float(w),
        }

        self.meta_file.write(json.dumps(record) + "\n")
        self.step_idx += 1

    def close(self):
        if self.meta_file is not None:
            try:
                self.meta_file.close()
            except Exception:
                pass
            self.meta_file = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--front", default="/dev/video0")
    parser.add_argument("--side",  default="/dev/video1")

    # Restored preview args
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview window (OpenCV GUI).")
    parser.add_argument("--preview-height", type=int, default=540,
                        help="Preview window height in pixels (only used with --preview).")
    parser.add_argument("--preview-width", type=int, default=920,
                        help="Preview window width in pixels (only used with --preview).")

    args = parser.parse_args()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir   = os.path.join("../data", session_id)
    os.makedirs(base_dir, exist_ok=True)

    if args.preview:
        cv2.namedWindow("Front | Side", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Front | Side", args.preview_width, args.preview_height)

    print("Initializing ESP32 serial link...")
    esp = ESP32Link(port=args.port, baud=args.baud)

    print("Initializing IMU...")
    imu = MPU6050GyroYaw()

    print("Initializing cameras...")
    cam_front = GstCam(device=args.front, width=FRAME_SIZE[0], height=FRAME_SIZE[1], fps=30)
    cam_side  = GstCam(device=args.side,  width=FRAME_SIZE[0], height=FRAME_SIZE[1], fps=30)

    print("Initializing gamepad...")
    js = init_gamepad()

    logger = EpisodeLoggerDual(base_dir, session_id)

    recording = False
    last_toggle_state = False
    last_new_ep_state = False

    print("Ready. Press A to toggle recording. Press B to start a new episode.")
    try:
        while True:
            t0 = time.time()

            pygame.event.pump()
            btn_toggle = js.get_button(BTN_TOGGLE_REC) == 1
            btn_new_ep = js.get_button(BTN_NEW_EPISODE) == 1

            if btn_toggle and not last_toggle_state:
                recording = not recording
                print(f"Recording: {recording}")
            last_toggle_state = btn_toggle

            if btn_new_ep and not last_new_ep_state:
                logger.start_new_episode()
            last_new_ep_state = btn_new_ep

            lx, rx, turbo_val = read_gamepad_axes(js)

            turbo_mult = 1.0
            if USE_TURBO and turbo_val >= TURBO_MIN:
                turbo_mult = 1.0 + (turbo_val - TURBO_MIN) / (1.0 - TURBO_MIN) * (TURBO_GAIN - 1.0)

            v = lx * MAX_V * turbo_mult
            w = rx * MAX_W * turbo_mult

            esp.send_cmd(v, w)

            dyaw_deg = imu.update_and_get_yaw_delta_deg()
            ax_g, ay_g, az_g = imu.read_accel_g()

            frame_front_rgb = cam_front.get_frame_rgb()
            frame_side_rgb  = cam_side.get_frame_rgb()

            if frame_front_rgb is None or frame_side_rgb is None:
                print("Camera frame missing; skipping.")
            else:
                if recording:
                    logger.log_step(frame_front_rgb, frame_side_rgb, dyaw_deg, ax_g, ay_g, az_g, v, w)

                if args.preview:
                    combined = cv2.hconcat([
                        cv2.cvtColor(frame_front_rgb, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(frame_side_rgb,  cv2.COLOR_RGB2BGR),
                    ])

                    # Resize only for display (does not affect saved images)
                    disp = cv2.resize(combined, (args.preview_width, args.preview_height))

                    cv2.imshow("Front | Side", disp)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            dt = time.time() - t0
            time.sleep(max(0.0, DT - dt))

    finally:
        logger.close()
        cam_front.release()
        cam_side.release()
        esp.close()

        if args.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()