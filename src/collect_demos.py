#!/usr/bin/env python3
import os
import cv2
import time
import json
import argparse
from datetime import datetime

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
      <base_dir>/epXXX/
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

        cv2.imwrite(
            front_path,
            cv2.cvtColor(frame_front_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        cv2.imwrite(
            side_path,
            cv2.cvtColor(frame_side_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )

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


class PreviewManagerDual:
    """
    Two separate windows: 'front' and 'side'
    Overlays REC/PAUSED and episode id, plus optional step counter.
    """

    def __init__(self, enable_preview: bool, preview_size: tuple[int, int]):
        self.enable = enable_preview
        self.preview_size = preview_size

        if self.enable:
            cv2.namedWindow("front", cv2.WINDOW_NORMAL)
            cv2.namedWindow("side",  cv2.WINDOW_NORMAL)
            cv2.resizeWindow("front", self.preview_size[0], self.preview_size[1])
            cv2.resizeWindow("side",  self.preview_size[0], self.preview_size[1])

    @staticmethod
    def _overlay(frame_bgr, recording: bool, episode_id: int, step_idx: int):
        state = "REC" if recording else "PAUSED"
        label = f"{state}  ep{episode_id:03d}  step{step_idx:06d}"

        # Red when recording, yellow when paused
        color = (0, 0, 255) if recording else (0, 255, 255)

        cv2.putText(
            frame_bgr,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame_bgr

    def update(self, frame_front_rgb, frame_side_rgb, recording: bool, episode_id: int, step_idx: int):
        if not self.enable:
            return

        if frame_front_rgb is not None:
            front_bgr = cv2.cvtColor(frame_front_rgb, cv2.COLOR_RGB2BGR)
            front_bgr = cv2.resize(front_bgr, self.preview_size, interpolation=cv2.INTER_LINEAR)
            front_bgr = self._overlay(front_bgr, recording, episode_id, step_idx)
            cv2.imshow("front", front_bgr)

        if frame_side_rgb is not None:
            side_bgr = cv2.cvtColor(frame_side_rgb, cv2.COLOR_RGB2BGR)
            side_bgr = cv2.resize(side_bgr, self.preview_size, interpolation=cv2.INTER_LINEAR)
            side_bgr = self._overlay(side_bgr, recording, episode_id, step_idx)
            cv2.imshow("side", side_bgr)

        cv2.waitKey(1)

    def close(self):
        if self.enable:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()

    # Serial
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=115200)

    # Cameras (CSI via nvarguscamerasrc)
    parser.add_argument("--front-sensor-id", type=int, default=0)
    parser.add_argument("--side-sensor-id", type=int, default=1)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=30)

    # Preview (two windows)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-height", type=int, default=540)
    parser.add_argument("--preview-width", type=int, default=920)

    # Logging verbosity behavior
    parser.add_argument("--print-while-recording", action="store_true",
                        help="If set, print periodic status lines while recording. Default: quiet.")

    args = parser.parse_args()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("../data", session_id)
    os.makedirs(base_dir, exist_ok=True)

    preview_size = (args.preview_width, args.preview_height)

    print("=== COLLECT DEMOS (DUAL CAM) ===")
    print("Session:", session_id)
    print("Base dir:", base_dir)
    print(f"Preview: {args.preview} ({preview_size[0]}x{preview_size[1]})")
    print()

    print("[1/6] Init ESP32 serial link...")
    esp = ESP32Link(port=args.port, baud=args.baud)

    print("[2/6] Init IMU...")
    imu = MPU6050GyroYaw()

    print("[3/6] Init cameras...")
    cam_front = GstCam(
        base_dir=base_dir,
        frame_size=FRAME_SIZE,
        jpeg_quality=JPEG_QUALITY,
        sensor_id=args.front_sensor_id,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        capture_fps=args.capture_fps,
    )
    cam_side = GstCam(
        base_dir=base_dir,
        frame_size=FRAME_SIZE,
        jpeg_quality=JPEG_QUALITY,
        sensor_id=args.side_sensor_id,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        capture_fps=args.capture_fps,
    )

    print(f"cam_front alive: {cam_front.alive}")
    print(f"cam_side  alive: {cam_side.alive}")

    if not cam_front.alive and not cam_side.alive:
        raise RuntimeError("Neither camera came up. Cannot record.")

    print("[4/6] Init gamepad...")
    js = init_gamepad()

    print("[5/6] Init logger...")
    logger = EpisodeLoggerDual(base_dir, session_id)

    print("[6/6] Init preview manager...")
    pv = PreviewManagerDual(enable_preview=args.preview, preview_size=preview_size)

    recording = False
    last_toggle_state = False
    last_new_ep_state = False

    # Print policy:
    # - Always print on state changes.
    # - While recording: default QUIET (no per-step spam). Optional via flag.
    # - While paused: print a slow heartbeat so you know it's alive.
    last_heartbeat = time.time()
    heartbeat_s = 2.0

    print()
    print("Controls:")
    print("  A: toggle recording on/off")
    print("  B: start a new episode (increments ep id)")
    print("  ESC (in preview): exit")
    print()

    try:
        while True:
            t0 = time.time()

            pygame.event.pump()
            btn_toggle = js.get_button(BTN_TOGGLE_REC) == 1
            btn_new_ep = js.get_button(BTN_NEW_EPISODE) == 1

            # Toggle recording on rising edge
            if btn_toggle and not last_toggle_state:
                recording = not recording
                print(f"Recording: {recording} (ep{logger.episode_id:03d})")
            last_toggle_state = btn_toggle

            # New episode on rising edge
            if btn_new_ep and not last_new_ep_state:
                logger.start_new_episode()
                print(f"Episode changed: ep{logger.episode_id:03d}")
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

            # Frames (already resized to FRAME_SIZE by GstCam)
            frame_front_rgb = None
            frame_side_rgb  = None

            if cam_front.alive:
                try:
                    frame_front_rgb = cam_front.get_frame_rgb()
                except Exception as e:
                    print(f"front cam err: {e}")
                    cam_front.alive = False

            if cam_side.alive:
                try:
                    frame_side_rgb = cam_side.get_frame_rgb()
                except Exception as e:
                    print(f"side cam err: {e}")
                    cam_side.alive = False

            # If both dead, stop
            if (not cam_front.alive) and (not cam_side.alive):
                raise RuntimeError("Both cameras are dead (no frames).")

            # Log only if recording AND frames exist
            if recording and (frame_front_rgb is not None) and (frame_side_rgb is not None):
                logger.log_step(frame_front_rgb, frame_side_rgb, dyaw_deg, ax_g, ay_g, az_g, v, w)

            # Preview overlay always shows current state + ep + step_idx
            pv.update(frame_front_rgb, frame_side_rgb, recording, logger.episode_id, logger.step_idx)

            # Exit via ESC only if preview enabled
            if args.preview:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

            # Console output policy
            if recording and args.print_while_recording:
                # light status line (not per-frame spam if you don't want it)
                # print at ~2 Hz
                if time.time() - last_heartbeat > heartbeat_s:
                    print(f"[REC ep{logger.episode_id:03d}] step={logger.step_idx} v={v:+.2f} w={w:+.2f} dyaw={dyaw_deg:+.2f}")
                    last_heartbeat = time.time()

            if (not recording) and (time.time() - last_heartbeat > heartbeat_s):
                print(f"[PAUSED ep{logger.episode_id:03d}] (press A to record)")
                last_heartbeat = time.time()

            dt = time.time() - t0
            time.sleep(max(0.0, DT - dt))

    except KeyboardInterrupt:
        print("\nStopping data collection... sending zero velocity.")
        try:
            esp.send_cmd(0.0, 0.0)
        except Exception:
            pass

    finally:
        logger.close()
        try:
            cam_front.release()
            cam_side.release()
        except Exception:
            pass
        esp.close()
        pv.close()
        print("Session saved at", base_dir)
        print("Episodes are in epXXX/ subfolders.")
        print("Done.")


if __name__ == "__main__":
    main()
