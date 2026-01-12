#!/usr/bin/env python3
import os
import cv2
import time
import json
import argparse
import socket
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple

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


def black_rgb_frame():
    # FRAME_SIZE is (W, H). Numpy expects (H, W, 3)
    w, h = FRAME_SIZE
    return np.zeros((h, w, 3), dtype=np.uint8)

def expo_curve(x, expo=0.25):
    return (1 - expo) * x + expo * (x**3)

def apply_deadzone(x, dz=0.2):
    if abs(x) < dz:
        return 0.0
    return (abs(x) - dz) / (1.0 - dz) * (1 if x > 0 else -1)

def clamp(x, lo, hi):
    return hi if x > hi else lo if x < lo else x

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("No gamepad found.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Gamepad connected: {js.get_name()}")
    return js

# IMPORTANT: no pygame.event.pump() here; we pump ONCE per loop in main
def read_gamepad_axes(js):
    lx = js.get_axis(1)  # left stick vertical
    rx = js.get_axis(2)  # right stick horizontal

    lx = -lx

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


class ArmU01Receiver:
    """
    Receives u01 arm packets from your leader PC WITHOUT modifying the leader program.

    Key point:
      - Your follower already binds UDP port 5005.
      - This receiver ALSO binds UDP port 5005 using SO_REUSEADDR and (if available) SO_REUSEPORT,
        so both processes can listen on the same UDP port on Linux.

    Leader payload example:
      {"t":..., "unit":"u01", "u":{"shoulder_lift":0.5,"elbow_flex":...}}

    We store u as IDs:
      {2:..., 3:..., 4:..., 6:...}
    """

    NAME_TO_ID = {
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "gripper": 6,
    }

    def __init__(self, bind_ip: str = "0.0.0.0", udp_port: int = 5005, verbose: bool = False):
        self.verbose = bool(verbose)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Allow sharing the same UDP port with follower_receive_u01.py
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        # SO_REUSEPORT is Linux-specific; try it if present
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except Exception:
            pass

        self.sock.bind((bind_ip, int(udp_port)))

        # Non-blocking
        self.sock.setblocking(False)

        self.last_u: Optional[Dict[int, float]] = None
        self.last_rx_time: float = 0.0

        if self.verbose:
            print(f"[ArmU01Receiver] listening on {bind_ip}:{udp_port} (reuseaddr/reuseport if available)")

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def poll(self) -> Optional[Dict[int, float]]:
        """
        Poll once (non-blocking). If a packet arrives, parse and return u_by_id.
        Otherwise return None.
        """
        try:
            data, _ = self.sock.recvfrom(8192)
        except BlockingIOError:
            return None
        except Exception:
            return None

        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            return None

        if msg.get("unit") != "u01":
            return None

        u_field = msg.get("u", {})
        if not isinstance(u_field, dict):
            return None

        out: Dict[int, float] = {}
        for k, v in u_field.items():
            mid: Optional[int] = None

            if isinstance(k, str) and k in self.NAME_TO_ID:
                mid = self.NAME_TO_ID[k]
            elif isinstance(k, str) and k.isdigit():
                mid = int(k)
            elif isinstance(k, int):
                mid = k

            if mid is None:
                continue

            try:
                out[mid] = clamp01(float(v))
            except Exception:
                pass

        if out:
            self.last_u = out
            self.last_rx_time = time.time()
            return out

        return None


class EpisodeLoggerDual:
    """
    Each episode:
      <base_dir>/epXXX/
        frames/
        episode.jsonl

    Required schema fields per step:
      img_front, img_side, dyaw_deg, ax_g, ay_g, az_g, v, w

    Additive field:
      u_arm: {"2":..., "3":..., ...}  (if --arm-log enabled)
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

    def log_step(
        self,
        frame_front_rgb,
        frame_side_rgb,
        dyaw_deg,
        ax_g,
        ay_g,
        az_g,
        v,
        w,
        u_arm: Optional[Dict[int, float]] = None,
    ):
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

        if u_arm is not None:
            record["u_arm"] = {str(k): float(clamp01(v)) for k, v in u_arm.items()}

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
    Overlays REC/PAUSED and episode id, plus step counter.
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

    # Cameras
    parser.add_argument("--front-sensor-id", type=int, default=0)
    parser.add_argument("--side-sensor-id", type=int, default=1)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=30)

    # Preview
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-height", type=int, default=540)
    parser.add_argument("--preview-width", type=int, default=920)

    # Logging verbosity
    parser.add_argument("--print-while-recording", action="store_true",
                        help="If set, print periodic status lines while recording. Default: quiet.")

    # Camera disable
    parser.add_argument("--disable-front-cam", action="store_true")
    parser.add_argument("--disable-side-cam", action="store_true")

    # Arm logging (receiver)
    parser.add_argument("--arm-log", action="store_true",
                        help="Log leader u01 arm packets (received via UDP) into episode.jsonl as u_arm.")
    parser.add_argument("--arm-udp-port", type=int, default=5005,
                        help="UDP port to listen on for u01 arm packets (default: 5005).")
    parser.add_argument("--arm-udp-bind", default="0.0.0.0",
                        help="Bind IP for arm receiver (default: 0.0.0.0).")
    parser.add_argument("--arm-timeout-s", type=float, default=1.0,
                        help="Warn if no arm packets received for this long while recording.")
    parser.add_argument("--arm-verbose", action="store_true")

    args = parser.parse_args()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("../data", session_id)
    os.makedirs(base_dir, exist_ok=True)

    preview_size = (args.preview_width, args.preview_height)

    print("=== COLLECT DEMOS (DUAL CAM) ===")
    print("Session:", session_id)
    print("Base dir:", base_dir)
    print(f"Preview: {args.preview} ({preview_size[0]}x{preview_size[1]})")
    print(f"Arm log: {args.arm_log} (bind {args.arm_udp_bind}:{args.arm_udp_port})")
    print()

    print("[1/6] Init ESP32 serial link...")
    esp = ESP32Link(port=args.port, baud=args.baud)

    print("[2/6] Init IMU...")
    imu = MPU6050GyroYaw()

    print("[3/6] Init cameras...")
    cam_front = None
    cam_side = None

    if args.disable_front_cam:
        print("Front cam disabled by flag.")
    else:
        cam_front = GstCam(
            base_dir=base_dir,
            frame_size=FRAME_SIZE,
            jpeg_quality=JPEG_QUALITY,
            sensor_id=args.front_sensor_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            capture_fps=args.capture_fps,
        )

    if args.disable_side_cam:
        print("Side cam disabled by flag.")
    else:
        cam_side = GstCam(
            base_dir=base_dir,
            frame_size=FRAME_SIZE,
            jpeg_quality=JPEG_QUALITY,
            sensor_id=args.side_sensor_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            capture_fps=args.capture_fps,
        )

    front_alive = (cam_front is not None) and getattr(cam_front, "alive", False)
    side_alive  = (cam_side  is not None) and getattr(cam_side,  "alive", False)
    print(f"cam_front alive: {front_alive}")
    print(f"cam_side  alive: {side_alive}")
    if not front_alive and not side_alive:
        print("[WARN] No cameras alive. Will run with black frames only.")

    print("[4/6] Init gamepad...")
    js = init_gamepad()

    print("[5/6] Init logger...")
    logger = EpisodeLoggerDual(base_dir, session_id)

    print("[6/6] Init preview manager...")
    pv = PreviewManagerDual(enable_preview=args.preview, preview_size=preview_size)

    arm_rx: Optional[ArmU01Receiver] = None
    if args.arm_log:
        arm_rx = ArmU01Receiver(bind_ip=args.arm_udp_bind, udp_port=args.arm_udp_port, verbose=args.arm_verbose)

    recording = False
    last_toggle_state = False
    last_new_ep_state = False

    # Debounce to prevent instant double toggles
    debounce_s = 0.25
    last_toggle_t = 0.0
    last_new_ep_t = 0.0

    # Heartbeat printing
    last_heartbeat = time.time()
    heartbeat_s = 2.0

    print()
    print("Controls:")
    print("  A: toggle recording on/off")
    print("  B: start a new episode (increments ep id)")
    print("  ESC (in preview): exit")
    if args.arm_log:
        print(f"  Arm logging: listening on {args.arm_udp_bind}:{args.arm_udp_port} (shared with follower)")
    print()

    try:
        while True:
            t0 = time.time()

            # Pump ONCE per loop (do not pump in helper functions)
            pygame.event.pump()

            btn_toggle = (js.get_button(BTN_TOGGLE_REC) == 1)
            btn_new_ep = (js.get_button(BTN_NEW_EPISODE) == 1)
            now = time.time()

            # Toggle recording on rising edge + debounce
            if btn_toggle and (not last_toggle_state) and ((now - last_toggle_t) > debounce_s):
                recording = not recording
                last_toggle_t = now
                print(f"Recording: {recording} (ep{logger.episode_id:03d})")
            last_toggle_state = btn_toggle

            # New episode on rising edge + debounce
            if btn_new_ep and (not last_new_ep_state) and ((now - last_new_ep_t) > debounce_s):
                logger.start_new_episode()
                last_new_ep_t = now
                print(f"Episode changed: ep{logger.episode_id:03d}")
            last_new_ep_state = btn_new_ep

            # Base drive controls
            lx, rx, turbo_val = read_gamepad_axes(js)

            turbo_mult = 1.0
            if USE_TURBO and turbo_val >= TURBO_MIN:
                turbo_mult = 1.0 + (turbo_val - TURBO_MIN) / (1.0 - TURBO_MIN) * (TURBO_GAIN - 1.0)

            v = lx * MAX_V * turbo_mult
            w = rx * MAX_W * turbo_mult

            esp.send_cmd(v, w)

            # Arm logging: poll non-blocking
            u_arm: Optional[Dict[int, float]] = None
            if arm_rx is not None:
                got = arm_rx.poll()
                if got is not None:
                    u_arm = got
                else:
                    # keep last known for logging continuity (optional)
                    u_arm = arm_rx.last_u

            # Sensors
            dyaw_deg = imu.update_and_get_yaw_delta_deg()
            ax_g, ay_g, az_g = imu.read_accel_g()

            # Frames: if a cam is missing/dead, use a black frame.
            frame_front_rgb = black_rgb_frame()
            frame_side_rgb  = black_rgb_frame()

            if cam_front is not None and getattr(cam_front, "alive", False):
                try:
                    f = cam_front.get_frame_rgb()
                    if f is not None:
                        frame_front_rgb = f
                except Exception as e:
                    print(f"front cam err: {e}")
                    cam_front.alive = False

            if cam_side is not None and getattr(cam_side, "alive", False):
                try:
                    s = cam_side.get_frame_rgb()
                    if s is not None:
                        frame_side_rgb = s
                except Exception as e:
                    print(f"side cam err: {e}")
                    cam_side.alive = False

            # Warn if arm packets are not arriving while recording
            if recording and arm_rx is not None and args.arm_timeout_s > 0:
                if arm_rx.last_rx_time > 0 and (time.time() - arm_rx.last_rx_time) > args.arm_timeout_s:
                    # print at heartbeat rate only
                    if time.time() - last_heartbeat > heartbeat_s:
                        print(f"[WARN] No arm u01 packets for {time.time()-arm_rx.last_rx_time:.2f}s on {args.arm_udp_bind}:{args.arm_udp_port}")
                        last_heartbeat = time.time()

            # Log only if recording
            if recording:
                logger.log_step(
                    frame_front_rgb,
                    frame_side_rgb,
                    dyaw_deg,
                    ax_g,
                    ay_g,
                    az_g,
                    v,
                    w,
                    u_arm=u_arm,
                )

            # Preview
            pv.update(frame_front_rgb, frame_side_rgb, recording, logger.episode_id, logger.step_idx)

            # Exit via ESC only if preview enabled
            if args.preview:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

            # Console output policy
            if recording and args.print_while_recording:
                if time.time() - last_heartbeat > heartbeat_s:
                    arm_str = ""
                    if u_arm is not None:
                        arm_str = " ".join([f"{k}={u_arm[k]:.2f}" for k in sorted(u_arm.keys())])
                    print(f"[REC ep{logger.episode_id:03d}] step={logger.step_idx} v={v:+.2f} w={w:+.2f} dyaw={dyaw_deg:+.2f} {arm_str}")
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
            if cam_front is not None:
                cam_front.release()
        except Exception:
            pass
        try:
            if cam_side is not None:
                cam_side.release()
        except Exception:
            pass

        try:
            if arm_rx is not None:
                arm_rx.close()
        except Exception:
            pass

        esp.close()
        pv.close()

        print("Session saved at", base_dir)
        print("Episodes are in epXXX/ subfolders.")
        print("Done.")


if __name__ == "__main__":
    main()
