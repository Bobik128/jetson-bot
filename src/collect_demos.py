#!/usr/bin/env python3
import os
import cv2
import time
import json
import argparse
import socket
import numpy as np

from datetime import datetime
from typing import Dict, List, Optional, Tuple

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


class ArmControlUDP:
    """
    Sends arm u01 packets over UDP to follower_receive_u01.py.

    - Maintains internal u targets for each servo id in [0,1]
    - Updates u from gamepad (hold-to-move) at rate u_per_s
    - Sends {"unit":"u01","u":{"2":0.5,...}} to UDP target

    Default mapping:
      - Servo 2: D-pad up/down (hat y)
      - Servo 3: D-pad left/right (hat x)
      - Servo 4: LB / RB (buttons 4/5)
      - Servo 6: triggers (tries several LT/RT axis pairs), RT closes (u+), LT opens (u-)
      - Servo 1 (optional): right stick horizontal (axis servo1_axis), proportional nudging
    """

    _TRIGGER_AXIS_CANDIDATES: List[Tuple[int, int]] = [
        (2, 5),
        (4, 5),
        (3, 4),
        (2, 3),
    ]

    def __init__(
        self,
        udp_ip: str = "127.0.0.1",
        udp_port: int = 5005,
        ids: Optional[List[int]] = None,
        *,
        send_hz: float = 30.0,
        u_per_s: float = 0.60,
        u_init: Optional[Dict[int, float]] = None,
        include_servo1: bool = False,
        servo1_axis: int = 3,
        servo1_init: float = 0.5,
        deadzone: float = 0.15,
        verbose: bool = False,
    ):
        if ids is None:
            ids = [2, 3, 4, 6]

        self.udp_ip = str(udp_ip)
        self.udp_port = int(udp_port)
        self.addr = (self.udp_ip, self.udp_port)

        self.ids: List[int] = list(ids)
        self.include_servo1 = bool(include_servo1)
        if self.include_servo1 and 1 not in self.ids:
            self.ids = [1] + self.ids

        self.send_hz = float(send_hz)
        self.dt = 1.0 / max(1.0, self.send_hz)
        self.u_per_s = float(u_per_s)

        self.deadzone = float(deadzone)
        self.servo1_axis = int(servo1_axis)
        self.verbose = bool(verbose)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.u: Dict[int, float] = {mid: 0.5 for mid in self.ids}
        if u_init:
            for k, v in u_init.items():
                try:
                    self.u[int(k)] = clamp01(float(v))
                except Exception:
                    pass

        if self.include_servo1:
            self.u[1] = clamp01(float(servo1_init))

        self._last_send_t = 0.0

        if self.verbose:
            print(f"[ArmControlUDP] UDP -> {self.udp_ip}:{self.udp_port} ids={self.ids}")

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def _apply_deadzone(self, x: float) -> float:
        if abs(x) < self.deadzone:
            return 0.0
        s = 1.0 if x > 0 else -1.0
        return s * (abs(x) - self.deadzone) / (1.0 - self.deadzone)

    def _nudge(self, mid: int, direction: float):
        if mid not in self.u:
            return
        self.u[mid] = clamp01(self.u[mid] + float(direction) * self.u_per_s * self.dt)

    def update_from_gamepad(self, js) -> Dict[int, float]:
        pygame.event.pump()

        # D-pad (hat) -> servo 2/3
        try:
            hx, hy = js.get_hat(0)
        except Exception:
            hx, hy = 0, 0

        if 2 in self.u and hy != 0:
            self._nudge(2, +1.0 if hy > 0 else -1.0)

        if 3 in self.u and hx != 0:
            self._nudge(3, +1.0 if hx > 0 else -1.0)

        # LB/RB -> servo 4
        if 4 in self.u:
            try:
                lb = js.get_button(4) == 1
            except Exception:
                lb = False
            try:
                rb = js.get_button(5) == 1
            except Exception:
                rb = False

            if lb and not rb:
                self._nudge(4, -1.0)
            elif rb and not lb:
                self._nudge(4, +1.0)

        # Triggers -> gripper (servo 6)
        if 6 in self.u:
            lt = 0.0
            rt = 0.0
            found = False
            for lt_axis, rt_axis in self._TRIGGER_AXIS_CANDIDATES:
                try:
                    lt = (js.get_axis(lt_axis) + 1.0) * 0.5
                    rt = (js.get_axis(rt_axis) + 1.0) * 0.5
                    found = True
                    break
                except Exception:
                    continue
            if not found:
                lt, rt = 0.0, 0.0

            lt = clamp01(lt)
            rt = clamp01(rt)

            g_dir = rt - lt  # RT close (u+), LT open (u-)
            if abs(g_dir) > 0.05:
                self._nudge(6, +1.0 if g_dir > 0 else -1.0)

        # Optional servo 1 -> right stick horizontal
        if self.include_servo1 and 1 in self.u:
            try:
                x = float(js.get_axis(self.servo1_axis))
                x = self._apply_deadzone(x)
                if abs(x) > 0.001:
                    self._nudge(1, x)
            except Exception:
                pass

        return dict(self.u)

    def build_packet(self, u_by_id: Dict[int, float]) -> bytes:
        msg = {
            "unit": "u01",
            "u": {str(k): float(clamp01(v)) for k, v in u_by_id.items()},
        }
        return json.dumps(msg).encode("utf-8")

    def send(self, u_by_id: Dict[int, float], *, rate_limit: bool = True) -> None:
        if rate_limit:
            now = time.time()
            if (now - self._last_send_t) < self.dt:
                return
            self._last_send_t = now

        data = self.build_packet(u_by_id)
        self.sock.sendto(data, self.addr)


class EpisodeLoggerDual:
    """
    Each episode:
      <base_dir>/epXXX/
        frames/
        episode.jsonl

    Required schema fields per step:
      img_front, img_side, dyaw_deg, ax_g, ay_g, az_g, v, w

    Additive field when arm is enabled:
      u_arm: {"2":..., "3":..., ...}
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

    def log_step(self, frame_front_rgb, frame_side_rgb, dyaw_deg, ax_g, ay_g, az_g, v, w, u_arm: Optional[Dict[int, float]] = None):
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

    # Arm control (UDP to follower_receive_u01.py)
    parser.add_argument("--arm-enable", action="store_true",
                        help="Enable arm control: send u01 UDP packets and log u_arm.")
    parser.add_argument("--arm-udp-ip", default="127.0.0.1")
    parser.add_argument("--arm-udp-port", type=int, default=5005)
    parser.add_argument("--arm-ids", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--arm-u-per-s", type=float, default=0.60)
    parser.add_argument("--arm-deadzone", type=float, default=0.15)
    parser.add_argument("--arm-verbose", action="store_true")

    # Optional servo 1 support
    parser.add_argument("--arm-include-servo1", action="store_true",
                        help="Include servo ID 1 in outgoing u01 packets.")
    parser.add_argument("--arm-servo1-axis", type=int, default=3,
                        help="Pygame axis index for servo 1 control (default: 3).")
    parser.add_argument("--arm-servo1-init", type=float, default=0.5)

    # Camera disable
    parser.add_argument("--disable-front-cam", action="store_true")
    parser.add_argument("--disable-side-cam", action="store_true")


    args = parser.parse_args()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("../data", session_id)
    os.makedirs(base_dir, exist_ok=True)

    preview_size = (args.preview_width, args.preview_height)

    print("=== COLLECT DEMOS (DUAL CAM) ===")
    print("Session:", session_id)
    print("Base dir:", base_dir)
    print(f"Preview: {args.preview} ({preview_size[0]}x{preview_size[1]})")
    print(f"Arm: {args.arm_enable} -> {args.arm_udp_ip}:{args.arm_udp_port} ids={args.arm_ids} servo1={args.arm_include_servo1}")
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

    # Allow running even if one (or both) cams are missing:
    if not front_alive and not side_alive:
        print("[WARN] No cameras alive. Will run with black frames only.")


    if not cam_front.alive and not cam_side.alive:
        raise RuntimeError("Neither camera came up. Cannot record.")

    print("[4/6] Init gamepad...")
    js = init_gamepad()

    print("[5/6] Init logger...")
    logger = EpisodeLoggerDual(base_dir, session_id)

    print("[6/6] Init preview manager...")
    pv = PreviewManagerDual(enable_preview=args.preview, preview_size=preview_size)

    # Arm controller instance (UDP sender)
    arm: Optional[ArmControlUDP] = None
    if args.arm_enable:
        arm = ArmControlUDP(
            udp_ip=args.arm_udp_ip,
            udp_port=args.arm_udp_port,
            ids=args.arm_ids,
            send_hz=SEND_HZ,
            u_per_s=args.arm_u_per_s,
            include_servo1=args.arm_include_servo1,
            servo1_axis=args.arm_servo1_axis,
            servo1_init=args.arm_servo1_init,
            deadzone=args.arm_deadzone,
            verbose=args.arm_verbose,
        )

    recording = False
    last_toggle_state = False
    last_new_ep_state = False

    last_heartbeat = time.time()
    heartbeat_s = 2.0

    print()
    print("Controls:")
    print("  A: toggle recording on/off")
    print("  B: start a new episode (increments ep id)")
    if args.arm_enable:
        print("  Arm control enabled: sends u01 UDP packets (check follower_receive_u01.py is running)")
        if args.arm_include_servo1:
            print(f"  Servo 1 control on axis {args.arm_servo1_axis}")
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

            # Base drive controls
            lx, rx, turbo_val = read_gamepad_axes(js)

            turbo_mult = 1.0
            if USE_TURBO and turbo_val >= TURBO_MIN:
                turbo_mult = 1.0 + (turbo_val - TURBO_MIN) / (1.0 - TURBO_MIN) * (TURBO_GAIN - 1.0)

            v = lx * MAX_V * turbo_mult
            w = rx * MAX_W * turbo_mult

            esp.send_cmd(v, w)

            # Arm controls (UDP -> follower)
            u_arm: Optional[Dict[int, float]] = None
            if arm is not None:
                u_arm = arm.update_from_gamepad(js)
                arm.send(u_arm, rate_limit=True)

            # Sensors
            dyaw_deg = imu.update_and_get_yaw_delta_deg()
            ax_g, ay_g, az_g = imu.read_accel_g()

            # Frames: if a cam is missing/dead, use a black frame so the loop never stalls.
            frame_front_rgb = black_rgb_frame()
            frame_side_rgb  = black_rgb_frame()

            if cam_front is not None and getattr(cam_front, "alive", False):
                try:
                    f = cam_front.get_frame_rgb()
                    if f is not None:
                        frame_front_rgb = f
                except Exception as e:
                    print(f"front cam err: {e}")
                    cam_front.alive = False  # fallback to black

            if cam_side is not None and getattr(cam_side, "alive", False):
                try:
                    s = cam_side.get_frame_rgb()
                    if s is not None:
                        frame_side_rgb = s
                except Exception as e:
                    print(f"side cam err: {e}")
                    cam_side.alive = False  # fallback to black

            # Log only if recording AND frames exist
            if recording and (frame_front_rgb is not None) and (frame_side_rgb is not None):
                logger.log_step(frame_front_rgb, frame_side_rgb, dyaw_deg, ax_g, ay_g, az_g, v, w, u_arm=u_arm)

            # Preview overlay always shows current state + ep + step_idx
            pv.update(frame_front_rgb, frame_side_rgb, recording, logger.episode_id, logger.step_idx)

            # Exit via ESC only if preview enabled
            if args.preview:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

            # Console output policy
            if recording and args.print_while_recording:
                if time.time() - last_heartbeat > heartbeat_s:
                    if arm is not None and u_arm is not None:
                        # small compact arm print
                        arm_str = " ".join([f"{k}={u_arm[k]:.2f}" for k in sorted(u_arm.keys())])
                    else:
                        arm_str = ""
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
            if arm is not None:
                arm.close()
        except Exception:
            pass
        esp.close()
        pv.close()
        print("Session saved at", base_dir)
        print("Episodes are in epXXX/ subfolders.")
        print("Done.")


if __name__ == "__main__":
    main()