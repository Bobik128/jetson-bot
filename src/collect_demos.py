#!/usr/bin/env python3
import os
import cv2
import time
import json
import argparse
import socket
import numpy as np
import sys
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pygame

import scservo_sdk as scs

from gst_cam import GstCam
from shared.constants import FRAME_SIZE, JPEG_QUALITY, MAX_V, MAX_W
from shared.esp32_link import ESP32Link
from shared.imu_mpu6050 import MPU6050GyroYaw


########################################
# Recording / control config
########################################

SEND_HZ = 30.0
DT = 1.0 / SEND_HZ

DEADZONE = 0.20
EXPO = 0.25

USE_TURBO = True
TURBO_AXIS = 4
TURBO_MIN = 0.6
TURBO_GAIN = 1.75

BTN_TOGGLE_REC = 0   # A
BTN_NEW_EPISODE = 1  # B


########################################
# Helpers
########################################

def black_rgb_frame():
    w, h = FRAME_SIZE  # (W,H)
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

# IMPORTANT: no pygame.event.pump() here (pump ONCE per loop in main)
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


########################################
# Arm follower (integrated)
########################################

CTRL_TABLE = {
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Operating_Mode": (33, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
}
SIGN_BITS = {"Goal_Position": 15}

def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value < 0:
        return (1 << sign_bit) | (abs(value) & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)

def clamp_ticks(x: int, lo: int, hi: int) -> int:
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    return a if x < a else b if x > b else x

def soft_range(lo: int, hi: int, margin: float) -> Tuple[int, int]:
    m = max(0.0, min(0.45, float(margin)))
    span = hi - lo
    slo = int(round(lo + m * span))
    shi = int(round(hi - m * span))
    if abs(shi - slo) < 2:
        return lo, hi
    return slo, shi

def u_to_ticks(u: float, lo: int, hi: int, invert: bool, trim: int) -> int:
    u = clamp01(u)
    if invert:
        u = 1.0 - u
    raw = int(round(lo + u * (hi - lo)))
    raw += int(trim)
    return clamp_ticks(raw, lo, hi)

def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50

class Bus:
    def __init__(self, port: str, baudrate: int, protocol: int = 0):
        self.port_handler = scs.PortHandler(port)
        self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(self.port_handler, scs.PortHandler)
        self.packet_handler = scs.PacketHandler(protocol)
        self.port = port
        self.baudrate = baudrate

    def connect(self):
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open {self.port}")
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate {self.baudrate}")

    def disconnect(self):
        try:
            self.port_handler.closePort()
        except Exception:
            pass

    def _unpack2or3(self, ret):
        if isinstance(ret, tuple) and len(ret) == 2:
            return None, ret[0], ret[1]
        if isinstance(ret, tuple) and len(ret) == 3:
            return ret[0], ret[1], ret[2]
        raise RuntimeError(f"Unexpected SDK return: {ret!r}")

    def write1(self, mid: int, addr: int, val: int) -> Tuple[bool, str]:
        try:
            ret = self.packet_handler.write1ByteTxRx(self.port_handler, mid, addr, int(val))
            _, comm, err = self._unpack2or3(ret)
            if comm != scs.COMM_SUCCESS:
                return False, self.packet_handler.getTxRxResult(comm)
            if err != 0:
                return False, self.packet_handler.getRxPacketError(err)
            return True, ""
        except Exception as e:
            return False, str(e)

    def write2(self, mid: int, addr: int, val: int) -> Tuple[bool, str]:
        try:
            ret = self.packet_handler.write2ByteTxRx(self.port_handler, mid, addr, int(val))
            _, comm, err = self._unpack2or3(ret)
            if comm != scs.COMM_SUCCESS:
                return False, self.packet_handler.getTxRxResult(comm)
            if err != 0:
                return False, self.packet_handler.getRxPacketError(err)
            return True, ""
        except Exception as e:
            return False, str(e)

    def read2(self, mid: int, addr: int) -> int:
        ret = self.packet_handler.read2ByteTxRx(self.port_handler, mid, addr)
        if isinstance(ret, tuple) and len(ret) == 3:
            val, comm, err = ret
        elif isinstance(ret, tuple) and len(ret) == 2:
            val, comm = ret
            err = 0
        else:
            raise RuntimeError(f"Unexpected SDK return: {ret!r}")
        if comm != scs.COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm))
        if err != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(err))
        return int(val)

def load_calib_ranges(path: str) -> Dict[int, Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    out: Dict[int, Tuple[int, int]] = {}
    for _, d in j.items():
        out[int(d["id"])] = (int(d["range_min"]), int(d["range_max"]))
    return out

class ArmFollowerU01:
    """
    Integrated follower:
      - Receives u01 packets from leader on UDP
      - Converts u -> servo ticks using calibration ranges
      - Applies EEPROM safety clamp + optional soft margin + rate limiting
      - Writes Goal_Position

    NOTE: This is intentionally close to your follower_receive_u01.py logic.
    """

    NAME_TO_ID = {
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "gripper": 6,
        "servo1": 1,
        "base": 1,
    }

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        udp_port: int,
        ids: List[int],
        follower_calib: str,
        invert: List[int],
        trim: str,
        soft_margin: float,
        hz: float,
        max_step: int,
        sock_timeout: float,
        gripper_range: Optional[Tuple[int, int]] = None,
        servo1_accel: Optional[int] = None,
        verbose: bool = False,
    ):
        self.verbose = bool(verbose)

        self.ids = list(ids)
        self.hz = float(hz)
        self.dt = 1.0 / max(1.0, self.hz)
        self.max_step = int(max_step)
        self.soft_margin = float(soft_margin)
        self.gr_override = tuple(gripper_range) if gripper_range else None

        self.INVERT = {mid: (mid in set(invert)) for mid in self.ids}
        self.TRIM = {mid: 0 for mid in self.ids}
        if trim.strip():
            for part in trim.split(","):
                k, v = part.strip().split(":")
                self.TRIM[int(k)] = int(v)

        # Load calibration ticks (range_min/range_max)
        calib_ranges = load_calib_ranges(follower_calib)

        # Connect bus
        self.bus = Bus(port, baudrate, protocol=0)
        self.bus.connect()

        # Read EEPROM safety limits and configure servos
        self.eeprom_limits: Dict[int, Tuple[int, int]] = {}
        self.map_limits: Dict[int, Tuple[int, int]] = {}

        for mid in self.ids:
            # Position mode, accel, lock, torque off while reading EEPROM
            self.bus.write1(mid, CTRL_TABLE["Operating_Mode"][0], 0)

            # Acceleration: allow per-servo1 override if provided
            accel_val = 254
            if servo1_accel is not None and mid == 1:
                accel_val = int(servo1_accel)
            self.bus.write1(mid, CTRL_TABLE["Acceleration"][0], accel_val)

            self.bus.write1(mid, CTRL_TABLE["Lock"][0], 1)
            self.bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 0)

            mn = self.bus.read2(mid, CTRL_TABLE["Min_Position_Limit"][0])
            mx = self.bus.read2(mid, CTRL_TABLE["Max_Position_Limit"][0])
            self.eeprom_limits[mid] = (mn, mx)

        # mapping limits: prefer calib, else EEPROM
        for mid in self.ids:
            self.map_limits[mid] = calib_ranges.get(mid, self.eeprom_limits[mid])

        # torque ON once (keep on)
        for mid in self.ids:
            self.bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 1)

        # UDP socket (non-blocking)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", int(udp_port)))
        self.sock.settimeout(max(0.001, float(sock_timeout)))

        # last u and command state
        self.last_u_by_id: Dict[int, float] = {mid: 0.5 for mid in self.ids}
        self.last_rx_t: float = 0.0
        self.q_cmd: Dict[int, Optional[int]] = {mid: None for mid in self.ids}

        if self.verbose:
            print("[arm] EEPROM safety limits:")
            for mid in self.ids:
                print(f"  ID {mid}: {self.eeprom_limits[mid]}")
            print("[arm] Mapping limits (from calib unless missing):")
            for mid in self.ids:
                print(f"  ID {mid}: {self.map_limits[mid]}")
            print(f"[arm] Listening u01 on UDP :{udp_port} @ {self.hz} Hz")

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self.bus.disconnect()
        except Exception:
            pass

    def poll_udp(self) -> Optional[Dict[int, float]]:
        """
        Non-blocking-ish: uses short socket timeout. Returns u_by_id if packet received, else None.
        """
        try:
            data, _ = self.sock.recvfrom(8192)
        except socket.timeout:
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

        u_by_id: Dict[int, float] = {}

        for k, v in u_field.items():
            mid: Optional[int] = None
            if isinstance(k, str) and k in self.NAME_TO_ID:
                mid = self.NAME_TO_ID[k]
            elif isinstance(k, str) and k.isdigit():
                mid = int(k)
            elif isinstance(k, int):
                mid = k

            if mid is None or mid not in self.ids:
                continue

            try:
                u_by_id[mid] = clamp01(float(v))
            except Exception:
                pass

        if u_by_id:
            self.last_u_by_id.update(u_by_id)
            self.last_rx_t = time.time()
            return dict(self.last_u_by_id)

        return None

    def step_write_servos(self) -> Dict[int, int]:
        """
        Convert last_u_by_id -> goal ticks and write to servos.
        Returns goals (ticks) dict.
        """
        goals: Dict[int, int] = {}

        for mid in self.ids:
            u = float(self.last_u_by_id.get(mid, 0.5))

            lo, hi = self.map_limits[mid]
            if mid == 6 and self.gr_override is not None:
                lo, hi = self.gr_override

            lo_s, hi_s = soft_range(lo, hi, self.soft_margin)

            mapped = u_to_ticks(u, lo_s, hi_s, self.INVERT.get(mid, False), self.TRIM.get(mid, 0))

            # EEPROM safety clamp
            e_lo, e_hi = self.eeprom_limits[mid]
            after_eeprom = clamp_ticks(mapped, e_lo, e_hi)

            # rate limit
            if self.q_cmd[mid] is None:
                goal = after_eeprom
            else:
                delta = after_eeprom - int(self.q_cmd[mid])
                step = delta
                if step > self.max_step:
                    step = self.max_step
                elif step < -self.max_step:
                    step = -self.max_step
                goal = int(self.q_cmd[mid]) + step

            self.q_cmd[mid] = goal
            goals[mid] = goal

        # write goals
        for mid, pos in goals.items():
            raw = encode_sign_magnitude(pos, SIGN_BITS["Goal_Position"])
            ok, err = self.bus.write2(mid, CTRL_TABLE["Goal_Position"][0], raw)
            if not ok and self.verbose:
                print(f"[arm][WARN] write goal failed ID {mid}: {err}", file=sys.stderr)

        return goals


########################################
# Logging / preview
########################################

class EpisodeLoggerDual:
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

    def log_step(self, frame_front_rgb, frame_side_rgb, dyaw_deg, ax_g, ay_g, az_g, v, w,
                 u_arm: Optional[Dict[int, float]] = None):
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
    def __init__(self, enable_preview: bool, preview_size: Tuple[int, int]):
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

        front_bgr = cv2.cvtColor(frame_front_rgb, cv2.COLOR_RGB2BGR)
        front_bgr = cv2.resize(front_bgr, self.preview_size, interpolation=cv2.INTER_LINEAR)
        front_bgr = self._overlay(front_bgr, recording, episode_id, step_idx)
        cv2.imshow("front", front_bgr)

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


########################################
# Main
########################################

def main():
    parser = argparse.ArgumentParser()

    # Base serial (ESP32 drive)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=115200)

    # Cameras
    parser.add_argument("--front-sensor-id", type=int, default=0)
    parser.add_argument("--side-sensor-id", type=int, default=1)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--disable-front-cam", action="store_true")
    parser.add_argument("--disable-side-cam", action="store_true")

    # Preview
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-height", type=int, default=540)
    parser.add_argument("--preview-width", type=int, default=920)

    # Logging verbosity
    parser.add_argument("--print-while-recording", action="store_true")

    # Arm follower (integrated)
    parser.add_argument("--arm-enable", action="store_true", help="Enable integrated arm follower (UDP u01 -> servos).")
    parser.add_argument("--arm-port", default="/dev/ttyACM0")
    parser.add_argument("--arm-baudrate", type=int, default=1_000_000)
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--follower_calib", required=False, default=None)
    parser.add_argument("--gripper_range", type=int, nargs=2, default=None)
    parser.add_argument("--invert", type=int, nargs="*", default=[])
    parser.add_argument("--trim", type=str, default="")
    parser.add_argument("--soft_margin", type=float, default=0.0)
    parser.add_argument("--arm-hz", type=float, default=120.0)
    parser.add_argument("--max_step", type=int, default=100)
    parser.add_argument("--sock_timeout", type=float, default=0.01)
    parser.add_argument("--servo1_accel", type=int, default=None)
    parser.add_argument("--arm-verbose", action="store_true")
    parser.add_argument("--arm-timeout-s", type=float, default=1.0,
                        help="Warn if no u01 packets for this long while recording (arm enabled).")

    args = parser.parse_args()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("../data", session_id)
    os.makedirs(base_dir, exist_ok=True)

    preview_size = (args.preview_width, args.preview_height)

    print("=== COLLECT DEMOS (UNIFIED: BASE + CAMS + ARM FOLLOWER) ===")
    print("Session:", session_id)
    print("Base dir:", base_dir)
    print(f"Preview: {args.preview} ({preview_size[0]}x{preview_size[1]})")
    print(f"Cams: disable_front={args.disable_front_cam} disable_side={args.disable_side_cam}")
    print(f"Arm: enable={args.arm_enable} udp={args.udp_port} ids={args.ids}")
    print()

    # Base
    print("[1/7] Init ESP32 serial link...")
    esp = ESP32Link(port=args.port, baud=args.baud)

    # IMU
    print("[2/7] Init IMU...")
    imu = MPU6050GyroYaw()

    # Cameras
    print("[3/7] Init cameras...")
    cam_front = None
    cam_side = None

    if not args.disable_front_cam:
        cam_front = GstCam(
            base_dir=base_dir,
            frame_size=FRAME_SIZE,
            jpeg_quality=JPEG_QUALITY,
            sensor_id=args.front_sensor_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            capture_fps=args.capture_fps,
        )
    else:
        print("Front cam disabled by flag.")

    if not args.disable_side_cam:
        cam_side = GstCam(
            base_dir=base_dir,
            frame_size=FRAME_SIZE,
            jpeg_quality=JPEG_QUALITY,
            sensor_id=args.side_sensor_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            capture_fps=args.capture_fps,
        )
    else:
        print("Side cam disabled by flag.")

    front_alive = (cam_front is not None) and getattr(cam_front, "alive", False)
    side_alive  = (cam_side  is not None) and getattr(cam_side,  "alive", False)
    print(f"cam_front alive: {front_alive}")
    print(f"cam_side  alive: {side_alive}")
    if not front_alive and not side_alive:
        print("[WARN] No cameras alive. Will run with black frames only.")

    # Gamepad
    print("[4/7] Init gamepad...")
    js = init_gamepad()

    # Logger
    print("[5/7] Init logger...")
    logger = EpisodeLoggerDual(base_dir, session_id)

    # Preview
    print("[6/7] Init preview manager...")
    pv = PreviewManagerDual(enable_preview=args.preview, preview_size=preview_size)

    # Arm follower integrated
    arm: Optional[ArmFollowerU01] = None
    if args.arm_enable:
        if not args.follower_calib:
            raise RuntimeError("--follower_calib is required when --arm-enable is set.")
        arm = ArmFollowerU01(
            port=args.arm_port,
            baudrate=args.arm_baudrate,
            udp_port=args.udp_port,
            ids=args.ids,
            follower_calib=args.follower_calib,
            invert=args.invert,
            trim=args.trim,
            soft_margin=args.soft_margin,
            hz=args.arm_hz,
            max_step=args.max_step,
            sock_timeout=args.sock_timeout,
            gripper_range=tuple(args.gripper_range) if args.gripper_range else None,
            servo1_accel=args.servo1_accel,
            verbose=args.arm_verbose,
        )

    print("[7/7] Ready.")
    print()
    print("Controls:")
    print("  A: toggle recording on/off")
    print("  B: start a new episode (increments ep id)")
    print("  ESC (in preview): exit")
    print()

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

    # Arm write scheduling
    next_arm_t = time.time()

    try:
        while True:
            t0 = time.time()

            # Pump ONCE per loop
            pygame.event.pump()

            # Buttons with debounce
            btn_toggle = (js.get_button(BTN_TOGGLE_REC) == 1)
            btn_new_ep = (js.get_button(BTN_NEW_EPISODE) == 1)
            now = time.time()

            if btn_toggle and (not last_toggle_state) and ((now - last_toggle_t) > debounce_s):
                recording = not recording
                last_toggle_t = now
                print(f"Recording: {recording} (ep{logger.episode_id:03d})")
            last_toggle_state = btn_toggle

            if btn_new_ep and (not last_new_ep_state) and ((now - last_new_ep_t) > debounce_s):
                logger.start_new_episode()
                last_new_ep_t = now
                print(f"Episode changed: ep{logger.episode_id:03d}")
            last_new_ep_state = btn_new_ep

            # Base drive
            lx, rx, turbo_val = read_gamepad_axes(js)

            turbo_mult = 1.0
            if USE_TURBO and turbo_val >= TURBO_MIN:
                turbo_mult = 1.0 + (turbo_val - TURBO_MIN) / (1.0 - TURBO_MIN) * (TURBO_GAIN - 1.0)

            v = lx * MAX_V * turbo_mult
            w = rx * MAX_W * turbo_mult
            esp.send_cmd(v, w)

            # Arm: receive UDP and write at arm-hz
            u_arm: Optional[Dict[int, float]] = None
            if arm is not None:
                got = arm.poll_udp()
                if got is not None:
                    u_arm = got
                else:
                    u_arm = dict(arm.last_u_by_id)

                if time.time() >= next_arm_t:
                    arm.step_write_servos()
                    next_arm_t = time.time() + arm.dt

                # warning if no packets
                if recording and args.arm_timeout_s > 0 and arm.last_rx_t > 0:
                    if (time.time() - arm.last_rx_t) > args.arm_timeout_s:
                        if time.time() - last_heartbeat > heartbeat_s:
                            print(f"[WARN] No arm u01 packets for {time.time()-arm.last_rx_t:.2f}s on UDP :{args.udp_port}")
                            last_heartbeat = time.time()

            # Sensors
            dyaw_deg = imu.update_and_get_yaw_delta_deg()
            ax_g, ay_g, az_g = imu.read_accel_g()

            # Frames (black fallback always)
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

            # Log
            if recording:
                logger.log_step(
                    frame_front_rgb, frame_side_rgb,
                    dyaw_deg, ax_g, ay_g, az_g,
                    v, w,
                    u_arm=u_arm
                )

            # Preview
            pv.update(frame_front_rgb, frame_side_rgb, recording, logger.episode_id, logger.step_idx)

            # Exit via ESC if preview enabled
            if args.preview:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

            # Status prints
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