#!/usr/bin/env python3
import os
import cv2
import time
import json
import argparse
import socket
import numpy as np
import sys
import threading
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
TURBO_AXIS = 3
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
    # lx = js.get_axis(1)  # left stick vertical
    # rx = js.get_axis(2)  # right stick horizontal

    lx = js.get_axis(1)  # left stick vertical
    rx = js.get_axis(0)  # right stick horizontal
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

def map_range(x, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        raise ValueError("in_min and in_max must be different")
    return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)

def remap_values_to_zone(u_by_id: Dict[int, float], *, verbose: bool = False) -> Dict[int, float]:
    """
    Keep-out/collision-avoidance remap (your function) integrated.
    Expects u_by_id to contain at least ids 2,3,4. Other ids pass through unchanged.
    """
    import math

    if 2 not in u_by_id or 3 not in u_by_id or 4 not in u_by_id:
        return u_by_id

    # ================= PARAMETERS =================
    fillet_r = 2.0      # radius of rounded corner geometry
    keepout_r = 5.0     # repulsion band thickness
    margin = 0.01

    # Forbidden quadrant boundary: x <= bx AND y <= by
    bx = 4.6
    by = -0.6

    # ================= MAP INPUT =================
    a_deg = map_range(u_by_id[2], 0, 0.25, 125, 90)
    b_deg = map_range(u_by_id[3], 1, 0.66, 19, 90)
    c_deg = map_range(u_by_id[4], 1, 0.47, 102, 180)

    a = math.radians(a_deg)
    b = math.radians(b_deg)
    c = math.radians(c_deg)

    # ================= FORWARD KINEMATICS =================
    x1 = math.cos(a) * 11.6
    y1 = math.sin(a) * 11.6

    omega = -(math.pi - a - b)
    x2 = math.cos(omega) * 10.5
    y2 = math.sin(omega) * 10.5

    fi = omega + (c - math.pi)
    x3 = math.cos(fi) * 5.5
    y3 = math.sin(fi) * 5.5

    finalX = x1 + x2 + x3
    finalY = y1 + y2 + y3

    # ================= ROUNDED SDF =================
    vx = max(finalX - bx, 0.0)
    vy = max(finalY - by, 0.0)

    dist_raw = math.hypot(vx, vy)
    dist = dist_raw - fillet_r

    if verbose:
        print(f"[keepout] X={finalX:.3f}, Y={finalY:.3f}, sdf={dist:.3f}")

    # ================= OUTSIDE KEEP-OUT =================
    if dist > keepout_r:
        return u_by_id

    # ================= NORMAL =================
    if dist_raw > 1e-9:
        nx = vx / dist_raw
        ny = vy / dist_raw
    else:
        nx = ny = 1.0 / math.sqrt(2.0)

    # ================= PUSH =================
    target = keepout_r + margin
    push = target - dist
    safeX = finalX + nx * push
    safeY = finalY + ny * push

    if verbose:
        print(f"[keepout] -> safeX={safeX:.3f}, safeY={safeY:.3f}, push={push:.3f}")

    # ================= IK =================
    length = math.hypot(safeX - x3, safeY - y3)
    if length < 1e-6:
        return u_by_id

    def _clamp(v):
        return max(-1.0, min(1.0, v))

    alpha2 = math.acos(_clamp((length * length + 11.6 * 11.6 - 10.5 * 10.5) / (2.0 * length * 11.6)))
    beta = math.acos(_clamp((10.5 * 10.5 + 11.6 * 11.6 - length * length) / (2.0 * 10.5 * 11.6)))
    alpha = math.atan2(safeY - y3, safeX - x3) + alpha2

    out = dict(u_by_id)
    out[2] = clamp01(map_range(math.degrees(alpha), 125, 90, 0, 0.25))
    out[3] = clamp01(map_range(math.degrees(beta), 19, 90, 1, 0.66))
    return out


########################################
# Arm follower (threaded, integrated)
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
    Threaded follower:
      - Receives u01 leader packets on UDP
      - Applies keepout remap_values_to_zone()
      - Writes servo goals at --arm-hz regardless of camera/logging load
      - Drains UDP queue to avoid backlog lag

    Use:
      arm = ArmFollowerU01(...)
      u_arm = arm.get_last_u()   # for logging
    """

    NAME_TO_ID = {
        "shoulder_pan": 1,     # <-- add this
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "gripper": 6,
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
        drain_all_udp: bool = True,
    ):
        self.verbose = bool(verbose)
        self.drain_all_udp = bool(drain_all_udp)

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

        calib_ranges = load_calib_ranges(follower_calib)

        self.bus = Bus(port, baudrate, protocol=0)
        self.bus.connect()

        self.eeprom_limits: Dict[int, Tuple[int, int]] = {}
        self.map_limits: Dict[int, Tuple[int, int]] = {}

        active_ids: List[int] = []
        for mid in self.ids:
            try:
                # Basic setup
                self.bus.write1(mid, CTRL_TABLE["Operating_Mode"][0], 0)

                accel_val = 254
                if servo1_accel is not None and mid == 1:
                    accel_val = int(servo1_accel)
                self.bus.write1(mid, CTRL_TABLE["Acceleration"][0], accel_val)

                self.bus.write1(mid, CTRL_TABLE["Lock"][0], 1)
                self.bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 0)

                # Probe by reading EEPROM limits (fast sanity check)
                mn = self.bus.read2(mid, CTRL_TABLE["Min_Position_Limit"][0])
                mx = self.bus.read2(mid, CTRL_TABLE["Max_Position_Limit"][0])

                self.eeprom_limits[mid] = (mn, mx)
                active_ids.append(mid)
            except Exception as e:
                print(f"[arm][WARN] ID {mid} not responding on {port} @ {baudrate}. Skipping. err={e}", file=sys.stderr)

        # Only keep motors that actually respond
        self.ids = active_ids
        if not self.ids:
            raise RuntimeError("[arm] No responding servos found. Check port/baud/IDs.")

        # Mapping limits from calib (fallback to EEPROM)
        for mid in self.ids:
            self.map_limits[mid] = calib_ranges.get(mid, self.eeprom_limits[mid])

        # Torque ON once for active motors
        for mid in self.ids:
            self.bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 1)

        # Re-init command state dicts to match filtered ids
        self.last_u_by_id = {mid: 0.5 for mid in self.ids}
        self.last_desired_u_by_id = {mid: 0.5 for mid in self.NAME_TO_ID.values()}
        self.q_cmd = {mid: None for mid in self.ids}


        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", int(udp_port)))
        self.sock.settimeout(max(0.001, float(sock_timeout)))

        self.last_u_by_id: Dict[int, float] = {mid: 0.5 for mid in self.ids}
        self.last_desired_u_by_id: Dict[int, float] = {mid: 0.5 for mid in self.NAME_TO_ID.values()}
        self.last_rx_t: float = 0.0
        self.q_cmd: Dict[int, Optional[int]] = {mid: None for mid in self.ids}

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ArmFollowerU01", daemon=True)

        if self.verbose:
            print("[arm] EEPROM safety limits:")
            for mid in self.ids:
                print(f"  ID {mid}: {self.eeprom_limits[mid]}")
            print("[arm] Mapping limits (from calib unless missing):")
            for mid in self.ids:
                print(f"  ID {mid}: {self.map_limits[mid]}")
            print(f"[arm] Listening u01 on UDP :{udp_port} @ {self.hz} Hz (threaded)")

        self._thread.start()

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self.bus.disconnect()
        except Exception:
            pass

    def get_last_u(self) -> Dict[int, float]:
        with self._lock:
            return dict(self.last_u_by_id)
        
    def get_last_desired_u(self) -> Dict[int, float]:
        with self._lock:
            return dict(self.last_desired_u_by_id)

    def _poll_one_udp(self) -> Optional[Dict[int, float]]:
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

            if mid is None:
                continue

            try:
                u_by_id[mid] = clamp01(float(v))
            except Exception:
                pass

        return u_by_id if u_by_id else None

    def _step_write_servos(self, u_by_id: Dict[int, float]) -> None:
        goals: Dict[int, int] = {}

        for mid in self.ids:
            u = float(u_by_id.get(mid, 0.5))

            lo, hi = self.map_limits[mid]
            if mid == 6 and self.gr_override is not None:
                lo, hi = self.gr_override

            lo_s, hi_s = soft_range(lo, hi, self.soft_margin)
            mapped = u_to_ticks(u, lo_s, hi_s, self.INVERT.get(mid, False), self.TRIM.get(mid, 0))

            e_lo, e_hi = self.eeprom_limits[mid]
            after_eeprom = clamp_ticks(mapped, e_lo, e_hi)

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

        for mid, pos in goals.items():
            raw = encode_sign_magnitude(pos, SIGN_BITS["Goal_Position"])
            ok, err = self.bus.write2(mid, CTRL_TABLE["Goal_Position"][0], raw)
            if not ok and self.verbose:
                print(f"[arm][WARN] write goal failed ID {mid}: {err}", file=sys.stderr)

    def _run(self):
        next_t = time.time()
        local_u = dict(self.last_u_by_id)
        now_desired_u = dict(self.last_desired_u_by_id)

        while not self._stop.is_set():
            # Drain UDP queue (prevents backlog -> lag)
            got_any = False
            while True:
                u_delta = self._poll_one_udp()
                if u_delta is None:
                    break
                got_any = True
                now_desired_u.update(u_delta)  # includes servo1 even if not on bus
                # only apply to actual follower servos for writing:
                local_u.update({k: v for k, v in u_delta.items() if k in self.ids})
                if not self.drain_all_udp:
                    break

            if got_any:
                self.last_rx_t = time.time()

            # Apply keep-out remap on full state
            local_u = remap_values_to_zone(local_u, verbose=self.verbose)

            # Publish latest u for logger
            with self._lock:
                self.last_u_by_id = dict(local_u)
                self.last_desired_u_by_id = dict(now_desired_u)

            # Write servos
            self._step_write_servos(local_u)

            # Rate keeping (resync if late)
            next_t += self.dt
            sleep_s = next_t - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.time()


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

        # print(record)

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
    parser.add_argument("--max_step", type=int, default=300)
    parser.add_argument("--sock_timeout", type=float, default=0.01)

    parser.add_argument("--servo1_accel", type=int, default=None)
    parser.add_argument("--arm-verbose", action="store_true")
    parser.add_argument("--arm-drain", action="store_true", help="Drain all pending UDP packets each arm tick (reduces lag).")
    parser.add_argument("--arm-timeout-s", type=float, default=1.0)
    
    parser.add_argument("--wheel-pan-enable", action="store_true",
                    help="Add leader servo1/shoulder_pan (u01, center=0.5) as extra wheel turn input.")
    parser.add_argument("--wheel-pan-deadzone", type=float, default=0.06,
                    help="Deadzone around u=0.5 for servo1 -> turn.")
    parser.add_argument("--wheel-pan-gain", type=float, default=1.0,
                    help="Gain multiplier for servo1 contribution to angular velocity (scaled by MAX_W).")


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
    print(f"Arm: enable={args.arm_enable} udp={args.udp_port} ids={args.ids} hz={args.arm_hz} max_step={args.max_step}")
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

    # Arm follower integrated (thread)
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
            drain_all_udp=args.arm_drain,
        )

    print("[7/7] Ready.")
    print()
    print("Controls:")
    print("  A: toggle recording on/off")
    print("  B: start a new episode (increments ep id)")
    print("  ESC (in preview): exit")
    print()

    recording = False
    rec_latch = False
    ep_latch = False

    # Heartbeat printing
    last_heartbeat = time.time()
    heartbeat_s = 2.0

    try:
        while True:
            t0 = time.time()

            # Handle button presses as events (no latch needed)
            events = pygame.event.get()

            # simple debounce (seconds)
            DEBOUNCE_S = 0.20
            if not hasattr(main, "_last_btn_t"):
                main._last_btn_t = 0.0

            now_t = time.time()

            for ev in events:
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt

                if ev.type == pygame.JOYBUTTONDOWN:
                    # Optional: print what button index was pressed to verify mapping
                    # print(f"[JOY] button down: {ev.button}")

                    if (now_t - main._last_btn_t) < DEBOUNCE_S:
                        continue  # ignore ultra-fast repeats/bounce

                    if ev.button == BTN_TOGGLE_REC:
                        recording = not recording
                        main._last_btn_t = now_t
                        print(f"Recording: {recording} (ep{logger.episode_id:03d})")

                    elif ev.button == BTN_NEW_EPISODE:
                        logger.start_new_episode()
                        main._last_btn_t = now_t
                        print(f"Episode changed: ep{logger.episode_id:03d}")

            # Base drive
            lx, rx, turbo_val = read_gamepad_axes(js)

            turbo_mult = 1.0
            if USE_TURBO and turbo_val >= TURBO_MIN:
                turbo_mult = 1.0 + (turbo_val - TURBO_MIN) / (1.0 - TURBO_MIN) * (TURBO_GAIN - 1.0)

            v = lx * MAX_V * turbo_mult
            w = rx * MAX_W * turbo_mult

            # Arm state for logging (thread handles servo writes)
            u_arm: Optional[Dict[int, float]] = None
            if arm is not None:
                u_arm = arm.get_last_u()
                if recording and args.arm_timeout_s > 0 and arm.last_rx_t > 0:
                    if (time.time() - arm.last_rx_t) > args.arm_timeout_s:
                        if time.time() - last_heartbeat > heartbeat_s:
                            print(f"[WARN] No arm u01 packets for {time.time()-arm.last_rx_t:.2f}s on UDP :{args.udp_port}")
                            last_heartbeat = time.time()
            
            desired_u_arm: Optional[Dict[int, float]] = None
            if arm is not None:
                desired_u_arm = arm.get_last_desired_u()

            if args.wheel_pan_enable and desired_u_arm is not None and 1 in desired_u_arm:
                u1 = float(desired_u_arm[1])  # [0..1], center=0.5
                d = u1 - 0.5

                dz = max(0.0, min(0.45, float(args.wheel_pan_deadzone)))
                if abs(d) < dz:
                    d_norm = 0.0
                else:
                    d_norm = (abs(d) - dz) / (0.5 - dz) * (1.0 if d > 0 else -1.0)

                w += d_norm * (MAX_W * float(args.wheel_pan_gain))

            w = clamp(w, -MAX_W, MAX_W)
            esp.send_cmd(v, w)

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