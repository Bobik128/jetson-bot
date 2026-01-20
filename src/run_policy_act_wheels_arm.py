#!/usr/bin/env python3
import time
import json
import argparse
import threading
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch

import scservo_sdk as scs

from gst_cam import GstCam
from shared.constants import FRAME_SIZE, MAX_V, MAX_W
from shared.esp32_link import ESP32Link
from shared.imu_mpu6050 import MPU6050GyroYaw

# LeRobot ACT policy
# (This import path is consistent with LeRobot examples/snippets.)
from lerobot.policies.act.modeling_act import ACTPolicy

############################################################
# Small utilities
############################################################

def clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def map_range(x, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        raise ValueError("in_min and in_max must be different")
    return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)

def remap_values_to_zone(u_by_id: Dict[int, float], *, verbose: bool = False) -> Dict[int, float]:
    """
    Keep-out/collision-avoidance remap integrated.
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
class RateMeter:
    """EMA loop Hz estimator."""
    def __init__(self, alpha: float = 0.12):
        self.alpha = float(alpha)
        self.last_t = None
        self.rate_hz = 0.0

    def tick(self, now: float = None) -> float:
        if now is None:
            now = time.time()
        if self.last_t is None:
            self.last_t = now
            return self.rate_hz
        dt = now - self.last_t
        self.last_t = now
        if dt <= 1e-9:
            return self.rate_hz
        inst = 1.0 / dt
        if self.rate_hz <= 0.0:
            self.rate_hz = inst
        else:
            self.rate_hz = (1.0 - self.alpha) * self.rate_hz + self.alpha * inst
        return self.rate_hz


############################################################
# Arm control (your existing servo stack, but with direct set_target_u)
############################################################

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


class ArmControllerU01:
    """
    Servo writer thread. You call set_target_u({2:..,3:..,4:..,6:..}).
    The thread runs at --arm-hz and pushes the latest target to the servos.
    """

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        ids: List[int],
        follower_calib: str,
        invert: List[int],
        trim: str,
        soft_margin: float,
        hz: float,
        max_step: int,
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

        calib_ranges = load_calib_ranges(follower_calib)

        self.bus = Bus(port, baudrate, protocol=0)
        self.bus.connect()

        self.eeprom_limits: Dict[int, Tuple[int, int]] = {}
        self.map_limits: Dict[int, Tuple[int, int]] = {}

        # loop Hz for overlay/debug
        self._rate = RateMeter(alpha=0.12)
        self.loop_hz = 0.0

        # setup motors
        active: List[int] = []
        for mid in self.ids:
            try:
                self.bus.write1(mid, CTRL_TABLE["Operating_Mode"][0], 0)

                accel_val = 254
                if servo1_accel is not None and mid == 1:
                    accel_val = int(servo1_accel)
                self.bus.write1(mid, CTRL_TABLE["Acceleration"][0], accel_val)

                self.bus.write1(mid, CTRL_TABLE["Lock"][0], 1)
                self.bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 0)

                mn = self.bus.read2(mid, CTRL_TABLE["Min_Position_Limit"][0])
                mx = self.bus.read2(mid, CTRL_TABLE["Max_Position_Limit"][0])
                self.eeprom_limits[mid] = (mn, mx)
                active.append(mid)
            except Exception as e:
                print(f"[arm][WARN] ID {mid} not responding. Skipping. err={e}", file=sys.stderr)

        self.ids = active
        if not self.ids:
            raise RuntimeError("[arm] No responding servos found.")

        for mid in self.ids:
            self.map_limits[mid] = calib_ranges.get(mid, self.eeprom_limits[mid])

        for mid in self.ids:
            self.bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 1)

        # target state
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ArmControllerU01", daemon=True)

        self._target_u = {mid: 0.5 for mid in self.ids}
        self._last_u   = {mid: 0.5 for mid in self.ids}
        self.q_cmd: Dict[int, Optional[int]] = {mid: None for mid in self.ids}

        if self.verbose:
            print("[arm] active ids:", self.ids)
            print("[arm] eeprom limits:", self.eeprom_limits)
            print("[arm] map limits:", self.map_limits)

        self._thread.start()

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.bus.disconnect()
        except Exception:
            pass

    def set_target_u(self, u_by_id: Dict[int, float]) -> None:
        with self._lock:
            for mid, u in u_by_id.items():
                if mid in self._target_u:
                    self._target_u[mid] = clamp01(float(u))

    def get_last_u(self) -> Dict[int, float]:
        with self._lock:
            return dict(self._last_u)

    def get_loop_hz(self) -> float:
        return float(self.loop_hz)

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
        while not self._stop.is_set():
            # 1) snapshot target safely
            with self._lock:
                base = self._target_u
                if not isinstance(base, dict):
                    # fall back to last known safe command or neutral
                    base = getattr(self, "_last_u", None)
                    if not isinstance(base, dict):
                        base = {mid: 0.5 for mid in self.ids}

                local_u = dict(base)

            # 2) ALWAYS keep-out remap (never skip)
            try:
                remapped = remap_values_to_zone(local_u, verbose=self.verbose)
            except Exception as e:
                if self.verbose:
                    print(f"[arm][WARN] remap_values_to_zone failed: {e}")
                remapped = local_u

            # remap must return dict; if not, keep last safe
            if not isinstance(remapped, dict):
                if self.verbose:
                    print("[arm][WARN] remap returned non-dict; keeping previous command")
                with self._lock:
                    prev = getattr(self, "_last_u", None)
                    if isinstance(prev, dict):
                        remapped = dict(prev)
                    else:
                        remapped = {mid: 0.5 for mid in self.ids}

            # 3) publish last_u for UI/logging
            with self._lock:
                self._last_u = dict(remapped)

            # 4) write servos
            self._step_write_servos(remapped)

            # 5) timing
            self.loop_hz = self._rate.tick()
            next_t += self.dt
            sleep_s = next_t - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.time()


############################################################
# LeRobot observation prep
############################################################

def frame_rgb_to_torch_chw_uint8(frame_rgb: np.ndarray, device: str) -> torch.Tensor:
    """
    LeRobot policies typically accept images as torch tensors.
    We provide uint8 CHW (0..255). Policy handles its own preprocessing.
    """
    if frame_rgb is None:
        raise ValueError("frame_rgb is None")
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"expected HWC RGB, got {frame_rgb.shape}")

    chw = torch.from_numpy(frame_rgb).to(torch.uint8).permute(2, 0, 1)  # CHW
    # add batch dim if needed (policy often accepts unbatched too, but batching is safer)
    chw = chw.unsqueeze(0).to(device, non_blocking=True)  # 1,C,H,W
    return chw

def state_to_torch(state_vec: np.ndarray, device: str) -> torch.Tensor:
    """
    observation.state should be float32.
    Shape: (1, D)
    """
    state_vec = np.asarray(state_vec, dtype=np.float32).reshape(1, -1)
    return torch.from_numpy(state_vec).to(device, non_blocking=True)


############################################################
# Main loop
############################################################

def parse_args():
    p = argparse.ArgumentParser()

    # Policy / checkpoint
    p.add_argument("--policy-path", required=True,
                   help="Path to LeRobot exported checkpoint dir (e.g. outputs/train/.../checkpoints/last/pretrained_model)")

    # Wheels
    p.add_argument("--port", default="/dev/ttyTHS1")
    p.add_argument("--baud", type=int, default=115200)

    # Cameras
    p.add_argument("--base-dir", default=".", help="Used for GstCam debug dumps.")
    p.add_argument("--front-sensor-id", type=int, default=0)
    p.add_argument("--side-sensor-id", type=int, default=1)
    p.add_argument("--capture-width", type=int, default=640)
    p.add_argument("--capture-height", type=int, default=480)
    p.add_argument("--capture-fps", type=int, default=30)
    p.add_argument("--disable-side", action="store_true")
    p.add_argument("--disable-front", action="store_true")
    p.add_argument("--debug-dir", default="../data", help="Directory for GstCam debug dumps (must be a real directory).")


    # Runtime
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--hz", type=float, default=30.0)

    # Arm
    p.add_argument("--arm-enable", action="store_true")
    p.add_argument("--arm-port", default="/dev/ttyACM0")
    p.add_argument("--arm-baudrate", type=int, default=1_000_000)
    p.add_argument("--arm-ids", type=int, nargs="+", default=[2, 3, 4, 6])
    p.add_argument("--follower_calib", default=None)
    p.add_argument("--invert", type=int, nargs="*", default=[])
    p.add_argument("--trim", type=str, default="")
    p.add_argument("--soft_margin", type=float, default=0.0)
    p.add_argument("--arm-hz", type=float, default=120.0)
    p.add_argument("--max_step", type=int, default=300)
    p.add_argument("--gripper_range", type=int, nargs=2, default=None)
    p.add_argument("--servo1_accel", type=int, default=None)
    p.add_argument("--arm-verbose", action="store_true")

    # Action mapping
    p.add_argument("--action-map", default="v,w,2,3,4,6",
                   help="Comma-separated mapping for the 6 action outputs. Example: v,w,2,3,4,6")

    # Preview
    p.add_argument("--preview", action="store_true")

    return p.parse_args()

def main():
    args = parse_args()

    # Device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    dt_target = 1.0 / max(1.0, float(args.hz))

    # Parse action map
    # Expected 6 tokens: first two are v,w, rest are servo IDs (or names)
    tokens = [t.strip() for t in args.action_map.split(",") if t.strip()]
    if len(tokens) != 6 or tokens[0] != "v" or tokens[1] != "w":
        raise RuntimeError("--action-map must look like: v,w,2,3,4,6 (6 entries, first two v,w).")
    arm_ids_out = [int(t) for t in tokens[2:]]

    print("Device:", device)
    print("Policy path:", args.policy_path)
    print("Action mapping: v,w then arm ids:", arm_ids_out)

    # Init wheels link
    print("Init ESP32 link...")
    esp = ESP32Link(port=args.port, baud=args.baud)
    # if your ESP32Link supports it, keep reader on:
    try:
        esp.start_reader()
    except Exception:
        pass

    # IMU
    print("Init IMU...")
    imu = MPU6050GyroYaw()

    # Cameras
    print("Init cameras...")
    cam_front = None
    cam_side = None

    if not args.disable_front:
        cam_front = GstCam(
            base_dir=args.base_dir,
            frame_size=FRAME_SIZE,
            sensor_id=args.front_sensor_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            capture_fps=args.capture_fps,
        )

    if not args.disable_side:
        cam_side = GstCam(
            base_dir=args.base_dir,
            frame_size=FRAME_SIZE,
            sensor_id=args.side_sensor_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            capture_fps=args.capture_fps,
        )

    # Load ACT policy
    # LeRobot exported checkpoints are typically in Hugging Face format.
    print("Load LeRobot ACT policy...")
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.eval()
    policy.to(device)

    # Arm controller
    arm: Optional[ArmControllerU01] = None
    if args.arm_enable:
        if not args.follower_calib:
            raise RuntimeError("--follower_calib is required when --arm-enable is set.")
        arm = ArmControllerU01(
            port=args.arm_port,
            baudrate=args.arm_baudrate,
            ids=args.arm_ids,
            follower_calib=args.follower_calib,
            invert=args.invert,
            trim=args.trim,
            soft_margin=args.soft_margin,
            hz=args.arm_hz,
            max_step=args.max_step,
            gripper_range=tuple(args.gripper_range) if args.gripper_range else None,
            servo1_accel=args.servo1_accel,
            verbose=args.arm_verbose,
        )

    # Preview
    if args.preview:
        cv2.namedWindow("policy", cv2.WINDOW_NORMAL)

    main_rate = RateMeter(alpha=0.12)

    try:
        while True:
            t0 = time.time()
            loop_hz = main_rate.tick()

            # Read sensors
            dyaw_deg = imu.update_and_get_yaw_delta_deg()
            ax_g, ay_g, az_g = imu.read_accel_g()

            # Build observation.state
            tel_v, tel_w, age = esp.get_latest()
            if age is None:
                tel_v, tel_w = 0.0, 0.0
            state_vec = np.array([dyaw_deg, ax_g, ay_g, az_g, tel_v, tel_w], dtype=np.float32)

            # Grab frames (RGB)
            # If camera is missing, fall back to black frame with FRAME_SIZE.
            h, w = FRAME_SIZE[1], FRAME_SIZE[0]
            front_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            side_rgb = np.zeros((h, w, 3), dtype=np.uint8)

            if cam_front is not None:
                f = cam_front.get_frame_rgb()
                if f is not None:
                    front_rgb = f

            if cam_side is not None:
                s = cam_side.get_frame_rgb()
                if s is not None:
                    side_rgb = s

            # Convert to torch tensors
            obs = {
                "observation.state": state_to_torch(state_vec, device),
                "observation.images.front": frame_rgb_to_torch_chw_uint8(front_rgb, device),
                "observation.images.side": frame_rgb_to_torch_chw_uint8(side_rgb, device),
            }

            # Inference
            with torch.no_grad():
                assert obs["observation.state"].shape[-1] == 6, obs["observation.state"].shape
                assert obs["observation.images.front"].shape[-2:] == (144, 256), obs["observation.images.front"].shape
                assert obs["observation.images.side"].shape[-2:] == (144, 256), obs["observation.images.side"].shape

                act = policy.select_action(obs)

            # Normalize shape: expect (1,6) or (6,)
            if isinstance(act, torch.Tensor):
                act_np = act.detach().float().cpu().numpy()
            else:
                act_np = np.asarray(act, dtype=np.float32)

            act_np = act_np.reshape(-1)
            if act_np.shape[0] != 6:
                raise RuntimeError(f"Expected 6 action outputs, got shape {act_np.shape}")

            # Map to wheel + arm
            v = float(act_np[0])
            w_cmd = float(act_np[1])

            v = clamp(v, -MAX_V, MAX_V)
            w_cmd = clamp(w_cmd, -MAX_W, MAX_W)

            # Send wheels
            esp.send_cmd(v, w_cmd)

            # Arm u01 outputs assumed in [0..1]
            if arm is not None:
                u = {
                    arm_ids_out[0]: clamp01(float(act_np[2])),
                    arm_ids_out[1]: clamp01(float(act_np[3])),
                    arm_ids_out[2]: clamp01(float(act_np[4])),
                    arm_ids_out[3]: clamp01(float(act_np[5])),
                }
                arm.set_target_u(u)

            # Preview overlay
            if args.preview:
                front_bgr = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
                side_bgr = cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR)
                combined = cv2.hconcat([front_bgr, side_bgr])
                txt = f"hz={loop_hz:5.1f} v={v:+.2f} w={w_cmd:+.2f} a={act_np[2]:.2f},{act_np[3]:.2f},{act_np[4]:.2f},{act_np[5]:.2f}"
                cv2.putText(combined, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("policy", combined)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

            # Rate keep
            dt = time.time() - t0
            time.sleep(max(0.0, dt_target - dt))

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping: sending wheels 0,0")
        try:
            esp.send_cmd(0.0, 0.0)
        except Exception:
            pass

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

        try:
            esp.close()
        except Exception:
            pass

        try:
            if args.preview:
                cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
