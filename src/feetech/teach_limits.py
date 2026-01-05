#!/usr/bin/env python3
"""
Teach / record hierarchical soft limits and forbidden combinations for Feetech STS3215 bus servos.

- You physically move servos by hand (torque OFF)
- Press keys to sample states, mark forbidden combos, etc.
- Saves to JSON at a path you choose.

Works universally for N motors, with a priority order (lower ID = higher priority by default,
but you can pass an explicit priority list).

Requires: scservo_sdk (as in your working setup)

Typical use:
  python3 teach_limits.py --port /dev/ttyACM0 --baudrate 1000000 --ids 2 3 4 6 --out limits.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

from scservo_sdk import PortHandler, PacketHandler


# ----------------------------
# STS3215 addresses (common)
# ----------------------------
ADDR_TORQUE_ENABLE = 40  # commonly used; if your setup works, keep it
TORQUE_ON = 1
TORQUE_OFF = 0

ADDR_PRESENT_POSITION = 0x38  # 2 bytes


# ----------------------------
# Units conversion
# ----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def raw_to_deg(raw: int) -> float:
    # STS3215 typical: 0..4095 == 0..360 deg
    raw = int(raw) & 0xFFFF
    return (raw % 4096) * (360.0 / 4096.0)

def deg_to_raw(deg: float) -> int:
    # map to 0..4095
    deg = deg % 360.0
    return int(round(deg * (4096.0 / 360.0))) & 0xFFFF

def bin_angle(deg: float, bin_deg: float) -> int:
    # integer bin index
    return int(round(deg / bin_deg))

def unbin_angle(bin_idx: int, bin_deg: float) -> float:
    return float(bin_idx) * bin_deg


# ----------------------------
# Bus helpers
# ----------------------------
def open_bus(port: str, baudrate: int, protocol_end: int = 0):
    ph = PortHandler(port)
    pk = PacketHandler(protocol_end)
    if not ph.openPort():
        raise RuntimeError(f"Failed to open port {port}")
    if not ph.setBaudRate(baudrate):
        raise RuntimeError(f"Failed to set baudrate {baudrate} on {port}")
    return ph, pk

def close_bus(ph: PortHandler):
    try:
        ph.closePort()
    except Exception:
        pass

def write_u8(pk: PacketHandler, ph: PortHandler, motor_id: int, addr: int, value: int):
    comm, err = pk.write1ByteTxRx(ph, motor_id, addr, int(value) & 0xFF)
    # Some writes may not return a status reliably; do not hard-fail here.
    return comm, err

def read_u16(pk: PacketHandler, ph: PortHandler, motor_id: int, addr: int) -> int:
    val, comm, err = pk.read2ByteTxRx(ph, motor_id, addr)
    if comm != 0 or err != 0:
        raise RuntimeError(f"read_u16 failed: id={motor_id} addr=0x{addr:02X} comm={comm} err={err}")
    return int(val)

def set_torque(pk: PacketHandler, ph: PortHandler, ids: List[int], on: bool):
    v = TORQUE_ON if on else TORQUE_OFF
    for mid in ids:
        write_u8(pk, ph, mid, ADDR_TORQUE_ENABLE, v)


# ----------------------------
# Data model
# ----------------------------
@dataclass
class LimitRange:
    min_deg: float = 9999.0
    max_deg: float = -9999.0

    def update(self, deg: float):
        self.min_deg = min(self.min_deg, deg)
        self.max_deg = max(self.max_deg, deg)

    def is_valid(self) -> bool:
        return self.max_deg >= self.min_deg

    def to_dict(self):
        return {"min_deg": self.min_deg, "max_deg": self.max_deg}

    @staticmethod
    def from_dict(d):
        return LimitRange(min_deg=float(d["min_deg"]), max_deg=float(d["max_deg"]))


@dataclass
class LimitsDB:
    version: int
    port: str
    baudrate: int
    ids: List[int]
    priority: List[int]
    bin_deg: float
    global_limits: Dict[str, Dict[str, float]]
    conditional_limits: Dict[str, Dict[str, Dict[str, float]]]
    forbidden: List[Dict[str, int]]
    notes: str = ""

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
        print(f"[OK] Saved limits to: {path}")

    @staticmethod
    def load(path: str) -> "LimitsDB":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return LimitsDB(**d)


# ----------------------------
# Constraint keying
# ----------------------------
def higher_key(priority: List[int], target_id: int, angles_deg: Dict[int, float], bin_deg: float) -> str:
    """
    Build a key for target_id based on binned angles of higher-priority servos only.
    Format: "2=10|3=22" (bin indices)
    """
    key_parts = []
    for pid in priority:
        if pid == target_id:
            break
        b = bin_angle(angles_deg[pid], bin_deg)
        key_parts.append(f"{pid}={b}")
    return "|".join(key_parts) if key_parts else "ROOT"

def snapshot_angles(pk: PacketHandler, ph: PortHandler, ids: List[int]) -> Dict[int, float]:
    out = {}
    for mid in ids:
        raw = read_u16(pk, ph, mid, ADDR_PRESENT_POSITION)
        out[mid] = raw_to_deg(raw)
    return out

def pretty_snapshot(priority: List[int], angles: Dict[int, float]) -> str:
    parts = []
    for mid in priority:
        parts.append(f"{mid}:{angles[mid]:7.2f}°")
    return "  ".join(parts)


# ----------------------------
# Teach loop
# ----------------------------
def teach(args) -> int:
    ids = args.ids
    priority = args.priority if args.priority else sorted(ids)  # default: numeric order
    if set(priority) != set(ids):
        print("[ERROR] --priority must contain exactly the same IDs as --ids")
        return 2

    ph, pk = open_bus(args.port, args.baudrate, protocol_end=0)

    # Init DB
    db = LimitsDB(
        version=1,
        port=args.port,
        baudrate=args.baudrate,
        ids=ids,
        priority=priority,
        bin_deg=args.bin_deg,
        global_limits={str(mid): {"min_deg": 9999.0, "max_deg": -9999.0} for mid in ids},
        conditional_limits={str(mid): {} for mid in ids},  # mid -> key -> {min_deg,max_deg}
        forbidden=[],
        notes=args.notes or "",
    )

    # Ensure torque off for manual movement
    set_torque(pk, ph, ids, on=False)
    print("\nTorque disabled. You can move servos by hand.\n")

    help_text = f"""
Controls:
  s  = sample/update limits (global + conditional for each servo)
  f  = mark current binned state as FORBIDDEN (combination not allowed)
  u  = undo last forbidden mark
  p  = print current angles
  w  = write/save JSON now
  t  = toggle torque ON/OFF (careful; ON prevents hand movement)
  q  = quit (auto-saves)

Notes:
  - Priority order: {priority}
  - Conditional limits: each servo's allowed range is recorded as a function of higher-priority servo bins.
  - Bin size: {args.bin_deg} degrees
"""
    print(help_text)

    torque_on = False
    last_angles: Optional[Dict[int, float]] = None

    try:
        while True:
            # show prompt
            cmd = input("> ").strip().lower()

            if cmd == "q":
                db.save(args.out)
                return 0

            if cmd == "p":
                angles = snapshot_angles(pk, ph, priority)
                last_angles = angles
                print(pretty_snapshot(priority, angles))
                continue

            if cmd == "t":
                torque_on = not torque_on
                set_torque(pk, ph, ids, on=torque_on)
                print(f"[OK] Torque {'ENABLED' if torque_on else 'DISABLED'}")
                continue

            if cmd == "w":
                db.save(args.out)
                continue

            if cmd == "u":
                if db.forbidden:
                    removed = db.forbidden.pop()
                    print(f"[OK] Removed last forbidden: {removed}")
                else:
                    print("[INFO] No forbidden states to remove.")
                continue

            # Always re-snapshot for s/f
            if cmd in ("s", "f"):
                angles = snapshot_angles(pk, ph, priority)
                last_angles = angles
                print(pretty_snapshot(priority, angles))

            if cmd == "s":
                # Update global limits and conditional limits
                for mid in ids:
                    deg = angles[mid]
                    g = db.global_limits[str(mid)]
                    g["min_deg"] = min(g["min_deg"], deg)
                    g["max_deg"] = max(g["max_deg"], deg)

                    key = higher_key(priority, mid, angles, db.bin_deg)
                    cm = db.conditional_limits[str(mid)]
                    if key not in cm:
                        cm[key] = {"min_deg": 9999.0, "max_deg": -9999.0}
                    cm[key]["min_deg"] = min(cm[key]["min_deg"], deg)
                    cm[key]["max_deg"] = max(cm[key]["max_deg"], deg)

                print("[OK] Sample recorded (global + conditional).")
                continue

            if cmd == "f":
                # Mark forbidden state based on binned angles of ALL servos
                state_bins = {str(mid): bin_angle(angles[mid], db.bin_deg) for mid in priority}
                db.forbidden.append(state_bins)
                print(f"[OK] Marked forbidden state: {state_bins}")
                continue

            print("[INFO] Unknown command. Use s/f/p/w/t/u/q.")

    finally:
        # leave torque off for safety
        try:
            set_torque(pk, ph, ids, on=False)
        except Exception:
            pass
        close_bus(ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--ids", type=int, nargs="+", required=True, help="Servo IDs on bus")
    ap.add_argument("--priority", type=int, nargs="+", default=None,
                    help="Priority order (must contain exactly the same IDs). Lower index = higher priority.")
    ap.add_argument("--bin-deg", type=float, default=5.0, help="Bin size for conditional constraints (degrees)")
    ap.add_argument("--out", required=True, help="Path to output JSON file")
    ap.add_argument("--notes", default="", help="Optional notes saved into JSON")
    args = ap.parse_args()
    return teach(args)

if __name__ == "__main__":
    raise SystemExit(main())
