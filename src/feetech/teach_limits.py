#!/usr/bin/env python3
"""
Teach hierarchical soft limits + forbidden combinations for Feetech STS3215.

- Uses scservo_sdk (same stack as your working setup).
- Default: torque OFF so you can move by hand.
- Manual sampling:
    s = record allowed state (updates global + conditional min/max)
    f = record forbidden binned combination
- Optional continuous sampling: --auto-sample-hz (records allowed states automatically)
- Saves JSON database to --out path (overwrites only on 'w' or 'q', or during auto-sample autosave if enabled)

Example:
  python3 teach_limits.py --port /dev/ttyACM0 --baudrate 1000000 --ids 2 3 4 6 --priority 2 3 4 6 --out limits.json

Notes:
- Priority defines constraints: lower-priority servos are conditioned on binned angles of higher-priority servos.
- Bin size affects resolution and teaching effort. Start with 5 degrees.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from scservo_sdk import PortHandler, PacketHandler

# ----------------------------
# Control table (common STS3215)
# ----------------------------
ADDR_TORQUE_ENABLE = 40
TORQUE_ON = 1
TORQUE_OFF = 0

ADDR_PRESENT_POSITION = 0x38  # read2ByteTxRx


# ----------------------------
# Conversion helpers
# ----------------------------
def raw_to_deg(raw: int) -> float:
    # STS3215 commonly: 0..4095 maps to 0..360 degrees
    return (int(raw) % 4096) * (360.0 / 4096.0)

def bin_angle(deg: float, bin_deg: float) -> int:
    return int(round(deg / bin_deg))

def snapshot_key_for_target(priority: List[int], target_id: int, angles_deg: Dict[int, float], bin_deg: float) -> str:
    """
    Conditional key for target_id based on binned angles of higher-priority servos.
    Example: "2=10|3=22" or "ROOT" if no higher servos.
    """
    parts = []
    for pid in priority:
        if pid == target_id:
            break
        parts.append(f"{pid}={bin_angle(angles_deg[pid], bin_deg)}")
    return "|".join(parts) if parts else "ROOT"


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
    # Some writes might not return status reliably; we do not hard-fail here.
    pk.write1ByteTxRx(ph, motor_id, addr, int(value) & 0xFF)

def read_u16(pk: PacketHandler, ph: PortHandler, motor_id: int, addr: int) -> int:
    val, comm, err = pk.read2ByteTxRx(ph, motor_id, addr)
    if comm != 0 or err != 0:
        raise RuntimeError(f"read_u16 failed: id={motor_id} addr=0x{addr:02X} comm={comm} err={err}")
    return int(val)

def set_torque(pk: PacketHandler, ph: PortHandler, ids: List[int], on: bool):
    v = TORQUE_ON if on else TORQUE_OFF
    for mid in ids:
        write_u8(pk, ph, mid, ADDR_TORQUE_ENABLE, v)

def snapshot_angles(pk: PacketHandler, ph: PortHandler, ids: List[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for mid in ids:
        raw = read_u16(pk, ph, mid, ADDR_PRESENT_POSITION)
        out[mid] = raw_to_deg(raw)
    return out


# ----------------------------
# DB model
# ----------------------------
@dataclass
class LimitsDB:
    version: int
    port: str
    baudrate: int
    ids: List[int]
    priority: List[int]
    bin_deg: float

    # per-servo global min/max
    global_limits: Dict[str, Dict[str, float]]
    # per-servo conditional limits: mid -> key -> {min_deg,max_deg}
    conditional_limits: Dict[str, Dict[str, Dict[str, float]]]
    # forbidden binned full states: list of {"2":bin, "3":bin, ...}
    forbidden: List[Dict[str, int]]

    notes: str = ""

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
        print(f"[OK] Saved: {path}")

    @staticmethod
    def new(port: str, baudrate: int, ids: List[int], priority: List[int], bin_deg: float, notes: str) -> "LimitsDB":
        return LimitsDB(
            version=1,
            port=port,
            baudrate=baudrate,
            ids=ids,
            priority=priority,
            bin_deg=bin_deg,
            global_limits={str(mid): {"min_deg": 9999.0, "max_deg": -9999.0} for mid in ids},
            conditional_limits={str(mid): {} for mid in ids},
            forbidden=[],
            notes=notes or "",
        )

def pretty_angles(priority: List[int], angles: Dict[int, float]) -> str:
    return "  ".join([f"{mid}:{angles[mid]:7.2f}°" for mid in priority])


# ----------------------------
# Auto-sampling thread
# ----------------------------
class AutoSampler:
    def __init__(self, hz: float):
        self.hz = hz
        self.enabled = hz > 0
        self.stop = False
        self._thread: Optional[threading.Thread] = None

    def start(self, fn_sample):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, args=(fn_sample,), daemon=True)
        self._thread.start()

    def _run(self, fn_sample):
        period = 1.0 / self.hz
        while not self.stop:
            t0 = time.time()
            try:
                fn_sample(auto=True)
            except Exception:
                # keep running; teach mode should be resilient
                pass
            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))


# ----------------------------
# Teach
# ----------------------------
def teach(args) -> int:
    ids = args.ids
    priority = args.priority if args.priority else sorted(ids)
    if set(priority) != set(ids) or len(priority) != len(ids):
        print("[ERROR] --priority must contain exactly the same IDs as --ids (same elements, no duplicates).")
        return 2

    ph, pk = open_bus(args.port, args.baudrate, protocol_end=0)
    db = LimitsDB.new(args.port, args.baudrate, ids, priority, args.bin_deg, args.notes)

    # default torque OFF so you can hand-move
    torque_on = False
    set_torque(pk, ph, ids, on=False)

    # optionally load existing DB and continue
    if args.load:
        with open(args.load, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        # minimal safety: ensure ids match
        if set(int(x) for x in loaded["ids"]) != set(ids):
            print("[ERROR] Loaded DB IDs do not match --ids.")
            close_bus(ph)
            return 2
        db = LimitsDB(**loaded)
        db.port = args.port
        db.baudrate = args.baudrate
        db.priority = priority
        db.bin_deg = args.bin_deg
        print(f"[OK] Loaded existing DB: {args.load}")

    last_save_time = 0.0

    def record_allowed(auto: bool = False):
        nonlocal last_save_time
        angles = snapshot_angles(pk, ph, priority)

        # update global + conditional for each servo
        for mid in ids:
            deg = angles[mid]
            g = db.global_limits[str(mid)]
            g["min_deg"] = min(g["min_deg"], deg)
            g["max_deg"] = max(g["max_deg"], deg)

            key = snapshot_key_for_target(priority, mid, angles, db.bin_deg)
            cm = db.conditional_limits[str(mid)]
            if key not in cm:
                cm[key] = {"min_deg": 9999.0, "max_deg": -9999.0}
            cm[key]["min_deg"] = min(cm[key]["min_deg"], deg)
            cm[key]["max_deg"] = max(cm[key]["max_deg"], deg)

        if not auto:
            print(pretty_angles(priority, angles))
            print("[OK] Sample recorded (allowed).")
        else:
            # optional autosave interval
            if args.auto_save_s > 0:
                now = time.time()
                if now - last_save_time >= args.auto_save_s:
                    db.save(args.out)
                    last_save_time = now

    sampler = AutoSampler(args.auto_sample_hz)
    sampler.start(record_allowed)

    help_text = f"""
Teach mode (STS3215). Torque is OFF by default (hand-move enabled).
IDs: {ids}
Priority: {priority}
Bin size: {db.bin_deg} deg

Commands:
  s  = record allowed sample (global + conditional)
  f  = mark current binned state as FORBIDDEN
  u  = undo last forbidden mark
  p  = print current angles
  w  = save JSON now (overwrites --out)
  t  = toggle torque ON/OFF
  q  = quit (saves once)

Auto-sampling:
  --auto-sample-hz {args.auto_sample_hz}  (0 means disabled)
  --auto-save-s {args.auto_save_s}  (autosave interval when auto-sampling)
"""
    print(help_text)

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd == "q":
                db.save(args.out)
                return 0

            if cmd == "w":
                db.save(args.out)
                continue

            if cmd == "t":
                torque_on = not torque_on
                set_torque(pk, ph, ids, on=torque_on)
                print(f"[OK] Torque {'ON' if torque_on else 'OFF'}")
                continue

            if cmd == "p":
                angles = snapshot_angles(pk, ph, priority)
                print(pretty_angles(priority, angles))
                continue

            if cmd == "u":
                if db.forbidden:
                    removed = db.forbidden.pop()
                    print(f"[OK] Removed forbidden: {removed}")
                else:
                    print("[INFO] No forbidden entries.")
                continue

            if cmd == "s":
                record_allowed(auto=False)
                continue

            if cmd == "f":
                angles = snapshot_angles(pk, ph, priority)
                state_bins = {str(mid): bin_angle(angles[mid], db.bin_deg) for mid in priority}
                db.forbidden.append(state_bins)
                print(pretty_angles(priority, angles))
                print(f"[OK] Forbidden recorded: {state_bins}")
                continue

            print("[INFO] Unknown command. Use s/f/u/p/w/t/q.")

    finally:
        sampler.stop = True
        # torque OFF for safety
        try:
            set_torque(pk, ph, ids, on=False)
        except Exception:
            pass
        close_bus(ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--priority", type=int, nargs="+", default=None,
                    help="Must contain exactly the same IDs as --ids. Earlier = higher priority.")
    ap.add_argument("--bin-deg", type=float, default=5.0)
    ap.add_argument("--out", required=True, help="Path to output JSON (overwritten only on save/quit/autosave).")
    ap.add_argument("--load", default=None, help="Load existing JSON and continue teaching into it.")
    ap.add_argument("--notes", default="")

    ap.add_argument("--auto-sample-hz", type=float, default=0.0,
                    help="If >0, continuously records allowed states at this rate (Hz).")
    ap.add_argument("--auto-save-s", type=float, default=10.0,
                    help="When auto-sampling, autosave every N seconds. Set 0 to disable autosave.")

    args = ap.parse_args()
    return teach(args)

if __name__ == "__main__":
    raise SystemExit(main())
