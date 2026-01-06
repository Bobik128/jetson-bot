#!/usr/bin/env python3
"""
Real-time limits teaching for Feetech STS3215 using read-only sampling (no servo writes).

Key properties:
- Samples at high rate (default 40 Hz) by reading Present_Position only.
- Expands allowed ranges:
    - global per joint
    - conditional per joint based on binned angles of higher-priority joints
- Stores sample_count per conditional key for confidence.
- Periodically autosaves to JSON (atomic write).
- Optional noise filtering: require angle change > deadband to update (default small).

Usage example:
  python3 teach_limits_realtime.py --port /dev/ttyACM0 --baudrate 1000000 \
      --ids 2 3 4 6 --priority 2 3 4 6 --bin-deg 5 \
      --hz 40 --out limits.json --autosave-s 5

Controls (interactive):
  p  print current angles
  s  force save now
  q  quit (saves)
  stats  show database size and sample counts
  torque on|off (optional; OFF lets you move by hand)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from scservo_sdk import PortHandler, PacketHandler

# ----------------------------
# STS3215 control table
# ----------------------------
ADDR_TORQUE_ENABLE = 40
TORQUE_ON = 1
TORQUE_OFF = 0

ADDR_PRESENT_POSITION = 0x38  # read2ByteTxRx (u16)


# ----------------------------
# Conversions
# ----------------------------
def raw_to_deg(raw: int) -> float:
    # STS3215 typical: 0..4095 => 0..360 degrees
    return (int(raw) % 4096) * (360.0 / 4096.0)

def bin_angle(deg: float, bin_deg: float) -> int:
    return int(round(deg / bin_deg))

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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
    # best-effort write; we do not fail hard here
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


# ----------------------------
# Limits DB
# ----------------------------
@dataclass
class LimitsDB:
    version: int
    port: str
    baudrate: int
    ids: List[int]
    priority: List[int]
    bin_deg: float

    # global_limits[id] = {min_deg,max_deg}
    global_limits: Dict[str, Dict[str, float]]

    # conditional_limits[id][key] = {min_deg,max_deg,count}
    conditional_limits: Dict[str, Dict[str, Dict[str, float]]]

    # optional forbidden pockets (still supported, not required)
    forbidden: List[Dict[str, int]]

    notes: str = ""

    def save_atomic(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    @staticmethod
    def new(port: str, baudrate: int, ids: List[int], priority: List[int], bin_deg: float, notes: str) -> "LimitsDB":
        return LimitsDB(
            version=2,
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

    @staticmethod
    def load(path: str) -> "LimitsDB":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # backward compatibility: ensure "count" exists
        for sid, mp in d.get("conditional_limits", {}).items():
            for k, rec in mp.items():
                rec.setdefault("count", 0)
        d.setdefault("forbidden", [])
        return LimitsDB(**d)


# ----------------------------
# Conditioning key
# ----------------------------
def key_for_target(priority: List[int], target_id: int, angles: Dict[int, float], bin_deg: float) -> str:
    parts = []
    for pid in priority:
        if pid == target_id:
            break
        parts.append(f"{pid}={bin_angle(angles[pid], bin_deg)}")
    return "|".join(parts) if parts else "ROOT"


# ----------------------------
# Sampler thread
# ----------------------------
class Sampler:
    def __init__(
        self,
        pk: PacketHandler,
        ph: PortHandler,
        db: LimitsDB,
        out_path: str,
        hz: float,
        autosave_s: float,
        deadband_deg: float,
        torque_off: bool,
        print_hz: float,
    ):
        self.pk = pk
        self.ph = ph
        self.db = db
        self.out_path = out_path
        self.hz = hz
        self.autosave_s = autosave_s
        self.deadband_deg = deadband_deg
        self.print_hz = print_hz

        self.ids = db.ids
        self.priority = db.priority
        self.bin_deg = db.bin_deg

        self.stop = False
        self.last_angles: Optional[Dict[int, float]] = None
        self.last_autosave = 0.0
        self.last_print = 0.0

        # recommended: torque OFF for manual teaching
        if torque_off:
            set_torque(self.pk, self.ph, self.ids, on=False)

        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def get_last_angles(self) -> Optional[Dict[int, float]]:
        with self._lock:
            return None if self.last_angles is None else dict(self.last_angles)

    def force_save(self):
        with self._lock:
            self.db.save_atomic(self.out_path)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            keys_total = sum(len(self.db.conditional_limits[str(mid)]) for mid in self.ids)
            return {
                "keys_total": keys_total,
                "forbidden": len(self.db.forbidden),
            }

    def _snapshot_angles(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for mid in self.priority:
            raw = read_u16(self.pk, self.ph, mid, ADDR_PRESENT_POSITION)
            out[mid] = raw_to_deg(raw)
        return out

    def _update_db(self, angles: Dict[int, float]):
        # deadband to reduce noise updates
        if self.last_angles is not None and self.deadband_deg > 0:
            # if nothing moved, skip update
            moved = any(abs(angles[mid] - self.last_angles[mid]) >= self.deadband_deg for mid in self.priority)
            if not moved:
                return

        # Global
        for mid in self.ids:
            deg = angles[mid]
            g = self.db.global_limits[str(mid)]
            g["min_deg"] = min(g["min_deg"], deg)
            g["max_deg"] = max(g["max_deg"], deg)

        # Conditional (per target servo)
        for mid in self.ids:
            deg = angles[mid]
            k = key_for_target(self.priority, mid, angles, self.bin_deg)
            mp = self.db.conditional_limits[str(mid)]
            if k not in mp:
                mp[k] = {"min_deg": 9999.0, "max_deg": -9999.0, "count": 0}
            mp[k]["min_deg"] = min(mp[k]["min_deg"], deg)
            mp[k]["max_deg"] = max(mp[k]["max_deg"], deg)
            mp[k]["count"] = int(mp[k].get("count", 0)) + 1

    def _run(self):
        period = 1.0 / self.hz if self.hz > 0 else 0.025
        self.last_autosave = time.time()
        self.last_print = time.time()

        while not self.stop:
            t0 = time.time()
            try:
                angles = self._snapshot_angles()
            except Exception:
                # if a read fails, skip this cycle
                time.sleep(period)
                continue

            with self._lock:
                self._update_db(angles)
                self.last_angles = angles

                now = time.time()
                if self.autosave_s > 0 and (now - self.last_autosave) >= self.autosave_s:
                    self.db.save_atomic(self.out_path)
                    self.last_autosave = now

                if self.print_hz > 0 and (now - self.last_print) >= (1.0 / self.print_hz):
                    # lightweight status line
                    keys_total = sum(len(self.db.conditional_limits[str(mid)]) for mid in self.ids)
                    sys.stdout.write(
                        f"\r[hz={self.hz:.1f}] keys={keys_total}  "
                        + "  ".join([f"{mid}:{angles[mid]:6.1f}°" for mid in self.priority])
                        + " " * 10
                    )
                    sys.stdout.flush()
                    self.last_print = now

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--priority", type=int, nargs="+", default=None,
                    help="Must contain exactly same IDs as --ids. Earlier = higher priority.")
    ap.add_argument("--bin-deg", type=float, default=5.0)

    ap.add_argument("--hz", type=float, default=40.0, help="Sampling rate (read-only).")
    ap.add_argument("--deadband-deg", type=float, default=0.25, help="Skip DB updates if nothing moved >= this.")
    ap.add_argument("--autosave-s", type=float, default=5.0, help="Autosave interval (seconds).")
    ap.add_argument("--print-hz", type=float, default=5.0, help="Print status line rate; 0 disables.")

    ap.add_argument("--out", required=True, help="JSON limits output path (atomic overwrite).")
    ap.add_argument("--load", default=None, help="Load existing limits JSON and keep expanding it.")
    ap.add_argument("--notes", default="")

    ap.add_argument("--torque-off", action="store_true",
                    help="Disable torque at start to allow hand-moving (recommended for teaching).")

    args = ap.parse_args()

    ids = args.ids
    priority = args.priority if args.priority else sorted(ids)
    if set(priority) != set(ids) or len(priority) != len(ids):
        print("[ERROR] --priority must contain exactly the same IDs as --ids (same elements, no duplicates).")
        return 2

    ph, pk = open_bus(args.port, args.baudrate, protocol_end=0)

    try:
        if args.load:
            db = LimitsDB.load(args.load)
            # normalize runtime fields
            db.port = args.port
            db.baudrate = args.baudrate
            db.ids = ids
            db.priority = priority
            db.bin_deg = args.bin_deg
            if args.notes:
                db.notes = args.notes
            print(f"[OK] Loaded existing DB: {args.load}")
        else:
            db = LimitsDB.new(args.port, args.baudrate, ids, priority, args.bin_deg, args.notes)

        sampler = Sampler(
            pk=pk,
            ph=ph,
            db=db,
            out_path=args.out,
            hz=args.hz,
            autosave_s=args.autosave_s,
            deadband_deg=args.deadband_deg,
            torque_off=args.torque_off,
            print_hz=args.print_hz,
        )
        sampler.start()

        print("\nReal-time teaching started (read-only).")
        print("Commands: p (print) | s (save now) | stats | torque on/off | q (quit)\n")

        while True:
            cmd = input("\n> ").strip().lower()

            if cmd in ("q", "quit", "exit"):
                sampler.stop = True
                sampler.force_save()
                print(f"[OK] Saved on exit: {args.out}")
                break

            if cmd == "p":
                angles = sampler.get_last_angles()
                if not angles:
                    print("[INFO] No angles yet.")
                else:
                    print("Angles:", "  ".join([f"{mid}:{angles[mid]:7.2f}°" for mid in priority]))
                continue

            if cmd == "s":
                sampler.force_save()
                print(f"[OK] Saved: {args.out}")
                continue

            if cmd == "stats":
                st = sampler.stats()
                print(f"keys_total={st['keys_total']} forbidden={st['forbidden']}")
                continue

            if cmd.startswith("torque "):
                v = cmd.split(" ", 1)[1].strip()
                if v == "on":
                    set_torque(pk, ph, ids, on=True)
                    print("[OK] Torque ON")
                elif v == "off":
                    set_torque(pk, ph, ids, on=False)
                    print("[OK] Torque OFF")
                else:
                    print("[ERROR] torque on|off")
                continue

            print("[INFO] Unknown command.")

        return 0

    finally:
        close_bus(ph)


if __name__ == "__main__":
    raise SystemExit(main())