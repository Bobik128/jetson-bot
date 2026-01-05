#!/usr/bin/env python3
"""
Verify/sweep learned limits from LimitsDB JSON by slowly traversing edges.
Instant terminate: press 'q' then Enter, or Ctrl+C.

Requires: scservo_sdk
"""

from __future__ import annotations

import argparse
import json
import time
import threading
from typing import Dict, List, Optional, Tuple

from scservo_sdk import PortHandler, PacketHandler

# Addresses
ADDR_TORQUE_ENABLE = 40
TORQUE_ON = 1
TORQUE_OFF = 0

ADDR_GOAL_POS_BLOCK = 0x2A  # pos/time/speed block (6 bytes)
ADDR_PRESENT_POSITION = 0x38

def lo(x: int) -> int: return x & 0xFF
def hi(x: int) -> int: return (x >> 8) & 0xFF

def raw_to_deg(raw: int) -> float:
    return (raw % 4096) * (360.0 / 4096.0)

def deg_to_raw(deg: float) -> int:
    deg = deg % 360.0
    return int(round(deg * (4096.0 / 360.0))) & 0xFFFF

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
    pk.write1ByteTxRx(ph, motor_id, addr, int(value) & 0xFF)

def read_u16(pk: PacketHandler, ph: PortHandler, motor_id: int, addr: int) -> int:
    val, comm, err = pk.read2ByteTxRx(ph, motor_id, addr)
    if comm != 0 or err != 0:
        raise RuntimeError(f"read_u16 failed: id={motor_id} addr=0x{addr:02X} comm={comm} err={err}")
    return int(val)

def write_goal_pos_block(pk: PacketHandler, ph: PortHandler, motor_id: int, pos_raw: int, time_ms: int, speed: int):
    data = [lo(pos_raw), hi(pos_raw), lo(time_ms), hi(time_ms), lo(speed), hi(speed)]
    if hasattr(pk, "writeTxRx"):
        pk.writeTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK, 6, data)
    elif hasattr(pk, "writeNByteTxRx"):
        pk.writeNByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK, 6, data)
    else:
        # fallback: not ideal but works on many forks
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 0, pos_raw)
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 2, time_ms)
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 4, speed)

def set_torque(pk: PacketHandler, ph: PortHandler, ids: List[int], on: bool):
    v = TORQUE_ON if on else TORQUE_OFF
    for mid in ids:
        write_u8(pk, ph, mid, ADDR_TORQUE_ENABLE, v)

def parse_key_to_bins(key: str) -> Dict[int, int]:
    # key like "2=10|3=22" or "ROOT"
    if key == "ROOT":
        return {}
    out = {}
    for part in key.split("|"):
        sid, b = part.split("=")
        out[int(sid)] = int(b)
    return out

class KillSwitch:
    def __init__(self):
        self.stop = False
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()

    def _run(self):
        while not self.stop:
            try:
                s = input()
            except EOFError:
                return
            if s.strip().lower() == "q":
                self.stop = True
                return

def load_db(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def move_deg(pk, ph, mid: int, deg: float, speed: int, time_ms: int):
    write_goal_pos_block(pk, ph, mid, deg_to_raw(deg), time_ms, speed)

def wait_until(pk, ph, mid: int, target_deg: float, tol_deg: float, timeout_s: float, kill: KillSwitch):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if kill.stop:
            return False
        cur = raw_to_deg(read_u16(pk, ph, mid, ADDR_PRESENT_POSITION))
        # handle wrap-around by using shortest circular distance
        diff = abs(((cur - target_deg + 180.0) % 360.0) - 180.0)
        if diff <= tol_deg:
            return True
        time.sleep(0.02)
    return False

def verify(args) -> int:
    db = load_db(args.limits)
    ids = [int(x) for x in db["ids"]]
    priority = [int(x) for x in db["priority"]]
    bin_deg = float(db["bin_deg"])

    ph, pk = open_bus(args.port, args.baudrate, protocol_end=0)
    kill = KillSwitch()
    kill.start()

    print("Verify mode started.")
    print("Instant stop: type 'q' + Enter, or Ctrl+C.\n")

    try:
        set_torque(pk, ph, ids, on=True)

        # 1) Global edges
        if args.global_edges:
            print("Sweeping global min/max edges...")
            for mid in priority:
                if kill.stop: break
                gl = db["global_limits"][str(mid)]
                mn, mx = float(gl["min_deg"]), float(gl["max_deg"])
                if mx < mn:
                    print(f"  ID {mid}: no valid global range recorded, skipping.")
                    continue

                print(f"  ID {mid}: min={mn:.2f} max={mx:.2f}")

                for target in (mn, mx, (mn + mx) / 2.0):
                    if kill.stop: break
                    move_deg(pk, ph, mid, target, args.speed, args.time_ms)
                    ok = wait_until(pk, ph, mid, target, args.tol_deg, args.timeout_s, kill)
                    if not ok:
                        print(f"    WARN: ID {mid} did not reach {target:.2f}° within timeout.")
                    time.sleep(args.pause_s)

        # 2) Conditional edges
        if args.conditional_edges and not kill.stop:
            print("\nSweeping conditional edges...")
            cond = db["conditional_limits"]  # mid -> key -> {min_deg,max_deg}

            for mid in priority:
                if kill.stop: break
                mid_str = str(mid)
                if mid_str not in cond:
                    continue
                keys = list(cond[mid_str].keys())
                if not keys:
                    continue

                print(f"  ID {mid}: {len(keys)} conditional keys")

                # Limit how many keys to sweep if huge
                keys = keys[:args.max_keys_per_servo]

                for key in keys:
                    if kill.stop: break
                    bins = parse_key_to_bins(key)

                    # Set higher-priority servos to the center of their bin
                    for hid in priority:
                        if hid == mid:
                            break
                        if hid in bins:
                            center_deg = bins[hid] * bin_deg
                            move_deg(pk, ph, hid, center_deg, args.speed, args.time_ms)
                            wait_until(pk, ph, hid, center_deg, args.tol_deg, args.timeout_s, kill)

                    rng = cond[mid_str][key]
                    mn, mx = float(rng["min_deg"]), float(rng["max_deg"])
                    if mx < mn:
                        continue

                    # Sweep target servo to its conditional edges
                    for target in (mn, mx, (mn + mx) / 2.0):
                        if kill.stop: break
                        move_deg(pk, ph, mid, target, args.speed, args.time_ms)
                        wait_until(pk, ph, mid, target, args.tol_deg, args.timeout_s, kill)
                        time.sleep(args.pause_s)

        print("\nVerify finished." if not kill.stop else "\nVerify terminated by user.")
        return 0

    except KeyboardInterrupt:
        print("\nVerify terminated (Ctrl+C).")
        return 130

    finally:
        # Disable torque on exit if requested
        if args.torque_off_on_exit:
            try:
                set_torque(pk, ph, ids, on=False)
            except Exception:
                pass
        close_bus(ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--limits", required=True, help="Path to limits JSON created by teach_limits.py")

    ap.add_argument("--speed", type=int, default=600, help="Feetech speed field (u16) in goal block")
    ap.add_argument("--time-ms", type=int, default=0, help="Feetech time field (u16) in goal block")
    ap.add_argument("--pause-s", type=float, default=0.3, help="Pause between moves")
    ap.add_argument("--tol-deg", type=float, default=2.0, help="Position tolerance (degrees)")
    ap.add_argument("--timeout-s", type=float, default=4.0, help="Timeout waiting for a move")

    ap.add_argument("--global-edges", action="store_true", help="Sweep global min/max per servo")
    ap.add_argument("--conditional-edges", action="store_true", help="Sweep conditional min/max per servo")
    ap.add_argument("--max-keys-per-servo", type=int, default=50, help="Limit conditional keys swept per servo")

    ap.add_argument("--torque-off-on-exit", action="store_true", help="Disable torque when exiting")
    args = ap.parse_args()

    if not args.global_edges and not args.conditional_edges:
        # sensible default
        args.global_edges = True
        args.conditional_edges = True

    return verify(args)

if __name__ == "__main__":
    raise SystemExit(main())
