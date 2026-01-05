#!/usr/bin/env python3
"""
Verify learned limits by sweeping edges with FULL-POSE moves.

For each conditional key:
- Move ALL servos into a consistent pose:
    - higher-priority servos -> bin centers from the key
    - target servo -> min edge, max edge, midpoint
    - lower-priority servos -> safe neutral (their conditioned midpoint if available, else global midpoint)
- This ensures the test actually matches the learned coupled limits.

Instant stop:
- type 'q' + Enter, or Ctrl+C.

Example:
  python3 verify_limits_fullpose.py --port /dev/ttyACM0 --baudrate 1000000 --limits limits.json --speed 300
"""

from __future__ import annotations

import argparse
import json
import time
import threading
from typing import Dict, List, Optional, Tuple

from scservo_sdk import PortHandler, PacketHandler

# ----------------------------
# Control table (common STS3215)
# ----------------------------
ADDR_TORQUE_ENABLE = 40
TORQUE_ON = 1
TORQUE_OFF = 0

ADDR_GOAL_POS_BLOCK = 0x2A  # pos/time/speed (6 bytes)
ADDR_PRESENT_POSITION = 0x38  # 2 bytes


# ----------------------------
# Conversions
# ----------------------------
def raw_to_deg(raw: int) -> float:
    return (int(raw) % 4096) * (360.0 / 4096.0)

def deg_to_raw(deg: float) -> int:
    deg = deg % 360.0
    return int(round(deg * (4096.0 / 360.0))) & 0xFFFF

def midpoint(a: float, b: float) -> float:
    return (a + b) / 2.0

def circular_diff_deg(a: float, b: float) -> float:
    # shortest circular distance
    return abs(((a - b + 180.0) % 360.0) - 180.0)

def lo(x: int) -> int: return x & 0xFF
def hi(x: int) -> int: return (x >> 8) & 0xFF


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

def write_goal_pos_block(pk: PacketHandler, ph: PortHandler, motor_id: int, pos_raw: int, time_ms: int, speed: int):
    data = [lo(pos_raw), hi(pos_raw), lo(time_ms), hi(time_ms), lo(speed), hi(speed)]
    if hasattr(pk, "writeTxRx"):
        pk.writeTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK, 6, data)
    elif hasattr(pk, "writeNByteTxRx"):
        pk.writeNByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK, 6, data)
    else:
        # fallback: sequential writes
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 0, pos_raw)
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 2, time_ms)
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 4, speed)

def move_deg(pk: PacketHandler, ph: PortHandler, mid: int, deg: float, speed: int, time_ms: int):
    write_goal_pos_block(pk, ph, mid, deg_to_raw(deg), time_ms, speed)

def wait_until(pk: PacketHandler, ph: PortHandler, mid: int, target_deg: float, tol_deg: float,
               timeout_s: float, kill) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if kill.stop:
            return False
        cur = raw_to_deg(read_u16(pk, ph, mid, ADDR_PRESENT_POSITION))
        if circular_diff_deg(cur, target_deg) <= tol_deg:
            return True
        time.sleep(0.02)
    return False


# ----------------------------
# Kill switch
# ----------------------------
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


# ----------------------------
# Limits DB helpers
# ----------------------------
def parse_key_to_bins(key: str) -> Dict[int, int]:
    if key == "ROOT":
        return {}
    out: Dict[int, int] = {}
    for part in key.split("|"):
        sid, b = part.split("=")
        out[int(sid)] = int(b)
    return out

def get_global_range(db: dict, mid: int) -> Optional[Tuple[float, float]]:
    gl = db["global_limits"].get(str(mid))
    if not gl:
        return None
    mn, mx = float(gl["min_deg"]), float(gl["max_deg"])
    if mx < mn:
        return None
    return mn, mx

def get_global_mid(db: dict, mid: int) -> float:
    r = get_global_range(db, mid)
    if not r:
        return 0.0
    return midpoint(r[0], r[1])

def get_cond_range(db: dict, mid: int, key: str) -> Optional[Tuple[float, float]]:
    cm = db.get("conditional_limits", {}).get(str(mid), {})
    rng = cm.get(key)
    if not rng:
        return None
    mn, mx = float(rng["min_deg"]), float(rng["max_deg"])
    if mx < mn:
        return None
    return mn, mx

def servo_key_from_pose(priority: List[int], target_id: int, pose_deg: Dict[int, float], bin_deg: float) -> str:
    parts = []
    for pid in priority:
        if pid == target_id:
            break
        # if not present, skip (should not happen in full pose)
        if pid not in pose_deg:
            continue
        b = int(round(pose_deg[pid] / bin_deg))
        parts.append(f"{pid}={b}")
    return "|".join(parts) if parts else "ROOT"

def build_full_pose(db: dict, priority: List[int], target_id: int, cond_key: str,
                    target_deg: float, bin_deg: float) -> Dict[int, float]:
    """
    Full pose for testing:
      - higher-priority servos -> bin centers from cond_key
      - target servo -> target_deg
      - lower-priority servos -> conditioned midpoint if available, else global midpoint
    """
    bins = parse_key_to_bins(cond_key)
    pose: Dict[int, float] = {}

    # Higher-priority setpoints
    for sid in priority:
        if sid == target_id:
            break
        if sid in bins:
            pose[sid] = bins[sid] * bin_deg
        else:
            pose[sid] = get_global_mid(db, sid)

    # Target
    pose[target_id] = target_deg

    # Lower-priority neutral positions
    past_target = False
    for sid in priority:
        if sid == target_id:
            past_target = True
            continue
        if not past_target:
            continue

        sid_key = servo_key_from_pose(priority, sid, pose, bin_deg)
        rng = get_cond_range(db, sid, sid_key)
        if rng:
            pose[sid] = midpoint(rng[0], rng[1])
        else:
            pose[sid] = get_global_mid(db, sid)

    return pose

def move_pose_priority(pk, ph, priority: List[int], pose: Dict[int, float], speed: int, time_ms: int):
    # Move in priority order for mechanical sanity
    for sid in priority:
        if sid in pose:
            move_deg(pk, ph, sid, pose[sid], speed, time_ms)


# ----------------------------
# Verification routine
# ----------------------------
def verify(args) -> int:
    with open(args.limits, "r", encoding="utf-8") as f:
        db = json.load(f)

    ids = [int(x) for x in db["ids"]]
    priority = [int(x) for x in db["priority"]]
    bin_deg = float(db["bin_deg"])

    ph, pk = open_bus(args.port, args.baudrate, protocol_end=0)
    kill = KillSwitch()
    kill.start()

    print("Verify FULL-POSE started.")
    print("Instant stop: type 'q' + Enter, or Ctrl+C.\n")

    try:
        set_torque(pk, ph, ids, on=True)

        # 1) Global edges (full pose: others at global mid, target at edge)
        if args.global_edges:
            print("Sweeping GLOBAL edges (full pose)...")
            for target_id in priority:
                if kill.stop:
                    break
                gr = get_global_range(db, target_id)
                if not gr:
                    print(f"  ID {target_id}: no valid global range recorded, skipping.")
                    continue

                mn, mx = gr
                mid = midpoint(mn, mx)
                print(f"  ID {target_id}: min={mn:.2f} max={mx:.2f}")

                for target_deg in (mid, mn, mx, mid):
                    if kill.stop:
                        break
                    pose = {sid: get_global_mid(db, sid) for sid in priority}
                    pose[target_id] = target_deg

                    move_pose_priority(pk, ph, priority, pose, args.speed, args.time_ms)
                    # wait for all joints in the pose
                    for sid in priority:
                        if kill.stop:
                            break
                        wait_until(pk, ph, sid, pose[sid], args.tol_deg, args.timeout_s, kill)

                    time.sleep(args.pause_s)

        # 2) Conditional edges (the important coupled test)
        if args.conditional_edges and not kill.stop:
            print("\nSweeping CONDITIONAL edges (full pose)...")
            cond = db.get("conditional_limits", {})

            for target_id in priority:
                if kill.stop:
                    break
                target_map = cond.get(str(target_id), {})
                if not target_map:
                    continue

                keys = list(target_map.keys())
                # optionally cap
                keys = keys[:args.max_keys_per_servo]

                print(f"  ID {target_id}: {len(keys)} conditional keys (capped={args.max_keys_per_servo})")

                for key in keys:
                    if kill.stop:
                        break
                    rng = get_cond_range(db, target_id, key)
                    if not rng:
                        continue
                    mn, mx = rng
                    mid = midpoint(mn, mx)

                    # Sweep midpoint first (safer), then edges
                    for target_deg in (mid, mn, mx, mid):
                        if kill.stop:
                            break

                        pose = build_full_pose(db, priority, target_id, key, target_deg, bin_deg)
                        move_pose_priority(pk, ph, priority, pose, args.edge_speed if target_deg in (mn, mx) else args.speed, args.time_ms)

                        # wait for all joints
                        for sid in priority:
                            if kill.stop:
                                break
                            wait_until(pk, ph, sid, pose[sid], args.tol_deg, args.timeout_s, kill)

                        time.sleep(args.pause_s)

        print("\nVerify finished." if not kill.stop else "\nVerify terminated by user.")
        return 0

    except KeyboardInterrupt:
        print("\nVerify terminated (Ctrl+C).")
        return 130

    finally:
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
    ap.add_argument("--limits", required=True)

    ap.add_argument("--speed", type=int, default=400, help="Normal sweep speed (u16 in goal block)")
    ap.add_argument("--edge-speed", type=int, default=250, help="Slower speed used at edges (min/max)")
    ap.add_argument("--time-ms", type=int, default=0, help="Time field in goal block (u16)")
    ap.add_argument("--pause-s", type=float, default=0.25)
    ap.add_argument("--tol-deg", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=5.0)

    ap.add_argument("--global-edges", action="store_true", help="Sweep global edges (full pose).")
    ap.add_argument("--conditional-edges", action="store_true", help="Sweep conditional edges (full pose).")
    ap.add_argument("--max-keys-per-servo", type=int, default=50)

    ap.add_argument("--torque-off-on-exit", action="store_true")
    args = ap.parse_args()

    if not args.global_edges and not args.conditional_edges:
        # sensible default
        args.global_edges = True
        args.conditional_edges = True

    return verify(args)

if __name__ == "__main__":
    raise SystemExit(main())
