#!/usr/bin/env python3
"""
Runtime controller CLI for Feetech STS3215 using limits.json from teach_limits_realtime.py.

- Priority enforcement: higher-priority servos keep their target; lower-priority are clamped.
- Conditional limits: used for non-global-only servos based on higher servos' binned angles.
- Global-only IDs: clamp only using global min/max (e.g., gripper servo 6).

Usage:
  python3 servo_controller_cmd.py --port /dev/ttyACM0 --baudrate 1000000 --limits limits.json --global-only-ids 6

Commands:
  help
  ids
  read
  show
  set <id> <deg>
  setmulti <id=deg ...>
  move
  neutral
  torque on|off
  speed <u16>
  wait on|off
  clear
  q
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional, Tuple, Set

from scservo_sdk import PortHandler, PacketHandler

# ----------------------------
# STS3215 common addresses
# ----------------------------
ADDR_TORQUE_ENABLE = 40
TORQUE_ON = 1
TORQUE_OFF = 0

ADDR_GOAL_POS_BLOCK = 0x2A  # pos/time/speed (6 bytes)
ADDR_PRESENT_POSITION = 0x38  # 2 bytes


# ----------------------------
# Conversion helpers
# ----------------------------
def raw_to_deg(raw: int) -> float:
    return (int(raw) % 4096) * (360.0 / 4096.0)

def deg_to_raw(deg: float) -> int:
    deg = deg % 360.0
    return int(round(deg * (4096.0 / 360.0))) & 0xFFFF

def lo(x: int) -> int: return x & 0xFF
def hi(x: int) -> int: return (x >> 8) & 0xFF

def midpoint(a: float, b: float) -> float:
    return (a + b) / 2.0

def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))

def circular_diff_deg(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)

def bin_angle(deg: float, bin_deg: float) -> int:
    return int(round(deg / bin_deg))


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
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 0, pos_raw)
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 2, time_ms)
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 4, speed)

def move_deg(pk: PacketHandler, ph: PortHandler, mid: int, deg: float, speed: int, time_ms: int):
    write_goal_pos_block(pk, ph, mid, deg_to_raw(deg), time_ms, speed)

def read_angles(pk: PacketHandler, ph: PortHandler, ids: List[int]) -> Dict[int, float]:
    return {mid: raw_to_deg(read_u16(pk, ph, mid, ADDR_PRESENT_POSITION)) for mid in ids}

def wait_until(pk: PacketHandler, ph: PortHandler, mid: int, target_deg: float, tol_deg: float, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        cur = raw_to_deg(read_u16(pk, ph, mid, ADDR_PRESENT_POSITION))
        if circular_diff_deg(cur, target_deg) <= tol_deg:
            return True
        time.sleep(0.02)
    return False


# ----------------------------
# Limits helpers
# ----------------------------
def servo_key_from_pose(priority: List[int], target_id: int, pose_deg: Dict[int, float], bin_deg: float) -> str:
    parts = []
    for pid in priority:
        if pid == target_id:
            break
        if pid not in pose_deg:
            continue
        parts.append(f"{pid}={bin_angle(pose_deg[pid], bin_deg)}")
    return "|".join(parts) if parts else "ROOT"

def get_global_range(db: dict, mid: int) -> Optional[Tuple[float, float]]:
    gl = db.get("global_limits", {}).get(str(mid))
    if not gl:
        return None
    mn, mx = float(gl["min_deg"]), float(gl["max_deg"])
    if mx < mn:
        return None
    return mn, mx

def get_global_mid(db: dict, mid: int) -> float:
    r = get_global_range(db, mid)
    return midpoint(r[0], r[1]) if r else 0.0

def get_cond_range(db: dict, mid: int, key: str) -> Optional[Tuple[float, float]]:
    cm = db.get("conditional_limits", {}).get(str(mid), {})
    rng = cm.get(key)
    if not rng:
        return None
    mn, mx = float(rng["min_deg"]), float(rng["max_deg"])
    if mx < mn:
        return None
    return mn, mx


# ----------------------------
# Pose solver (priority + limits)
# ----------------------------
def solve_pose(
    db: dict,
    priority: List[int],
    desired: Dict[int, float],
    current: Dict[int, float],
    global_only_ids: Set[int],
    prefer_current_for_unspecified: bool = True,
) -> Dict[int, float]:
    """
    Build a final pose:
      - iterate in priority order
      - for each servo, base = desired if specified else current (or global mid)
      - clamp:
          - if global-only -> use global range only
          - else prefer conditional for current higher-pose key, fallback to global
    """
    bin_deg = float(db["bin_deg"])
    pose: Dict[int, float] = {}

    for mid in priority:
        base = desired.get(mid)
        if base is None:
            base = current[mid] if prefer_current_for_unspecified else get_global_mid(db, mid)
        base = base % 360.0

        gr = get_global_range(db, mid)

        if mid in global_only_ids:
            # global-only: ignore conditional
            if gr:
                pose[mid] = clamp(base, gr[0], gr[1])
            else:
                pose[mid] = base
            continue

        key = servo_key_from_pose(priority, mid, pose, bin_deg)
        cr = get_cond_range(db, mid, key)

        if cr:
            pose[mid] = clamp(base, cr[0], cr[1])
        elif gr:
            pose[mid] = clamp(base, gr[0], gr[1])
        else:
            pose[mid] = base

    return pose


# ----------------------------
# Controller REPL
# ----------------------------
def fmt_pose(priority: List[int], pose: Dict[int, float]) -> str:
    return "  ".join([f"{mid}:{pose[mid]:7.2f}°" for mid in priority])

def move_pose_priority(pk, ph, priority: List[int], pose: Dict[int, float], speed: int, time_ms: int):
    for mid in priority:
        move_deg(pk, ph, mid, pose[mid], speed, time_ms)

HELP = """
Commands:
  help
  ids
  read
  show
  set <id> <deg>
  setmulti <id=deg ...>     e.g. setmulti 2=30 3=120 6=200
  move                      apply desired targets with limits
  neutral                   move all to neutral (global midpoints)
  torque on|off
  speed <u16>
  wait on|off
  clear                     clear desired targets
  q / quit
""".strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--limits", required=True)
    ap.add_argument("--global-only-ids", type=int, nargs="*", default=[],
                    help="IDs that should clamp only to global min/max (e.g., gripper). Example: --global-only-ids 6")
    ap.add_argument("--speed", type=int, default=450)
    ap.add_argument("--time-ms", type=int, default=0)
    ap.add_argument("--tol-deg", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=4.0)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    with open(args.limits, "r", encoding="utf-8") as f:
        db = json.load(f)

    ids = [int(x) for x in db["ids"]]
    priority = [int(x) for x in db["priority"]]
    global_only = set(int(x) for x in (args.global_only_ids or db.get("global_only_ids", []) or []))

    for gid in global_only:
        if gid not in ids:
            raise SystemExit(f"[ERROR] global-only id {gid} not in limits IDs {ids}")

    ph, pk = open_bus(args.port, args.baudrate, protocol_end=0)

    desired: Dict[int, float] = {}
    speed = args.speed
    do_wait = not args.no_wait

    try:
        set_torque(pk, ph, ids, on=True)
        print(HELP)

        def apply_move():
            cur = read_angles(pk, ph, priority)
            pose = solve_pose(db, priority, desired, cur, global_only_ids=global_only)

            move_pose_priority(pk, ph, priority, pose, speed, args.time_ms)

            if do_wait:
                for mid in priority:
                    wait_until(pk, ph, mid, pose[mid], args.tol_deg, args.timeout_s)
            return cur, pose

        while True:
            line = input("> ").strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("q", "quit", "exit"):
                break

            if cmd == "help":
                print(HELP)
                continue

            if cmd == "ids":
                print(f"IDs: {ids}")
                print(f"Priority: {priority}")
                print(f"Global-only: {sorted(global_only)}")
                print(f"Bin size: {db['bin_deg']} deg")
                continue

            if cmd == "read":
                cur = read_angles(pk, ph, priority)
                print("Current:", fmt_pose(priority, cur))
                continue

            if cmd == "show":
                cur = read_angles(pk, ph, priority)
                pose = solve_pose(db, priority, desired, cur, global_only_ids=global_only)
                print("Desired:", "  ".join([f"{mid}:{desired[mid]:.2f}°" for mid in priority if mid in desired]) or "(none)")
                print("Final:  ", fmt_pose(priority, pose))
                continue

            if cmd == "clear":
                desired.clear()
                print("[OK] Desired targets cleared.")
                continue

            if cmd == "torque" and len(parts) == 2:
                v = parts[1].lower()
                if v == "on":
                    set_torque(pk, ph, ids, on=True)
                    print("[OK] Torque ON")
                elif v == "off":
                    set_torque(pk, ph, ids, on=False)
                    print("[OK] Torque OFF")
                else:
                    print("[ERROR] torque on|off")
                continue

            if cmd == "wait" and len(parts) == 2:
                v = parts[1].lower()
                if v == "on":
                    do_wait = True
                    print("[OK] wait ON")
                elif v == "off":
                    do_wait = False
                    print("[OK] wait OFF")
                else:
                    print("[ERROR] wait on|off")
                continue

            if cmd == "speed" and len(parts) == 2:
                speed = int(parts[1])
                print(f"[OK] speed={speed}")
                continue

            if cmd == "neutral":
                desired.clear()
                for mid in priority:
                    desired[mid] = get_global_mid(db, mid)
                cur, pose = apply_move()
                print("Current:", fmt_pose(priority, cur))
                print("Final:  ", fmt_pose(priority, pose))
                continue

            if cmd == "set" and len(parts) == 3:
                mid = int(parts[1])
                deg = float(parts[2])
                if mid not in ids:
                    print(f"[ERROR] Unknown id {mid}")
                    continue
                desired[mid] = deg % 360.0
                cur, pose = apply_move()
                print("Current:", fmt_pose(priority, cur))
                print("Final:  ", fmt_pose(priority, pose))
                continue

            if cmd == "setmulti" and len(parts) >= 2:
                ok = True
                for tok in parts[1:]:
                    if "=" not in tok:
                        ok = False
                        break
                    sid, sdeg = tok.split("=", 1)
                    mid = int(sid)
                    deg = float(sdeg)
                    if mid not in ids:
                        print(f"[ERROR] Unknown id {mid}")
                        ok = False
                        break
                    desired[mid] = deg % 360.0
                if not ok:
                    print("[ERROR] Usage: setmulti 2=30 3=120 6=200")
                    continue
                cur, pose = apply_move()
                print("Current:", fmt_pose(priority, cur))
                print("Final:  ", fmt_pose(priority, pose))
                continue

            if cmd == "move":
                cur, pose = apply_move()
                print("Current:", fmt_pose(priority, cur))
                print("Final:  ", fmt_pose(priority, pose))
                continue

            print("[INFO] Unknown command. Type 'help'.")

    finally:
        close_bus(ph)


if __name__ == "__main__":
    raise SystemExit(main())