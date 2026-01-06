#!/usr/bin/env python3
"""
Feetech STS3215 Runtime Controller (CMD/REPL) with hierarchical limits from limits.json.

Features:
- Load LimitsDB JSON produced by teach_limits.py
- Interactive command-line control (REPL)
- Priority enforcement:
    - Higher-priority servos keep requested angles
    - Lower-priority servos are auto-clamped to conditional/global limits based on higher ones
- Forbidden-state avoidance:
    - If a pose matches a forbidden binned combination, adjust lowest-priority servos toward neutral until safe
- Safe motion:
    - Moves servos in priority order
    - Optional wait-to-target

Requirements:
- scservo_sdk available in your Python env
- Your physical bus is already working

Usage:
  python3 servo_controller_cmd.py --port /dev/ttyACM0 --baudrate 1000000 --limits ./limits.json

REPL commands:
  help
  ids
  read                    (read current angles)
  show                    (show desired + final clamped pose)
  set <id> <deg>          (set target angle for one servo and move immediately)
  setmulti <id=deg ...>   (set multiple targets at once and move)
  move                    (apply current desired targets)
  neutral                 (set all servos to their neutral/midpoints and move)
  torque on|off
  speed <u16>             (set move speed)
  edgespeed <u16>         (set speed used for forbidden-resolution micro-steps)
  wait on|off             (toggle waiting for each move)
  q / quit

Notes:
- Angles are interpreted in the same 0..360 domain your teaching used.
- If your learned ranges cross 0° wrap-around, you should re-teach in a consistent region
  or we can extend the model to handle circular intervals.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional, Tuple

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
# Conversion
# ----------------------------
def raw_to_deg(raw: int) -> float:
    return (int(raw) % 4096) * (360.0 / 4096.0)

def deg_to_raw(deg: float) -> int:
    deg = deg % 360.0
    return int(round(deg * (4096.0 / 360.0))) & 0xFFFF

def lo(x: int) -> int:
    return x & 0xFF

def hi(x: int) -> int:
    return (x >> 8) & 0xFF

def midpoint(a: float, b: float) -> float:
    return (a + b) / 2.0

def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))

def circular_diff_deg(a: float, b: float) -> float:
    # shortest circular distance
    return abs(((a - b + 180.0) % 360.0) - 180.0)


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
        # fallback: sequential
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 0, pos_raw)
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 2, time_ms)
        pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POS_BLOCK + 4, speed)

def move_deg(pk: PacketHandler, ph: PortHandler, mid: int, deg: float, speed: int, time_ms: int):
    write_goal_pos_block(pk, ph, mid, deg_to_raw(deg), time_ms, speed)

def read_angles(pk: PacketHandler, ph: PortHandler, ids: List[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for mid in ids:
        out[mid] = raw_to_deg(read_u16(pk, ph, mid, ADDR_PRESENT_POSITION))
    return out

def wait_until(pk: PacketHandler, ph: PortHandler, mid: int, target_deg: float,
               tol_deg: float, timeout_s: float) -> bool:
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
def bin_angle(deg: float, bin_deg: float) -> int:
    return int(round(deg / bin_deg))

def parse_key_to_bins(key: str) -> Dict[int, int]:
    if key == "ROOT":
        return {}
    out: Dict[int, int] = {}
    for part in key.split("|"):
        sid, b = part.split("=")
        out[int(sid)] = int(b)
    return out

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

def forbidden_bins_set(db: dict) -> List[Dict[str, int]]:
    return db.get("forbidden", []) or []


# ----------------------------
# Pose solver (priority + limits + forbidden avoidance)
# ----------------------------
def solve_pose(
    db: dict,
    priority: List[int],
    desired: Dict[int, float],
    current: Dict[int, float],
    prefer_current_for_unspecified: bool = True,
) -> Dict[int, float]:
    """
    Build a final pose that respects:
      - global range for each servo
      - conditional range for each servo given higher-priority bins
    Strategy:
      - iterate in priority order
      - for each servo:
          base = desired if specified else (current if prefer_current else global mid)
          clamp to conditional range if exists else global range if exists else passthrough
    """
    bin_deg = float(db["bin_deg"])
    pose: Dict[int, float] = {}

    for mid in priority:
        base = desired.get(mid)
        if base is None:
            base = current[mid] if prefer_current_for_unspecified else get_global_mid(db, mid)

        # Determine applicable range
        key = servo_key_from_pose(priority, mid, pose, bin_deg)  # uses already-fixed higher joints
        cr = get_cond_range(db, mid, key)
        gr = get_global_range(db, mid)

        if cr:
            mn, mx = cr
            pose[mid] = clamp(base, mn, mx)
        elif gr:
            mn, mx = gr
            pose[mid] = clamp(base, mn, mx)
        else:
            pose[mid] = base % 360.0

    return pose

def is_forbidden_pose(db: dict, pose: Dict[int, float], priority: List[int]) -> bool:
    fb = forbidden_bins_set(db)
    if not fb:
        return False
    bin_deg = float(db["bin_deg"])
    state = {str(mid): bin_angle(pose[mid], bin_deg) for mid in priority}
    for entry in fb:
        # forbidden entry must match all IDs present
        if all(str(mid) in entry and entry[str(mid)] == state[str(mid)] for mid in priority):
            return True
    return False

def resolve_forbidden(
    db: dict,
    priority: List[int],
    desired: Dict[int, float],
    current: Dict[int, float],
    max_iters: int = 60,
    step_deg: float = 2.0,
) -> Dict[int, float]:
    """
    If a solved pose hits a forbidden binned combination, adjust *lowest-priority* joints only
    toward neutral (their global midpoint), re-solving each time so constraints remain valid.

    This is intentionally conservative and deterministic.
    """
    pose = solve_pose(db, priority, desired, current)
    if not is_forbidden_pose(db, pose, priority):
        return pose

    # Work from lowest priority upward
    low_to_high = list(reversed(priority))

    # Precompute neutral points
    neutral = {mid: get_global_mid(db, mid) for mid in priority}

    # Copy desired so we can nudge only low-priority commands
    nudged = dict(desired)

    for _ in range(max_iters):
        if not is_forbidden_pose(db, pose, priority):
            return pose

        changed = False
        for mid in low_to_high:
            # Do not modify high-priority if it is explicitly commanded and is among top-most constraints
            # But per your rule, lower numbers have higher priority => we are allowed to adjust lower-priority only.
            # So we always adjust from lowest upwards.
            cur_cmd = nudged.get(mid, pose[mid])
            tgt = neutral[mid]

            # Nudge toward neutral
            if circular_diff_deg(cur_cmd, tgt) < 1e-6:
                continue

            # Move small step toward tgt in linear domain (0..360) assuming your taught ranges do not wrap
            if cur_cmd < tgt:
                cur_cmd = min(cur_cmd + step_deg, tgt)
            else:
                cur_cmd = max(cur_cmd - step_deg, tgt)

            nudged[mid] = cur_cmd
            changed = True

            pose = solve_pose(db, priority, nudged, current)
            if not is_forbidden_pose(db, pose, priority):
                return pose

        if not changed:
            break

    # If still forbidden after attempts, return best effort pose
    return pose


# ----------------------------
# Controller (REPL)
# ----------------------------
def fmt_pose(priority: List[int], pose: Dict[int, float]) -> str:
    return "  ".join([f"{mid}:{pose[mid]:7.2f}°" for mid in priority])

def move_pose_priority(pk, ph, priority: List[int], pose: Dict[int, float], speed: int, time_ms: int):
    for mid in priority:
        move_deg(pk, ph, mid, pose[mid], speed, time_ms)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--limits", required=True, help="Path to limits.json from teach_limits.py")
    ap.add_argument("--tol-deg", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=4.0)
    ap.add_argument("--time-ms", type=int, default=0)
    ap.add_argument("--speed", type=int, default=450)
    ap.add_argument("--edge-speed", type=int, default=250, help="Used for forbidden-resolution micro-steps.")
    ap.add_argument("--no-wait", action="store_true", help="Do not wait for move completion.")
    args = ap.parse_args()

    with open(args.limits, "r", encoding="utf-8") as f:
        db = json.load(f)

    ids = [int(x) for x in db["ids"]]
    priority = [int(x) for x in db["priority"]]

    ph, pk = open_bus(args.port, args.baudrate, protocol_end=0)

    desired: Dict[int, float] = {}  # user-set targets
    speed = args.speed
    edge_speed = args.edge_speed
    do_wait = not args.no_wait

    try:
        set_torque(pk, ph, ids, on=True)

        def apply_move():
            nonlocal desired
            cur = read_angles(pk, ph, priority)

            # Solve with forbidden-resolution
            pose = resolve_forbidden(db, priority, desired, cur)

            # Move
            move_pose_priority(pk, ph, priority, pose, speed, args.time_ms)

            # Wait (optional)
            if do_wait:
                for mid in priority:
                    wait_until(pk, ph, mid, pose[mid], args.tol_deg, args.timeout_s)

            # If pose differs from desired for low-priority joints, keep desired as-is (it remains user's intent),
            # but show final pose.
            return cur, pose

        help_text = """
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
  edgespeed <u16>
  wait on|off
  clear                     clear desired targets (controller will hold current as neutral basis)
  quit / q
"""
        print(help_text.strip())

        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("q", "quit", "exit"):
                break

            if cmd == "help":
                print(help_text.strip())
                continue

            if cmd == "ids":
                print(f"IDs: {ids}")
                print(f"Priority: {priority}")
                print(f"Bin size: {db['bin_deg']} deg")
                continue

            if cmd == "read":
                cur = read_angles(pk, ph, priority)
                print("Current:", fmt_pose(priority, cur))
                continue

            if cmd == "show":
                cur = read_angles(pk, ph, priority)
                final_pose = resolve_forbidden(db, priority, desired, cur)
                print("Desired:", "  ".join([f"{mid}:{desired[mid]:7.2f}°" for mid in priority if mid in desired]) or "(none)")
                print("Final:  ", fmt_pose(priority, final_pose))
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

            if cmd == "edgespeed" and len(parts) == 2:
                edge_speed = int(parts[1])
                print(f"[OK] edge_speed={edge_speed}")
                continue

            if cmd == "neutral":
                # Set desired to neutral for all and move
                desired = {mid: get_global_mid(db, mid) for mid in priority}
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
        # torque on exit: leave ON by default; you can change here if you prefer
        close_bus(ph)


if __name__ == "__main__":
    main()
