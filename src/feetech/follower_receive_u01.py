#!/usr/bin/env python3
import argparse
import json
import socket
import sys
import time
import math
from typing import Dict, Tuple, Optional

import scservo_sdk as scs

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

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value < 0:
        return (1 << sign_bit) | (abs(value) & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)

def clamp_ticks(x: int, lo: int, hi: int) -> int:
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    return a if x < a else b if x > b else x

def soft_range(lo: int, hi: int, margin: float) -> tuple[int, int]:
    # Shrink inward while preserving direction (lo->hi). Works even if lo>hi.
    m = max(0.0, min(0.45, float(margin)))
    span = hi - lo  # signed
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

def map_range(x, in_min, in_max, out_min, out_max):
    """
    Linearly maps x from [in_min, in_max] to [out_min, out_max].
    """
    if in_max == in_min:
        raise ValueError("in_min and in_max must be different")

    return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)

def is_touching_danger_zone(a, b, c, r) -> bool:
    x1 = math.cos(a) * 11.6
    y1 = math.sin(a) * 11.6

    omega = math.pi - a - b
    x2 = math.cos(omega) * 10.5
    y2 = -math.sin(omega) * 10.5

    fi = omega + (c - math.pi)
    x3 = math.cos(fi) * 5.7
    y3 = math.sin(fi) * 5.7

    finalX = x1 + x2 + x3
    finalY = y1 + y2 + y3

    closestX = min(finalX, 6)
    closestY = min(finalY, 1)

    dx = finalX - closestX
    dy = finalY - closestY

    print(f"posX={finalX}, posY={finalY}")

    return (dx * dx + dy * dy) <= (r * r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6])

    ap.add_argument("--follower_calib", required=True)
    ap.add_argument("--gripper_range", type=int, nargs=2, default=None)
    ap.add_argument("--invert", type=int, nargs="*", default=[])
    ap.add_argument("--trim", type=str, default="")
    ap.add_argument("--soft_margin", type=float, default=0.0)

    ap.add_argument("--hz", type=float, default=120.0)
    ap.add_argument("--max_step", type=int, default=300)
    ap.add_argument("--sock_timeout", type=float, default=0.01)
    ap.add_argument("--print_gripper", action="store_true", help="Print gripper clamp diagnostics")

    args = ap.parse_args()

    ids = args.ids
    INVERT = {mid: (mid in set(args.invert)) for mid in ids}
    TRIM = {mid: 0 for mid in ids}
    if args.trim.strip():
        for part in args.trim.split(","):
            k, v = part.strip().split(":")
            TRIM[int(k)] = int(v)

    NAME_TO_ID = {"shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "gripper": 6}

    calib_ranges = load_calib_ranges(args.follower_calib)

    bus = Bus(args.port, args.baudrate, protocol=0)
    bus.connect()

    # EEPROM safety limits
    eeprom_limits: Dict[int, Tuple[int, int]] = {}
    for mid in ids:
        bus.write1(mid, CTRL_TABLE["Operating_Mode"][0], 0)
        bus.write1(mid, CTRL_TABLE["Acceleration"][0], 254)
        bus.write1(mid, CTRL_TABLE["Lock"][0], 1)
        bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 0)
        mn = bus.read2(mid, CTRL_TABLE["Min_Position_Limit"][0])
        mx = bus.read2(mid, CTRL_TABLE["Max_Position_Limit"][0])
        eeprom_limits[mid] = (mn, mx)

    # Mapping limits from calib (fallback to EEPROM)
    map_limits: Dict[int, Tuple[int, int]] = {}
    for mid in ids:
        map_limits[mid] = calib_ranges.get(mid, eeprom_limits[mid])

    gr_override = tuple(args.gripper_range) if args.gripper_range else None

    print("[follower] EEPROM safety limits:")
    for mid in ids:
        print(f"  ID {mid}: {eeprom_limits[mid]}")
    print("[follower] Mapping limits (from calib unless missing):")
    for mid in ids:
        print(f"  ID {mid}: {map_limits[mid]}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.settimeout(max(0.001, float(args.sock_timeout)))

    dt = 1.0 / max(1.0, args.hz)
    q_cmd: Dict[int, Optional[int]] = {mid: None for mid in ids}

    try:
        while True:
            try:
                data, _ = sock.recvfrom(8192)
                msg = json.loads(data.decode("utf-8"))
            except socket.timeout:
                time.sleep(dt)
                continue

            if msg.get("unit") != "u01":
                time.sleep(dt)
                continue

            u_field = msg.get("u", {})
            u_by_id: Dict[int, float] = {}

            for k, v in u_field.items():
                if isinstance(k, str) and k in NAME_TO_ID:
                    mid = NAME_TO_ID[k]
                elif isinstance(k, str) and k.isdigit():
                    mid = int(k)
                elif isinstance(k, int):
                    mid = k
                else:
                    continue
                if mid in ids:
                    try:
                        u_by_id[mid] = clamp01(float(v))
                    except Exception:
                        pass

            # torque on
            for mid in ids:
                bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 1)


            # for mid, u in u_by_id.items():
            #     if mid == 6:
            #         u = 0.5

            shoulder_lift_rad = map_range(u_by_id[2], 0, 0.25, 125, 90)
            elbow_flex_rad = map_range(u_by_id[3], 1, 0.66, 19, 90)
            wrist_flex_rad = map_range(u_by_id[4], 0, 0.47, 256, 180)

            touchung: bool = is_touching_danger_zone(math.radians(shoulder_lift_rad), math.radians(elbow_flex_rad), math.radians(wrist_flex_rad), 6.5)

            print(f"shoulder={shoulder_lift_rad}, elbow={elbow_flex_rad}, wrist={wrist_flex_rad}, touching={touchung}")

            goals: Dict[int, int] = {}
            for mid, u in u_by_id.items():
                lo, hi = map_limits[mid]
                if mid == 6 and gr_override is not None:
                    lo, hi = gr_override

                lo_s, hi_s = soft_range(lo, hi, args.soft_margin)

                pre = int(round(lo_s + (clamp01(u) if not INVERT[mid] else (1.0 - clamp01(u))) * (hi_s - lo_s)))
                mapped = u_to_ticks(u, lo_s, hi_s, INVERT[mid], TRIM[mid])

                # EEPROM safety clamp
                e_lo, e_hi = eeprom_limits[mid]
                after_eeprom = clamp_ticks(mapped, e_lo, e_hi)

                # rate limit
                if q_cmd[mid] is None:
                    goal = after_eeprom
                else:
                    delta = after_eeprom - int(q_cmd[mid])
                    step = delta
                    if step > args.max_step:
                        step = args.max_step
                    elif step < -args.max_step:
                        step = -args.max_step
                    goal = int(q_cmd[mid]) + step
                q_cmd[mid] = goal
                goals[mid] = goal

                if args.print_gripper and mid == 6:
                    print(
                        f"[grip] u={u:.3f} invert={INVERT[6]} trim={TRIM[6]} "
                        f"map=({lo},{hi}) soft=({lo_s},{hi_s}) "
                        f"pre={pre} mapped={mapped} eeprom=({e_lo},{e_hi}) after_eeprom={after_eeprom} goal={goal}"
                    )

            # write
            for mid, pos in goals.items():
                raw = encode_sign_magnitude(pos, SIGN_BITS["Goal_Position"])
                ok, err = bus.write2(mid, CTRL_TABLE["Goal_Position"][0], raw)
                if not ok:
                    print(f"[WARN] write goal failed ID {mid}: {err}", file=sys.stderr)

            time.sleep(dt)

    finally:
        bus.disconnect()

if __name__ == "__main__":
    main()