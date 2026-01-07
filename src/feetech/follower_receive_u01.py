#!/usr/bin/env python3
"""
Jetson-side UDP follower for Feetech STS3215 (protocol 0), consuming u∈[0,1] packets.

NEW:
- Can use FOLLOWER calibration JSON (your produced file) for mapping instead of servo EEPROM limits.
  This is the correct approach when EEPROM limits aren't aligned with your mechanical "safe/useful" range,
  especially for grippers.

Packet format expected:
  {"t": <float>, "unit": "u01", "u": {"shoulder_lift":0..1, "elbow_flex":0..1, "wrist_flex":0..1, "gripper":0..1}}
Keys may be names OR numeric IDs ("2","3","4","6").

Recommended start:
  python3 follower_receive_u01_with_calib.py --port /dev/ttyACM0 --udp_port 5005 --ids 2 3 4 6 \
    --follower_calib hand_calibration.json --hz 120 --max_step 300 --soft_margin 0.05 --auto_trim --invert 6 --print
"""

import argparse
import json
import socket
import sys
import time
from typing import Dict, Tuple, Optional

import scservo_sdk as scs


CTRL_TABLE = {
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
}

SIGN_BITS = {
    "Homing_Offset": 11,
    "Goal_Position": 15,
    "Present_Position": 15,
}


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value < 0:
        return (1 << sign_bit) | (abs(value) & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value & (1 << sign_bit):
        return -(value & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def compute_soft_limits(lo: int, hi: int, margin: float) -> Tuple[int, int]:
    lo, hi = (lo, hi) if lo <= hi else (hi, lo)
    span = hi - lo
    m = max(0.0, min(0.45, float(margin)))
    smn = int(round(lo + m * span))
    smx = int(round(hi - m * span))
    if smx <= smn:
        return lo, hi
    return smn, smx


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class Bus:
    def __init__(self, port: str, baudrate: int = 1_000_000, protocol: int = 0):
        self.port = port
        self.baudrate = baudrate
        self.port_handler = scs.PortHandler(port)
        self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(self.port_handler, scs.PortHandler)
        self.packet_handler = scs.PacketHandler(protocol)

    def connect(self):
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port: {self.port}")
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate {self.baudrate} on {self.port}")

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

    def read_present_ticks(self, mid: int) -> int:
        addr, _ = CTRL_TABLE["Present_Position"]
        raw = self.read2(mid, addr)
        return decode_sign_magnitude(raw, SIGN_BITS["Present_Position"])


def load_follower_calib_ranges(path: str) -> Dict[int, Tuple[int, int]]:
    """
    Your calibration JSON looks like:
      { "shoulder_lift": {"id":2,"range_min":...,"range_max":...}, ... }
    Returns: {2: (min,max), 3: (...), ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    out: Dict[int, Tuple[int, int]] = {}
    for _, d in j.items():
        mid = int(d["id"])
        out[mid] = (int(d["range_min"]), int(d["range_max"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6])

    ap.add_argument("--hz", type=float, default=120.0)
    ap.add_argument("--max_step", type=int, default=300)
    ap.add_argument("--sock_timeout", type=float, default=0.01)
    ap.add_argument("--timeout_s", type=float, default=0.8)

    ap.add_argument("--soft_margin", type=float, default=0.05)

    ap.add_argument("--invert", type=int, nargs="*", default=[], help="IDs to invert (u -> 1-u)")
    ap.add_argument("--trim", type=str, default="", help='Per-joint trim ticks, e.g. "2:120,3:-80,4:0,6:50"')
    ap.add_argument("--auto_trim", action="store_true", help="Compute trim at start from current follower pose")

    ap.add_argument("--no_gripper", action="store_true")
    ap.add_argument("--gripper_range", type=int, nargs=2, default=None, metavar=("GMIN", "GMAX"),
                    help="Override follower mapping range for gripper only (ticks)")

    # NEW: follower calibration JSON
    ap.add_argument("--follower_calib", type=str, default=None,
                    help="Follower calibration JSON produced by your calibrator; used for mapping ranges.")

    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    ids = [i for i in args.ids if not (args.no_gripper and i == 6)]
    dt = 1.0 / max(1.0, args.hz)

    NAME_TO_ID = {"shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "gripper": 6}

    INVERT_U = {mid: (mid in set(args.invert)) for mid in ids}

    TRIM_TICKS: Dict[int, int] = {mid: 0 for mid in ids}
    if args.trim.strip():
        for part in args.trim.split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":")
            TRIM_TICKS[int(k.strip())] = int(v.strip())

    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.settimeout(max(0.001, float(args.sock_timeout)))

    # Bus
    bus = Bus(args.port, args.baudrate, protocol=0)
    bus.connect()

    # Read EEPROM limits (always useful as absolute safety clamp)
    eeprom_limits: Dict[int, Tuple[int, int]] = {}
    for mid in ids:
        bus.write1(mid, CTRL_TABLE["Operating_Mode"][0], 0)
        bus.write1(mid, CTRL_TABLE["Acceleration"][0], 254)
        bus.write1(mid, CTRL_TABLE["Lock"][0], 1)
        bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 0)

        mn = bus.read2(mid, CTRL_TABLE["Min_Position_Limit"][0])
        mx = bus.read2(mid, CTRL_TABLE["Max_Position_Limit"][0])
        eeprom_limits[mid] = (mn, mx)

    # Mapping limits (source = follower_calib OR EEPROM)
    map_limits: Dict[int, Tuple[int, int]] = {}
    if args.follower_calib:
        calib_ranges = load_follower_calib_ranges(args.follower_calib)
        for mid in ids:
            if mid in calib_ranges:
                map_limits[mid] = calib_ranges[mid]
            else:
                map_limits[mid] = eeprom_limits[mid]
    else:
        map_limits = dict(eeprom_limits)

    # Optional gripper override
    GRIPPER_RANGE_OVERRIDE: Optional[Tuple[int, int]] = None
    if args.gripper_range is not None and 6 in ids:
        GRIPPER_RANGE_OVERRIDE = (int(args.gripper_range[0]), int(args.gripper_range[1]))

    # Compute soft limits on mapping limits
    soft_map_limits: Dict[int, Tuple[int, int]] = {}
    for mid in ids:
        lo, hi = map_limits[mid]
        soft_map_limits[mid] = compute_soft_limits(lo, hi, args.soft_margin)

    print(f"[follower] Listening UDP :{args.udp_port}, controlling IDs={ids}")
    print("[follower] EEPROM limits (absolute safety):")
    for mid in ids:
        mn, mx = eeprom_limits[mid]
        print(f"  ID {mid}: min={mn} max={mx}")
    print("[follower] Mapping limits (used for u->ticks):")
    src = "follower_calib" if args.follower_calib else "EEPROM"
    print(f"  source={src}")
    for mid in ids:
        lo, hi = map_limits[mid]
        print(f"  ID {mid}: map_min={lo} map_max={hi}")
    print("[follower] Soft mapping limits used:")
    for mid in ids:
        lo, hi = soft_map_limits[mid]
        print(f"  ID {mid}: soft_min={lo} soft_max={hi}")
    if GRIPPER_RANGE_OVERRIDE is not None:
        print(f"[follower] Gripper override range (ticks): {GRIPPER_RANGE_OVERRIDE}")
    print(f"[follower] Invert map: {INVERT_U}")
    print(f"[follower] Trim ticks (initial): {TRIM_TICKS}")
    if args.auto_trim:
        print("[follower] Auto-trim enabled: will compute trims after first packet.")

    got_first = False
    last_packet_t = time.time()
    q_cmd: Dict[int, Optional[int]] = {mid: None for mid in ids}
    auto_trim_done = False
    n = 0

    def torque_on(mid: int) -> bool:
        ok, err = bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 1)
        if not ok and mid == 6:
            # do not crash; gripper may overload
            return False
        return ok

    try:
        while True:
            msg = None
            try:
                data, _ = sock.recvfrom(8192)
                msg = json.loads(data.decode("utf-8"))
                got_first = True
                last_packet_t = time.time()
            except socket.timeout:
                pass
            except Exception as e:
                if n % 50 == 0:
                    print(f"[follower] WARN: bad UDP packet: {e}", file=sys.stderr)

            if got_first and (time.time() - last_packet_t) > args.timeout_s:
                print("[follower] Timeout -> torque off + unlock")
                for mid in ids:
                    bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 0)
                    bus.write1(mid, CTRL_TABLE["Lock"][0], 0)
                break

            if msg is None or msg.get("unit") != "u01":
                time.sleep(dt)
                continue

            u_field = msg.get("u", {})
            u_by_id: Dict[int, float] = {}

            for k, v in u_field.items():
                mid = None
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

            if not u_by_id:
                time.sleep(dt)
                continue

            # auto trim once: align current follower pose to first packet
            if args.auto_trim and not auto_trim_done:
                # compute target ticks for first packet, then set trim so current==target
                for mid, u in u_by_id.items():
                    if INVERT_U.get(mid, False):
                        u = 1.0 - u

                    if mid == 6 and GRIPPER_RANGE_OVERRIDE is not None:
                        lo, hi = GRIPPER_RANGE_OVERRIDE
                        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
                    else:
                        lo, hi = soft_map_limits[mid]
                        lo, hi = (lo, hi) if lo <= hi else (hi, lo)

                    tgt = int(round(lo + u * (hi - lo)))
                    tgt = clamp_int(tgt, lo, hi)

                    try:
                        cur = bus.read_present_ticks(mid)
                        TRIM_TICKS[mid] += int(cur - tgt)
                    except Exception:
                        pass

                auto_trim_done = True
                print(f"[follower] Auto-trim computed. Trim ticks now: {TRIM_TICKS}")

            # torque on
            for mid in ids:
                _ = torque_on(mid)

            # map u -> target
            targets: Dict[int, int] = {}
            for mid, u in u_by_id.items():
                if INVERT_U.get(mid, False):
                    u = 1.0 - u

                if mid == 6 and GRIPPER_RANGE_OVERRIDE is not None:
                    lo, hi = GRIPPER_RANGE_OVERRIDE
                    lo, hi = (lo, hi) if lo <= hi else (hi, lo)
                else:
                    lo, hi = soft_map_limits[mid]
                    lo, hi = (lo, hi) if lo <= hi else (hi, lo)

                tgt = int(round(lo + u * (hi - lo)))
                tgt = clamp_int(tgt, lo, hi)

                tgt += int(TRIM_TICKS.get(mid, 0))
                tgt = clamp_int(tgt, lo, hi)

                # absolute safety clamp to EEPROM limits (important!)
                s_lo, s_hi = eeprom_limits[mid]
                s_lo, s_hi = (s_lo, s_hi) if s_lo <= s_hi else (s_hi, s_lo)
                tgt = clamp_int(tgt, s_lo, s_hi)

                targets[mid] = tgt

            # rate limit and write
            goals: Dict[int, int] = {}
            for mid, tgt in targets.items():
                if q_cmd[mid] is None:
                    goals[mid] = tgt
                else:
                    delta = tgt - int(q_cmd[mid])
                    step = clamp_int(delta, -args.max_step, args.max_step)
                    goals[mid] = int(q_cmd[mid]) + step
                q_cmd[mid] = goals[mid]

            for mid, pos in goals.items():
                raw = encode_sign_magnitude(pos, SIGN_BITS["Goal_Position"])
                ok, err = bus.write2(mid, CTRL_TABLE["Goal_Position"][0], raw)
                if not ok and n % 50 == 0:
                    print(f"[follower] WARN: write goal failed ID {mid}: {err}", file=sys.stderr)

            if args.print and n % 30 == 0:
                dbg = {mid: {"u": u_by_id.get(mid), "goal": goals.get(mid)} for mid in sorted(u_by_id)}
                print(dbg)

            n += 1
            time.sleep(dt)

    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
