#!/usr/bin/env python3
"""
follower_receive_u01_tuned.py

Jetson-side UDP follower for Feetech STS3215 (protocol 0), tuned for "u01" leader packets.

Solves:
1) "Centers are offset" -> per-joint trim in ticks + optional auto-trim at startup
2) "Painful latency"     -> higher loop rate, smaller socket timeout, higher max_step
3) "Gripper out of range"-> per-joint inversion + optional gripper-range override

Packet format expected from leader:
  {"t": <float>, "unit": "u01", "u": {"shoulder_lift":0..1, "elbow_flex":0..1, "wrist_flex":0..1, "gripper":0..1}}

Key features:
- Accepts u keys by NAME or numeric ID ("2","3","4","6")
- Reads Min/Max limits from servo EEPROM and clamps safely
- Optional "soft margin" inside EEPROM limits (recommended)
- Per-joint inversion (u -> 1-u) for direction mismatch
- Per-joint trim in ticks (static), plus optional auto-trim computed at startup
- Robust torque enable (won't crash on gripper overload; will skip gripper if needed)
- Rate limiting (max_step ticks per update) to avoid violent jumps

Usage (recommended):
  python3 follower_receive_u01_tuned.py --port /dev/ttyACM0 --udp_port 5005 --ids 2 3 4 6 --hz 120 --max_step 300 --soft_margin 0.05 --auto_trim

If gripper still misbehaves, try:
  --no_gripper
or set a custom safe gripper range:
  --gripper_range 1050 1850
and/or invert gripper:
  --invert 6
"""

import argparse
import json
import socket
import sys
import time
from typing import Dict, Tuple, Optional

import scservo_sdk as scs


# ---------------------------- Control table (STS3215 protocol 0) ----------------------------

CTRL_TABLE = {
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Torque_Limit": (16, 2),       # EEPROM
    "Protection_Current": (28, 2),     # EEPROM
    "Overload_Torque": (36, 1),        # EEPROM
    "Operating_Mode": (33, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
}

SIGN_BITS = {
    "Goal_Position": 15,
    "Present_Position": 15,
}


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value < 0:
        return (1 << sign_bit) | (abs(value) & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value & (1 << sign_bit):
        return - (value & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    # Same workaround used in LeRobot for some scservo_sdk builds
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class Bus:
    """
    Minimal robust wrapper around scservo_sdk for protocol 0.
    Handles SDK variants that return 2-tuple vs 3-tuple.
    """
    def __init__(self, port: str, baudrate: int = 1_000_000, protocol: int = 0):
        self.port = port
        self.baudrate = baudrate
        self.protocol = protocol

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
        # writeXByteTxRx sometimes returns (comm, err) or (value, comm, err)
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

    def read_present_pos_ticks(self, mid: int) -> int:
        addr, _ = CTRL_TABLE["Present_Position"]
        raw = self.read2(mid, addr)
        return decode_sign_magnitude(raw, SIGN_BITS["Present_Position"])


def compute_soft_limits(hard: Tuple[int, int], margin: float) -> Tuple[int, int]:
    mn, mx = hard
    lo, hi = (mn, mx) if mn <= mx else (mx, mn)
    span = hi - lo
    m = max(0.0, min(0.45, float(margin)))
    smn = int(round(lo + m * span))
    smx = int(round(hi - m * span))
    if smx <= smn:
        smn, smx = lo, hi
    return smn, smx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6])

    # Responsiveness
    ap.add_argument("--hz", type=float, default=120.0, help="Control loop rate (higher = less perceived lag)")
    ap.add_argument("--max_step", type=int, default=300, help="Max ticks per update (higher = faster response)")
    ap.add_argument("--sock_timeout", type=float, default=0.01, help="UDP socket timeout in seconds")

    # Safety
    ap.add_argument("--timeout_s", type=float, default=0.8, help="Stop and torque-off if no packets after first receive")
    ap.add_argument("--soft_margin", type=float, default=0.05, help="Use only inner portion of EEPROM limits")

    # Mapping tweaks
    ap.add_argument("--invert", type=int, nargs="*", default=[], help="IDs to invert (u -> 1-u)")
    ap.add_argument("--trim", type=str, default="", help='Per-joint trim ticks, e.g. "2:120,3:-80,4:0,6:50"')
    ap.add_argument("--auto_trim", action="store_true", help="Compute trim at start from current follower pose")
    ap.add_argument("--no_gripper", action="store_true", help="Ignore gripper ID 6 entirely")
    ap.add_argument("--gripper_range", type=int, nargs=2, default=None, metavar=("GMIN", "GMAX"),
                    help="Override gripper mapping range in ticks on the follower")

    # Gripper safety settings
    ap.add_argument("--gripper_safe", action="store_true", help="Apply conservative gripper current/torque settings")

    # Debug
    ap.add_argument("--print", action="store_true", help="Print incoming u and goals periodically")

    args = ap.parse_args()

    ids = [i for i in args.ids if not (args.no_gripper and i == 6)]
    dt = 1.0 / max(1.0, args.hz)

    # Leader names commonly used by your publisher
    NAME_TO_ID = {"shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "gripper": 6}

    # Inversion map
    invert_ids = set(args.invert)
    INVERT_U: Dict[int, bool] = {mid: (mid in invert_ids) for mid in ids}

    # Trim map (ticks)
    TRIM_TICKS: Dict[int, int] = {mid: 0 for mid in ids}
    if args.trim.strip():
        # format "2:120,3:-80,4:0,6:50"
        for part in args.trim.split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":")
            TRIM_TICKS[int(k.strip())] = int(v.strip())

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.settimeout(max(0.001, float(args.sock_timeout)))

    # Servo bus
    bus = Bus(args.port, args.baudrate, protocol=0)
    bus.connect()

    # Configure + read EEPROM limits
    hard_limits: Dict[int, Tuple[int, int]] = {}
    soft_limits: Dict[int, Tuple[int, int]] = {}

    for mid in ids:
        # POSITION mode, lock
        bus.write1(mid, CTRL_TABLE["Operating_Mode"][0], 0)
        bus.write1(mid, CTRL_TABLE["Acceleration"][0], 254)
        bus.write1(mid, CTRL_TABLE["Lock"][0], 1)
        bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 0)

        mn = bus.read2(mid, CTRL_TABLE["Min_Position_Limit"][0])
        mx = bus.read2(mid, CTRL_TABLE["Max_Position_Limit"][0])
        hard_limits[mid] = (mn, mx)
        soft_limits[mid] = compute_soft_limits((mn, mx), args.soft_margin)

    # Optional gripper safety parameters (like LeRobot intent)
    if args.gripper_safe and 6 in ids:
        bus.write2(6, CTRL_TABLE["Max_Torque_Limit"][0], 500)
        bus.write2(6, CTRL_TABLE["Protection_Current"][0], 250)
        bus.write1(6, CTRL_TABLE["Overload_Torque"][0], 25)

    # Apply gripper range override on follower side if requested
    GRIPPER_RANGE_OVERRIDE: Optional[Tuple[int, int]] = None
    if args.gripper_range is not None and 6 in ids:
        gmin, gmax = args.gripper_range
        GRIPPER_RANGE_OVERRIDE = (int(gmin), int(gmax))

    print(f"[follower] Listening UDP :{args.udp_port}, controlling IDs={ids}")
    print("[follower] Hard limits (from servo EEPROM):")
    for mid in ids:
        mn, mx = hard_limits[mid]
        print(f"  ID {mid}: min={mn} max={mx}")
    print("[follower] Soft limits used:")
    for mid in ids:
        smn, smx = soft_limits[mid]
        print(f"  ID {mid}: soft_min={smn} soft_max={smx}")
    if GRIPPER_RANGE_OVERRIDE is not None:
        print(f"[follower] Gripper range override: {GRIPPER_RANGE_OVERRIDE}")
    print(f"[follower] Invert map: {INVERT_U}")
    print(f"[follower] Trim ticks: {TRIM_TICKS}")
    if args.auto_trim:
        print("[follower] Auto-trim: waiting for first u01 packet, then computing trims...")

    got_first = False
    last_packet_t = time.time()

    # Last commanded ticks for rate limiting
    q_cmd: Dict[int, Optional[int]] = {mid: None for mid in ids}

    # Gripper torque may fail under overload; do not crash, just skip
    gripper_ok = True

    # Helper: enable torque best-effort
    def torque_on(mid: int) -> bool:
        ok, err = bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 1)
        if not ok and mid == 6:
            return False
        return ok

    n = 0

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

            # Timeout only after first packet
            if got_first and (time.time() - last_packet_t) > args.timeout_s:
                print("[follower] Timeout -> torque off + unlock")
                for mid in ids:
                    bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 0)
                    bus.write1(mid, CTRL_TABLE["Lock"][0], 0)
                break

            if msg is None:
                time.sleep(dt)
                continue

            if msg.get("unit") != "u01":
                continue

            u_field = msg.get("u", {})
            u_by_id: Dict[int, float] = {}

            # Normalize keys -> ids
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

                if mid not in ids:
                    continue

                try:
                    u_by_id[mid] = clamp01(float(v))
                except Exception:
                    continue

            if not u_by_id:
                time.sleep(dt)
                continue

            # Auto-trim once: align current follower pose to the first received u.
            if args.auto_trim and not hasattr(main, "_auto_trim_done"):
                # Enable torque briefly so present position reads reflect reality; then torque off is fine too.
                # We will read current follower ticks and compute trim to match target.
                targets_now: Dict[int, int] = {}
                for mid, u in u_by_id.items():
                    if INVERT_U.get(mid, False):
                        u = 1.0 - u

                    if mid == 6 and GRIPPER_RANGE_OVERRIDE is not None:
                        lo, hi = GRIPPER_RANGE_OVERRIDE
                        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
                    else:
                        lo, hi = soft_limits[mid]
                        lo, hi = (lo, hi) if lo <= hi else (hi, lo)

                    tgt = int(round(lo + u * (hi - lo)))
                    tgt = clamp_int(tgt, lo, hi)
                    targets_now[mid] = tgt

                # Read current ticks and compute trim = current - target
                for mid, tgt in targets_now.items():
                    try:
                        cur = bus.read_present_pos_ticks(mid)
                        TRIM_TICKS[mid] += int(cur - tgt)
                    except Exception:
                        pass

                setattr(main, "_auto_trim_done", True)
                print(f"[follower] Auto-trim computed. New trim ticks: {TRIM_TICKS}")

            # Enable torque best effort (do not crash on gripper overload)
            for mid in ids:
                if mid == 6 and not gripper_ok:
                    continue
                ok = torque_on(mid)
                if not ok and mid == 6:
                    gripper_ok = False
                    # Keep lock off to allow manual movement
                    bus.write1(6, CTRL_TABLE["Torque_Enable"][0], 0)
                    bus.write1(6, CTRL_TABLE["Lock"][0], 0)
                    print("[follower] WARN: gripper torque enable failed (overload). Skipping gripper.", file=sys.stderr)

            # Map u -> target ticks with inversion, range override, trim
            targets: Dict[int, int] = {}
            for mid, u in u_by_id.items():
                if mid == 6 and (args.no_gripper or not gripper_ok):
                    continue

                if INVERT_U.get(mid, False):
                    u = 1.0 - u

                # Choose mapping range
                if mid == 6 and GRIPPER_RANGE_OVERRIDE is not None:
                    lo, hi = GRIPPER_RANGE_OVERRIDE
                    lo, hi = (lo, hi) if lo <= hi else (hi, lo)
                else:
                    lo, hi = soft_limits[mid]
                    lo, hi = (lo, hi) if lo <= hi else (hi, lo)

                tgt = int(round(lo + u * (hi - lo)))
                tgt = clamp_int(tgt, lo, hi)

                # Apply trim (ticks)
                tgt += int(TRIM_TICKS.get(mid, 0))
                tgt = clamp_int(tgt, lo, hi)

                targets[mid] = tgt

            # Rate limit and write goals
            goals: Dict[int, int] = {}
            for mid, tgt in targets.items():
                if q_cmd[mid] is None:
                    goals[mid] = tgt
                else:
                    delta = tgt - int(q_cmd[mid])
                    step = clamp_int(delta, -args.max_step, args.max_step)
                    goals[mid] = int(q_cmd[mid]) + step
                q_cmd[mid] = goals[mid]

            # Write goals sequentially (robust across SDK variants)
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
