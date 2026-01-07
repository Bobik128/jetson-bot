#!/usr/bin/env python3
"""
follower_receive_sts3215.py

Jetson-side UDP receiver that teleoperates Feetech STS3215 servos (protocol 0)
using packets from a leader (e.g., LeRobot leader arm).

Features:
- Accepts leader packets with either:
    q keys as numeric ids: {"q":{"2":1234,"3":...}}   OR
    q keys as names:      {"q":{"shoulder_lift":12.3,"elbow_flex":...}}
- Accepts units:
    unit="ticks"  -> values are servo ticks (0..4095-ish)
    unit="deg"    -> values are degrees (converted to ticks)
    unit="lerobot_norm" -> assumes:
        shoulder_lift/elbow_flex/wrist_flex are DEGREES
        gripper is 0..100 (mapped to ticks using servo EEPROM min/max)
- Safety:
    - rate limiting (max ticks per update)
    - clamp to servo EEPROM min/max limits
    - timeout after first packet -> torque off + unlock

Usage:
  python3 follower_receive_sts3215.py --port /dev/ttyACM0 --ids 2 3 4 6 --udp_port 5005
"""

import argparse
import json
import socket
import sys
import time
from typing import Dict, Tuple


# ------------------ STS3215 constants (LeRobot-compatible) ------------------

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

STS_RESOLUTION = 4096  # ticks per full turn (360 deg)


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value < 0:
        return (1 << sign_bit) | (abs(value) & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value & (1 << sign_bit):
        return - (value & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def deg_to_ticks(deg: float) -> int:
    return int(round(deg * STS_RESOLUTION / 360.0))


def pct_to_ticks(pct_0_100: float, mn: int, mx: int) -> int:
    pct = max(0.0, min(100.0, float(pct_0_100)))
    return int(round(mn + (mx - mn) * (pct / 100.0)))


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    # Same workaround LeRobot uses for some scservo_sdk builds
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class STS3215Bus:
    """
    Minimal STS3215 bus wrapper (protocol 0) that works across scservo_sdk variants.
    Uses sequential writes for maximum compatibility.
    """
    def __init__(self, port: str, baudrate: int = 1_000_000, protocol_version: int = 0):
        import scservo_sdk as scs
        self.scs = scs
        self.port = port
        self.baudrate = baudrate
        self.protocol_version = protocol_version

        self.port_handler = scs.PortHandler(port)
        self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(self.port_handler, scs.PortHandler)
        self.packet_handler = scs.PacketHandler(protocol_version)

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

    def _check(self, comm: int, err: int, motor_id: int, ctx: str):
        if comm != self.scs.COMM_SUCCESS:
            raise RuntimeError(f"[ID {motor_id}] COMM error in {ctx}: {self.packet_handler.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID {motor_id}] Servo error in {ctx}: {self.packet_handler.getRxPacketError(err)}")

    def _write_1b(self, motor_id: int, addr: int, value: int) -> None:
        ret = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, addr, value)
        if isinstance(ret, tuple) and len(ret) == 2:
            comm, err = ret
        elif isinstance(ret, tuple) and len(ret) == 3:
            _, comm, err = ret
        else:
            raise RuntimeError(f"Unexpected return from write1ByteTxRx: {ret!r}")
        self._check(comm, err, motor_id, f"write1B addr={addr} value={value}")

    def _write_2b(self, motor_id: int, addr: int, value: int) -> None:
        ret = self.packet_handler.write2ByteTxRx(self.port_handler, motor_id, addr, value)
        if isinstance(ret, tuple) and len(ret) == 2:
            comm, err = ret
        elif isinstance(ret, tuple) and len(ret) == 3:
            _, comm, err = ret
        else:
            raise RuntimeError(f"Unexpected return from write2ByteTxRx: {ret!r}")
        self._check(comm, err, motor_id, f"write2B addr={addr} value={value}")

    def _read_2b(self, motor_id: int, addr: int) -> int:
        ret = self.packet_handler.read2ByteTxRx(self.port_handler, motor_id, addr)
        if isinstance(ret, tuple) and len(ret) == 3:
            val, comm, err = ret
        elif isinstance(ret, tuple) and len(ret) == 2:
            val, comm = ret
            err = 0
        else:
            raise RuntimeError(f"Unexpected return from read2ByteTxRx: {ret!r}")
        self._check(comm, err, motor_id, f"read2B addr={addr}")
        return int(val)

    def torque_enable(self, motor_id: int, enable: bool):
        addr, _ = CTRL_TABLE["Torque_Enable"]
        self._write_1b(motor_id, addr, 1 if enable else 0)

    def lock(self, motor_id: int, enable: bool):
        addr, _ = CTRL_TABLE["Lock"]
        self._write_1b(motor_id, addr, 1 if enable else 0)

    def set_operating_mode_position(self, motor_id: int):
        addr, _ = CTRL_TABLE["Operating_Mode"]
        self._write_1b(motor_id, addr, 0)  # POSITION

    def set_acceleration(self, motor_id: int, accel: int):
        addr, _ = CTRL_TABLE["Acceleration"]
        self._write_1b(motor_id, addr, int(accel))

    def read_present_position(self, motor_id: int) -> int:
        addr, _ = CTRL_TABLE["Present_Position"]
        raw = self._read_2b(motor_id, addr)
        return decode_sign_magnitude(raw, SIGN_BITS["Present_Position"])

    def read_limits(self, motor_id: int) -> Tuple[int, int]:
        mn_addr, _ = CTRL_TABLE["Min_Position_Limit"]
        mx_addr, _ = CTRL_TABLE["Max_Position_Limit"]
        mn = self._read_2b(motor_id, mn_addr)
        mx = self._read_2b(motor_id, mx_addr)
        return mn, mx

    def write_goal_positions(self, ids_to_pos_ticks: Dict[int, int]):
        addr, _ = CTRL_TABLE["Goal_Position"]
        for motor_id, pos in ids_to_pos_ticks.items():
            raw = encode_sign_magnitude(int(pos), SIGN_BITS["Goal_Position"])
            self._write_2b(motor_id, addr, raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--max_step", type=int, default=40, help="Max ticks per update per joint")
    ap.add_argument("--timeout_s", type=float, default=0.5, help="Timeout AFTER first packet")
    ap.add_argument("--acceleration", type=int, default=254)
    ap.add_argument("--torque_on_start", action="store_true")
    args = ap.parse_args()

    # Map leader joint names -> follower IDs (adjust if you rename)
    NAME_TO_ID = {
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "gripper": 6,
    }

    ids = args.ids

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.settimeout(0.1)

    # Servo bus
    bus = STS3215Bus(args.port, args.baudrate, protocol_version=0)
    bus.connect()

    # Configure servos
    for i in ids:
        bus.set_operating_mode_position(i)
        bus.set_acceleration(i, args.acceleration)
        bus.lock(i, True)
        bus.torque_enable(i, bool(args.torque_on_start))

    # Read limits from servo EEPROM for clamp safety
    limits: Dict[int, Tuple[int, int]] = {}
    for i in ids:
        limits[i] = bus.read_limits(i)

    print(f"[follower] Listening UDP :{args.udp_port}, controlling IDs={ids}")
    print("[follower] Limits (from servo):")
    for i in ids:
        mn, mx = limits[i]
        print(f"  ID {i}: min={mn} max={mx}")

    dt = 1.0 / args.hz
    got_first_packet = False
    last_packet_t = time.time()

    # last commanded positions (ticks)
    q_cmd: Dict[int, int] = {i: None for i in ids}

    try:
        while True:
            msg = None
            try:
                data, _ = sock.recvfrom(8192)
                msg = json.loads(data.decode("utf-8"))
                last_packet_t = time.time()
                got_first_packet = True
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[follower] WARN: bad UDP packet: {e}", file=sys.stderr)

            # Timeout only after first packet (so you can start receiver first)
            if got_first_packet and (time.time() - last_packet_t) > args.timeout_s:
                print("[follower] Teleop timeout -> torque off + unlock")
                for i in ids:
                    try:
                        bus.torque_enable(i, False)
                        bus.lock(i, False)
                    except Exception:
                        pass
                break

            if msg is None:
                time.sleep(dt)
                continue

            unit = msg.get("unit", "ticks")
            q_field = msg.get("q", {})

            # Normalize incoming q keys to servo IDs
            q_in_raw: Dict[int, float] = {}
            for k, v in q_field.items():
                sid = None
                if isinstance(k, int):
                    sid = k
                elif isinstance(k, str) and k.isdigit():
                    sid = int(k)
                elif isinstance(k, str) and k in NAME_TO_ID:
                    sid = NAME_TO_ID[k]
                else:
                    continue

                if sid not in ids:
                    continue

                try:
                    q_in_raw[sid] = float(v)
                except Exception:
                    continue

            if not q_in_raw:
                time.sleep(dt)
                continue

            # Enable torque when actively receiving commands
            for i in ids:
                bus.torque_enable(i, True)

            # Convert units -> ticks
            q_target_ticks: Dict[int, int] = {}
            for sid, val in q_in_raw.items():
                mn, mx = limits[sid]

                if unit in ("ticks", "tick"):
                    tgt = int(round(val))

                elif unit in ("deg", "degree", "degrees"):
                    tgt = deg_to_ticks(val)

                elif unit == "lerobot_norm":
                    # Assumption: joints are DEGREES, gripper is 0..100
                    if sid == 6:
                        tgt = pct_to_ticks(val, mn, mx)
                    else:
                        tgt = deg_to_ticks(val)

                else:
                    # Unknown unit -> assume ticks
                    tgt = int(round(val))

                # Clamp to EEPROM limits for safety
                tgt = clamp(tgt, mn, mx)
                q_target_ticks[sid] = tgt

            # Apply rate limiting and write goals
            goals: Dict[int, int] = {}
            for sid, tgt in q_target_ticks.items():
                if q_cmd[sid] is None:
                    goals[sid] = tgt
                else:
                    step = clamp(tgt - q_cmd[sid], -args.max_step, args.max_step)
                    goals[sid] = q_cmd[sid] + step
                q_cmd[sid] = goals[sid]

            bus.write_goal_positions(goals)
            time.sleep(dt)

    finally:
        try:
            bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
