#!/usr/bin/env python3
"""
Jetson UDP follower for STS3215 (protocol 0) with overload-safe torque enable.

- Accepts q keys as numeric ids ("2") OR names ("shoulder_lift").
- Accepts unit: ticks | deg | lerobot_norm
  - lerobot_norm: joints in degrees, gripper in 0..100 (%)
- Safety:
  - rate limiting
  - clamp to servo EEPROM min/max
  - timeout after first packet -> torque off + unlock
- Robustness:
  - Overload errors on torque enable (common on grippers) do not crash the program.
"""

import argparse
import json
import socket
import sys
import time
from typing import Dict, Tuple


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

STS_RESOLUTION = 4096


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
    return int(round(float(deg) * STS_RESOLUTION / 360.0))


def pct_to_ticks(pct_0_100: float, mn: int, mx: int) -> int:
    pct = max(0.0, min(100.0, float(pct_0_100)))
    return int(round(mn + (mx - mn) * (pct / 100.0)))


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class STS3215Bus:
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
            # Keep the raw err code available for handling
            raise ServoError(motor_id, ctx, err, self.packet_handler.getRxPacketError(err))


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
        ret = self.packet_handler.write2ByteTxRx(self.port_handler, motor_id, addrladdr := addr, value)
        # Some scservo_sdk builds have quirks; keep robust unpack:
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

    def _write_1b_safe(self, motor_id: int, addr: int, value: int, retries: int = 3, delay_s: float = 0.05) -> bool:
        for _ in range(retries):
            try:
                self._write_1b(motor_id, addr, value)
                return True
            except ServoError as e:
                # Overload or other servo error: retry a few times
                time.sleep(delay_s)
        return False

    def torque_enable(self, motor_id: int, enable: bool, retries: int = 3) -> bool:
        addr, _ = CTRL_TABLE["Torque_Enable"]
        return self._write_1b_safe(motor_id, addr, 1 if enable else 0, retries=retries)

    def lock(self, motor_id: int, enable: bool, retries: int = 3) -> bool:
        addr, _ = CTRL_TABLE["Lock"]
        return self._write_1b_safe(motor_id, addr, 1 if enable else 0, retries=retries)

    def set_operating_mode_position(self, motor_id: int) -> bool:
        addr, _ = CTRL_TABLE["Operating_Mode"]
        return self._write_1b_safe(motor_id, addr, 0)

    def set_acceleration(self, motor_id: int, accel: int) -> bool:
        addr, _ = CTRL_TABLE["Acceleration"]
        return self._write_1b_safe(motor_id, addr, int(accel))

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
            # Goal_Position is 2 bytes
            self._write_2b(motor_id, addr, raw)

    # Gripper safety settings like LeRobot configure()
    def set_gripper_safety(self, motor_id: int = 6) -> None:
        # LeRobot uses (for gripper):
        # Max_Torque_Limit = 500 (50%)
        # Protection_Current = 250 (50%)
        # Overload_Torque = 25 (25%)
        # These are relatively conservative.
        try:
            addr, _ = CTRL_TABLE["Max_Torque_Limit"]
            self._write_2b(motor_id, addr, 500)
            addr, _ = CTRL_TABLE["Protection_Current"]
            self._write_2b(motor_id, addr, 250)
            addr, _ = CTRL_TABLE["Overload_Torque"]
            self._write_1b(motor_id, addr, 25)
        except ServoError:
            # If it fails, do not crash; continue.
            pass


class ServoError(RuntimeError):
    def __init__(self, motor_id: int, ctx: str, err_code: int, err_text: str):
        super().__init__(f"[ID {motor_id}] Servo error in {ctx}: {err_text}")
        self.motor_id = motor_id
        self.ctx = ctx
        self.err_code = err_code
        self.err_text = err_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--max_step", type=int, default=40)
    ap.add_argument("--timeout_s", type=float, default=0.5)
    ap.add_argument("--acceleration", type=int, default=254)
    ap.add_argument("--torque_on_start", action="store_true")
    args = ap.parse_args()

    NAME_TO_ID = {
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "gripper": 6,
    }

    ids = args.ids

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.settimeout(0.1)

    bus = STS3215Bus(args.port, args.baudrate, protocol_version=0)
    bus.connect()

    # Configure motors
    for i in ids:
        bus.set_operating_mode_position(i)
        bus.set_acceleration(i, args.acceleration)
        bus.lock(i, True)
        # Apply gripper safety params before torque enable attempts
        if i == 6:
            bus.set_gripper_safety(6)
        bus.torque_enable(i, bool(args.torque_on_start), retries=2)

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
    q_cmd: Dict[int, int] = {i: None for i in ids}
    gripper_enabled = bool(args.torque_on_start)

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

            if got_first_packet and (time.time() - last_packet_t) > args.timeout_s:
                print("[follower] Teleop timeout -> torque off + unlock")
                for i in ids:
                    bus.torque_enable(i, False, retries=1)
                    bus.lock(i, False, retries=1)
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

            # Enable torque for arm joints (2/3/4) every time (safe).
            for sid in [i for i in ids if i != 6]:
                ok = bus.torque_enable(sid, True, retries=2)
                if not ok:
                    print(f"[follower] WARN: could not enable torque on ID {sid}", file=sys.stderr)

            # Enable gripper torque carefully; if overload persists, keep it off.
            if 6 in ids:
                if not gripper_enabled:
                    # Try to enable with safety settings; if overload occurs, continue without gripper torque.
                    bus.set_gripper_safety(6)
                    ok = bus.torque_enable(6, True, retries=2)
                    if ok:
                        gripper_enabled = True
                    else:
                        # Overload likely; keep it disabled and do not crash.
                        bus.torque_enable(6, False, retries=1)
                        bus.lock(6, False, retries=1)
                        gripper_enabled = False
                        print("[follower] WARN: gripper torque enable failed (overload). Continuing without gripper.", file=sys.stderr)

            # Convert incoming units -> ticks
            q_target_ticks: Dict[int, int] = {}
            for sid, val in q_in_raw.items():
                mn, mx = limits[sid]

                if unit in ("ticks", "tick"):
                    tgt = int(round(val))
                elif unit in ("deg", "degree", "degrees"):
                    tgt = deg_to_ticks(val)
                elif unit == "lerobot_norm":
                    # joints are degrees; gripper is 0..100
                    if sid == 6:
                        tgt = pct_to_ticks(val, mn, mx)
                    else:
                        tgt = deg_to_ticks(val)
                else:
                    tgt = int(round(val))

                tgt = clamp(tgt, mn, mx)
                q_target_ticks[sid] = tgt

            # Apply rate limiting and write goals
            goals: Dict[int, int] = {}
            for sid, tgt in q_target_ticks.items():
                # Skip commanding gripper if torque isn't enabled (prevents fighting under load)
                if sid == 6 and not gripper_enabled:
                    continue

                if q_cmd[sid] is None:
                    goals[sid] = tgt
                else:
                    step = clamp(tgt - q_cmd[sid], -args.max_step, args.max_step)
                    goals[sid] = q_cmd[sid] + step
                q_cmd[sid] = goals[sid]

            if goals:
                bus.write_goal_positions(goals)

            time.sleep(dt)

    finally:
        try:
            bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()