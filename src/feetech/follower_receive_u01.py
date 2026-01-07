#!/usr/bin/env python3
import argparse
import json
import socket
import sys
import time
from typing import Dict, Tuple

import scservo_sdk as scs


CTRL_TABLE = {
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Torque_Limit": (16, 2),
    "Protection_Current": (28, 2),
    "Overload_Torque": (36, 1),
    "Operating_Mode": (33, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Lock": (55, 1),
}

SIGN_BITS = {"Goal_Position": 15}


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value < 0:
        return (1 << sign_bit) | (abs(value) & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class Bus:
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

    def _unpack_2_or_3(self, ret):
        if isinstance(ret, tuple) and len(ret) == 2:
            return None, ret[0], ret[1]
        if isinstance(ret, tuple) and len(ret) == 3:
            return ret[0], ret[1], ret[2]
        raise RuntimeError(f"Unexpected SDK return: {ret!r}")

    def write1(self, mid: int, addr: int, val: int) -> Tuple[bool, str]:
        try:
            ret = self.packet_handler.write1ByteTxRx(self.port_handler, mid, addr, val)
            _, comm, err = self._unpack_2_or_3(ret)
            if comm != scs.COMM_SUCCESS:
                return False, self.packet_handler.getTxRxResult(comm)
            if err != 0:
                return False, self.packet_handler.getRxPacketError(err)
            return True, ""
        except Exception as e:
            return False, str(e)

    def write2(self, mid: int, addr: int, val: int) -> Tuple[bool, str]:
        try:
            ret = self.packet_handler.write2ByteTxRx(self.port_handler, mid, addr, val)
            _, comm, err = self._unpack_2_or_3(ret)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--max_step", type=int, default=40)
    ap.add_argument("--timeout_s", type=float, default=0.6)
    ap.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6])
    args = ap.parse_args()

    NAME_TO_ID = {"shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "gripper": 6}

    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.settimeout(0.1)

    # Servo bus
    bus = Bus(args.port, args.baudrate, protocol=0)
    bus.connect()

    ids = args.ids

    # Configure + read limits
    limits: Dict[int, Tuple[int, int]] = {}
    for mid in ids:
        bus.write1(mid, CTRL_TABLE["Operating_Mode"][0], 0)  # POSITION
        bus.write1(mid, CTRL_TABLE["Acceleration"][0], 254)
        bus.write1(mid, CTRL_TABLE["Lock"][0], 1)

        mn = bus.read2(mid, CTRL_TABLE["Min_Position_Limit"][0])
        mx = bus.read2(mid, CTRL_TABLE["Max_Position_Limit"][0])
        limits[mid] = (mn, mx)

    print(f"[follower] Listening UDP :{args.udp_port}, controlling IDs={ids}")
    print("[follower] Limits (from servo):")
    for mid in ids:
        mn, mx = limits[mid]
        print(f"  ID {mid}: min={mn} max={mx}")

    # Gripper safer settings (best effort)
    if 6 in ids:
        bus.write2(6, CTRL_TABLE["Max_Torque_Limit"][0], 500)
        bus.write2(6, CTRL_TABLE["Protection_Current"][0], 250)
        bus.write1(6, CTRL_TABLE["Overload_Torque"][0], 25)

    dt = 1.0 / args.hz
    got_first = False
    last_packet = time.time()

    q_cmd: Dict[int, int] = {mid: None for mid in ids}
    gripper_ok = True

    try:
        while True:
            msg = None
            try:
                data, _ = sock.recvfrom(8192)
                msg = json.loads(data.decode("utf-8"))
                got_first = True
                last_packet = time.time()
            except socket.timeout:
                pass

            if got_first and (time.time() - last_packet) > args.timeout_s:
                print("[follower] Timeout -> torque off + unlock")
                for mid in ids:
                    bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 0)
                    bus.write1(mid, CTRL_TABLE["Lock"][0], 0)
                break

            if msg is None:
                time.sleep(dt)
                continue

            if msg.get("unit") != "u01":
                # Ignore other packet types to avoid wrong mappings
                continue

            u_field = msg.get("u", {})
            u_by_id: Dict[int, float] = {}

            for k, v in u_field.items():
                if isinstance(k, str) and k in NAME_TO_ID:
                    mid = NAME_TO_ID[k]
                elif isinstance(k, str) and k.isdigit():
                    mid = int(k)
                else:
                    continue
                if mid not in ids:
                    continue
                try:
                    u_by_id[mid] = float(v)
                except Exception:
                    continue

            if not u_by_id:
                time.sleep(dt)
                continue

            # Enable torque on demand; do not crash on gripper overload
            for mid in ids:
                if mid == 6 and not gripper_ok:
                    continue
                ok, err = bus.write1(mid, CTRL_TABLE["Torque_Enable"][0], 1)
                if not ok and mid == 6:
                    gripper_ok = False
                    print(f"[follower] WARN: gripper torque enable failed: {err}", file=sys.stderr)

            # Map u∈[0,1] -> ticks within follower limits
            targets: Dict[int, int] = {}
            for mid, u in u_by_id.items():
                if mid == 6 and not gripper_ok:
                    continue
                mn, mx = limits[mid]
                u = 0.0 if u < 0.0 else 1.0 if u > 1.0 else u
                tgt = int(round(mn + u * (mx - mn)))
                tgt = clamp(tgt, mn, mx)
                targets[mid] = tgt

            # Rate limit
            goals: Dict[int, int] = {}
            for mid, tgt in targets.items():
                if q_cmd[mid] is None:
                    goals[mid] = tgt
                else:
                    step = clamp(tgt - q_cmd[mid], -args.max_step, args.max_step)
                    goals[mid] = q_cmd[mid] + step
                q_cmd[mid] = goals[mid]

            # Write goals
            for mid, pos in goals.items():
                raw = encode_sign_magnitude(pos, SIGN_BITS["Goal_Position"])
                ok, err = bus.write2(mid, CTRL_TABLE["Goal_Position"][0], raw)
                if not ok:
                    print(f"[follower] WARN: write goal failed ID {mid}: {err}", file=sys.stderr)

            time.sleep(dt)

    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()