#!/usr/bin/env python3
"""
Calibrate Feetech STS3215 (protocol 0) like LeRobot SO101 calibration.

What it does:
- Disables torque so you can move joints by hand.
- Computes homing offsets from a "mid-range" pose:
    homing_offset = present_position - 2047  (4096 resolution => half-turn)
- Records min/max position limits by sampling Present_Position while you move joints.
- Optionally writes Homing_Offset, Min_Position_Limit, Max_Position_Limit into the servo EEPROM.
- Saves calibration JSON in the same structure as LeRobot.

WARNING:
- Torque will be DISABLED during calibration.
- Move joints slowly; do not slam into hard stops.
"""

import argparse
import json
import sys
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---- LeRobot control table for STS3215 (STS/SMS series) ----
ADDR_FIRMWARE_MAJOR = (0, 1)
ADDR_FIRMWARE_MINOR = (1, 1)
ADDR_MODEL_NUMBER   = (3, 2)

CTRL_TABLE = {
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
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

# Sign-magnitude encoding bits used by LeRobot for STS3215
SIGN_BITS = {
    "Homing_Offset": 11,
    "Goal_Position": 15,
    "Present_Position": 15,
}

STS3215_RESOLUTION = 4096
HALF_TURN = (STS3215_RESOLUTION - 1) // 2  # 4095//2 = 2047


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    """
    Encode signed int into sign-magnitude integer with sign bit at position sign_bit.
    """
    if value < 0:
        return (1 << sign_bit) | (abs(value) & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    """
    Decode sign-magnitude integer with sign bit at position sign_bit into signed int.
    """
    if value & (1 << sign_bit):
        return - (value & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    """
    Same workaround used by LeRobot to fix timeout behavior in some scservo_sdk builds.
    """
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


@dataclass
class CalibEntry:
    id: int
    drive_mode: int = 0
    homing_offset: int = 0
    range_min: int = 0
    range_max: int = 0

    def to_dict(self):
        return {
            "id": self.id,
            "drive_mode": self.drive_mode,
            "homing_offset": self.homing_offset,
            "range_min": self.range_min,
            "range_max": self.range_max,
        }


class STS3215Bus:
    def __init__(self, port: str, baudrate: int = 1_000_000, timeout_ms: int = 1000, protocol_version: int = 0):
        self.port = port
        self.baudrate = baudrate
        self.timeout_ms = timeout_ms
        self.protocol_version = protocol_version

        import scservo_sdk as scs
        self.scs = scs
        self.port_handler = scs.PortHandler(port)
        # monkeypatch like LeRobot
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

    def _write_1b(self, motor_id: int, addr: int, value: int) -> None:
        ret = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, addr, value)
        # Some scservo_sdk builds return (comm, err); others return (something, comm, err)
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

    def _read_1b(self, motor_id: int, addr: int) -> int:
        ret = self.packet_handler.read1ByteTxRx(self.port_handler, motor_id, addr)
        # Usually returns (val, comm, err). Some builds might return (val, comm) or similar.
        if isinstance(ret, tuple) and len(ret) == 3:
            val, comm, err = ret
        elif isinstance(ret, tuple) and len(ret) == 2:
            val, comm = ret
            err = 0
        else:
            raise RuntimeError(f"Unexpected return from read1ByteTxRx: {ret!r}")
        self._check(comm, err, motor_id, f"read1B addr={addr}")
        return int(val)

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

    def _check(self, comm: int, err: int, motor_id: int, ctx: str):
        if comm != self.scs.COMM_SUCCESS:
            raise RuntimeError(f"[ID {motor_id}] COMM error in {ctx}: {self.packet_handler.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID {motor_id}] Servo error in {ctx}: {self.packet_handler.getRxPacketError(err)}")

    # High-level helpers matching LeRobot behavior
    def torque_enable(self, motor_id: int, enable: bool):
        addr, _ = CTRL_TABLE["Torque_Enable"]
        self._write_1b(motor_id, addr, 1 if enable else 0)

    def lock(self, motor_id: int, enable: bool):
        addr, _ = CTRL_TABLE["Lock"]
        self._write_1b(motor_id, addr, 1 if enable else 0)

    def set_operating_mode_position(self, motor_id: int):
        addr, _ = CTRL_TABLE["Operating_Mode"]
        self._write_1b(motor_id, addr, 0)  # POSITION

    def read_present_position(self, motor_id: int) -> int:
        addr, _ = CTRL_TABLE["Present_Position"]
        raw = self._read_2b(motor_id, addr)
        return decode_sign_magnitude(raw, SIGN_BITS["Present_Position"])

    def write_homing_offset(self, motor_id: int, offset: int):
        addr, _ = CTRL_TABLE["Homing_Offset"]
        raw = encode_sign_magnitude(offset, SIGN_BITS["Homing_Offset"])
        self._write_2b(motor_id, addr, raw)

    def write_min_limit(self, motor_id: int, limit: int):
        addr, _ = CTRL_TABLE["Min_Position_Limit"]
        self._write_2b(motor_id, addr, int(limit))

    def write_max_limit(self, motor_id: int, limit: int):
        addr, _ = CTRL_TABLE["Max_Position_Limit"]
        self._write_2b(motor_id, addr, int(limit))

    def configure_basic(self, motor_id: int, return_delay_time: int = 0, acceleration: int = 254):
        addr, _ = CTRL_TABLE["Return_Delay_Time"]
        self._write_1b(motor_id, addr, return_delay_time)
        addr, _ = CTRL_TABLE["Acceleration"]
        self._write_1b(motor_id, addr, acceleration)

    def sync_write_goal_position(self, ids_to_pos: dict[int, int]):
        """
        Minimal sequential write (works everywhere). We avoid GroupSyncWrite because
        scservo_sdk variants differ a lot.
        """
        addr, _ = CTRL_TABLE["Goal_Position"]
        for motor_id, pos in ids_to_pos.items():
            raw = encode_sign_magnitude(int(pos), SIGN_BITS["Goal_Position"])
            self._write_2b(motor_id, addr, raw)


def wait_for_enter_in_thread(prompt: str) -> threading.Event:
    evt = threading.Event()

    def _run():
        try:
            input(prompt)
        except EOFError:
            pass
        evt.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return evt


def record_ranges(bus: STS3215Bus, ids: List[int], hz: float = 50.0) -> Tuple[Dict[int, int], Dict[int, int]]:
    dt = 1.0 / hz
    mins = {i: 10**9 for i in ids}
    maxs = {i: -10**9 for i in ids}

    stop_evt = wait_for_enter_in_thread(
        "Move all joints through their FULL range now.\nPress ENTER to stop recording ranges...\n"
    )

    while not stop_evt.is_set():
        for i in ids:
            try:
                p = bus.read_present_position(i)
            except Exception as e:
                print(f"[WARN] failed reading ID {i}: {e}", file=sys.stderr)
                continue
            if p < mins[i]:
                mins[i] = p
            if p > maxs[i]:
                maxs[i] = p
        time.sleep(dt)

    # Replace unset values if something went wrong
    for i in ids:
        if mins[i] == 10**9 or maxs[i] == -10**9:
            raise RuntimeError(f"Did not record any valid samples for ID {i}. Check wiring/baud/protocol.")

    return mins, maxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--ids", type=int, nargs="+", required=True,
                    help="Servo IDs to calibrate, e.g. --ids 2 3 4 6")
    ap.add_argument("--names", type=str, default=None,
                    help="Optional names matching ids order, comma-separated. "
                         "Example: --names shoulder_lift,elbow_flex,wrist_flex,gripper")
    ap.add_argument("--out", type=str, default="calibration.json")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--write", action="store_true", help="Write calibration to servo EEPROM (default).")
    ap.add_argument("--no-write", action="store_true", help="Do not write to servos; only save JSON.")
    ap.add_argument("--acceleration", type=int, default=254)
    ap.add_argument("--return-delay", type=int, default=0)
    args = ap.parse_args()

    do_write = True
    if args.no_write:
        do_write = False
    if args.write:
        do_write = True

    ids = args.ids
    if args.names:
        names = [s.strip() for s in args.names.split(",")]
        if len(names) != len(ids):
            raise ValueError("--names must have same count as --ids")
        name_by_id = {i: n for i, n in zip(ids, names)}
    else:
        # Default key names "id_2", "id_3", ...
        name_by_id = {i: f"id_{i}" for i in ids}

    bus = STS3215Bus(args.port, args.baudrate)

    try:
        bus.connect()
        print(f"Connected to {args.port} @ {args.baudrate} (protocol 0)")

        # Safety: torque off + unlock
        for i in ids:
            bus.torque_enable(i, False)
            bus.lock(i, False)

        # Configure + ensure position mode (like LeRobot does during calibration)
        for i in ids:
            bus.configure_basic(i, return_delay_time=args.return_delay, acceleration=args.acceleration)
            bus.set_operating_mode_position(i)

        input(
            f"\nMove the arm so EACH joint is approximately in the MIDDLE of its mechanical range.\n"
            f"Then press ENTER to capture center pose and compute homing offsets...\n"
        )

        # Read present positions and compute homing offsets like LeRobot:
        # half_turn_homing = present - 2047
        homing_offsets: Dict[int, int] = {}
        present_at_center: Dict[int, int] = {}
        for i in ids:
            p = bus.read_present_position(i)
            present_at_center[i] = p
            homing_offsets[i] = p - HALF_TURN

        print("\nCenter capture:")
        for i in ids:
            print(f"  ID {i}: present={present_at_center[i]} -> homing_offset={homing_offsets[i]}")

        # Record ranges of motion
        mins, maxs = record_ranges(bus, ids, hz=args.hz)

        print("\nRecorded ranges:")
        for i in ids:
            print(f"  ID {i}: min={mins[i]}  max={maxs[i]}  (span={maxs[i]-mins[i]})")

        # Build calibration dict
        calib: Dict[str, CalibEntry] = {}
        for i in ids:
            calib[name_by_id[i]] = CalibEntry(
                id=i,
                drive_mode=0,
                homing_offset=homing_offsets[i],
                range_min=mins[i],
                range_max=maxs[i],
            )

        # Write to servos (optional)
        if do_write:
            print("\nWriting calibration to servos (EEPROM)...")
            # Keep torque disabled while writing
            for i in ids:
                bus.torque_enable(i, False)
                bus.lock(i, False)

            for key, entry in calib.items():
                i = entry.id
                bus.write_homing_offset(i, entry.homing_offset)
                bus.write_min_limit(i, entry.range_min)
                bus.write_max_limit(i, entry.range_max)

            print("Done writing calibration.")

        # Save JSON
        out_obj = {k: v.to_dict() for k, v in calib.items()}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=4)
        print(f"\nSaved calibration to: {args.out}")

    finally:
        # Leave torque disabled after calibration by default (safer during setup)
        try:
            for i in ids:
                bus.torque_enable(i, False)
                bus.lock(i, False)
        except Exception:
            pass
        bus.disconnect()


if __name__ == "__main__":
    main()
