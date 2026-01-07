#!/usr/bin/env python3
"""
Calibrate Feetech STS3215 (protocol 0) in a self-consistent way.

Key fix vs many naive scripts:
- We WRITE Homing_Offset immediately after capturing the mid pose.
- Then we record Min/Max while the new offset is active.
This prevents the common "limits shifted / can't reach calibrated region" bug.

Process:
1) Torque OFF, unlock, set position mode
2) User sets each joint to mid-range -> ENTER
3) Compute homing_offset = present_position - HALF_TURN
4) WRITE Homing_Offset to each servo (EEPROM) right away
5) Record ranges while user moves joints -> ENTER
6) WRITE Min_Position_Limit and Max_Position_Limit (EEPROM)
7) Save JSON in LeRobot structure

Notes:
- Keep torque disabled during calibration.
- Gripper mechanisms often require a smaller safe range than full mechanical range; you can still calibrate it,
  but it may be better to override gripper range later in teleop mapping.
"""

import argparse
import json
import sys
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import scservo_sdk as scs

CTRL_TABLE = {
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

SIGN_BITS = {
    "Homing_Offset": 11,
    "Goal_Position": 15,
    "Present_Position": 15,
}

STS3215_RESOLUTION = 4096
HALF_TURN = (STS3215_RESOLUTION - 1) // 2  # 2047


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value < 0:
        return (1 << sign_bit) | (abs(value) & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value & (1 << sign_bit):
        return -(value & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
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
    def __init__(self, port: str, baudrate: int = 1_000_000, protocol_version: int = 0):
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

    def _unpack2or3(self, ret):
        # writeXByteTxRx: some builds return (comm, err), others (something, comm, err)
        if isinstance(ret, tuple) and len(ret) == 2:
            return None, ret[0], ret[1]
        if isinstance(ret, tuple) and len(ret) == 3:
            return ret[0], ret[1], ret[2]
        raise RuntimeError(f"Unexpected SDK return: {ret!r}")

    def _check(self, comm: int, err: int, mid: int, ctx: str):
        if comm != scs.COMM_SUCCESS:
            raise RuntimeError(f"[ID {mid}] COMM error in {ctx}: {self.packet_handler.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID {mid}] Servo error in {ctx}: {self.packet_handler.getRxPacketError(err)}")

    def write1(self, mid: int, addr: int, value: int):
        ret = self.packet_handler.write1ByteTxRx(self.port_handler, mid, addr, int(value))
        _, comm, err = self._unpack2or3(ret)
        self._check(comm, err, mid, f"write1 addr={addr} value={value}")

    def write2(self, mid: int, addr: int, value: int):
        ret = self.packet_handler.write2ByteTxRx(self.port_handler, mid, addr, int(value))
        _, comm, err = self._unpack2or3(ret)
        self._check(comm, err, mid, f"write2 addr={addr} value={value}")

    def read2(self, mid: int, addr: int) -> int:
        ret = self.packet_handler.read2ByteTxRx(self.port_handler, mid, addr)
        if isinstance(ret, tuple) and len(ret) == 3:
            val, comm, err = ret
        elif isinstance(ret, tuple) and len(ret) == 2:
            val, comm = ret
            err = 0
        else:
            raise RuntimeError(f"Unexpected SDK return: {ret!r}")
        self._check(comm, err, mid, f"read2 addr={addr}")
        return int(val)

    # High level helpers
    def torque_enable(self, mid: int, enable: bool):
        self.write1(mid, CTRL_TABLE["Torque_Enable"][0], 1 if enable else 0)

    def lock(self, mid: int, enable: bool):
        self.write1(mid, CTRL_TABLE["Lock"][0], 1 if enable else 0)

    def set_position_mode(self, mid: int):
        self.write1(mid, CTRL_TABLE["Operating_Mode"][0], 0)

    def configure_basic(self, mid: int, return_delay_time: int = 0, acceleration: int = 254):
        self.write1(mid, CTRL_TABLE["Return_Delay_Time"][0], int(return_delay_time))
        self.write1(mid, CTRL_TABLE["Acceleration"][0], int(acceleration))

    def read_present_position(self, mid: int) -> int:
        raw = self.read2(mid, CTRL_TABLE["Present_Position"][0])
        return decode_sign_magnitude(raw, SIGN_BITS["Present_Position"])

    def write_homing_offset(self, mid: int, offset: int):
        raw = encode_sign_magnitude(int(offset), SIGN_BITS["Homing_Offset"])
        self.write2(mid, CTRL_TABLE["Homing_Offset"][0], raw)

    def write_min_limit(self, mid: int, limit: int):
        self.write2(mid, CTRL_TABLE["Min_Position_Limit"][0], int(limit))

    def write_max_limit(self, mid: int, limit: int):
        self.write2(mid, CTRL_TABLE["Max_Position_Limit"][0], int(limit))


def wait_for_enter(prompt: str) -> threading.Event:
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


def record_ranges(bus: STS3215Bus, ids: List[int], hz: float) -> Tuple[Dict[int, int], Dict[int, int]]:
    dt = 1.0 / max(1.0, hz)
    mins = {i: 10**9 for i in ids}
    maxs = {i: -10**9 for i in ids}

    stop_evt = wait_for_enter(
        "Move all joints sequentially through their FULL desired range now.\n"
        "Press ENTER to stop recording ranges...\n"
    )

    while not stop_evt.is_set():
        for i in ids:
            try:
                p = bus.read_present_position(i)
            except Exception as e:
                print(f"[WARN] read failed ID {i}: {e}", file=sys.stderr)
                continue
            if p < mins[i]:
                mins[i] = p
            if p > maxs[i]:
                maxs[i] = p
        time.sleep(dt)

    for i in ids:
        if mins[i] == 10**9 or maxs[i] == -10**9:
            raise RuntimeError(f"No valid samples for ID {i}. Check wiring/baud/protocol.")
    return mins, maxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--names", type=str, default=None,
                    help="Comma-separated names matching ids order, e.g. shoulder_lift,elbow_flex,wrist_flex,gripper")
    ap.add_argument("--out", type=str, default="calibration.json")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--acceleration", type=int, default=254)
    ap.add_argument("--return-delay", type=int, default=0)

    ap.add_argument("--write", action="store_true", help="Write homing offset + limits into servos (EEPROM).")
    ap.add_argument("--no-write", action="store_true", help="Do not write EEPROM; only save JSON.")
    ap.add_argument("--verify", action="store_true", help="After writing, read back EEPROM and print diffs.")
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
        name_by_id = {i: f"id_{i}" for i in ids}

    bus = STS3215Bus(args.port, args.baudrate, protocol_version=0)

    try:
        bus.connect()
        print(f"Connected to {args.port} @ {args.baudrate} (protocol 0)")

        # torque off + unlock + configure
        for i in ids:
            bus.torque_enable(i, False)
            bus.lock(i, False)
            bus.configure_basic(i, return_delay_time=args.return_delay, acceleration=args.acceleration)
            bus.set_position_mode(i)

        input(
            "\nStep 1/2: Put EACH joint approximately at the MIDDLE of its desired range.\n"
            "Press ENTER to capture centers and compute homing offsets...\n"
        )

        # Capture center and compute homing offsets
        homing_offsets: Dict[int, int] = {}
        present_at_center: Dict[int, int] = {}
        for i in ids:
            p = bus.read_present_position(i)
            present_at_center[i] = p
            homing_offsets[i] = p - HALF_TURN

        print("\nCenter capture (before writing offsets):")
        for i in ids:
            print(f"  ID {i}: present={present_at_center[i]} -> homing_offset={homing_offsets[i]}")

        # IMPORTANT FIX: write offsets FIRST (so all subsequent readings are in the correct frame)
        if do_write:
            print("\nWriting Homing_Offset to servos (EEPROM) ...")
            for i in ids:
                bus.torque_enable(i, False)
                bus.lock(i, False)
                bus.write_homing_offset(i, homing_offsets[i])
            print("Homing_Offset written.")

            # Give servos a moment; some setups benefit from a short pause
            time.sleep(0.2)

            # Optional: show what present reads now (it should be near HALF_TURN if you haven't moved)
            print("\nPresent_Position after writing offsets (do not move joints for this check):")
            for i in ids:
                p2 = bus.read_present_position(i)
                print(f"  ID {i}: present_now={p2} (ideal near {HALF_TURN})")

        else:
            print("\nNOTE: --no-write was used; offsets not written to servos.")
            print("Ranges will be recorded in the CURRENT servo frame. If you later write offsets, limits will shift.")

        print(
            "\nStep 2/2: Now move joints through their FULL desired ranges.\n"
            "Recording min/max. Press ENTER when done.\n"
        )
        mins, maxs = record_ranges(bus, ids, hz=args.hz)

        print("\nRecorded ranges (in CURRENT Present_Position frame):")
        for i in ids:
            print(f"  ID {i}: min={mins[i]}  max={maxs[i]}  span={maxs[i]-mins[i]}")

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

        # Write min/max
        if do_write:
            print("\nWriting Min/Max limits to servos (EEPROM) ...")
            for key, entry in calib.items():
                mid = entry.id
                bus.torque_enable(mid, False)
                bus.lock(mid, False)
                bus.write_min_limit(mid, entry.range_min)
                bus.write_max_limit(mid, entry.range_max)
            print("Min/Max limits written.")

        # Save JSON
        out_obj = {k: v.to_dict() for k, v in calib.items()}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=4)
        print(f"\nSaved calibration to: {args.out}")

        # Optional verification: read back EEPROM and compare to saved
        if args.verify and do_write:
            print("\nVerification (EEPROM vs saved JSON):")
            for name, entry in calib.items():
                mid = entry.id
                e_min = bus.read2(mid, CTRL_TABLE["Min_Position_Limit"][0])
                e_max = bus.read2(mid, CTRL_TABLE["Max_Position_Limit"][0])
                e_off_raw = bus.read2(mid, CTRL_TABLE["Homing_Offset"][0])
                e_off = decode_sign_magnitude(e_off_raw, SIGN_BITS["Homing_Offset"])
                print(f"{name} ID{mid}:")
                print(f"  JSON:   min={entry.range_min} max={entry.range_max} off={entry.homing_offset}")
                print(f"  EEPROM: min={e_min} max={e_max} off={e_off}")
                print(f"  DIFF:   min={e_min-entry.range_min:+d} max={e_max-entry.range_max:+d} off={e_off-entry.homing_offset:+d}")

    finally:
        # Leave safe state
        try:
            for i in ids:
                bus.torque_enable(i, False)
                bus.lock(i, False)
        except Exception:
            pass
        bus.disconnect()


if __name__ == "__main__":
    main()
