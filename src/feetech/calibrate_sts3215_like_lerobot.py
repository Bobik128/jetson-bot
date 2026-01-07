#!/usr/bin/env python3
import argparse
import json
import sys
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import scservo_sdk as scs

CTRL = {
    "Return_Delay_Time": (7, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
}

SIGN_BITS = {
    "Homing_Offset": 11,
    "Present_Position": 15,
}

RES = 4096
HALF_TURN = (RES - 1) // 2  # 2047


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


class Bus:
    def __init__(self, port: str, baudrate: int = 1_000_000, protocol: int = 0):
        self.port = port
        self.baudrate = baudrate
        self.port_handler = scs.PortHandler(port)
        self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(self.port_handler, scs.PortHandler)
        self.packet_handler = scs.PacketHandler(protocol)

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
        # writeXByteTxRx: (comm, err) or (something, comm, err)
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

    def write1(self, mid: int, addr: int, val: int):
        ret = self.packet_handler.write1ByteTxRx(self.port_handler, mid, addr, int(val))
        _, comm, err = self._unpack2or3(ret)
        self._check(comm, err, mid, f"write1 addr={addr} val={val}")

    def write2(self, mid: int, addr: int, val: int):
        ret = self.packet_handler.write2ByteTxRx(self.port_handler, mid, addr, int(val))
        _, comm, err = self._unpack2or3(ret)
        self._check(comm, err, mid, f"write2 addr={addr} val={val}")

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

    # High level
    def torque(self, mid: int, enable: bool):
        self.write1(mid, CTRL["Torque_Enable"][0], 1 if enable else 0)

    def unlock(self, mid: int):
        self.write1(mid, CTRL["Lock"][0], 0)

    def position_mode(self, mid: int):
        self.write1(mid, CTRL["Operating_Mode"][0], 0)

    def configure(self, mid: int, return_delay: int, acceleration: int):
        self.write1(mid, CTRL["Return_Delay_Time"][0], int(return_delay))
        self.write1(mid, CTRL["Acceleration"][0], int(acceleration))

    def present_pos(self, mid: int) -> int:
        raw = self.read2(mid, CTRL["Present_Position"][0])
        return decode_sign_magnitude(raw, SIGN_BITS["Present_Position"])

    def write_offset(self, mid: int, offset: int):
        raw = encode_sign_magnitude(int(offset), SIGN_BITS["Homing_Offset"])
        self.write2(mid, CTRL["Homing_Offset"][0], raw)

    def write_limits(self, mid: int, mn: int, mx: int):
        self.write2(mid, CTRL["Min_Position_Limit"][0], int(mn))
        self.write2(mid, CTRL["Max_Position_Limit"][0], int(mx))


def wait_enter(prompt: str) -> threading.Event:
    evt = threading.Event()

    def _run():
        try:
            input(prompt)
        except EOFError:
            pass
        evt.set()

    threading.Thread(target=_run, daemon=True).start()
    return evt


def record_ranges(bus: Bus, ids: List[int], hz: float) -> Tuple[Dict[int, int], Dict[int, int]]:
    mins = {i: 10**9 for i in ids}
    maxs = {i: -10**9 for i in ids}
    dt = 1.0 / max(1.0, hz)

    stop = wait_enter(
        "Move each joint through its full range now.\n"
        "Press ENTER to stop recording...\n"
    )

    while not stop.is_set():
        for i in ids:
            try:
                p = bus.present_pos(i)
            except Exception as e:
                print(f"[WARN] read ID {i}: {e}", file=sys.stderr)
                continue
            if p < mins[i]:
                mins[i] = p
            if p > maxs[i]:
                maxs[i] = p
        time.sleep(dt)

    for i in ids:
        if mins[i] == 10**9 or maxs[i] == -10**9:
            raise RuntimeError(f"No samples recorded for ID {i}")
    return mins, maxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--names", type=str, default=None)
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--hz", type=float, default=50.0)

    ap.add_argument("--return-delay", type=int, default=0)
    ap.add_argument("--acceleration", type=int, default=254)

    ap.add_argument("--write", action="store_true", help="Write to EEPROM (default)")
    ap.add_argument("--no-write", action="store_true")

    ap.add_argument("--mode", choices=["offset_first", "record_first_autofix"], default="offset_first",
                    help="offset_first: write offset before recording ranges (recommended). "
                         "record_first_autofix: record ranges before writing offset but auto-shift limits when writing.")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    do_write = not args.no_write
    if args.write:
        do_write = True

    ids = args.ids
    if args.names:
        names = [s.strip() for s in args.names.split(",")]
        if len(names) != len(ids):
            raise ValueError("--names count must equal --ids")
        name_by_id = {i: n for i, n in zip(ids, names)}
    else:
        name_by_id = {i: f"id_{i}" for i in ids}

    bus = Bus(args.port, args.baudrate, protocol=0)
    bus.connect()

    try:
        # Safe config
        for i in ids:
            bus.torque(i, False)
            bus.unlock(i)
            bus.configure(i, args.return_delay, args.acceleration)
            bus.position_mode(i)

        input(
            "\nPut all joints at MID-RANGE pose.\n"
            "Press ENTER to capture mid pose...\n"
        )

        mid_pos = {i: bus.present_pos(i) for i in ids}
        offsets = {i: mid_pos[i] - HALF_TURN for i in ids}

        print("\nCaptured mid pose:")
        for i in ids:
            print(f"  ID {i}: present={mid_pos[i]} -> homing_offset={offsets[i]}")

        if args.mode == "offset_first":
            if do_write:
                print("\nWriting Homing_Offset first (recommended)...")
                for i in ids:
                    bus.torque(i, False)
                    bus.unlock(i)
                    bus.write_offset(i, offsets[i])
                time.sleep(0.2)

                print("Present after offset write (should be near 2047 if you didn't move):")
                for i in ids:
                    print(f"  ID {i}: present_now={bus.present_pos(i)}")

            print("\nNow record ranges in the SAME frame as EEPROM enforcement...")
            mins, maxs = record_ranges(bus, ids, args.hz)

            mins_write, maxs_write = mins, maxs

        else:
            print("\nRecording ranges BEFORE writing offsets (will auto-fix limits on write)...")
            mins, maxs = record_ranges(bus, ids, args.hz)

            # Auto-fix: shift limits into the new Present_Position frame after applying offset
            mins_write = {i: mins[i] - offsets[i] for i in ids}
            maxs_write = {i: maxs[i] - offsets[i] for i in ids}

            if do_write:
                print("\nWriting Homing_Offset...")
                for i in ids:
                    bus.torque(i, False)
                    bus.unlock(i)
                    bus.write_offset(i, offsets[i])
                time.sleep(0.2)

        print("\nFinal ranges to write:")
        for i in ids:
            print(f"  ID {i}: min={mins_write[i]} max={maxs_write[i]}")

        calib: Dict[str, CalibEntry] = {}
        for i in ids:
            calib[name_by_id[i]] = CalibEntry(
                id=i,
                drive_mode=0,
                homing_offset=offsets[i],
                range_min=mins_write[i],
                range_max=maxs_write[i],
            )

        if do_write:
            print("\nWriting Min/Max limits...")
            for i in ids:
                bus.torque(i, False)
                bus.unlock(i)
                bus.write_limits(i, mins_write[i], maxs_write[i])
            print("EEPROM written.")
            print("Recommendation: power-cycle servo power to ensure enforcement updates cleanly.")

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in calib.items()}, f, indent=4)
        print(f"\nSaved calibration JSON to: {args.out}")

        if args.verify and do_write:
            print("\nVerify EEPROM:")
            for i in ids:
                mn = bus.read2(i, CTRL["Min_Position_Limit"][0])
                mx = bus.read2(i, CTRL["Max_Position_Limit"][0])
                off_raw = bus.read2(i, CTRL["Homing_Offset"][0])
                off = decode_sign_magnitude(off_raw, SIGN_BITS["Homing_Offset"])
                print(f"  ID {i}: EEPROM min={mn} max={mx} offset={off}")

    finally:
        try:
            for i in ids:
                bus.torque(i, False)
                bus.unlock(i)
        except Exception:
            pass
        bus.disconnect()


if __name__ == "__main__":
    main()
