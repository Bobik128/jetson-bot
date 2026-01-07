#!/usr/bin/env python3
"""
Set "center" (zero) on Feetech STS3215 by writing POSITION_CORRECTION so that
the *current physical position* becomes logical center (2047).

Port: /dev/ttyACM0 (default)
Baud: 1_000_000 (default)

Registers (STS):
  POSITION_CORRECTION = 0x1F (2 bytes, signed)
  WRITE_LOCK          = 0x37 (1 byte) 0=unlock, 1=lock
  CURRENT_POSITION    = 0x38 (2 bytes, unsigned)
  TORQUE_SWITCH       = 0x28 (1 byte) 0=off, 1=on (optional)

Protocol: Dynamixel v1-like (0x02 READ, 0x03 WRITE)
"""

from __future__ import annotations
import argparse
import struct
import sys
import time
from typing import Optional, Tuple, List

import serial

# ---- STS register map (from STS_Servos library) ----
REG_POSITION_CORRECTION = 0x1F
REG_TORQUE_SWITCH = 0x28
REG_WRITE_LOCK = 0x37
REG_CURRENT_POSITION = 0x38

# ---- Instructions ----
INST_READ = 0x02
INST_WRITE = 0x03

CENTER_COUNT = 2047  # "middle" for 0..4095


def checksum(packet_wo_checksum: bytes) -> int:
    # checksum is ~sum(ID..last_param) & 0xFF (excluding 0xFF 0xFF header)
    s = sum(packet_wo_checksum[2:]) & 0xFF
    return (~s) & 0xFF


def build_packet(servo_id: int, instruction: int, params: bytes) -> bytes:
    if not (0 <= servo_id <= 253):
        raise ValueError(f"Invalid servo_id {servo_id}")
    length = len(params) + 2  # instruction + params + checksum
    pkt_wo_chk = bytes([0xFF, 0xFF, servo_id, length, instruction]) + params
    chk = checksum(pkt_wo_chk)
    return pkt_wo_chk + bytes([chk])


def read_status_packet(ser: serial.Serial, timeout_s: float = 0.08) -> Optional[Tuple[int, int, bytes]]:
    """
    Returns (id, error, params) or None on timeout/bad packet.
    """
    deadline = time.time() + timeout_s

    # Find header 0xFF 0xFF
    buf = bytearray()
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if len(buf) >= 2 and buf[-2] == 0xFF and buf[-1] == 0xFF:
            break
        if len(buf) > 64:
            buf = buf[-2:]  # keep tail
    else:
        return None

    # Now read id, length, error
    hdr_rest = ser.read(3)
    if len(hdr_rest) != 3:
        return None
    sid = hdr_rest[0]
    length = hdr_rest[1]
    err = hdr_rest[2]

    # length includes error+params+checksum => params_len = length - 2
    params_len = length - 2
    payload = ser.read(params_len + 1)  # params + checksum
    if len(payload) != params_len + 1:
        return None
    params = payload[:-1]
    chk = payload[-1]

    # Validate checksum
    pkt_wo_chk = bytes([0xFF, 0xFF, sid, length, err]) + params
    expected = checksum(pkt_wo_chk)
    if chk != expected:
        return None

    return sid, err, params


def write_byte(ser: serial.Serial, servo_id: int, address: int, value: int) -> bool:
    params = bytes([address, value & 0xFF])
    pkt = build_packet(servo_id, INST_WRITE, params)
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    st = read_status_packet(ser)
    return st is not None and st[0] == servo_id and st[1] == 0


def write_i16_le(ser: serial.Serial, servo_id: int, address: int, value: int) -> bool:
    # signed 16-bit little endian
    if not (-32768 <= value <= 32767):
        raise ValueError("value out of int16 range")
    lohi = struct.pack("<h", int(value))
    params = bytes([address]) + lohi
    pkt = build_packet(servo_id, INST_WRITE, params)
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    st = read_status_packet(ser)
    return st is not None and st[0] == servo_id and st[1] == 0


def read_u16_le(ser: serial.Serial, servo_id: int, address: int) -> Optional[int]:
    # READ params: [address, size]
    params = bytes([address, 2])
    pkt = build_packet(servo_id, INST_READ, params)
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    st = read_status_packet(ser)
    if st is None:
        return None
    sid, err, data = st
    if sid != servo_id or err != 0 or len(data) < 2:
        return None
    return struct.unpack("<H", data[:2])[0]


def center_one_servo(
    ser: serial.Serial,
    servo_id: int,
    torque_off: bool,
    torque_on_after: bool,
) -> bool:
    if torque_off:
        if not write_byte(ser, servo_id, REG_TORQUE_SWITCH, 0):
            print(f"[ID {servo_id}] Failed to torque OFF", file=sys.stderr)
            return False
        time.sleep(0.01)

    pos = read_u16_le(ser, servo_id, REG_CURRENT_POSITION)
    if pos is None:
        print(f"[ID {servo_id}] Failed to read current position", file=sys.stderr)
        return False

    # Compute offset so that current position becomes CENTER_COUNT.
    # STS position is 0..4095; POSITION_CORRECTION is signed.
    offset = int(CENTER_COUNT) - int(pos)

    print(f"[ID {servo_id}] current_pos={pos} => writing POSITION_CORRECTION={offset}")

    # Unlock EEPROM (WRITE_LOCK = 0)
    if not write_byte(ser, servo_id, REG_WRITE_LOCK, 0):
        print(f"[ID {servo_id}] Failed to unlock EEPROM (WRITE_LOCK=0)", file=sys.stderr)
        return False
    time.sleep(0.01)

    # Write offset
    if not write_i16_le(ser, servo_id, REG_POSITION_CORRECTION, offset):
        print(f"[ID {servo_id}] Failed to write POSITION_CORRECTION", file=sys.stderr)
        # try to relock anyway
        write_byte(ser, servo_id, REG_WRITE_LOCK, 1)
        return False
    time.sleep(0.01)

    # Lock EEPROM (WRITE_LOCK = 1)
    if not write_byte(ser, servo_id, REG_WRITE_LOCK, 1):
        print(f"[ID {servo_id}] Failed to lock EEPROM (WRITE_LOCK=1)", file=sys.stderr)
        return False

    if torque_on_after:
        time.sleep(0.01)
        if not write_byte(ser, servo_id, REG_TORQUE_SWITCH, 1):
            print(f"[ID {servo_id}] Failed to torque ON", file=sys.stderr)
            return False

    # Read back current pos again (sanity check; may change depending on firmware behavior)
    pos2 = read_u16_le(ser, servo_id, REG_CURRENT_POSITION)
    if pos2 is not None:
        print(f"[ID {servo_id}] done. current_pos_now={pos2} (power-cycle recommended)")
    else:
        print(f"[ID {servo_id}] done. (power-cycle recommended)")

    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Set current pose as center (2047) for Feetech STS3215.")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ids", type=int, nargs="+", required=True, help="Servo IDs to center (e.g. --ids 1 2 3)")
    ap.add_argument("--torque-off", action="store_true", help="Torque OFF before centering each servo")
    ap.add_argument("--torque-on-after", action="store_true", help="Torque ON after centering each servo")
    ap.add_argument("--timeout", type=float, default=0.02, help="Serial read timeout seconds")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            timeout=args.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
    except Exception as e:
        print(f"Failed to open {args.port}: {e}", file=sys.stderr)
        return 2

    ok_all = True
    try:
        # small settle
        time.sleep(0.05)

        for sid in args.ids:
            ok = center_one_servo(
                ser=ser,
                servo_id=int(sid),
                torque_off=bool(args.torque_off),
                torque_on_after=bool(args.torque_on_after),
            )
            ok_all = ok_all and ok
            time.sleep(0.05)

    finally:
        try:
            ser.close()
        except Exception:
            pass

    if ok_all:
        print("All done. For persistent behavior, power-cycle the servos/controller now.")
        return 0
    else:
        print("Completed with errors (see stderr).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
