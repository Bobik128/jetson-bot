#!/usr/bin/env python3
"""
Feetech STS3215 limit tool (serial bus)

Reads:
- Max angle limit  (addr 9)   [2 bytes, little-endian]
- Min angle limit  (addr 11)  [2 bytes, little-endian]
- Present position (addr 0x38)[2 bytes, little-endian]

Optionally writes new min/max limits and verifies by re-reading.

Protocol:
FF FF ID LEN INST PARAMS... CHK
Response:
FF FF ID LEN ERR  PARAMS... CHK
Checksum = ~(ID + LEN + INST + sum(PARAMS)) & 0xFF

References:
- Present position read example uses address 0x38 and low byte then high byte. :contentReference[oaicite:1]{index=1}
- Angle limit addresses 9 and 11 are referenced as max/min angles. :contentReference[oaicite:2]{index=2}
"""

import argparse
import sys
import time
from typing import Optional, Tuple, List

import serial


# -----------------------------
# Feetech protocol constants
# -----------------------------
INST_PING  = 0x01
INST_READ  = 0x02
INST_WRITE = 0x03

ADDR_MAX_ANGLE_LIMIT = 0x09  # 2 bytes
ADDR_MIN_ANGLE_LIMIT = 0x0B  # 2 bytes
ADDR_PRESENT_POS     = 0x38  # 2 bytes


def checksum(pkt_wo_checksum: bytes) -> int:
    # pkt_wo_checksum includes: ID, LEN, INST, PARAMS...
    s = sum(pkt_wo_checksum) & 0xFF
    return (~s) & 0xFF


def build_packet(servo_id: int, inst: int, params: bytes) -> bytes:
    length = len(params) + 2  # instruction + checksum
    core = bytes([servo_id & 0xFF, length & 0xFF, inst & 0xFF]) + params
    chk = checksum(core)
    return b"\xFF\xFF" + core + bytes([chk])


def read_exact(ser: serial.Serial, n: int, timeout_s: float) -> bytes:
    """Read exactly n bytes or return fewer if timeout."""
    end = time.time() + timeout_s
    buf = bytearray()
    while len(buf) < n and time.time() < end:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
        else:
            time.sleep(0.001)
    return bytes(buf)


def read_response(ser: serial.Serial, timeout_s: float) -> Optional[Tuple[int, int, bytes]]:
    """
    Parse one response frame:
    FF FF ID LEN ERR PARAMS... CHK
    Returns (id, err, params) or None on timeout/invalid.
    """
    end = time.time() + timeout_s
    # Find header 0xFF 0xFF
    state = 0
    while time.time() < end:
        b = ser.read(1)
        if not b:
            time.sleep(0.001)
            continue
        if state == 0:
            state = 1 if b == b"\xFF" else 0
        elif state == 1:
            if b == b"\xFF":
                break
            state = 1 if b == b"\xFF" else 0
    else:
        return None

    hdr = read_exact(ser, 3, timeout_s)  # ID, LEN, ERR
    if len(hdr) != 3:
        return None
    sid, length, err = hdr[0], hdr[1], hdr[2]
    # length includes ERR? In Feetech reply, LENGTH = PARAMS + 2 (ERR + CHK)
    params_len = max(0, length - 2)
    params = read_exact(ser, params_len, timeout_s)
    chk_b = read_exact(ser, 1, timeout_s)
    if len(params) != params_len or len(chk_b) != 1:
        return None

    # Validate checksum
    core = bytes([sid, length, err]) + params
    if checksum(core) != chk_b[0]:
        return None

    return sid, err, params


def transact(ser: serial.Serial, pkt: bytes, expect_id: Optional[int], timeout_s: float) -> Optional[Tuple[int, int, bytes]]:
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    # Read until we get a valid response (matching ID if specified) or timeout
    end = time.time() + timeout_s
    while time.time() < end:
        resp = read_response(ser, timeout_s=max(0.0, end - time.time()))
        if resp is None:
            return None
        sid, err, params = resp
        if expect_id is None or sid == expect_id:
            return resp
    return None


def ping(ser: serial.Serial, sid: int, timeout_s: float) -> bool:
    pkt = build_packet(sid, INST_PING, b"")
    resp = transact(ser, pkt, expect_id=sid, timeout_s=timeout_s)
    return resp is not None and resp[1] == 0


def read_bytes(ser: serial.Serial, sid: int, addr: int, n: int, timeout_s: float) -> Optional[bytes]:
    params = bytes([addr & 0xFF, n & 0xFF])
    pkt = build_packet(sid, INST_READ, params)
    resp = transact(ser, pkt, expect_id=sid, timeout_s=timeout_s)
    if resp is None:
        return None
    _, err, data = resp
    if err != 0 or len(data) != n:
        return None
    return data


def write_bytes(ser: serial.Serial, sid: int, addr: int, data: bytes, timeout_s: float) -> bool:
    params = bytes([addr & 0xFF]) + data
    pkt = build_packet(sid, INST_WRITE, params)
    resp = transact(ser, pkt, expect_id=sid, timeout_s=timeout_s)
    return resp is not None and resp[1] == 0


def le_u16(b: bytes) -> int:
    return int(b[0]) | (int(b[1]) << 8)


def u16_to_le(v: int) -> bytes:
    v &= 0xFFFF
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def ticks_to_deg(ticks: int) -> float:
    # STS3215 uses 0..4095 over ~360 degrees (12-bit magnetic encoder typical)
    return (ticks % 4096) * 360.0 / 4096.0


def deg_to_ticks(deg: float) -> int:
    # Clamp into [0, 360)
    d = deg % 360.0
    return int(round(d * 4096.0 / 360.0)) % 4096


def read_status(ser: serial.Serial, sid: int, timeout_s: float) -> Optional[dict]:
    max_b = read_bytes(ser, sid, ADDR_MAX_ANGLE_LIMIT, 2, timeout_s)
    min_b = read_bytes(ser, sid, ADDR_MIN_ANGLE_LIMIT, 2, timeout_s)
    pos_b = read_bytes(ser, sid, ADDR_PRESENT_POS,     2, timeout_s)
    if max_b is None or min_b is None or pos_b is None:
        return None

    max_ticks = le_u16(max_b)
    min_ticks = le_u16(min_b)
    pos_ticks = le_u16(pos_b)

    return {
        "id": sid,
        "max_ticks": max_ticks,
        "min_ticks": min_ticks,
        "pos_ticks": pos_ticks,
        "max_deg": ticks_to_deg(max_ticks),
        "min_deg": ticks_to_deg(min_ticks),
        "pos_deg": ticks_to_deg(pos_ticks),
    }


def main():
    ap = argparse.ArgumentParser(description="Read/write STS3215 min/max angle limits over serial.")
    ap.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    ap.add_argument("--baudrate", type=int, default=1_000_000, help="Baudrate (default: 1000000)")
    ap.add_argument("--scan-min", type=int, default=1, help="Min ID to scan (default: 1)")
    ap.add_argument("--scan-max", type=int, default=20, help="Max ID to scan (default: 20)")
    ap.add_argument("--timeout", type=float, default=0.05, help="Per-transaction timeout seconds (default: 0.05)")

    # Optional write
    ap.add_argument("--set-id", type=int, help="Servo ID to set limits for")
    ap.add_argument("--min-deg", type=float, help="New min limit in degrees (0..360)")
    ap.add_argument("--max-deg", type=float, help="New max limit in degrees (0..360)")
    ap.add_argument("--min-ticks", type=int, help="New min limit in ticks (0..4095), overrides --min-deg")
    ap.add_argument("--max-ticks", type=int, help="New max limit in ticks (0..4095), overrides --max-deg")

    args = ap.parse_args()

    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.001,   # non-blocking-ish reads; we manage timing ourselves
            write_timeout=0.2
        )
    except Exception as e:
        print(f"ERROR: could not open {args.port}: {e}", file=sys.stderr)
        sys.exit(2)

    with ser:
        # Scan
        found: List[int] = []
        for sid in range(args.scan_min, args.scan_max + 1):
            if ping(ser, sid, args.timeout):
                found.append(sid)

        if not found:
            print(f"No servos found on {args.port} in ID range {args.scan_min}..{args.scan_max}.")
            print("Notes:")
            print("- Ensure external servo power is connected and ground is common with the USB adapter.")
            print("- Verify baudrate (many Feetech servos default to 1,000,000).")
            sys.exit(1)

        # Print current status for each found servo
        print(f"Found servos: {found}")
        for sid in found:
            st = read_status(ser, sid, args.timeout)
            if st is None:
                print(f"ID {sid:3d}: read failed")
                continue
            print(
                f"ID {sid:3d}: "
                f"MIN={st['min_ticks']:4d} ({st['min_deg']:7.2f}°)  "
                f"MAX={st['max_ticks']:4d} ({st['max_deg']:7.2f}°)  "
                f"POS={st['pos_ticks']:4d} ({st['pos_deg']:7.2f}°)"
            )

        # Optional write
        if args.set_id is not None:
            sid = args.set_id
            if sid not in found:
                print(f"\nERROR: --set-id {sid} was not found during scan.", file=sys.stderr)
                sys.exit(3)

            if args.min_ticks is not None:
                new_min = int(args.min_ticks)
            elif args.min_deg is not None:
                new_min = deg_to_ticks(args.min_deg)
            else:
                new_min = None

            if args.max_ticks is not None:
                new_max = int(args.max_ticks)
            elif args.max_deg is not None:
                new_max = deg_to_ticks(args.max_deg)
            else:
                new_max = None

            if new_min is None or new_max is None:
                print("\nERROR: To set limits you must provide BOTH min and max (use --min-deg/--max-deg or --min-ticks/--max-ticks).", file=sys.stderr)
                sys.exit(4)

            if not (0 <= new_min <= 4095 and 0 <= new_max <= 4095):
                print("\nERROR: min/max ticks must be in 0..4095.", file=sys.stderr)
                sys.exit(5)

            print(f"\nWriting limits on ID {sid}: MIN={new_min} ({ticks_to_deg(new_min):.2f}°), MAX={new_max} ({ticks_to_deg(new_max):.2f}°)")
            ok1 = write_bytes(ser, sid, ADDR_MIN_ANGLE_LIMIT, u16_to_le(new_min), args.timeout)
            ok2 = write_bytes(ser, sid, ADDR_MAX_ANGLE_LIMIT, u16_to_le(new_max), args.timeout)

            if not (ok1 and ok2):
                print("WARNING: write did not get a clean ACK.")
                print("If values do not persist after power-cycle, the servo's EEPROM lock/protection may be enabled.")
            else:
                # verify
                st = read_status(ser, sid, args.timeout)
                if st is None:
                    print("Wrote limits, but verify read failed.")
                else:
                    print(
                        f"Verified ID {sid}: "
                        f"MIN={st['min_ticks']} ({st['min_deg']:.2f}°)  "
                        f"MAX={st['max_ticks']} ({st['max_deg']:.2f}°)  "
                        f"POS={st['pos_ticks']} ({st['pos_deg']:.2f}°)"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
