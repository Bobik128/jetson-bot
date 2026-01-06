#!/usr/bin/env python3
"""
STS3215 continuous limit teaching (read continuously, write once at the end)

What it does:
- Scan IDs (PING)
- Loop at --hz reading Present_Position (addr 0x38) for each found servo
- Track learned min/max ticks in RAM (expand when position goes outside range)
- Print live status
- Optional autosave to JSON
- On Ctrl+C (or 'q' + Enter), write learned min/max to each servo once:
    MinAngleLimit addr 0x0B (2 bytes)
    MaxAngleLimit addr 0x09 (2 bytes)
  Then verify by re-reading.

Notes:
- This assumes the Feetech packet format: FF FF ID LEN INST PARAMS... CHK
- Checksum = ~(ID + LEN + INST + sum(PARAMS)) & 0xFF
- Response: FF FF ID LEN ERR PARAMS... CHK

Run:
  python3 teach_limits_end_write.py --port /dev/ttyACM0 --baudrate 1000000 --scan-max 20 --hz 40 --out limits.json
Stop:
  Ctrl+C  (writes limits to servos once and exits)
"""

import argparse
import json
import os
import sys
import time
import select
from typing import Optional, Tuple, Dict, List

import serial

# Instructions
INST_PING  = 0x01
INST_READ  = 0x02
INST_WRITE = 0x03

# Registers (STS series)
ADDR_MAX_ANGLE_LIMIT = 0x09  # 2 bytes
ADDR_MIN_ANGLE_LIMIT = 0x0B  # 2 bytes
ADDR_PRESENT_POS     = 0x38  # 2 bytes


# -----------------------------
# Low-level protocol helpers
# -----------------------------
def checksum(core: bytes) -> int:
    # core includes: ID, LEN, INST/ERR, PARAMS...
    return (~(sum(core) & 0xFF)) & 0xFF


def build_packet(servo_id: int, inst: int, params: bytes) -> bytes:
    length = len(params) + 2  # inst + checksum
    core = bytes([servo_id & 0xFF, length & 0xFF, inst & 0xFF]) + params
    chk = checksum(core)
    return b"\xFF\xFF" + core + bytes([chk])


def read_exact(ser: serial.Serial, n: int, timeout_s: float) -> bytes:
    end = time.time() + timeout_s
    buf = bytearray()
    while len(buf) < n and time.time() < end:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
        else:
            time.sleep(0.0005)
    return bytes(buf)


def read_response(ser: serial.Serial, timeout_s: float) -> Optional[Tuple[int, int, bytes]]:
    """
    Parse one response:
    FF FF ID LEN ERR PARAMS... CHK
    Returns (id, err, params) or None.
    """
    end = time.time() + timeout_s

    # Find header 0xFF 0xFF
    state = 0
    while time.time() < end:
        b = ser.read(1)
        if not b:
            time.sleep(0.0005)
            continue
        if state == 0:
            state = 1 if b == b"\xFF" else 0
        else:
            if b == b"\xFF":
                break
            state = 1 if b == b"\xFF" else 0
    else:
        return None

    hdr = read_exact(ser, 3, timeout_s)  # ID, LEN, ERR
    if len(hdr) != 3:
        return None

    sid, length, err = hdr[0], hdr[1], hdr[2]
    params_len = max(0, length - 2)  # (ERR + CHK) = 2
    params = read_exact(ser, params_len, timeout_s)
    chk_b  = read_exact(ser, 1, timeout_s)

    if len(params) != params_len or len(chk_b) != 1:
        return None

    core = bytes([sid, length, err]) + params
    if checksum(core) != chk_b[0]:
        return None

    return sid, err, params


def transact(ser: serial.Serial, pkt: bytes, expect_id: Optional[int], timeout_s: float) -> Optional[Tuple[int, int, bytes]]:
    # Clear stale bytes; avoids matching an old response
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()

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
    pkt = build_packet(sid, INST_READ, bytes([addr & 0xFF, n & 0xFF]))
    resp = transact(ser, pkt, expect_id=sid, timeout_s=timeout_s)
    if resp is None:
        return None
    _, err, data = resp
    if err != 0 or len(data) != n:
        return None
    return data


def write_bytes(ser: serial.Serial, sid: int, addr: int, data: bytes, timeout_s: float) -> bool:
    pkt = build_packet(sid, INST_WRITE, bytes([addr & 0xFF]) + data)
    resp = transact(ser, pkt, expect_id=sid, timeout_s=timeout_s)
    return resp is not None and resp[1] == 0


def le_u16(b: bytes) -> int:
    return int(b[0]) | (int(b[1]) << 8)


def u16_to_le(v: int) -> bytes:
    v &= 0xFFFF
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def ticks_to_deg(ticks: int) -> float:
    # Common STS position scale: 0..4095 over 360°
    return (ticks % 4096) * 360.0 / 4096.0


# -----------------------------
# Persistence helpers
# -----------------------------
def atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# -----------------------------
# Main
# -----------------------------
def stdin_has_line() -> bool:
    # Non-blocking check for user commands
    r, _, _ = select.select([sys.stdin], [], [], 0.0)
    return bool(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuously teach STS3215 angle limits; write once at exit.")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--scan-min", type=int, default=1)
    ap.add_argument("--scan-max", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=0.05, help="Per transaction timeout (s)")
    ap.add_argument("--hz", type=float, default=40.0, help="Sample rate (Hz)")
    ap.add_argument("--print-hz", type=float, default=10.0, help="Console print rate (Hz)")
    ap.add_argument("--out", default="limits_learned.json", help="Autosave JSON path")
    ap.add_argument("--autosave-s", type=float, default=5.0, help="Autosave interval (s); 0 disables")
    ap.add_argument("--no-autosave", action="store_true", help="Disable autosave")
    ap.add_argument("--write-on-exit", action="store_true", default=True,
                    help="Write learned limits to servos on exit (default: enabled)")
    ap.add_argument("--no-write-on-exit", dest="write_on_exit", action="store_false",
                    help="Do not write to servos on exit (just report/save)")
    args = ap.parse_args()

    dt = 1.0 / max(1e-6, args.hz)
    print_dt = 1.0 / max(1e-6, args.print_hz)

    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.001,      # small read timeout; we handle timeouts ourselves
            write_timeout=0.2,
        )
    except Exception as e:
        print(f"ERROR: Could not open {args.port}: {e}", file=sys.stderr)
        return 2

    with ser:
        # Scan
        found: List[int] = []
        for sid in range(args.scan_min, args.scan_max + 1):
            if ping(ser, sid, args.timeout):
                found.append(sid)

        if not found:
            print(f"No servos found on {args.port} in ID range {args.scan_min}..{args.scan_max}.")
            return 1

        print(f"Found servos: {found}")
        print("Teaching limits now. Move the arm by hand.")
        print("Stop with Ctrl+C, or type 'q' then Enter.")
        print("")

        # Learned limits in ticks: {id: {"min": int, "max": int, "last": int}}
        learned: Dict[int, Dict[str, int]] = {}

        # Initialize from current position (and optionally could read current limits too)
        for sid in found:
            pos_b = read_bytes(ser, sid, ADDR_PRESENT_POS, 2, args.timeout)
            if pos_b is None:
                # still initialize to a safe default
                continue
            pos = le_u16(pos_b)
            learned[sid] = {"min": pos, "max": pos, "last": pos}

        # Timing for printing/autosave
        t_last_print = 0.0
        t_last_save = 0.0
        start = time.time()

        try:
            while True:
                t0 = time.time()

                # user command
                if stdin_has_line():
                    line = sys.stdin.readline().strip().lower()
                    if line in ("q", "quit", "exit"):
                        break

                # sample all servos
                for sid in found:
                    pos_b = read_bytes(ser, sid, ADDR_PRESENT_POS, 2, args.timeout)
                    if pos_b is None:
                        # keep going; bus noise happens
                        continue
                    pos = le_u16(pos_b)

                    if sid not in learned:
                        learned[sid] = {"min": pos, "max": pos, "last": pos}
                    else:
                        if pos < learned[sid]["min"]:
                            learned[sid]["min"] = pos
                        if pos > learned[sid]["max"]:
                            learned[sid]["max"] = pos
                        learned[sid]["last"] = pos

                now = time.time()

                # Print
                if now - t_last_print >= print_dt:
                    t_last_print = now
                    elapsed = now - start
                    print(f"t={elapsed:7.2f}s")
                    for sid in found:
                        if sid not in learned:
                            print(f"  ID {sid:3d}: (no data)")
                            continue
                        mn = learned[sid]["min"]
                        mx = learned[sid]["max"]
                        cur = learned[sid]["last"]
                        print(
                            f"  ID {sid:3d}: "
                            f"CUR={cur:4d} ({ticks_to_deg(cur):7.2f}°)  "
                            f"MIN={mn:4d} ({ticks_to_deg(mn):7.2f}°)  "
                            f"MAX={mx:4d} ({ticks_to_deg(mx):7.2f}°)"
                        )
                    print("")

                # Autosave
                if (not args.no_autosave) and args.autosave_s > 0 and (now - t_last_save >= args.autosave_s):
                    t_last_save = now
                    payload = {
                        "port": args.port,
                        "baudrate": args.baudrate,
                        "ids": found,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "limits_ticks": {str(k): {"min": v["min"], "max": v["max"]} for k, v in learned.items()},
                    }
                    try:
                        atomic_write_json(args.out, payload)
                    except Exception as e:
                        print(f"WARNING: autosave failed: {e}", file=sys.stderr)

                # Rate control
                sleep_s = dt - (time.time() - t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except KeyboardInterrupt:
            pass

        # Final save
        payload = {
            "port": args.port,
            "baudrate": args.baudrate,
            "ids": found,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "limits_ticks": {str(k): {"min": v["min"], "max": v["max"]} for k, v in learned.items()},
        }
        try:
            atomic_write_json(args.out, payload)
            print(f"Saved learned limits to: {args.out}")
        except Exception as e:
            print(f"WARNING: final save failed: {e}", file=sys.stderr)

        # Write once at the end
        if args.write_on_exit:
            print("\nWriting learned limits to servos (once)...")
            for sid in found:
                if sid not in learned:
                    print(f"  ID {sid:3d}: skipped (no learned data)")
                    continue
                mn = int(learned[sid]["min"])
                mx = int(learned[sid]["max"])

                # Basic sanity clamp
                mn = max(0, min(4095, mn))
                mx = max(0, min(4095, mx))
                if mx < mn:
                    mn, mx = mx, mn

                ok_min = write_bytes(ser, sid, ADDR_MIN_ANGLE_LIMIT, u16_to_le(mn), args.timeout)
                ok_max = write_bytes(ser, sid, ADDR_MAX_ANGLE_LIMIT, u16_to_le(mx), args.timeout)

                # Verify
                vmin_b = read_bytes(ser, sid, ADDR_MIN_ANGLE_LIMIT, 2, args.timeout)
                vmax_b = read_bytes(ser, sid, ADDR_MAX_ANGLE_LIMIT, 2, args.timeout)
                vmin = le_u16(vmin_b) if vmin_b else None
                vmax = le_u16(vmax_b) if vmax_b else None

                print(
                    f"  ID {sid:3d}: "
                    f"WRITE(min={mn}, max={mx}) "
                    f"ACK(min={ok_min}, max={ok_max}) "
                    f"VERIFY(min={vmin}, max={vmax})"
                )

            print("Done.")
        else:
            print("\nNot writing to servos (--no-write-on-exit enabled).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
