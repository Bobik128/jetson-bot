#!/usr/bin/env python3
"""
Feetech STS/SCS serial-bus servo control via scservo_sdk (no LeRobot).

Tested assumptions / common STS map:
- ID register address = 5
- Goal position write block starts at 0x2A:
    [pos_L, pos_H, time_L, time_H, speed_L, speed_H]
- Present position read at 0x38 (2 bytes, low then high)
- Min angle limit at address 9, Max angle limit at address 11 (often 2 bytes each)
- protocol_end = 0 for STS/SMS, protocol_end = 1 for SCS (common convention)

IMPORTANT:
- If multiple servos share the same current ID, you must connect/configure one at a time.
- Some servos require disabling an EEPROM lock before ID changes will persist across power cycles.
"""

from __future__ import annotations
import argparse
import sys
from typing import List, Tuple

from scservo_sdk import PortHandler, PacketHandler  # provided by feetech-servo-sdk / ftservo-python-sdk

# ----------------------------
# Control table addresses
# ----------------------------
ADDR_ID                 = 5

ADDR_MIN_ANGLE_LIMIT    = 9     # from Feetech tutorial doc
ADDR_MAX_ANGLE_LIMIT    = 11    # from Feetech tutorial doc

ADDR_GOAL_POS_BLOCK     = 0x2A  # from protocol manual example
LEN_GOAL_POS_BLOCK      = 6     # pos(2) + time(2) + speed(2)

ADDR_PRESENT_POSITION   = 0x38  # from protocol manual example (2 bytes, low then high)
LEN_PRESENT_POSITION    = 2

# Some commonly-used runtime registers (seen in many STS examples)
ADDR_TORQUE_ENABLE      = 40    # often used for STS/SCS series
TORQUE_ON               = 1
TORQUE_OFF              = 0


# ----------------------------
# Helpers
# ----------------------------
def lo(x: int) -> int:
    return x & 0xFF

def hi(x: int) -> int:
    return (x >> 8) & 0xFF

def clamp(v: int, vmin: int, vmax: int) -> int:
    return max(vmin, min(vmax, v))

def deg_to_sts_units(deg: float) -> int:
    """
    STS “360° / 4096 steps” convention is common for ST3215/STS3215-class devices.
    0..4095 corresponds to 0..360 degrees, with 2048 near mid.
    """
    # Map degrees to [0, 4095]
    units = int(round((deg % 360.0) * (4096.0 / 360.0)))
    return clamp(units, 0, 4095)

def open_bus(port: str, baudrate: int, protocol_end: int) -> Tuple[PortHandler, PacketHandler]:
    port_handler = PortHandler(port)
    packet_handler = PacketHandler(protocol_end)

    if not port_handler.openPort():
        raise RuntimeError(f"Failed to open port: {port}")

    if not port_handler.setBaudRate(baudrate):
        raise RuntimeError(f"Failed to set baudrate {baudrate} on {port}")

    return port_handler, packet_handler

def close_bus(port_handler: PortHandler) -> None:
    try:
        port_handler.closePort()
    except Exception:
        pass

def ping(packet: PacketHandler, port: PortHandler, motor_id: int) -> bool:
    model_number, comm_result, error = packet.ping(port, motor_id)
    return comm_result == 0 and error == 0

def read_u16(packet: PacketHandler, port: PortHandler, motor_id: int, addr: int) -> int:
    val, comm_result, error = packet.read2ByteTxRx(port, motor_id, addr)
    if comm_result != 0 or error != 0:
        raise RuntimeError(f"read_u16 failed: id={motor_id} addr=0x{addr:02X} comm={comm_result} err={error}")
    return int(val)

def write_u8(packet: PacketHandler, port: PortHandler, motor_id: int, addr: int, value: int) -> None:
    comm_result, error = packet.write1ByteTxRx(port, motor_id, addr, int(value) & 0xFF)
    if comm_result != 0 or error != 0:
        raise RuntimeError(f"write_u8 failed: id={motor_id} addr=0x{addr:02X} comm={comm_result} err={error}")

def write_u16(packet: PacketHandler, port: PortHandler, motor_id: int, addr: int, value: int) -> None:
    comm_result, error = packet.write2ByteTxRx(port, motor_id, addr, int(value) & 0xFFFF)
    if comm_result != 0 or error != 0:
        raise RuntimeError(f"write_u16 failed: id={motor_id} addr=0x{addr:02X} comm={comm_result} err={error}")

def write_goal_pos_block(packet: PacketHandler, port: PortHandler, motor_id: int,
                         pos: int, time_ms: int, speed: int) -> None:
    """
    Write 6 bytes to 0x2A: position (u16), time (u16), speed (u16)
    Based on Feetech protocol manual example.
    """
    # SDK provides writeTxRx for arbitrary blocks as writeNByteTxRx in some variants;
    # if unavailable, you can fall back to sequential writes.
    data = [lo(pos), hi(pos), lo(time_ms), hi(time_ms), lo(speed), hi(speed)]

    # Try to use writeNByteTxRx if present; else do sequential writes.
    if hasattr(packet, "writeTxRx"):
        # Some forks expose writeTxRx(port, id, addr, length, data)
        comm_result, error = packet.writeTxRx(port, motor_id, ADDR_GOAL_POS_BLOCK, LEN_GOAL_POS_BLOCK, data)
        if comm_result != 0 or error != 0:
            raise RuntimeError(f"write_goal_pos_block failed: comm={comm_result} err={error}")
    elif hasattr(packet, "writeNByteTxRx"):
        comm_result, error = packet.writeNByteTxRx(port, motor_id, ADDR_GOAL_POS_BLOCK, LEN_GOAL_POS_BLOCK, data)
        if comm_result != 0 or error != 0:
            raise RuntimeError(f"write_goal_pos_block failed: comm={comm_result} err={error}")
    else:
        # Fallback: write as three u16 values
        write_u16(packet, port, motor_id, ADDR_GOAL_POS_BLOCK + 0, pos)
        write_u16(packet, port, motor_id, ADDR_GOAL_POS_BLOCK + 2, time_ms)
        write_u16(packet, port, motor_id, ADDR_GOAL_POS_BLOCK + 4, speed)


# ----------------------------
# Commands
# ----------------------------
def cmd_scan(args) -> int:
    port, baud, proto = args.port, args.baudrate, args.protocol_end
    ph, pk = open_bus(port, baud, proto)
    try:
        found = []
        for mid in range(args.start_id, args.end_id + 1):
            if ping(pk, ph, mid):
                found.append(mid)
        if found:
            print("Found servos:", found)
            return 0
        print("No servos found in range.")
        return 1
    finally:
        close_bus(ph)

def cmd_set_id(args) -> int:
    ph, pk = open_bus(args.port, args.baudrate, args.protocol_end)
    try:
        # Disable torque (recommended)
        try:
            write_u8(pk, ph, args.current_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
        except Exception:
            pass

        # Write new ID
        write_u8(pk, ph, args.current_id, ADDR_ID, args.new_id)

        # Verify by pinging the new ID
        if not ping(pk, ph, args.new_id):
            print("Wrote ID, but could not ping new ID. If multiple servos shared the old ID, configure one at a time.")
            return 2

        print(f"OK: ID changed {args.current_id} -> {args.new_id}")
        print("Note: some servos require disabling an EEPROM lock for ID changes to persist after power-off.")
        return 0
    finally:
        close_bus(ph)

def cmd_set_bounds(args) -> int:
    ph, pk = open_bus(args.port, args.baudrate, args.protocol_end)
    try:
        # Bounds are often stored as u16 units (0..4095) for STS/ST class devices.
        min_u = deg_to_sts_units(args.min_deg)
        max_u = deg_to_sts_units(args.max_deg)

        # Disable torque (recommended)
        try:
            write_u8(pk, ph, args.id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
        except Exception:
            pass

        # Write limits
        write_u16(pk, ph, args.id, ADDR_MIN_ANGLE_LIMIT, min_u)
        write_u16(pk, ph, args.id, ADDR_MAX_ANGLE_LIMIT, max_u)

        # Re-enable torque
        try:
            write_u8(pk, ph, args.id, ADDR_TORQUE_ENABLE, TORQUE_ON)
        except Exception:
            pass

        # Readback (optional)
        rb_min = read_u16(pk, ph, args.id, ADDR_MIN_ANGLE_LIMIT)
        rb_max = read_u16(pk, ph, args.id, ADDR_MAX_ANGLE_LIMIT)
        print(f"OK: bounds set for ID={args.id}: min={rb_min} max={rb_max} (units 0..4095)")
        return 0
    finally:
        close_bus(ph)

def cmd_move(args) -> int:
    ph, pk = open_bus(args.port, args.baudrate, args.protocol_end)
    try:
        pos_u = deg_to_sts_units(args.deg)

        # Enable torque
        try:
            write_u8(pk, ph, args.id, ADDR_TORQUE_ENABLE, TORQUE_ON)
        except Exception:
            pass

        # Write goal position block (pos/time/speed) at 0x2A
        # time_ms and speed are protocol-level fields; 0 is commonly accepted for "use internal default"
        write_goal_pos_block(pk, ph, args.id, pos_u, args.time_ms, args.speed)

        if args.readback:
            present = read_u16(pk, ph, args.id, ADDR_PRESENT_POSITION)
            print(f"Commanded: deg={args.deg} -> pos={pos_u}; Present_Position={present}")
        else:
            print(f"Commanded: deg={args.deg} -> pos={pos_u}")

        return 0
    finally:
        close_bus(ph)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feetech STS/SCS servo CLI using scservo_sdk (no LeRobot)")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baudrate", type=int, default=1_000_000)
    p.add_argument("--protocol-end", type=int, default=0,
                   help="0 for STS/SMS, 1 for SCS (common convention). Default 0.")

    sub = p.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Ping a range of IDs")
    scan.add_argument("--start-id", type=int, default=1)
    scan.add_argument("--end-id", type=int, default=20)
    scan.set_defaults(func=cmd_scan)

    sid = sub.add_parser("set-id", help="Change a servo ID")
    sid.add_argument("--current-id", type=int, required=True)
    sid.add_argument("--new-id", type=int, required=True)
    sid.set_defaults(func=cmd_set_id)

    bnd = sub.add_parser("set-bounds", help="Set min/max angle limits (degrees) (stored as 0..4095 units)")
    bnd.add_argument("--id", type=int, required=True)
    bnd.add_argument("--min-deg", type=float, required=True)
    bnd.add_argument("--max-deg", type=float, required=True)
    bnd.set_defaults(func=cmd_set_bounds)

    mv = sub.add_parser("move", help="Move to an absolute angle (degrees)")
    mv.add_argument("--id", type=int, required=True)
    mv.add_argument("--deg", type=float, required=True)
    mv.add_argument("--time-ms", type=int, default=0, help="Move time field in the 0x2A block (u16). Default 0.")
    mv.add_argument("--speed", type=int, default=1000, help="Speed field in the 0x2A block (u16). Default 1000.")
    mv.add_argument("--readback", action="store_true")
    mv.set_defaults(func=cmd_move)

    return p

def main() -> int:
    p = build_parser()
    args = p.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
