#!/usr/bin/env python3
"""
Feetech (STS3215 / scs_series) control via LeRobot FeetechMotorsBus on Jetson.

Supports:
  - Set motor ID (requires you can uniquely address the current ID)
  - Set angle limits (bounds) in degrees
  - Move to a target position in degrees (servo/position mode)

Notes:
  - Use one motor at a time when changing IDs if multiple motors share the same ID.
  - Default bus baudrate used by LeRobot is typically 1_000_000.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, TorqueMode


# --- Helpers ---
def degrees_to_raw_steps(deg: float, resolution: int = 4096) -> int:
    """
    Convert degrees in [-180, +180] (nominal) to Feetech raw step offset around center.
    LeRobot uses a centered degree convention for calibration, but for raw limits we keep it simple:
      center = resolution/2
      steps_per_deg = resolution/360
      raw = center + deg * steps_per_deg
    Then clamp to [0, resolution-1].
    """
    center = resolution // 2
    raw = int(round(center + deg * (resolution / 360.0)))
    return max(0, min(resolution - 1, raw))


def build_single_motor_bus(port: str, motor_id: int, model: str, name: str = "m") -> FeetechMotorsBus:
    cfg = FeetechMotorsBusConfig(
        port=port,
        motors={name: (motor_id, model)},
        mock=False,
    )
    return FeetechMotorsBus(cfg)


@dataclass(frozen=True)
class BusParams:
    port: str
    baudrate: int
    model: str
    motor_name: str


def connect_bus(bus: FeetechMotorsBus, baudrate: int) -> None:
    bus.connect()
    # LeRobot exposes a bus-level baudrate setter.
    bus.set_bus_baudrate(baudrate)


# --- Commands ---
def cmd_set_id(args: argparse.Namespace, bp: BusParams) -> int:
    bus = build_single_motor_bus(bp.port, args.current_id, bp.model, bp.motor_name)
    connect_bus(bus, bp.baudrate)

    try:
        # Verify we can talk to the motor we think we are addressing
        present_id = int(bus.read("ID")[0])
        if present_id != args.current_id:
            print(f"[WARN] Read back ID={present_id}, expected {args.current_id}. Continuing anyway.")

        # Recommended: disable torque before changing persistent config
        bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Write new ID
        bus.write("ID", int(args.new_id))

        # Read back (still addressed via old mapping in this bus object; reconnect with new ID)
        bus.disconnect()

        verify_bus = build_single_motor_bus(bp.port, args.new_id, bp.model, bp.motor_name)
        connect_bus(verify_bus, bp.baudrate)
        verified_id = int(verify_bus.read("ID")[0])
        verify_bus.disconnect()

        if verified_id != args.new_id:
            print(f"[ERROR] Failed to set ID. Read back {verified_id}, expected {args.new_id}.")
            return 2

        print(f"[OK] Motor ID changed: {args.current_id} -> {args.new_id}")
        return 0

    finally:
        try:
            if getattr(bus, "is_connected", False):
                bus.disconnect()
        except Exception:
            pass


def cmd_set_bounds(args: argparse.Namespace, bp: BusParams) -> int:
    bus = build_single_motor_bus(bp.port, args.id, bp.model, bp.motor_name)
    connect_bus(bus, bp.baudrate)

    try:
        # For sts3215, LeRobot declares resolution 4096 in its table.
        resolution = 4096

        min_raw = degrees_to_raw_steps(args.min_deg, resolution)
        max_raw = degrees_to_raw_steps(args.max_deg, resolution)

        if min_raw >= max_raw:
            print(f"[ERROR] Computed min_raw={min_raw} >= max_raw={max_raw}. Check your degree inputs.")
            return 2

        # Disable torque for config
        bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Ensure position/servo mode (commonly Mode=0 in STS/SCS series tables)
        bus.write("Mode", 0)

        # Apply angle limits
        bus.write("Min_Angle_Limit", min_raw)
        bus.write("Max_Angle_Limit", max_raw)

        # Re-enable torque
        bus.write("Torque_Enable", TorqueMode.ENABLED.value)

        # Read back for confirmation
        rb_min = int(bus.read("Min_Angle_Limit")[0])
        rb_max = int(bus.read("Max_Angle_Limit")[0])

        print(f"[OK] Bounds set for ID={args.id}:")
        print(f"     min_deg={args.min_deg} -> Min_Angle_Limit={rb_min}")
        print(f"     max_deg={args.max_deg} -> Max_Angle_Limit={rb_max}")
        return 0

    finally:
        bus.disconnect()


def cmd_move(args: argparse.Namespace, bp: BusParams) -> int:
    bus = build_single_motor_bus(bp.port, args.id, bp.model, bp.motor_name)
    connect_bus(bus, bp.baudrate)

    try:
        resolution = 4096

        # Put motor in position/servo mode and enable torque
        bus.write("Mode", 0)
        bus.write("Torque_Enable", TorqueMode.ENABLED.value)

        # Optional motion shaping
        if args.goal_speed is not None:
            bus.write("Goal_Speed", int(args.goal_speed))
        if args.acceleration is not None:
            bus.write("Acceleration", int(args.acceleration))

        target_raw = degrees_to_raw_steps(args.deg, resolution)
        bus.write("Goal_Position", target_raw)

        if args.readback:
            pos = int(bus.read("Present_Position")[0])
            print(f"[OK] Commanded deg={args.deg} (raw={target_raw}), present raw={pos}")

        return 0

    finally:
        bus.disconnect()


def main() -> int:
    p = argparse.ArgumentParser(description="Feetech on Jetson via LeRobot FeetechMotorsBus")
    p.add_argument("--port", default="/dev/ttyACM0", help="Serial adapter path (default: /dev/ttyACM0)")
    p.add_argument("--baudrate", type=int, default=1_000_000, help="Bus baudrate (default: 1000000)")
    p.add_argument("--model", default="sts3215", help="Motor model key used by LeRobot (default: sts3215)")

    sub = p.add_subparsers(dest="cmd", required=True)

    s_id = sub.add_parser("set-id", help="Change a motor ID (requires you can uniquely address current ID)")
    s_id.add_argument("--current-id", type=int, required=True)
    s_id.add_argument("--new-id", type=int, required=True)

    s_bounds = sub.add_parser("set-bounds", help="Set Min/Max angle limits (degrees) for a motor")
    s_bounds.add_argument("--id", type=int, required=True)
    s_bounds.add_argument("--min-deg", type=float, required=True)
    s_bounds.add_argument("--max-deg", type=float, required=True)

    s_move = sub.add_parser("move", help="Move motor to a target in degrees (servo/position mode)")
    s_move.add_argument("--id", type=int, required=True)
    s_move.add_argument("--deg", type=float, required=True)
    s_move.add_argument("--goal-speed", type=int, default=None, help="Optional Goal_Speed register value")
    s_move.add_argument("--acceleration", type=int, default=None, help="Optional Acceleration register value")
    s_move.add_argument("--readback", action="store_true", help="Read Present_Position after command")

    args = p.parse_args()

    bp = BusParams(
        port=args.port,
        baudrate=args.baudrate,
        model=args.model,
        motor_name="m",
    )

    if args.cmd == "set-id":
        return cmd_set_id(args, bp)
    if args.cmd == "set-bounds":
        return cmd_set_bounds(args, bp)
    if args.cmd == "move":
        return cmd_move(args, bp)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
