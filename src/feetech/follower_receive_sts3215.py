#!/usr/bin/env python3
import argparse
import json
import socket
import sys
import time

# Reuse the same wrapper concepts as your calibration script.
# If you still have calibrate_sts3215_like_lerobot.py in this folder,
# we can import STS3215Bus and CTRL_TABLE from it to avoid duplication.
try:
    from calibrate_sts3215_like_lerobot import STS3215Bus, CTRL_TABLE, SIGN_BITS, decode_sign_magnitude
except Exception as e:
    print("ERROR: Could not import STS3215Bus from calibrate_sts3215_like_lerobot.py", file=sys.stderr)
    print("Make sure calibrate_sts3215_like_lerobot.py is in the same directory.", file=sys.stderr)
    raise

MIN_ADDR = CTRL_TABLE["Min_Position_Limit"][0]
MAX_ADDR = CTRL_TABLE["Max_Position_Limit"][0]


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0", help="Follower servo bus port")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--hz", type=float, default=50.0, help="Control loop rate")
    ap.add_argument("--max_step", type=int, default=40, help="Max ticks per update per joint")
    ap.add_argument("--timeout_s", type=float, default=0.5, help="Stop if no packets for this long")
    ap.add_argument("--torque_on_start", action="store_true", help="Enable torque immediately")
    args = ap.parse_args()

    # UDP listener
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.settimeout(0.1)

    # Servo bus
    bus = STS3215Bus(args.port, args.baudrate)
    bus.connect()

    # Set position mode + optionally torque enable
    for i in args.ids:
        bus.set_operating_mode_position(i)
        bus.lock(i, True)
        bus.torque_enable(i, bool(args.torque_on_start))

    # Read hard limits from EEPROM for extra clamp safety
    limits = {}
    for i in args.ids:
        mn = bus._read_2b(i, MIN_ADDR)
        mx = bus._read_2b(i, MAX_ADDR)
        limits[i] = (mn, mx)

    print(f"[follower] Listening UDP :{args.udp_port}, controlling IDs={args.ids}")
    print("[follower] Limits (from servo):")
    for i in args.ids:
        mn, mx = limits[i]
        print(f"  ID {i}: min={mn} max={mx}")

    dt = 1.0 / args.hz
    last_packet_t = time.time()

    # Internal state (last commanded)
    q_cmd = {i: None for i in args.ids}

    try:
        while True:
            # Receive latest packet if any (non-blocking-ish)
            got = False
            try:
                data, _ = sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                q_in = {int(k): int(v) for k, v in msg["q"].items()}
                last_packet_t = time.time()
                got = True
            except socket.timeout:
                q_in = None

            # Safety timeout
            if time.time() - last_packet_t > args.timeout_s:
                print("[follower] Teleop timeout -> torque off")
                for i in args.ids:
                    bus.torque_enable(i, False)
                    bus.lock(i, False)
                break

            # If no new packet this cycle, just idle at loop rate
            if not got:
                time.sleep(dt)
                continue

            # Enable torque upon first packet if not enabled at start
            for i in args.ids:
                bus.torque_enable(i, True)

            # Compute safe targets
            goal = {}
            for i in args.ids:
                if i not in q_in:
                    continue
                target = q_in[i]

                mn, mx = limits[i]
                target = clamp(target, mn, mx)

                if q_cmd[i] is None:
                    # First command snaps to target (still clamped)
                    goal[i] = target
                else:
                    step = clamp(target - q_cmd[i], -args.max_step, args.max_step)
                    goal[i] = q_cmd[i] + step

                q_cmd[i] = goal[i]

            # Write goal positions
            # Uses the same address and encoding semantics as calibration script.
            # For STS3215 Goal_Position is sign-magnitude encoded, and your wrapper handles that.
            bus.sync_write_goal_position(goal)

            time.sleep(dt)

    finally:
        try:
            bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()