#!/usr/bin/env python3
import argparse
import json
import socket
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--jetson_ip", required=True)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6])
    args = ap.parse_args()

    # Use DEGREES (available in LeRobot; you used it in SO101Follower snippet too)
    motors = {f"id_{i}": Motor(i, "sts3215", MotorNormMode.DEGREES) for i in args.ids}

    bus = FeetechMotorsBus(
        port=args.port,
        motors=motors,
        calibration=None,
        protocol_version=0,
    )

    print(f"[leader] Connecting to {args.port} @ {args.baudrate} ...")
    bus.connect()
    bus.set_baudrate(args.baudrate)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt = 1.0 / args.hz

    print(f"[leader] Streaming DEGREES for IDs {args.ids} -> {args.jetson_ip}:{args.udp_port} @ {args.hz} Hz")
    try:
        while True:
            pos = bus.sync_read("Present_Position")  # values in degrees now
            payload = {
                "t": time.time(),
                "unit": "deg",
                "q": {int(name.split('_')[1]): float(val) for name, val in pos.items()},
            }
            sock.sendto(json.dumps(payload).encode("utf-8"), (args.jetson_ip, args.udp_port))
            time.sleep(dt)
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()