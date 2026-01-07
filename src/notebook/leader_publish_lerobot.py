#!/usr/bin/env python3
import argparse
import json
import socket
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0", help="Leader arm USB serial port")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--jetson_ip", required=True, help="Jetson Tailscale IP (100.x.y.z)")
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--ids", type=int, nargs="+", default=[2, 3, 4, 6], help="Servo IDs to publish")
    args = ap.parse_args()

    # Minimal motors dict for LeRobot bus.
    # Names are arbitrary, IDs must match your leader arm IDs.
    motors = {f"id_{i}": Motor(i, "sts3215", MotorNormMode.RAW) for i in args.ids}

    bus = FeetechMotorsBus(
        port=args.port,
        motors=motors,
        calibration=None,      # We just need Present_Position; calibration optional
        protocol_version=0,    # STS3215 uses protocol 0
    )

    print(f"[leader] Connecting to {args.port} @ {args.baudrate} ...")
    bus.connect()
    bus.set_baudrate(args.baudrate)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt = 1.0 / args.hz

    print(f"[leader] Streaming IDs {args.ids} to {args.jetson_ip}:{args.udp_port} at {args.hz} Hz")
    try:
        while True:
            # Returns dict keyed by motor names (id_2, id_3, ...)
            pos = bus.sync_read("Present_Position")  # RAW ticks if MotorNormMode.RAW
            payload = {
                "t": time.time(),
                "q": {int(name.split("_")[1]): int(val) for name, val in pos.items()},
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