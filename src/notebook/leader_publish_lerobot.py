#!/usr/bin/env python3
import argparse
import json
import socket
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def load_calibration(calib_path: str) -> dict[str, MotorCalibration]:
    with open(calib_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    calib = {}
    for name, d in raw.items():
        calib[name] = MotorCalibration(
            id=int(d["id"]),
            drive_mode=int(d.get("drive_mode", 0)),
            homing_offset=int(d.get("homing_offset", 0)),
            range_min=int(d["range_min"]),
            range_max=int(d["range_max"]),
        )
    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--jetson_ip", required=True)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--calib", required=True, help="Path to LeRobot calibration json for the LEADER arm")
    args = ap.parse_args()

    calibration = load_calibration(args.calib)

    # IMPORTANT: motor names MUST match the keys in the calibration JSON
    # Use the canonical SO101 names so it matches your file.
    motors = {
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),  # matches LeRobot convention
    }

    bus = FeetechMotorsBus(
        port=args.port,
        motors=motors,
        calibration=calibration,
        protocol_version=0,
    )

    print(f"[leader] Connecting to {args.port} @ {args.baudrate} ...")
    bus.connect()
    bus.set_baudrate(args.baudrate)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt = 1.0 / args.hz

    print(f"[leader] Streaming -> {args.jetson_ip}:{args.udp_port} @ {args.hz} Hz")
    try:
        while True:
            pos = bus.sync_read("Present_Position")  # now works because calibration exists
            # pos keys: shoulder_lift, elbow_flex, wrist_flex, gripper
            payload = {
                "t": time.time(),
                "unit": "lerobot_norm",
                "q": {k: float(v) for k, v in pos.items()},
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