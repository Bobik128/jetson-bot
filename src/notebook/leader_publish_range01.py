#!/usr/bin/env python3
import argparse
import json
import socket
import time

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode


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


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--jetson_ip", required=True)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--calib", required=True, help="Leader LeRobot calibration JSON path")
    args = ap.parse_args()

    calibration = load_calibration(args.calib)

    # IMPORTANT: names must match keys in the calibration JSON
    motors = {
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),  # norm mode irrelevant if we read raw
        "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
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

    names = list(motors.keys())
    print(f"[leader] Streaming range fractions u∈[0,1] for {names} -> {args.jetson_ip}:{args.udp_port} @ {args.hz} Hz")

    try:
        while True:
            # Read *raw ticks* per motor to avoid LeRobot normalization ambiguity.
            # We do sequential reads for compatibility across LeRobot versions.
            u = {}
            for name in names:
                cal = calibration[name]
                # read raw present position (ticks), bypass normalization
                pos_ticks = bus.read("Present_Position", name, normalize=False)

                lo, hi = cal.range_min, cal.range_max
                span = (hi - lo) if (hi - lo) != 0 else 1
                frac = (pos_ticks - lo) / span
                u[name] = clamp01(float(frac))

            payload = {
                "t": time.time(),
                "unit": "u01",   # range fraction
                "u": u,
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