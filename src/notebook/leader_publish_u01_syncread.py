#!/usr/bin/env python3
import argparse
import json
import socket
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def load_calibration_as_objects(calib_path: str) -> dict[str, MotorCalibration]:
    """
    Convert LeRobot-style calibration JSON into dict[str, MotorCalibration]
    (what FeetechMotorsBus normalization expects).
    """
    with open(calib_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: dict[str, MotorCalibration] = {}
    for name, d in raw.items():
        out[name] = MotorCalibration(
            id=int(d["id"]),
            drive_mode=int(d.get("drive_mode", 0)),
            homing_offset=int(d.get("homing_offset", 0)),
            range_min=int(d["range_min"]),
            range_max=int(d["range_max"]),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--jetson_ip", required=True)
    ap.add_argument("--udp_port", type=int, default=5005)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--calib", required=True, help="Path to leader calibration JSON (e.g. the_leader.json)")
    ap.add_argument("--no-gripper", action="store_true")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    calibration = load_calibration_as_objects(args.calib)

    # Use LeRobot-normalized ranges so we can convert to u01 without needing ticks.
    motors = {
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    }
    if not args.no_gripper:
        motors["gripper"] = Motor(6, "sts3215", MotorNormMode.RANGE_0_100)

    bus = FeetechMotorsBus(
        port=args.port,
        motors=motors,
        calibration=calibration,   # <-- FIX: MotorCalibration objects, not dicts
        protocol_version=0,
    )

    print(f"[leader] Connecting to {args.port} @ {args.baudrate} ...")
    bus.connect()
    bus.set_baudrate(args.baudrate)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt = 1.0 / args.hz

    names = list(motors.keys())
    print(f"[leader] Streaming u∈[0,1] via sync_read for {names} -> {args.jetson_ip}:{args.udp_port} @ {args.hz} Hz")

    n = 0
    last_u = {name: 0.5 for name in names}

    try:
        while True:
            # Same path as LeRobot teleop leader:
            pos = bus.sync_read("Present_Position")  # normalized based on calibration

            u = {}
            for name in names:
                v = float(pos[name])

                if name == "gripper":
                    # RANGE_0_100 -> u01
                    u01 = v / 100.0
                else:
                    # RANGE_M100_100 -> u01
                    u01 = (v + 100.0) / 200.0

                u01 = clamp01(u01)
                u[name] = u01
                last_u[name] = u01

            payload = {"t": time.time(), "unit": "u01", "u": u}
            sock.sendto(json.dumps(payload).encode("utf-8"), (args.jetson_ip, args.udp_port))

            if args.print and n % 50 == 0:
                print(u)

            n += 1
            time.sleep(dt)

    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
