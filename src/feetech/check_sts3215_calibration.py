#!/usr/bin/env python3
import argparse
import json
import scservo_sdk as scs

CTRL_TABLE = {
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Homing_Offset": (31, 2),
    "Present_Position": (56, 2),
}

SIGN_BITS = {"Homing_Offset": 11, "Present_Position": 15}


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value & (1 << sign_bit):
        return -(value & ((1 << sign_bit) - 1))
    return value & ((1 << sign_bit) - 1)


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


def read2(packet, port, mid, addr):
    ret = packet.read2ByteTxRx(port, mid, addr)
    if isinstance(ret, tuple) and len(ret) == 3:
        val, comm, err = ret
    elif isinstance(ret, tuple) and len(ret) == 2:
        val, comm = ret
        err = 0
    else:
        raise RuntimeError(f"Unexpected SDK return: {ret!r}")
    if comm != scs.COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(comm))
    if err != 0:
        raise RuntimeError(packet.getRxPacketError(err))
    return int(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--json", required=True, help="Calibration JSON produced by your calibrator")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        calib = json.load(f)

    port = scs.PortHandler(args.port)
    port.setPacketTimeout = patch_setPacketTimeout.__get__(port, scs.PortHandler)
    packet = scs.PacketHandler(0)

    if not port.openPort():
        raise RuntimeError("Failed to open port")
    if not port.setBaudRate(args.baudrate):
        raise RuntimeError("Failed to set baudrate")

    try:
        print(f"Checking servos on {args.port} @ {args.baudrate}")
        for name, d in calib.items():
            mid = int(d["id"])
            j_min = int(d["range_min"])
            j_max = int(d["range_max"])
            j_off = int(d.get("homing_offset", 0))

            e_min = read2(packet, port, mid, CTRL_TABLE["Min_Position_Limit"][0])
            e_max = read2(packet, port, mid, CTRL_TABLE["Max_Position_Limit"][0])
            e_off_raw = read2(packet, port, mid, CTRL_TABLE["Homing_Offset"][0])
            e_off = decode_sign_magnitude(e_off_raw, SIGN_BITS["Homing_Offset"])

            p_raw = read2(packet, port, mid, CTRL_TABLE["Present_Position"][0])
            p = decode_sign_magnitude(p_raw, SIGN_BITS["Present_Position"])

            print(f"\n{name} (ID {mid})")
            print(f"  JSON:   min={j_min} max={j_max} homing_offset={j_off}")
            print(f"  EEPROM: min={e_min} max={e_max} homing_offset={e_off}")
            print(f"  DIFF:   min={e_min-j_min:+d} max={e_max-j_max:+d} off={e_off-j_off:+d}")
            print(f"  Present_Position (ticks): {p}")

        print("\nDone.")
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
