#!/usr/bin/env python3
import argparse
import scservo_sdk as scs

CTRL = {
    "Torque_Enable": (40, 1),
    "Lock": (55, 1),

    # EPROM (persistent)
    "Max_Torque_Limit": (16, 2),
    "Protection_Current": (28, 2),
    "Overload_Torque": (36, 1),

    # optional: make motion gentler
    "Acceleration": (41, 1),
}

def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50

def _unpack2or3(ret):
    if isinstance(ret, tuple) and len(ret) == 2:
        return None, ret[0], ret[1]
    if isinstance(ret, tuple) and len(ret) == 3:
        return ret[0], ret[1], ret[2]
    raise RuntimeError(f"Unexpected SDK return: {ret!r}")

def write1(packet, port, mid, addr, val):
    ret = packet.write1ByteTxRx(port, mid, addr, int(val))
    _, comm, err = _unpack2or3(ret)
    if comm != scs.COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(comm))
    if err != 0:
        raise RuntimeError(packet.getRxPacketError(err))

def write2(packet, port, mid, addr, val):
    ret = packet.write2ByteTxRx(port, mid, addr, int(val))
    _, comm, err = _unpack2or3(ret)
    if comm != scs.COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(comm))
    if err != 0:
        raise RuntimeError(packet.getRxPacketError(err))

def read1(packet, port, mid, addr):
    ret = packet.read1ByteTxRx(port, mid, addr)
    if len(ret) == 3:
        val, comm, err = ret
    else:
        val, comm = ret
        err = 0
    if comm != scs.COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(comm))
    if err != 0:
        raise RuntimeError(packet.getRxPacketError(err))
    return int(val)

def read2(packet, port, mid, addr):
    ret = packet.read2ByteTxRx(port, mid, addr)
    if len(ret) == 3:
        val, comm, err = ret
    else:
        val, comm = ret
        err = 0
    if comm != scs.COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(comm))
    if err != 0:
        raise RuntimeError(packet.getRxPacketError(err))
    return int(val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--id", type=int, default=6)

    # Conservative defaults (adjust later)
    ap.add_argument("--max_torque_limit", type=int, default=500, help="0..1000 typically on STS series")
    ap.add_argument("--protection_current", type=int, default=250, help="units depend on firmware; lower = safer")
    ap.add_argument("--overload_torque", type=int, default=25, help="0..100 (%) threshold-ish")
    ap.add_argument("--acceleration", type=int, default=80, help="0..254")

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    port = scs.PortHandler(args.port)
    port.setPacketTimeout = patch_setPacketTimeout.__get__(port, scs.PortHandler)
    packet = scs.PacketHandler(0)

    if not port.openPort():
        raise RuntimeError("Failed to open port")
    if not port.setBaudRate(args.baudrate):
        raise RuntimeError("Failed to set baudrate")

    try:
        mid = args.id

        if args.show:
            print("Current gripper settings:")
            print("  Max_Torque_Limit:", read2(packet, port, mid, CTRL["Max_Torque_Limit"][0]))
            print("  Protection_Current:", read2(packet, port, mid, CTRL["Protection_Current"][0]))
            print("  Overload_Torque:", read1(packet, port, mid, CTRL["Overload_Torque"][0]))
            print("  Acceleration:", read1(packet, port, mid, CTRL["Acceleration"][0]))

        if args.write:
            # torque off and unlock for EPROM writes
            write1(packet, port, mid, CTRL["Torque_Enable"][0], 0)
            write1(packet, port, mid, CTRL["Lock"][0], 0)

            write2(packet, port, mid, CTRL["Max_Torque_Limit"][0], args.max_torque_limit)
            write2(packet, port, mid, CTRL["Protection_Current"][0], args.protection_current)
            write1(packet, port, mid, CTRL["Overload_Torque"][0], args.overload_torque)
            write1(packet, port, mid, CTRL["Acceleration"][0], args.acceleration)

            print("Written. Power-cycle servo power to ensure EPROM changes take effect cleanly.")
    finally:
        port.closePort()

if __name__ == "__main__":
    main()
