#!/usr/bin/env python3
import argparse
import scservo_sdk as scs

CTRL = {
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Torque_Enable": (40, 1),
    "Lock": (55, 1),
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
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--min", dest="mn", type=int, default=0)
    ap.add_argument("--max", dest="mx", type=int, default=4095)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    port = scs.PortHandler(args.port)
    port.setPacketTimeout = patch_setPacketTimeout.__get__(port, scs.PortHandler)
    packet = scs.PacketHandler(0)

    if not port.openPort():
        raise RuntimeError(f"Failed to open {args.port}")
    if not port.setBaudRate(args.baudrate):
        raise RuntimeError(f"Failed to set baudrate {args.baudrate}")

    try:
        for mid in args.ids:
            # safe state for EEPROM write
            write1(packet, port, mid, CTRL["Torque_Enable"][0], 0)
            write1(packet, port, mid, CTRL["Lock"][0], 0)

            write2(packet, port, mid, CTRL["Min_Position_Limit"][0], args.mn)
            write2(packet, port, mid, CTRL["Max_Position_Limit"][0], args.mx)

            if args.verify:
                mn = read2(packet, port, mid, CTRL["Min_Position_Limit"][0])
                mx = read2(packet, port, mid, CTRL["Max_Position_Limit"][0])
                print(f"ID {mid}: min={mn} max={mx}")
            else:
                print(f"ID {mid}: wrote min={args.mn} max={args.mx}")

        print("\nIMPORTANT: Power-cycle the servo bus after changing EEPROM limits.")
    finally:
        port.closePort()

if __name__ == "__main__":
    main()
