# shared/esp32_link.py
import serial

class ESP32Link:
    def __init__(self, port="/dev/ttyTHS1", baud=115200, timeout=0.01):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def send_cmd(self, v, w):
        msg = f"CMD V={v:.3f} W={w:.3f}\n"
        self.ser.write(msg.encode("utf-8"))

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass