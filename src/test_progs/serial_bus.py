# serial_bus.py
import serial
import threading
import time

class SerialBus:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0.01)
        self.lock = threading.Lock()
        self.latest_state = {
            "yaw": 0.0,   # robot heading [rad or deg, be consistent]
            "vx": 0.0,    # estimated forward speed m/s
            "vw": 0.0,    # estimated angular speed rad/s
            "batt": 0.0,  # battery volts
        }
        self._run_rx = True
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

    def _rx_loop(self):
        buffer = b""
        while self._run_rx:
            chunk = self.ser.read(128)
            if not chunk:
                continue
            buffer += chunk
            if b"\n" in buffer:
                lines = buffer.split(b"\n")
                buffer = lines[-1]
                for line in lines[:-1]:
                    self._parse_line(line.decode("utf-8", errors="ignore"))

    def _parse_line(self, line: str):
        # Expect format from ESP like:
        # STATE yaw=1.23 vx=0.10 vw=0.02 batt=11.7
        if not line.startswith("STATE"):
            return
        parts = line.split()[1:]
        with self.lock:
            for p in parts:
                k, v = p.split("=")
                self.latest_state[k] = float(v)

    def send_velocity_cmd(self, v_lin, v_ang):
        # Send a simple command string. ESP firmware must parse this.
        cmd = f"CMD V={v_lin:.3f} W={v_ang:.3f}\n"
        with self.lock:
            self.ser.write(cmd.encode("utf-8"))

    def get_state_snapshot(self):
        with self.lock:
            return dict(self.latest_state)

    def close(self):
        self._run_rx = False
        self.rx_thread.join()
        self.ser.close()
