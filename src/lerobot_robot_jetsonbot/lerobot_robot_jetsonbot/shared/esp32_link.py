# shared/esp32_link.py
import serial
import threading
import time
import re
from typing import Optional, Tuple


class ESP32Link:
    """
    Simple serial link to an ESP32.

    TX (to ESP32):  "CMD V=0.123 W=0.456\n"
    RX (from ESP32): expected telemetry lines containing "v=<float> w=<float>"
                     e.g. " v=0.123 w=0.456"
    """

    _TELEM_RE = re.compile(r"\bv\s*=\s*([+-]?\d+(?:\.\d+)?)\s+\bw\s*=\s*([+-]?\d+(?:\.\d+)?)\b")

    def __init__(self, port="/dev/ttyTHS1", baud=115200, timeout=0.05):
        self.port = port
        self.baud = baud
        self.timeout = timeout

        self.ser = serial.Serial(port, baud, timeout=timeout)

        # Reader thread state
        self._rx_thread: Optional[threading.Thread] = None
        self._rx_stop = threading.Event()

        # Latest values (protected by lock)
        self._lock = threading.Lock()
        self._last_v: Optional[float] = None
        self._last_w: Optional[float] = None
        self._last_rx_time: Optional[float] = None
        self._last_line: Optional[str] = None

    def send_cmd(self, v: float, w: float) -> None:
        msg = f"CMD V={v:.3f} W={w:.3f}\n"
        self.ser.write(msg.encode("utf-8"))

    # --------------------------
    # Background RX loop
    # --------------------------
    def start_reader(self, daemon: bool = True) -> None:
        """Start background thread that reads and parses telemetry."""
        if self._rx_thread and self._rx_thread.is_alive():
            return  # already running
        self._rx_stop.clear()
        self._rx_thread = threading.Thread(target=self._reader_loop, daemon=daemon)
        self._rx_thread.start()

    def stop_reader(self, join_timeout: float = 1.0) -> None:
        """Stop background reader thread."""
        self._rx_stop.set()
        t = self._rx_thread
        if t and t.is_alive():
            t.join(timeout=join_timeout)

    def _reader_loop(self) -> None:
        """
        Continuously read lines. This uses readline(), which returns on '\n'
        or after serial timeout.
        """
        # Ensure we don't block forever even if something goes weird
        while not self._rx_stop.is_set():
            try:
                raw = self.ser.readline()
            except (serial.SerialException, OSError):
                # Port broke / disconnected
                time.sleep(0.1)
                continue

            if not raw:
                continue  # timeout, no data

            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if not line:
                continue

            m = self._TELEM_RE.search(line)
            with self._lock:
                self._last_line = line
                if m:
                    try:
                        v = float(m.group(1))
                        w = float(m.group(2))
                        self._last_v = v
                        self._last_w = w
                        self._last_rx_time = time.time()
                    except ValueError:
                        # keep last values, just store last line
                        pass

    # --------------------------
    # Accessors
    # --------------------------
    def get_latest(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Returns (v, w, age_seconds).
        age_seconds is None if we've never received valid telemetry.
        """
        with self._lock:
            v = self._last_v
            w = self._last_w
            t = self._last_rx_time

        if t is None:
            return v, w, None

        age = time.time() - t
        return v, w, age

    def get_last_line(self) -> Optional[str]:
        """Return the most recent received line (parsed or not)."""
        with self._lock:
            return self._last_line
        
    def is_connected(self, max_age: float = 0.5) -> bool:
        """Return True if we have received telemetry within max_age seconds."""
        with self._lock:
            t = self._last_rx_time
        if t is None:
            return False
        return (time.time() - t) <= max_age
        # return True

    def close(self) -> None:
        """Stop reader and close serial port."""
        self.stop_reader()
        try:
            self.ser.close()
        except Exception:
            pass