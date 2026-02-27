import time
import errno
import smbus2
from dataclasses import dataclass
from typing import Optional, Tuple

from .constants import (
    I2C_BUS_IDX, MPU6050_ADDR,
    PWR_MGMT_1,
    ACCEL_XOUT_H, ACCEL_SCALE_2G,
    GYRO_XOUT_H, GYRO_SCALE_250,
)

G = 9.80665

def _i2c_write_retry(bus, addr, reg, val, retries=5, delay=0.01):
    for _ in range(retries):
        try:
            bus.write_byte_data(addr, reg, val)
            return
        except OSError as e:
            if e.errno in (errno.EIO, 121):
                time.sleep(delay)
                delay = min(delay * 2, 0.2)
                continue
            raise
    raise RuntimeError("I2C write failed after retries.")

def _i2c_read_retry(bus, addr, reg, length=1, retries=5, delay=0.01):
    for _ in range(retries):
        try:
            if length == 1:
                return bus.read_byte_data(addr, reg)
            return bus.read_i2c_block_data(addr, reg, length)
        except OSError as e:
            if e.errno in (errno.EIO, 121):
                time.sleep(delay)
                delay = min(delay * 2, 0.2)
                continue
            raise
    raise RuntimeError("I2C read failed after retries.")


@dataclass
class IMURates:
    dt: float
    gyro: Tuple[float, float, float]                  # deg/s or rad/s
    accel: Optional[Tuple[float, float, float]]       # m/s^2 or g
    vel: Optional[Tuple[float, float, float]]         # m/s (integrated from accel)


class MPU6050Rates:
    """
    Outputs IMU rates for ML:
      - gyro angular velocity (deg/s default)
      - accel (m/s^2 default)
      - velocity estimate (m/s) by integrating accel over time (optional)

    WARNING: velocity integration drifts unless accel bias + gravity are handled.
    """

    def __init__(
        self,
        bus_idx: int = I2C_BUS_IDX,
        addr: int = MPU6050_ADDR,
        calibrate: bool = True,
        include_accel: bool = True,
        gyro_units: str = "dps",       # "dps" or "rads"
        accel_units: str = "mps2",     # "mps2" or "g"
        accel_remove_gravity: bool = False,
        integrate_velocity: bool = True,              # <-- new
        vel_clamp_mps: Optional[float] = None,         # <-- new (limits drift)
        dt_clamp: Tuple[float, float] = (1e-4, 0.1),
    ):
        self.bus = smbus2.SMBus(bus_idx)
        self.addr = addr

        _i2c_write_retry(self.bus, self.addr, PWR_MGMT_1, 0x00)  # wake

        self.include_accel = include_accel
        self.gyro_units = gyro_units
        self.accel_units = accel_units
        self.accel_remove_gravity = accel_remove_gravity
        self.integrate_velocity = integrate_velocity and include_accel
        self.vel_clamp_mps = vel_clamp_mps
        self.dt_min, self.dt_max = dt_clamp

        # biases in raw LSB
        self.bias_gx = self.bias_gy = self.bias_gz = 0.0
        self.bias_ax = self.bias_ay = self.bias_az = 0.0

        # integrated velocity state (m/s)
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0

        self._last_t = time.time()

        if calibrate:
            self._calibrate_gyro()
            if self.include_accel:
                self._calibrate_accel()

    # ------- raw reads -------

    def _read_word(self, reg: int) -> int:
        hi = _i2c_read_retry(self.bus, self.addr, reg, 1)
        lo = _i2c_read_retry(self.bus, self.addr, reg + 1, 1)
        v = (hi << 8) | lo
        if v >= 0x8000:
            v = -((65535 - v) + 1)
        return v

    def _read_gyro_raw_all(self) -> Tuple[int, int, int]:
        gx = self._read_word(GYRO_XOUT_H)
        gy = self._read_word(GYRO_XOUT_H + 2)
        gz = self._read_word(GYRO_XOUT_H + 4)
        return gx, gy, gz

    def _read_accel_raw_all(self) -> Tuple[int, int, int]:
        ax = self._read_word(ACCEL_XOUT_H)
        ay = self._read_word(ACCEL_XOUT_H + 2)
        az = self._read_word(ACCEL_XOUT_H + 4)
        return ax, ay, az

    # ------- calibration -------

    def _calibrate_gyro(self, n_samples=500, delay=0.002):
        print(f"Calibrating gyro ({n_samples} samples)... keep IMU still.")
        sx = sy = sz = 0.0
        for _ in range(n_samples):
            gx, gy, gz = self._read_gyro_raw_all()
            sx += gx; sy += gy; sz += gz
            time.sleep(delay)
        self.bias_gx = sx / n_samples
        self.bias_gy = sy / n_samples
        self.bias_gz = sz / n_samples
        print(f"Gyro bias (raw): gx={self.bias_gx:.1f}, gy={self.bias_gy:.1f}, gz={self.bias_gz:.1f}")

    def _calibrate_accel(self, n_samples=500, delay=0.002):
        """
        Stationary calibration, assumes Z sees +1g.
        After this, accel in 'g' is ~ (0,0,+1) at rest.
        """
        print(f"Calibrating accel ({n_samples} samples)... keep IMU still.")
        sx = sy = sz = 0.0
        for _ in range(n_samples):
            ax, ay, az = self._read_accel_raw_all()
            sx += ax; sy += ay; sz += az
            time.sleep(delay)

        mean_ax = sx / n_samples
        mean_ay = sy / n_samples
        mean_az = sz / n_samples

        self.bias_ax = mean_ax
        self.bias_ay = mean_ay
        self.bias_az = mean_az - ACCEL_SCALE_2G  # so az_g ~ +1 at rest

        print(f"Accel bias (raw): ax={self.bias_ax:.1f}, ay={self.bias_ay:.1f}, az={self.bias_az:.1f}")

    # ------- scaled -------

    def read_gyro(self) -> Tuple[float, float, float]:
        gx_raw, gy_raw, gz_raw = self._read_gyro_raw_all()
        gx_dps = (gx_raw - self.bias_gx) / GYRO_SCALE_250
        gy_dps = (gy_raw - self.bias_gy) / GYRO_SCALE_250
        gz_dps = (gz_raw - self.bias_gz) / GYRO_SCALE_250

        if self.gyro_units == "dps":
            return gx_dps, gy_dps, gz_dps
        if self.gyro_units == "rads":
            k = 3.141592653589793 / 180.0
            return gx_dps * k, gy_dps * k, gz_dps * k
        raise ValueError("gyro_units must be 'dps' or 'rads'")

    def read_accel(self) -> Tuple[float, float, float]:
        ax_raw, ay_raw, az_raw = self._read_accel_raw_all()
        ax_g = (ax_raw - self.bias_ax) / ACCEL_SCALE_2G
        ay_g = (ay_raw - self.bias_ay) / ACCEL_SCALE_2G
        az_g = (az_raw - self.bias_az) / ACCEL_SCALE_2G

        if self.accel_remove_gravity:
            # only valid if your Z axis is consistently aligned with gravity
            az_g -= 1.0

        if self.accel_units == "g":
            return ax_g, ay_g, az_g
        if self.accel_units == "mps2":
            return ax_g * G, ay_g * G, az_g * G
        raise ValueError("accel_units must be 'mps2' or 'g'")

    # ------- helpers -------

    def reset_timebase(self):
        self._last_t = time.time()

    def reset_velocity(self):
        self._vx = self._vy = self._vz = 0.0

    def reset_all(self):
        self.reset_timebase()
        self.reset_velocity()

    def _clamp_vel(self):
        if self.vel_clamp_mps is None:
            return
        m = self.vel_clamp_mps
        self._vx = max(-m, min(m, self._vx))
        self._vy = max(-m, min(m, self._vy))
        self._vz = max(-m, min(m, self._vz))

    # ------- main API -------

    def sample(self) -> IMURates:
        now = time.time()
        dt = now - self._last_t
        self._last_t = now

        if dt < self.dt_min:
            dt = self.dt_min
        elif dt > self.dt_max:
            dt = self.dt_max

        gyro = self.read_gyro()

        accel = None
        vel = None

        if self.include_accel:
            accel = self.read_accel()

            # If accel is in g, convert to m/s^2 before integrating velocity
            ax, ay, az = accel
            if self.accel_units == "g":
                ax *= G; ay *= G; az *= G

            if self.integrate_velocity:
                self._vx += ax * dt
                self._vy += ay * dt
                self._vz += az * dt
                self._clamp_vel()
                vel = (self._vx, self._vy, self._vz)

        return IMURates(dt=dt, gyro=gyro, accel=accel, vel=vel)