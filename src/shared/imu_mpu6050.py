# shared/imu_mpu6050.py
import time
import errno
import smbus2

from .constants import (
    I2C_BUS_IDX, MPU6050_ADDR,
    PWR_MGMT_1,
    ACCEL_XOUT_H, ACCEL_SCALE_2G,
    GYRO_XOUT_H, GYRO_SCALE_250,
)

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

class MPU6050GyroYaw:
    """
    MPU6050 reader.
    'Yaw' is integrated from gyro X axis (gx) by design (your mounting).
    Provides:
      - update_and_get_yaw_delta_deg(): Δyaw in degrees since last call
      - read_accel_g(): (ax, ay, az) in g, bias-corrected
    """

    def __init__(self, bus_idx=I2C_BUS_IDX, addr=MPU6050_ADDR):
        self.bus_idx = bus_idx
        self.addr = addr
        self.bus = smbus2.SMBus(bus_idx)

        _i2c_write_retry(self.bus, self.addr, PWR_MGMT_1, 0x00)  # wake

        # Gyro bias (raw LSB)
        self.bias_gx = 0.0
        self.bias_gy = 0.0
        self.bias_gz = 0.0

        # Accel bias (raw LSB)
        self.bias_ax = 0.0
        self.bias_ay = 0.0
        self.bias_az = 0.0

        self._last_t = time.time()

        self._calibrate_gyro(n_samples=500, delay=0.002)
        self._calibrate_accel(n_samples=500, delay=0.002)

    def _read_word(self, reg):
        hi = _i2c_read_retry(self.bus, self.addr, reg, 1)
        lo = _i2c_read_retry(self.bus, self.addr, reg + 1, 1)
        val = (hi << 8) | lo
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val

    def _read_accel_raw_all(self):
        ax = self._read_word(ACCEL_XOUT_H)
        ay = self._read_word(ACCEL_XOUT_H + 2)
        az = self._read_word(ACCEL_XOUT_H + 4)
        return ax, ay, az

    def _read_gyro_raw_all(self):
        gx = self._read_word(GYRO_XOUT_H)
        gy = self._read_word(GYRO_XOUT_H + 2)
        gz = self._read_word(GYRO_XOUT_H + 4)
        return gx, gy, gz

    def _calibrate_gyro(self, n_samples=500, delay=0.002):
        print(f"Calibrating gyro ({n_samples} samples)... keep robot still.")
        sx = sy = sz = 0.0
        for _ in range(n_samples):
            gx, gy, gz = self._read_gyro_raw_all()
            sx += gx
            sy += gy
            sz += gz
            time.sleep(delay)

        self.bias_gx = sx / n_samples
        self.bias_gy = sy / n_samples
        self.bias_gz = sz / n_samples
        print(f"Gyro bias (raw): gx={self.bias_gx:.1f}, gy={self.bias_gy:.1f}, gz={self.bias_gz:.1f}")

    def _calibrate_accel(self, n_samples=500, delay=0.002):
        """
        Stationary calibration. Assumes Z sees +1g. Removes that so outputs are ~0 at rest.
        """
        print(f"Calibrating accel ({n_samples} samples)... keep robot still.")
        sx = sy = sz = 0.0
        for _ in range(n_samples):
            ax, ay, az = self._read_accel_raw_all()
            sx += ax
            sy += ay
            sz += az
            time.sleep(delay)

        mean_ax = sx / n_samples
        mean_ay = sy / n_samples
        mean_az = sz / n_samples

        self.bias_ax = mean_ax
        self.bias_ay = mean_ay
        self.bias_az = mean_az - ACCEL_SCALE_2G

        print(f"Accel bias (raw): ax={self.bias_ax:.1f}, ay={self.bias_ay:.1f}, az={self.bias_az:.1f}")

    def read_gyro_dps(self):
        gx_raw, gy_raw, gz_raw = self._read_gyro_raw_all()
        gx = (gx_raw - self.bias_gx) / GYRO_SCALE_250
        gy = (gy_raw - self.bias_gy) / GYRO_SCALE_250
        gz = (gz_raw - self.bias_gz) / GYRO_SCALE_250
        return gx, gy, gz

    def read_accel_g(self):
        ax_raw, ay_raw, az_raw = self._read_accel_raw_all()
        ax = (ax_raw - self.bias_ax) / ACCEL_SCALE_2G
        ay = (ay_raw - self.bias_ay) / ACCEL_SCALE_2G
        az = (az_raw - self.bias_az) / ACCEL_SCALE_2G
        return ax, ay, az

    def update_and_get_yaw_delta_deg(self):
        gx_dps, _, _ = self.read_gyro_dps()
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        return gx_dps * dt