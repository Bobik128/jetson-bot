# shared/constants.py

# ---------- Dataset / logging ----------
FRAME_SIZE      = (256, 144)
JPEG_QUALITY    = 90

# ---------- Robot command limits ----------
MAX_V           = 0.30
MAX_W           = 2.00

# ---------- IMU (MPU6050) ----------
I2C_BUS_IDX     = 7
MPU6050_ADDR    = 0x68

PWR_MGMT_1      = 0x6B

ACCEL_XOUT_H    = 0x3B
ACCEL_SCALE_2G  = 16384.0  # LSB per g at +/-2g

GYRO_XOUT_H     = 0x43
GYRO_SCALE_250  = 131.0    # LSB per dps at +/-250 dps