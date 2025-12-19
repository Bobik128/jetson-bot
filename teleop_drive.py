#!/usr/bin/env python3
import time, math, sys, serial, pygame, smbus2

########################################
# CONFIG
########################################

SERIAL_PORT = "/dev/ttyTHS1"   # adapt if needed (/dev/ttyUSB0 etc.)
BAUD        = 115200
SEND_HZ     = 30.0
DT          = 1.0 / SEND_HZ

DEADZONE    = 0.20
EXPO        = 0.25  # response shaping
MAX_V       = 0.3   # m/s safe indoor
MAX_W       = 1.0   # rad/s safe indoor

USE_TURBO   = True
TURBO_AXIS  = 4     # adjust if needed
TURBO_MIN   = 0.6
TURBO_GAIN  = 1.75

# MPU6050 I2C
I2C_BUS_IDX      = 7          # <- you used 7
MPU6050_ADDR     = 0x68
PWR_MGMT_1       = 0x6B
ACCEL_XOUT_H     = 0x3B
GYRO_XOUT_H      = 0x43
ACCEL_SCALE_2G   = 16384.0
GYRO_SCALE_250   = 131.0

########################################
# HELPERS
########################################

def dz(x, d=DEADZONE):
    if abs(x) < d:
        return 0.0
    return (x - math.copysign(d, x)) / (1.0 - d)

def expo_fn(x, k=EXPO):
    return math.copysign(abs(x) ** (1.0 + k*1.5), x)

def stick_to_cmd(lx, ly, turbo_raw):
    # NOTE: you had lx=axis1 ly=axis0 swapped; let's define:
    # left stick vertical = forward/back
    # left stick horizontal = turn
    forward = -dz(ly)  # forward stick usually negative up
    turn    =  dz(lx)

    forward = expo_fn(forward, EXPO)
    turn    = expo_fn(turn, EXPO)

    scale = 1.0
    if USE_TURBO and turbo_raw >= TURBO_MIN:
        scale = TURBO_GAIN

    v = forward * MAX_V * scale      # m/s
    w = turn    * MAX_W * scale      # rad/s
    return v, w

########################################
# IMU
########################################

class MPU6050:
    def __init__(self, bus_idx=I2C_BUS_IDX, addr=MPU6050_ADDR):
        self.bus = smbus2.SMBus(bus_idx)
        self.addr = addr
        # wake MPU
        self.bus.write_byte_data(self.addr, PWR_MGMT_1, 0)

        self.last_time = time.time()
        self.yaw_deg = 0.0  # integrate gyro_z

    def _read_word(self, reg):
        high = self.bus.read_byte_data(self.addr, reg)
        low  = self.bus.read_byte_data(self.addr, reg+1)
        val  = (high << 8) + low
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val

    def read_and_update(self):
        # read gyro Z only for yaw rate
        gyro_z_raw = self._read_word(GYRO_XOUT_H + 4)
        gyro_z_dps = gyro_z_raw / GYRO_SCALE_250  # deg/sec

        now = time.time()
        dt  = now - self.last_time
        self.last_time = now

        # integrate yaw
        self.yaw_deg += gyro_z_dps * dt
        return self.yaw_deg

########################################
# SERIAL
########################################

class ESPLink:
    def __init__(self, port=SERIAL_PORT, baud=BAUD):
        self.ser = serial.Serial(port, baud, timeout=0)
        time.sleep(0.2)

    def send_cmd(self, v_lin, v_ang):
        # same protocol we want ESP to parse
        msg = f"CMD V={v_lin:.3f} W={v_ang:.3f}\n"
        self.ser.write(msg.encode("utf-8"))

    def read_state_lines(self):
        # non-blocking read
        data = self.ser.read(256)
        return data.decode("utf-8", errors="ignore")

########################################
# GAMEPAD
########################################

def init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick detected")
        sys.exit(1)
    js = pygame.joystick.Joystick(0)
    js.init()
    print("Using:", js.get_name())
    return js

def read_axes(js):
    pygame.event.pump()
    # adjust to match actual axes on your DualSense under pygame
    # commonly:
    # axis 1 = left stick vertical (-1 up, +1 down)
    # axis 0 = left stick horizontal (-1 left, +1 right)
    ly = js.get_axis(1)
    lx = js.get_axis(0)

    turbo = 0.0
    if USE_TURBO:
        try:
            raw = js.get_axis(TURBO_AXIS)
            turbo = (raw + 1.0) * 0.5  # normalize to 0..1
        except Exception:
            turbo = 0.0
    return lx, ly, turbo

########################################
# MAIN LOOP
########################################

def main():
    js = init_gamepad()
    imu = MPU6050()
    link = ESPLink()

    print("Teleop active. Ctrl+C to quit.")

    next_t = time.perf_counter()
    while True:
        try:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += DT

            # read controller
            lx, ly, turbo = read_axes(js)
            v, w = stick_to_cmd(lx, ly, turbo)

            # send to ESP
            link.send_cmd(v, w)

            # update yaw estimate
            yaw_deg = imu.read_and_update()

            # (optional now) read any ESP feedback
            esp_debug = link.read_state_lines()

            # debug print
            print(f"v={v:+.2f} m/s  w={w:+.2f} rad/s  yaw≈{yaw_deg:7.2f}°   {esp_debug.strip()}")

        except KeyboardInterrupt:
            print("\nSTOP")
            link.send_cmd(0.0, 0.0)
            break
        except Exception as e:
            print("ERR:", e)
            time.sleep(0.05)

if __name__ == "__main__":
    main()
