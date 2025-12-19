#!/usr/bin/env python3
import time
import math
import sys
import serial
import pygame
import smbus2

########################################
#              CONFIG
########################################

# ---------- UART to ESP32 ----------
SERIAL_PORT = "/dev/ttyTHS1"   # Jetson UART on 40-pin header. Change if needed.
BAUD        = 115200

# ---------- Control rate ----------
SEND_HZ     = 30.0
DT          = 1.0 / SEND_HZ

# ---------- Gamepad mapping ----------
# We'll assume DualSense over USB/Bluetooth recognized by pygame.
# left stick vertical = forward/back
# left stick horizontal = turn
# R2 trigger = turbo

DEADZONE    = 0.20       # ignore tiny stick noise
EXPO        = 0.25       # nonlinear response (0 = linear)

MAX_V       = 0.30       # m/s forward speed at full stick (### TUNE ME)
MAX_W       = 2.00       # rad/s yaw rate at full stick (### TUNE ME)

USE_TURBO   = True
TURBO_AXIS  = 4          # depends on driver; sometimes 4 or 5 for R2 on DualSense
TURBO_MIN   = 0.6        # how far you have to press before turbo applies
TURBO_GAIN  = 1.75       # scale speeds when turbo held (### TUNE ME)

# ---------- MPU6050 config ----------
# This is your IMU connected directly to the Jetson's I2C.
I2C_BUS_IDX     = 7       # you said bus 7
MPU6050_ADDR    = 0x68
PWR_MGMT_1      = 0x6B
GYRO_XOUT_H     = 0x43
GYRO_SCALE_250  = 131.0   # LSB -> deg/s for ±250 dps range

########################################
#     HELPER FUNCTIONS
########################################

def dz(x, dead):
    """Apply deadzone and rescale the remaining range to [-1,1]."""
    if abs(x) < dead:
        return 0.0
    return (x - math.copysign(dead, x)) / (1.0 - dead)

def expo_fn(x, k):
    """Softens center response to make control less twitchy."""
    # same shape you used: raise |x| to power >1
    return math.copysign(abs(x) ** (1.0 + k*1.5), x)

def stick_to_cmd(lx, ly, turbo_raw):
    """
    Convert raw stick values to (v, w).
    lx = left stick horizontal  (-1 left, +1 right)
    ly = left stick vertical    (-1 up, +1 down)
    turbo_raw = trigger value (0..1 after normalization)
    """
    # Note: on most controllers, up = -1. So we invert ly for forward.
    forward_raw = -dz(ly, DEADZONE)
    turn_raw    =  dz(lx, DEADZONE)

    forward_shaped = expo_fn(forward_raw, EXPO)
    turn_shaped    = expo_fn(turn_raw,    EXPO)

    scale = 1.0
    if USE_TURBO and turbo_raw >= TURBO_MIN:
        scale = TURBO_GAIN

    v = forward_shaped * MAX_V * scale   # m/s
    w = turn_shaped    * MAX_W * scale   # rad/s

    return v, w

########################################
#     MPU6050 READER (YAW ESTIMATE)
########################################

class MPU6050GyroYaw:
    """
    Super lightweight IMU reader:
    - wake up MPU6050
    - read gyro Z (deg/s)
    - integrate to yaw_deg
    This is drift-y, but enough for IL dataset context and debug.
    """
    def __init__(self, bus_idx=I2C_BUS_IDX, addr=MPU6050_ADDR):
        self.addr = addr
        self.bus = smbus2.SMBus(bus_idx)

        # Wake MPU6050
        self.bus.write_byte_data(self.addr, PWR_MGMT_1, 0)

        self.last_time = time.time()
        self.yaw_deg = 0.0

    def _read_word(self, reg):
        high = self.bus.read_byte_data(self.addr, reg)
        low  = self.bus.read_byte_data(self.addr, reg + 1)
        val  = (high << 8) | low
        # convert 16-bit signed
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val

    def update_and_get_yaw_deg(self):
        """Read gyro Z, integrate to yaw (degrees), return yaw."""
        # gyro_z is at GYRO_XOUT_H + 4
        gyro_z_raw = self._read_word(GYRO_XOUT_H + 4)
        gyro_z_dps = gyro_z_raw / GYRO_SCALE_250  # deg/sec

        now = time.time()
        dt  = now - self.last_time
        self.last_time = now

        self.yaw_deg += gyro_z_dps * dt
        return self.yaw_deg

########################################
#     ESP32 LINK
########################################

class ESP32Link:
    """
    Handles UART to the ESP32.
    New protocol:
        CMD V=<v> W=<w>\\n
    ESP will respond with:
        STATE vx=<v> vw=<w> batt=<...>
    We'll print that for debug.
    """
    def __init__(self, port=SERIAL_PORT, baud=BAUD):
        self.ser = serial.Serial(port, baud, timeout=0)
        time.sleep(0.2)

    def send_velocity(self, v_lin, v_ang):
        """
        v_lin [m/s], v_ang [rad/s]
        """
        msg = f"CMD V={v_lin:.3f} W={v_ang:.3f}\n"
        self.ser.write(msg.encode("utf-8"))
        # flush to reduce buffering weirdness during debug
        self.ser.flush()
        return msg  # for printing

    def read_state_nonblocking(self):
        """
        Grab whatever ESP sent since last call.
        We'll just dump it to console for now.
        """
        data = self.ser.read(256)
        if not data:
            return None
        return data.decode("utf-8", errors="ignore")

########################################
#     GAMEPAD
########################################

def init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick detected. Connect DualSense.")
        sys.exit(1)
    js = pygame.joystick.Joystick(0)
    js.init()
    print("Using controller:", js.get_name())
    return js

def read_axes(js):
    """
    Read sticks and trigger each frame.
    Returns lx, ly, turbo in [~ -1..1] or [0..1] for turbo.
    You MAY have to adjust axis indices depending on how the DualSense shows up.
    To debug, print all axes on startup.
    """
    pygame.event.pump()

    # NOTE: On many systems:
    # axis 0 = left stick horizontal (-1 left, +1 right)
    # axis 1 = left stick vertical   (-1 up, +1 down)
    # R2 trigger can be axis 4 or 5 depending on driver.
    lx = js.get_axis(0)
    ly = js.get_axis(1)

    turbo_val = 0.0
    if USE_TURBO:
        try:
            raw = js.get_axis(TURBO_AXIS)
            # normalize trigger from [-1..1] to [0..1]
            turbo_val = (raw + 1.0) * 0.5
        except Exception:
            turbo_val = 0.0

    return lx, ly, turbo_val

########################################
#     MAIN TELEOP LOOP
########################################

def main():
    print("Initializing gamepad...")
    js = init_gamepad()

    print("Initializing IMU (MPU6050)...")
    imu = MPU6050GyroYaw()

    print("Opening UART to ESP32 on", SERIAL_PORT)
    link = ESP32Link()

    print("Teleop active. Ctrl+C to quit.")
    print("Sending at", SEND_HZ, "Hz")

    next_t = time.perf_counter()

    while True:
        try:
            # pacing for ~30 Hz loop
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += DT

            # read controller
            lx, ly, turbo = read_axes(js)

            # convert sticks -> chassis velocity command
            v, w = stick_to_cmd(lx, ly, turbo)

            # send command to ESP32
            msg = link.send_velocity(v, w)

            # get yaw estimate from IMU
            yaw_deg = imu.update_and_get_yaw_deg()

            # read any reply/telemetry from ESP32 (non-blocking)
            esp_reply = link.read_state_nonblocking()

            # debug print
            # You can comment this out later when you don't need spam.
            # v is m/s, w is rad/s, yaw_deg is just for you (not sent to ESP).
            line = f"TX:{msg.strip()} | yaw≈{yaw_deg:7.2f}°"
            if esp_reply:
                # esp_reply can contain multiple STATE lines jammed together
                line += f" | ESP:{esp_reply.strip()}"
            print(line)

        except KeyboardInterrupt:
            print("\nSTOP pressed. Sending zero velocity.")
            try:
                link.send_velocity(0.0, 0.0)
            except Exception:
                pass
            break

        except Exception as e:
            # don't die on minor I/O hiccups, just show it
            print("ERR:", e)
            time.sleep(0.05)

########################################
#     ENTRY POINT
########################################

if __name__ == "__main__":
    main()
