#!/usr/bin/env python3
import os
import cv2
import time
import math
import json
import sys
import serial
import pygame
import smbus2
import argparse
from datetime import datetime

from gst_cam import GstCam  # multi-camera capable now


########################################
#                CONFIG
########################################

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_DIR   = os.path.join("data", SESSION_ID)
# We still create a top-level frames dir just for info / compatibility,
# but episodes will use their own subfolders.
FRAMES_DIR = os.path.join(BASE_DIR, "frames_root_unused")
META_PATH  = os.path.join(BASE_DIR, "episode_root_unused.jsonl")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

SERIAL_PORT = "/dev/ttyTHS1"
BAUD        = 115200

SEND_HZ     = 30.0
DT          = 1.0 / SEND_HZ

DEADZONE    = 0.20
EXPO        = 0.25

MAX_V       = 0.30
MAX_W       = 2.00

USE_TURBO   = True
TURBO_AXIS  = 4
TURBO_MIN   = 0.6
TURBO_GAIN  = 1.75

I2C_BUS_IDX     = 7
MPU6050_ADDR    = 0x68
PWR_MGMT_1      = 0x6B
GYRO_XOUT_H     = 0x43
GYRO_SCALE_250  = 131.0

FRAME_SIZE      = (256, 144)
JPEG_QUALITY    = 90

# Gamepad button index for toggling recording
CROSS_BUTTON_IDX = 0  # adjust if your controller maps Cross differently


########################################
#   helper funcs
########################################

def dz(x, dead):
    if abs(x) < dead:
        return 0.0
    return (x - math.copysign(dead, x)) / (1.0 - dead)

def expo_fn(x, k):
    return math.copysign(abs(x) ** (1.0 + k*1.5), x)

def stick_to_cmd(lx, ly, turbo_raw):
    forward_raw = -dz(ly, DEADZONE)
    turn_raw    =  dz(lx, DEADZONE)

    forward_shaped = expo_fn(forward_raw, EXPO)
    turn_shaped    = expo_fn(turn_raw,    EXPO)

    scale = 1.0
    if USE_TURBO and turbo_raw >= TURBO_MIN:
        scale = TURBO_GAIN

    v = forward_shaped * MAX_V * scale
    w = turn_shaped    * MAX_W * scale

    return v, w


########################################
#   IMU
########################################

import errno
from smbus2 import SMBus, i2c_msg

def _open_bus(n, force=True):
    return SMBus(n, force=force)

def _retry(op, tries=5, base=0.01):
    delay = base
    for i in range(tries):
        try:
            return op()
        except OSError as e:
            if e.errno in (errno.EIO, 121):  # Remote I/O (NACK)
                time.sleep(delay)
                delay = min(delay * 2, 0.2)
                continue
            raise
    raise

class MPU6050GyroYaw:
    """Robust MPU6050 yaw integrator with auto bus/address probe + retries.
       Here 'yaw' is integrated from the GYRO X axis (gx)."""

    def __init__(self,
                 bus_idx=I2C_BUS_IDX,
                 addr=MPU6050_ADDR,
                 candidate_buses=(0,1,2,3,4,5,6,7,8,9),
                 candidate_addrs=(0x68, 0x69),
                 force=True):

        self.addr = addr
        self.bus_idx = bus_idx
        self.force = force
        self.bus = None

        # 1) Probe WHO_AM_I on given bus/addr; if it fails, scan candidates.
        def try_open_and_probe(bi, a):
            b = _open_bus(bi, force=self.force)
            # WHO_AM_I should return 0x68 for MPU6050
            w = self._read_reg_retry(b, a, 0x75, 1)[0]
            if w != 0x68:
                b.close()
                raise OSError(f"WHO_AM_I mismatch on bus {bi}, addr 0x{a:02x}: 0x{w:02x}")
            return b

        try:
            self.bus = try_open_and_probe(self.bus_idx, self.addr)
        except Exception:
            # scan for a responding combo
            found = False
            for bi in candidate_buses:
                for a in candidate_addrs:
                    try:
                        self.bus = try_open_and_probe(bi, a)
                        self.bus_idx, self.addr = bi, a
                        found = True
                        break
                    except Exception:
                        pass
                if found:
                    break
            if not found:
                raise RuntimeError(
                    "MPU6050 not found on any candidate bus/address. "
                    "Check wiring/power and run `i2cdetect -y <bus>`."
                )

        # 2) Wake device (PWR_MGMT_1=0), with a short settle time.
        self._write_reg_retry(self.bus, self.addr, PWR_MGMT_1, 0)
        time.sleep(0.05)

        # Bias (raw units) – will be filled by calibration
        self.bias_gx = 0.0
        self.bias_gy = 0.0
        self.bias_gz = 0.0

        # 3) Calibrate gyro while robot is still
        self._calibrate_gyro(n_samples=500, delay=0.002)

        self.last_time = time.time()
        self.yaw_deg = 0.0   # integrated angle from X axis

    # ---- low-level helpers with combined transactions and retries ----

    @staticmethod
    def _read_reg_retry(bus, addr, reg, nbytes):
        def _op():
            w = i2c_msg.write(addr, [reg])
            r = i2c_msg.read(addr, nbytes)
            bus.i2c_rdwr(w, r)  # repeated-start read
            return list(r)
        return _retry(_op)

    @staticmethod
    def _write_reg_retry(bus, addr, reg, val):
        def _op():
            bus.write_byte_data(addr, reg, val)
        return _retry(_op)

    def _read_word(self, reg):
        # Read two bytes with a repeated-start
        b = self._read_reg_retry(self.bus, self.addr, reg, 2)
        val = (b[0] << 8) | b[1]
        if val & 0x8000:
            val = -((0x10000 - val) & 0xFFFF)
        return val

    def _read_gyro_raw_all(self):
        """
        Read raw gyro X/Y/Z (signed int16), bias *not* yet subtracted.
        """
        try:
            gx = self._read_word(GYRO_XOUT_H)       # X
            gy = self._read_word(GYRO_XOUT_H + 2)   # Y
            gz = self._read_word(GYRO_XOUT_H + 4)   # Z
            return gx, gy, gz
        except OSError:
            # attempt one quick reopen and retry if bus flaked
            try:
                if self.bus:
                    try:
                        self.bus.close()
                    except Exception:
                        pass
                self.bus = _open_bus(self.bus_idx, force=self.force)
                gx = self._read_word(GYRO_XOUT_H)
                gy = self._read_word(GYRO_XOUT_H + 2)
                gz = self._read_word(GYRO_XOUT_H + 4)
                return gx, gy, gz
            except Exception as e2:
                raise e2

    def _calibrate_gyro(self, n_samples=500, delay=0.002):
        """
        Measure gyro bias while the sensor is stationary.
        Stores bias in raw units (LSB) for each axis.
        """
        print(f"Calibrating gyro ({n_samples} samples)... keep robot still.")
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0

        for _ in range(n_samples):
            gx, gy, gz = self._read_gyro_raw_all()
            sum_x += gx
            sum_y += gy
            sum_z += gz
            time.sleep(delay)

        self.bias_gx = sum_x / n_samples
        self.bias_gy = sum_y / n_samples
        self.bias_gz = sum_z / n_samples

        print(
            f"Gyro bias (raw): "
            f"gx={self.bias_gx:.1f}, gy={self.bias_gy:.1f}, gz={self.bias_gz:.1f}"
        )

    def read_gyro_dps(self):
        """
        Return current gyro (gx, gy, gz) in deg/s with bias subtracted.
        """
        gx_raw, gy_raw, gz_raw = self._read_gyro_raw_all()

        gx = (gx_raw - self.bias_gx) / GYRO_SCALE_250
        gy = (gy_raw - self.bias_gy) / GYRO_SCALE_250
        gz = (gz_raw - self.bias_gz) / GYRO_SCALE_250

        return gx, gy, gz

    def update_and_get_yaw_delta_deg(self):
        """
        Return only the change in yaw (Δangle in degrees) since the last update.
        Does NOT accumulate an absolute yaw angle.
        """
        gx_dps, _, _ = self.read_gyro_dps()  # deg/s for X axis

        now = time.time()
        dt  = now - self.last_time
        self.last_time = now

        delta_yaw = gx_dps * dt
        return delta_yaw


    def update_and_get_yaw_deg(self):
        """
        Integrate angle from the X axis (gx) and return yaw_deg.
        (Assumes sensor is mounted so that gyro X corresponds to your yaw axis.)
        """
        gx_dps, _, _ = self.read_gyro_dps()  # X-axis in deg/s

        now = time.time()
        dt  = now - self.last_time
        self.last_time = now

        self.yaw_deg += gx_dps * dt
        return self.yaw_deg


########################################
#   ESP32 link
########################################

class ESP32Link:
    def __init__(self, port=SERIAL_PORT, baud=BAUD):
        self.ser = serial.Serial(port, baud, timeout=0)
        time.sleep(0.2)

    def send_velocity(self, v_lin, v_ang):
        msg = f"CMD V={v_lin:.3f} W={v_ang:.3f}\n"
        self.ser.write(msg.encode("utf-8"))
        self.ser.flush()
        return msg

    def read_state_nonblocking(self):
        data = self.ser.read(256)
        if not data:
            return None
        return data.decode("utf-8", errors="ignore")


########################################
#   gamepad
########################################

def init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick detected. Connect controller.")
        sys.exit(1)
    js = pygame.joystick.Joystick(0)
    js.init()
    print("Using controller:", js.get_name())
    return js

def read_axes(js):
    pygame.event.pump()
    lx = js.get_axis(2)
    ly = js.get_axis(1)

    turbo_val = 0.0
    if USE_TURBO:
        try:
            raw = js.get_axis(TURBO_AXIS)
            turbo_val = (raw + 1.0) * 0.5
        except Exception:
            turbo_val = 0.0

    return lx, ly, turbo_val


########################################
#   Episode logger for TWO cams
########################################

class EpisodeLoggerDual:
    """
    Each episode gets its own folder:
      BASE_DIR/epXXX/
        frames/
        episode.jsonl

    JSONL schema:
      {
        "session": session_id,
        "episode": episode_id,
        "t": step_idx,
        "timestamp": ...,
        "img_front": "frames/..._front.jpg" or null,
        "img_side":  "frames/..._side.jpg" or null,
        "yaw_deg": yaw_deg,
        "v": v,
        "w": w
      }
    """

    def __init__(self, base_dir, session_id, cam_front, cam_side):
        self.base_dir    = base_dir
        self.session_id  = session_id

        self.episode_id  = -1   # will become 0 on first start_new_episode()
        self.step_idx    = 0

        self.cam_front   = cam_front
        self.cam_side    = cam_side

        self.episode_dir = None
        self.frames_dir  = None
        self.meta_file   = None

    def _open_new_files_for_episode(self):
        # Close previous meta file if any
        if self.meta_file is not None:
            try:
                self.meta_file.close()
            except Exception:
                pass
            self.meta_file = None

        # Create episode folder and frames subfolder
        self.episode_dir = os.path.join(self.base_dir, f"ep{self.episode_id:03d}")
        self.frames_dir  = os.path.join(self.episode_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        meta_path = os.path.join(self.episode_dir, "episode.jsonl")
        self.meta_file = open(meta_path, "w")

        print(f"Episode directory: {self.episode_dir}")
        print(f"Frames will be saved in: {self.frames_dir}")
        print(f"Meta path: {meta_path}")

    def start_new_episode(self):
        """
        Begin a new logical episode/sequence.
        Resets step_idx and increments episode_id.
        Opens a new episode.jsonl and frames folder.
        """
        self.episode_id += 1
        self.step_idx = 0
        print(f"--- New episode started: episode_id={self.episode_id} ---")
        self._open_new_files_for_episode()

    def log_step(self, frame_front_rgb, frame_side_rgb, yaw_deg, v, w):
        if self.meta_file is None:
            # Should not happen if start_new_episode was called,
            # but guard anyway.
            self.start_new_episode()

        ts = time.time()

        front_rel = None
        side_rel  = None

        if frame_front_rgb is not None:
            img_name_front = (
                f"{self.session_id}_ep{self.episode_id:03d}_{self.step_idx:06d}_front.jpg"
            )
            path_rel_front = os.path.join("frames", img_name_front)
            path_abs_front = os.path.join(self.frames_dir, img_name_front)
            self.cam_front.save_frame(frame_front_rgb, path_abs_front)
            front_rel = path_rel_front

        if frame_side_rgb is not None:
            img_name_side = (
                f"{self.session_id}_ep{self.episode_id:03d}_{self.step_idx:06d}_side.jpg"
            )
            path_rel_side = os.path.join("frames", img_name_side)
            path_abs_side = os.path.join(self.frames_dir, img_name_side)
            self.cam_side.save_frame(frame_side_rgb, path_abs_side)
            side_rel = path_rel_side

        record = {
            "session":   self.session_id,
            "episode":   self.episode_id,
            "t":         self.step_idx,
            "timestamp": ts,
            "img_front": front_rel,
            "img_side":  side_rel,
            "yaw_deg":   yaw_deg,
            "v":         v,
            "w":         w,
        }

        self.meta_file.write(json.dumps(record) + "\n")
        self.meta_file.flush()

        self.step_idx += 1

    def close(self):
        if self.meta_file is not None:
            try:
                self.meta_file.close()
            except Exception:
                pass
            self.meta_file = None


########################################
#   Live Preview (local only now)
########################################

class PreviewManager:
    """
    Local on-screen preview of both cameras using OpenCV windows.
    We scale frames to the requested preview_size so you actually
    see them bigger/smaller.
    Also overlays recording state + episode id on each view.
    """

    def __init__(self, enable_preview, preview_size=(640, 480)):
        self.enable = enable_preview
        self.preview_size = preview_size

        if self.enable:
            cv2.namedWindow("front", cv2.WINDOW_NORMAL)
            cv2.namedWindow("side",  cv2.WINDOW_NORMAL)
            cv2.resizeWindow("front", self.preview_size[0], self.preview_size[1])
            cv2.resizeWindow("side",  self.preview_size[0], self.preview_size[1])

    def update(self, frame_front_bgr, frame_side_bgr, recording, episode_id):
        if not self.enable:
            return

        # Label text
        state = "REC" if recording else "PAUSED"
        label = f"{state} ep{episode_id:03d}"

        # Colors: red when recording, yellow when paused
        if recording:
            color = (0, 0, 255)   # BGR red
        else:
            color = (0, 255, 255) # BGR yellow

        if frame_front_bgr is not None:
            disp_front = cv2.resize(
                frame_front_bgr,
                self.preview_size,
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.putText(
                disp_front,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("front", disp_front)

        if frame_side_bgr is not None:
            disp_side = cv2.resize(
                frame_side_bgr,
                self.preview_size,
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.putText(
                disp_side,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("side", disp_side)

        # Poll key press without blocking
        cv2.waitKey(1)

    def close(self):
        if self.enable:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


########################################
#   main loop
########################################

def main():
    parser = argparse.ArgumentParser(
        description="Teleop demo collector with dual camera logging + local preview"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show cv2 windows with live front/side cameras."
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=640,
        help="Width of on-screen preview window if --preview."
    )
    parser.add_argument(
        "--preview-height",
        type=int,
        default=480,
        help="Height of on-screen preview window if --preview."
    )

    args = parser.parse_args()

    mode_preview = args.preview
    preview_size = (args.preview_width, args.preview_height)

    print("=== COLLECT DEMOS (DUAL CAM) ===")
    print("Session:", SESSION_ID)
    print("Base dir:", BASE_DIR)
    print("(Episodes will be in epXXX/ subfolders)")
    print()
    print(f"Preview enabled: {mode_preview}")
    if mode_preview:
        print(f"Preview size: {preview_size[0]}x{preview_size[1]}")
    print()

    print("[1/6] Init gamepad...")
    js = init_gamepad()

    print("[2/6] Init IMU...")
    imu = MPU6050GyroYaw()

    print("[3/6] Init UART...")
    esp = ESP32Link()

    print("[4/6] Init cameras...")
    # Assume:
    #   sensor_id=0 -> front (IMX219)
    #   sensor_id=1 -> side / high-res (IMX477)
    # If one fails, .alive will be False.

    cam_front = GstCam(
        base_dir=BASE_DIR,
        frame_size=FRAME_SIZE,
        jpeg_quality=JPEG_QUALITY,
        sensor_id=0,
        capture_width=640,
        capture_height=480,
        capture_fps=30,
    )

    cam_side = GstCam(
        base_dir=BASE_DIR,
        frame_size=FRAME_SIZE,
        jpeg_quality=JPEG_QUALITY,
        sensor_id=1,
        capture_width=640,
        capture_height=480,
        capture_fps=30,
    )

    print(f"cam_front alive: {cam_front.alive}")
    print(f"cam_side  alive: {cam_side.alive}")

    if not cam_front.alive and not cam_side.alive:
        print("ERROR: neither camera came up. Cannot record.")
        cam_front.release()
        cam_side.release()
        return

    print("[5/6] Init logger (dual cams)...")
    logger = EpisodeLoggerDual(BASE_DIR, SESSION_ID, cam_front, cam_side)

    print("[6/6] Init preview manager...")
    pv = PreviewManager(
        enable_preview=mode_preview,
        preview_size=preview_size,
    )

    # recording state
    recording   = True      # start with recording enabled
    prev_cross  = 0         # for edge detection on Cross button
    logger.start_new_episode()  # start first episode (ep000)

    print()
    print("Controls live. Drive your robot and collect demonstrations.")
    print("Cross button: toggle recording (new folder/episode when re-enabled).")
    print("Press Ctrl+C to stop and finalize.\n")

    next_t = time.perf_counter()

    try:
        while True:
            # pacing
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += DT

            # joystick -> (v,w)
            lx, ly, turbo_val = read_axes(js)
            v, w = stick_to_cmd(lx, ly, turbo_val)

            # --- Cross button: toggle recording / new episode ---
            cross = js.get_button(CROSS_BUTTON_IDX)
            if cross and not prev_cross:
                if recording:
                    recording = False
                    print("=== RECORDING PAUSED ===")
                else:
                    recording = True
                    logger.start_new_episode()
                    print(f"=== RECORDING RESUMED (episode {logger.episode_id}) ===")
            prev_cross = cross
            # -----------------------------------------------------

            # send cmd
            esp_msg = esp.send_velocity(v, w)

            # imu yaw
            yaw_deg = imu.update_and_get_yaw_delta_deg()

            # grab frames
            frame_front_rgb = None
            frame_side_rgb  = None

            if cam_front.alive:
                try:
                    frame_front_rgb = cam_front.get_frame_rgb()
                except Exception as e:
                    print("front cam err:", e)
                    cam_front.alive = False

            if cam_side.alive:
                try:
                    frame_side_rgb = cam_side.get_frame_rgb()
                except Exception as e:
                    print("side cam err:", e)
                    cam_side.alive = False

            # log to disk (only if recording)
            if recording:
                logger.log_step(frame_front_rgb, frame_side_rgb, yaw_deg, v, w)

            # preview (convert RGB -> BGR for cv2)
            frame_front_bgr = None
            frame_side_bgr  = None

            if frame_front_rgb is not None:
                frame_front_bgr = cv2.cvtColor(frame_front_rgb, cv2.COLOR_RGB2BGR)
            if frame_side_rgb is not None:
                frame_side_bgr = cv2.cvtColor(frame_side_rgb, cv2.COLOR_RGB2BGR)

            pv.update(frame_front_bgr, frame_side_bgr, recording, logger.episode_id)

            # read any telemetry from ESP
            esp_reply = esp.read_state_nonblocking()

            # debug print
            step_str = logger.step_idx - 1 if recording else "NA"
            dbg = (
                f"[REC={'Y' if recording else 'N'} ep={logger.episode_id:03d}] "
                f"step={step_str} "
                f"v={v:+.2f} w={w:+.2f} yaw={yaw_deg:7.2f}° | "
                f"TX:{esp_msg.strip()} | "
                f"front={'Y' if frame_front_rgb is not None else 'N'} "
                f"side={'Y' if frame_side_rgb  is not None else 'N'}"
            )
            if esp_reply:
                dbg += " | ESP:" + esp_reply.strip().replace("\n", " ")
            print(dbg)

    except KeyboardInterrupt:
        print("\nStopping data collection... sending zero velocity.")
        try:
            esp.send_velocity(0.0, 0.0)
        except Exception:
            pass

    finally:
        logger.close()
        cam_front.release()
        cam_side.release()
        pv.close()
        print("Session saved at", BASE_DIR)
        print("Episodes are in epXXX/ subfolders.")
        print("Done.\n")


if __name__ == "__main__":
    main()

