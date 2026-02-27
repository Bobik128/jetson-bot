#!/usr/bin/env python
import logging
import time
from functools import cached_property
from itertools import chain
from typing import Any
from typing import Dict, List, Optional, Tuple

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from .config_jetsonbot import JetsonBotConfig

from .shared.esp32_link import ESP32Link
from .shared.imu_mpu import MPU6050Rates

logger = logging.getLogger(__name__)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def map_range(x, in_min, in_max, out_min, out_max):
    return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)

class JetsonBot(Robot):

    config_class = JetsonBotConfig
    name = "jetsonbot"

    def __init__(self, config: JetsonBotConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                # arm
                "arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.arm_motors = [motor for motor in self.bus.motors if motor.startswith("arm")]

        # Base with 2 wheels
        self.esp_link = ESP32Link(self.config.esp_port, self.config.esp_baud, self.config.esp_timeout)
        self.esp_link.start_reader()

        self.imu = MPU6050Rates(include_accel=True, gyro_units="dps", accel_units="mps2")

        time.sleep(1)
        
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_shoulder_lift.pos",
                "arm_elbow_flex.pos",
                "arm_wrist_flex.pos",
                "arm_gripper.pos",
                "motor_linear.vel",
                "motor_angular.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values()) and self.esp_link.is_connected()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not self.esp_link.is_connected():
            raise ConnectionError(f"Could not connect to ESP32 on port {self.config.esp_port}")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\nRunning calibration of {self}")

        motors = self.arm_motors

        self.bus.disable_torque(self.arm_motors)
        for name in self.arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings(self.arm_motors)

        # homing_offsets.update(dict.fromkeys(self.base_motors, 0))

        full_turn_motor = [
            motor for motor in motors if any(keyword in motor for keyword in ["wheel", "wrist_roll"])
        ]
        unknown_range_motors = [motor for motor in motors if motor not in full_turn_motor]

        print(
            f"Move all arm joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for name in full_turn_motor:
            range_mins[name] = 0
            range_maxes[name] = 4095

        self.calibration = {}
        for name, motor in self.bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        # Set-up arm actuators (position mode)
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        self.bus.disable_torque()
        self.bus.configure_motors()
        for name in self.arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus.write("I_Coefficient", name, 0)
            self.bus.write("D_Coefficient", name, 32)

        self.bus.enable_torque()

    def setup_motors(self) -> None:
        for motor in chain(reversed(self.arm_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read actuators position for arm and vel for base
        start = time.perf_counter()
        arm_pos = self.bus.sync_read("Present_Position", self.arm_motors)

        v, w, age = self.esp_link.get_latest()
        if age is None or age > 0.5:
            v = 0.0
            w = 0.0

        base_vel: dict = {
            "motor_linear.vel": v,
            "motor_angular.vel": w,
        }

        arm_state = {f"{k}.pos": v for k, v in arm_pos.items()}

        imu_sample = self.imu.sample()
        wx, wy, wz = imu_sample.gyro
        vx, vy, vz = imu_sample.vel

        imu_state = {
            "gyro_yaw": float(wx), 
            "accel_x":  float(vx),
        }

        obs_dict = {**arm_state, **base_vel, **imu_state}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
    
    def remap_arm_goal_pos_to_zone(self, arm_goal_pos: Dict[str, float], *, verbose: bool = False) -> Dict[str, float]:
        """
        Keep-out/collision-avoidance remap for LeRobot action dict.

        Input dict keys: "... .pos" (e.g. "arm_shoulder_lift.pos")
        Input value range: 0..100

        Only modifies:
        - arm_shoulder_lift.pos  (id 2)
        - arm_elbow_flex.pos     (id 3)
        - arm_wrist_flex.pos     (id 4)
        """

        import math

        # print(arm_goal_pos)

        K2 = "arm_shoulder_lift.pos"  # id 2
        K3 = "arm_elbow_flex.pos"     # id 3
        K4 = "arm_wrist_flex.pos"     # id 4

        if K2 not in arm_goal_pos or K3 not in arm_goal_pos or K4 not in arm_goal_pos:
            return arm_goal_pos

        # Copy-through by default
        out_goal = dict(arm_goal_pos)

        # ================= PARAMETERS =================
        fillet_r = 2.0      # radius of rounded corner geometry
        keepout_r = 5.0     # repulsion band thickness
        margin = 0.01

        # Forbidden quadrant boundary: x <= bx AND y <= by
        bx = 4.8
        by = -0.6

        # ================= MAP INPUT (0..100 scale) =================
        # Your original ranges were:
        #   u2: 0..0.25
        #   u3: 1..0.66
        #   u4: 1..0.47
        #
        # With 0..100 scaling, that becomes:
        #   u2: 0..25
        #   u3: 100..66
        #   u4: 100..47

        u2 = clamp(arm_goal_pos[K2], -100.0, 100.0)
        u3 = clamp(arm_goal_pos[K3], -100.0, 100.0)
        u4 = clamp(arm_goal_pos[K4], -100.0, 100.0)

        a_deg = map_range(u2, -100.0, -50.0, 125.0, 90.0)
        b_deg = map_range(u3, 100.0, 32.0, 19.0, 90.0)
        c_deg = map_range(u4, 100.0, -6.0, 102.0, 180.0)

        a = math.radians(a_deg)
        b = math.radians(b_deg)
        c = math.radians(c_deg)

        # ================= FORWARD KINEMATICS =================
        x1 = math.cos(a) * 11.6
        y1 = math.sin(a) * 11.6

        omega = -(math.pi - a - b)
        x2 = math.cos(omega) * 10.5
        y2 = math.sin(omega) * 10.5

        fi = omega + (c - math.pi)
        x3 = math.cos(fi) * 5.5
        y3 = math.sin(fi) * 5.5

        finalX = x1 + x2 + x3
        finalY = y1 + y2 + y3

        # ================= ROUNDED SDF =================
        vx = max(finalX - bx, 0.0)
        vy = max(finalY - by, 0.0)

        dist_raw = math.hypot(vx, vy)
        dist = dist_raw - fillet_r

        if verbose:
            print(f"[keepout] X={finalX:.3f}, Y={finalY:.3f}, sdf={dist:.3f}")

        # ================= OUTSIDE KEEP-OUT =================
        if dist > keepout_r:
            return out_goal

        # ================= NORMAL =================
        if dist_raw > 1e-9:
            nx = vx / dist_raw
            ny = vy / dist_raw
        else:
            nx = ny = 1.0 / math.sqrt(2.0)

        # ================= PUSH =================
        target = keepout_r + margin
        push = target - dist
        safeX = finalX + nx * push
        safeY = finalY + ny * push

        if verbose:
            print(f"[keepout] -> safeX={safeX:.3f}, safeY={safeY:.3f}, push={push:.3f}")

        # ================= IK =================
        length = math.hypot(safeX - x3, safeY - y3)
        if length < 1e-6:
            return out_goal

        def _clamp_unit(v: float) -> float:
            return max(-1.0, min(1.0, v))

        alpha2 = math.acos(_clamp_unit((length * length + 11.6 * 11.6 - 10.5 * 10.5) / (2.0 * length * 11.6)))
        beta   = math.acos(_clamp_unit((10.5 * 10.5 + 11.6 * 11.6 - length * length) / (2.0 * 10.5 * 11.6)))
        alpha  = math.atan2(safeY - y3, safeX - x3) + alpha2

        # ================= MAP OUTPUT BACK (0..100 scale) =================
        u2_out = map_range(math.degrees(alpha), 125.0, 90.0, -100.0, -50.0)
        u2_out = clamp(u2_out, -100.0, 100.0)
        out_goal[K2] = u2_out

        u3_out = map_range(math.degrees(beta), 19.0, 90.0, 100.0, 32.0)
        u3_out = clamp(u3_out, -100.0, 100.0)
        out_goal[K3] = u3_out

        # NOTE:
        # Your original code did not update id4 (wrist) in output.
        # If you want to also adjust wrist based on keepout, say so and we’ll extend the IK mapping.

        return out_goal

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command jetsonbot to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        arm_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        if "arm_gripper.pos" in arm_goal_pos:
            arm_goal_pos["arm_gripper.pos"] = 100.0 - arm_goal_pos["arm_gripper.pos"]

        arm_goal_pos = self.remap_arm_goal_pos_to_zone(arm_goal_pos, verbose=False)

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", self.arm_motors)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in arm_goal_pos.items()}
            arm_safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            arm_goal_pos = arm_safe_goal_pos

        # Send goal position to the actuators
        arm_goal_pos_raw = {k.replace(".pos", ""): v for k, v in arm_goal_pos.items()}

        # Only write if we actually have arm targets
        if arm_goal_pos_raw:
            self.bus.sync_write("Goal_Position", arm_goal_pos_raw)

        # Base: allow base-only actions
        v = float(base_goal_vel.get("motor_linear.vel", 0.0))
        w = float(base_goal_vel.get("motor_angular.vel", 0.0))
        self.esp_link.send_cmd(v, w)

        return {**arm_goal_pos, **base_goal_vel}

    def stop_base(self):
        # watchdog stop should NOT tear down the connection
        self.esp_link.send_cmd(0.0, 0.0)
        logger.info("Base motors stopped")

    @check_if_not_connected
    def disconnect(self):
        self.stop_base()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")