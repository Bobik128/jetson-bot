# cube_robot_interface.py
import time

class CubeRobotInterface:
    """
    This is the single interface LeRobot-style code will talk to.
    """

    def __init__(self, drive_system, sensors):
        self.drive = drive_system
        self.sensors = sensors

    def reset(self):
        # mobile robot likely can't 'teleport reset'
        # just return first obs
        return self.get_observation()

    def get_observation(self):
        img = self.sensors.get_image_front()
        yaw = self.sensors.get_yaw()

        obs = {
            # visual input
            "image_front": img,      # uint8 HxWx3
            # proprio / robot state
            "imu_yaw": float(yaw),   # consistent units!
        }

        return obs

    def step(self, action):
        """
        action: dict-like or tensor-like with 2 floats:
            { "linear_vel": v, "angular_vel": w }
        """
        v = float(action["linear_vel"])
        w = float(action["angular_vel"])

        # send command to motors
        self.drive.set_body_velocity(v, w)

        # small wait for movement / sensor update
        time.sleep(0.03)  # ~30 Hz

        # gather new obs + low-level feedback for logging
        new_obs = self.get_observation()
        fb = self.drive.get_feedback()

        info = {
            "yaw_feedback": fb.get("yaw", 0.0),
            "vx_feedback": fb.get("vx", 0.0),
            "vw_feedback": fb.get("vw", 0.0),
            "battery": fb.get("batt", 0.0),
        }

        return new_obs, info

    def emergency_stop(self):
        self.drive.stop()
