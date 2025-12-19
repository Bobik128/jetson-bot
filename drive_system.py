# drive_system.py
import numpy as np
import time

class DriveSystem:
    def __init__(self, bus, max_lin=0.4, max_ang=1.0):
        self.bus = bus
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.last_cmd_time = time.time()

    def set_body_velocity(self, linear_vel, angular_vel):
        v = float(np.clip(linear_vel, -self.max_lin, self.max_lin))
        w = float(np.clip(angular_vel, -self.max_ang, self.max_ang))
        self.bus.send_velocity_cmd(v, w)
        self.last_cmd_time = time.time()

    def get_feedback(self):
        # expose whatever ESP sends (yaw, vx, vw, batt)
        return self.bus.get_state_snapshot()

    def stop(self):
        self.set_body_velocity(0.0, 0.0)

