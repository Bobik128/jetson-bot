# policy_runner.py
import torch
import time
import numpy as np

class PolicyRunner:
    def __init__(self, robot_interface, policy_path):
        self.iface = robot_interface
        # could be torch.jit.trace()'d model or plain torch.load()
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()

    def _prep_obs(self, obs):
        # image: uint8 [H,W,3] -> torch [1,3,H,W] float32 0..1
        img = obs["image_front"]
        img_t = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)/255.0

        yaw = np.array([[obs["imu_yaw"]]], dtype=np.float32)
        yaw_t = torch.from_numpy(yaw)

        return img_t, yaw_t

    def run_loop(self):
        while True:
            obs = self.iface.get_observation()
            img_t, yaw_t = self._prep_obs(obs)

            with torch.no_grad():
                pred = self.policy(img_t, yaw_t)  # -> [1,2] = [linear_vel, angular_vel]

            v = float(pred[0,0].cpu())
            w = float(pred[0,1].cpu())

            action = {
                "linear_vel": v,
                "angular_vel": w,
            }

            _, info = self.iface.step(action)

            # safety: slow control rate
            time.sleep(0.03)
