#!/usr/bin/env python3
import time
import argparse
import numpy as np

import cv2
import torch
import torch.nn as nn

from gst_cam import GstCam

from shared.constants import MAX_V, MAX_W
from shared.esp32_link import ESP32Link
from shared.imu_mpu6050 import MPU6050GyroYaw
from shared.vision import preprocess_frame_rgb


USE_CUDA     = True
MODEL_PATH   = "trained_policies/il_policy_final.pt"
IMG_W, IMG_H = 128, 72

SEND_HZ      = 30.0
DT           = 1.0 / SEND_HZ

# roughly 5 s memory at 30 Hz
MEMORY_STEPS = 150


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--front", default="/dev/video0")
    p.add_argument("--side",  default="/dev/video1")
    return p.parse_args()


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 9 * 16, 128)

    def forward(self, x):
        # x: (B,T,3,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.conv(x)
        feats = feats.view(B * T, -1)
        feats = torch.relu(self.fc(feats))
        return feats.view(B, T, -1)


class PolicyNet(nn.Module):
    def __init__(self, hidden_size=128, imu_embed=32):
        super().__init__()
        self.encoder_front = CNNEncoder()
        self.encoder_side  = CNNEncoder()

        self.imu_mlp = nn.Sequential(
            nn.Linear(4, imu_embed),
            nn.ReLU(),
            nn.Linear(imu_embed, imu_embed),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=256 + imu_embed,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, front, side, imu, h0=None):
        f_f = self.encoder_front(front)
        f_s = self.encoder_side(side)
        feats_img = torch.cat([f_f, f_s], dim=-1)    # (B,T,256)

        imu_e = self.imu_mlp(imu)                    # (B,T,imu_embed)
        feats = torch.cat([feats_img, imu_e], dim=-1)

        out, h = self.gru(feats, h0)
        actions = self.head(out)
        return actions, h


def preprocess_frame_tensor(frame_rgb, device):
    img_chw = preprocess_frame_rgb(frame_rgb, IMG_W, IMG_H)  # (3,H,W)
    x = img_chw[None, None, ...]  # (1,1,3,H,W)
    return torch.from_numpy(x).to(device)

def preprocess_imu_tensor(dyaw_deg, ax_g, ay_g, az_g, device):
    x = np.array([[[dyaw_deg, ax_g, ay_g, az_g]]], dtype=np.float32)  # (1,1,4)
    return torch.from_numpy(x).to(device)


def main():
    args = parse_args()
    device = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"
    print("Device:", device)

    print("Init ESP32 link...")
    esp = ESP32Link(port=args.port, baud=args.baud)

    print("Init IMU...")
    imu = MPU6050GyroYaw()

    print("Init cameras...")
    cam_front = GstCam(args.front)
    cam_side  = GstCam(args.side)

    print("Load model:", args.model)
    model = PolicyNet(hidden_size=128, imu_embed=32).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    h = None
    step_count = 0

    try:
        while True:
            t0 = time.time()

            step_count += 1
            if step_count % MEMORY_STEPS == 0 and h is not None:
                h = h.detach()

            dyaw_deg = imu.update_and_get_yaw_delta_deg()
            ax_g, ay_g, az_g = imu.read_accel_g()

            frame_front_rgb = cam_front.get_frame_rgb()
            frame_side_rgb  = cam_side.get_frame_rgb()
            if frame_front_rgb is None or frame_side_rgb is None:
                time.sleep(0.01)
                continue

            inp_front = preprocess_frame_tensor(frame_front_rgb, device)
            inp_side  = preprocess_frame_tensor(frame_side_rgb, device)
            inp_imu   = preprocess_imu_tensor(dyaw_deg, ax_g, ay_g, az_g, device)

            with torch.no_grad():
                actions_pred, h = model(inp_front, inp_side, inp_imu, h)

            v = float(actions_pred[0, -1, 0].item())
            w = float(actions_pred[0, -1, 1].item())

            v = max(-MAX_V, min(MAX_V, v))
            w = max(-MAX_W, min(MAX_W, w))

            esp.send_cmd(v, w)

            combined = cv2.hconcat([
                cv2.cvtColor(frame_front_rgb, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(frame_side_rgb,  cv2.COLOR_RGB2BGR),
            ])
            cv2.imshow("Policy (Front | Side)", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            dt = time.time() - t0
            time.sleep(max(0.0, DT - dt))

    finally:
        try:
            cam_front.release()
            cam_side.release()
        except Exception:
            pass
        esp.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
