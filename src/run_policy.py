#!/usr/bin/env python3
import os
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn

from gst_cam import GstCam
from collect_demos import (
    ESP32Link,
    MPU6050GyroYaw,
    FRAME_SIZE,
    JPEG_QUALITY,
    MAX_V,
    MAX_W,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="trained_policies/il_policy_final.pt",
                        help="Path to the trained model .pt file")
    return parser.parse_args()

args = parse_args()
MODEL_PATH = args.model

USE_CUDA    = True
IMG_W, IMG_H = 128, 72
SEND_HZ     = 30.0
DT          = 1.0 / SEND_HZ

# zhruba 5 s paměť při 30 Hz -> 150 kroků
MEMORY_STEPS = 150


# =============== Model (stejný jako v tréninku) ===============

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 5 * 8, 128)

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            feats = self.conv(x)
            feats = feats.view(B*T, -1)
            feats = self.fc(feats)
            feats = torch.relu(feats)
            feats = feats.view(B, T, -1)
            return feats
        else:
            feats = self.conv(x)
            feats = feats.view(x.shape[0], -1)
            feats = self.fc(feats)
            feats = torch.relu(feats)
            return feats


class PolicyNet(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.encoder_front = CNNEncoder()
        self.encoder_side  = CNNEncoder()
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, frames_front, frames_side, h0=None):
        f_front = self.encoder_front(frames_front)  # (B,T,128)
        f_side  = self.encoder_side(frames_side)    # (B,T,128)
        feats = torch.cat([f_front, f_side], dim=-1)  # (B,T,256)
        out, h = self.gru(feats, h0)
        actions = self.head(out)
        return actions, h


def preprocess_frame(frame_rgb):
    """
    frame_rgb: H_orig x W_orig x 3 (RGB 0–255)
    return: tensor (1,1,3,H,W) v rozsahu [0,1]
    """
    img = cv2.resize(frame_rgb, (IMG_W, IMG_H))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # C,H,W
    img = np.expand_dims(img, axis=0)   # 1,C,H,W
    img = np.expand_dims(img, axis=0)   # 1,1,C,H,W
    return torch.from_numpy(img)


def clamp(x, lo, hi):
    return hi if x > hi else lo if x < lo else x


def main():
    device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # load model
    model = PolicyNet(hidden_size=128).to(device)
    if not os.path.exists(MODEL_PATH):
        print("Model path not found:", MODEL_PATH)
        return
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model:", MODEL_PATH)

    # init hardware
    print("Init IMU...")
    imu = MPU6050GyroYaw()

    print("Init UART...")
    esp = ESP32Link()

    print("Init front camera...")
    cam_front = GstCam(
        base_dir=".",
        frame_size=FRAME_SIZE,
        jpeg_quality=JPEG_QUALITY,
        sensor_id=0,
        capture_width=640,
        capture_height=480,
        capture_fps=30,
    )

    print("Init side camera...")
    cam_side = GstCam(
        base_dir=".",
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
        print("ERROR: neither camera came up.")
        cam_front.release()
        cam_side.release()
        return

    print("Running IL policy loop (front+side). Ctrl+C to stop.")
    next_t = time.perf_counter()

    # GRU hidden state + počítadlo kroků (pro cca 5 s paměť)
    h = None
    step_count = 0

    try:
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += DT

            step_count += 1
            # každých ~5 s odpoj hidden state (paměť se “resetne”)
            if step_count % MEMORY_STEPS == 0 and h is not None:
                h = h.detach()

            yaw_deg = imu.update_and_get_yaw_delta_deg()

            frame_front_rgb = cam_front.get_frame_rgb()  # H,W,3 RGB
            frame_side_rgb  = cam_side.get_frame_rgb()

            if frame_front_rgb is None and frame_side_rgb is None:
                print("No frames from cameras.")
                continue

            # když náhodou jedna chybí, nahradíme druhou / černou
            if frame_front_rgb is None and frame_side_rgb is not None:
                frame_front_rgb = frame_side_rgb
            if frame_side_rgb is None and frame_front_rgb is not None:
                frame_side_rgb = frame_front_rgb

            inp_front = preprocess_frame(frame_front_rgb).to(device)
            inp_side  = preprocess_frame(frame_side_rgb).to(device)

            with torch.no_grad():
                actions_pred, h = model(inp_front, inp_side, h)
                a = actions_pred[0, -1]  # (2,)

            v_pred = float(a[0].item())
            w_pred = float(a[1].item())

            v_pred = clamp(v_pred, -MAX_V, MAX_V)
            w_pred = clamp(w_pred, -MAX_W, MAX_W)

            msg = esp.send_velocity(v_pred, w_pred)

            print(
                f"v={v_pred:+.3f} w={w_pred:+.3f} yaw={yaw_deg:7.2f} "
                f"| TX:{msg.strip()} "
                f"| step={step_count}"
            )

    except KeyboardInterrupt:
        print("\nStopping IL policy, sending zero velocity.")
        try:
            esp.send_velocity(0.0, 0.0)
        except Exception:
            pass
    finally:
        cam_front.release()
        cam_side.release()
        print("Done.")


if __name__ == "__main__":
    main()
