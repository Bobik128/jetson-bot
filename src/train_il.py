#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from shared.vision import preprocess_frame_rgb


# ============================
# CONFIG
# ============================

DATA_ROOT   = "../data"
MODEL_OUT   = "../trained_policies/il_policy_final.pt"

SEQ_LEN     = 20
STRIDE      = 5
BATCH_SIZE  = 16
NUM_EPOCHS  = 10
LR          = 1e-3

IMG_W, IMG_H = 128, 72

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(ep_dir, rel_path):
    full = os.path.join(ep_dir, rel_path)
    img_bgr = cv2.imread(full, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(full)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


class Episode:
    def __init__(self, episode_dir):
        self.episode_dir = episode_dir
        self.records = []
        meta = os.path.join(episode_dir, "episode.jsonl")
        if not os.path.exists(meta):
            raise FileNotFoundError(meta)

        with open(meta, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        if len(self.records) < SEQ_LEN:
            raise ValueError(f"Episode too short: {episode_dir}")


class ILSequenceDataset(Dataset):
    """
    Returns:
      front:  (T,3,H,W)
      side:   (T,3,H,W)
      imu:    (T,4) => [dyaw_deg, ax_g, ay_g, az_g]
      action: (T,2) => [v, w]
    """

    def __init__(self, data_root, seq_len=20, stride=5):
        self.seq_len = seq_len
        self.stride = stride
        self.episodes = []
        self.samples = []

        sessions = sorted(glob.glob(os.path.join(data_root, "*")))
        ep_dirs = []
        for s in sessions:
            ep_dirs.extend(sorted(glob.glob(os.path.join(s, "ep*"))))

        for ep_dir in ep_dirs:
            ep = Episode(ep_dir)  # no compatibility mode
            self.episodes.append(ep)

        if not self.episodes:
            raise RuntimeError(f"No episodes found under {data_root}")

        for ei, ep in enumerate(self.episodes):
            n = len(ep.records)
            for start in range(0, n - seq_len + 1, stride):
                self.samples.append((ei, start))

        print(f"Episodes: {len(self.episodes)}")
        print(f"Sequences: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ei, start = self.samples[idx]
        ep = self.episodes[ei]
        end = start + self.seq_len

        fronts, sides, imus, actions = [], [], [], []

        for i in range(start, end):
            rec = ep.records[i]

            img_front = load_image(ep.episode_dir, rec["img_front"])
            img_side  = load_image(ep.episode_dir, rec["img_side"])

            fronts.append(preprocess_frame_rgb(img_front, IMG_W, IMG_H))
            sides.append(preprocess_frame_rgb(img_side,  IMG_W, IMG_H))

            imus.append([
                float(rec["dyaw_deg"]),
                float(rec["ax_g"]),
                float(rec["ay_g"]),
                float(rec["az_g"]),
            ])

            actions.append([
                float(rec["v"]),
                float(rec["w"]),
            ])

        fronts_np  = np.stack(fronts, axis=0).astype(np.float32)   # (T,3,H,W)
        sides_np   = np.stack(sides,  axis=0).astype(np.float32)   # (T,3,H,W)
        imus_np    = np.stack(imus,   axis=0).astype(np.float32)   # (T,4)
        actions_np = np.stack(actions,axis=0).astype(np.float32)   # (T,2)

        return (
            torch.from_numpy(fronts_np),
            torch.from_numpy(sides_np),
            torch.from_numpy(imus_np),
            torch.from_numpy(actions_np),
        )


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
        self.fc = nn.Linear(128 * 9 * 16, 128)  # for 72x128 -> 9x16 after /8

    def forward(self, x):
        # x: (B,T,3,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.conv(x)
        feats = feats.view(B * T, -1)
        feats = torch.relu(self.fc(feats))
        return feats.view(B, T, -1)  # (B,T,128)


class PolicyNet(nn.Module):
    def __init__(self, hidden_size=128, imu_embed=32):
        super().__init__()
        self.encoder_front = CNNEncoder()
        self.encoder_side  = CNNEncoder()

        # Embedding on [dyaw_deg, ax_g, ay_g, az_g]
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
        f_f = self.encoder_front(front)              # (B,T,128)
        f_s = self.encoder_side(side)                # (B,T,128)
        feats_img = torch.cat([f_f, f_s], dim=-1)    # (B,T,256)

        imu_e = self.imu_mlp(imu)                    # (B,T,imu_embed)
        feats = torch.cat([feats_img, imu_e], dim=-1)# (B,T,256+imu_embed)

        out, h = self.gru(feats, h0)
        actions = self.head(out)
        return actions, h


def main():
    ds = ILSequenceDataset(DATA_ROOT, seq_len=SEQ_LEN, stride=STRIDE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    model = PolicyNet(hidden_size=128, imu_embed=32).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0.0
        n = 0

        for step, (front, side, imu, actions) in enumerate(dl, start=1):
            front   = front.to(DEVICE).float()
            side    = side.to(DEVICE).float()
            imu     = imu.to(DEVICE).float()
            actions = actions.to(DEVICE).float()

            pred, _ = model(front, side, imu)
            loss = criterion(pred, actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total += float(loss.item())
            n += 1

            if step % 50 == 0:
                print(f"Epoch {epoch}/{NUM_EPOCHS} step {step}: loss={loss.item():.6f}")

        print(f"Epoch {epoch} avg loss: {total / max(1, n):.6f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True) if os.path.dirname(MODEL_OUT) else None
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved model: {MODEL_OUT}")


if __name__ == "__main__":
    main()
