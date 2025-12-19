#!/usr/bin/env python3
import os
import glob
import json
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================
# CONFIG
# ============================

DATA_ROOT       = "data"        # kde jsou session složky
USE_CUDA        = True
BATCH_SIZE      = 16
SEQ_LEN         = 150            # délka sekvence (počet snímků) pro trénování
NUM_EPOCHS      = 30
LR              = 1e-4
IMG_W, IMG_H    = 128, 72       # tréninková velikost obrázků
STRIDE          = 3


# ============================
# Episode Loader
# ============================

class Episode:
    def __init__(self, records, episode_dir):
        self.records = records          # list[dict]
        self.episode_dir = episode_dir  # cesta k epXXX

    def __len__(self):
        return len(self.records)


def load_all_episodes(data_root: str) -> List[Episode]:
    """
    Najde všechny epizody a robustně načte JSON řádky.
    Špatné JSON řádky přeskočí.
    """
    episodes: List[Episode] = []
    pattern = os.path.join(data_root, "*", "ep*", "episode.jsonl")
    jsonl_paths = sorted(glob.glob(pattern))

    print(f"Found {len(jsonl_paths)} episode.jsonl files total.")

    for jp in jsonl_paths:
        ep_dir = os.path.dirname(jp)
        recs = []

        with open(jp, "r") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping bad JSON in {jp} line {lineno}: {e}")
                    continue

                # musí být aspoň jedna kamera
                if obj.get("img_front") is None and obj.get("img_side") is None:
                    continue

                recs.append(obj)

        if len(recs) >= SEQ_LEN:
            episodes.append(Episode(recs, ep_dir))
        else:
            print(f"[INFO] Skipping {jp} – only {len(recs)} valid records (< SEQ_LEN {SEQ_LEN}).")

    print(f"Episodes usable for training: {len(episodes)}")
    for i, ep in enumerate(episodes):
        print(f"  Episode {i}: len={len(ep.records)} frames, dir={ep.episode_dir}")

    return episodes


# ============================
# Image helpers
# ============================

def load_image(ep_dir: str, rel_path: str):
    if rel_path is None:
        return None
    abs_path = os.path.join(ep_dir, rel_path)
    img = cv2.imread(abs_path)
    return img


def preprocess_img(img):
    """
    img: BGR nebo None
    → np.array (3,H,W) float32 [0,1] v RGB
    """
    if img is None:
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # C,H,W
    return img


# ============================
# Dataset
# ============================

class ILSequenceDataset(Dataset):
    """
    Vytvoří VŠECHNY možné startovací indexy pro každou epizodu:
      sample = (episode_idx, start_idx)
    Kde start_idx jde od 0 do n-SEQ_LEN.

    Vrací:
      front_frames: (T, C, H, W)
      side_frames:  (T, C, H, W)
      actions:      (T, 2)
    """
    def __init__(self, episodes: List[Episode], seq_len: int):
        self.episodes = episodes
        self.seq_len = seq_len

        # předpočítáme všechny okna (episode_idx, start_idx)
        self.samples: List[Tuple[int, int]] = []
        for ei, ep in enumerate(self.episodes):
            n = len(ep)
            max_start = n - seq_len
            for s in range(0, max_start + 1, STRIDE):
                self.samples.append((ei, s))

        print(f"Total sequences (all episodes): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ei, start = self.samples[idx]
        ep = self.episodes[ei]
        end = start + self.seq_len

        fronts = []
        sides = []
        actions = []

        for i in range(start, end):
            rec = ep.records[i]

            img_front = load_image(ep.episode_dir, rec.get("img_front"))
            img_side  = load_image(ep.episode_dir, rec.get("img_side"))

            fronts.append(preprocess_img(img_front))
            sides.append(preprocess_img(img_side))

            actions.append([
                float(rec["v"]),
                float(rec["w"]),
            ])

        fronts_np  = np.stack(fronts,  axis=0).astype(np.float32)   # (T,C,H,W)
        sides_np   = np.stack(sides,   axis=0).astype(np.float32)   # (T,C,H,W)
        actions_np = np.stack(actions, axis=0).astype(np.float32)   # (T,2)

        return (
            torch.from_numpy(fronts_np),
            torch.from_numpy(sides_np),
            torch.from_numpy(actions_np),
        )


# ============================
# Model
# ============================

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),   # 3x72x128 -> 16x36x64
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 32x18x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 64x9x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> 128x5x8
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 5 * 8, 128)

    def forward(self, x):
        """
        x: (B,T,C,H,W)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.conv(x)
        feats = feats.view(B * T, -1)
        feats = self.fc(feats)
        feats = torch.relu(feats)
        return feats.view(B, T, -1)   # (B,T,128)


class PolicyNet(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.encoder_front = CNNEncoder()
        self.encoder_side  = CNNEncoder()

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, front, side, h0=None):
        f_f = self.encoder_front(front)   # (B,T,128)
        f_s = self.encoder_side(side)     # (B,T,128)
        feats = torch.cat([f_f, f_s], dim=-1)  # (B,T,256)
        out, h = self.gru(feats, h0)          # out: (B,T,H)
        actions = self.head(out)              # (B,T,2)
        return actions, h


# ============================
# Training
# ============================

def train():
    device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    episodes = load_all_episodes(DATA_ROOT)
    if not episodes:
        print("No valid episodes found. Exiting.")
        return

    dataset = ILSequenceDataset(episodes, SEQ_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    approx_batches = len(dataset) // BATCH_SIZE + 1
    print(f"Approx batches per epoch: {approx_batches}")

    model = PolicyNet(hidden_size=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for step, (front, side, actions) in enumerate(dataloader, start=1):
            front   = front.to(device).float()   # (B,T,C,H,W)
            side    = side.to(device).float()
            actions = actions.to(device).float() # (B,T,2)

            optimizer.zero_grad()
            pred, _ = model(front, side)         # (B,T,2)
            loss = criterion(pred, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch}/{NUM_EPOCHS} - steps: {count}, loss: {avg_loss:.6f}")

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("models", f"il_policy_epoch{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join("models", "il_policy_final.pt"))
    print("Training complete -> models/il_policy_final.pt")


if __name__ == "__main__":
    train()
