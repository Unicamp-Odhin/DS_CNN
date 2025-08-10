#!/usr/bin/env python
"""
Train a tiny DS‑CNN for binary keyword spotting.

Folder layout:
data/train/keyword/*.wav
data/train/background/*.wav
data/val/keyword/*.wav
data/val/background/*.wav
"""

import argparse, glob, os, random
from pathlib import Path

import torch, torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# 1.  Hyper‑parameters
# ──────────────────────────────────────────────────────────────
SR              = 16000           # expected sample‑rate
WIN_LEN         = 0.025          # 25 ms
WIN_STEP        = 0.010          # 10 ms
N_MFCC          = 12
N_FRAMES        = 399             # ~1 s → ceil((1‑WIN_LEN)/WIN_STEP)+1
BATCH_SIZE      = 32
EPOCHS          = 25
LR              = 1e-3
SEED            = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────
# 2.  Dataset
# ──────────────────────────────────────────────────────────────
from mfcc_dataset_from_csv import MFCCFileDataset


# ──────────────────────────────────────────────────────────────
# 3.  DS‑CNN Model
# ──────────────────────────────────────────────────────────────
class DepthwiseBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_in),
            nn.Conv2d(channels_in, channels_out, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DS_CNN_Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            DepthwiseBlock(32, 64),
            DepthwiseBlock(64, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,) logit


# ──────────────────────────────────────────────────────────────
# 4.  Training / Validation loops
# ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    running = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    running, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        running += loss.item() * x.size(0)
        preds = torch.sigmoid(logits) > 0.5
        correct += (preds.float() == y).sum().item()
    return running / len(loader.dataset), correct / len(loader.dataset)


# ──────────────────────────────────────────────────────────────
# 5.  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--out", type=str, default="dscnn_binary.pt")
    args = parser.parse_args()

    train_set = MFCCFileDataset("mfccs/train")
    val_set   = MFCCFileDataset("mfccs/val")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DS_CNN_Binary().to(device)

    loss_fn  = nn.BCEWithLogitsLoss()
    optim    = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optim, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} "
              f"| val loss {val_loss:.4f} | val acc {val_acc:.3f}")

    torch.save(model.state_dict(), args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
