"""
train.py — WeWill PS01
Standalone training script for the KLA SEM image restoration model.

Usage:
    python train.py --data_dir data/train --epochs 20 --batch_size 16

Run `python train.py --help` for all options.
"""

import argparse
import glob
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import HybridLoss, UNet


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────

class KLASEMDataset(Dataset):
    """
    Loads paired KLA SEM .npy files:
        NoisyLR/ → (128, 128) degraded input   [float32, 0–1]
        GT/      → (256, 256) clean target      [float32, 0–1]

    Files are matched by sorted filename order (e.g., 000000.npy ↔ 000000.npy).
    A 90/10 train/val split is applied deterministically.

    Args:
        root_dir (str): Path containing GT/ and NoisyLR/ subdirectories.
        split    (str): 'train' or 'val'.
    """
    def __init__(self, root_dir: str, split: str = 'train'):
        super().__init__()
        gt_dir    = os.path.join(root_dir, 'GT')
        noisy_dir = os.path.join(root_dir, 'NoisyLR')

        gt_paths    = sorted(glob.glob(os.path.join(gt_dir,    '*.npy')))
        noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, '*.npy')))

        if len(gt_paths) == 0:
            raise FileNotFoundError(f'No .npy files found in {gt_dir}')
        if len(gt_paths) != len(noisy_paths):
            raise ValueError(f'GT ({len(gt_paths)}) and NoisyLR ({len(noisy_paths)}) counts mismatch.')

        split_idx = int(0.9 * len(gt_paths))
        if split == 'train':
            self.gt_paths    = gt_paths[:split_idx]
            self.noisy_paths = noisy_paths[:split_idx]
        else:
            self.gt_paths    = gt_paths[split_idx:]
            self.noisy_paths = noisy_paths[split_idx:]

    def __len__(self) -> int:
        return len(self.gt_paths)

    def __getitem__(self, idx: int):
        clean    = np.load(self.gt_paths[idx])
        degraded = np.clip(np.load(self.noisy_paths[idx]), 0.0, 1.0)

        # Add channel dim: (H, W) → (1, H, W)
        if clean.ndim == 2:
            clean    = clean[np.newaxis]
            degraded = degraded[np.newaxis]

        return torch.from_numpy(degraded).float(), torch.from_numpy(clean).float()


# ─────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────

def batch_metrics(pred: torch.Tensor, target: torch.Tensor):
    """Compute mean PSNR and SSIM for a batch on CPU."""
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    psnrs, ssims = [], []
    for pi, ti in zip(p, t):
        psnrs.append(compute_psnr(ti.squeeze(), pi.squeeze(), data_range=1.0))
        ssims.append(compute_ssim(ti.squeeze(), pi.squeeze(), data_range=1.0))
    return float(np.mean(psnrs)), float(np.mean(ssims))


# ─────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────

def train(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device     : {device}')

    # Data
    train_ds = KLASEMDataset(args.data_dir, split='train')
    val_ds   = KLASEMDataset(args.data_dir, split='val')
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    print(f'Train      : {len(train_ds)} samples | Val: {len(val_ds)} samples')

    # Model
    model     = UNet().to(device)
    total_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters : {total_p / 1e6:.2f}M')

    # Optimisation
    criterion = HybridLoss(alpha=args.ssim_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    history   = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}
    best_psnr = 0.0
    t_start   = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        tloss = 0.0
        for deg, cln in tqdm(train_dl, desc=f'[{epoch:02d}/{args.epochs}] Train', leave=False):
            deg, cln = deg.to(device), cln.to(device)
            optimizer.zero_grad()
            loss = criterion(model(deg), cln)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        tloss /= len(train_dl)

        # ── Validate ───────────────────────────────────────
        model.eval()
        vloss = vp = vs = 0.0
        with torch.no_grad():
            for deg, cln in tqdm(val_dl, desc=f'[{epoch:02d}/{args.epochs}] Val  ', leave=False):
                deg, cln = deg.to(device), cln.to(device)
                out = model(deg)
                vloss += criterion(out, cln).item()
                p, s   = batch_metrics(out, cln)
                vp += p; vs += s
        vloss /= len(val_dl)
        vp    /= len(val_dl)
        vs    /= len(val_dl)
        scheduler.step()

        history['train_loss'].append(tloss)
        history['val_loss'].append(vloss)
        history['val_psnr'].append(vp)
        history['val_ssim'].append(vs)

        if vp > best_psnr:
            best_psnr = vp
            torch.save(model.state_dict(), args.save_path)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = (time.time() - t_start) / 60
            print(f'[{epoch:02d}/{args.epochs}] '
                  f'Train: {tloss:.4f} | Val: {vloss:.4f} | '
                  f'PSNR: {vp:.2f} dB | SSIM: {vs:.4f} | '
                  f'Elapsed: {elapsed:.1f} min')

    print(f'\n✅ Best Val PSNR : {best_psnr:.2f} dB')
    print(f'   Weights saved : {args.save_path}')

    # Save history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(history['train_loss'], label='Train', color='royalblue')
    axs[0].plot(history['val_loss'],   label='Val',   color='tomato')
    axs[0].legend(); axs[0].set_title('Loss'); axs[0].set_xlabel('Epoch')
    axs[1].plot(history['val_psnr'], color='seagreen')
    axs[1].set_title('Val PSNR (dB)'); axs[1].set_xlabel('Epoch')
    axs[2].plot(history['val_ssim'], color='darkorange')
    axs[2].set_title('Val SSIM'); axs[2].set_xlabel('Epoch')
    plt.suptitle('WeWill PS01 — Training Metrics', fontweight='bold')
    plt.tight_layout()
    curve_path = os.path.join(args.output_dir, 'training_curves.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'   Curves saved  : {curve_path}')


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train WeWill PS01 U-Net on KLA SEM data')
    p.add_argument('--data_dir',     type=str,   default='data/train',
                   help='Path to directory containing GT/ and NoisyLR/ subdirs')
    p.add_argument('--epochs',       type=int,   default=20)
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--ssim_weight',  type=float, default=0.7,
                   help='Weight for SSIM in hybrid loss (rest is L1)')
    p.add_argument('--save_path',    type=str,   default='checkpoints/unet_kla.pth')
    p.add_argument('--output_dir',   type=str,   default='results')
    p.add_argument('--num_workers',  type=int,   default=2)
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
