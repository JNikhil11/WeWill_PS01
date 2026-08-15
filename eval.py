"""
eval.py — WeWill PS01
Standalone evaluation script for the trained KLA SEM restoration model.

Computes:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - Inference speed (images/sec on GPU)

Usage:
    python eval.py --data_dir data/train --weights checkpoints/unet_kla.pth

Run `python eval.py --help` for all options.
"""

import argparse
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import UNet

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print('⚠️  lpips not installed. Run: pip install lpips')


# ─────────────────────────────────────────────────────────
# Dataset (same as train.py — val split only)
# ─────────────────────────────────────────────────────────

class KLASEMDataset(Dataset):
    """
    Loads the validation split (last 10%) of the KLA SEM dataset.
    See train.py for full documentation.
    """
    def __init__(self, root_dir: str, split: str = 'val'):
        super().__init__()
        gt_dir    = os.path.join(root_dir, 'GT')
        noisy_dir = os.path.join(root_dir, 'NoisyLR')

        gt_paths    = sorted(glob.glob(os.path.join(gt_dir,    '*.npy')))
        noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, '*.npy')))

        if len(gt_paths) == 0:
            raise FileNotFoundError(f'No .npy files found in {gt_dir}')

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
        if clean.ndim == 2:
            clean    = clean[np.newaxis]
            degraded = degraded[np.newaxis]
        return torch.from_numpy(degraded).float(), torch.from_numpy(clean).float()


# ─────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device  : {device}')
    print(f'Weights : {args.weights}')

    # Load model
    model = UNet().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    total_p = sum(p.numel() for p in model.parameters())
    print(f'Params  : {total_p / 1e6:.2f}M')

    # Data
    val_ds = KLASEMDataset(args.data_dir, split='val')
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f'Val set : {len(val_ds)} samples\n')

    # LPIPS network
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips_lib.LPIPS(net='alex').to(device)

    # ── Metric accumulation ────────────────────────────────
    all_psnr, all_ssim, all_lpips = [], [], []

    with torch.no_grad():
        for deg, cln in tqdm(val_dl, desc='Evaluating'):
            deg, cln = deg.to(device), cln.to(device)
            rst = model(deg)

            p_arr = rst.cpu().numpy()
            t_arr = cln.cpu().numpy()
            for pi, ti in zip(p_arr, t_arr):
                all_psnr.append(compute_psnr(ti.squeeze(), pi.squeeze(), data_range=1.0))
                all_ssim.append(compute_ssim(ti.squeeze(), pi.squeeze(), data_range=1.0))

            if lpips_fn is not None:
                r3 = rst.repeat(1, 3, 1, 1) * 2 - 1
                c3 = cln.repeat(1, 3, 1, 1) * 2 - 1
                all_lpips.append(lpips_fn(r3, c3).mean().item())

    # ── Inference speed benchmark ──────────────────────────
    dummy = torch.randn(1, 1, 128, 128).to(device)
    with torch.no_grad():
        for _ in range(10):   # warm-up
            model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(200):
            model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    fps = 200 / (time.perf_counter() - t0)

    # ── Print results ──────────────────────────────────────
    print('━' * 48)
    print(f'  PSNR    : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB')
    print(f'  SSIM    : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}')
    if all_lpips:
        print(f'  LPIPS   : {np.mean(all_lpips):.4f}  (lower is better)')
    print(f'  Speed   : {fps:.1f} img/sec  ({1000 / fps:.2f} ms/image) on 128×128 input')
    print('━' * 48)

    # ── Visual comparison grid ─────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    n_vis = min(args.num_vis, len(val_ds))
    fig, axs = plt.subplots(n_vis, 3, figsize=(12, 4 * n_vis))
    if n_vis == 1:
        axs = axs[np.newaxis]

    for col, lbl in enumerate(['Degraded Input (128×128)', 'Restored U-Net (256×256)', 'Ground Truth (256×256)']):
        axs[0, col].set_title(lbl, fontweight='bold', fontsize=10)

    for row in range(n_vis):
        deg, cln = val_ds[row]
        with torch.no_grad():
            rst = model(deg.unsqueeze(0).to(device)).squeeze().cpu()

        p = compute_psnr(cln.squeeze().numpy(), rst.numpy(), data_range=1.0)
        s = compute_ssim(cln.squeeze().numpy(), rst.numpy(), data_range=1.0)

        for col, img in enumerate([deg.squeeze(), rst, cln.squeeze()]):
            axs[row, col].imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
            axs[row, col].axis('off')
            if col == 1:
                axs[row, col].set_xlabel(f'PSNR: {p:.2f} dB | SSIM: {s:.4f}', fontsize=9)

    fig.suptitle('WeWill PS01 — KLA Restoration Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, 'visual_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Visuals : {out_path}')


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate WeWill PS01 U-Net on KLA SEM data')
    p.add_argument('--data_dir',    type=str, default='data/train',
                   help='Path containing GT/ and NoisyLR/ subdirs')
    p.add_argument('--weights',     type=str, default='checkpoints/unet_kla.pth',
                   help='Path to trained model .pth file')
    p.add_argument('--batch_size',  type=int, default=16)
    p.add_argument('--num_vis',     type=int, default=4,
                   help='Number of sample images to include in the visual grid')
    p.add_argument('--output_dir',  type=str, default='results')
    p.add_argument('--num_workers', type=int, default=2)
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
