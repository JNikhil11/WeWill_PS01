"""
model.py — WeWill PS01
U-Net architecture for joint denoising + 2x super-resolution of KLA SEM images.

Input : [B, 1, 128, 128]  — noisy, low-resolution SEM patch
Output: [B, 1, 256, 256]  — clean, high-resolution restored patch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two Conv2d-BN-ReLU layers. The standard U-Net building block."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Super-Resolution U-Net for joint denoising and 2x upscaling.

    Architecture:
        - 4-level encoder with MaxPool downsampling
        - Bottleneck
        - 4-level decoder with ConvTranspose2d upsampling + skip connections
        - Final SR upsampling layer: 128x128 → 256x256
        - Output: Sigmoid activation maps values to [0, 1]

    Parameters: ~7.4M
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_ch,    base)
        self.enc2 = ConvBlock(base,     base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = ConvBlock(base * 8, base * 16)

        # Decoder
        self.up4  = nn.ConvTranspose2d(base * 16, base * 8,  kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)

        self.up3  = nn.ConvTranspose2d(base * 8,  base * 4,  kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8,  base * 4)

        self.up2  = nn.ConvTranspose2d(base * 4,  base * 2,  kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4,  base * 2)

        self.up1  = nn.ConvTranspose2d(base * 2,  base,      kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2,  base)

        # Super-Resolution upsampling: 128x128 → 256x256
        self.sr_up = nn.ConvTranspose2d(base, base, kernel_size=2, stride=2)

        # Output head
        self.out = nn.Sequential(
            nn.Conv2d(base, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Super-resolution upsampling
        sr = self.sr_up(d1)
        return self.out(sr)


# ─────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss computed in PyTorch (GPU-compatible).
    Returns 1 - SSIM(pred, target), so minimising this loss maximises SSIM.

    Args:
        win (int): Gaussian window size. Default: 11.
        C1, C2 (float): SSIM stability constants.
    """
    def __init__(self, win: int = 11, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.C1, self.C2, self.win = C1, C2, win

        # Build 2D Gaussian kernel
        coords = torch.arange(win, dtype=torch.float) - win // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g /= g.sum()
        kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', kernel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ch  = pred.shape[1]
        w   = self.window.expand(ch, -1, -1, -1)
        pad = self.win // 2

        mu_x   = F.conv2d(pred,         w, padding=pad, groups=ch)
        mu_y   = F.conv2d(target,        w, padding=pad, groups=ch)
        sig_x  = F.conv2d(pred * pred,   w, padding=pad, groups=ch) - mu_x ** 2
        sig_y  = F.conv2d(target * target, w, padding=pad, groups=ch) - mu_y ** 2
        sig_xy = F.conv2d(pred * target, w, padding=pad, groups=ch) - mu_x * mu_y

        num = (2 * mu_x * mu_y + self.C1) * (2 * sig_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sig_x + sig_y + self.C2)

        return 1.0 - (num / den).mean()


class HybridLoss(nn.Module):
    """
    Combined SSIM + L1 loss.

    L = alpha * SSIM_Loss + (1 - alpha) * L1_Loss

    SSIM encourages structural similarity; L1 prevents blurring
    and anchors predictions to ground-truth pixel values.

    Args:
        alpha (float): Weight for SSIM loss. Default: 0.7.
    """
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.ssim  = SSIMLoss()
        self.l1    = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ssim(pred, target) + (1 - self.alpha) * self.l1(pred, target)


# ─────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = UNet().to(device)
    dummy  = torch.randn(2, 1, 128, 128).to(device)
    out    = model(dummy)

    total  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Input  shape : {dummy.shape}')
    print(f'Output shape : {out.shape}')
    print(f'Parameters   : {total / 1e6:.2f}M')
    assert out.shape == (2, 1, 256, 256), 'Shape mismatch!'
    print('[PASS] model.py sanity check passed.')
