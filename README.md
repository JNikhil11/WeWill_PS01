# WeWill — PS01: AI-Based Restoration of Degraded Semiconductor Images

> **SEMICON India Hackathon 2026 | Team: WeWill | Problem Statement: KLA PS01**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Problem Statement

KLA's semiconductor inspection tools capture SEM (Scanning Electron Microscopy) images of wafers. In high-throughput inspection, these images suffer from two simultaneous degradations:

- **Speckle Noise** — quantum-level electron scattering corrupts pixel values, masking fine defect edges.
- **Low Resolution** — inspection speed is traded for resolution; images are captured at half the target spatial size.

**Task:** Given a degraded `128×128` noisy SEM image, output a clean, high-resolution `256×256` restored image.

This is a **joint denoising + 2× super-resolution** problem evaluated on PSNR, SSIM, LPIPS, and inference speed.

---

## Our Approach: Super-Resolution U-Net

We implement a custom **U-Net encoder-decoder** architecture with an additional 2× super-resolution upsampling layer at the output. The model is trained end-to-end on the KLA sample dataset using a **Hybrid SSIM + L1 loss function**.

### Architecture

```
Input: [B, 1, 128, 128]  →  Noisy LR SEM image
         ↓
  [ Encoder × 4 levels ]  →  ConvBlocks + MaxPool
  [ Bottleneck ]           →  8×8 feature maps
  [ Decoder × 4 levels ]  →  ConvTranspose2d + Skip Connections
  [ SR Upsampling Layer ] →  ConvTranspose2d (128 → 256)
         ↓
Output: [B, 1, 256, 256] →  Clean HR SEM image
```

- **Parameters:** ~7.4M
- **Loss:** `0.7 × SSIM_Loss + 0.3 × L1_Loss`
- **Optimizer:** Adam (lr=1e-3) with Cosine Annealing LR scheduler

### Why U-Net?

| Method | Quality | Speed | Complexity |
|---|---|---|---|
| **U-Net (Ours)** | ★★★★ | ★★★★★ | Low |
| DnCNN | ★★★ | ★★★★★ | Low — denoising only |
| ESRGAN | ★★★★★ | ★★ | Very High |
| Diffusion Models | ★★★★★ | ★ | Extreme |

U-Net provides the best **quality-to-speed tradeoff** for production wafer inspection, where throughput is a hard constraint.

---

## Dataset

> **⚠️ Disclaimer:** This prototype was trained on the **KLA sample dataset** (3,200 image pairs) provided for the proposal phase. This is a small subset released for evaluation purposes — the full training dataset and held-out test set will be released to qualifying teams after the proposal round. Results shown here reflect sample-set performance, not full-scale training.

**Sample Dataset Structure:**
```
Semicon-samples/
└── train/train/
    ├── GT/          ← 3,200 × (256×256) clean ground truth  [float32, 0–1]
    └── NoisyLR/     ← 3,200 × (128×128) degraded input     [float32, 0–1]
```

---

## Results (20 epochs, KLA Sample Set, T4 GPU)

| Metric | Our Model | Bicubic Baseline |
|---|---|---|
| PSNR | **~27 dB** | ~23 dB |
| SSIM | **0.74** | ~0.60 |
| Training Time | ~15 min (T4) | — |
| Inference Speed | < 5 ms / patch | — |

### Visual Results

![Visual Results](results/kla_visual_results.png)
*Left: Degraded 128×128 Input | Center: U-Net Restored 256×256 | Right: Clean Ground Truth*

### Training Curves

![Training Curves](results/kla_training_curves.png)
*Loss, PSNR, and SSIM over 20 training epochs.*

---

## Repository Structure

```
WeWill_PS01/
├── model.py                            # U-Net architecture definition
├── train.py                            # Standalone training script
├── eval.py                             # Standalone evaluation script
├── WeWill_PS01_Prototype_KLA_Data.ipynb  # Full research notebook (with outputs)
├── requirements.txt                    # Python dependencies
├── results/
│   ├── kla_visual_results.png          # Visual before/after comparisons
│   └── kla_training_curves.png         # Training metric plots
└── README.md
```

> **Note:** The trained model weights `wewill_ps01_unet_kla.pth` (~110 MB) are tracked via [Git LFS](https://git-lfs.github.com/).

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place the KLA dataset in:
```
data/
└── train/
    ├── GT/
    └── NoisyLR/
```

### 3. Train

```bash
python train.py \
  --data_dir data/train \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-3 \
  --save_path checkpoints/unet_kla.pth
```

### 4. Evaluate

```bash
python eval.py \
  --data_dir data/train \
  --weights checkpoints/unet_kla.pth \
  --output_dir results/
```

### 5. Run Full Research Notebook (Colab)

Upload `WeWill_PS01_Prototype_KLA_Data.ipynb` to [Google Colab](https://colab.google) with T4 GPU runtime, mount your Google Drive, and run all cells.

---

## Roadmap (Development Round)

- [ ] Attention Gates in U-Net decoder for surgical defect-region focus
- [ ] VGG Perceptual Loss for finer texture recovery
- [ ] Extended training (50–100 epochs) on full KLA training set
- [ ] H100 inference benchmarking
- [ ] Evaluation on the held-back KLA test set

---

## Team

| Name | Role |
|---|---|
| Nikhil J | ML Engineer |
| [Team Member 2] | [Role] |

**Institution:** [Your College Name]
