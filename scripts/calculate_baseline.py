import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as ssim_func
from dataset import ProstateSliceDataset
import torch.nn.functional as F
from math import log10

# ======================================================
# PATHS (match your setup)
# ======================================================
BASE_DIR = "/blue/pinaki.sarder/s.savant/prostate_sr_project"
DATA_DIR = f"{BASE_DIR}/data/processed"
SPLITS_FILE = f"{DATA_DIR}/splits.pkl"

# ======================================================
# Metrics (same as your training scripts)
# ======================================================
def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100
    return 10 * log10(1 / mse)

def calc_ssim(pred, target):
    return ssim_func(pred, target, data_range=1.0, size_average=True).item()

# ======================================================
# Baseline Methods
# ======================================================
def linear_interpolation(x):
    """
    Simple averaging of two input slices
    x: tensor of shape (batch, 2, H, W)
    returns: tensor of shape (batch, 1, H, W)
    """
    # Average the two input slices
    avg = x.mean(dim=1, keepdim=True)
    return avg

def nearest_neighbor(x):
    """
    Just copy the first input slice
    x: tensor of shape (batch, 2, H, W)
    returns: tensor of shape (batch, 1, H, W)
    """
    return x[:, 0:1, :, :]  # Use slice i-1

# ======================================================
# Evaluate Baseline
# ======================================================
def evaluate_baseline():
    # Load validation set
    val_ds = ProstateSliceDataset(DATA_DIR, SPLITS_FILE, "val", augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Validation samples: {len(val_ds)}")
    print("="*60)
    
    # Storage for metrics
    results = {
        'linear': {'psnr': [], 'ssim': []},
        'nearest': {'psnr': [], 'ssim': []}
    }
    
    # Evaluate
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            # Method 1: Linear interpolation (averaging)
            pred_linear = linear_interpolation(x)
            
            # Method 2: Nearest neighbor
            pred_nearest = nearest_neighbor(x)
            
            # Calculate metrics for each sample in batch
            for i in range(x.size(0)):
                # Linear interpolation metrics
                results['linear']['psnr'].append(
                    calc_psnr(pred_linear[i:i+1], y[i:i+1])
                )
                results['linear']['ssim'].append(
                    calc_ssim(pred_linear[i:i+1], y[i:i+1])
                )
                
                # Nearest neighbor metrics
                results['nearest']['psnr'].append(
                    calc_psnr(pred_nearest[i:i+1], y[i:i+1])
                )
                results['nearest']['ssim'].append(
                    calc_ssim(pred_nearest[i:i+1], y[i:i+1])
                )
    
    # Print results
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    
    print("\n1. LINEAR INTERPOLATION (Simple Averaging):")
    print(f"   Average PSNR: {np.mean(results['linear']['psnr']):.3f} dB")
    print(f"   Average SSIM: {np.mean(results['linear']['ssim']):.4f}")
    
    print("\n2. NEAREST NEIGHBOR (Copy closest slice):")
    print(f"   Average PSNR: {np.mean(results['nearest']['psnr']):.3f} dB")
    print(f"   Average SSIM: {np.mean(results['nearest']['ssim']):.4f}")
    
    print("\n" + "="*60)
    print("COMPARISON TO YOUR MODELS:")
    print("="*60)
    print(f"Your U-Net CNN:     PSNR ~29.2 dB, SSIM ~0.883")
    print(f"Your Improved GAN:  PSNR ~29.1 dB, SSIM ~0.882")
    print(f"Linear Baseline:    PSNR ~{np.mean(results['linear']['psnr']):.1f} dB, SSIM ~{np.mean(results['linear']['ssim']):.3f}")
    print("="*60)
    
    improvement_ssim = 0.883 - np.mean(results['linear']['ssim'])
    improvement_psnr = 29.2 - np.mean(results['linear']['psnr'])
    
    print(f"\nYour model improvement over baseline:")
    print(f"  SSIM: +{improvement_ssim:.4f} ({improvement_ssim/np.mean(results['linear']['ssim'])*100:.1f}% better)")
    print(f"  PSNR: +{improvement_psnr:.2f} dB ({improvement_psnr/np.mean(results['linear']['psnr'])*100:.1f}% better)")
    print("="*60)

if __name__ == "__main__":
    evaluate_baseline()