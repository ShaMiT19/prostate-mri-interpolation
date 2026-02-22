"""
Evaluation script for Conditional Diffusion Model
Optimized with proper progress tracking, baseline comparison, and visualizations
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusion_model import ConditionalUNet, GaussianDiffusion
from dataset import ProstateSliceDataset

# ======================================================
# HIPERGATOR PATHS
# ======================================================
BASE_DIR = "/blue/pinaki.sarder/s.savant/prostate_sr_project"
DATA_DIR = f"{BASE_DIR}/data/processed"
SPLITS = f"{DATA_DIR}/splits.pkl"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints/diffusion"
OUTPUT_DIR = f"{BASE_DIR}/results/diffusion"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# CONFIGURATION
# ======================================================
BATCH_SIZE = 8           # Optimized batch size
NUM_WORKERS = 8          # Parallel data loading
NUM_SAMPLES = 200       # Number of samples to evaluate (good statistics)
TIMESTEPS = 100          # MUST MATCH TRAINING (you trained with 100)


@torch.no_grad()
def evaluate(model, diffusion, dataloader, device, num_samples=None):
    """Evaluate model with proper early stopping and progress tracking"""
    
    psnr_vals = []
    ssim_vals = []
    vis_samples = []
    processed = 0

    # Calculate exact number of batches needed
    if num_samples:
        max_batches = (num_samples + dataloader.batch_size - 1) // dataloader.batch_size
        progress_bar = tqdm(dataloader, desc="Evaluating", total=max_batches)
    else:
        progress_bar = tqdm(dataloader, desc="Evaluating")

    for batch_idx, (conditioning, target) in enumerate(progress_bar):
        if num_samples and processed >= num_samples:
            break

        conditioning = conditioning.to(device)
        target = target.to(device)

        # Full DDPM sampling
        generated = diffusion.sample(model, conditioning, device)
        gen_np = generated.clamp(0, 1).cpu().numpy()
        tgt_np = target.cpu().numpy()

        batch_size = gen_np.shape[0]

        for i in range(batch_size):
            if num_samples and processed >= num_samples:
                break

            g = gen_np[i, 0]
            t = tgt_np[i, 0]

            psnr_vals.append(psnr(t, g, data_range=1.0))
            ssim_vals.append(ssim(t, g, data_range=1.0))

            # Save first 5 samples for visualization
            if len(vis_samples) < 5:
                vis_samples.append({
                    "cond1": conditioning[i, 0].cpu().numpy(),
                    "cond2": conditioning[i, 1].cpu().numpy(),
                    "gen": g,
                    "gt": t,
                    "psnr": psnr_vals[-1],
                    "ssim": ssim_vals[-1]
                })

            processed += 1

        # Update progress bar with current metrics
        if psnr_vals:
            progress_bar.set_postfix({
                'PSNR': f'{np.mean(psnr_vals):.2f}',
                'SSIM': f'{np.mean(ssim_vals):.4f}',
                'samples': processed
            })

    return np.array(psnr_vals), np.array(ssim_vals), vis_samples


def compute_baseline(dataloader, device, num_samples=None):
    """Compute linear interpolation baseline"""
    print("\nComputing baseline (linear interpolation)...")
    
    psnr_vals = []
    ssim_vals = []
    processed = 0
    
    if num_samples:
        max_batches = (num_samples + dataloader.batch_size - 1) // dataloader.batch_size
        progress_bar = tqdm(dataloader, desc="Baseline", total=max_batches)
    else:
        progress_bar = tqdm(dataloader, desc="Baseline")
    
    for batch_idx, (conditioning, target) in enumerate(progress_bar):
        if num_samples and processed >= num_samples:
            break
        
        # Linear interpolation = average of adjacent slices
        linear_interp = conditioning.mean(dim=1, keepdim=True)
        linear_np = linear_interp.cpu().numpy()
        target_np = target.cpu().numpy()
        
        batch_size = linear_np.shape[0]
        
        for i in range(batch_size):
            if num_samples and processed >= num_samples:
                break
            
            lin = np.clip(linear_np[i, 0], 0, 1)
            tgt = np.clip(target_np[i, 0], 0, 1)
            
            psnr_vals.append(psnr(tgt, lin, data_range=1.0))
            ssim_vals.append(ssim(tgt, lin, data_range=1.0))
            processed += 1
        
        if psnr_vals:
            progress_bar.set_postfix({
                'PSNR': f'{np.mean(psnr_vals):.2f}',
                'SSIM': f'{np.mean(ssim_vals):.4f}'
            })
    
    return np.array(psnr_vals), np.array(ssim_vals)


def save_visualizations(vis_samples, output_dir):
    """Save visualization samples with comparison to baseline and ground truth"""
    vis_dir = os.path.join(output_dir, "visualizations")
    
    print(f"\nSaving visualizations to {vis_dir}...")
    
    for idx, sample in enumerate(vis_samples):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Linear interpolation baseline
        cond_avg = (sample['cond1'] + sample['cond2']) / 2
        axes[0].imshow(cond_avg, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Linear Interpolation\n(Baseline)')
        axes[0].axis('off')
        
        # Diffusion model output
        axes[1].imshow(sample['gen'], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Diffusion Model\nPSNR: {sample["psnr"]:.2f} dB')
        axes[1].axis('off')
        
        # Ground truth
        axes[2].imshow(sample['gt'], cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        # Error map
        error = np.abs(sample['gen'] - sample['gt'])
        im = axes[3].imshow(error, cmap='hot', vmin=0, vmax=0.3)
        axes[3].set_title(f'Error Map\nSSIM: {sample["ssim"]:.4f}')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'sample_{idx:03d}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(vis_samples)} visualizations")


def main():
    print(f"Device: {DEVICE}")
    print(f"Configuration:")
    print(f"  - Samples to evaluate: {NUM_SAMPLES}")
    print(f"  - Diffusion timesteps: {TIMESTEPS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    
    # Load checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: No checkpoint found in {CHECKPOINT_DIR}")
        sys.exit(1)

    print(f"\nUsing checkpoint: {ckpt_path}")

    # Load model
    model = ConditionalUNet(
        img_channels=1,
        cond_channels=2,
        base_channels=64,
        time_emb_dim=256
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    if 'val_loss' in ckpt:
        print(f"Validation loss at checkpoint: {ckpt['val_loss']:.4f}")

    # Load diffusion
    diffusion = GaussianDiffusion(
        timesteps=TIMESTEPS,
        beta_start=1e-4,
        beta_end=0.02
    )

    # Move all diffusion schedule buffers to device
    for attr in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "posterior_variance"
    ]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(DEVICE))

    # Load dataset
    print("\nLoading test dataset...")
    dataset = ProstateSliceDataset(DATA_DIR, SPLITS, mode="test", augment=False)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        shuffle=True,
        pin_memory=True
    )

    print(f"Total test samples: {len(dataset)}")
    print(f"Will evaluate: {NUM_SAMPLES} samples")
    print(f"Expected batches: {(NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE}\n")

    # Evaluate diffusion model
    print("Evaluating diffusion model...")
    psnr_diff, ssim_diff, vis_samples = evaluate(model, diffusion, loader, DEVICE, NUM_SAMPLES)

    # Compute baseline
    psnr_base, ssim_base = compute_baseline(loader, DEVICE, NUM_SAMPLES)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nSamples evaluated: {len(psnr_diff)}")
    print(f"\nDiffusion Model:")
    print(f"  PSNR: {psnr_diff.mean():.2f} ± {psnr_diff.std():.2f} dB")
    print(f"  SSIM: {ssim_diff.mean():.4f} ± {ssim_diff.std():.4f}")
    print(f"\nLinear Baseline:")
    print(f"  PSNR: {psnr_base.mean():.2f} ± {psnr_base.std():.2f} dB")
    print(f"  SSIM: {ssim_base.mean():.4f} ± {ssim_base.std():.4f}")
    
    psnr_imp = ((psnr_diff.mean() - psnr_base.mean()) / psnr_base.mean() * 100)
    ssim_imp = ((ssim_diff.mean() - ssim_base.mean()) / ssim_base.mean() * 100)
    
    print(f"\nImprovement over Baseline:")
    print(f"  PSNR: {psnr_imp:+.2f}%")
    print(f"  SSIM: {ssim_imp:+.2f}%")
    print("="*80)
    
    # Save visualizations
    save_visualizations(vis_samples, OUTPUT_DIR)
    
    # Save numerical results to file
    results_file = os.path.join(OUTPUT_DIR, "results.txt")
    with open(results_file, 'w') as f:
        f.write("DIFFUSION MODEL EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Epoch: {ckpt.get('epoch', 'unknown')}\n")
        f.write(f"Samples evaluated: {len(psnr_diff)}\n")
        f.write(f"Diffusion timesteps: {TIMESTEPS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n\n")
        
        f.write("Diffusion Model:\n")
        f.write(f"  PSNR: {psnr_diff.mean():.2f} ± {psnr_diff.std():.2f} dB\n")
        f.write(f"  SSIM: {ssim_diff.mean():.4f} ± {ssim_diff.std():.4f}\n\n")
        
        f.write("Linear Baseline:\n")
        f.write(f"  PSNR: {psnr_base.mean():.2f} ± {psnr_base.std():.2f} dB\n")
        f.write(f"  SSIM: {ssim_base.mean():.4f} ± {ssim_base.std():.4f}\n\n")
        
        f.write("Improvement over Baseline:\n")
        f.write(f"  PSNR: {psnr_imp:+.2f}%\n")
        f.write(f"  SSIM: {ssim_imp:+.2f}%\n")
    
    print(f"\nResults saved to: {results_file}")
    print("\nEvaluation complete! ✓")


if __name__ == "__main__":
    main()