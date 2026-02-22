#!/usr/bin/env python3
"""
================================================================
MRI SLICE INTERPOLATION - COMPREHENSIVE EVALUATION SCRIPT
Improved Visualization Layout + Fixed Checkpoint Loading
================================================================

Evaluates:
- Linear interpolation (baseline)
- Nearest neighbor (baseline)
- CNN (EDSR)
- U-Net
- GAN (Basic)
- GAN (Improved)

Outputs:
- PSNR & SSIM metrics
- Beautiful 3-row visual layout (Inputs → Predictions → Error maps)
- CSV, text summary, ranking tables

Author: Shamit Savant
Updated: Nov 2024
================================================================
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ======================================================
# PATHS
# ======================================================
BASE_DIR = "/blue/pinaki.sarder/s.savant/prostate_sr_project"
DATA_DIR = f"{BASE_DIR}/data/processed"
SPLITS_FILE = f"{DATA_DIR}/splits.pkl"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints"
OUTPUT_DIR = f"{BASE_DIR}/final_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# MODEL DEFINITIONS
# ======================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


class EDSR(nn.Module):
    def __init__(self, num_res_blocks=8):
        super().__init__()
        self.entry = nn.Conv2d(2, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.exit = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        return self.exit(self.blocks(self.entry(x)))


class UNetInterpolator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return self.final(d2)


class GANResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


class Generator(nn.Module):
    def __init__(self, num_res_blocks=8):
        super().__init__()
        self.entry = nn.Conv2d(2, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[GANResidualBlock(64) for _ in range(num_res_blocks)])
        self.exit = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        return self.exit(self.blocks(self.entry(x)))


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))


class ImprovedGenerator(nn.Module):
    def __init__(self, num_res_blocks=16):
        super().__init__()
        self.entry = nn.Conv2d(2, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[GANResidualBlock(64) for _ in range(num_res_blocks)])

        # NEW NAME: self.att
        self.att = AttentionBlock(64)

        self.pre_exit = nn.Conv2d(64, 64, 3, padding=1)
        self.exit = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        f = self.entry(x)
        r = self.blocks(f)
        a = self.att(r)
        return self.exit(self.pre_exit(a + f))


# ======================================================
# DATASET (fallback loader included)
# ======================================================

try:
    from dataset import ProstateSliceDataset
except:
    print("dataset.py not found — using fallback dataset.")
    class ProstateSliceDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, split_file, mode="test"):
            with open(split_file, "rb") as f:
                splits = pickle.load(f)

            self.ids = splits[mode]
            self.vols = {
                pid: np.load(os.path.join(data_dir, pid + ".npy")).astype(np.float32)
                for pid in self.ids
            }

            self.index = []
            for pid, V in self.vols.items():
                for i in range(1, V.shape[0] - 1):
                    self.index.append((pid, i))

        def __len__(self):
            return len(self.index)

        def __getitem__(self, idx):
            pid, i = self.index[idx]
            V = self.vols[pid]
            X = np.stack([V[i - 1], V[i + 1]], axis=0)
            y = V[i][None]
            return torch.tensor(X), torch.tensor(y)


# ======================================================
# BASELINES
# ======================================================
def linear_interpolation(x):
    return x.mean(dim=1, keepdim=True)


def nearest_neighbor(x):
    return x[:, 0:1]


# ======================================================
# METRICS
# ======================================================
def calculate_metrics(pred, gt):
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, data_range=1.0)
    return psnr, ssim


# ======================================================
# BEAUTIFUL CLEAN VISUALIZATION (NEW)
# ======================================================
def create_comparison_visualization(sample_idx, inputs, preds, gt, save_dir):

    order = ["Linear", "Nearest", "CNN_EDSR", "CNN_UNet", "GAN_Basic", "GAN_Improved"]
    N = len(order)

    fig = plt.figure(figsize=(4*N, 12))
    gs = fig.add_gridspec(3, N)

    # ---------- ROW 1: Inputs + GT ----------
    ax1 = fig.add_subplot(gs[0, :N//3])
    ax1.imshow(inputs[0], cmap="gray")
    ax1.set_title("Input (i-1)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, N//3:2*N//3])
    ax2.imshow(inputs[1], cmap="gray")
    ax2.set_title("Input (i+1)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2*N//3:])
    ax3.imshow(gt[0], cmap="gray")
    ax3.set_title("Ground Truth")
    ax3.axis("off")

    # ---------- ROW 2: Predictions ----------
    for col, m in enumerate(order):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(preds[m]["pred"], cmap="gray")
        ax.set_title(
            f"{m}\nPSNR={preds[m]['psnr']:.2f}  SSIM={preds[m]['ssim']:.3f}",
            fontsize=10
        )
        ax.axis("off")

    # ---------- ROW 3: Error maps ----------
    for col, m in enumerate(order):
        err = np.abs(preds[m]["pred"] - gt[0])
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(err, cmap="hot", vmin=0, vmax=0.3)
        ax.set_title(f"{m} Error", fontsize=10)
        ax.axis("off")

    # # Shared colorbar
    # cbar = fig.colorbar(im, ax=fig.get_axes(), orientation="horizontal",
    #                     fraction=0.03, pad=0.02)
    # cbar.set_label("Absolute Error")

    # ===== CLEAN SEPARATE COLORBAR BELOW THE ENTIRE FIGURE =====
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # create a new axis below all rows
    cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.015])  
    # [left, bottom, width, height] in figure coordinates

    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Absolute Error", fontsize=10)


    plt.suptitle(f"Sample {sample_idx}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = os.path.join(save_dir, f"comparison_{sample_idx:03d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    return out


# ======================================================
# FIXED CHECKPOINT LOADING (Renames attention.* → att.*)
# ======================================================
def load_model(model_class, ckpt_path, name):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading {name}...")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    # Extract right dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    fixed = {}
    for k, v in state_dict.items():

        # Rename old keys (attention.* → att.*)
        if k.startswith("attention."):
            new_k = k.replace("attention.", "att.")
            print(f"  Renamed: {k} → {new_k}")
            fixed[new_k] = v
        else:
            fixed[k] = v

    model = model_class().to(DEVICE)
    missing, unexpected = model.load_state_dict(fixed, strict=False)

    if missing:
        print("  Missing keys:", missing)
    if unexpected:
        print("  Unexpected keys:", unexpected)

    print(f"✓ Loaded {name}")
    return model.eval()


# ======================================================
# MAIN EVALUATION PIPELINE
# ======================================================
def evaluate_all_models():

    print("\n================ MRI SLICE INTERPOLATION - FINAL EVAL ================\n")

    test_ds = ProstateSliceDataset(DATA_DIR, SPLITS_FILE, "test")
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test samples: {len(test_ds)}")

    # Load models
    models = {
        "CNN_EDSR": load_model(EDSR, f"{CHECKPOINT_DIR}/cnn/cnn_best.pth", "CNN (EDSR)"),
        "CNN_UNet": load_model(UNetInterpolator, f"{CHECKPOINT_DIR}/cnn_unet/cnn_unet_best.pth", "U-Net"),
        "GAN_Basic": load_model(Generator, f"{CHECKPOINT_DIR}/gan/G_best.pth", "GAN Basic"),
        "GAN_Improved": load_model(ImprovedGenerator, f"{CHECKPOINT_DIR}/gan_improved/G_best.pth", "GAN Improved")
    }

    results = {m: {"psnr": [], "ssim": []} for m in
               ["Linear", "Nearest", "CNN_EDSR", "CNN_UNet", "GAN_Basic", "GAN_Improved"]}

    # ---------------- EVALUATE ----------------
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader)):

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            gt_np = y.cpu().numpy()[0]

            preds = {}

            # Baseline 1: Linear
            p = linear_interpolation(x).cpu().numpy()[0, 0]
            ps, ss = calculate_metrics(p, gt_np[0])
            preds["Linear"] = {"pred": p, "psnr": ps, "ssim": ss}
            results["Linear"]["psnr"].append(ps)
            results["Linear"]["ssim"].append(ss)

            # Baseline 2: Nearest
            p = nearest_neighbor(x).cpu().numpy()[0, 0]
            ps, ss = calculate_metrics(p, gt_np[0])
            preds["Nearest"] = {"pred": p, "psnr": ps, "ssim": ss}
            results["Nearest"]["psnr"].append(ps)
            results["Nearest"]["ssim"].append(ss)

            # Models
            for key in ["CNN_EDSR", "CNN_UNet", "GAN_Basic", "GAN_Improved"]:
                p = models[key](x).cpu().numpy()[0, 0]
                ps, ss = calculate_metrics(p, gt_np[0])
                preds[key] = {"pred": p, "psnr": ps, "ssim": ss}
                results[key]["psnr"].append(ps)
                results[key]["ssim"].append(ss)

            # Save visualization for first 10
            if idx < 10:
                create_comparison_visualization(
                    idx,
                    x.cpu().numpy()[0],
                    preds,
                    gt_np,
                    f"{OUTPUT_DIR}/visualizations"
                )

    # ---------------- SAVE SUMMARY ----------------
    summary = f"{OUTPUT_DIR}/results_summary.txt"
    with open(summary, "w") as f:
        f.write("===== FINAL RESULTS =====\n\n")
        for m in results:
            ps = np.mean(results[m]["psnr"])
            ss = np.mean(results[m]["ssim"])
            f.write(f"{m:15s} PSNR={ps:.3f}  SSIM={ss:.4f}\n")

    print(f"\nSaved summary → {summary}")
    print("\nDONE.\n")


# 
# 
# 
# 
# 
# 
# 
# ======================================================
# PAPER FIGURES DIRECTORY
# ======================================================
PAPER_FIG_DIR = f"{BASE_DIR}/paper_figures"
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# ======================================================
# PAPER VISUALIZATION SET 1: CLEAN SIDE-BY-SIDE GRID
# ======================================================
def viz_side_by_side(inputs, preds, gt, sample_idx):
    models = ["Linear", "Nearest", "CNN_EDSR", "CNN_UNet", "GAN_Basic", "GAN_Improved"]

    fig, axs = plt.subplots(2, len(models) + 1, figsize=(4 * (len(models) + 1), 8))

    # ---- Row 1: Ground truth + predictions ----
    axs[0, 0].imshow(gt[0], cmap="gray")
    axs[0, 0].set_title("Ground Truth")
    axs[0, 0].axis("off")

    for i, m in enumerate(models):
        axs[0, i + 1].imshow(preds[m]["pred"], cmap="gray")
        axs[0, i + 1].set_title(m)
        axs[0, i + 1].axis("off")

    # ---- Row 2: predictions with metrics ----
    axs[1, 0].imshow(gt[0], cmap="gray")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Ground Truth")

    for i, m in enumerate(models):
        axs[1, i + 1].imshow(preds[m]["pred"], cmap="gray")
        axs[1, i + 1].axis("off")
        axs[1, i + 1].set_title(f"PSNR={preds[m]['psnr']:.2f}\nSSIM={preds[m]['ssim']:.3f}")

    plt.tight_layout()
    out = f"{PAPER_FIG_DIR}/side_by_side_{sample_idx:03d}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


# ======================================================
# PAPER VISUALIZATION SET 2: ZOOM-IN PATCHES (ROI)
# ======================================================
def viz_zoom_patches(inputs, preds, gt, sample_idx, patch_size=64, top=100, left=100):
    models = ["Linear", "Nearest", "CNN_EDSR", "CNN_UNet", "GAN_Basic", "GAN_Improved"]

    gt_patch = gt[0][top:top+patch_size, left:left+patch_size]

    fig = plt.figure(figsize=(5 * len(models), 10))
    gs = fig.add_gridspec(2, len(models))

    # Row 1 – Prediction patches
    for col, m in enumerate(models):
        patch = preds[m]["pred"][top:top+patch_size, left:left+patch_size]
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(patch, cmap="gray")
        ax.set_title(m)
        ax.axis("off")

    # Row 2 – Error patches
    for col, m in enumerate(models):
        err = np.abs(preds[m]["pred"] - gt[0])
        patch_err = err[top:top+patch_size, left:left+patch_size]
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(patch_err, cmap="hot")
        ax.set_title(f"{m} Error")
        ax.axis("off")

    plt.tight_layout()
    out = f"{PAPER_FIG_DIR}/zoom_{sample_idx:03d}.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    return out


# ======================================================
# PAPER VISUALIZATION SET 3: ERROR-ONLY GRID
# ======================================================
def viz_error_maps(preds, gt, sample_idx):
    models = ["Linear", "Nearest", "CNN_EDSR", "CNN_UNet", "GAN_Basic", "GAN_Improved"]

    fig, axs = plt.subplots(1, len(models), figsize=(5 * len(models), 4))

    for i, m in enumerate(models):
        err = np.abs(preds[m]["pred"] - gt[0])
        axs[i].imshow(err, cmap="hot", vmin=0, vmax=0.3)
        axs[i].set_title(m)
        axs[i].axis("off")

    out = f"{PAPER_FIG_DIR}/errors_{sample_idx:03d}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


# ======================================================
# PAPER VISUALIZATION SET 4: BAR CHART (PSNR + SSIM)
# ======================================================
def viz_metric_barchart(results):
    labels = list(results.keys())
    psnr_vals = [np.mean(results[m]["psnr"]) for m in labels]
    ssim_vals = [np.mean(results[m]["ssim"]) for m in labels]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 0.2, psnr_vals, width=0.4, label="PSNR")
    ax.bar(x + 0.2, ssim_vals, width=0.4, label="SSIM")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()

    out = f"{PAPER_FIG_DIR}/metric_barchart.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    return out



# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    evaluate_all_models()

















# #!/usr/bin/env python3
# """
# ================================================================
# MRI SLICE INTERPOLATION - COMPREHENSIVE EVAL SCRIPT + PAPER FIGURES
# ================================================================
# Author: Shamit Savant
# Updated: Dec 2024
# ================================================================
# """

# import os
# import sys
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity, peak_signal_noise_ratio
# from datetime import datetime
# warnings = __import__("warnings")
# warnings.filterwarnings("ignore")


# # ======================================================
# # PATHS
# # ======================================================
# BASE_DIR = "/blue/pinaki.sarder/s.savant/prostate_sr_project"
# DATA_DIR = f"{BASE_DIR}/data/processed"
# SPLITS_FILE = f"{DATA_DIR}/splits.pkl"
# CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints"
# OUTPUT_DIR = f"{BASE_DIR}/final_evaluation"
# PAPER_FIG_DIR = f"{BASE_DIR}/paper_figures"

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)
# os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# # ======================================================
# # MODEL DEFINITIONS
# # ======================================================

# class ResidualBlock(nn.Module):
#     def __init__(self, channels=64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

#     def forward(self, x):
#         return self.conv2(self.relu(self.conv1(x))) + x


# class EDSR(nn.Module):
#     def __init__(self, num_res_blocks=8):
#         super().__init__()
#         self.entry = nn.Conv2d(2, 64, 3, padding=1)
#         self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
#         self.exit = nn.Conv2d(64, 1, 3, padding=1)

#     def forward(self, x):
#         return self.exit(self.blocks(self.entry(x)))


# class UNetInterpolator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
#         )
#         self.pool1 = nn.MaxPool2d(2)

#         self.enc2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
#         )
#         self.pool2 = nn.MaxPool2d(2)

#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
#         )

#         self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
#         )

#         self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
#         )

#         self.final = nn.Conv2d(64, 1, 1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool1(e1))
#         b = self.bottleneck(self.pool2(e2))
#         d1 = self.up1(b)
#         d1 = self.dec1(torch.cat([d1, e2], dim=1))
#         d2 = self.up2(d1)
#         d2 = self.dec2(torch.cat([d2, e1], dim=1))
#         return self.final(d2)


# class GANResidualBlock(nn.Module):
#     def __init__(self, channels=64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

#     def forward(self, x):
#         return self.conv2(self.relu(self.conv1(x))) + x


# class Generator(nn.Module):
#     def __init__(self, num_res_blocks=8):
#         super().__init__()
#         self.entry = nn.Conv2d(2, 64, 3, padding=1)
#         self.blocks = nn.Sequential(*[GANResidualBlock(64) for _ in range(num_res_blocks)])
#         self.exit = nn.Conv2d(64, 1, 3, padding=1)

#     def forward(self, x):
#         return self.exit(self.blocks(self.entry(x)))


# class AttentionBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv2d(channels, 1, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         return x * self.sigmoid(self.conv(x))


# class ImprovedGenerator(nn.Module):
#     def __init__(self, num_res_blocks=16):
#         super().__init__()
#         self.entry = nn.Conv2d(2, 64, 3, padding=1)
#         self.blocks = nn.Sequential(*[GANResidualBlock(64) for _ in range(num_res_blocks)])
#         self.att = AttentionBlock(64)
#         self.pre_exit = nn.Conv2d(64, 64, 3, padding=1)
#         self.exit = nn.Conv2d(64, 1, 3, padding=1)

#     def forward(self, x):
#         f = self.entry(x)
#         r = self.blocks(f)
#         a = self.att(r)
#         return self.exit(self.pre_exit(a + f))


# # ======================================================
# # DATASET FALLBACK
# # ======================================================
# try:
#     from dataset import ProstateSliceDataset
# except:
#     print("dataset.py missing — using fallback loader.")
#     class ProstateSliceDataset(torch.utils.data.Dataset):
#         def __init__(self, data_dir, split_file, mode="test"):
#             with open(split_file, "rb") as f:
#                 splits = pickle.load(f)
#             self.ids = splits[mode]
#             self.vols = {pid: np.load(os.path.join(data_dir, pid + ".npy")).astype(np.float32)
#                          for pid in self.ids}
#             self.index = [(pid, i)
#                           for pid, V in self.vols.items()
#                           for i in range(1, V.shape[0]-1)]

#         def __len__(self):
#             return len(self.index)

#         def __getitem__(self, idx):
#             pid, i = self.index[idx]
#             V = self.vols[pid]
#             X = np.stack([V[i-1], V[i+1]], axis=0)
#             y = V[i][None]
#             return torch.tensor(X), torch.tensor(y)


# # ======================================================
# # BASELINES
# # ======================================================
# def linear_interpolation(x):
#     return x.mean(dim=1, keepdim=True)

# def nearest_neighbor(x):
#     return x[:, 0:1]


# # ======================================================
# # METRICS
# # ======================================================
# def calculate_metrics(pred, gt):
#     pred = np.clip(pred, 0, 1)
#     gt = np.clip(gt, 0, 1)
#     return (
#         peak_signal_noise_ratio(gt, pred, data_range=1.0),
#         structural_similarity(gt, pred, data_range=1.0)
#     )


# # ======================================================
# # MAIN VISUALIZATION GRID
# # ======================================================
# def create_comparison_visualization(sample_idx, inputs, preds, gt, save_dir):
#     os.makedirs(save_dir, exist_ok=True)

#     order = ["Linear", "Nearest", "CNN_EDSR", "CNN_UNet", "GAN_Basic", "GAN_Improved"]
#     N = len(order)

#     fig = plt.figure(figsize=(4*N, 12))
#     gs = fig.add_gridspec(3, N)

#     # Row 1: Inputs + GT
#     for ax, img, title in zip(
#         [fig.add_subplot(gs[0, :N//3]), fig.add_subplot(gs[0, N//3:2*N//3]), fig.add_subplot(gs[0, 2*N//3:])],
#         [inputs[0], inputs[1], gt[0]],
#         ["Input (i-1)", "Input (i+1)", "Ground Truth"]
#     ):
#         ax.imshow(img, cmap="gray")
#         ax.set_title(title)
#         ax.axis("off")

#     # Row 2: Predictions
#     for col, m in enumerate(order):
#         ax = fig.add_subplot(gs[1, col])
#         ax.imshow(preds[m]["pred"], cmap="gray")
#         ax.set_title(f"{m}\nPSNR={preds[m]['psnr']:.2f}  SSIM={preds[m]['ssim']:.3f}")
#         ax.axis("off")

#     # Row 3: Error maps
#     for col, m in enumerate(order):
#         ax = fig.add_subplot(gs[2, col])
#         err = np.abs(preds[m]["pred"] - gt[0])
#         im = ax.imshow(err, cmap="hot", vmin=0, vmax=0.3)
#         ax.set_title(f"{m} Error")
#         ax.axis("off")

#     # Shared colorbar below figure
#     cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.02])
#     plt.colorbar(im, cax=cbar_ax, orientation="horizontal")

#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     out = f"{save_dir}/comparison_{sample_idx:03d}.png"
#     plt.savefig(out, dpi=150)
#     plt.close()
#     return out


# # ======================================================
# # MODEL LOADING
# # ======================================================
# def load_model(model_class, ckpt_path, name):
#     print(f"\nLoading {name}...")
#     ckpt = torch.load(ckpt_path, map_location=DEVICE)
#     sd = ckpt.get("model_state_dict", ckpt)

#     # Fix attention naming
#     fixed = { (k.replace("attention.", "att.") if k.startswith("attention.") else k): v
#               for k, v in sd.items() }

#     model = model_class().to(DEVICE)
#     missing, unexpected = model.load_state_dict(fixed, strict=False)
#     if missing: print("Missing keys:", missing)
#     if unexpected: print("Unexpected keys:", unexpected)

#     print(f"✓ Loaded {name}")
#     return model.eval()


# # ======================================================
# # PAPER FIGURE: RADAR CHART
# # ======================================================
# def viz_radar_chart(results):
#     import matplotlib.pyplot as plt
#     from math import pi

#     labels = list(results.keys())
#     psnr_vals = [np.mean(results[m]["psnr"]) for m in labels]
#     ssim_vals = [np.mean(results[m]["ssim"]) for m in labels]

#     categories = ["PSNR", "SSIM"]
#     values = [psnr_vals, ssim_vals]

#     N = len(labels)
#     angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

#     fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

#     for metric_idx, metric_name in enumerate(categories):
#         vals = values[metric_idx] + values[metric_idx][:1]
#         ax.plot(angles, vals, linewidth=2, label=metric_name)
#         ax.fill(angles, vals, alpha=0.15)

#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, rotation=15)
#     ax.set_title("Model Comparison Radar Chart", fontweight="bold")
#     ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))

#     out = f"{PAPER_FIG_DIR}/radar_chart.png"
#     plt.savefig(out, dpi=220, bbox_inches="tight")
#     plt.close()
#     return out


# # ======================================================
# # PAPER FIGURE: SSIM MAP
# # ======================================================
# from skimage.metrics import structural_similarity as ssim_map

# def viz_ssim_map(preds, gt, sample_idx, model="CNN_UNet"):
#     pred = preds[model]["pred"]
#     gt_im = gt[0]
#     _, ssim_out = ssim_map(gt_im, pred, data_range=1.0, full=True)

#     plt.figure(figsize=(8,4))
#     plt.subplot(1,2,1)
#     plt.imshow(pred, cmap="gray")
#     plt.title(f"{model} Prediction")
#     plt.axis("off")

#     plt.subplot(1,2,2)
#     plt.imshow(ssim_out, cmap="viridis")
#     plt.title(f"{model} SSIM Map")
#     plt.colorbar()

#     out = f"{PAPER_FIG_DIR}/ssim_map_{model}_{sample_idx:03d}.png"
#     plt.savefig(out, dpi=220, bbox_inches="tight")
#     plt.close()
#     return out


# # ======================================================
# # PAPER FIGURE: PSNR vs SSIM SCATTER
# # ======================================================
# def viz_psnr_ssim_scatter(results):
#     plt.figure(figsize=(10,6))

#     for m in results:
#         plt.scatter(
#             results[m]["psnr"], results[m]["ssim"],
#             s=18, alpha=0.6, label=m
#         )

#     plt.xlabel("PSNR")
#     plt.ylabel("SSIM")
#     plt.title("PSNR vs SSIM Scatter Plot")
#     plt.legend()

#     out = f"{PAPER_FIG_DIR}/psnr_ssim_scatter.png"
#     plt.savefig(out, dpi=220, bbox_inches="tight")
#     plt.close()
#     return out


# # ======================================================
# # MAIN EVALUATION LOOP
# # ======================================================
# def evaluate_all_models():

#     print("\n================ MRI SLICE INTERPOLATION - FINAL EVAL ================\n")

#     test_ds = ProstateSliceDataset(DATA_DIR, SPLITS_FILE, "test")
#     loader = DataLoader(test_ds, batch_size=1, shuffle=False)

#     models = {
#         "CNN_EDSR": load_model(EDSR, f"{CHECKPOINT_DIR}/cnn/cnn_best.pth", "CNN (EDSR)"),
#         "CNN_UNet": load_model(UNetInterpolator, f"{CHECKPOINT_DIR}/cnn_unet/cnn_unet_best.pth", "U-Net"),
#         "GAN_Basic": load_model(Generator, f"{CHECKPOINT_DIR}/gan/G_best.pth", "GAN Basic"),
#         "GAN_Improved": load_model(ImprovedGenerator, f"{CHECKPOINT_DIR}/gan_improved/G_best.pth", "GAN Improved")
#     }

#     results = {m: {"psnr": [], "ssim": []} for m in
#                ["Linear", "Nearest", "CNN_EDSR", "CNN_UNet", "GAN_Basic", "GAN_Improved"]}

#     global_last_preds = None
#     global_last_gt = None

#     # -------------------- EVALUATION --------------------
#     for idx, (x, y) in enumerate(tqdm(loader)):
#         x = x.to(DEVICE)
#         y = y.to(DEVICE)
#         gt_np = y.cpu().numpy()[0]
#         global_last_gt = gt_np

#         preds = {}

#         # Baseline 1
#         p = linear_interpolation(x).detach().cpu().numpy()[0,0]
#         ps, ss = calculate_metrics(p, gt_np[0])
#         preds["Linear"] = {"pred": p, "psnr": ps, "ssim": ss}
#         results["Linear"]["psnr"].append(ps)
#         results["Linear"]["ssim"].append(ss)

#         # Baseline 2
#         p = nearest_neighbor(x).detach().cpu().numpy()[0,0]
#         ps, ss = calculate_metrics(p, gt_np[0])
#         preds["Nearest"] = {"pred": p, "psnr": ps, "ssim": ss}
#         results["Nearest"]["psnr"].append(ps)
#         results["Nearest"]["ssim"].append(ss)

#         # Models
#         for m in ["CNN_EDSR", "CNN_UNet", "GAN_Basic", "GAN_Improved"]:
#             p = models[m](x).detach().cpu().numpy()[0,0]
#             ps, ss = calculate_metrics(p, gt_np[0])
#             preds[m] = {"pred": p, "psnr": ps, "ssim": ss}
#             results[m]["psnr"].append(ps)
#             results[m]["ssim"].append(ss)

#         # Save visualization for first 10
#         if idx < 10:
#             create_comparison_visualization(
#                 idx, x.cpu().numpy()[0], preds, gt_np,
#                 f"{OUTPUT_DIR}/visualizations"
#             )

#         global_last_preds = preds

#     # -------------------- SAVE SUMMARY --------------------
#     summary_path = f"{OUTPUT_DIR}/results_summary.txt"
#     with open(summary_path, "w") as f:
#         f.write("===== FINAL RESULTS =====\n\n")
#         for m in results:
#             f.write(f"{m:15s} PSNR={np.mean(results[m]['psnr']):.3f}  SSIM={np.mean(results[m]['ssim']):.4f}\n")

#     print(f"\nSaved summary → {summary_path}")

#     # -------------------- PAPER FIGURES --------------------
#     print("\nGenerating paper-quality figures...")

#     viz_radar_chart(results)
#     viz_psnr_ssim_scatter(results)
#     viz_ssim_map(global_last_preds, global_last_gt, 0, model="CNN_UNet")

#     print("\nSaved paper figures →", PAPER_FIG_DIR)
#     print("\nDONE.\n")


# # ======================================================
# # ENTRY POINT
# # ======================================================
# if __name__ == "__main__":
#     evaluate_all_models()
