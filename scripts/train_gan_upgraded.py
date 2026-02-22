import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from math import log10
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_func

from dataset import ProstateSliceDataset

# ======================================================
# HIPERGATOR PATHS
# ======================================================
BASE_DIR = "/blue/pinaki.sarder/s.savant/prostate_sr_project"
DATA_DIR = f"{BASE_DIR}/data/processed"
SPLITS = f"{DATA_DIR}/splits.pkl"
OUT_DIR = f"{BASE_DIR}/checkpoints/gan_improved"

os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================
# CONFIGURATION
# ======================================================
NUM_EPOCHS = 10  # Total epochs (fewer due to slower training with 16 blocks + attention)
RESUME = True    # Set to False to start fresh


# ======================================================
# SMART CHECKPOINT FINDER FOR IMPROVED GAN
# ======================================================
def find_latest_checkpoint():
    """
    Automatically finds the latest Generator and Discriminator checkpoints
    by scanning the directory for the highest epoch number.
    
    Returns:
        (G_path, D_path, epoch_number) or (None, None, 0) if no checkpoint found
    """
    G_files = glob.glob(f"{OUT_DIR}/G_epoch*.pth")
    D_files = glob.glob(f"{OUT_DIR}/D_epoch*.pth")
    
    if not G_files or not D_files:
        return None, None, 0
    
    # Extract epoch numbers from Generator filenames
    epochs = []
    for g_file in G_files:
        match = re.search(r'G_epoch(\d+)\.pth', os.path.basename(g_file))
        if match:
            epoch_num = int(match.group(1))
            # Check if corresponding D file exists
            d_file = f"{OUT_DIR}/D_epoch{epoch_num}.pth"
            if os.path.exists(d_file):
                epochs.append((epoch_num, g_file, d_file))
    
    if not epochs:
        return None, None, 0
    
    # Sort by epoch number (highest first) and return the latest
    epochs.sort(reverse=True, key=lambda x: x[0])
    latest_epoch, latest_g, latest_d = epochs[0]
    
    return latest_g, latest_d, latest_epoch


# ==== Improved Generator ====

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


class AttentionBlock(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class ImprovedGenerator(nn.Module):
    """Deeper generator with attention and more residual blocks"""
    def __init__(self, num_res_blocks=16):
        super().__init__()
        self.entry = nn.Conv2d(2, 64, 3, padding=1)
        
        # More residual blocks (16 instead of 8)
        self.resblocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )
        
        # Attention mechanism
        self.attention = AttentionBlock(64)
        
        # Additional conv before exit
        self.pre_exit = nn.Conv2d(64, 64, 3, padding=1)
        self.exit = nn.Conv2d(64, 1, 3, padding=1)
    
    def forward(self, x):
        feat = self.entry(x)
        res = self.resblocks(feat)
        att = self.attention(res)
        out = self.pre_exit(att + feat)  # Global residual
        return self.exit(out)


# ==== Improved Discriminator ====

class ImprovedDiscriminator(nn.Module):
    """Deeper discriminator with batch normalization"""
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, stride, use_bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(1, 64, 2, use_bn=False),
            block(64, 128, 2),
            block(128, 256, 2),
            block(256, 512, 1),  # Extra layer
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# ==== Feature Extractor for Feature Matching ====

class FeatureExtractor(nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.features = nn.Sequential(*list(discriminator.model.children())[:-1])
    
    def forward(self, x):
        return self.features(x)


def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    mse = max(mse, 1e-8)
    return 10 * log10(1 / mse)


def calc_ssim(pred, target):
    return ssim_func(pred, target, data_range=1.0, size_average=True).item()


def train_gan():

    train_ds = ProstateSliceDataset(DATA_DIR, SPLITS, "train", augment=True)
    val_ds = ProstateSliceDataset(DATA_DIR, SPLITS, "val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    G = ImprovedGenerator().to(device)
    D = ImprovedDiscriminator().to(device)
    feat_extractor = FeatureExtractor(D).to(device)

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=1e-4)

    # ======================================================
    # AUTOMATIC CHECKPOINT RESUMPTION FOR IMPROVED GAN
    # ======================================================
    start_epoch = 0
    best_ssim = 0.0
    
    if RESUME:
        ckpt_G_path, ckpt_D_path, ckpt_epoch = find_latest_checkpoint()
        
        if ckpt_G_path and ckpt_D_path:
            print(f"\n{'='*60}")
            print(f"CHECKPOINT FOUND:")
            print(f"  Generator: {os.path.basename(ckpt_G_path)}")
            print(f"  Discriminator: {os.path.basename(ckpt_D_path)}")
            print(f"RESUMING from epoch {ckpt_epoch}")
            print(f"{'='*60}\n")
            
            # Load Generator checkpoint
            ckpt_G = torch.load(ckpt_G_path, map_location=device, weights_only=False)
            
            # Handle both old and new checkpoint formats for Generator
            if isinstance(ckpt_G, dict):
                if 'model_state_dict' in ckpt_G:
                    # New format with full training state
                    G.load_state_dict(ckpt_G['model_state_dict'])
                    if 'optimizer_state_dict' in ckpt_G:
                        opt_G.load_state_dict(ckpt_G['optimizer_state_dict'])
                    best_ssim = ckpt_G.get('best_ssim', 0.0)
                    print(f"Loaded Generator (new format), best SSIM: {best_ssim:.4f}")
                else:
                    # Old format - just state dict
                    G.load_state_dict(ckpt_G)
                    print("Loaded Generator (old format)")
            else:
                # Very old format - direct state dict
                G.load_state_dict(ckpt_G)
                print("Loaded Generator (very old format)")
            
            # Load Discriminator checkpoint
            ckpt_D = torch.load(ckpt_D_path, map_location=device, weights_only=False)
            
            # Handle both old and new checkpoint formats for Discriminator
            if isinstance(ckpt_D, dict):
                if 'model_state_dict' in ckpt_D:
                    D.load_state_dict(ckpt_D['model_state_dict'])
                    if 'optimizer_state_dict' in ckpt_D:
                        opt_D.load_state_dict(ckpt_D['optimizer_state_dict'])
                    print("Loaded Discriminator (new format)")
                else:
                    D.load_state_dict(ckpt_D)
                    print("Loaded Discriminator (old format)")
            else:
                D.load_state_dict(ckpt_D)
                print("Loaded Discriminator (very old format)")
            
            # Recreate feature extractor with loaded discriminator
            feat_extractor = FeatureExtractor(D).to(device)
            
            start_epoch = ckpt_epoch
            print(f"Will continue training from epoch {start_epoch + 1}\n")
        else:
            print(f"\nNo checkpoint found. Starting fresh training.\n")
    else:
        print(f"\nRESUME=False. Starting fresh training.\n")

    # ======================================================
    # TRAINING LOOP (now starts from start_epoch instead of 0)
    # ======================================================
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n====== Epoch {epoch+1}/{NUM_EPOCHS} ======")

        G.train()
        D.train()

        for x, y in tqdm(train_loader, desc="Training"):
            x, y = x.to(device), y.to(device)

            # ---- Train D ----
            opt_D.zero_grad()

            real_out = D(y)
            fake = G(x).detach()
            fake_out = D(fake)

            d_loss = 0.5 * (
                adv_loss(real_out, torch.ones_like(real_out)) +
                adv_loss(fake_out, torch.zeros_like(fake_out))
            )
            d_loss.backward()
            opt_D.step()

            # ---- Train G ----
            opt_G.zero_grad()

            gen = G(x)
            pred_fake = D(gen)

            # Feature matching loss
            feat_real = feat_extractor(y).detach()
            feat_fake = feat_extractor(gen)
            feat_match_loss = F.l1_loss(feat_fake, feat_real)

            # Combined loss
            adv = adv_loss(pred_fake, torch.ones_like(pred_fake))
            rec = l1_loss(gen, y)
            
            g_loss = rec + 0.01 * adv + 0.1 * feat_match_loss
            g_loss.backward()
            opt_G.step()

        # ---- Validation ----
        G.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                fake = G(x)
                for i in range(fake.shape[0]):
                    psnrs.append(calc_psnr(fake[i:i+1], y[i:i+1]))
                    ssims.append(calc_ssim(fake[i:i+1], y[i:i+1]))

        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)
        
        print(f"Val PSNR: {avg_psnr:.3f} dB")
        print(f"Val SSIM: {avg_ssim:.4f}")

        # ======================================================
        # SAVE CHECKPOINTS (new format with full training state)
        # ======================================================
        ckpt_G = {
            'epoch': epoch + 1,
            'model_state_dict': G.state_dict(),
            'optimizer_state_dict': opt_G.state_dict(),
            'val_psnr': avg_psnr,
            'val_ssim': avg_ssim,
            'best_ssim': best_ssim
        }
        
        ckpt_D = {
            'epoch': epoch + 1,
            'model_state_dict': D.state_dict(),
            'optimizer_state_dict': opt_D.state_dict()
        }
        
        # Always save current epoch
        g_path = f"{OUT_DIR}/G_epoch{epoch+1}.pth"
        d_path = f"{OUT_DIR}/D_epoch{epoch+1}.pth"
        
        torch.save(ckpt_G, g_path)
        torch.save(ckpt_D, d_path)
        print(f"Saved: {os.path.basename(g_path)}, {os.path.basename(d_path)}")
        
        # Save best model
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            ckpt_G['best_ssim'] = best_ssim
            torch.save(ckpt_G, f"{OUT_DIR}/G_best.pth")
            torch.save(ckpt_D, f"{OUT_DIR}/D_best.pth")
            print(f"★ NEW BEST SSIM: {best_ssim:.4f} - saved to G_best.pth, D_best.pth")
        
        # Save training log
        with open(f"{OUT_DIR}/training_log.txt", 'a') as f:
            f.write(f"Epoch {epoch+1}: PSNR={avg_psnr:.3f}, SSIM={avg_ssim:.4f}, Best={best_ssim:.4f}\n")


if __name__ == "__main__":
    train_gan()