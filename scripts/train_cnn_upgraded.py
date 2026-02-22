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
SPLITS_FILE = f"{DATA_DIR}/splits.pkl"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints/cnn_unet"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ======================================================
# CONFIGURATION
# ======================================================
NUM_EPOCHS = 25  # Total epochs you want to train for
RESUME = True    # Set to False to start fresh


# ======================================================
# SMART CHECKPOINT FINDER
# ======================================================
def find_latest_checkpoint():
    """
    Automatically finds the latest checkpoint by scanning the directory
    for the highest epoch number.
    
    Returns:
        (checkpoint_path, epoch_number) or (None, 0) if no checkpoint found
    """
    checkpoint_files = glob.glob(f"{CHECKPOINT_DIR}/cnn_unet_epoch*.pth")
    
    if not checkpoint_files:
        return None, 0
    
    # Extract epoch numbers from filenames
    epochs = []
    for ckpt_file in checkpoint_files:
        match = re.search(r'cnn_unet_epoch(\d+)\.pth', os.path.basename(ckpt_file))
        if match:
            epoch_num = int(match.group(1))
            epochs.append((epoch_num, ckpt_file))
    
    if not epochs:
        return None, 0
    
    # Sort by epoch number (highest first) and return the latest
    epochs.sort(reverse=True, key=lambda x: x[0])
    latest_epoch, latest_file = epochs[0]
    
    return latest_file, latest_epoch


# ======================================================
# U-NET MODEL
# ======================================================
class UNetInterpolator(nn.Module):
    """U-Net style with skip connections for better detail preservation"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(64, 1, 1)
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder with skip connections
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        
        return self.final(d2)

# ======================================================
# Metrics
# ======================================================
def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100
    return 10 * log10(1 / mse)

def calc_ssim(pred, target):
    return ssim_func(pred, target, data_range=1.0, size_average=True).item()

# ======================================================
# Training
# ======================================================
def train_model():
    train_ds = ProstateSliceDataset(DATA_DIR, SPLITS_FILE, "train", augment=True)
    val_ds = ProstateSliceDataset(DATA_DIR, SPLITS_FILE, "val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    model = UNetInterpolator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    # ======================================================
    # AUTOMATIC CHECKPOINT RESUMPTION
    # ======================================================
    start_epoch = 0
    best_ssim = 0.0
    
    if RESUME:
        ckpt_path, ckpt_epoch = find_latest_checkpoint()
        
        if ckpt_path:
            print(f"\n{'='*60}")
            print(f"CHECKPOINT FOUND: {os.path.basename(ckpt_path)}")
            print(f"RESUMING from epoch {ckpt_epoch}")
            print(f"{'='*60}\n")
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # Handle both old and new checkpoint formats
            if isinstance(checkpoint, dict):
                # New format with full training state
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    best_ssim = checkpoint.get('best_ssim', 0.0)
                    print(f"Loaded model, optimizer, best SSIM: {best_ssim:.4f}")
                else:
                    # Old format - just state dict
                    model.load_state_dict(checkpoint)
                    print("Loaded model weights (old format)")
            else:
                # Very old format - direct state dict
                model.load_state_dict(checkpoint)
                print("Loaded model weights (very old format)")
            
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
        print(f"\n==== Epoch {epoch+1}/{NUM_EPOCHS} ====")

        model.train()
        running_loss = 0
        for x, y in tqdm(train_loader, desc="Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.5f}")

        # === Validation ===
        model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                for i in range(pred.size(0)):
                    psnrs.append(calc_psnr(pred[i:i+1], y[i:i+1]))
                    ssims.append(calc_ssim(pred[i:i+1], y[i:i+1]))

        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)
        
        print(f"Val PSNR: {avg_psnr:.3f} dB")
        print(f"Val SSIM: {avg_ssim:.4f}")

        # ======================================================
        # SAVE CHECKPOINT (new format with full training state)
        # ======================================================
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_psnr': avg_psnr,
            'val_ssim': avg_ssim,
            'best_ssim': best_ssim
        }
        
        # Always save current epoch
        ckpt_path = f"{CHECKPOINT_DIR}/cnn_unet_epoch{epoch+1}.pth"
        torch.save(checkpoint, ckpt_path)
        print(f"Saved: {os.path.basename(ckpt_path)}")
        
        # Save best model
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            checkpoint['best_ssim'] = best_ssim
            torch.save(checkpoint, f"{CHECKPOINT_DIR}/cnn_unet_best.pth")
            print(f"★ NEW BEST SSIM: {best_ssim:.4f} - saved to cnn_unet_best.pth")
        
        # Save training log
        with open(f"{CHECKPOINT_DIR}/training_log.txt", 'a') as f:
            f.write(f"Epoch {epoch+1}: Loss={avg_loss:.5f}, PSNR={avg_psnr:.3f}, SSIM={avg_ssim:.4f}, Best={best_ssim:.4f}\n")

if __name__ == "__main__":
    train_model()