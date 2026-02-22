import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# === MODEL IMPORTS (adjust if needed) ===
from cnn_model import CNN
from unet import UNet
from gan_model import Generator as GANGen
from gan_improved import ImprovedGenerator

# === Paths ===
CHECKPOINTS = {
    'cnn': 'checkpoints/cnn/cnn_best.pth',
    'unet': 'checkpoints/cnn_unet/cnn_unet_best.pth',
    'gan': 'checkpoints/gan/G_best.pth',
    'gan_improved': 'checkpoints/gan_improved/G_best.pth',
}
DATA_DIR = 'data/processed/'
SAVE_DIR = 'paper_figures/'
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load models ===
def load_model(model_name):
    if model_name == 'cnn':
        model = CNN()
    elif model_name == 'unet':
        model = UNet()
    elif model_name == 'gan':
        model = GANGen()
    elif model_name == 'gan_improved':
        model = ImprovedGenerator()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    ckpt = torch.load(CHECKPOINTS[model_name], map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model

models = {name: load_model(name) for name in CHECKPOINTS}

# === Load data ===
all_samples = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npy')])
selected_samples = all_samples[:5]

# === Utility functions ===
def preprocess(x):
    x = torch.tensor(x, dtype=torch.float32)
    if len(x.shape) == 2:
        x = x.unsqueeze(0)  # (1, H, W)
    return x.to(DEVICE)

def compute_metrics(pred, target):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    psnr_val = psnr(target_np, pred_np, data_range=1.0)
    return psnr_val, ssim_val

def plot_and_save(inputs, preds, gt, filename_prefix):
    methods = list(preds.keys())
    fig, axes = plt.subplots(3, len(methods), figsize=(4 * len(methods), 10))

    for i, method in enumerate(methods):
        pred = preds[method].squeeze().cpu().numpy()
        diff = np.abs(gt - pred)
        axes[0, i].imshow(pred, cmap='gray')
        axes[0, i].set_title(f'{method.upper()} output')
        axes[1, i].imshow(diff, cmap='hot')
        axes[1, i].set_title(f'{method.upper()} error map')
        psnr_val, ssim_val = compute_metrics(torch.tensor(pred), torch.tensor(gt))
        axes[2, i].text(0.1, 0.5, f'PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.3f}', fontsize=14)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{filename_prefix}.png'), dpi=300)
    plt.close()

# === Main Loop ===
for idx, fname in enumerate(selected_samples):
    arr = np.load(os.path.join(DATA_DIR, fname))  # shape: (3, H, W)
    slice_prev, slice_target, slice_next = arr[0], arr[1], arr[2]
    input_tensor = torch.stack([preprocess(slice_prev), preprocess(slice_next)], dim=1).squeeze(0)  # (2, H, W)
    
    gt_tensor = preprocess(slice_target).squeeze(0)

    preds = {}
    with torch.no_grad():
        for name, model in models.items():
            inp = input_tensor.unsqueeze(0).to(DEVICE)  # (1, 2, H, W)
            pred = model(inp).squeeze(0)
            preds[name] = pred.clamp(0, 1)

    plot_and_save((slice_prev, slice_next), preds, slice_target, f'fig_sample{idx}')

print("✅ Figures saved to:", SAVE_DIR)
