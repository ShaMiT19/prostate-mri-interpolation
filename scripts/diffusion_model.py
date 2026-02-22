"""
Conditional Diffusion Model for MRI Slice Interpolation
Based on DDPM (Denoising Diffusion Probabilistic Models)
CORRECT VERSION - Noise prediction as trained
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings for diffusion process"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        k = k.reshape(B, C, H * W)  # B, C, HW
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        
        # Attention
        attn = torch.bmm(q, k) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class ConditionalUNet(nn.Module):
    """
    U-Net for conditional diffusion model
    Conditions on adjacent slices (i-1, i+1) to generate intermediate slice (i)
    Outputs: NOISE (not image) - trained with noise prediction objective
    """
    def __init__(self, img_channels=1, cond_channels=2, base_channels=64, 
                 time_emb_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial projection combines noisy image and conditioning
        self.init_conv = nn.Conv2d(img_channels + cond_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.enc1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.enc4 = ResidualBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck with attention
        self.bottleneck1 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        self.attn = AttentionBlock(base_channels * 8)
        self.bottleneck2 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec4 = ResidualBlock(base_channels * 16, base_channels * 4, time_emb_dim)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        
        # Output: 1-channel NOISE map (not image!)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1)
        )

    def forward(self, x, conditioning, timestep):
        """
        Args:
            x: Noisy image [B, 1, H, W]
            conditioning: Adjacent slices [B, 2, H, W]
            timestep: Diffusion timestep [B]
        Returns:
            Predicted NOISE [B, 1, H, W] - NOT the clean image
        """
        # Get time embedding
        t_emb = self.time_mlp(timestep)
        
        # Concatenate noisy image with conditioning
        h = torch.cat([x, conditioning], dim=1)
        h = self.init_conv(h)
        
        # Encoder with skip connections
        skip1 = self.enc1(h, t_emb)
        h = self.pool(skip1)
        
        skip2 = self.enc2(h, t_emb)
        h = self.pool(skip2)
        
        skip3 = self.enc3(h, t_emb)
        h = self.pool(skip3)
        
        skip4 = self.enc4(h, t_emb)
        h = self.pool(skip4)
        
        # Bottleneck
        h = self.bottleneck1(h, t_emb)
        h = self.attn(h)
        h = self.bottleneck2(h, t_emb)
        
        # Decoder with skip connections
        h = self.up(h)
        h = torch.cat([h, skip4], dim=1)
        h = self.dec4(h, t_emb)
        
        h = self.up(h)
        h = torch.cat([h, skip3], dim=1)
        h = self.dec3(h, t_emb)
        
        h = self.up(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.dec2(h, t_emb)
        
        h = self.up(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec1(h, t_emb)
        
        # Output predicted noise
        return self.out_conv(h)


class GaussianDiffusion:
    """
    Gaussian Diffusion Process for training and sampling
    Standard DDPM implementation with NOISE prediction
    """
    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to x_start
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, conditioning, t, noise=None):
        """
        Training loss: predict noise at timestep t
        This is the standard DDPM training objective
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise to ground truth
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = model(x_noisy, conditioning, t)
        
        # MSE loss between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, conditioning, t, t_index):
        """
        Reverse diffusion: single step denoising
        Standard DDPM sampling using predicted noise
        
        Args:
            model: the noise prediction model
            x: current noisy image [B, 1, H, W]
            conditioning: adjacent slices [B, 2, H, W]
            t: timestep tensor for model input [B]
            t_index: scalar timestep index for schedule lookup
        """
        device = x.device
        
        # Extract schedule values for this timestep
        betas_t = self.betas[t_index].to(device).view(1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index].to(device).view(1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t_index]).to(device).view(1, 1, 1, 1)
        
        # Model predicts noise
        predicted_noise = model(x, conditioning, t)
        
        # Compute mean of posterior distribution p(x_{t-1} | x_t)
        # This is the standard DDPM formula from Ho et al. 2020
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            # At t=0, return mean without adding noise
            return model_mean
        else:
            # Sample from posterior distribution
            posterior_variance_t = self.posterior_variance[t_index].to(device).view(1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, conditioning, device):
        """
        Full reverse diffusion process: generate image from noise
        Iteratively denoises from T -> 0 using the model
        """
        batch_size = conditioning.shape[0]
        img_size = conditioning.shape[-1]
        
        # Start from pure Gaussian noise
        img = torch.randn(batch_size, 1, img_size, img_size, device=device)
        
        # Reverse diffusion loop (T -> 0)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, conditioning, t, i)
        
        return img


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ConditionalUNet().to(device)
    diffusion = GaussianDiffusion(timesteps=200)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    conditioning = torch.randn(batch_size, 2, 256, 256).to(device)
    x_start = torch.randn(batch_size, 1, 256, 256).to(device)
    t = torch.randint(0, diffusion.timesteps, (batch_size,)).to(device)
    
    # Test training
    loss = diffusion.p_losses(model, x_start, conditioning, t)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling
    sample = diffusion.sample(model, conditioning, device)
    print(f"Generated sample shape: {sample.shape}")
    
    print("Model test passed!")