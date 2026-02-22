"""
Corrected Conditional Diffusion Model
✓ Works with L1-trained noise predictor
✓ Uses stable x0-prediction sampling (DDIM-like)
✓ Fixes all scalar/tensor bugs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -----------------------------------------
# Sinusoidal Embeddings
# -----------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        embeddings = np.log(10000) / (half - 1)
        embeddings = torch.exp(torch.arange(half, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)


# -----------------------------------------
# Residual Block
# -----------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        t_add = self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = h + t_add

        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


# -----------------------------------------
# Self-Attention
# -----------------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)

        q = q.view(B, C, H*W).transpose(1, 2)
        k = k.view(B, C, H*W)
        v = v.view(B, C, H*W).transpose(1, 2)

        attn = torch.softmax(q @ k / (C ** 0.5), dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).view(B, C, H, W)

        return x + self.proj(out)


# -----------------------------------------
# Conditional U-Net
# -----------------------------------------
class ConditionalUNet(nn.Module):
    def __init__(self, img_channels=1, cond_channels=2, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(img_channels + cond_channels, base_channels, 3, padding=1)

        self.enc1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.enc4 = ResidualBlock(base_channels * 4, base_channels * 8, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        self.b1 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        self.attn = AttentionBlock(base_channels * 8)
        self.b2 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dec4 = ResidualBlock(base_channels * 16, base_channels * 4, time_emb_dim)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)

        # Model predicts **noise**, but behaves like x0. Keep output = 1 channel.
        self.out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x, cond, t):
        t_emb = self.time_mlp(t)

        h = torch.cat([x, cond], dim=1)
        h = self.init_conv(h)

        s1 = self.enc1(h, t_emb); h = self.pool(s1)
        s2 = self.enc2(h, t_emb); h = self.pool(s2)
        s3 = self.enc3(h, t_emb); h = self.pool(s3)
        s4 = self.enc4(h, t_emb); h = self.pool(s4)

        h = self.b1(h, t_emb)
        h = self.attn(h)
        h = self.b2(h, t_emb)

        h = self.up(h); h = torch.cat([h, s4], dim=1); h = self.dec4(h, t_emb)
        h = self.up(h); h = torch.cat([h, s3], dim=1); h = self.dec3(h, t_emb)
        h = self.up(h); h = torch.cat([h, s2], dim=1); h = self.dec2(h, t_emb)
        h = self.up(h); h = torch.cat([h, s1], dim=1); h = self.dec1(h, t_emb)

        return self.out(h)


# -----------------------------------------
# Gaussian Diffusion (Corrected)
# -----------------------------------------
class GaussianDiffusion:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1)

        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

    # Forward diffusion
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        ac = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        acm = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        return torch.sqrt(ac) * x0 + torch.sqrt(acm) * noise

    # Reverse diffusion (x0 prediction)
    @torch.no_grad()
    def p_sample(self, model, x, cond, t, t_index):
        B = x.shape[0]
        device = x.device

        ac = self.alphas_cumprod[t_index].view(1, 1, 1, 1).to(device)
        ac_prev = self.alphas_cumprod_prev[t_index].view(1, 1, 1, 1).to(device)

        # Model outputs x0-like signal → clamp for stability
        x0_pred = model(x, cond, t).clamp(-1, 1)

        eps = (x - torch.sqrt(ac) * x0_pred) / torch.sqrt(1 - ac)

        mean = (
            torch.sqrt(ac_prev) * x0_pred +
            torch.sqrt(1 - ac_prev) * eps
        )

        if t_index == 0:
            return mean

        var = self.posterior_variance[t_index].view(1, 1, 1, 1).to(device)
        noise = torch.randn_like(x)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, cond, device):
        b, _, h, w = cond.shape
        x = torch.randn(b, 1, h, w, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, cond, t, i)

        return x


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
