"""Multi-scale video encoder with local self-attention and depthwise strided downsampling.

Paper Eqs. (1)-(4):
  Z^0 = Conv(v_1, ..., v_T)
  Z_bar^l = alpha^l * LocalMSA(LN(Z^{l-1})) + Z^{l-1}
  Z_hat^l = alpha_bar^l * MLP(LN(Z_bar^l)) + Z_bar^l
  Z^l = Downsample(Z_hat^l)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VideoProjection(nn.Module):
    """1D convolution to project raw video features to hidden dim (Eq. 1)."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D_in) -> (B, T, D_out)"""
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.ln(x)


class LocalMultiHeadSelfAttention(nn.Module):
    """Local windowed multi-head self-attention.

    Each position attends only to positions within a local window of size W.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D), mask: (B, T) with 1=valid, 0=pad."""
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
        q, k, v = qkv.unbind(0)

        # Build local attention mask: position i attends to [i-W//2, i+W//2]
        W = self.window_size
        half_w = W // 2
        indices = torch.arange(T, device=x.device)
        # (T, T) relative position mask
        rel = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()  # (T, T)
        local_mask = (rel <= half_w).float()  # 1 if within window

        if mask is not None:
            # Combine with padding mask: keys and queries
            # Key mask: (B, 1, 1, T) — mask out padded keys
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            # Query mask: (B, 1, T, 1) — we'll zero out padded query rows after
            local_mask = local_mask.unsqueeze(0).unsqueeze(0) * key_mask  # (B, 1, T, T)
        else:
            local_mask = local_mask.unsqueeze(0).unsqueeze(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = attn.masked_fill(local_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # Replace NaN rows (from all-inf) with zeros — padded positions
        attn = attn.nan_to_num(0.0)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class DepthwiseDownsample(nn.Module):
    """Strided depthwise 1D convolution for downsampling (Eq. 4).

    Reduces temporal dimension by `stride` (default 2).
    """

    def __init__(self, dim: int, stride: int = 2, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, groups=dim,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: (B, T, D) -> (B, T', D) where T' = ceil(T/stride).

        Returns (x_down, mask_down).
        """
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T', D)
        out = self.norm(out)
        T_out = out.shape[1]
        mask_down = None
        if mask is not None:
            # Downsample mask to exactly match conv output length
            # Use max_pool with same padding logic, then trim/pad to T_out
            mask_down = F.max_pool1d(
                mask.unsqueeze(1),
                kernel_size=self.conv.stride[0],
                stride=self.conv.stride[0],
                padding=0,
            ).squeeze(1)
            # Ensure mask matches feature length
            if mask_down.shape[1] > T_out:
                mask_down = mask_down[:, :T_out]
            elif mask_down.shape[1] < T_out:
                pad = torch.zeros(mask_down.shape[0], T_out - mask_down.shape[1],
                                  device=mask_down.device)
                mask_down = torch.cat([mask_down, pad], dim=1)
        return out, mask_down


class TransformerBlock(nn.Module):
    """One encoder layer: LocalMSA + MLP with learnable per-channel scaling (Eqs. 2-3)."""

    def __init__(self, dim: int, num_heads: int, window_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LocalMultiHeadSelfAttention(dim, num_heads, window_size, dropout)
        self.alpha = nn.Parameter(torch.ones(dim))       # per-channel scale for attn

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)
        self.alpha_bar = nn.Parameter(torch.ones(dim))   # per-channel scale for MLP

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.alpha * self.attn(self.norm1(x), mask) + x
        x = self.alpha_bar * self.mlp(self.norm2(x)) + x
        return x


class VideoEncoder(nn.Module):
    """Multi-scale video encoder producing a feature pyramid {Z^l}_{l=0..L}.

    Returns:
        feats:  list of (B, T_l, D) tensors  [Z^0, Z^1, ..., Z^L]
        masks:  list of (B, T_l) tensors
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        window_size: int = 19,
        downsample_rate: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = VideoProjection(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(hidden_dim, num_heads, window_size, dropout))
            self.downsamples.append(DepthwiseDownsample(hidden_dim, stride=downsample_rate))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        z = self.proj(x)  # Z^0
        feats = [z]
        masks = [mask]

        for layer, ds in zip(self.layers, self.downsamples):
            z = layer(z, mask)
            z, mask = ds(z, mask)
            feats.append(z)
            masks.append(mask)

        return feats, masks
