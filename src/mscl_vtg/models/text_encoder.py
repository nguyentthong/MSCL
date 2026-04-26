"""Text encoder: lightweight Transformer over pre-extracted text features.

If features are already contextualised (e.g. BERT cls), a single linear
projection suffices.  Otherwise we apply a few Transformer layers.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, K, D = x.shape
        qkv = self.qkv(x).reshape(B, K, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            pad = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,K)
            attn = attn.masked_fill(pad == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, K, D)
        return self.proj(out)


class TextTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TextSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    """Projects text features to hidden_dim, then applies Transformer layers.

    Input:  (B, K, D_t)  pre-extracted text features
    Output: (B, K, D)    contextualised text features
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [TextTransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
