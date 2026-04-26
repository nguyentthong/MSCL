"""Cross-modal fusion: video ↔ text cross-attention (Eqs. 5–7).

  Z_tilde = LN(Z^l),  E_tilde = LN(E)
  O^l = softmax(Z_tilde^T · E_tilde / sqrt(D)) · Z_tilde
  X^l = beta^l · MLP(LN(O^l)) + O^l
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """Fuse one pyramid level with text features."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.v_norm = nn.LayerNorm(dim)
        self.t_norm = nn.LayerNorm(dim)
        self.scale = dim ** -0.5

        self.out_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.beta = nn.Parameter(torch.ones(dim))  # per-channel scale

    def forward(
        self,
        video: torch.Tensor,       # (B, T_l, D)
        text: torch.Tensor,         # (B, K, D)
        video_mask: torch.Tensor | None = None,   # (B, T_l)
        text_mask: torch.Tensor | None = None,     # (B, K)
    ) -> torch.Tensor:
        z = self.v_norm(video)  # (B, T, D)
        e = self.t_norm(text)   # (B, K, D)

        # Attention: (B, T, D) x (B, D, K) -> (B, T, K)
        attn = torch.bmm(z, e.transpose(1, 2)) * self.scale
        if text_mask is not None:
            attn = attn.masked_fill(text_mask.unsqueeze(1) == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)
        # Weighted sum of text features → text-conditioned video representation
        # NOTE: Paper Eq.6 notation is σ(Z̃^T·Ẽ/√D)·Z̃. We interpret this as
        # standard cross-attention: each video position attends to text tokens
        # and aggregates text features, producing a text-modulated output.
        o = torch.bmm(attn, e)  # (B, T, K) x (B, K, D) -> (B, T, D)

        x = self.beta * self.mlp(self.out_norm(o)) + o
        return x


class MultiScaleFusion(nn.Module):
    """Apply cross-modal fusion at every pyramid level."""

    def __init__(self, num_levels: int, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fusions = nn.ModuleList([CrossModalFusion(dim, dropout) for _ in range(num_levels)])

    def forward(
        self,
        video_feats: list[torch.Tensor],
        video_masks: list[torch.Tensor | None],
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Fuse each pyramid level with text. Skips level 0 (pre-transformer).

        Args:
            video_feats: [Z^0, Z^1, ..., Z^L]  — we fuse levels 1..L
            ...
        Returns:
            fused: list of (B, T_l, D) for levels 1..L
        """
        fused = []
        for i, fusion in enumerate(self.fusions):
            # Fuse level i+1 (skip Z^0 which is pre-transformer)
            level = i + 1
            v = video_feats[level]
            m = video_masks[level] if video_masks[level] is not None else None
            fused.append(fusion(v, text, m, text_mask))
        return fused
