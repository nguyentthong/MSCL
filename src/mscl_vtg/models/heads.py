"""Classification and boundary regression heads for moment decoding.

Paper Sec. Moment decoding:
  - 1D conv classification head → score p_t^l
  - 1D conv regression head + ReLU → distances (d_s, d_e)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvHead(nn.Module):
    """Stack of 1D conv layers used for both classification and regression."""

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 3, kernel_size: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Conv1d(in_dim, in_dim, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(1, in_dim),  # equivalent to LayerNorm over channels
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, out_dim)"""
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class PredictionHead(nn.Module):
    """Produces classification logits and boundary offset predictions.

    cls_logits: (B, T_l, 1)  — raw logit (apply sigmoid for probability)
    reg_offsets: (B, T_l, 2) — (d_s, d_e), ReLU-activated (non-negative)
    """

    def __init__(self, dim: int, num_layers: int = 3, kernel_size: int = 3):
        super().__init__()
        self.cls_head = ConvHead(dim, 1, num_layers, kernel_size)
        self.reg_head = ConvHead(dim, 2, num_layers, kernel_size)

    def forward(self, x: torch.Tensor):
        cls_logits = self.cls_head(x)           # (B, T, 1)
        reg_offsets = F.relu(self.reg_head(x))  # (B, T, 2)
        return cls_logits.squeeze(-1), reg_offsets
