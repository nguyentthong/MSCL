"""Full MSCL-VTG model: video encoder → text encoder → fusion → heads.

The forward pass returns everything needed to compute all four losses
and to decode predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from mscl_vtg.config import ModelConfig
from .video_encoder import VideoEncoder
from .text_encoder import TextEncoder
from .fusion import MultiScaleFusion
from .heads import PredictionHead


@dataclass
class ModelOutput:
    """Structured output from the model's forward pass."""
    cls_logits: list[torch.Tensor]      # L tensors of (B, T_l)
    reg_offsets: list[torch.Tensor]     # L tensors of (B, T_l, 2)
    video_feats: list[torch.Tensor]     # L+1 tensors [Z^0, Z^1, ..., Z^L] — for contrastive loss
    fused_feats: list[torch.Tensor]     # L tensors [X^1, ..., X^L] — after fusion
    masks: list[torch.Tensor | None]    # L+1 masks


class MSCLModel(nn.Module):
    """Multi-Scale Contrastive Learning model for Video Temporal Grounding."""

    def __init__(
        self,
        video_feat_dim: int,
        text_feat_dim: int,
        cfg: ModelConfig | None = None,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        D = cfg.hidden_dim
        L = cfg.num_layers

        self.video_encoder = VideoEncoder(
            in_dim=video_feat_dim,
            hidden_dim=D,
            num_layers=L,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            downsample_rate=cfg.downsample_rate,
            dropout=cfg.dropout,
        )
        self.text_encoder = TextEncoder(
            in_dim=text_feat_dim,
            hidden_dim=D,
            num_layers=cfg.text_encoder_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.fusion = MultiScaleFusion(num_levels=L, dim=D, dropout=cfg.dropout)

        # One shared head applied at each pyramid level
        self.head = PredictionHead(D, num_layers=cfg.head_layers, kernel_size=cfg.head_kernel_size)

        self.num_layers = L
        self.downsample_rate = cfg.downsample_rate

    def forward(
        self,
        video_feat: torch.Tensor,       # (B, T, D_v)
        video_mask: torch.Tensor,        # (B, T)
        text_feat: torch.Tensor,         # (B, K, D_t)
        text_mask: torch.Tensor,         # (B, K)
    ) -> ModelOutput:
        # 1. Encode video → feature pyramid [Z^0, ..., Z^L]
        video_feats, video_masks = self.video_encoder(video_feat, video_mask)

        # 2. Encode text
        text_enc = self.text_encoder(text_feat, text_mask)  # (B, K, D)

        # 3. Cross-modal fusion at levels 1..L
        fused = self.fusion(video_feats, video_masks, text_enc, text_mask)  # L items

        # 4. Prediction heads at each fused level
        cls_logits = []
        reg_offsets = []
        for f in fused:
            cl, ro = self.head(f)
            cls_logits.append(cl)
            reg_offsets.append(ro)

        return ModelOutput(
            cls_logits=cls_logits,
            reg_offsets=reg_offsets,
            video_feats=video_feats,
            fused_feats=fused,
            masks=video_masks,
        )

    def get_pyramid_lengths(self, T: int) -> list[int]:
        """Compute sequence lengths at each pyramid level given input length T."""
        import math
        lengths = [T]
        cur = T
        for _ in range(self.num_layers):
            cur = math.ceil(cur / self.downsample_rate)
            lengths.append(cur)
        return lengths
