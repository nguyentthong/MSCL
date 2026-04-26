"""Temporal segment operations: IoU, moment decoding."""
from __future__ import annotations

import torch


def temporal_iou(
    pred_s: torch.Tensor, pred_e: torch.Tensor,
    gt_s: torch.Tensor, gt_e: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise temporal IoU.

    All inputs can be scalars or 1-D tensors of the same length.
    """
    inter_s = torch.max(pred_s, gt_s)
    inter_e = torch.min(pred_e, gt_e)
    inter = (inter_e - inter_s).clamp(min=0)
    union = (pred_e - pred_s) + (gt_e - gt_s) - inter
    return inter / union.clamp(min=1e-8)


def decode_moments(
    cls_logits: list[torch.Tensor],     # L tensors of (B, T_l)
    reg_offsets: list[torch.Tensor],    # L tensors of (B, T_l, 2)
    feature_stride: torch.Tensor,       # (B,) seconds per clip at level 0
    downsample_rate: int = 2,
) -> list[dict[str, torch.Tensor]]:
    """Decode per-sample predictions across all pyramid levels.

    Returns list (len B) of dicts with keys:
        scores: (N,)  sigmoid scores
        starts: (N,)  predicted start in seconds
        ends:   (N,)  predicted end in seconds
    """
    B = cls_logits[0].shape[0]
    results = [{"scores": [], "starts": [], "ends": []} for _ in range(B)]

    for l, (logits, offsets) in enumerate(zip(cls_logits, reg_offsets)):
        # logits: (B, T_l), offsets: (B, T_l, 2)
        scores = torch.sigmoid(logits)  # (B, T_l)
        scale = downsample_rate ** (l + 1)  # +1 because level 0 is pre-transformer; head levels are 1..L

        # TODO: Paper uses 2^{l-1} in Eq. 9 — here l starts from 1 in the head,
        # so scale = 2^l for the l-th head output. Adjust if needed for exact matching.
        for b in range(B):
            T_l = scores.shape[1]
            t = torch.arange(T_l, device=scores.device).float()
            d_s = offsets[b, :, 0]  # (T_l,)
            d_e = offsets[b, :, 1]  # (T_l,)
            fs = feature_stride[b].item()

            # Eq. 9: s_hat = scale * (t - d_s), e_hat = scale * (t + d_e)
            # Convert to seconds
            starts = fs * scale * (t - d_s)
            ends = fs * scale * (t + d_e)

            results[b]["scores"].append(scores[b])
            results[b]["starts"].append(starts)
            results[b]["ends"].append(ends)

    # Concatenate across levels
    for b in range(B):
        results[b]["scores"] = torch.cat(results[b]["scores"])
        results[b]["starts"] = torch.cat(results[b]["starts"])
        results[b]["ends"] = torch.cat(results[b]["ends"])

    return results
