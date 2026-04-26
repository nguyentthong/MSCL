"""Sampling utilities for center-based target assignment and contrastive learning.

Paper reference (Sec. Methodology):
  Center sampling: given a moment centered at t, any time step
  c ∈ [t - α·T/T_l, t + α·T/T_l] in layer l is a target (positive).

  Negative sampling: draw |N(l)| = |P(l)| random non-target moments per layer.
"""
from __future__ import annotations

import torch


def center_sample_targets(
    gt_start_idx: torch.Tensor,     # (B,) ground-truth start in clip indices
    gt_end_idx: torch.Tensor,       # (B,) ground-truth end in clip indices
    seq_lens: list[int],            # length T_l for each pyramid level
    downsample_rate: int = 2,
    alpha: float = 1.5,
) -> list[torch.Tensor]:
    """Assign positive labels per pyramid level via center sampling.

    Returns:
        targets: list of (B, T_l) boolean tensors, True = positive.
    """
    B = gt_start_idx.shape[0]
    device = gt_start_idx.device
    targets = []
    T0 = seq_lens[0]

    for l, T_l in enumerate(seq_lens):
        scale = downsample_rate ** l
        # Map gt to this level's time axis
        center = ((gt_start_idx + gt_end_idx) / 2.0) / scale  # (B,)
        radius = alpha * T0 / T_l  # scalar
        t_grid = torch.arange(T_l, device=device).unsqueeze(0).expand(B, -1).float()  # (B, T_l)
        pos = (t_grid >= (center.unsqueeze(1) - radius)) & (t_grid <= (center.unsqueeze(1) + radius))

        # Also ensure the point falls within the gt span at this scale
        gt_s_l = gt_start_idx.unsqueeze(1) / scale
        gt_e_l = gt_end_idx.unsqueeze(1) / scale
        in_span = (t_grid >= gt_s_l) & (t_grid <= gt_e_l)
        targets.append(pos & in_span)

    return targets


def sample_negatives(
    pos_mask: torch.Tensor,         # (T_l,) boolean for one sample
    valid_mask: torch.Tensor,       # (T_l,) boolean for valid (non-padded) positions
    num_neg: int | None = None,     # if None, use |P(l)|
) -> torch.Tensor:
    """Sample negative indices for one sample at one level.

    Returns tensor of indices into the T_l dimension.
    """
    neg_candidates = valid_mask & (~pos_mask)
    neg_indices = neg_candidates.nonzero(as_tuple=False).squeeze(-1)
    if num_neg is None:
        num_neg = pos_mask.sum().item()
    num_neg = int(min(num_neg, len(neg_indices)))
    if num_neg == 0:
        return torch.empty(0, dtype=torch.long, device=pos_mask.device)
    perm = torch.randperm(len(neg_indices), device=pos_mask.device)[:num_neg]
    return neg_indices[perm]
