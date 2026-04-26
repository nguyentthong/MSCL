"""Soft-NMS for merging overlapping temporal moment predictions (Bodla et al., 2017)."""
from __future__ import annotations

import torch


def soft_nms(
    scores: torch.Tensor,   # (N,)
    starts: torch.Tensor,   # (N,)
    ends: torch.Tensor,     # (N,)
    sigma: float = 0.5,
    threshold: float = 0.001,
    max_predictions: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Soft-NMS and return filtered (scores, starts, ends).

    Uses Gaussian decay: score_i *= exp(-iou^2 / sigma).
    """
    N = scores.shape[0]
    if N == 0:
        return scores, starts, ends

    # Work on clones
    sc = scores.clone()
    st = starts.clone()
    en = ends.clone()

    keep_scores = []
    keep_starts = []
    keep_ends = []

    for _ in range(min(N, max_predictions)):
        max_idx = sc.argmax()
        if sc[max_idx] < threshold:
            break

        keep_scores.append(sc[max_idx].item())
        keep_starts.append(st[max_idx].item())
        keep_ends.append(en[max_idx].item())

        # Compute IoU with max
        inter_s = torch.max(st, st[max_idx])
        inter_e = torch.min(en, en[max_idx])
        inter = (inter_e - inter_s).clamp(min=0)
        union = (en - st) + (en[max_idx] - st[max_idx]) - inter
        iou = inter / union.clamp(min=1e-8)

        # Gaussian decay
        decay = torch.exp(-(iou ** 2) / sigma)
        sc = sc * decay
        sc[max_idx] = 0  # already selected

    device = scores.device
    if not keep_scores:
        return (
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            torch.empty(0, device=device),
        )
    return (
        torch.tensor(keep_scores, device=device),
        torch.tensor(keep_starts, device=device),
        torch.tensor(keep_ends, device=device),
    )
