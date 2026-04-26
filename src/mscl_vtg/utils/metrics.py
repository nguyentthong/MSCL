"""Evaluation metrics: Recall@K at tIoU thresholds."""
from __future__ import annotations

from collections import defaultdict

import torch


def compute_recall(
    predictions: list[dict],   # list of {scores, starts, ends} per query
    gt_starts: list[float],
    gt_ends: list[float],
    recall_k: list[int] = (1, 5),
    tiou_thresholds: list[float] = (0.5, 0.7),
) -> dict[str, float]:
    """Compute Recall@K at multiple tIoU thresholds.

    Returns dict like {"R@1_tIoU=0.5": 0.42, "R@5_tIoU=0.7": 0.61, ...}
    """
    results: dict[str, float] = {}
    N = len(predictions)

    for theta in tiou_thresholds:
        for k in recall_k:
            hits = 0
            for i in range(N):
                pred = predictions[i]
                if pred["scores"].numel() == 0:
                    continue
                # top-k by score
                topk = min(k, pred["scores"].numel())
                _, idx = pred["scores"].topk(topk)
                p_s = pred["starts"][idx]
                p_e = pred["ends"][idx]
                g_s = gt_starts[i]
                g_e = gt_ends[i]

                # Compute IoU for each of top-k
                inter_s = torch.max(p_s, torch.tensor(g_s))
                inter_e = torch.min(p_e, torch.tensor(g_e))
                inter = (inter_e - inter_s).clamp(min=0)
                union = (p_e - p_s) + (g_e - g_s) - inter
                iou = inter / union.clamp(min=1e-8)

                if (iou >= theta).any():
                    hits += 1

            results[f"R@{k}_tIoU={theta}"] = hits / max(N, 1) * 100.0
    return results
