"""Distance-IoU (DIoU) loss for 1D temporal boundary regression.

Adapted from the 2D box DIoU loss to 1D segments.
"""
from __future__ import annotations

import torch


def temporal_iou(
    pred_s: torch.Tensor, pred_e: torch.Tensor,
    gt_s: torch.Tensor, gt_e: torch.Tensor,
) -> torch.Tensor:
    """Compute IoU between predicted and ground-truth 1D segments."""
    inter_s = torch.max(pred_s, gt_s)
    inter_e = torch.min(pred_e, gt_e)
    inter = (inter_e - inter_s).clamp(min=0)
    union = (pred_e - pred_s) + (gt_e - gt_s) - inter
    return inter / union.clamp(min=1e-8)


def diou_loss(
    pred_s: torch.Tensor,   # predicted start
    pred_e: torch.Tensor,   # predicted end
    gt_s: torch.Tensor,     # ground-truth start
    gt_e: torch.Tensor,     # ground-truth end
    reduction: str = "mean",
) -> torch.Tensor:
    """Distance-IoU loss for 1D temporal segments.

    DIoU = IoU - (center_distance^2) / (enclosing_length^2)
    Loss = 1 - DIoU
    """
    iou = temporal_iou(pred_s, pred_e, gt_s, gt_e)

    # Center distance
    pred_center = (pred_s + pred_e) / 2
    gt_center = (gt_s + gt_e) / 2
    center_dist_sq = (pred_center - gt_center) ** 2

    # Enclosing segment
    enclose_s = torch.min(pred_s, gt_s)
    enclose_e = torch.max(pred_e, gt_e)
    enclose_len_sq = (enclose_e - enclose_s) ** 2

    diou = iou - center_dist_sq / enclose_len_sq.clamp(min=1e-8)
    loss = 1.0 - diou

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
