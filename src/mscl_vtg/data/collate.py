"""Collate function that pads variable-length videos and queries."""
from __future__ import annotations

from typing import Any

import torch


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad video/text features to the max length within the batch."""
    B = len(batch)

    # --- determine max lengths in this batch ---
    max_vlen = max(b["video_feat"].shape[0] for b in batch)
    max_qlen = max(b["text_feat"].shape[0] for b in batch)
    D_v = batch[0]["video_feat"].shape[1]
    D_t = batch[0]["text_feat"].shape[1]

    video_feats = torch.zeros(B, max_vlen, D_v)
    video_masks = torch.zeros(B, max_vlen)
    text_feats = torch.zeros(B, max_qlen, D_t)
    text_masks = torch.zeros(B, max_qlen)

    gt_starts = []
    gt_ends = []
    durations = []
    feature_strides = []
    video_ids = []
    query_ids = []
    all_gts = []

    for i, b in enumerate(batch):
        vl = b["video_feat"].shape[0]
        ql = b["text_feat"].shape[0]
        video_feats[i, :vl] = b["video_feat"]
        video_masks[i, :vl] = 1.0
        text_feats[i, :ql] = b["text_feat"]
        text_masks[i, :ql] = 1.0
        gt_starts.append(b["gt_start"])
        gt_ends.append(b["gt_end"])
        durations.append(b["duration"])
        feature_strides.append(b["feature_stride"])
        video_ids.append(b["video_id"])
        query_ids.append(b["query_id"])
        all_gts.append(b["all_gt"])

    return {
        "video_feat": video_feats,                          # (B, T, D_v)
        "video_mask": video_masks,                          # (B, T)
        "text_feat": text_feats,                            # (B, K, D_t)
        "text_mask": text_masks,                            # (B, K)
        "gt_start": torch.tensor(gt_starts, dtype=torch.float32),   # (B,)
        "gt_end": torch.tensor(gt_ends, dtype=torch.float32),       # (B,)
        "duration": torch.tensor(durations, dtype=torch.float32),
        "feature_stride": torch.tensor(feature_strides, dtype=torch.float32),
        "video_ids": video_ids,
        "query_ids": query_ids,
        "all_gt": all_gts,
    }
