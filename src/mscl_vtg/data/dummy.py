"""Dummy dataset that generates random features — no files needed.

Used for smoke-testing the full pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from mscl_vtg.config import DataConfig


class DummyDataset(Dataset):
    """Generates random video/text features."""

    def __init__(self, cfg: DataConfig, split: str = "train", num_samples: int = 8):
        self.cfg = cfg
        self.split = split
        self.num_samples = num_samples
        self.rng = np.random.RandomState(42 if split == "train" else 123)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.rng.randint(16, self.cfg.max_video_len + 1)
        K = self.rng.randint(3, self.cfg.max_query_len + 1)
        duration = T * self.cfg.feature_stride
        gt_start = self.rng.uniform(0, duration * 0.5)
        gt_end = self.rng.uniform(gt_start + 1.0, min(gt_start + duration * 0.5, duration))

        return {
            "video_id": f"dummy_{idx}",
            "query_id": f"dummy_{idx}_q0",
            "video_feat": torch.randn(T, self.cfg.video_feat_dim),
            "video_mask": torch.ones(T),
            "text_feat": torch.randn(K, self.cfg.text_feat_dim),
            "text_mask": torch.ones(K),
            "gt_start": gt_start,
            "gt_end": gt_end,
            "duration": duration,
            "feature_stride": self.cfg.feature_stride,
            "all_gt": [{"query_id": f"dummy_{idx}_q0", "start": gt_start, "end": gt_end}],
        }
