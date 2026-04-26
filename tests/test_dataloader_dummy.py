"""Dummy dataset that generates random features for testing.

Also used by train.py when dataset='dummy'.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from mscl_vtg.config import DataConfig


class DummyDataset(Dataset):
    """Generates random video/text features — no files needed."""

    def __init__(self, cfg: DataConfig, split: str = "train", num_samples: int = 8):
        self.cfg = cfg
        self.split = split
        self.num_samples = num_samples
        self.rng = np.random.RandomState(42)

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


def test_dummy_dataloader():
    from torch.utils.data import DataLoader
    from mscl_vtg.data.collate import collate_fn

    cfg = DataConfig(
        video_feat_dim=256, text_feat_dim=128,
        max_video_len=64, max_query_len=8,
    )
    ds = DummyDataset(cfg, num_samples=6)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    batch = next(iter(loader))
    assert batch["video_feat"].shape[0] == 2
    assert batch["video_feat"].shape[2] == 256
    assert batch["text_feat"].shape[2] == 128
    assert batch["video_mask"].shape == batch["video_feat"].shape[:2]
    assert len(batch["video_ids"]) == 2
    print("test_dummy_dataloader PASSED")


if __name__ == "__main__":
    test_dummy_dataloader()
