"""Base dataset for temporal grounding with pre-extracted features.

Each dataset adapter subclasses `TemporalGroundingDataset` and implements
`_load_annotations()` which returns a list of `VideoItem`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from mscl_vtg.config import DataConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class QueryAnnotation:
    query_id: str
    query_text: str
    start: float          # ground-truth start (seconds)
    end: float            # ground-truth end (seconds)


@dataclass
class VideoItem:
    """One video with all its associated queries."""
    video_id: str
    duration: float                         # video duration in seconds
    feature_stride: float                   # seconds per feature frame
    queries: list[QueryAnnotation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TemporalGroundingDataset(Dataset):
    """Video-centric dataset: each __getitem__ returns one video with a sampled query.

    Expected feature layout:
        video: <video_feat_dir>/<video_id>.npy   shape (T, D_v)
        text:  <text_feat_dir>/<video_id>_<query_id>.npy  shape (K, D_t)
              OR <text_feat_dir>/<query_id>.npy

    Subclasses must implement `_load_annotations`.
    """

    def __init__(self, cfg: DataConfig, split: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.video_feat_dir = Path(cfg.video_feat_dir)
        self.text_feat_dir = Path(cfg.text_feat_dir)

        # Build flat list: one entry per (video, query) pair
        self.videos: list[VideoItem] = self._load_annotations(split)
        self.samples: list[tuple[int, int]] = []  # (video_idx, query_idx)
        for vi, v in enumerate(self.videos):
            for qi in range(len(v.queries)):
                self.samples.append((vi, qi))

    # ----- subclass hook -----
    def _load_annotations(self, split: str) -> list[VideoItem]:
        raise NotImplementedError

    # ----- feature loading helpers -----
    def _load_video_feat(self, video_id: str) -> np.ndarray:
        p = self.video_feat_dir / f"{video_id}.npy"
        if p.exists():
            return np.load(str(p)).astype(np.float32)
        # Fallback: try .npz
        pz = self.video_feat_dir / f"{video_id}.npz"
        if pz.exists():
            d = np.load(str(pz))
            return d[list(d.keys())[0]].astype(np.float32)
        raise FileNotFoundError(f"Video features not found: {p}")

    def _load_text_feat(self, video_id: str, query_id: str) -> np.ndarray:
        # Try video_id + query_id first, then query_id only
        for name in [f"{video_id}_{query_id}", query_id]:
            p = self.text_feat_dir / f"{name}.npy"
            if p.exists():
                return np.load(str(p)).astype(np.float32)
        raise FileNotFoundError(
            f"Text features not found for video={video_id}, query={query_id}"
        )

    # ----- main -----
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        vi, qi = self.samples[idx]
        video = self.videos[vi]
        query = video.queries[qi]

        video_feat = self._load_video_feat(video.video_id)  # (T, D_v)
        text_feat = self._load_text_feat(video.video_id, query.query_id)  # (K, D_t)

        T = video_feat.shape[0]
        K = text_feat.shape[0]

        # Truncate / pad video
        max_v = self.cfg.max_video_len
        if T > max_v:
            video_feat = video_feat[:max_v]
            T = max_v
        video_mask = np.ones(T, dtype=np.float32)

        # Truncate / pad text
        max_q = self.cfg.max_query_len
        if K > max_q:
            text_feat = text_feat[:max_q]
            K = max_q
        text_mask = np.ones(K, dtype=np.float32)

        # Collect all query annotations for this video (for contrastive sampling)
        all_gt: list[dict] = []
        for q in video.queries:
            all_gt.append({"query_id": q.query_id, "start": q.start, "end": q.end})

        return {
            "video_id": video.video_id,
            "query_id": query.query_id,
            "video_feat": torch.from_numpy(video_feat),    # (T, D_v)
            "video_mask": torch.from_numpy(video_mask),     # (T,)
            "text_feat": torch.from_numpy(text_feat),       # (K, D_t)
            "text_mask": torch.from_numpy(text_mask),        # (K,)
            "gt_start": query.start,
            "gt_end": query.end,
            "duration": video.duration,
            "feature_stride": video.feature_stride,
            "all_gt": all_gt,
        }


# ---------------------------------------------------------------------------
# JSON-based generic loader (works for TACoS, ActivityNet, Charades)
# ---------------------------------------------------------------------------
class JsonAnnotationDataset(TemporalGroundingDataset):
    """Generic dataset that reads annotations from a single JSON file.

    Expected JSON format (one of):
      { "video_id": { "duration": ..., "timestamps": [[s,e],...], "sentences": [...] } }
    OR list of dicts with video_id, query, start, end fields.
    """

    def _load_annotations(self, split: str) -> list[VideoItem]:
        anno_path = Path(self.cfg.anno_dir) / f"{split}.json"
        if not anno_path.exists():
            # Return empty – allows dummy runs
            return []
        with open(anno_path) as f:
            raw = json.load(f)

        videos: dict[str, VideoItem] = {}
        if isinstance(raw, dict):
            for vid, info in raw.items():
                dur = float(info.get("duration", 0))
                fs = float(info.get("feature_stride", self.cfg.feature_stride))
                vi = VideoItem(video_id=vid, duration=dur, feature_stride=fs)
                for i, (ts, sent) in enumerate(
                    zip(info.get("timestamps", []), info.get("sentences", []))
                ):
                    vi.queries.append(
                        QueryAnnotation(
                            query_id=f"{vid}_{i}",
                            query_text=sent,
                            start=float(ts[0]),
                            end=float(ts[1]),
                        )
                    )
                if vi.queries:
                    videos[vid] = vi
        elif isinstance(raw, list):
            for item in raw:
                vid = item["video_id"]
                if vid not in videos:
                    videos[vid] = VideoItem(
                        video_id=vid,
                        duration=float(item.get("duration", 0)),
                        feature_stride=float(item.get("feature_stride", self.cfg.feature_stride)),
                    )
                qid = item.get("query_id", f"{vid}_{len(videos[vid].queries)}")
                videos[vid].queries.append(
                    QueryAnnotation(
                        query_id=str(qid),
                        query_text=item.get("query", ""),
                        start=float(item["start"]),
                        end=float(item["end"]),
                    )
                )
        return list(videos.values())
