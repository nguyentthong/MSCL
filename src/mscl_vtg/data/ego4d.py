"""Ego4D-NLQ dataset adapter.

Annotation format: JSON list from the official Ego4D-NLQ benchmark.
Each entry: {clip_uid, video_uid, query, exact_s, exact_e, ...}
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import QueryAnnotation, TemporalGroundingDataset, VideoItem


class Ego4DDataset(TemporalGroundingDataset):
    """Ego4D-NLQ with SlowFast+BERT or EgoVLP features."""

    def _load_annotations(self, split: str) -> list[VideoItem]:
        anno_path = Path(self.cfg.anno_dir) / f"ego4d_nlq_{split}.json"
        if not anno_path.exists():
            return []
        with open(anno_path) as f:
            raw = json.load(f)

        videos: dict[str, VideoItem] = {}

        # Handle both flat-list and nested Ego4D formats
        entries = raw if isinstance(raw, list) else raw.get("annotations", raw.get("data", []))
        for item in entries:
            vid = item.get("clip_uid", item.get("video_id", ""))
            if vid not in videos:
                videos[vid] = VideoItem(
                    video_id=vid,
                    duration=float(item.get("duration", item.get("clip_duration", 0))),
                    feature_stride=self.cfg.feature_stride,
                )
            # Ego4D may nest queries under "language_queries"
            queries = item.get("language_queries", [item])
            for qi, q in enumerate(queries):
                qid = q.get("query_id", f"{vid}_{len(videos[vid].queries)}")
                start = float(q.get("exact_s", q.get("start", 0)))
                end = float(q.get("exact_e", q.get("end", 0)))
                text = q.get("query", q.get("template", ""))
                videos[vid].queries.append(
                    QueryAnnotation(query_id=str(qid), query_text=text, start=start, end=end)
                )
        return list(videos.values())
