"""MAD (Movie Audio Descriptions) dataset adapter.

Uses CLIP features for both video and text.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import QueryAnnotation, TemporalGroundingDataset, VideoItem


class MADDataset(TemporalGroundingDataset):
    def _load_annotations(self, split: str) -> list[VideoItem]:
        anno_path = Path(self.cfg.anno_dir) / f"mad_{split}.json"
        if not anno_path.exists():
            return []
        with open(anno_path) as f:
            raw = json.load(f)

        videos: dict[str, VideoItem] = {}
        entries = raw if isinstance(raw, list) else list(raw.values())
        for item in entries:
            vid = str(item.get("movie", item.get("video_id", "")))
            if vid not in videos:
                videos[vid] = VideoItem(
                    video_id=vid,
                    duration=float(item.get("duration", item.get("movie_duration", 0))),
                    feature_stride=self.cfg.feature_stride,
                )
            qid = item.get("query_id", f"{vid}_{len(videos[vid].queries)}")
            videos[vid].queries.append(
                QueryAnnotation(
                    query_id=str(qid),
                    query_text=item.get("sentence", item.get("query", "")),
                    start=float(item.get("start", item.get("timestamp", [0, 0])[0])),
                    end=float(item.get("end", item.get("timestamp", [0, 0])[1])),
                )
            )
        return list(videos.values())
