"""TACoS dataset adapter.  Uses C3D video + GloVe text features.

Expected annotation: data/annotations/tacos_{split}.json
Format: { video_id: { duration, timestamps: [[s,e],...], sentences: [...] } }
"""
from .base import JsonAnnotationDataset


class TACoSDataset(JsonAnnotationDataset):
    """TACoS inherits generic JSON loader with no customization needed."""

    def _load_annotations(self, split: str):
        # Override anno filename pattern
        import json
        from pathlib import Path
        from .base import VideoItem, QueryAnnotation

        anno_path = Path(self.cfg.anno_dir) / f"tacos_{split}.json"
        if not anno_path.exists():
            # fallback to generic name
            return super()._load_annotations(split)
        with open(anno_path) as f:
            raw = json.load(f)

        videos: dict[str, VideoItem] = {}
        if isinstance(raw, dict):
            for vid, info in raw.items():
                dur = float(info.get("duration", 0))
                vi = VideoItem(video_id=vid, duration=dur, feature_stride=self.cfg.feature_stride)
                for i, (ts, sent) in enumerate(
                    zip(info.get("timestamps", []), info.get("sentences", []))
                ):
                    vi.queries.append(
                        QueryAnnotation(
                            query_id=f"{vid}_{i}", query_text=sent,
                            start=float(ts[0]), end=float(ts[1]),
                        )
                    )
                if vi.queries:
                    videos[vid] = vi
        return list(videos.values())
