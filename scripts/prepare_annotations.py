#!/usr/bin/env python3
"""Convert raw dataset annotations to the unified JSON format expected by dataloaders.

Usage:
    uv run python scripts/prepare_annotations.py --dataset tacos --raw_dir data/raw/ --out_dir data/annotations/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_tacos(raw_dir: Path, out_dir: Path):
    """Convert TACoS annotations to {video_id: {duration, timestamps, sentences}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        src = raw_dir / f"{split}.json"
        if not src.exists():
            print(f"  [SKIP] {src} not found")
            continue
        with open(src) as f:
            raw = json.load(f)
        # Already in expected format for many repos — just copy
        with open(out_dir / f"tacos_{split}.json", "w") as f:
            json.dump(raw, f, indent=2)
        print(f"  Wrote {out_dir / f'tacos_{split}.json'}")


def prepare_activitynet(raw_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [("train.json", "train.json"), ("val_1.json", "val.json")]:
        src = raw_dir / src_name
        if not src.exists():
            print(f"  [SKIP] {src} not found")
            continue
        with open(src) as f:
            raw = json.load(f)
        with open(out_dir / dst_name, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"  Wrote {out_dir / dst_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tacos", "activitynet", "charades", "ego4d", "mad"])
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.dataset == "tacos":
        prepare_tacos(raw_dir, out_dir)
    elif args.dataset == "activitynet":
        prepare_activitynet(raw_dir, out_dir)
    else:
        print(f"Preparation for {args.dataset}: copy your annotations to {out_dir}/")
        print("See scripts/download_datasets.py for expected format.")


if __name__ == "__main__":
    main()
