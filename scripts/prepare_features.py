#!/usr/bin/env python3
"""Convert pre-extracted features to the .npy format expected by dataloaders.

Expected output format:
  Video: <out_dir>/video/<video_id>.npy  →  shape (T, D_v)
  Text:  <out_dir>/text/<query_id>.npy   →  shape (K, D_t)

This script provides conversion skeletons.  Adapt to your raw feature format.

Usage:
    uv run python scripts/prepare_features.py --dataset tacos \
        --raw_video_dir data/raw_features/c3d/ \
        --raw_text_dir data/raw_features/glove/ \
        --out_dir data/features/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def convert_npy_dir(src: Path, dst: Path):
    """Copy/convert .npy files from src to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.npy")):
        arr = np.load(str(f))
        np.save(str(dst / f.name), arr.astype(np.float32))
    # Also handle .npz
    for f in sorted(src.glob("*.npz")):
        d = np.load(str(f))
        key = list(d.keys())[0]
        out_name = f.stem + ".npy"
        np.save(str(dst / out_name), d[key].astype(np.float32))
    print(f"  Converted {src} → {dst}  ({len(list(dst.glob('*.npy')))} files)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw_video_dir", required=True)
    parser.add_argument("--raw_text_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    print(f"Preparing features for {args.dataset}")
    convert_npy_dir(Path(args.raw_video_dir), out / "video")
    convert_npy_dir(Path(args.raw_text_dir), out / "text")
    print("Done!")


if __name__ == "__main__":
    main()
