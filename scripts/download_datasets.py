#!/usr/bin/env python3
"""Download instructions and skeleton for obtaining datasets.

Most temporal grounding datasets require registration.  This script
prints clear instructions and verifies whether expected files exist.

Usage:
    uv run python scripts/download_datasets.py --dataset tacos --data_root data/
"""
from __future__ import annotations

import argparse
from pathlib import Path

DATASET_INFO = {
    "ego4d": {
        "name": "Ego4D-NLQ",
        "instructions": (
            "1. Register at https://ego4d-data.org/ and accept the licence.\n"
            "2. Use the Ego4D CLI to download NLQ annotations:\n"
            "      ego4d --output_directory data/ego4d --datasets nlq\n"
            "3. For SlowFast features: download from the Ego4D model zoo or extract\n"
            "   using the official SlowFast repo.\n"
            "4. For EgoVLP features: see https://github.com/showlab/EgoVLP\n"
            "5. For BERT text features: extract using HuggingFace transformers.\n"
        ),
        "expected_files": [
            "annotations/ego4d_nlq_train.json",
            "annotations/ego4d_nlq_val.json",
            "features/video/",
            "features/text/",
        ],
    },
    "mad": {
        "name": "MAD",
        "instructions": (
            "1. Download MAD annotations from https://github.com/Soldelli/MAD\n"
            "2. CLIP features are provided by the MAD authors.\n"
            "3. Place annotations as mad_train.json, mad_val.json, mad_test.json\n"
        ),
        "expected_files": [
            "annotations/mad_train.json",
            "annotations/mad_val.json",
            "features/video/",
            "features/text/",
        ],
    },
    "tacos": {
        "name": "TACoS",
        "instructions": (
            "1. Download TACoS annotations from https://www.mpi-inf.mpg.de/tacos\n"
            "   or from common temporal grounding repos (e.g. 2D-TAN).\n"
            "2. C3D features: download from https://github.com/jiyanggao/TALL\n"
            "3. GloVe embeddings: download glove.840B.300d from\n"
            "   https://nlp.stanford.edu/projects/glove/\n"
            "4. Pre-extract query features using GloVe word embeddings.\n"
        ),
        "expected_files": [
            "annotations/tacos_train.json",
            "annotations/tacos_val.json",
            "annotations/tacos_test.json",
            "features/video/",
            "features/text/",
        ],
    },
    "activitynet": {
        "name": "ActivityNet-Captions",
        "instructions": (
            "1. Download annotations from http://activity-net.org/download.html\n"
            "2. C3D features: download from the ActivityNet challenge page.\n"
            "3. GloVe text features: same as TACoS.\n"
        ),
        "expected_files": [
            "annotations/train.json",
            "annotations/val.json",
            "features/video/",
            "features/text/",
        ],
    },
    "charades": {
        "name": "Charades-STA",
        "instructions": (
            "1. Download Charades-STA annotations from\n"
            "   https://github.com/jiyanggao/TALL\n"
            "2. I3D features pre-trained on Kinetics: extract or download from\n"
            "   common temporal grounding repos.\n"
            "3. GloVe text features: same as TACoS.\n"
        ),
        "expected_files": [
            "annotations/train.json",
            "annotations/val.json",
            "features/video/",
            "features/text/",
        ],
    },
}


def check_files(data_root: Path, expected: list[str]) -> None:
    for f in expected:
        p = data_root / f
        status = "✓" if p.exists() else "✗ MISSING"
        print(f"  [{status}] {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_INFO.keys()))
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()

    info = DATASET_INFO[args.dataset]
    print(f"\n{'='*60}")
    print(f"  Dataset: {info['name']}")
    print(f"{'='*60}\n")
    print(info["instructions"])
    print("\nExpected file layout:")
    check_files(Path(args.data_root), info["expected_files"])
    print()


if __name__ == "__main__":
    main()
