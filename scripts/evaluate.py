#!/usr/bin/env python3
"""Evaluate MSCL-VTG.

Usage:
    uv run python scripts/evaluate.py --config configs/tacos_c3d_glove.yaml --checkpoint checkpoints/best.ckpt
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from mscl_vtg.config import Config
from mscl_vtg.data.collate import collate_fn
from mscl_vtg.engine.evaluator import Evaluator
from mscl_vtg.models.model import MSCLModel
from mscl_vtg.utils.checkpoint import load_checkpoint
from mscl_vtg.utils.seed import set_seed
from mscl_vtg.utils.logging import console


def _build_dataset(cfg, split):
    """Same factory as train.py."""
    name = cfg.data.dataset.lower()
    if name == "ego4d":
        from mscl_vtg.data.ego4d import Ego4DDataset
        return Ego4DDataset(cfg.data, split)
    elif name == "mad":
        from mscl_vtg.data.mad import MADDataset
        return MADDataset(cfg.data, split)
    elif name == "tacos":
        from mscl_vtg.data.tacos import TACoSDataset
        return TACoSDataset(cfg.data, split)
    elif name in ("activitynet", "anet"):
        from mscl_vtg.data.activitynet import ActivityNetDataset
        return ActivityNetDataset(cfg.data, split)
    elif name in ("charades", "charades_sta"):
        from mscl_vtg.data.charades import CharadesDataset
        return CharadesDataset(cfg.data, split)
    elif name == "dummy":
        from mscl_vtg.data.dummy import DummyDataset
        return DummyDataset(cfg.data, split)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MSCL-VTG")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSCLModel(
        video_feat_dim=cfg.data.video_feat_dim,
        text_feat_dim=cfg.data.text_feat_dim,
        cfg=cfg.model,
    )
    ckpt = load_checkpoint(args.checkpoint, model, map_location=str(device))
    console.print(f"[bold]Loaded checkpoint[/bold] epoch={ckpt.get('epoch', '?')}")

    test_ds = _build_dataset(cfg, args.split)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.eval.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
    )

    evaluator = Evaluator(cfg, model, test_loader, device)
    metrics = evaluator.evaluate()
    for k, v in sorted(metrics.items()):
        console.print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
