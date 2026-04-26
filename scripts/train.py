#!/usr/bin/env python3
"""Train MSCL-VTG.

Usage:
    uv run python scripts/train.py --config configs/tacos_c3d_glove.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mscl_vtg.config import Config
from mscl_vtg.data.collate import collate_fn
from mscl_vtg.engine.trainer import Trainer
from mscl_vtg.models.model import MSCLModel
from mscl_vtg.utils.seed import set_seed
from mscl_vtg.utils.logging import console


def _build_dataset(cfg: Config, split: str):
    """Instantiate the correct dataset class based on cfg.data.dataset."""
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
    parser = argparse.ArgumentParser(description="Train MSCL-VTG")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.seed is not None:
        cfg.train.seed = args.seed
    set_seed(cfg.train.seed)

    console.print(f"[bold]Config:[/bold] {args.config}")
    console.print(f"[bold]Dataset:[/bold] {cfg.data.dataset}")
    console.print(f"[bold]Device:[/bold] {'cuda' if torch.cuda.is_available() else 'cpu'}")

    train_ds = _build_dataset(cfg, "train")
    val_ds = _build_dataset(cfg, "val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn, pin_memory=True,
    ) if len(val_ds) > 0 else None

    model = MSCLModel(
        video_feat_dim=cfg.data.video_feat_dim,
        text_feat_dim=cfg.data.text_feat_dim,
        cfg=cfg.model,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold]Model params:[/bold] {n_params:,}")

    trainer = Trainer(cfg, model, train_loader, val_loader)
    trainer.fit()


if __name__ == "__main__":
    main()
