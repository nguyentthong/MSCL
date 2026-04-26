"""Training engine for MSCL-VTG.

Handles:
- Combined loss computation (focal + DIoU + within-scale + cross-scale)
- AMP mixed-precision
- Gradient clipping
- Warmup + cosine schedule
- Periodic validation and best-checkpoint saving
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from mscl_vtg.config import Config
from mscl_vtg.data.sampling import center_sample_targets, sample_negatives
from mscl_vtg.losses import (
    cross_scale_contrastive_loss,
    diou_loss,
    sigmoid_focal_loss,
    within_scale_contrastive_loss,
)
from mscl_vtg.models.model import MSCLModel, ModelOutput
from mscl_vtg.utils.checkpoint import save_checkpoint, load_checkpoint
from mscl_vtg.utils.logging import log_losses, console


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: MSCLModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        total_steps = cfg.train.epochs * len(train_loader)
        warmup_steps = cfg.train.warmup_epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine(step, warmup_steps, total_steps),
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = GradScaler("cpu" if not torch.cuda.is_available() else "cuda", enabled=cfg.train.amp)
        self.best_metric = 0.0
        self.start_epoch = 0

        # Resume
        if cfg.train.resume:
            console.print(f"[yellow]Resuming from {cfg.train.resume}[/yellow]")
            ckpt = load_checkpoint(cfg.train.resume, model, self.optimizer, self.scheduler)
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_metric = ckpt.get("best_metric", 0.0)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_loss(self, batch: dict[str, Any], output: ModelOutput) -> dict[str, torch.Tensor]:
        lcfg = self.cfg.loss
        device = self.device

        gt_start = batch["gt_start"].to(device)
        gt_end = batch["gt_end"].to(device)
        feat_stride = batch["feature_stride"].to(device)
        B = gt_start.shape[0]

        # Convert GT seconds → clip indices at level 0
        gt_s_idx = gt_start / feat_stride  # (B,)
        gt_e_idx = gt_end / feat_stride    # (B,)

        # Pyramid level lengths (from actual output shapes)
        seq_lens = [f.shape[1] for f in output.video_feats]  # L+1 levels

        # Center-sample targets for head levels (1..L)
        # We need targets for levels 1..L (the head levels)
        head_seq_lens = seq_lens[1:]  # skip level 0
        targets_head = center_sample_targets(
            gt_s_idx, gt_e_idx, head_seq_lens,
            downsample_rate=self.model.downsample_rate,
            alpha=lcfg.center_sampling_alpha,
        )

        # Also generate targets for level 0 (needed for cross-scale)
        targets_all = center_sample_targets(
            gt_s_idx, gt_e_idx, seq_lens,
            downsample_rate=self.model.downsample_rate,
            alpha=lcfg.center_sampling_alpha,
        )

        # ---- Classification focal loss ----
        cls_loss = torch.tensor(0.0, device=device)
        for l, (logits, tgt) in enumerate(zip(output.cls_logits, targets_head)):
            mask_l = output.masks[l + 1]
            if mask_l is not None:
                valid = mask_l.bool()
                cls_loss = cls_loss + sigmoid_focal_loss(
                    logits[valid], tgt[valid].float(),
                    alpha=lcfg.focal_alpha, gamma=lcfg.focal_gamma,
                )
            else:
                cls_loss = cls_loss + sigmoid_focal_loss(
                    logits.reshape(-1), tgt.reshape(-1).float(),
                    alpha=lcfg.focal_alpha, gamma=lcfg.focal_gamma,
                )
        cls_loss = cls_loss / max(len(output.cls_logits), 1)

        # ---- Regression DIoU loss (only on positive positions) ----
        reg_loss = torch.tensor(0.0, device=device)
        reg_count = 0
        for l, (offsets, tgt) in enumerate(zip(output.reg_offsets, targets_head)):
            scale = self.model.downsample_rate ** (l + 1)
            for b in range(B):
                pos = tgt[b].nonzero(as_tuple=False).squeeze(-1)
                if len(pos) == 0:
                    continue
                t = pos.float()
                d_s = offsets[b, pos, 0]
                d_e = offsets[b, pos, 1]
                fs = feat_stride[b].item()
                pred_s = fs * scale * (t - d_s)
                pred_e = fs * scale * (t + d_e)
                reg_loss = reg_loss + diou_loss(
                    pred_s, pred_e,
                    gt_start[b].expand_as(pred_s),
                    gt_end[b].expand_as(pred_e),
                )
                reg_count += 1
        reg_loss = reg_loss / max(reg_count, 1)

        # ---- Within-scale contrastive loss (Eq. 11) ----
        # Uses video_feats at levels 1..L (same as head levels)
        within_feats = output.video_feats[1:]           # [Z^1, ..., Z^L]
        within_masks_valid = [output.masks[l + 1] for l in range(len(within_feats))]
        within_loss = within_scale_contrastive_loss(
            within_feats, targets_head, within_masks_valid, lcfg.temperature,
        )

        # ---- Cross-scale contrastive loss (Eq. 12) ----
        cross_loss = cross_scale_contrastive_loss(
            output.video_feats, targets_all, output.masks, lcfg.temperature,
        )

        # ---- Total loss (Eq. 13) ----
        total = cls_loss + lcfg.rho_reg * reg_loss + lcfg.rho_within * within_loss + lcfg.rho_cross * cross_loss

        return {
            "total": total,
            "cls": cls_loss,
            "reg": reg_loss,
            "within": within_loss,
            "cross": cross_loss,
        }

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}", leave=False)
        for step, batch in enumerate(pbar):
            v = batch["video_feat"].to(self.device)
            vm = batch["video_mask"].to(self.device)
            t = batch["text_feat"].to(self.device)
            tm = batch["text_mask"].to(self.device)

            with autocast(self.device.type, enabled=self.cfg.train.amp):
                output = self.model(v, vm, t, tm)
                losses = self._compute_loss(batch, output)

            self.optimizer.zero_grad()
            self.scaler.scale(losses["total"]).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += losses["total"].item()
            if (step + 1) % self.cfg.train.log_interval == 0:
                log_losses({k: v.item() for k, v in losses.items()}, epoch, step + 1)
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        return total_loss / max(len(self.train_loader), 1)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def fit(self) -> None:
        save_dir = Path(self.cfg.train.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            avg_loss = self.train_one_epoch(epoch)
            console.print(f"[bold green]Epoch {epoch}[/bold green] avg_loss={avg_loss:.4f}")

            if self.val_loader is not None and (epoch + 1) % self.cfg.train.val_interval == 0:
                from .evaluator import Evaluator
                evaluator = Evaluator(self.cfg, self.model, self.val_loader, self.device)
                metrics = evaluator.evaluate()
                # Pick primary metric: first Recall entry
                primary = list(metrics.values())[0] if metrics else 0.0
                if primary > self.best_metric:
                    self.best_metric = primary
                    save_checkpoint(
                        save_dir / "best.ckpt", self.model, self.optimizer,
                        self.scheduler, epoch, self.best_metric,
                    )
                    console.print(f"[bold cyan]New best: {primary:.2f}[/bold cyan]")

            # Always save latest
            save_checkpoint(
                save_dir / "latest.ckpt", self.model, self.optimizer,
                self.scheduler, epoch, self.best_metric,
            )
        console.print("[bold green]Training complete![/bold green]")


def _warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))
