"""Evaluation engine for MSCL-VTG."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mscl_vtg.config import Config
from mscl_vtg.models.model import MSCLModel
from mscl_vtg.utils.box_ops import decode_moments
from mscl_vtg.utils.soft_nms import soft_nms
from mscl_vtg.utils.metrics import compute_recall
from mscl_vtg.utils.logging import log_metrics


class Evaluator:
    def __init__(
        self,
        cfg: Config,
        model: MSCLModel,
        dataloader: DataLoader,
        device: torch.device | None = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        ecfg = self.cfg.eval

        all_preds: list[dict[str, torch.Tensor]] = []
        all_gt_s: list[float] = []
        all_gt_e: list[float] = []

        for batch in tqdm(self.dataloader, desc="Eval", leave=False):
            v = batch["video_feat"].to(self.device)
            vm = batch["video_mask"].to(self.device)
            t = batch["text_feat"].to(self.device)
            tm = batch["text_mask"].to(self.device)

            output = self.model(v, vm, t, tm)

            # Decode predictions
            decoded = decode_moments(
                output.cls_logits,
                output.reg_offsets,
                batch["feature_stride"].to(self.device),
                downsample_rate=self.model.downsample_rate,
            )

            # Apply Soft-NMS per sample
            B = v.shape[0]
            for b in range(B):
                d = decoded[b]
                # Pre-filter: top-k before NMS
                if d["scores"].numel() > ecfg.pre_nms_topk:
                    _, topk_idx = d["scores"].topk(ecfg.pre_nms_topk)
                    d = {k: v[topk_idx] if isinstance(v, torch.Tensor) else v for k, v in d.items()}

                # Clamp starts/ends to valid range
                dur = batch["duration"][b].item()
                d["starts"] = d["starts"].clamp(min=0, max=dur)
                d["ends"] = d["ends"].clamp(min=0, max=dur)

                # Soft-NMS
                sc, st, en = soft_nms(
                    d["scores"].cpu(), d["starts"].cpu(), d["ends"].cpu(),
                    sigma=ecfg.nms_sigma,
                    threshold=ecfg.nms_threshold,
                    max_predictions=ecfg.max_predictions,
                )
                all_preds.append({"scores": sc, "starts": st, "ends": en})
                all_gt_s.append(batch["gt_start"][b].item())
                all_gt_e.append(batch["gt_end"][b].item())

        metrics = compute_recall(
            all_preds, all_gt_s, all_gt_e,
            recall_k=ecfg.recall_k,
            tiou_thresholds=ecfg.tiou_thresholds,
        )
        log_metrics(metrics, prefix="Eval")
        return metrics
