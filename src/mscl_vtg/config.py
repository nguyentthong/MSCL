"""Pydantic-based hierarchical configuration for MSCL-VTG."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------
class DataConfig(BaseModel):
    dataset: str = "tacos"                       # ego4d | mad | tacos | activitynet | charades
    anno_dir: str = "data/annotations"
    video_feat_dir: str = "data/features/video"
    text_feat_dir: str = "data/features/text"
    video_feat_dim: int = 2048
    text_feat_dim: int = 300
    max_video_len: int = 2304                    # max #clips after padding
    max_query_len: int = 32
    feature_stride: float = 1.0                  # seconds per clip feature
    num_workers: int = 4
    # video-centric sampling
    num_queries_per_video: int = -1              # -1 = all queries


class ModelConfig(BaseModel):
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6                          # L in paper
    window_size: int = 19                        # W for local self-attention
    downsample_rate: int = 2
    dropout: float = 0.0
    pre_norm: bool = True
    text_encoder_layers: int = 2
    # head
    head_layers: int = 3
    head_kernel_size: int = 3


class LossConfig(BaseModel):
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    rho_reg: float = 1.0
    rho_within: float = 1.0
    rho_cross: float = 1.0
    center_sampling_alpha: float = 1.5           # α for center sampling
    temperature: float = 1.0                     # optional temperature for contrastive


class TrainConfig(BaseModel):
    epochs: int = 30
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    amp: bool = True
    val_interval: int = 1
    save_dir: str = "checkpoints"
    log_interval: int = 50
    resume: str = ""                             # path to checkpoint to resume from
    seed: int = 42


class EvalConfig(BaseModel):
    batch_size: int = 4
    # per-dataset recall thresholds
    recall_k: list[int] = Field(default_factory=lambda: [1, 5])
    tiou_thresholds: list[float] = Field(default_factory=lambda: [0.5, 0.7])
    # soft-nms
    nms_sigma: float = 0.5
    nms_threshold: float = 0.001
    max_predictions: int = 200
    pre_nms_topk: int = 2000


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------
class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
