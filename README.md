# MSCL: Multi-Scale Contrastive Learning for Video Temporal Grounding

A PyTorch implementation of the method described in:

> **Multi-Scale Contrastive Learning for Video Temporal Grounding**
> Thong Thanh Nguyen, Yi Bin, Xiaobao Wu, Zhiyuan Hu, Cong-Duy Nguyen, See-Kiong Ng, Anh Tuan Luu
> *AAAI 2025* — [arXiv:2412.07157](https://arxiv.org/abs/2412.07157)

---

## Overview

This codebase implements the full MSCL-VTG pipeline:

- **Video Encoder**: Multi-scale Transformer with local self-attention and depthwise strided downsampling, producing a feature pyramid.
- **Text Encoder**: Lightweight Transformer over pre-extracted text features.
- **Cross-Modal Fusion**: Cross-attention fusing each pyramid level with text representations.
- **Prediction Heads**: 1D convolutional heads for classification (focal loss) and boundary regression (DIoU loss).
- **Multi-Scale Contrastive Learning**: Within-scale and cross-scale contrastive losses that pull target moments together and push non-target moments apart across the feature pyramid.

Supported datasets: **Ego4D-NLQ**, **MAD**, **TACoS**, **ActivityNet-Captions**, **Charades-STA**.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone <repo-url> && cd mscl_vtg

# Install dependencies
uv sync

# Run tests
uv run pytest
```

## Repository Structure

```
mscl_vtg/
├── configs/                    # YAML experiment configs
│   ├── dummy.yaml              # Minimal config for testing
│   ├── tacos_c3d_glove.yaml
│   ├── ego4d_slowfast_bert.yaml
│   ├── ego4d_egovlp.yaml
│   ├── mad_clip.yaml
│   ├── activitynet_c3d_glove.yaml
│   └── charades_i3d_glove.yaml
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation entry point
│   ├── download_datasets.py    # Download instructions per dataset
│   ├── prepare_annotations.py  # Convert raw annotations to unified JSON
│   └── prepare_features.py     # Convert raw features to .npy format
├── src/mscl_vtg/
│   ├── config.py               # Pydantic config schema
│   ├── data/                   # Dataloaders per dataset
│   ├── models/                 # Video encoder, text encoder, fusion, heads
│   ├── losses/                 # Focal, DIoU, within/cross-scale contrastive
│   ├── engine/                 # Trainer and Evaluator
│   └── utils/                  # Soft-NMS, metrics, checkpointing, seeding
├── tests/                      # Unit tests
├── pyproject.toml
└── README.md
```

## Data Preparation

### Expected Feature Format

All dataloaders expect **pre-extracted** features as `.npy` files:

| Type | Path | Shape |
|------|------|-------|
| Video | `<video_feat_dir>/<video_id>.npy` | `(T, D_v)` |
| Text | `<text_feat_dir>/<video_id>_<query_id>.npy` or `<query_id>.npy` | `(K, D_t)` |

### Annotation Format

Annotations are JSON files, one per split. Two formats are accepted:

**Dict format** (TACoS, ActivityNet style):
```json
{
  "video_001": {
    "duration": 120.0,
    "timestamps": [[5.2, 12.8], [30.0, 45.5]],
    "sentences": ["person picks up the bowl", "person stirs the soup"]
  }
}
```

**List format** (Ego4D, MAD style):
```json
[
  {"video_id": "v001", "query_id": "q001", "query": "...", "start": 5.2, "end": 12.8, "duration": 120.0}
]
```

### Per-Dataset Setup

| Dataset | Video Features | Text Features | Feature Dim (V/T) |
|---------|---------------|---------------|-------------------|
| Ego4D-NLQ (SF+BERT) | SlowFast | BERT | 2304 / 768 |
| Ego4D-NLQ (EgoVLP) | EgoVLP | EgoVLP | 256 / 256 |
| MAD | CLIP | CLIP | 512 / 512 |
| TACoS | C3D | GloVe | 4096 / 300 |
| ActivityNet-Captions | C3D | GloVe | 4096 / 300 |
| Charades-STA | I3D | GloVe | 1024 / 300 |

Run `uv run python scripts/download_datasets.py --dataset <name>` for per-dataset instructions.

## Training

```bash
uv run python scripts/train.py --config configs/tacos_c3d_glove.yaml
```

Key training features:
- **AMP** mixed-precision (disable with `train.amp: false`)
- **Warmup + cosine** learning rate schedule
- **Gradient clipping** (default 1.0)
- **Periodic validation** with best-checkpoint saving (by R@1)
- **Resume** training: set `train.resume: path/to/latest.ckpt`
- Individual loss terms logged every `log_interval` steps

## Evaluation

```bash
uv run python scripts/evaluate.py \
    --config configs/tacos_c3d_glove.yaml \
    --checkpoint checkpoints/tacos_c3d_glove/best.ckpt
```

### Metrics per Dataset

| Dataset | R@K | tIoU |
|---------|-----|------|
| Ego4D-NLQ | 1, 5 | 0.3, 0.5 |
| MAD | 1, 5, 10, 50 | 0.1, 0.3, 0.5 |
| TACoS | 1, 5 | 0.5, 0.7 |
| ActivityNet-Captions | 1, 5 | 0.3, 0.5 |
| Charades-STA | 1, 5 | 0.5, 0.7 |

## Testing

```bash
# Run all unit tests
uv run pytest

# Quick smoke test with dummy data
uv run python scripts/train.py --config configs/dummy.yaml
```

## Configuration

All configs are YAML files loaded into a Pydantic schema (`src/mscl_vtg/config.py`). Key sections:

- **data**: dataset name, feature paths, dimensions, max lengths
- **model**: hidden dim, num layers (L), window size (W), heads, downsample rate
- **loss**: focal loss α/γ, loss weight ρ values, center sampling α, temperature
- **train**: epochs, batch size, LR, AMP, checkpoint paths
- **eval**: recall K values, tIoU thresholds, Soft-NMS parameters

## Assumptions and Notes

The following design decisions were made where the paper omits implementation details:

1. **Video projection**: 1D convolution with kernel size 3 for initial projection (Eq. 1). The paper says "convolution-based projection" without specifying kernel size.

2. **Cross-modal fusion (Eq. 6)**: The paper's formula `σ(Z̃^T · Ẽ / √D) · Z̃` is ambiguous about whether the softmax output re-weights video or text features. We follow the literal formula: attention weights from video-text similarity applied to video features.

3. **Downsampling**: Implemented as strided depthwise 1D convolution (kernel=3, stride=2) following the SnAG paper referenced.

4. **Learnable scaling factors** (α, ᾱ, β): Initialized to ones, following the CaiT paper (Touvron et al., 2021).

5. **Prediction head**: Shared across pyramid levels. The paper doesn't specify separate vs shared heads.

6. **Feature stride**: Dataset-dependent; values in configs are reasonable defaults. Adjust based on your actual feature extraction setup.

7. **Center sampling α**: Default 1.5, following ActionFormer/SnAG conventions.

8. **Contrastive temperature**: Default 1.0 (paper uses dot product directly in Eqs. 11-12 without explicit temperature; we add it as a tunable parameter).

9. **Negative sampling**: `|N(l)| = |P(l)|` as stated in the paper.

10. **Scale factor in Eq. 9**: The paper uses `2^{l-1}`; our implementation uses `downsample_rate^l` since head level indexing starts from 1.

All assumptions are marked with `TODO` or `NOTE` comments in the code for easy modification.

## Tips for Reproduction

- For **TACoS**: The most commonly used C3D features can be found in repos like 2D-TAN or TALL.
- For **Ego4D**: Use the official Ego4D CLI for consistent feature extraction.
- For **MAD**: CLIP features are provided by the MAD authors on their GitHub.
- Start with `configs/dummy.yaml` to verify the pipeline runs end-to-end before using real data.
- Monitor all four loss terms during training — the contrastive losses should decrease steadily.

## Citation

```bibtex
@inproceedings{nguyen2025multi,
  title={Multi-scale contrastive learning for video temporal grounding},
  author={Nguyen, Thong Thanh and Bin, Yi and Wu, Xiaobao and Hu, Zhiyuan and Nguyen, Cong-Duy T and Ng, See-Kiong and Luu, Anh Tuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={6227--6235},
  year={2025}
}
```

## License

This is a research reimplementation. Please cite the original paper if you use this code.
