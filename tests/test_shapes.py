"""Test that model forward pass produces correct output shapes."""
import math

import torch

from mscl_vtg.config import ModelConfig
from mscl_vtg.models.model import MSCLModel


def test_forward_shapes():
    B, T, D_v, D_t, K = 2, 64, 256, 128, 8
    cfg = ModelConfig(hidden_dim=64, num_heads=2, num_layers=3, window_size=7,
                      downsample_rate=2, text_encoder_layers=1, head_layers=2)
    model = MSCLModel(video_feat_dim=D_v, text_feat_dim=D_t, cfg=cfg)
    model.eval()

    video = torch.randn(B, T, D_v)
    vmask = torch.ones(B, T)
    text = torch.randn(B, K, D_t)
    tmask = torch.ones(B, K)

    with torch.no_grad():
        out = model(video, vmask, text, tmask)

    # Feature pyramid: L+1 levels
    assert len(out.video_feats) == cfg.num_layers + 1
    assert out.video_feats[0].shape == (B, T, cfg.hidden_dim)
    # Each subsequent level halves
    expected_T = T
    for l in range(cfg.num_layers + 1):
        assert out.video_feats[l].shape[1] == expected_T, \
            f"Level {l}: expected T={expected_T}, got {out.video_feats[l].shape[1]}"
        if l < cfg.num_layers:
            expected_T = math.ceil(expected_T / cfg.downsample_rate)

    # Fused feats: L levels (1..L)
    assert len(out.fused_feats) == cfg.num_layers
    # Heads: L levels
    assert len(out.cls_logits) == cfg.num_layers
    assert len(out.reg_offsets) == cfg.num_layers

    for l in range(cfg.num_layers):
        T_l = out.fused_feats[l].shape[1]
        assert out.cls_logits[l].shape == (B, T_l)
        assert out.reg_offsets[l].shape == (B, T_l, 2)

    print("test_forward_shapes PASSED")


def test_model_backward():
    """Ensure gradients flow."""
    B, T, D_v, D_t, K = 1, 32, 64, 32, 4
    cfg = ModelConfig(hidden_dim=32, num_heads=2, num_layers=2, window_size=5,
                      downsample_rate=2, text_encoder_layers=1, head_layers=2)
    model = MSCLModel(video_feat_dim=D_v, text_feat_dim=D_t, cfg=cfg)

    video = torch.randn(B, T, D_v)
    vmask = torch.ones(B, T)
    text = torch.randn(B, K, D_t)
    tmask = torch.ones(B, K)

    out = model(video, vmask, text, tmask)
    loss = sum(l.sum() for l in out.cls_logits)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "No gradients found!"
    print("test_model_backward PASSED")


if __name__ == "__main__":
    test_forward_shapes()
    test_model_backward()
