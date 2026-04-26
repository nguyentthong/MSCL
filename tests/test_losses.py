"""Tests for all loss functions."""
import torch

from mscl_vtg.losses.focal import sigmoid_focal_loss
from mscl_vtg.losses.diou import diou_loss
from mscl_vtg.losses.contrastive import within_scale_contrastive_loss, cross_scale_contrastive_loss


def test_focal_loss_basic():
    logits = torch.tensor([2.0, -2.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = sigmoid_focal_loss(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0
    print("test_focal_loss_basic PASSED")


def test_focal_loss_perfect():
    """Perfect predictions should give low loss."""
    logits = torch.tensor([10.0, -10.0])
    targets = torch.tensor([1.0, 0.0])
    loss = sigmoid_focal_loss(logits, targets)
    assert loss.item() < 0.01
    print("test_focal_loss_perfect PASSED")


def test_diou_loss_identical():
    """Identical segments → loss ≈ 0."""
    s = torch.tensor([1.0, 5.0])
    e = torch.tensor([3.0, 10.0])
    loss = diou_loss(s, e, s, e)
    assert loss.item() < 1e-5
    print("test_diou_loss_identical PASSED")


def test_diou_loss_nonoverlap():
    """Non-overlapping segments → loss ≈ 2."""
    pred_s = torch.tensor([0.0])
    pred_e = torch.tensor([1.0])
    gt_s = torch.tensor([5.0])
    gt_e = torch.tensor([6.0])
    loss = diou_loss(pred_s, pred_e, gt_s, gt_e)
    assert loss.item() > 1.0
    print("test_diou_loss_nonoverlap PASSED")


def test_within_scale_contrastive():
    B, T, D = 1, 16, 32
    feats = [torch.randn(B, T, D)]
    pos_masks = [torch.zeros(B, T, dtype=torch.bool)]
    pos_masks[0][0, 3:7] = True
    valid = [torch.ones(B, T)]
    loss = within_scale_contrastive_loss(feats, pos_masks, valid)
    assert loss.ndim == 0
    assert loss.item() >= 0
    print("test_within_scale_contrastive PASSED")


def test_cross_scale_contrastive():
    B, D = 1, 32
    feats = [
        torch.randn(B, 16, D),   # level 0
        torch.randn(B, 8, D),    # level 1
    ]
    pos_masks = [
        torch.zeros(B, 16, dtype=torch.bool),
        torch.zeros(B, 8, dtype=torch.bool),
    ]
    pos_masks[0][0, 3:7] = True
    pos_masks[1][0, 1:4] = True
    valid = [torch.ones(B, 16), torch.ones(B, 8)]
    loss = cross_scale_contrastive_loss(feats, pos_masks, valid)
    assert loss.ndim == 0
    assert loss.item() >= 0
    print("test_cross_scale_contrastive PASSED")


if __name__ == "__main__":
    test_focal_loss_basic()
    test_focal_loss_perfect()
    test_diou_loss_identical()
    test_diou_loss_nonoverlap()
    test_within_scale_contrastive()
    test_cross_scale_contrastive()
