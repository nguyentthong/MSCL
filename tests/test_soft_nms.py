"""Tests for Soft-NMS."""
import torch

from mscl_vtg.utils.soft_nms import soft_nms


def test_soft_nms_basic():
    scores = torch.tensor([0.9, 0.8, 0.3, 0.1])
    starts = torch.tensor([0.0, 0.5, 5.0, 10.0])
    ends = torch.tensor([2.0, 2.5, 7.0, 12.0])

    sc, st, en = soft_nms(scores, starts, ends, sigma=0.5, threshold=0.01)
    assert len(sc) > 0
    # Highest score should be first
    assert sc[0].item() >= sc[-1].item()
    print("test_soft_nms_basic PASSED")


def test_soft_nms_identical():
    """Identical segments: second should be suppressed."""
    scores = torch.tensor([0.9, 0.85])
    starts = torch.tensor([1.0, 1.0])
    ends = torch.tensor([3.0, 3.0])

    sc, st, en = soft_nms(scores, starts, ends, sigma=0.3, threshold=0.01)
    # After NMS, second score should be much reduced
    assert len(sc) >= 1
    if len(sc) > 1:
        assert sc[1].item() < 0.5  # significantly suppressed
    print("test_soft_nms_identical PASSED")


def test_soft_nms_empty():
    sc, st, en = soft_nms(torch.empty(0), torch.empty(0), torch.empty(0))
    assert len(sc) == 0
    print("test_soft_nms_empty PASSED")


def test_soft_nms_nonoverlap():
    """Non-overlapping segments should all survive."""
    scores = torch.tensor([0.9, 0.8, 0.7])
    starts = torch.tensor([0.0, 10.0, 20.0])
    ends = torch.tensor([2.0, 12.0, 22.0])

    sc, st, en = soft_nms(scores, starts, ends, sigma=0.5, threshold=0.01)
    assert len(sc) == 3
    print("test_soft_nms_nonoverlap PASSED")


if __name__ == "__main__":
    test_soft_nms_basic()
    test_soft_nms_identical()
    test_soft_nms_empty()
    test_soft_nms_nonoverlap()
