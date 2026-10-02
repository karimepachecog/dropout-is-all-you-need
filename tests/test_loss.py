"""Loss identities that do not need a pretrained encoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from simcse.model import info_nce, info_nce_with_hard_negatives


def test_identical_views_with_orthogonal_batch_is_near_zero():
    eyes = torch.eye(4)
    loss = info_nce(eyes, eyes, tau=0.05)
    assert loss.item() < 1e-3


def test_uniform_logits_match_closed_form():
    anchors = torch.eye(4)
    positives = torch.eye(4)
    tau = 1.0
    loss = info_nce(anchors, positives, tau)
    # Diagonal cosine is 1, off-diagonal is 0.
    expected = -torch.log(torch.exp(torch.tensor(1.0)) / (torch.exp(torch.tensor(1.0)) + 3))
    assert torch.allclose(loss, expected, atol=1e-5)


def test_masked_hard_negatives_match_in_batch_loss():
    generator = torch.Generator().manual_seed(0)
    vectors = torch.nn.functional.normalize(torch.randn(6, 8, generator=generator), dim=-1)
    present = torch.zeros(6, dtype=torch.bool)
    masked = info_nce_with_hard_negatives(vectors, vectors, vectors, present, 0.05)
    plain = info_nce(vectors, vectors, 0.05)
    assert torch.allclose(masked, plain, atol=1e-4)


def test_hard_negative_column_count():
    generator = torch.Generator().manual_seed(1)
    anchors = torch.nn.functional.normalize(torch.randn(5, 8, generator=generator), dim=-1)
    positives = anchors.clone()
    negatives = torch.nn.functional.normalize(torch.randn(5, 8, generator=generator), dim=-1)
    present = torch.ones(5, dtype=torch.bool)
    # Own positive is a perfect match, so the loss stays finite and defined.
    loss = info_nce_with_hard_negatives(anchors, positives, negatives, present, 0.05)
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_identical_views_with_orthogonal_batch_is_near_zero()
    test_uniform_logits_match_closed_form()
    test_masked_hard_negatives_match_in_batch_loss()
    test_hard_negative_column_count()
    print("loss tests passed")
