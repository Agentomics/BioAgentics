"""Tests for CXR long-tail loss functions."""

from __future__ import annotations

import torch
import pytest

from bioagentics.cxr_rare.losses import (
    AsymmetricLoss,
    ClassBalancedFocalLoss,
    FocalLoss,
    LDAMLoss,
    LDAMDRW,
)


@pytest.fixture
def sample_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample logits and targets for testing."""
    torch.manual_seed(42)
    logits = torch.randn(8, 10, requires_grad=True)
    targets = (torch.rand(8, 10) > 0.7).float()
    return logits, targets


class TestFocalLoss:
    def test_output_is_scalar(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        loss = FocalLoss(gamma=2.0)(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        loss = FocalLoss(gamma=2.0)(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_gamma_zero_matches_bce(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        focal = FocalLoss(gamma=0.0)(logits.detach(), targets)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits.detach(), targets)
        assert torch.allclose(focal, bce, atol=1e-5)

    def test_all_zero_labels(self) -> None:
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.zeros(4, 5)
        loss = FocalLoss()(logits, targets)
        loss.backward()
        assert torch.isfinite(loss)

    def test_reduction_none(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        loss = FocalLoss(reduction="none")(logits, targets)
        assert loss.shape == logits.shape


class TestClassBalancedFocalLoss:
    def test_output_is_scalar(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        loss = ClassBalancedFocalLoss(counts, gamma=2.0)(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        loss = ClassBalancedFocalLoss(counts)(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_rare_classes_weighted_higher(self) -> None:
        counts = [1000, 1000, 1000, 1, 1]
        cb = ClassBalancedFocalLoss(counts, gamma=0.0)
        # Weights for rare classes should be higher
        weights = cb.weights
        assert weights[3] > weights[0]

    def test_single_sample_class(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        loss = ClassBalancedFocalLoss(counts)(logits, targets)
        assert torch.isfinite(loss)


class TestAsymmetricLoss:
    def test_output_is_scalar(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        loss = AsymmetricLoss()(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        loss = AsymmetricLoss()(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_all_zero_labels(self) -> None:
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.zeros(4, 5)
        loss = AsymmetricLoss()(logits, targets)
        loss.backward()
        assert torch.isfinite(loss)

    def test_asymmetric_behavior(self) -> None:
        # With higher gamma_neg, negative loss should be more suppressed
        logits = torch.randn(16, 5)
        targets = torch.zeros(16, 5)
        targets[:2] = 1.0

        asl_low = AsymmetricLoss(gamma_pos=0, gamma_neg=0, clip=0.0)(logits, targets)
        asl_high = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)(logits, targets)
        # Higher negative focusing should reduce loss on easy negatives
        assert asl_high < asl_low


class TestLDAMLoss:
    def test_output_is_scalar(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        loss = LDAMLoss(counts)(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        loss = LDAMLoss(counts)(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_rare_classes_get_larger_margins(self) -> None:
        counts = [10000, 10000, 1, 1, 1]
        ldam = LDAMLoss(counts, max_margin=0.5)
        # Rare classes (index 2,3,4) should have larger margins
        assert ldam.margins[2] > ldam.margins[0]

    def test_with_class_weights(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        weights = torch.ones(10) * 2.0
        loss = LDAMLoss(counts, class_weights=weights)(logits, targets)
        assert torch.isfinite(loss)

    def test_reduction_none(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        loss = LDAMLoss(counts, reduction="none")(logits, targets)
        assert loss.shape == logits.shape


class TestLDAMDRW:
    def test_output_is_scalar(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        loss_fn = LDAMDRW(counts, drw_start_epoch=3)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        loss_fn = LDAMDRW(counts)
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_drw_schedule_uniform_before(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        loss_fn = LDAMDRW(counts, drw_start_epoch=5)

        loss_fn.set_epoch(0)
        assert not loss_fn.is_reweighting
        assert loss_fn.ldam.class_weights is None

    def test_drw_schedule_reweighted_after(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [100, 50, 30, 20, 10, 5, 3, 2, 1, 1]
        loss_fn = LDAMDRW(counts, drw_start_epoch=5)

        loss_fn.set_epoch(5)
        assert loss_fn.is_reweighting
        assert loss_fn.ldam.class_weights is not None

    def test_drw_changes_loss(self, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        logits, targets = sample_data
        counts = [10000, 10000, 10000, 1, 1, 1, 1, 1, 1, 1]
        loss_fn = LDAMDRW(counts, drw_start_epoch=3)

        loss_fn.set_epoch(0)
        loss_uniform = loss_fn(logits.detach(), targets).item()

        loss_fn.set_epoch(3)
        loss_reweighted = loss_fn(logits.detach(), targets).item()

        # Loss values should differ once reweighting kicks in
        assert loss_uniform != loss_reweighted

    def test_current_epoch_tracking(self) -> None:
        counts = [10, 5, 1]
        loss_fn = LDAMDRW(counts, drw_start_epoch=2)
        assert loss_fn.current_epoch == 0
        loss_fn.set_epoch(3)
        assert loss_fn.current_epoch == 3
