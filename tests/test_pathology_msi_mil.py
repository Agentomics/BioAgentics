"""Tests for MIL models (ABMIL, CLAM, TransMIL)."""

import torch
import pytest

from bioagentics.models.pathology_msi.mil_models import (
    ABMIL,
    CLAM,
    TransMIL,
    GatedAttention,
    MILOutput,
    create_mil_model,
)


@pytest.fixture
def sample_input():
    """Sample input: batch of 2 slides, 50 patches each, 1024-dim features."""
    return torch.randn(2, 50, 1024)


@pytest.fixture
def small_input():
    """Small input for quick testing."""
    return torch.randn(1, 10, 512)


class TestGatedAttention:
    def test_output_shapes(self):
        ga = GatedAttention(256, 128)
        h = torch.randn(2, 50, 256)
        attn, weighted = ga(h)
        assert attn.shape == (2, 50)
        assert weighted.shape == (2, 256)

    def test_attention_sums_to_one(self):
        ga = GatedAttention(256, 128)
        h = torch.randn(2, 50, 256)
        attn, _ = ga(h)
        sums = attn.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


class TestABMIL:
    def test_output_types(self, sample_input):
        model = ABMIL(input_dim=1024)
        output = model(sample_input)
        assert isinstance(output, MILOutput)

    def test_output_shapes(self, sample_input):
        model = ABMIL(input_dim=1024, n_classes=2)
        output = model(sample_input)
        assert output.logits.shape == (2, 2)
        assert output.attention_weights.shape == (2, 50)
        assert output.bag_repr.shape[0] == 2

    def test_attention_sums_to_one(self, sample_input):
        model = ABMIL(input_dim=1024)
        output = model(sample_input)
        sums = output.attention_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_variable_length_input(self):
        model = ABMIL(input_dim=1024)
        # Different number of patches per slide
        x1 = torch.randn(1, 20, 1024)
        x2 = torch.randn(1, 100, 1024)
        o1 = model(x1)
        o2 = model(x2)
        assert o1.logits.shape == (1, 2)
        assert o2.logits.shape == (1, 2)
        assert o1.attention_weights.shape == (1, 20)
        assert o2.attention_weights.shape == (1, 100)


class TestCLAM:
    def test_output_shapes(self, sample_input):
        model = CLAM(input_dim=1024, n_classes=2)
        output = model(sample_input)
        assert output.logits.shape == (2, 2)
        assert output.attention_weights.shape == (2, 50)

    def test_multiclass(self, sample_input):
        model = CLAM(input_dim=1024, n_classes=3)
        output = model(sample_input)
        assert output.logits.shape == (2, 3)

    def test_attention_sums_to_one(self, sample_input):
        model = CLAM(input_dim=1024)
        output = model(sample_input)
        sums = output.attention_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


class TestTransMIL:
    def test_output_shapes(self, small_input):
        model = TransMIL(input_dim=512, hidden_dim=256, n_classes=2, n_layers=1)
        output = model(small_input)
        assert output.logits.shape == (1, 2)
        assert output.attention_weights.shape == (1, 10)

    def test_attention_sums_to_one(self, small_input):
        model = TransMIL(input_dim=512, hidden_dim=256, n_layers=1)
        output = model(small_input)
        sums = output.attention_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(1), atol=1e-5)

    def test_larger_input_nystrom(self):
        model = TransMIL(
            input_dim=512, hidden_dim=256, n_classes=2,
            n_layers=1, n_landmarks=8,
        )
        # More patches than landmarks to trigger Nystrom
        x = torch.randn(1, 100, 512)
        output = model(x)
        assert output.logits.shape == (1, 2)
        assert output.attention_weights.shape == (1, 100)


class TestFactory:
    def test_create_abmil(self):
        model = create_mil_model("abmil", input_dim=768)
        assert isinstance(model, ABMIL)

    def test_create_clam(self):
        model = create_mil_model("clam", input_dim=768)
        assert isinstance(model, CLAM)

    def test_create_transmil(self):
        model = create_mil_model("transmil", input_dim=768)
        assert isinstance(model, TransMIL)

    def test_case_insensitive(self):
        model = create_mil_model("ABMIL", input_dim=768)
        assert isinstance(model, ABMIL)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_mil_model("unknown_model")


class TestGradientFlow:
    """Verify gradients flow through all models."""

    def test_abmil_gradient(self):
        model = ABMIL(input_dim=512, hidden_dim=128)
        x = torch.randn(1, 20, 512, requires_grad=True)
        output = model(x)
        loss = output.logits.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_clam_gradient(self):
        model = CLAM(input_dim=512, hidden_dim=128)
        x = torch.randn(1, 20, 512, requires_grad=True)
        output = model(x)
        loss = output.logits.sum()
        loss.backward()
        assert x.grad is not None

    def test_transmil_gradient(self):
        model = TransMIL(input_dim=512, hidden_dim=128, n_layers=1)
        x = torch.randn(1, 20, 512, requires_grad=True)
        output = model(x)
        loss = output.logits.sum()
        loss.backward()
        assert x.grad is not None
