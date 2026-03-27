"""Tests for ConvNeXt-Large CXR baseline model."""

from __future__ import annotations

import torch

from bioagentics.cxr_rare.config import NUM_CLASSES
from bioagentics.cxr_rare.models.convnext_baseline import ConvNeXtLargeCXR


class TestConvNeXtLargeCXR:
    def test_output_shape(self) -> None:
        model = ConvNeXtLargeCXR(num_classes=NUM_CLASSES, pretrained=False, dropout=0.0)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, NUM_CLASSES)

    def test_gradient_flows(self) -> None:
        model = ConvNeXtLargeCXR(num_classes=5, pretrained=False, dropout=0.0)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        # Check gradients flow to the backbone
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.all(p.grad == 0)
                break

    def test_custom_num_classes(self) -> None:
        model = ConvNeXtLargeCXR(num_classes=10, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (1, 10)

    def test_dropout_variant(self) -> None:
        model = ConvNeXtLargeCXR(num_classes=5, pretrained=False, dropout=0.5)
        model.train()
        x = torch.randn(4, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (4, 5)
