"""Tests for feature extraction pipelines."""

import numpy as np
import pytest
import torch

from bioagentics.models.pathology_msi.feature_extraction import (
    ResNet50Extractor,
    create_feature_extractor,
)


class TestResNet50Extractor:
    """Test ResNet50 extractor (always available, no special weights needed)."""

    def test_output_dimensions(self):
        extractor = ResNet50Extractor(device="cpu", batch_size=4)
        model = extractor.get_model()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = model(dummy)

        assert features.shape == (2, 2048)

    def test_feature_dim(self):
        extractor = ResNet50Extractor()
        assert extractor.get_feature_dim() == 2048

    def test_transform_output(self):
        from PIL import Image

        extractor = ResNet50Extractor()
        transform = extractor.get_transform()

        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_extract_from_patches(self):
        from torch.utils.data import DataLoader, TensorDataset

        extractor = ResNet50Extractor(device="cpu", batch_size=2)

        # Create a small dummy dataset
        dummy_patches = torch.randn(4, 3, 224, 224)
        loader = DataLoader(TensorDataset(dummy_patches), batch_size=2)

        # Wrap to return tensors directly
        class SimpleLoader:
            def __init__(self, loader):
                self.loader = loader

            def __iter__(self):
                for (batch,) in self.loader:
                    yield batch

        features = extractor.extract_from_patches(SimpleLoader(loader))
        assert features.shape == (4, 2048)
        assert isinstance(features, np.ndarray)


class TestFactory:
    def test_create_resnet50(self):
        extractor = create_feature_extractor("resnet50")
        assert isinstance(extractor, ResNet50Extractor)

    def test_case_insensitive(self):
        extractor = create_feature_extractor("ResNet50")
        assert isinstance(extractor, ResNet50Extractor)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown extractor"):
            create_feature_extractor("unknown")

    def test_uni2_creates(self):
        # Just test it creates without loading weights (timm may not be installed)
        try:
            extractor = create_feature_extractor("uni2")
            assert extractor.get_feature_dim() == 1024
        except ImportError:
            pytest.skip("timm not installed")

    def test_conch_creates(self):
        try:
            extractor = create_feature_extractor("conch")
            assert extractor.get_feature_dim() == 512
        except ImportError:
            pytest.skip("timm not installed")
