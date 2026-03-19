"""Tests for attention heatmap visualization module."""

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bioagentics.models.pathology_msi.heatmaps import (
    extract_attention,
    generate_heatmap,
    generate_slide_heatmap,
)
from bioagentics.models.pathology_msi.mil_models import create_mil_model


@pytest.fixture
def slide_data(tmp_path):
    """Create test slide data: features, tile coords, thumbnail."""
    n_patches = 50
    feat_dim = 256

    # Feature file
    features_path = tmp_path / "test_slide.h5"
    features = np.random.randn(n_patches, feat_dim).astype(np.float32)
    with h5py.File(features_path, "w") as f:
        f.create_dataset("features", data=features)

    # Tile coordinates (5x10 grid of 256px tiles)
    rows = []
    for i in range(n_patches):
        rows.append({
            "x": (i % 10) * 256,
            "y": (i // 10) * 256,
            "tile_size": 256,
            "level": 0,
            "downsample": 1.0,
        })
    coords_path = tmp_path / "test_slide_tiles.csv"
    pd.DataFrame(rows).to_csv(coords_path, index=False)

    # Slide dimensions (10 tiles wide, 5 tiles tall)
    slide_dims = (2560, 1280)

    # Thumbnail image
    thumbnail = Image.fromarray(
        np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
    )

    return features_path, coords_path, slide_dims, thumbnail


@pytest.fixture
def trained_model():
    """Create a simple trained model."""
    return create_mil_model("abmil", input_dim=256, n_classes=2)


class TestExtractAttention:
    def test_returns_attention_and_probs(self, slide_data, trained_model):
        features_path, _, _, _ = slide_data
        attention, probs = extract_attention(trained_model, features_path)

        assert attention.ndim == 1
        assert attention.shape[0] == 50  # n_patches
        assert probs.ndim == 1
        assert probs.shape[0] == 2  # n_classes

    def test_attention_sums_to_one(self, slide_data, trained_model):
        features_path, _, _, _ = slide_data
        attention, _ = extract_attention(trained_model, features_path)
        assert np.isclose(attention.sum(), 1.0, atol=1e-5)

    def test_probs_sum_to_one(self, slide_data, trained_model):
        features_path, _, _, _ = slide_data
        _, probs = extract_attention(trained_model, features_path)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)


class TestGenerateHeatmap:
    def test_returns_image(self, slide_data):
        _, coords_path, slide_dims, _ = slide_data
        attention = np.random.rand(50)
        coords = pd.read_csv(coords_path)

        heatmap = generate_heatmap(attention, coords, slide_dims, thumbnail_size=512)
        assert isinstance(heatmap, Image.Image)
        assert heatmap.size[0] <= 512
        assert heatmap.size[1] <= 512

    def test_with_thumbnail_overlay(self, slide_data):
        _, coords_path, slide_dims, thumbnail = slide_data
        attention = np.random.rand(50)
        coords = pd.read_csv(coords_path)

        heatmap = generate_heatmap(
            attention, coords, slide_dims,
            thumbnail_size=512,
            thumbnail_image=thumbnail,
        )
        assert isinstance(heatmap, Image.Image)

    def test_uniform_attention(self, slide_data):
        _, coords_path, slide_dims, _ = slide_data
        attention = np.ones(50)  # uniform
        coords = pd.read_csv(coords_path)

        heatmap = generate_heatmap(attention, coords, slide_dims, thumbnail_size=256)
        assert isinstance(heatmap, Image.Image)

    def test_handles_mismatched_lengths(self, slide_data):
        _, coords_path, slide_dims, _ = slide_data
        # More attention values than coords
        attention = np.random.rand(100)
        coords = pd.read_csv(coords_path)

        heatmap = generate_heatmap(attention, coords, slide_dims, thumbnail_size=256)
        assert isinstance(heatmap, Image.Image)


class TestGenerateSlideHeatmap:
    def test_saves_heatmap(self, slide_data, trained_model, tmp_path):
        features_path, coords_path, slide_dims, _ = slide_data
        output_path = tmp_path / "heatmap.png"

        result = generate_slide_heatmap(
            trained_model,
            features_path,
            coords_path,
            slide_dims,
            output_path,
        )

        assert output_path.exists()
        assert result["slide"] == "test_slide"
        assert result["n_patches"] == 50
        assert "prediction_class" in result
        assert result["prediction_class"] in ("MSI-H", "MSS")
        assert 0.0 <= result["prediction_msi_h_prob"] <= 1.0

    def test_with_thumbnail(self, slide_data, trained_model, tmp_path):
        features_path, coords_path, slide_dims, thumbnail = slide_data
        output_path = tmp_path / "heatmap_overlay.png"

        result = generate_slide_heatmap(
            trained_model,
            features_path,
            coords_path,
            slide_dims,
            output_path,
            thumbnail_image=thumbnail,
        )

        assert output_path.exists()
        img = Image.open(output_path)
        assert img.mode == "RGB"
