"""Tests for DR screening preprocessing pipeline."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bioagentics.diagnostics.retinal_dr_screening.preprocessing import (
    PreprocessingParams,
    ben_graham_preprocess,
    crop_fundus_circle,
    normalize_for_model,
    preprocess_dataset,
    preprocess_image,
    resize_image,
    score_image_quality,
)


def _make_fundus_image(size: int = 400, border: int = 50) -> np.ndarray:
    """Create a synthetic fundus-like image with black borders."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius = size // 2 - border
    cv2.circle(img, (center, center), radius, (80, 60, 120), -1)
    # Add some texture
    noise = np.random.default_rng(42).integers(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def test_crop_fundus_circle():
    img = _make_fundus_image(400, border=80)
    cropped = crop_fundus_circle(img)
    # Cropped should be smaller or equal (borders removed where possible)
    assert cropped.shape[0] <= img.shape[0]
    assert cropped.shape[1] <= img.shape[1]
    assert cropped.shape[2] == 3
    # With padding, the bounding rect may cover full image, but crop should still work
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0


def test_crop_fundus_circle_no_border():
    """Image without black borders should be returned mostly unchanged."""
    img = np.full((200, 200, 3), 100, dtype=np.uint8)
    cropped = crop_fundus_circle(img)
    assert cropped.shape[2] == 3


def test_resize_image_square():
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    resized = resize_image(img, 128)
    assert resized.shape == (128, 128, 3)


def test_resize_image_preserves_content():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    resized = resize_image(img, 64)
    # Center should have content
    center = resized[32, 32]
    assert np.any(center > 0)


def test_ben_graham_preprocess():
    img = _make_fundus_image(200, border=20)
    result = ben_graham_preprocess(img, sigma=10, weight=4.0)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    # Result should have enhanced local contrast (mean near 128 for uniform regions)
    mean_val = np.mean(result)
    assert 50 < mean_val < 200


def test_ben_graham_output_range():
    img = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    result = ben_graham_preprocess(img)
    assert result.min() >= 0
    assert result.max() <= 255


def test_normalize_for_model():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    normalized = normalize_for_model(img)
    assert normalized.dtype == np.float32
    assert normalized.shape == (64, 64, 3)
    # With uniform 128/255 ≈ 0.502, after ImageNet normalization values should be near 0
    assert abs(np.mean(normalized)) < 1.0


def test_score_image_quality_good():
    """Bright, sharp image should be gradable."""
    img = _make_fundus_image(200, border=20)
    params = PreprocessingParams(
        quality_min_brightness=10.0,
        quality_min_contrast=5.0,
        quality_min_laplacian_var=1.0,
    )
    score = score_image_quality(img, params)
    assert score.brightness > 0
    assert score.contrast > 0
    assert score.sharpness > 0
    assert score.is_gradable is True


def test_score_image_quality_dark():
    """Very dark image should not be gradable."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = 5  # Very dark
    score = score_image_quality(img)
    assert score.brightness < 30
    assert score.is_gradable is False


def test_score_image_quality_blurry():
    """Very blurry image should not be gradable."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Uniform = zero Laplacian variance = extremely blurry
    score = score_image_quality(img)
    assert score.sharpness < 1.0
    assert score.is_gradable is False


def test_preprocess_image_full_pipeline():
    img = _make_fundus_image(400, border=50)
    params = PreprocessingParams(target_size=128)
    processed, quality = preprocess_image(img, params)
    assert processed.shape == (128, 128, 3)
    assert processed.dtype == np.uint8
    assert isinstance(quality.brightness, float)


def test_preprocess_dataset(tmp_path):
    """Full dataset preprocessing pipeline."""
    # Create mock images
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        img = _make_fundus_image(200, border=30)
        cv2.imwrite(str(img_dir / f"img_{i}.png"), img)

    # Create mock catalog
    catalog = pd.DataFrame({
        "image_path": [str(img_dir / f"img_{i}.png") for i in range(3)],
        "dr_grade": [0, 2, 4],
        "dataset_source": ["test"] * 3,
        "original_filename": [f"img_{i}.png" for i in range(3)],
    })
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    output_dir = tmp_path / "processed"
    params = PreprocessingParams(target_size=64)

    result = preprocess_dataset(catalog_path, output_dir, params)

    assert len(result) == 3
    assert "quality_brightness" in result.columns
    assert "is_gradable" in result.columns
    assert (output_dir / "catalog_processed.csv").exists()
    assert (output_dir / "preprocessing_params.json").exists()

    # Check params saved correctly
    with open(output_dir / "preprocessing_params.json") as f:
        saved_params = json.load(f)
    assert saved_params["target_size"] == 64

    # Check processed images exist
    for _, row in result.iterrows():
        assert Path(row["image_path"]).exists()


def test_preprocessing_params_defaults():
    params = PreprocessingParams()
    assert params.target_size == 512
    assert params.ben_graham_sigma == 10
    assert params.quality_min_brightness == 30.0
