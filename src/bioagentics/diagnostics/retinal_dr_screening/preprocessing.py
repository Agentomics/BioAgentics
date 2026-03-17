"""Image preprocessing pipeline for DR screening.

Implements:
  1. Black border cropping via circular mask detection
  2. Resizing to standard resolution (512x512 training, 224x224 mobile)
  3. Ben Graham-style preprocessing (local average color subtraction)
  4. Pixel normalization (ImageNet or dataset-computed stats)
  5. Image quality scoring to filter ungradable images

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.preprocessing \\
        --catalog data/diagnostics/smartphone-retinal-dr-screening/catalog.csv \\
        --output-dir data/diagnostics/smartphone-retinal-dr-screening/processed \\
        --size 512
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bioagentics.diagnostics.retinal_dr_screening.config import (
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOBILE_IMAGE_SIZE,
    TRAIN_IMAGE_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingParams:
    """Parameters for reproducible preprocessing."""

    target_size: int = TRAIN_IMAGE_SIZE
    ben_graham_sigma: int = 10  # Gaussian blur kernel = sigma * 2 + 1
    ben_graham_weight: float = 4.0
    quality_min_brightness: float = 30.0
    quality_min_contrast: float = 20.0
    quality_min_laplacian_var: float = 50.0  # focus/sharpness threshold


# ── Step 1: Black border cropping ──


def crop_fundus_circle(image: np.ndarray) -> np.ndarray:
    """Crop black borders from fundus image by detecting the circular field of view.

    Converts to grayscale, thresholds to find the bright fundus region,
    finds the largest contour, and crops to its bounding rect.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to separate fundus from black background
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find the largest contour (the fundus circle)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add small padding to avoid cutting the edge
    pad = max(2, min(w, h) // 100)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2 * pad)
    h = min(image.shape[0] - y, h + 2 * pad)

    return image[y : y + h, x : x + w]


# ── Step 2: Resize ──


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize image to square (size x size) preserving aspect ratio with padding."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to square
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


# ── Step 3: Ben Graham preprocessing ──


def ben_graham_preprocess(
    image: np.ndarray,
    sigma: int = 10,
    weight: float = 4.0,
) -> np.ndarray:
    """Apply Ben Graham-style preprocessing for DR detection.

    Subtracts local average color to enhance local contrast of lesions.
    Formula: output = weight * (image - GaussianBlur(image)) + 128

    Args:
        image: BGR uint8 image.
        sigma: Controls blur kernel size (kernel = sigma * 2 + 1).
        weight: Scaling factor for the high-pass signal.

    Returns:
        Preprocessed uint8 image.
    """
    ksize = sigma * 2 + 1
    blur = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    result = weight * (image.astype(np.float32) - blur.astype(np.float32)) + 128.0
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Step 4: Normalization ──


def normalize_for_model(
    image: np.ndarray,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> np.ndarray:
    """Normalize pixel values to float32 with given mean/std (channel-wise).

    Converts from BGR uint8 to RGB float32 normalized.
    """
    # BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    for c in range(3):
        normalized[:, :, c] = (normalized[:, :, c] - mean[c]) / std[c]
    return normalized


# ── Step 5: Quality scoring ──


@dataclass
class QualityScore:
    """Image quality assessment results."""

    brightness: float
    contrast: float
    sharpness: float  # Laplacian variance
    is_gradable: bool


def score_image_quality(
    image: np.ndarray,
    params: PreprocessingParams | None = None,
) -> QualityScore:
    """Score image quality for DR grading suitability.

    Evaluates:
    - Brightness: mean pixel intensity (too dark = unusable)
    - Contrast: standard deviation of pixel intensities
    - Sharpness: variance of Laplacian (low = blurry/out of focus)
    """
    if params is None:
        params = PreprocessingParams()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_gradable = (
        brightness >= params.quality_min_brightness
        and contrast >= params.quality_min_contrast
        and sharpness >= params.quality_min_laplacian_var
    )

    return QualityScore(
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        sharpness=round(sharpness, 2),
        is_gradable=is_gradable,
    )


# ── Full pipeline ──


def preprocess_image(
    image: np.ndarray,
    params: PreprocessingParams | None = None,
) -> tuple[np.ndarray, QualityScore]:
    """Run the full preprocessing pipeline on a single image.

    Returns the preprocessed image (uint8, BGR) and quality score.
    """
    if params is None:
        params = PreprocessingParams()

    # 1. Crop black borders
    cropped = crop_fundus_circle(image)

    # 2. Resize
    resized = resize_image(cropped, params.target_size)

    # 3. Quality score (before Ben Graham, which alters statistics)
    quality = score_image_quality(resized, params)

    # 4. Ben Graham preprocessing
    processed = ben_graham_preprocess(
        resized,
        sigma=params.ben_graham_sigma,
        weight=params.ben_graham_weight,
    )

    return processed, quality


def preprocess_dataset(
    catalog_path: Path,
    output_dir: Path,
    params: PreprocessingParams | None = None,
) -> pd.DataFrame:
    """Preprocess all images in a catalog CSV.

    Args:
        catalog_path: Path to unified catalog CSV.
        output_dir: Directory for preprocessed images.
        params: Preprocessing parameters.

    Returns:
        Updated catalog DataFrame with quality scores and processed image paths.
    """
    if params is None:
        params = PreprocessingParams()

    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(catalog_path)
    records = []

    for idx, row in catalog.iterrows():
        src_path = Path(row["image_path"])
        if not src_path.exists():
            logger.warning("Image not found: %s (skipping)", src_path)
            continue

        image = cv2.imread(str(src_path))
        if image is None:
            logger.warning("Failed to read: %s (skipping)", src_path)
            continue

        processed, quality = preprocess_image(image, params)

        # Save processed image
        dataset = row["dataset_source"]
        out_subdir = output_dir / dataset
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_name = f"{src_path.stem}.png"
        out_path = out_subdir / out_name
        cv2.imwrite(str(out_path), processed)

        records.append({
            "image_path": str(out_path),
            "original_path": str(src_path),
            "dr_grade": row["dr_grade"],
            "dataset_source": dataset,
            "original_filename": row["original_filename"],
            "quality_brightness": quality.brightness,
            "quality_contrast": quality.contrast,
            "quality_sharpness": quality.sharpness,
            "is_gradable": quality.is_gradable,
        })

        if (idx + 1) % 500 == 0:
            logger.info("Processed %d / %d images", idx + 1, len(catalog))

    df = pd.DataFrame(records)

    # Save updated catalog
    out_catalog = output_dir / "catalog_processed.csv"
    df.to_csv(out_catalog, index=False)

    # Save preprocessing params
    params_path = output_dir / "preprocessing_params.json"
    with open(params_path, "w") as f:
        json.dump(asdict(params), f, indent=2)

    total = len(df)
    gradable = df["is_gradable"].sum() if total > 0 else 0
    logger.info(
        "Preprocessing complete: %d images processed, %d gradable (%.1f%%)",
        total,
        gradable,
        100 * gradable / total if total > 0 else 0,
    )
    return df


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Preprocess DR screening images")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DATA_DIR / "catalog.csv",
        help="Path to unified catalog CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "processed",
        help="Output directory for processed images",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=TRAIN_IMAGE_SIZE,
        help=f"Target image size (default: {TRAIN_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--mobile",
        action="store_true",
        help=f"Use mobile image size ({MOBILE_IMAGE_SIZE}x{MOBILE_IMAGE_SIZE})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    size = MOBILE_IMAGE_SIZE if args.mobile else args.size
    params = PreprocessingParams(target_size=size)

    df = preprocess_dataset(args.catalog, args.output_dir, params)
    print(f"\nProcessed: {len(df)} images")
    if not df.empty:
        print(f"Gradable:  {df['is_gradable'].sum()} ({100 * df['is_gradable'].mean():.1f}%)")
        print(f"\nQuality stats:")
        for col in ["quality_brightness", "quality_contrast", "quality_sharpness"]:
            print(f"  {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}")


if __name__ == "__main__":
    main()
