"""Stream-and-delete WSI processing pipeline.

Downloads one WSI at a time from GDC, tiles it, extracts features, saves to
HDF5, then deletes the WSI to reclaim disk. Designed for 8GB RAM systems with
limited disk space.

Usage:
    uv run python -m bioagentics.models.pathology_msi.stream_pipeline \
        --manifest data/diagnostics/pathology-fm-msi-prescreening/pilot_cohort_manifest.json \
        --feature-dir data/diagnostics/pathology-fm-msi-prescreening/features/conch \
        --extractor conch
"""

import gc
import json
import logging
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import torch

from bioagentics.models.pathology_msi.feature_extraction import (
    create_feature_extractor,
)
from bioagentics.models.pathology_msi.gdc_client import GDC_DATA_ENDPOINT
from bioagentics.models.pathology_msi.tiling import compute_tile_coordinates

try:
    import openslide
except ImportError:
    openslide = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Streaming download chunk size (8 MB)
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024

# Feature extraction batch size — conservative for 8GB RAM
DEFAULT_BATCH_SIZE = 16


def download_wsi(file_uuid: str, output_path: Path, expected_size: int | None = None) -> bool:
    """Stream-download a single WSI from GDC by file UUID.

    Args:
        file_uuid: GDC file UUID.
        output_path: Path to save the downloaded file.
        expected_size: Expected file size in bytes (for progress logging).

    Returns:
        True if download succeeded.
    """
    url = f"{GDC_DATA_ENDPOINT}/{file_uuid}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Downloading {file_uuid} -> {output_path.name}"
        + (f" ({expected_size / 1e6:.0f} MB)" if expected_size else "")
    )

    try:
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            downloaded = 0
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)

            logger.info(f"  Downloaded {downloaded / 1e6:.0f} MB")
            return True

    except requests.RequestException as e:
        logger.error(f"  Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def _extract_features_streaming(
    slide_path: Path,
    tile_coords: pd.DataFrame,
    extractor_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray | None:
    """Extract features from tiles in small batches to limit memory.

    Reads tiles directly from the slide and processes in batches,
    accumulating feature vectors without holding all patches in memory.

    Args:
        slide_path: Path to the WSI file.
        tile_coords: DataFrame with tile coordinates from tiling module.
        extractor_name: Name of the feature extractor ('conch', 'uni2', 'resnet50').
        batch_size: Number of patches per batch.

    Returns:
        Feature matrix (n_patches, feature_dim) or None on failure.
    """
    if openslide is None:
        logger.error("openslide-python required: uv add --optional research openslide-python")
        return None

    extractor = create_feature_extractor(extractor_name, device="cpu", batch_size=batch_size)
    model = extractor.get_model().eval()
    transform = extractor.get_transform()

    slide = openslide.OpenSlide(str(slide_path))
    all_features = []

    try:
        n_tiles = len(tile_coords)
        for batch_start in range(0, n_tiles, batch_size):
            batch_end = min(batch_start + batch_size, n_tiles)
            batch_tiles = tile_coords.iloc[batch_start:batch_end]

            # Read and transform patches for this batch
            patches = []
            for _, row in batch_tiles.iterrows():
                x, y = int(row["x"]), int(row["y"])
                level = int(row["level"])
                downsample = float(row["downsample"])
                level_downsample = float(row["level_downsample"])
                tile_size = int(row["tile_size"])

                read_size = int(tile_size * downsample / level_downsample)
                region = slide.read_region((x, y), level, (read_size, read_size))
                region = region.convert("RGB")
                if region.size[0] != tile_size or region.size[1] != tile_size:
                    from PIL import Image
                    region = region.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

                patches.append(transform(region))

            batch_tensor = torch.stack(patches)

            with torch.no_grad():
                features = model(batch_tensor)
                if isinstance(features, tuple):
                    features = features[0]
                all_features.append(features.cpu().numpy())

            # Free batch memory
            del patches, batch_tensor, features
            gc.collect()

            if (batch_start // batch_size) % 10 == 0:
                logger.info(
                    f"  Extracted {min(batch_end, n_tiles)}/{n_tiles} patches"
                )

    finally:
        slide.close()

    if not all_features:
        return None

    return np.concatenate(all_features, axis=0)


def process_single_slide(
    slide_path: Path,
    output_h5_path: Path,
    extractor_name: str = "conch",
    batch_size: int = DEFAULT_BATCH_SIZE,
    tile_size: int = 256,
    target_mag: float = 20.0,
    tissue_threshold: float = 0.5,
) -> bool:
    """Tile a WSI and extract features, saving to HDF5.

    Args:
        slide_path: Path to the downloaded WSI.
        output_h5_path: Path for output HDF5 feature file.
        extractor_name: Feature extractor name.
        batch_size: Batch size for feature extraction.
        tile_size: Tile size in pixels.
        target_mag: Target magnification.
        tissue_threshold: Minimum tissue fraction for tiles.

    Returns:
        True if processing succeeded.
    """
    if openslide is None:
        logger.error("openslide-python required")
        return False

    try:
        slide = openslide.OpenSlide(str(slide_path))
    except Exception as e:
        logger.error(f"Failed to open slide {slide_path}: {e}")
        return False

    try:
        logger.info(f"Tiling {slide_path.name} (dims={slide.dimensions})")
        tile_coords = compute_tile_coordinates(
            slide,
            tile_size=tile_size,
            target_mag=target_mag,
            tissue_threshold=tissue_threshold,
        )
    finally:
        slide.close()

    if tile_coords.empty:
        logger.warning(f"No tissue tiles found for {slide_path.name}")
        return False

    logger.info(f"Extracting {extractor_name} features for {len(tile_coords)} tiles")
    features = _extract_features_streaming(
        slide_path, tile_coords, extractor_name, batch_size=batch_size
    )

    if features is None:
        logger.error(f"Feature extraction failed for {slide_path.name}")
        return False

    # Save to HDF5
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5_path, "w") as f:
        f.create_dataset("features", data=features, compression="gzip")
        f.create_dataset(
            "coords",
            data=tile_coords[["x", "y"]].values,
            compression="gzip",
        )
        f.attrs["extractor"] = extractor_name
        f.attrs["feature_dim"] = features.shape[1]
        f.attrs["n_patches"] = features.shape[0]
        f.attrs["slide_path"] = str(slide_path)
        f.attrs["tile_size"] = tile_size
        f.attrs["target_mag"] = target_mag

    logger.info(
        f"  Saved {features.shape[0]} x {features.shape[1]} features -> {output_h5_path}"
    )

    # Free memory
    del features, tile_coords
    gc.collect()

    return True


def run_stream_pipeline(
    manifest_path: str | Path,
    feature_dir: str | Path,
    extractor_name: str = "conch",
    batch_size: int = DEFAULT_BATCH_SIZE,
    tmp_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> dict:
    """Run the stream-and-delete pipeline over a slide manifest.

    For each slide in the manifest:
    1. Download WSI from GDC
    2. Tile and extract features
    3. Save features to HDF5
    4. Delete WSI to reclaim disk

    Args:
        manifest_path: Path to JSON manifest (list of slide entries with
            file_uuid, patient_id, file_name, file_size_bytes).
        feature_dir: Directory to save HDF5 feature files.
        extractor_name: Feature extractor name ('conch', 'uni2', 'resnet50').
        batch_size: Batch size for feature extraction.
        tmp_dir: Temporary directory for WSI downloads. Defaults to system temp.
        skip_existing: Skip slides that already have feature files.

    Returns:
        Dict with keys: completed, failed, skipped (lists of patient_ids).
    """
    manifest_path = Path(manifest_path)
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    results = {"completed": [], "failed": [], "skipped": []}
    total = len(manifest)

    for i, entry in enumerate(manifest):
        patient_id = entry["patient_id"]
        file_uuid = entry["file_uuid"]
        file_name = entry["file_name"]
        file_size = entry.get("file_size_bytes")

        output_h5 = feature_dir / f"{patient_id}.h5"

        logger.info(f"\n[{i + 1}/{total}] {patient_id} ({entry.get('msi_label', '?')})")

        # Skip if already processed
        if skip_existing and output_h5.exists():
            logger.info(f"  Skipping (features exist): {output_h5}")
            results["skipped"].append(patient_id)
            continue

        # Use a temp directory for WSI download
        with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
            wsi_path = Path(td) / file_name

            # Step 1: Download
            if not download_wsi(file_uuid, wsi_path, expected_size=file_size):
                results["failed"].append(patient_id)
                continue

            # Step 2 & 3: Tile + extract + save
            success = process_single_slide(
                wsi_path, output_h5, extractor_name=extractor_name, batch_size=batch_size
            )

            if success:
                results["completed"].append(patient_id)
            else:
                results["failed"].append(patient_id)

            # Step 4: WSI is deleted automatically when TemporaryDirectory exits

        # Force garbage collection between slides
        gc.collect()

    logger.info(
        f"\nPipeline complete: {len(results['completed'])} completed, "
        f"{len(results['failed'])} failed, {len(results['skipped'])} skipped"
    )
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Stream-and-delete WSI processing pipeline")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSON slide manifest",
    )
    parser.add_argument(
        "--feature-dir",
        required=True,
        help="Output directory for HDF5 feature files",
    )
    parser.add_argument(
        "--extractor",
        default="conch",
        choices=["conch", "uni2", "resnet50"],
        help="Feature extractor (default: conch)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for feature extraction (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-process slides that already have feature files",
    )
    args = parser.parse_args()

    results = run_stream_pipeline(
        manifest_path=args.manifest,
        feature_dir=args.feature_dir,
        extractor_name=args.extractor,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip,
    )

    print(f"\nResults: {json.dumps(results, indent=2)}")
