"""WSI tiling pipeline with background/artifact filtering.

Extracts 256x256 patches at 20x magnification from whole-slide images.
Uses tissue detection via Otsu thresholding on HSV saturation to filter
background and artifact tiles. Outputs per-slide tile coordinate CSVs.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# Default tiling parameters
TILE_SIZE = 256  # pixels at target magnification
TARGET_MAG = 20.0  # 20x magnification
TISSUE_THRESHOLD = 0.5  # minimum fraction of tile that must be tissue
THUMBNAIL_MAX_DIM = 2048  # max dimension for thumbnail used in tissue detection


def _get_slide_magnification(slide) -> float:
    """Extract objective magnification from slide properties."""
    props = slide.properties
    # Try standard openslide property
    mag = props.get("openslide.objective-power")
    if mag is not None:
        return float(mag)
    # Try aperio-specific
    mag = props.get("aperio.AppMag")
    if mag is not None:
        return float(mag)
    logger.warning("Could not determine slide magnification, assuming 40x")
    return 40.0


def _compute_downsample_for_target_mag(slide, target_mag: float) -> float:
    """Compute downsample factor to achieve target magnification."""
    native_mag = _get_slide_magnification(slide)
    return native_mag / target_mag


def create_tissue_mask(
    slide,
    thumbnail_max_dim: int = THUMBNAIL_MAX_DIM,
) -> tuple[np.ndarray, float]:
    """Create a binary tissue mask from a WSI thumbnail using Otsu thresholding.

    Args:
        slide: OpenSlide slide object.
        thumbnail_max_dim: Maximum dimension for the thumbnail.

    Returns:
        (tissue_mask, downsample_factor) where tissue_mask is a boolean array
        and downsample_factor maps thumbnail coords back to level-0 coords.
    """
    # Get thumbnail at manageable resolution
    dims = slide.dimensions  # (width, height) at level 0
    scale = max(dims[0], dims[1]) / thumbnail_max_dim
    thumb_size = (int(dims[0] / scale), int(dims[1] / scale))
    thumbnail = slide.get_thumbnail(thumb_size)
    thumbnail = np.array(thumbnail.convert("RGB"))

    # Convert to HSV and use saturation channel for tissue detection
    from skimage.color import rgb2hsv
    from skimage.filters import threshold_otsu

    hsv = rgb2hsv(thumbnail)
    saturation = hsv[:, :, 1]

    # Otsu threshold on saturation
    try:
        thresh = threshold_otsu(saturation)
    except ValueError:
        # If image is uniform, use a default threshold
        thresh = 0.05

    tissue_mask = saturation > thresh

    # Optional: apply morphological operations to clean up mask
    from scipy.ndimage import binary_closing, binary_opening

    tissue_mask = binary_closing(tissue_mask, iterations=3)
    tissue_mask = binary_opening(tissue_mask, iterations=3)

    return tissue_mask, scale


def compute_tile_coordinates(
    slide,
    tile_size: int = TILE_SIZE,
    target_mag: float = TARGET_MAG,
    tissue_threshold: float = TISSUE_THRESHOLD,
    thumbnail_max_dim: int = THUMBNAIL_MAX_DIM,
) -> pd.DataFrame:
    """Compute tile coordinates for a WSI, filtering by tissue content.

    Args:
        slide: OpenSlide slide object.
        tile_size: Tile size in pixels at target magnification.
        target_mag: Target magnification (default 20x).
        tissue_threshold: Minimum fraction of tile that must be tissue.
        thumbnail_max_dim: Max dimension for tissue mask thumbnail.

    Returns:
        DataFrame with columns: x, y, tile_size, level, downsample,
        tissue_fraction, width_at_level0, height_at_level0.
    """
    dims = slide.dimensions  # level 0 dimensions
    downsample = _compute_downsample_for_target_mag(slide, target_mag)

    # Tile size at level 0
    tile_size_l0 = int(tile_size * downsample)

    # Find best level for reading
    best_level = slide.get_best_level_for_downsample(downsample)
    level_downsample = slide.level_downsamples[best_level]

    # Create tissue mask
    tissue_mask, mask_scale = create_tissue_mask(slide, thumbnail_max_dim)
    mask_h, mask_w = tissue_mask.shape

    # Generate tile grid
    n_cols = dims[0] // tile_size_l0
    n_rows = dims[1] // tile_size_l0

    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * tile_size_l0
            y = row * tile_size_l0

            # Check tissue fraction in mask coordinates
            mx_start = int(x / mask_scale)
            my_start = int(y / mask_scale)
            mx_end = min(int((x + tile_size_l0) / mask_scale), mask_w)
            my_end = min(int((y + tile_size_l0) / mask_scale), mask_h)

            if mx_end <= mx_start or my_end <= my_start:
                continue

            tile_mask = tissue_mask[my_start:my_end, mx_start:mx_end]
            tissue_fraction = tile_mask.mean()

            if tissue_fraction >= tissue_threshold:
                tiles.append(
                    {
                        "x": x,
                        "y": y,
                        "tile_size": tile_size,
                        "level": best_level,
                        "downsample": downsample,
                        "level_downsample": level_downsample,
                        "tissue_fraction": round(tissue_fraction, 3),
                        "width_at_level0": tile_size_l0,
                        "height_at_level0": tile_size_l0,
                    }
                )

    df = pd.DataFrame(tiles)
    logger.info(
        f"  {len(df)}/{n_rows * n_cols} tiles pass tissue threshold "
        f"({tissue_threshold:.0%})"
    )
    return df


def tile_slide(
    slide_path: str | Path,
    output_dir: str | Path,
    tile_size: int = TILE_SIZE,
    target_mag: float = TARGET_MAG,
    tissue_threshold: float = TISSUE_THRESHOLD,
) -> Path | None:
    """Tile a single WSI and save coordinates to CSV.

    Args:
        slide_path: Path to the WSI file (.svs, .ndpi, .tiff, etc.)
        output_dir: Directory to save tile coordinate CSV.
        tile_size: Tile size in pixels at target magnification.
        target_mag: Target magnification.
        tissue_threshold: Minimum tissue fraction for tiles.

    Returns:
        Path to the saved CSV, or None if tiling failed.
    """
    slide_path = Path(slide_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import openslide
    except ImportError:
        logger.error("openslide-python is required for WSI tiling: uv add openslide-python")
        return None

    try:
        slide = openslide.OpenSlide(str(slide_path))
    except Exception as e:
        logger.error(f"Failed to open slide {slide_path}: {e}")
        return None

    try:
        logger.info(f"Tiling {slide_path.name} (dims={slide.dimensions})")
        coords = compute_tile_coordinates(
            slide,
            tile_size=tile_size,
            target_mag=target_mag,
            tissue_threshold=tissue_threshold,
        )

        if coords.empty:
            logger.warning(f"No tiles found for {slide_path.name}")
            return None

        # Save coordinates
        csv_name = slide_path.stem + "_tiles.csv"
        csv_path = output_dir / csv_name
        coords.to_csv(csv_path, index=False)
        logger.info(f"  Saved {len(coords)} tile coordinates to {csv_path}")
        return csv_path

    finally:
        slide.close()


def tile_batch(
    slide_paths: list[str | Path],
    output_dir: str | Path,
    tile_size: int = TILE_SIZE,
    target_mag: float = TARGET_MAG,
    tissue_threshold: float = TISSUE_THRESHOLD,
) -> dict[str, Path | None]:
    """Tile a batch of WSIs.

    Returns:
        Dict mapping slide filename to output CSV path (or None if failed).
    """
    results = {}
    for i, path in enumerate(slide_paths):
        path = Path(path)
        logger.info(f"[{i + 1}/{len(slide_paths)}] Processing {path.name}")
        result = tile_slide(
            path,
            output_dir,
            tile_size=tile_size,
            target_mag=target_mag,
            tissue_threshold=tissue_threshold,
        )
        results[path.name] = result
    return results


def read_tile_from_slide(
    slide,
    x: int,
    y: int,
    tile_size: int,
    level: int,
    downsample: float,
    level_downsample: float,
) -> Image.Image:
    """Read a single tile from an OpenSlide object.

    Args:
        slide: OpenSlide slide object.
        x, y: Top-left corner at level 0.
        tile_size: Desired output tile size.
        level: OpenSlide level to read from.
        downsample: Target downsample factor.
        level_downsample: Actual downsample of the chosen level.

    Returns:
        PIL Image of the tile at the target size.
    """
    # Read at the best level
    read_size = int(tile_size * downsample / level_downsample)
    region = slide.read_region((x, y), level, (read_size, read_size))
    region = region.convert("RGB")

    # Resize to target tile size if needed
    if region.size[0] != tile_size or region.size[1] != tile_size:
        region = region.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

    return region
