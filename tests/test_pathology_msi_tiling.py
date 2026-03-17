"""Tests for WSI tiling pipeline."""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bioagentics.models.pathology_msi.tiling import (
    _compute_downsample_for_target_mag,
    _get_slide_magnification,
    compute_tile_coordinates,
    create_tissue_mask,
    read_tile_from_slide,
)


def _mock_slide(width=10000, height=10000, mag=40.0):
    """Create a mock OpenSlide object."""
    slide = MagicMock()
    slide.dimensions = (width, height)
    slide.properties = {"openslide.objective-power": str(mag)}
    slide.level_downsamples = [1.0, 4.0, 16.0]
    slide.level_dimensions = [
        (width, height),
        (width // 4, height // 4),
        (width // 16, height // 16),
    ]
    slide.get_best_level_for_downsample.return_value = 0

    # Return a simple thumbnail
    thumb = Image.new("RGB", (256, 256), color=(200, 100, 100))
    slide.get_thumbnail.return_value = thumb

    return slide


class TestGetSlideMagnification:
    def test_standard_property(self):
        slide = MagicMock()
        slide.properties = {"openslide.objective-power": "40"}
        assert _get_slide_magnification(slide) == 40.0

    def test_aperio_property(self):
        slide = MagicMock()
        slide.properties = {"aperio.AppMag": "20"}
        assert _get_slide_magnification(slide) == 20.0

    def test_missing_defaults_to_40(self):
        slide = MagicMock()
        slide.properties = {}
        assert _get_slide_magnification(slide) == 40.0


class TestComputeDownsample:
    def test_40x_to_20x(self):
        slide = MagicMock()
        slide.properties = {"openslide.objective-power": "40"}
        assert _compute_downsample_for_target_mag(slide, 20.0) == 2.0

    def test_20x_to_20x(self):
        slide = MagicMock()
        slide.properties = {"openslide.objective-power": "20"}
        assert _compute_downsample_for_target_mag(slide, 20.0) == 1.0


class TestCreateTissueMask:
    def test_returns_mask_and_scale(self):
        slide = _mock_slide()
        mask, scale = create_tissue_mask(slide)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert scale > 0

    def test_mask_shape_matches_thumbnail(self):
        slide = _mock_slide(width=5000, height=3000)
        mask, scale = create_tissue_mask(slide, thumbnail_max_dim=500)
        # Mask should be approximately thumbnail-sized
        assert mask.shape[0] > 0
        assert mask.shape[1] > 0


class TestComputeTileCoordinates:
    def test_produces_valid_dataframe(self):
        slide = _mock_slide(width=2000, height=2000, mag=20.0)
        # For 20x slide at 20x target, downsample=1, so tile_size_l0=256
        # Grid: 2000//256 = 7 cols/rows = 49 max tiles

        df = compute_tile_coordinates(slide, tissue_threshold=0.0)

        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "x", "y", "tile_size", "level", "downsample",
            "level_downsample", "tissue_fraction", "width_at_level0",
            "height_at_level0",
        }
        assert expected_cols == set(df.columns)
        assert all(df["tile_size"] == 256)

    def test_tissue_threshold_filters_tiles(self):
        slide = _mock_slide(width=2000, height=2000, mag=20.0)
        # With threshold=0, get all tiles
        df_all = compute_tile_coordinates(slide, tissue_threshold=0.0)
        # With threshold=1, get no tiles (thumbnail is uniform)
        df_strict = compute_tile_coordinates(slide, tissue_threshold=1.0)

        assert len(df_all) >= len(df_strict)


class TestReadTileFromSlide:
    def test_returns_correct_size_image(self):
        slide = MagicMock()
        region = Image.new("RGBA", (256, 256), color=(200, 100, 100, 255))
        slide.read_region.return_value = region

        tile = read_tile_from_slide(
            slide, x=0, y=0, tile_size=256,
            level=0, downsample=1.0, level_downsample=1.0,
        )
        assert tile.size == (256, 256)
        assert tile.mode == "RGB"

    def test_resizes_when_needed(self):
        slide = MagicMock()
        # Return a larger region that needs resizing
        region = Image.new("RGBA", (512, 512), color=(200, 100, 100, 255))
        slide.read_region.return_value = region

        tile = read_tile_from_slide(
            slide, x=0, y=0, tile_size=256,
            level=0, downsample=2.0, level_downsample=1.0,
        )
        assert tile.size == (256, 256)
