"""Tests for the stream-and-delete WSI processing pipeline."""

import json
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd

from bioagentics.models.pathology_msi.stream_pipeline import (
    _extract_features_streaming,
    download_wsi,
    process_single_slide,
    run_stream_pipeline,
)


class TestDownloadWsi:
    """Test GDC WSI download function."""

    def test_successful_download(self, tmp_path):
        """Verify streaming download writes file and returns True."""
        fake_content = b"fake-svs-content" * 100
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [fake_content]
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        out = tmp_path / "test.svs"
        with patch("bioagentics.models.pathology_msi.stream_pipeline.requests.get", return_value=mock_resp):
            result = download_wsi("fake-uuid", out, expected_size=len(fake_content))

        assert result is True
        assert out.exists()
        assert out.read_bytes() == fake_content

    def test_failed_download_cleans_up(self, tmp_path):
        """Verify failed download removes partial file."""
        import requests

        with patch(
            "bioagentics.models.pathology_msi.stream_pipeline.requests.get",
            side_effect=requests.ConnectionError("timeout"),
        ):
            out = tmp_path / "test.svs"
            result = download_wsi("fake-uuid", out)

        assert result is False
        assert not out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        """Verify parent directories are created."""
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"data"]
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        out = tmp_path / "sub" / "dir" / "test.svs"
        with patch("bioagentics.models.pathology_msi.stream_pipeline.requests.get", return_value=mock_resp):
            result = download_wsi("fake-uuid", out)

        assert result is True
        assert out.exists()


class TestExtractFeaturesStreaming:
    """Test streaming feature extraction."""

    def test_returns_correct_shape(self, tmp_path):
        """Verify feature matrix shape matches tiles x feature_dim."""
        tile_coords = pd.DataFrame({
            "x": [0, 256],
            "y": [0, 0],
            "tile_size": [256, 256],
            "level": [0, 0],
            "downsample": [1.0, 1.0],
            "level_downsample": [1.0, 1.0],
            "tissue_fraction": [0.9, 0.8],
        })

        # Mock openslide
        mock_slide = MagicMock()
        from PIL import Image
        fake_region = Image.new("RGBA", (256, 256), (150, 100, 80, 255))
        mock_slide.read_region.return_value = fake_region

        import bioagentics.models.pathology_msi.stream_pipeline as sp
        mock_openslide = MagicMock()
        mock_openslide.OpenSlide.return_value = mock_slide

        with patch.object(sp, "openslide", mock_openslide):
            features = _extract_features_streaming(
                tmp_path / "fake.svs",
                tile_coords,
                "resnet50",  # Use resnet50 since it doesn't need special weights
                batch_size=2,
            )

        assert features is not None
        assert features.shape == (2, 2048)  # 2 tiles, ResNet50 = 2048-d

    def test_empty_tiles_returns_none(self, tmp_path):
        """Verify empty tile coords produces None."""
        tile_coords = pd.DataFrame(
            columns=["x", "y", "tile_size", "level", "downsample", "level_downsample", "tissue_fraction"]
        )

        import bioagentics.models.pathology_msi.stream_pipeline as sp
        mock_openslide = MagicMock()

        with patch.object(sp, "openslide", mock_openslide):
            features = _extract_features_streaming(
                tmp_path / "fake.svs", tile_coords, "resnet50", batch_size=2
            )

        assert features is None


class TestProcessSingleSlide:
    """Test the combined tile + extract + save pipeline."""

    def test_saves_valid_h5(self, tmp_path):
        """Verify process_single_slide produces a valid HDF5 file."""
        slide_path = tmp_path / "slide.svs"
        output_h5 = tmp_path / "features" / "test.h5"

        mock_slide = MagicMock()
        mock_slide.close = MagicMock()

        from PIL import Image
        fake_region = Image.new("RGBA", (256, 256), (150, 100, 80, 255))
        mock_slide.read_region.return_value = fake_region

        # Fake tile coordinates (bypass actual tiling)
        fake_coords = pd.DataFrame({
            "x": [0, 512], "y": [0, 0],
            "tile_size": [256, 256], "level": [0, 0],
            "downsample": [2.0, 2.0], "level_downsample": [1.0, 1.0],
            "tissue_fraction": [0.9, 0.8],
        })

        import bioagentics.models.pathology_msi.stream_pipeline as sp
        mock_openslide = MagicMock()
        mock_openslide.OpenSlide.return_value = mock_slide

        with patch.object(sp, "openslide", mock_openslide), \
             patch.object(sp, "compute_tile_coordinates", return_value=fake_coords):
            result = process_single_slide(
                slide_path, output_h5, extractor_name="resnet50", batch_size=4
            )

        assert result is True
        assert output_h5.exists()

        with h5py.File(output_h5, "r") as f:
            assert "features" in f
            assert "coords" in f
            feats = f["features"]
            assert feats.shape[1] == 2048  # type: ignore[union-attr]
            assert f.attrs["extractor"] == "resnet50"
            assert f.attrs["tile_size"] == 256
            assert f.attrs["target_mag"] == 20.0


class TestRunStreamPipeline:
    """Test the main pipeline runner."""

    def test_skips_existing(self, tmp_path):
        """Verify skip_existing correctly skips processed slides."""
        manifest = [
            {
                "case_id": "c1",
                "file_uuid": "uuid-1",
                "patient_id": "TCGA-TEST-001",
                "msi_label": "MSI-H",
                "file_name": "test1.svs",
                "file_size_bytes": 1000,
            },
            {
                "case_id": "c2",
                "file_uuid": "uuid-2",
                "patient_id": "TCGA-TEST-002",
                "msi_label": "MSS",
                "file_name": "test2.svs",
                "file_size_bytes": 1000,
            },
        ]
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()

        # Create existing feature file for first slide
        existing_h5 = feature_dir / "TCGA-TEST-001.h5"
        with h5py.File(existing_h5, "w") as f:
            f.create_dataset("features", data=np.zeros((10, 512)))

        # Mock download to fail (should only be called for slide 2)
        with patch(
            "bioagentics.models.pathology_msi.stream_pipeline.download_wsi",
            return_value=False,
        ) as mock_dl:
            results = run_stream_pipeline(
                manifest_path, feature_dir, skip_existing=True
            )

        assert "TCGA-TEST-001" in results["skipped"]
        assert "TCGA-TEST-002" in results["failed"]
        # download only called once (for slide 2)
        assert mock_dl.call_count == 1

    def test_reports_download_failures(self, tmp_path):
        """Verify download failures are recorded."""
        manifest = [
            {
                "case_id": "c1",
                "file_uuid": "uuid-1",
                "patient_id": "TCGA-FAIL-001",
                "msi_label": "MSI-H",
                "file_name": "fail.svs",
                "file_size_bytes": 1000,
            }
        ]
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        with patch(
            "bioagentics.models.pathology_msi.stream_pipeline.download_wsi",
            return_value=False,
        ):
            results = run_stream_pipeline(
                manifest_path, tmp_path / "features", skip_existing=False
            )

        assert "TCGA-FAIL-001" in results["failed"]
        assert len(results["completed"]) == 0

    def test_no_skip_reprocesses(self, tmp_path):
        """Verify skip_existing=False reprocesses existing slides."""
        manifest = [
            {
                "case_id": "c1",
                "file_uuid": "uuid-1",
                "patient_id": "TCGA-REDO-001",
                "msi_label": "MSS",
                "file_name": "redo.svs",
                "file_size_bytes": 1000,
            }
        ]
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        existing_h5 = feature_dir / "TCGA-REDO-001.h5"
        with h5py.File(existing_h5, "w") as f:
            f.create_dataset("features", data=np.zeros((5, 512)))

        with patch(
            "bioagentics.models.pathology_msi.stream_pipeline.download_wsi",
            return_value=False,
        ) as mock_dl:
            results = run_stream_pipeline(
                manifest_path, feature_dir, skip_existing=False
            )

        # Should attempt download even though features exist
        assert mock_dl.call_count == 1
        assert len(results["skipped"]) == 0
