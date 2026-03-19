"""Tests for voice_pd data loader modules.

Tests loader initialization, manifest format, and availability checks
using temporary directories with synthetic test files.
"""

import csv

import pytest


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a minimal dataset directory with dummy files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    # Create dummy wav files (not real audio, just for path discovery)
    for i in range(3):
        (audio_dir / f"subj{i:02d}_vowel_a.wav").write_bytes(b"\x00" * 100)
    return tmp_path


@pytest.fixture
def uci_tele_dataset(tmp_path):
    """Create a minimal UCI Telemonitoring data file."""
    header = [
        "subject#", "age", "sex", "test_time",
        "motor_UPDRS", "total_UPDRS",
        "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
        "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
    ]
    rows = [
        ["1", "72", "0", "5.6", "28.2", "34.4",
         "0.00662", "0.00005", "0.00317", "0.00358", "0.00952",
         "0.02841", "0.258", "0.01462", "0.01693", "0.02264",
         "0.04386", "0.01125", "25.211", "0.4857", "0.7238", "0.1312"],
    ]
    data_file = tmp_path / "parkinsons_updrs.data"
    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return tmp_path


class TestBaseLoader:
    def test_is_available_returns_false_for_missing_dir(self):
        from bioagentics.voice_pd.data_loaders.base import BaseDatasetLoader

        class DummyLoader(BaseDatasetLoader):
            name = "dummy"
            def load_manifest(self):
                return []

        loader = DummyLoader("/nonexistent/path")
        assert not loader.is_available()

    def test_validate_returns_summary(self):
        from bioagentics.voice_pd.data_loaders.base import BaseDatasetLoader

        class DummyLoader(BaseDatasetLoader):
            name = "dummy"
            def load_manifest(self):
                return [
                    {"pd_status": "pd"},
                    {"pd_status": "healthy"},
                    {"pd_status": "healthy"},
                ]

        loader = DummyLoader("/nonexistent/path")
        result = loader.validate()
        assert result["available"] is False
        assert result["name"] == "dummy"


class TestUCISpeechLoader:
    def test_load_from_audio(self, tmp_dataset):
        from bioagentics.voice_pd.data_loaders.uci_speech import UCISpeechLoader

        loader = UCISpeechLoader(tmp_dataset)
        manifest = loader.load_manifest()
        assert len(manifest) == 3
        for row in manifest:
            assert "recording_id" in row
            assert "audio_path" in row
            assert "dataset" in row
            assert row["dataset"] == "uci_speech"

    def test_is_available(self, tmp_dataset):
        from bioagentics.voice_pd.data_loaders.uci_speech import UCISpeechLoader

        loader = UCISpeechLoader(tmp_dataset)
        assert loader.is_available()


class TestUCITelemonitoringLoader:
    def test_load_manifest(self, uci_tele_dataset):
        from bioagentics.voice_pd.data_loaders.uci_telemonitoring import (
            UCITelemonitoringLoader,
        )

        loader = UCITelemonitoringLoader(uci_tele_dataset)
        manifest = loader.load_manifest()
        assert len(manifest) == 1
        row = manifest[0]
        assert row["pd_status"] == "pd"
        assert row["dataset"] == "uci_telemonitoring"
        assert row["preextracted"] is True
        assert isinstance(row["motor_updrs"], float)
        assert isinstance(row["total_updrs"], float)


class TestPCGITALoader:
    def test_load_from_audio(self, tmp_dataset):
        from bioagentics.voice_pd.data_loaders.pcgita import PCGITALoader

        loader = PCGITALoader(tmp_dataset)
        manifest = loader.load_manifest()
        assert len(manifest) == 3
        for row in manifest:
            assert row["language"] == "es"
            assert row["dataset"] == "pcgita"


class TestItalianLoader:
    def test_load_from_audio(self, tmp_dataset):
        from bioagentics.voice_pd.data_loaders.italian import ItalianLoader

        loader = ItalianLoader(tmp_dataset)
        manifest = loader.load_manifest()
        assert len(manifest) == 3
        for row in manifest:
            assert row["language"] == "it"
            assert row["dataset"] == "italian"


class TestGetLoader:
    def test_unknown_dataset_raises(self):
        from bioagentics.voice_pd.data_loaders import get_loader

        with pytest.raises(ValueError, match="Unknown dataset"):
            get_loader("nonexistent")
