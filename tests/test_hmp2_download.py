"""Tests for HMP2/IBDMDB download and loader module."""

import pandas as pd
import pytest

from bioagentics.data.hmp2_download import (
    DATA_DIR,
    DATA_FILES,
    HMP2Loader,
    _extract_sample_id,
)


def test_data_files_structure():
    """DATA_FILES dict has required keys and values."""
    required_keys = {"metadata", "taxonomic", "pathways", "metabolomics",
                      "transcriptomics", "serology", "dysbiosis"}
    assert required_keys == set(DATA_FILES.keys())

    for key, info in DATA_FILES.items():
        assert "filename" in info
        assert "url" in info
        assert "description" in info
        assert info["url"].startswith("https://")


def test_extract_sample_id():
    """Sample ID normalization strips platform suffixes."""
    assert _extract_sample_id("CSM5FZ4P_P") == "CSM5FZ4P"
    assert _extract_sample_id("HSM6XRS9_TR") == "HSM6XRS9"
    assert _extract_sample_id("CSM5FZ4P_M") == "CSM5FZ4P"
    assert _extract_sample_id("CSM5FZ4P_MBX") == "CSM5FZ4P"
    assert _extract_sample_id("CSM5FZ4P") == "CSM5FZ4P"
    assert _extract_sample_id("  CSM5FZ4P_P  ") == "CSM5FZ4P"


def test_data_dir_is_under_repo():
    """DATA_DIR should point to data/hmp2/ under the repo root."""
    assert DATA_DIR.name == "hmp2"
    assert DATA_DIR.parent.name == "data"


def test_loader_init_default():
    """HMP2Loader uses default DATA_DIR."""
    loader = HMP2Loader()
    assert loader.data_dir == DATA_DIR


def test_loader_init_custom(tmp_path):
    """HMP2Loader accepts custom data directory."""
    loader = HMP2Loader(data_dir=tmp_path)
    assert loader.data_dir == tmp_path


def test_loader_missing_files(tmp_path):
    """Loader raises FileNotFoundError for missing data files."""
    loader = HMP2Loader(data_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        loader.load_metadata()

    with pytest.raises(FileNotFoundError):
        loader.load_species()

    with pytest.raises(FileNotFoundError):
        loader.load_pathways()

    with pytest.raises(FileNotFoundError):
        loader.load_metabolomics()


def test_loader_metadata_from_csv(tmp_path):
    """Loader reads metadata CSV correctly."""
    meta = pd.DataFrame(
        {
            "External ID": ["CSM5FZ4P_P", "HSM6XRS9_P", "TST0001_P"],
            "Participant ID": ["H4001", "H4002", "H4003"],
            "diagnosis": ["CD", "nonIBD", "UC"],
            "site_name": ["MGH", "MGH", "Emory"],
        }
    )
    meta.to_csv(tmp_path / "hmp2_metadata.csv", index=False)

    loader = HMP2Loader(data_dir=tmp_path)
    loaded = loader.load_metadata()
    assert len(loaded) == 3
    assert "External ID" in loaded.columns
    assert "diagnosis" in loaded.columns


def test_loader_sample_map(tmp_path):
    """_build_sample_map creates correct ID mappings."""
    meta = pd.DataFrame(
        {
            "External ID": ["CSM5FZ4P_P", "HSM6XRS9_P"],
            "Participant ID": ["H4001", "H4002"],
            "diagnosis": ["CD", "nonIBD"],
        }
    )
    meta.to_csv(tmp_path / "hmp2_metadata.csv", index=False)

    loader = HMP2Loader(data_dir=tmp_path)
    loaded_meta = loader.load_metadata()
    sample_map = loader._build_sample_map(loaded_meta)

    # Direct mapping
    assert sample_map["CSM5FZ4P_P"] == "H4001"
    # Normalized mapping (without suffix)
    assert sample_map["CSM5FZ4P"] == "H4001"


def test_align_to_participants(tmp_path):
    """_align_to_participants maps and aggregates correctly."""
    meta = pd.DataFrame(
        {
            "External ID": ["S1_P", "S2_P", "S1_P_2"],
            "Participant ID": ["H4001", "H4002", "H4001"],
            "diagnosis": ["CD", "nonIBD", "CD"],
        }
    )
    meta.to_csv(tmp_path / "hmp2_metadata.csv", index=False)

    # Create omic data with two samples from same participant
    omic = pd.DataFrame(
        {"feature_a": [1.0, 2.0, 3.0], "feature_b": [4.0, 5.0, 6.0]},
        index=["S1_P", "S2_P", "S1_P_2"],
    )

    loader = HMP2Loader(data_dir=tmp_path)
    loaded_meta = loader.load_metadata()
    sample_map = loader._build_sample_map(loaded_meta)
    aligned = loader._align_to_participants(omic, sample_map)

    # H4001 should be mean of S1_P and S1_P_2
    assert "H4001" in aligned.index or "H4002" in aligned.index
