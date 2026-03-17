"""Tests for DR screening download/catalog module."""

import pandas as pd
import pytest

from bioagentics.diagnostics.retinal_dr_screening.download import (
    CATALOG_FUNCTIONS,
    build_unified_catalog,
    catalog_aptos2019,
    catalog_eyepacs,
    catalog_idrid,
    catalog_messidor2,
    catalog_odir5k,
    save_catalog,
)


def test_catalog_functions_registry():
    assert len(CATALOG_FUNCTIONS) == 5
    for name in ["eyepacs", "aptos2019", "idrid", "messidor2", "odir5k"]:
        assert name in CATALOG_FUNCTIONS
        download_fn, catalog_fn = CATALOG_FUNCTIONS[name]
        assert callable(download_fn)
        assert callable(catalog_fn)


def test_catalog_eyepacs_no_data(tmp_path):
    """Catalog returns empty DataFrame when no data is present."""
    df = catalog_eyepacs(data_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_catalog_aptos2019_no_data(tmp_path):
    df = catalog_aptos2019(data_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_catalog_idrid_no_data(tmp_path):
    df = catalog_idrid(data_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_catalog_messidor2_no_data(tmp_path):
    df = catalog_messidor2(data_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_catalog_odir5k_no_data(tmp_path):
    df = catalog_odir5k(data_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_build_unified_catalog_empty():
    """Unified catalog returns proper columns even with no data."""
    df = build_unified_catalog(datasets=[])
    assert list(df.columns) == ["image_path", "dr_grade", "dataset_source", "original_filename"]


def test_catalog_eyepacs_with_mock_data(tmp_path):
    """Catalog correctly parses EyePACS-format data."""
    # Create mock label file
    labels = pd.DataFrame({"image": ["10_left", "10_right", "15_left"], "level": [0, 2, 4]})
    labels.to_csv(tmp_path / "trainLabels.csv", index=False)

    # Create mock image files in train/ subdirectory
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for name in ["10_left.jpeg", "10_right.jpeg", "15_left.jpeg"]:
        (train_dir / name).write_bytes(b"\xff\xd8\xff\xe0")  # JPEG magic bytes

    df = catalog_eyepacs(data_dir=tmp_path)
    assert len(df) == 3
    assert set(df["dr_grade"]) == {0, 2, 4}
    assert all(df["dataset_source"] == "eyepacs")


def test_catalog_aptos2019_with_mock_data(tmp_path):
    """Catalog correctly parses APTOS 2019-format data."""
    labels = pd.DataFrame({"id_code": ["abc123", "def456"], "diagnosis": [1, 3]})
    labels.to_csv(tmp_path / "train.csv", index=False)

    img_dir = tmp_path / "train_images"
    img_dir.mkdir()
    for name in ["abc123.png", "def456.png"]:
        (img_dir / name).write_bytes(b"\x89PNG")

    df = catalog_aptos2019(data_dir=tmp_path)
    assert len(df) == 2
    assert set(df["dr_grade"]) == {1, 3}
    assert all(df["dataset_source"] == "aptos2019")


def test_save_catalog(tmp_path):
    """Save and reload catalog round-trips correctly."""
    df = pd.DataFrame({
        "image_path": ["/a/b.jpg", "/c/d.jpg"],
        "dr_grade": [0, 3],
        "dataset_source": ["eyepacs", "aptos2019"],
        "original_filename": ["b.jpg", "d.jpg"],
    })
    out_path = tmp_path / "test_catalog.csv"
    save_catalog(df, out_path)

    loaded = pd.read_csv(out_path)
    assert len(loaded) == 2
    assert list(loaded.columns) == ["image_path", "dr_grade", "dataset_source", "original_filename"]


def test_build_unified_catalog_with_mock_data(tmp_path):
    """Unified catalog concatenates multiple datasets."""
    # Create mock APTOS data
    aptos_dir = tmp_path / "aptos"
    aptos_dir.mkdir()
    labels = pd.DataFrame({"id_code": ["img1"], "diagnosis": [2]})
    labels.to_csv(aptos_dir / "train.csv", index=False)
    img_dir = aptos_dir / "train_images"
    img_dir.mkdir()
    (img_dir / "img1.png").write_bytes(b"\x89PNG")

    # Patch the dict entry so build_unified_catalog uses our tmp dir
    original = CATALOG_FUNCTIONS["aptos2019"]
    CATALOG_FUNCTIONS["aptos2019"] = (
        original[0],
        lambda data_dir=None: catalog_aptos2019(data_dir=aptos_dir),
    )
    try:
        df = build_unified_catalog(datasets=["aptos2019"])
        assert len(df) == 1
        assert df.iloc[0]["dr_grade"] == 2
    finally:
        CATALOG_FUNCTIONS["aptos2019"] = original
