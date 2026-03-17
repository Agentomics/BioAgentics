"""Tests for MIMIC-CXR dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bioagentics.cxr_rare.config import LABEL_NAMES
from bioagentics.cxr_rare.datasets.mimic_cxr import (
    MIMICCXRDataset,
    _load_image,
)


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """Create a mock MIMIC-CXR data directory with sample images and CSVs."""
    data_dir = tmp_path / "mimic-cxr"
    files_dir = data_dir / "files"
    files_dir.mkdir(parents=True)

    # Create mock images (flat directory for simplicity)
    dicom_ids = [f"d{i:04d}" for i in range(5)]
    for did in dicom_ids:
        img = Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8), mode="L")
        img.save(files_dir / f"{did}.png")

    # Create labels CSV
    rows = []
    for i, did in enumerate(dicom_ids):
        row = {
            "dicom_id": did,
            "subject_id": f"1000{i}",
            "study_id": f"5000{i}",
        }
        for lbl in LABEL_NAMES:
            row[lbl] = 0
        # Assign some labels
        row[LABEL_NAMES[0]] = 1  # "No Finding" for first
        if i >= 2:
            row[LABEL_NAMES[1]] = 1  # "Support Devices"
        if i == 4:
            row[LABEL_NAMES[-1]] = 1  # tail class
        rows.append(row)
    labels_df = pd.DataFrame(rows)
    labels_df.to_csv(data_dir / "cxr-lt-2026-labels.csv", index=False)

    # Create split CSV
    split_rows = []
    for i, did in enumerate(dicom_ids):
        split = "train" if i < 3 else ("validate" if i == 3 else "test")
        split_rows.append({"dicom_id": did, "split": split})
    splits_df = pd.DataFrame(split_rows)
    splits_df.to_csv(data_dir / "mimic-cxr-2.0.0-split.csv", index=False)

    return data_dir


def test_dataset_loads_train_split(mock_data_dir: Path) -> None:
    ds = MIMICCXRDataset(data_dir=mock_data_dir, split="train")
    assert len(ds) == 3


def test_dataset_loads_val_split(mock_data_dir: Path) -> None:
    ds = MIMICCXRDataset(data_dir=mock_data_dir, split="validate")
    assert len(ds) == 1


def test_dataset_loads_test_split(mock_data_dir: Path) -> None:
    ds = MIMICCXRDataset(data_dir=mock_data_dir, split="test")
    assert len(ds) == 1


def test_getitem_returns_correct_shape(mock_data_dir: Path) -> None:
    ds = MIMICCXRDataset(data_dir=mock_data_dir, split="train")
    sample = ds[0]
    assert "image" in sample
    assert "labels" in sample
    assert sample["image"].shape == (3, 224, 224)
    assert sample["labels"].shape == (len(LABEL_NAMES),)


def test_labels_are_binary(mock_data_dir: Path) -> None:
    ds = MIMICCXRDataset(data_dir=mock_data_dir, split="train")
    sample = ds[0]
    labels = sample["labels"].numpy()
    assert set(np.unique(labels)).issubset({0.0, 1.0})


def test_class_counts(mock_data_dir: Path) -> None:
    ds = MIMICCXRDataset(data_dir=mock_data_dir, split="train")
    counts = ds.get_class_counts()
    assert counts["No Finding"] == 3  # all train samples
    assert counts["Support Devices"] == 1  # only d0002


def test_empty_dataset_when_no_files(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = MIMICCXRDataset(data_dir=empty_dir, split="train")
    assert len(ds) == 0
    assert ds.get_class_counts() == {name: 0 for name in LABEL_NAMES}


def test_load_image_grayscale_to_rgb(tmp_path: Path) -> None:
    img = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
    path = tmp_path / "test.png"
    img.save(path)
    loaded = _load_image(path)
    assert loaded.mode == "RGB"
    assert loaded.size == (32, 32)
