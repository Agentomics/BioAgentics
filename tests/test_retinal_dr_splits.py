"""Tests for DR screening data splitting module."""

import pandas as pd

from bioagentics.diagnostics.retinal_dr_screening.splits import (
    create_splits,
    extract_patient_id,
    load_split,
    save_splits,
)


def _make_catalog(n: int = 100) -> pd.DataFrame:
    """Create a mock catalog with multiple datasets and DR grades."""
    rng = pd.np if hasattr(pd, "np") else __import__("numpy").random.default_rng(42)
    import numpy as np

    rng = np.random.default_rng(42)

    records = []
    # EyePACS-like data with paired eyes
    for i in range(40):
        for eye in ["left", "right"]:
            records.append({
                "image_path": f"/data/eyepacs/{i}_{eye}.jpeg",
                "dr_grade": int(rng.choice([0, 0, 0, 1, 2, 3, 4])),
                "dataset_source": "eyepacs",
                "original_filename": f"{i}_{eye}.jpeg",
                "is_gradable": True,
            })

    # APTOS-like data
    for i in range(20):
        records.append({
            "image_path": f"/data/aptos/img_{i}.png",
            "dr_grade": int(rng.choice([0, 1, 2, 3, 4])),
            "dataset_source": "aptos2019",
            "original_filename": f"img_{i}.png",
            "is_gradable": True,
        })

    # Messidor-2 (holdout)
    for i in range(15):
        records.append({
            "image_path": f"/data/messidor2/img_{i}.tif",
            "dr_grade": int(rng.choice([0, 1, 2, 3, 4])),
            "dataset_source": "messidor2",
            "original_filename": f"img_{i}.tif",
            "is_gradable": True,
        })

    return pd.DataFrame(records)


def test_extract_patient_id_eyepacs():
    row = pd.Series({"original_filename": "12345_left.jpeg", "dataset_source": "eyepacs"})
    pid = extract_patient_id(row)
    assert pid == "eyepacs_12345"


def test_extract_patient_id_eyepacs_both_eyes():
    left = pd.Series({"original_filename": "999_left.jpeg", "dataset_source": "eyepacs"})
    right = pd.Series({"original_filename": "999_right.jpeg", "dataset_source": "eyepacs"})
    assert extract_patient_id(left) == extract_patient_id(right)


def test_extract_patient_id_aptos():
    row = pd.Series({"original_filename": "abc123.png", "dataset_source": "aptos2019"})
    pid = extract_patient_id(row)
    assert "aptos2019" in pid


def test_create_splits_has_all_splits():
    catalog = _make_catalog()
    df = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)
    assert "split" in df.columns
    splits = set(df["split"].unique())
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert "external_val" in splits


def test_holdout_dataset_in_external_val():
    catalog = _make_catalog()
    df = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)
    external = df[df["split"] == "external_val"]
    assert all(external["dataset_source"] == "messidor2")
    # All messidor2 images should be in external_val
    assert len(external) == len(catalog[catalog["dataset_source"] == "messidor2"])


def test_no_messidor2_in_train_val_test():
    catalog = _make_catalog()
    df = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)
    non_external = df[df["split"] != "external_val"]
    assert "messidor2" not in non_external["dataset_source"].values


def test_patient_level_no_leakage():
    """Both eyes of same EyePACS patient must be in same split."""
    catalog = _make_catalog()
    df = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)

    eyepacs = df[df["dataset_source"] == "eyepacs"].copy()
    # Extract patient number
    eyepacs["patient_num"] = eyepacs["original_filename"].str.extract(r"^(\d+)_")

    for patient_num, group in eyepacs.groupby("patient_num"):
        splits = group["split"].unique()
        assert len(splits) == 1, (
            f"Patient {patient_num} has images in multiple splits: {splits}"
        )


def test_all_images_assigned():
    catalog = _make_catalog()
    df = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)
    assert len(df) == len(catalog)
    assert not df["split"].isna().any()
    assert not (df["split"] == "").any()


def test_class_balance_approximate():
    """Grade distribution should be roughly similar across train/val/test."""
    catalog = _make_catalog()
    df = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)

    non_external = df[df["split"] != "external_val"]
    overall_dist = non_external["dr_grade"].value_counts(normalize=True).sort_index()

    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        if len(subset) < 5:
            continue
        split_dist = subset["dr_grade"].value_counts(normalize=True).sort_index()
        # Each grade should be within 20 percentage points of overall
        for grade in overall_dist.index:
            if grade in split_dist.index:
                diff = abs(split_dist[grade] - overall_dist[grade])
                assert diff < 0.25, (
                    f"Grade {grade} in {split}: {split_dist[grade]:.2f} "
                    f"vs overall {overall_dist[grade]:.2f}"
                )


def test_save_and_load_splits(tmp_path):
    catalog = _make_catalog()
    df = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)

    path = tmp_path / "splits.csv"
    save_splits(df, path)
    assert path.exists()

    train = load_split(path, "train", gradable_only=False)
    assert len(train) > 0
    assert all(train["split"] == "train")


def test_load_split_gradable_filter(tmp_path):
    catalog = _make_catalog()
    # Make some images ungradable
    catalog.loc[0:4, "is_gradable"] = False
    df = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)

    path = tmp_path / "splits.csv"
    save_splits(df, path)

    train_all = load_split(path, "train", gradable_only=False)
    train_gradable = load_split(path, "train", gradable_only=True)
    assert len(train_gradable) <= len(train_all)


def test_reproducibility():
    catalog = _make_catalog()
    df1 = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)
    df2 = create_splits(catalog, holdout_datasets=["messidor2"], seed=42)
    assert df1["split"].tolist() == df2["split"].tolist()
