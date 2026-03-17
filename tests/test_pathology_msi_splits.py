"""Tests for stratified train/val/test splits."""

import pandas as pd
import pytest

from bioagentics.models.pathology_msi.splits import (
    _extract_patient_id,
    create_stratified_splits,
    save_splits,
    validate_splits,
)


def _make_labels_df(n_per_type=50):
    """Create a synthetic labels DataFrame for testing."""
    rows = []
    for ct in ["COAD", "READ", "UCEC", "STAD"]:
        for i in range(n_per_type):
            # ~20% MSI-H rate
            msi = "MSI-H" if i < n_per_type * 0.2 else "MSS"
            rows.append(
                {
                    "case_id": f"case-{ct}-{i:04d}",
                    "submitter_id": f"TCGA-{ct[:2]}-{i:04d}-01A",
                    "project_id": f"TCGA-{ct}",
                    "cancer_type": ct,
                    "msi_status": msi,
                }
            )
    return pd.DataFrame(rows)


class TestExtractPatientId:
    def test_standard_barcode(self):
        assert _extract_patient_id("TCGA-AA-0001-01A") == "TCGA-AA-0001"

    def test_short_barcode(self):
        assert _extract_patient_id("TCGA-AA-0001") == "TCGA-AA-0001"

    def test_single_part(self):
        assert _extract_patient_id("ABC") == "ABC"


class TestCreateStratifiedSplits:
    def test_basic_split(self):
        df = _make_labels_df()
        result = create_stratified_splits(df)

        assert "split" in result.columns
        assert "patient_id" in result.columns
        assert set(result["split"].unique()) == {"train", "val", "test"}

    def test_split_proportions(self):
        df = _make_labels_df(n_per_type=100)
        result = create_stratified_splits(df)

        total = len(result)
        train_frac = (result["split"] == "train").sum() / total
        val_frac = (result["split"] == "val").sum() / total
        test_frac = (result["split"] == "test").sum() / total

        # Allow 5% tolerance
        assert abs(train_frac - 0.70) < 0.05
        assert abs(val_frac - 0.15) < 0.05
        assert abs(test_frac - 0.15) < 0.05

    def test_no_patient_leakage(self):
        df = _make_labels_df()
        result = create_stratified_splits(df)

        train_patients = set(result[result["split"] == "train"]["patient_id"])
        val_patients = set(result[result["split"] == "val"]["patient_id"])
        test_patients = set(result[result["split"] == "test"]["patient_id"])

        assert not train_patients & val_patients
        assert not train_patients & test_patients
        assert not val_patients & test_patients

    def test_reproducible_with_same_seed(self):
        df = _make_labels_df()
        r1 = create_stratified_splits(df, seed=123)
        r2 = create_stratified_splits(df, seed=123)
        assert list(r1["split"]) == list(r2["split"])

    def test_different_seed_gives_different_splits(self):
        df = _make_labels_df()
        r1 = create_stratified_splits(df, seed=1)
        r2 = create_stratified_splits(df, seed=2)
        assert list(r1["split"]) != list(r2["split"])

    def test_excludes_unknown_status(self):
        df = _make_labels_df(n_per_type=20)
        df.loc[0, "msi_status"] = "unknown"
        result = create_stratified_splits(df)
        assert len(result) == len(df) - 1

    def test_stratification_preserves_msi_rates(self):
        df = _make_labels_df(n_per_type=100)
        result = create_stratified_splits(df)

        # Check per-cancer-type MSI-H rates are similar across splits
        for ct in result["cancer_type"].unique():
            ct_df = result[result["cancer_type"] == ct]
            rates = {}
            for split_name in ["train", "val", "test"]:
                split_df = ct_df[ct_df["split"] == split_name]
                if len(split_df) > 0:
                    rates[split_name] = (split_df["msi_status"] == "MSI-H").mean()

            # MSI-H rate should be within 10% across splits
            if len(rates) > 1:
                rate_values = list(rates.values())
                assert max(rate_values) - min(rate_values) < 0.15


class TestValidateSplits:
    def test_valid_splits(self):
        df = _make_labels_df()
        result = create_stratified_splits(df)
        validation = validate_splits(result)
        assert validation["valid"] is True
        assert len(validation["issues"]) == 0

    def test_detects_leakage(self):
        df = _make_labels_df(n_per_type=20)
        result = create_stratified_splits(df)
        # Manually introduce leakage
        result.loc[result["split"] == "val", "patient_id"] = result[
            result["split"] == "train"
        ]["patient_id"].iloc[0]
        validation = validate_splits(result)
        assert validation["valid"] is False


def test_save_splits(tmp_path):
    df = _make_labels_df(n_per_type=10)
    result = create_stratified_splits(df)
    path = save_splits(result, tmp_path)
    assert path.exists()
    loaded = pd.read_csv(path)
    assert "split" in loaded.columns
