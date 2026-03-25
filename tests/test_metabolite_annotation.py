"""Tests for metabolite_annotation module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Neutral Mass Computation ──


class TestComputeNeutralMass:
    def test_negative_mode_c18(self):
        from bioagentics.data.metabolite_annotation import PROTON_MASS, compute_neutral_mass

        mz = 313.2387
        result = compute_neutral_mass(mz, "C18-neg")
        expected = mz + PROTON_MASS  # [M-H]⁻: M = m/z + H
        assert abs(result - expected) < 1e-6

    def test_negative_mode_hilic(self):
        from bioagentics.data.metabolite_annotation import PROTON_MASS, compute_neutral_mass

        mz = 200.0
        result = compute_neutral_mass(mz, "HILIC-neg")
        expected = mz + PROTON_MASS
        assert abs(result - expected) < 1e-6

    def test_positive_mode_c8(self):
        from bioagentics.data.metabolite_annotation import PROTON_MASS, compute_neutral_mass

        mz = 500.0
        result = compute_neutral_mass(mz, "C8-pos")
        expected = mz - PROTON_MASS  # [M+H]⁺: M = m/z - H
        assert abs(result - expected) < 1e-6

    def test_positive_mode_hilic(self):
        from bioagentics.data.metabolite_annotation import PROTON_MASS, compute_neutral_mass

        mz = 300.0
        result = compute_neutral_mass(mz, "HILIC-pos")
        expected = mz - PROTON_MASS
        assert abs(result - expected) < 1e-6

    def test_unknown_method_returns_nan(self):
        from bioagentics.data.metabolite_annotation import compute_neutral_mass

        result = compute_neutral_mass(100.0, "UNKNOWN")
        assert np.isnan(result)

    def test_empty_method_returns_nan(self):
        from bioagentics.data.metabolite_annotation import compute_neutral_mass

        result = compute_neutral_mass(100.0, "")
        assert np.isnan(result)


class TestComputeAllNeutralMasses:
    def test_negative_mode_returns_multiple_adducts(self):
        from bioagentics.data.metabolite_annotation import compute_all_neutral_masses

        results = compute_all_neutral_masses(313.2387, "C18-neg")
        assert len(results) >= 2  # primary + extra adducts
        names = [name for name, _ in results]
        assert "[M-H]-" in names

    def test_positive_mode_returns_multiple_adducts(self):
        from bioagentics.data.metabolite_annotation import compute_all_neutral_masses

        results = compute_all_neutral_masses(500.0, "C8-pos")
        assert len(results) >= 2
        names = [name for name, _ in results]
        assert "[M+H]+" in names

    def test_unknown_method_returns_empty(self):
        from bioagentics.data.metabolite_annotation import compute_all_neutral_masses

        results = compute_all_neutral_masses(100.0, "UNKNOWN")
        assert results == []


# ── Mass Matching ──


class TestMatchByMass:
    @pytest.fixture()
    def hmdb_df(self):
        return pd.DataFrame({
            "hmdb_id": ["HMDB0001", "HMDB0002", "HMDB0003", "HMDB0004"],
            "name": ["metabolite_A", "metabolite_B", "metabolite_C", "metabolite_D"],
            "monoisotopic_weight": [314.2460, 314.2462, 500.0000, 200.0000],
            "chemical_formula": ["C20H34O3", "C20H34O3", "C30H50O5", "C10H15O3"],
            "super_class": ["Lipids", "Lipids", "Steroids", "Amines"],
        })

    def test_exact_match(self, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_by_mass

        result = match_by_mass(314.2460, hmdb_df, ppm_tolerance=10.0)
        assert len(result) >= 1
        assert result.iloc[0]["hmdb_id"] == "HMDB0001"
        assert result.iloc[0]["ppm_error"] < 1.0

    def test_close_match_within_tolerance(self, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_by_mass

        # 5 ppm from 314.2460 is ~0.00157 Da
        query = 314.2460 + 0.001  # ~3.2 ppm
        result = match_by_mass(query, hmdb_df, ppm_tolerance=10.0)
        assert len(result) >= 1

    def test_no_match_outside_tolerance(self, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_by_mass

        # 100 ppm from 314.2460 is ~0.0314 Da; 0.05 is beyond 10 ppm
        query = 314.2460 + 0.05
        result = match_by_mass(query, hmdb_df, ppm_tolerance=10.0)
        assert len(result) == 0

    def test_nan_mass_returns_empty(self, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_by_mass

        result = match_by_mass(float("nan"), hmdb_df, ppm_tolerance=10.0)
        assert len(result) == 0

    def test_zero_mass_returns_empty(self, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_by_mass

        result = match_by_mass(0.0, hmdb_df, ppm_tolerance=10.0)
        assert len(result) == 0

    def test_results_sorted_by_ppm(self, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_by_mass

        # Query between the two close masses
        query = 314.2461
        result = match_by_mass(query, hmdb_df, ppm_tolerance=10.0)
        if len(result) > 1:
            assert result.iloc[0]["ppm_error"] <= result.iloc[1]["ppm_error"]


# ── Feature Matching Pipeline ──


class TestMatchFeaturesToHmdb:
    @pytest.fixture()
    def biom_meta(self):
        return pd.DataFrame({
            "feature_id": ["F001", "F002", "F003"],
            "RT": [9.75, 5.0, 3.0],
            "mz": [313.2387, 500.0, 200.0],
            "Method": ["C18-neg", "C8-pos", "C18-neg"],
            "Metabolite": ["known_compound", "", ""],
            "HMDB_ID": ["HMDB12345", "", ""],
            "QC_CV": [0.03, 0.05, 0.04],
            "is_annotated": [True, False, False],
        })

    @pytest.fixture()
    def hmdb_df(self):
        return pd.DataFrame({
            "hmdb_id": ["HMDB0001", "HMDB0002"],
            "name": ["metabolite_A", "metabolite_B"],
            "monoisotopic_weight": [
                314.2460,  # close to 313.2387 + proton (314.2460)
                498.9927,  # close to 500.0 - proton (498.9927)
            ],
            "chemical_formula": ["C20H34O3", "C30H50O5"],
            "super_class": ["Lipids", "Steroids"],
        })

    def test_only_unannotated_are_matched(self, biom_meta, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_features_to_hmdb

        result = match_features_to_hmdb(
            biom_meta, hmdb_df, ppm_tolerance=10.0,
            include_extra_adducts=False,
        )
        # F001 is annotated, should not have putative matches
        f001 = result[result["feature_id"] == "F001"]
        assert f001["putative_hmdb_id"].isna().all()

    def test_feature_subset_filter(self, biom_meta, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_features_to_hmdb

        result = match_features_to_hmdb(
            biom_meta, hmdb_df,
            feature_subset=["F001", "F002"],
            include_extra_adducts=False,
        )
        assert set(result["feature_id"].unique()) <= {"F001", "F002"}

    def test_neutral_mass_column_added(self, biom_meta, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_features_to_hmdb

        result = match_features_to_hmdb(
            biom_meta, hmdb_df, include_extra_adducts=False,
        )
        assert "neutral_mass" in result.columns

    def test_output_columns_present(self, biom_meta, hmdb_df):
        from bioagentics.data.metabolite_annotation import match_features_to_hmdb

        result = match_features_to_hmdb(
            biom_meta, hmdb_df, include_extra_adducts=False,
        )
        expected_cols = [
            "feature_id", "RT", "mz", "Method", "Metabolite", "HMDB_ID",
            "QC_CV", "is_annotated", "neutral_mass", "putative_hmdb_id",
            "putative_name", "putative_formula", "putative_super_class",
            "adduct", "ppm_error", "match_rank",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


# ── Biom Metadata Extraction (Integration Test) ──


class TestExtractBiomMetadata:
    @pytest.fixture()
    def biom_path(self):
        path = Path("data/hmp2/HMP2_metabolomics_w_metadata.biom.gz")
        if not path.exists():
            pytest.skip("HMP2 biom file not available")
        return path

    def test_extraction_returns_expected_shape(self, biom_path):
        from bioagentics.data.metabolite_annotation import extract_biom_metadata

        df = extract_biom_metadata(biom_path)
        assert len(df) == 81867
        assert "feature_id" in df.columns
        assert "mz" in df.columns
        assert "RT" in df.columns
        assert "Method" in df.columns

    def test_annotated_count(self, biom_path):
        from bioagentics.data.metabolite_annotation import extract_biom_metadata

        df = extract_biom_metadata(biom_path)
        assert df["is_annotated"].sum() == 592

    def test_mz_values_are_numeric(self, biom_path):
        from bioagentics.data.metabolite_annotation import extract_biom_metadata

        df = extract_biom_metadata(biom_path)
        # All features should have numeric m/z
        assert df["mz"].notna().sum() > 80000

    def test_methods_are_expected(self, biom_path):
        from bioagentics.data.metabolite_annotation import extract_biom_metadata

        df = extract_biom_metadata(biom_path)
        expected_methods = {"C18-neg", "HILIC-neg", "HILIC-pos", "C8-pos", ""}
        assert set(df["Method"].unique()) <= expected_methods


# ── Safe Float ──


class TestSafeFloat:
    def test_valid_string(self):
        from bioagentics.data.metabolite_annotation import _safe_float

        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_empty_string(self):
        from bioagentics.data.metabolite_annotation import _safe_float

        assert np.isnan(_safe_float(""))

    def test_none(self):
        from bioagentics.data.metabolite_annotation import _safe_float

        assert np.isnan(_safe_float(None))

    def test_numeric(self):
        from bioagentics.data.metabolite_annotation import _safe_float

        assert _safe_float(42) == 42.0
