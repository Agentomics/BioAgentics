"""Tests for pre-flare metabolite candidate features (task 1321).

Tests cover:
- Named candidate metabolite extraction (features_metabolomics.py)
- Succinate/OXPHOS pathway features (features_auxiliary.py)
- Levhar urate-to-creatinine+CRP index baseline (baselines.py)
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from bioagentics.crohns.flare_prediction.features_metabolomics import (
    CANDIDATE_METABOLITES,
    _find_candidate_columns,
    compute_metabolomic_features,
)
from bioagentics.crohns.flare_prediction.features_auxiliary import (
    KEY_PATHWAYS,
    _compute_pathway_features,
    _match_columns,
    compute_auxiliary_features,
)
from bioagentics.crohns.flare_prediction.baselines import (
    extract_levhar_index_features,
)
from bioagentics.crohns.flare_prediction.flare_events import Window


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_windows(subject_id: str = "S001", n_windows: int = 2) -> list[Window]:
    """Create synthetic classification windows."""
    base = pd.Timestamp("2015-01-01")
    windows = []
    for i in range(n_windows):
        start = base + timedelta(weeks=i * 4)
        end = start + timedelta(weeks=2)
        label = "pre_flare" if i % 2 == 0 else "stable"
        windows.append(Window(
            subject_id=subject_id,
            window_start=start,
            window_end=end,
            label=label,
            anchor_visit=i + 1,
        ))
    return windows


def _make_metabolomics(windows: list[Window]) -> pd.DataFrame:
    """Create synthetic metabolomics DataFrame with candidate metabolite columns."""
    rng = np.random.default_rng(42)
    rows = []
    for w in windows:
        for day_offset in range(0, 14, 3):
            date = w.window_start + timedelta(days=day_offset)
            row = {
                "subject_id": w.subject_id,
                "visit_num": day_offset // 3 + 1,
                "date": date,
                "urate_plasma": rng.uniform(3, 8),
                "uric_acid_serum": rng.uniform(3, 8),
                "3-hydroxybutyrate": rng.uniform(0.1, 2.0),
                "acetoacetate_level": rng.uniform(0.05, 1.0),
                "trehalose_concentration": rng.uniform(0.01, 0.5),
                "mesaconate_abundance": rng.uniform(0, 0.3),
                "creatinine_plasma": rng.uniform(0.6, 1.2),
                "random_metabolite_A": rng.uniform(0, 10),
                "random_metabolite_B": rng.uniform(0, 10),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.set_index(["subject_id", "visit_num"])


def _make_pathways(windows: list[Window]) -> pd.DataFrame:
    """Create synthetic pathway DataFrame with succinate/OXPHOS columns."""
    rng = np.random.default_rng(99)
    rows = []
    for w in windows:
        for day_offset in range(0, 14, 3):
            date = w.window_start + timedelta(days=day_offset)
            row = {
                "subject_id": w.subject_id,
                "visit_num": day_offset // 3 + 1,
                "date": date,
                "succinate_pathway": rng.uniform(0, 1),
                "oxidative_phosphorylation_complex": rng.uniform(0, 1),
                "TCA_cycle": rng.uniform(0, 1),
                "butyrate_synthesis": rng.uniform(0, 1),
                "glutathione_metabolism": rng.uniform(0, 1),
                "sulfate_assimilation": rng.uniform(0, 1),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.set_index(["subject_id", "visit_num"])


def _make_metadata_with_crp(windows: list[Window]) -> pd.DataFrame:
    """Create metadata with CRP values."""
    rng = np.random.default_rng(7)
    rows = []
    for w in windows:
        for day_offset in range(0, 14, 3):
            date = w.window_start + timedelta(days=day_offset)
            rows.append({
                "subject_id": w.subject_id,
                "visit_num": day_offset // 3 + 1,
                "date": date,
                "diagnosis": "CD",
                "crp": rng.uniform(0.1, 15.0),
            })
    df = pd.DataFrame(rows)
    return df.set_index(["subject_id", "visit_num"])


# ---------------------------------------------------------------------------
# Tests: Candidate metabolite matching
# ---------------------------------------------------------------------------


class TestCandidateMetabolites:
    def test_candidate_dict_has_expected_entries(self):
        expected = {"urate", "3hb", "acetoacetate", "trehalose", "mesaconic_acid"}
        assert set(CANDIDATE_METABOLITES.keys()) == expected

    def test_find_candidate_columns_matches(self):
        cols = [
            "subject_id", "visit_num", "date",
            "urate_plasma", "3-hydroxybutyrate", "acetoacetate_level",
            "trehalose_concentration", "mesaconate_abundance",
            "random_metabolite_A",
        ]
        meta_cols = {"subject_id", "visit_num", "date"}
        result = _find_candidate_columns(cols, meta_cols)
        assert "urate_plasma" in result["urate"]
        assert "3-hydroxybutyrate" in result["3hb"]
        assert "acetoacetate_level" in result["acetoacetate"]
        assert "trehalose_concentration" in result["trehalose"]
        assert "mesaconate_abundance" in result["mesaconic_acid"]

    def test_find_candidate_columns_empty_for_unmatched(self):
        cols = ["subject_id", "random_X", "random_Y"]
        result = _find_candidate_columns(cols, {"subject_id"})
        for cand_name, matched in result.items():
            assert matched == [], f"Expected no match for {cand_name}"


class TestMetabolomicCandidateFeatures:
    def test_candidate_features_present(self):
        windows = _make_windows()
        met = _make_metabolomics(windows)
        result = compute_metabolomic_features(met, windows)
        # Check that candidate features are computed
        cand_cols = [c for c in result.columns if "met_cand_" in c]
        assert len(cand_cols) > 0

    def test_candidate_slope_mean_std_range(self):
        windows = _make_windows()
        met = _make_metabolomics(windows)
        result = compute_metabolomic_features(met, windows)
        for cand in CANDIDATE_METABOLITES:
            assert f"met_cand_slope__{cand}" in result.columns, f"Missing slope for {cand}"
            assert f"met_cand_mean__{cand}" in result.columns, f"Missing mean for {cand}"
            assert f"met_cand_std__{cand}" in result.columns, f"Missing std for {cand}"
            assert f"met_cand_range__{cand}" in result.columns, f"Missing range for {cand}"

    def test_candidate_features_nan_when_absent(self):
        windows = _make_windows()
        # Metabolomics with no candidate columns
        rng = np.random.default_rng(1)
        rows = []
        for w in windows:
            for d in range(0, 14, 3):
                rows.append({
                    "subject_id": w.subject_id,
                    "visit_num": d // 3 + 1,
                    "date": w.window_start + timedelta(days=d),
                    "generic_feature_1": rng.uniform(0, 1),
                    "generic_feature_2": rng.uniform(0, 1),
                })
        met = pd.DataFrame(rows).set_index(["subject_id", "visit_num"])
        result = compute_metabolomic_features(met, windows)
        for cand in CANDIDATE_METABOLITES:
            assert np.isnan(result[f"met_cand_slope__{cand}"].iloc[0])

    def test_correct_number_of_rows(self):
        windows = _make_windows(n_windows=3)
        met = _make_metabolomics(windows)
        result = compute_metabolomic_features(met, windows)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: Succinate/OXPHOS pathway features
# ---------------------------------------------------------------------------


class TestSuccinateOxphosFeatures:
    def test_succinate_oxphos_in_key_pathways(self):
        assert "succinate_oxphos" in KEY_PATHWAYS

    def test_keywords_match_succinate_columns(self):
        cols = ["succinate_pathway", "oxidative_phosphorylation_complex", "TCA_cycle"]
        keywords = KEY_PATHWAYS["succinate_oxphos"]
        matched = _match_columns(cols, keywords)
        assert len(matched) == 3

    def test_pathway_features_include_succinate(self):
        windows = _make_windows()
        pw = _make_pathways(windows)
        data: dict[str, pd.DataFrame | None] = {
            "pathways": pw,
            "transcriptomics": None,
            "serology": None,
        }
        result = compute_auxiliary_features(data, windows)
        assert "pw_slope__succinate_oxphos" in result.columns
        assert "pw_mean__succinate_oxphos" in result.columns

    def test_succinate_features_not_nan(self):
        windows = _make_windows()
        pw = _make_pathways(windows)
        data: dict[str, pd.DataFrame | None] = {
            "pathways": pw,
            "transcriptomics": None,
            "serology": None,
        }
        result = compute_auxiliary_features(data, windows)
        assert result["pw_mean__succinate_oxphos"].notna().all()


# ---------------------------------------------------------------------------
# Tests: Levhar urate-to-creatinine+CRP index baseline
# ---------------------------------------------------------------------------


class TestLevharIndex:
    def test_basic_extraction(self):
        windows = _make_windows()
        met = _make_metabolomics(windows)
        meta = _make_metadata_with_crp(windows)
        result = extract_levhar_index_features(met, meta, windows)
        assert "levhar_index_mean" in result.columns
        assert "levhar_index_slope" in result.columns
        assert "levhar_index_last" in result.columns
        assert len(result) == len(windows)

    def test_values_are_finite(self):
        windows = _make_windows()
        met = _make_metabolomics(windows)
        meta = _make_metadata_with_crp(windows)
        result = extract_levhar_index_features(met, meta, windows)
        assert result["levhar_index_mean"].notna().all()

    def test_returns_nan_when_no_metabolomics(self):
        windows = _make_windows()
        result = extract_levhar_index_features(None, None, windows)
        assert result["levhar_index_mean"].isna().all()
        assert result["levhar_index_slope"].isna().all()

    def test_returns_nan_when_empty_metabolomics(self):
        windows = _make_windows()
        empty = pd.DataFrame(columns=["subject_id", "visit_num", "date", "urate"])
        result = extract_levhar_index_features(empty, None, windows)
        assert result["levhar_index_mean"].isna().all()

    def test_works_without_creatinine(self):
        windows = _make_windows()
        met = _make_metabolomics(windows)
        # Remove creatinine column
        met_reset = met.reset_index()
        met_no_creat = met_reset.drop(columns=["creatinine_plasma"]).set_index(
            ["subject_id", "visit_num"]
        )
        result = extract_levhar_index_features(met_no_creat, None, windows)
        # Should still compute using urate-only
        assert result["levhar_index_mean"].notna().all()

    def test_correct_row_count(self):
        windows = _make_windows(n_windows=4)
        met = _make_metabolomics(windows)
        result = extract_levhar_index_features(met, None, windows)
        assert len(result) == 4
