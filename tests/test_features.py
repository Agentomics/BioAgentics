"""Tests for Phase 2 feature engineering modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.crohns.flare_prediction.flare_events import (
    Window,
    detect_flares,
    extract_windows,
)
from bioagentics.crohns.flare_prediction.features_microbiome import (
    compute_microbiome_features,
    _shannon_diversity,
    _bray_curtis_consecutive,
)
from bioagentics.crohns.flare_prediction.features_metabolomics import (
    compute_metabolomic_features,
)
from bioagentics.crohns.flare_prediction.features_auxiliary import (
    compute_auxiliary_features,
)


def _make_windows_and_data(n_subjects=3, visits=10, interval_days=14):
    """Create synthetic data with flares and stable periods."""
    rng = np.random.default_rng(42)
    idx_cols = ["subject_id", "visit_num"]
    species_names = [f"sp_{i}" for i in range(5)]
    pw_names = ["PWY-butyrate-synth", "PWY-sulfur-red", "PWY-oxidative-stress"]
    met_names = [f"met_{i}" for i in range(15)]
    sero_markers = ["ASCA_IgA", "ASCA_IgG", "anti_CBir1", "anti_OmpC"]

    hbi_rows, species_rows, pw_rows, met_rows, sero_rows, meta_rows = [], [], [], [], [], []

    for s in range(n_subjects):
        sid = f"P{s:03d}"
        dates = pd.date_range("2015-01-01", periods=visits, freq=f"{interval_days}D")

        if s == 0:
            scores = [2, 1, 3, 2, 8, 3, 2, 1, 2, 1][:visits]
        else:
            scores = [rng.integers(0, 4) for _ in range(visits)]

        for v in range(visits):
            meta_rows.append({"subject_id": sid, "visit_num": v + 1, "date": dates[v], "diagnosis": "CD"})
            hbi_rows.append({"subject_id": sid, "visit_num": v + 1, "date": dates[v], "hbi_score": scores[v]})

            row_sp = {"subject_id": sid, "visit_num": v + 1, "date": dates[v]}
            abund = rng.dirichlet(np.ones(5))
            for sn, ab in zip(species_names, abund):
                row_sp[sn] = float(ab)
            species_rows.append(row_sp)

            row_pw = {"subject_id": sid, "visit_num": v + 1, "date": dates[v]}
            for pw in pw_names:
                row_pw[pw] = float(rng.exponential(10))
            pw_rows.append(row_pw)

            row_met = {"subject_id": sid, "visit_num": v + 1, "date": dates[v]}
            for m in met_names:
                row_met[m] = float(rng.normal(5, 2))
            met_rows.append(row_met)

            sero_rows.append({
                "subject_id": sid, "visit_num": v + 1, "date": dates[v],
                **{m: float(rng.exponential(5)) for m in sero_markers},
            })

    hbi = pd.DataFrame(hbi_rows).set_index(idx_cols)
    species = pd.DataFrame(species_rows).set_index(idx_cols)
    pathways = pd.DataFrame(pw_rows).set_index(idx_cols)
    metabolomics = pd.DataFrame(met_rows).set_index(idx_cols)
    serology = pd.DataFrame(sero_rows).set_index(idx_cols)
    metadata = pd.DataFrame(meta_rows).set_index(idx_cols)

    data = {
        "metadata": metadata,
        "hbi": hbi,
        "species": species,
        "pathways": pathways,
        "metabolomics": metabolomics,
        "serology": serology,
        "transcriptomics": None,
    }

    flares = detect_flares(hbi)
    windows = extract_windows(hbi, flares, lead_weeks=2)
    return windows, data, species, metabolomics


class TestMicrobiomeFeatures:
    def test_basic_output_shape(self):
        windows, data, species, _ = _make_windows_and_data()
        result = compute_microbiome_features(species, windows)
        assert result.shape[0] == len(windows)
        assert result.shape[1] > 0

    def test_contains_expected_columns(self):
        windows, data, species, _ = _make_windows_and_data()
        result = compute_microbiome_features(species, windows)
        assert "mb_shannon_mean" in result.columns
        assert "mb_shannon_slope" in result.columns
        assert "mb_bc_mean" in result.columns
        assert "mb_volatility_index" in result.columns

    def test_species_slopes_present(self):
        windows, data, species, _ = _make_windows_and_data()
        result = compute_microbiome_features(species, windows)
        slope_cols = [c for c in result.columns if c.startswith("mb_slope__")]
        assert len(slope_cols) > 0

    def test_empty_windows(self):
        _, data, species, _ = _make_windows_and_data()
        result = compute_microbiome_features(species, [])
        assert result.empty

    def test_shannon_diversity_values(self):
        # Uniform distribution should have high diversity
        uniform = np.ones(10) / 10
        assert _shannon_diversity(uniform) > 0
        # Single species = 0 diversity
        single = np.array([1.0, 0.0, 0.0])
        assert _shannon_diversity(single) == 0.0

    def test_bray_curtis_identical_samples(self):
        samples = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        dists = _bray_curtis_consecutive(samples)
        assert len(dists) == 1
        assert abs(dists[0]) < 1e-10


class TestMetabolomicFeatures:
    def test_basic_output_shape(self):
        windows, data, _, metabolomics = _make_windows_and_data()
        result = compute_metabolomic_features(metabolomics, windows)
        assert result.shape[0] == len(windows)
        assert result.shape[1] > 0

    def test_contains_global_stats(self):
        windows, data, _, metabolomics = _make_windows_and_data()
        result = compute_metabolomic_features(metabolomics, windows)
        assert "met_mean_global" in result.columns
        assert "met_std_global" in result.columns

    def test_empty_windows(self):
        _, data, _, metabolomics = _make_windows_and_data()
        result = compute_metabolomic_features(metabolomics, [])
        assert result.empty


class TestAuxiliaryFeatures:
    def test_basic_output_shape(self):
        windows, data, _, _ = _make_windows_and_data()
        result = compute_auxiliary_features(data, windows)
        assert result.shape[0] == len(windows)
        assert result.shape[1] > 0

    def test_pathway_features_present(self):
        windows, data, _, _ = _make_windows_and_data()
        result = compute_auxiliary_features(data, windows)
        pw_cols = [c for c in result.columns if c.startswith("pw_")]
        assert len(pw_cols) > 0

    def test_serology_features_present(self):
        windows, data, _, _ = _make_windows_and_data()
        result = compute_auxiliary_features(data, windows)
        sero_cols = [c for c in result.columns if c.startswith("sero_")]
        assert len(sero_cols) > 0

    def test_missing_layers_handled(self):
        windows, data, _, _ = _make_windows_and_data()
        # Remove optional layers
        data["serology"] = None
        data["transcriptomics"] = None
        result = compute_auxiliary_features(data, windows)
        assert result.shape[0] == len(windows)
        # Should only have pathway features
        sero_cols = [c for c in result.columns if c.startswith("sero_")]
        assert len(sero_cols) == 0

    def test_empty_windows(self):
        _, data, _, _ = _make_windows_and_data()
        result = compute_auxiliary_features(data, [])
        assert result.empty
