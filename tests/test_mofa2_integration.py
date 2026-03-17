"""Tests for Phase 3 MOFA2 integration module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.crohns.flare_prediction.flare_events import Window
from bioagentics.crohns.flare_prediction.mofa2_integration import (
    prepare_mofa_input,
    run_mofa,
    extract_factor_trajectories,
    factor_flare_association,
)


def _make_windows(n=20):
    """Create synthetic windows."""
    windows = []
    base = pd.Timestamp("2015-01-01")
    for i in range(n):
        windows.append(
            Window(
                subject_id=f"P{i % 5:03d}",
                window_start=base + pd.Timedelta(days=i * 14),
                window_end=base + pd.Timedelta(days=i * 14 + 14),
                label="pre_flare" if i % 2 == 0 else "stable",
                anchor_visit=i + 1,
            )
        )
    return windows


def _make_views(n_samples=40, seed=42):
    """Create synthetic multi-view data with shared latent structure."""
    rng = np.random.default_rng(seed)
    windows = _make_windows(n_samples)

    # Create strong shared latent structure across views
    # 3 latent factors driving both views
    Z = rng.normal(0, 1, (n_samples, 3))
    # Add class signal to first factor
    labels = np.array([w.label for w in windows])
    Z[labels == "pre_flare", 0] += 3.0

    # View 1: microbiome = Z @ W1 + noise
    W1 = rng.normal(0, 1, (3, 15))
    mb = Z @ W1 + rng.normal(0, 0.3, (n_samples, 15))

    # View 2: metabolomics = Z @ W2 + noise
    W2 = rng.normal(0, 1, (3, 20))
    met = Z @ W2 + rng.normal(0, 0.3, (n_samples, 20))

    features_by_layer = {
        "microbiome": pd.DataFrame(mb, columns=[f"mb_{i}" for i in range(15)]),
        "metabolomics": pd.DataFrame(met, columns=[f"met_{i}" for i in range(20)]),
    }
    return features_by_layer, windows


class TestPrepareInput:
    def test_basic_output(self):
        features_by_layer, windows = _make_views()
        views = prepare_mofa_input(features_by_layer, windows)
        assert "microbiome" in views
        assert "metabolomics" in views
        assert views["microbiome"].shape == (40, 15)
        assert views["metabolomics"].shape == (40, 20)

    def test_skips_empty_layers(self):
        features_by_layer, windows = _make_views()
        features_by_layer["empty"] = pd.DataFrame()
        views = prepare_mofa_input(features_by_layer, windows)
        assert "empty" not in views

    def test_skips_mismatched_rows(self):
        features_by_layer, windows = _make_views()
        features_by_layer["bad"] = pd.DataFrame(np.zeros((5, 3)), columns=["a", "b", "c"])
        views = prepare_mofa_input(features_by_layer, windows)
        assert "bad" not in views


class TestRunMofa:
    def test_basic_run(self):
        features_by_layer, windows = _make_views()
        views = prepare_mofa_input(features_by_layer, windows)
        result = run_mofa(views, n_factors=3, seed=42, max_iter=100)

        assert "factors" in result
        assert "weights" in result
        assert "variance_explained" in result
        assert result["factors"].shape[0] == 40
        assert result["n_factors"] <= 3

    def test_factor_values_finite(self):
        features_by_layer, windows = _make_views()
        views = prepare_mofa_input(features_by_layer, windows)
        result = run_mofa(views, n_factors=3, seed=42, max_iter=100)
        assert np.all(np.isfinite(result["factors"]))

    def test_weights_per_view(self):
        features_by_layer, windows = _make_views()
        views = prepare_mofa_input(features_by_layer, windows)
        result = run_mofa(views, n_factors=3, seed=42, max_iter=100)
        for vname in views:
            assert vname in result["weights"]


class TestExtractTrajectories:
    def test_output_shape(self):
        features_by_layer, windows = _make_views()
        views = prepare_mofa_input(features_by_layer, windows)
        result = run_mofa(views, n_factors=3, seed=42, max_iter=100)
        traj = extract_factor_trajectories(result, windows)
        assert traj.shape[0] == len(windows)
        assert traj.shape[1] == result["n_factors"]


class TestFactorAssociation:
    def test_basic_output(self):
        features_by_layer, windows = _make_views()
        views = prepare_mofa_input(features_by_layer, windows)
        result = run_mofa(views, n_factors=3, seed=42, max_iter=100)
        assoc = factor_flare_association(result, windows)
        assert "factor" in assoc.columns
        assert "mean_diff" in assoc.columns
        assert "p_value" in assoc.columns
        assert len(assoc) == result["n_factors"]
