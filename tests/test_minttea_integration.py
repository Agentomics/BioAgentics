"""Tests for Phase 3 MintTea multi-omic module discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bioagentics.crohns.flare_prediction.flare_events import Window
from bioagentics.crohns.flare_prediction.minttea_integration import (
    discover_modules,
    compute_module_scores,
    compare_with_mofa,
    save_module_results,
)


def _make_windows(n_patients=6, per_patient=4):
    base = pd.Timestamp("2015-01-01")
    windows = []
    for p in range(n_patients):
        sid = f"P{p:03d}"
        for i in range(per_patient):
            idx = p * per_patient + i
            label = "pre_flare" if i < per_patient // 2 else "stable"
            windows.append(
                Window(
                    subject_id=sid,
                    window_start=base + pd.Timedelta(days=idx * 14),
                    window_end=base + pd.Timedelta(days=idx * 14 + 14),
                    label=label,
                    anchor_visit=idx + 1,
                )
            )
    return windows


def _make_features_by_layer(windows, seed=42):
    rng = np.random.default_rng(seed)
    n = len(windows)

    # Create features with signal in pre-flare
    mb_data = rng.normal(0, 1, (n, 8))
    met_data = rng.normal(0, 1, (n, 6))
    pw_data = rng.normal(0, 1, (n, 5))

    for i, w in enumerate(windows):
        if w.label == "pre_flare":
            mb_data[i, :2] += 2.0
            met_data[i, :2] += 1.5

    return {
        "microbiome": pd.DataFrame(
            mb_data, columns=[f"mb_feat_{j}" for j in range(8)]
        ),
        "metabolomics": pd.DataFrame(
            met_data, columns=[f"met_feat_{j}" for j in range(6)]
        ),
        "pathways": pd.DataFrame(
            pw_data, columns=[f"pw_feat_{j}" for j in range(5)]
        ),
    }


class TestDiscoverModules:
    def test_basic(self):
        windows = _make_windows()
        features = _make_features_by_layer(windows)
        result = discover_modules(features, windows, n_modules=3)

        assert len(result.modules) > 0
        assert not result.module_scores.empty
        assert not result.feature_memberships.empty

    def test_module_structure(self):
        windows = _make_windows()
        features = _make_features_by_layer(windows)
        result = discover_modules(features, windows, n_modules=3)

        for mod in result.modules:
            assert len(mod.features) > 0
            assert len(mod.layers) > 0
            assert mod.weights is not None
            assert isinstance(mod.label_correlation, float)

    def test_module_scores_shape(self):
        windows = _make_windows()
        features = _make_features_by_layer(windows)
        result = discover_modules(features, windows, n_modules=3)

        assert result.module_scores.shape[0] == len(windows)
        assert result.module_scores.shape[1] > 0

    def test_feature_memberships(self):
        windows = _make_windows()
        features = _make_features_by_layer(windows)
        result = discover_modules(features, windows, n_modules=3)

        assert "module_id" in result.feature_memberships.columns
        assert "feature" in result.feature_memberships.columns
        assert "omic_layer" in result.feature_memberships.columns
        assert "weight" in result.feature_memberships.columns

    def test_empty_input(self):
        windows = _make_windows()
        result = discover_modules({}, windows)
        assert len(result.modules) == 0

    def test_single_layer(self):
        windows = _make_windows()
        features = {"microbiome": _make_features_by_layer(windows)["microbiome"]}
        result = discover_modules(features, windows, n_modules=2)
        assert len(result.modules) > 0


class TestComputeModuleScores:
    def test_basic(self):
        windows = _make_windows()
        features = _make_features_by_layer(windows)
        result = discover_modules(features, windows, n_modules=3)

        # Re-compute scores
        all_feats = []
        X_blocks = []
        for layer_name in sorted(features.keys()):
            all_feats.extend(features[layer_name].columns.tolist())
            X_blocks.append(features[layer_name].values.astype(float))

        X = np.hstack(X_blocks)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            X[m, j] = medians[j] if np.isfinite(medians[j]) else 0.0
        X_scaled = scaler.fit_transform(X)

        scores = compute_module_scores(result.modules, X_scaled, all_feats)
        assert scores.shape[0] == len(windows)


class TestCompareWithMOFA:
    def test_with_mofa_result(self):
        windows = _make_windows()
        features = _make_features_by_layer(windows)
        result = discover_modules(features, windows, n_modules=3)

        # Simulate MOFA result
        mofa_result = {
            "factors": np.random.randn(len(windows), 3),
            "weights": {
                "microbiome": np.random.randn(8, 3),
                "metabolomics": np.random.randn(6, 3),
            },
            "view_names": ["microbiome", "metabolomics"],
            "n_factors": 3,
        }

        comparison = compare_with_mofa(result, mofa_result)
        assert not comparison.empty
        assert "jaccard_similarity" in comparison.columns
        assert "module_id" in comparison.columns
        assert "mofa_factor" in comparison.columns

    def test_without_mofa(self):
        windows = _make_windows()
        features = _make_features_by_layer(windows)
        result = discover_modules(features, windows, n_modules=3)

        comparison = compare_with_mofa(result, None)
        assert comparison.empty


class TestSaveModuleResults:
    def test_save(self, tmp_path):
        windows = _make_windows()
        features = _make_features_by_layer(windows)
        result = discover_modules(features, windows, n_modules=3)

        save_module_results(result, None, tmp_path)
        assert (tmp_path / "module_scores.csv").exists()
        assert (tmp_path / "module_feature_memberships.csv").exists()
        assert (tmp_path / "module_summary.csv").exists()

    def test_save_with_comparison(self, tmp_path):
        windows = _make_windows()
        features = _make_features_by_layer(windows)
        result = discover_modules(features, windows, n_modules=3)

        comparison = pd.DataFrame(
            [{"module_id": 0, "mofa_factor": 0, "jaccard_similarity": 0.3}]
        )
        save_module_results(result, comparison, tmp_path)
        assert (tmp_path / "module_mofa_comparison.csv").exists()
