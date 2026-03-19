"""Tests for FP clustering pipeline."""

import numpy as np

from bioagentics.diagnostics.fp_mining.adapters.sepsis_adapter import MockSepsisAdapter
from bioagentics.diagnostics.fp_mining.cluster import (
    cluster_dbscan,
    cluster_kmeans,
    compute_cluster_profiles,
    find_optimal_k,
    prepare_features,
    reduce_for_visualization,
    run_clustering,
)
from bioagentics.diagnostics.fp_mining.extract import extract_at_operating_points


def _get_result():
    adapter = MockSepsisAdapter(n_admissions=500, seed=42)
    results = extract_at_operating_points(adapter, specificities=[0.90])
    return results[0]


class TestPrepareFeatures:
    def test_output_shape(self) -> None:
        result = _get_result()
        X, names, scaler = prepare_features(result.false_positives)
        assert X.shape[0] == len(result.false_positives)
        assert X.shape[1] == len(names)
        assert len(names) > 0

    def test_scaled(self) -> None:
        result = _get_result()
        X, _, _ = prepare_features(result.false_positives)
        # Standardized: mean ~0, std ~1
        assert abs(X.mean()) < 0.1
        assert abs(X.std() - 1.0) < 0.2


class TestFindOptimalK:
    def test_returns_valid_k(self) -> None:
        result = _get_result()
        X, _, _ = prepare_features(result.false_positives)
        best_k, scores = find_optimal_k(X)
        assert best_k >= 2
        assert len(scores) > 0


class TestClusterKmeans:
    def test_correct_n_clusters(self) -> None:
        result = _get_result()
        X, _, _ = prepare_features(result.false_positives)
        labels, model = cluster_kmeans(X, n_clusters=3)
        assert len(set(labels)) == 3
        assert len(labels) == X.shape[0]


class TestClusterDbscan:
    def test_returns_labels(self) -> None:
        result = _get_result()
        X, _, _ = prepare_features(result.false_positives)
        labels = cluster_dbscan(X)
        assert len(labels) == X.shape[0]


class TestComputeClusterProfiles:
    def test_profiles_per_cluster(self) -> None:
        result = _get_result()
        fp = result.false_positives
        X, names, _ = prepare_features(fp)
        labels, _ = cluster_kmeans(X, n_clusters=3)
        profiles = compute_cluster_profiles(fp, labels, names)
        assert len(profiles) == 3
        assert "cluster_id" in profiles.columns
        assert "n_samples" in profiles.columns


class TestReduceForVisualization:
    def test_output_2d(self) -> None:
        result = _get_result()
        X, _, _ = prepare_features(result.false_positives)
        X_2d = reduce_for_visualization(X)
        assert X_2d.shape[1] == 2
        assert X_2d.shape[0] == X.shape[0]


class TestRunClustering:
    def test_full_pipeline(self, tmp_path) -> None:
        result = _get_result()
        output = run_clustering(result, output_dir=tmp_path)
        assert "summary" in output
        assert "assignments" in output
        assert "profiles" in output
        assert output["summary"]["kmeans_k"] >= 2

    def test_saves_files(self, tmp_path) -> None:
        result = _get_result()
        run_clustering(result, output_dir=tmp_path)
        parquets = list(tmp_path.glob("*.parquet"))
        csvs = list(tmp_path.glob("*.csv"))
        assert len(parquets) == 1
        assert len(csvs) == 1
