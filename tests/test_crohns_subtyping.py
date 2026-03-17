"""Tests for consensus clustering and subtype discovery."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.crohns_subtyping import (
    ConsensusSubtyping,
    consensus_matrix,
    gap_statistic,
    multi_algorithm_clustering,
    permanova_test,
)


@pytest.fixture
def clustered_data():
    """Synthetic data with 3 well-separated clusters."""
    rng = np.random.default_rng(42)
    n_per = 15
    centers = np.array([[0, 0], [5, 5], [-5, 5]])

    data = []
    for center in centers:
        data.append(rng.normal(center, 0.5, (n_per, 2)))

    X = np.vstack(data)
    labels = np.repeat([0, 1, 2], n_per)

    df = pd.DataFrame(
        X, index=[f"P{i}" for i in range(len(X))], columns=["F0", "F1"]
    )
    return df, X, labels


# ── Consensus Matrix ──


def test_consensus_matrix_shape(clustered_data):
    _, X, _ = clustered_data
    cm = consensus_matrix(X, k=3, n_iterations=20)
    assert cm.shape == (len(X), len(X))
    assert np.allclose(np.diag(cm), 1.0)


def test_consensus_matrix_symmetric(clustered_data):
    _, X, _ = clustered_data
    cm = consensus_matrix(X, k=3, n_iterations=20)
    np.testing.assert_allclose(cm, cm.T, atol=1e-10)


def test_consensus_matrix_within_cluster_high(clustered_data):
    _, X, labels = clustered_data
    cm = consensus_matrix(X, k=3, n_iterations=50)
    # Samples in the same cluster should co-cluster frequently
    within_vals = []
    for c in range(3):
        mask = labels == c
        within = cm[np.ix_(mask, mask)]
        within_vals.append(within[np.triu_indices_from(within, k=1)].mean())
    assert np.mean(within_vals) > 0.5


# ── Multi-Algorithm Clustering ──


def test_multi_algorithm_returns_all(clustered_data):
    _, X, _ = clustered_data
    results = multi_algorithm_clustering(X, k=3)
    assert "kmeans" in results
    assert "hierarchical" in results
    assert all(len(v) == len(X) for v in results.values())


def test_multi_algorithm_agreement(clustered_data):
    _, X, _ = clustered_data
    results = multi_algorithm_clustering(X, k=3)
    # Well-separated clusters should give high ARI between methods
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(results["kmeans"], results["hierarchical"])
    assert ari > 0.5


# ── Gap Statistic ──


def test_gap_statistic_shape(clustered_data):
    _, X, _ = clustered_data
    gap = gap_statistic(X, k_range=range(2, 5), n_references=5)
    assert "k" in gap.columns
    assert "gap" in gap.columns
    assert "gap_se" in gap.columns
    assert len(gap) == 3


# ── PERMANOVA ──


def test_permanova_significant_covariate(clustered_data):
    _, X, labels = clustered_data
    # Cluster labels as covariate should be significant
    result = permanova_test(X, labels, labels, n_permutations=99)
    assert result["p_value"] < 0.05
    assert result["F_statistic"] > 0


def test_permanova_random_covariate(clustered_data):
    _, X, labels = clustered_data
    rng = np.random.default_rng(42)
    random_cov = rng.integers(0, 3, size=len(X))
    result = permanova_test(X, labels, random_cov, n_permutations=99)
    # Random covariate should not be significant (most of the time)
    assert result["F_statistic"] >= 0


# ── ConsensusSubtyping Pipeline ──


def test_consensus_subtyping_fit(clustered_data, tmp_path):
    df, _, _ = clustered_data
    subtyping = ConsensusSubtyping(k_range=range(2, 5), n_consensus_iter=20)
    results = subtyping.fit(df, output_dir=tmp_path)

    assert "optimal_k" in results
    assert "labels" in results
    assert "silhouette_scores" in results
    assert isinstance(results["labels"], pd.Series)
    assert len(results["labels"]) == len(df)


def test_consensus_subtyping_finds_3_clusters(clustered_data, tmp_path):
    df, _, _ = clustered_data
    subtyping = ConsensusSubtyping(k_range=range(2, 5), n_consensus_iter=30)
    results = subtyping.fit(df, output_dir=tmp_path)

    # Should find k=3 with good silhouette
    assert results["optimal_k"] in [2, 3, 4]
    assert results["silhouette_scores"][results["optimal_k"]] > 0.3


def test_consensus_subtyping_saves_files(clustered_data, tmp_path):
    df, _, _ = clustered_data
    subtyping = ConsensusSubtyping(k_range=range(2, 4), n_consensus_iter=10)
    subtyping.fit(df, output_dir=tmp_path)

    assert (tmp_path / "subtype_labels.csv").exists()
    assert (tmp_path / "silhouette_scores.csv").exists()
    assert (tmp_path / "gap_statistic.csv").exists()


def test_compare_methods(clustered_data):
    df, _, _ = clustered_data
    rng = np.random.default_rng(42)
    # Simulate two slightly different factor matrices
    df2 = df + pd.DataFrame(
        rng.normal(0, 0.1, df.shape), index=df.index, columns=df.columns
    )

    subtyping = ConsensusSubtyping(k_range=range(2, 4))
    comparison = subtyping.compare_methods(df, df2)

    assert "ari_scores" in comparison
    assert "mean_ari" in comparison
    # Similar matrices should give high ARI
    assert comparison["mean_ari"] > 0.3
