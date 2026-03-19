"""Tests for the ensemble matcher combining IC, node2vec, and GAT scores."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.rare_disease.ensemble_matcher import (
    EnsembleConfig,
    EnsembleMatcher,
    rank_diseases,
)


@pytest.fixture
def training_data() -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic training data where model_0 is most predictive.

    For positive samples (label=1): model_0 scores high, others moderate.
    For negative samples (label=0): all scores low/mixed.
    """
    rng = np.random.RandomState(42)
    n_pos = 50
    n_neg = 150

    # Positive: model_0=high, model_1=moderate, model_2=moderate
    pos_scores = np.column_stack([
        rng.uniform(0.7, 1.0, n_pos),  # model_0: strong predictor
        rng.uniform(0.3, 0.7, n_pos),  # model_1: moderate
        rng.uniform(0.3, 0.7, n_pos),  # model_2: moderate
    ])
    # Negative: all low-to-moderate
    neg_scores = np.column_stack([
        rng.uniform(0.0, 0.4, n_neg),  # model_0: low
        rng.uniform(0.2, 0.6, n_neg),  # model_1: mixed
        rng.uniform(0.2, 0.6, n_neg),  # model_2: mixed
    ])

    scores = np.vstack([pos_scores, neg_scores])
    labels = np.array([1] * n_pos + [0] * n_neg)

    return scores, labels


@pytest.fixture
def trained_ensemble(training_data: tuple[np.ndarray, np.ndarray]) -> EnsembleMatcher:
    scores, labels = training_data
    matcher = EnsembleMatcher(model_names=["ic", "node2vec", "gat"])
    matcher.train(scores, labels)
    return matcher


class TestTraining:
    def test_train_sets_weights(self, trained_ensemble: EnsembleMatcher):
        assert trained_ensemble.trained
        assert trained_ensemble.weights is not None
        assert len(trained_ensemble.weights) == 3

    def test_best_model_gets_highest_weight(self, trained_ensemble: EnsembleMatcher):
        """model_0 (ic) is most predictive, should get highest weight."""
        weights = trained_ensemble.weights
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]

    def test_training_returns_weight_map(self, training_data: tuple[np.ndarray, np.ndarray]):
        scores, labels = training_data
        matcher = EnsembleMatcher(model_names=["ic", "node2vec", "gat"])
        weight_map = matcher.train(scores, labels)
        assert "ic" in weight_map
        assert "node2vec" in weight_map
        assert "gat" in weight_map

    def test_mismatched_dimensions_raises(self):
        matcher = EnsembleMatcher(model_names=["a", "b"])
        scores = np.array([[0.5, 0.3, 0.7]])  # 3 models but 2 names
        labels = np.array([1])
        with pytest.raises(ValueError, match="model_names"):
            matcher.train(scores, labels)

    def test_mismatched_samples_raises(self):
        matcher = EnsembleMatcher(model_names=["a", "b"])
        scores = np.array([[0.5, 0.3], [0.6, 0.4]])
        labels = np.array([1, 0, 1])  # 3 labels but 2 samples
        with pytest.raises(ValueError, match="rows"):
            matcher.train(scores, labels)


class TestScoring:
    def test_trained_scores_high_for_positive_pattern(self, trained_ensemble: EnsembleMatcher):
        """Positive pattern (high ic, moderate others) should score high."""
        score = trained_ensemble.combine_scores(
            {"ic": 0.9, "node2vec": 0.5, "gat": 0.5}
        )
        assert score > 0.5

    def test_trained_scores_low_for_negative_pattern(self, trained_ensemble: EnsembleMatcher):
        """Negative pattern (low ic) should score low."""
        score = trained_ensemble.combine_scores(
            {"ic": 0.1, "node2vec": 0.4, "gat": 0.4}
        )
        assert score < 0.5

    def test_score_bounded_01(self, trained_ensemble: EnsembleMatcher):
        """Ensemble scores should be valid probabilities."""
        for ic in [0.0, 0.5, 1.0]:
            for n2v in [0.0, 0.5, 1.0]:
                for gat in [0.0, 0.5, 1.0]:
                    score = trained_ensemble.combine_scores(
                        {"ic": ic, "node2vec": n2v, "gat": gat}
                    )
                    assert 0.0 <= score <= 1.0

    def test_untrained_uses_uniform_average(self):
        """Without training, ensemble averages scores."""
        matcher = EnsembleMatcher(model_names=["ic", "node2vec", "gat"])
        score = matcher.combine_scores({"ic": 0.3, "node2vec": 0.6, "gat": 0.9})
        assert score == pytest.approx(0.6, abs=1e-6)

    def test_empty_model_names_averages(self):
        """Totally unconfigured matcher averages provided scores."""
        matcher = EnsembleMatcher()
        score = matcher.combine_scores({"a": 0.4, "b": 0.8})
        assert score == pytest.approx(0.6, abs=1e-6)


class TestRanking:
    def test_ranking_returns_all(self, trained_ensemble: EnsembleMatcher):
        disease_scores = {
            "D1": {"ic": 0.9, "node2vec": 0.5, "gat": 0.6},
            "D2": {"ic": 0.2, "node2vec": 0.3, "gat": 0.3},
            "D3": {"ic": 0.5, "node2vec": 0.4, "gat": 0.4},
        }
        results = rank_diseases(trained_ensemble, disease_scores)
        assert len(results) == 3

    def test_ranks_sequential(self, trained_ensemble: EnsembleMatcher):
        disease_scores = {
            "D1": {"ic": 0.9, "node2vec": 0.5, "gat": 0.6},
            "D2": {"ic": 0.2, "node2vec": 0.3, "gat": 0.3},
        }
        results = rank_diseases(trained_ensemble, disease_scores)
        assert [r.rank for r in results] == [1, 2]

    def test_scores_descending(self, trained_ensemble: EnsembleMatcher):
        disease_scores = {
            "D1": {"ic": 0.9, "node2vec": 0.5, "gat": 0.6},
            "D2": {"ic": 0.2, "node2vec": 0.3, "gat": 0.3},
            "D3": {"ic": 0.5, "node2vec": 0.4, "gat": 0.4},
        }
        results = rank_diseases(trained_ensemble, disease_scores)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ensemble_outperforms_individual(self, training_data: tuple[np.ndarray, np.ndarray]):
        """Ensemble should outperform each individual model on held-out data."""
        from sklearn.metrics import roc_auc_score

        scores, labels = training_data
        rng = np.random.RandomState(99)
        n = len(labels)
        idx = rng.permutation(n)
        split = int(0.6 * n)
        train_idx, test_idx = idx[:split], idx[split:]

        train_scores, train_labels = scores[train_idx], labels[train_idx]
        test_scores, test_labels = scores[test_idx], labels[test_idx]

        # Train ensemble
        matcher = EnsembleMatcher(model_names=["ic", "node2vec", "gat"])
        matcher.train(train_scores, train_labels)

        # Ensemble AUC
        ensemble_preds = []
        for row in test_scores:
            s = matcher.combine_scores(dict(zip(matcher.model_names, row)))
            ensemble_preds.append(s)
        ensemble_auc = roc_auc_score(test_labels, ensemble_preds)

        # Individual AUCs
        for i, name in enumerate(matcher.model_names):
            individual_auc = roc_auc_score(test_labels, test_scores[:, i])
            assert ensemble_auc >= individual_auc - 0.02, (
                f"Ensemble AUC {ensemble_auc:.3f} should be >= "
                f"{name} AUC {individual_auc:.3f} (within tolerance)"
            )

    def test_empty_disease_scores(self, trained_ensemble: EnsembleMatcher):
        results = rank_diseases(trained_ensemble, {})
        assert results == []
