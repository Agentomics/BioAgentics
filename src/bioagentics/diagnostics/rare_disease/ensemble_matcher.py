"""Ensemble matcher combining IC, node2vec, and GAT scores.

Learns optimal combination weights via logistic regression on a validation
set. Falls back to uniform averaging if no training data is provided.

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.ensemble_matcher
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from bioagentics.diagnostics.rare_disease.ic_matcher import RankResult

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble matcher."""

    regularization: float = 1.0
    normalize_scores: bool = True


@dataclass
class EnsembleMatcher:
    """Ensemble model combining multiple matcher scores.

    Learns weights via logistic regression on (score_vector, is_correct) pairs.
    If not trained, uses uniform weight averaging.

    Attributes:
        model_names: Names of the individual matchers.
        weights: Learned weights per model (after training).
        intercept: Learned intercept (after training).
        trained: Whether the ensemble has been trained.
    """

    model_names: list[str] = field(default_factory=list)
    weights: np.ndarray | None = None
    intercept: float = 0.0
    trained: bool = False
    _scaler_means: np.ndarray | None = None
    _scaler_stds: np.ndarray | None = None

    def train(
        self,
        score_matrix: np.ndarray,
        labels: np.ndarray,
        config: EnsembleConfig | None = None,
    ) -> dict[str, float]:
        """Train the ensemble weights from validation data.

        Args:
            score_matrix: Array of shape [n_samples, n_models] with scores
                from each individual matcher.
            labels: Binary array [n_samples] — 1 if the disease is the
                correct diagnosis, 0 otherwise.
            config: Ensemble configuration.

        Returns:
            Dict mapping model names to learned weights.
        """
        if config is None:
            config = EnsembleConfig()

        if score_matrix.shape[0] != len(labels):
            raise ValueError("score_matrix rows must match labels length")

        n_models = score_matrix.shape[1]
        if self.model_names and len(self.model_names) != n_models:
            raise ValueError("model_names length must match score_matrix columns")
        if not self.model_names:
            self.model_names = [f"model_{i}" for i in range(n_models)]

        # Normalize scores per model
        if config.normalize_scores:
            self._scaler_means = score_matrix.mean(axis=0)
            self._scaler_stds = score_matrix.std(axis=0)
            self._scaler_stds[self._scaler_stds == 0] = 1.0
            X = (score_matrix - self._scaler_means) / self._scaler_stds
        else:
            X = score_matrix

        # Train logistic regression
        lr = LogisticRegression(
            C=config.regularization,
            max_iter=1000,
            solver="lbfgs",
        )
        lr.fit(X, labels)

        self.weights = lr.coef_[0]
        self.intercept = lr.intercept_[0]
        self.trained = True

        weight_map = dict(zip(self.model_names, self.weights))
        logger.info("Ensemble weights: %s, intercept=%.4f", weight_map, self.intercept)
        return weight_map

    def combine_scores(self, scores: dict[str, float]) -> float:
        """Combine individual matcher scores into an ensemble score.

        Args:
            scores: Dict mapping model name to score for a single
                disease-query pair.

        Returns:
            Combined ensemble score.
        """
        if not self.model_names:
            # Fallback: uniform average
            vals = list(scores.values())
            return sum(vals) / len(vals) if vals else 0.0

        score_vec = np.array([scores.get(name, 0.0) for name in self.model_names])

        if self.trained and self.weights is not None:
            if self._scaler_means is not None and self._scaler_stds is not None:
                score_vec = (score_vec - self._scaler_means) / self._scaler_stds
            logit = float(np.dot(self.weights, score_vec) + self.intercept)
            # Sigmoid for probability
            return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
        else:
            # Uniform average fallback
            return float(score_vec.mean())


def rank_diseases(
    matcher: EnsembleMatcher,
    disease_scores: dict[str, dict[str, float]],
) -> list[RankResult]:
    """Rank diseases using the ensemble model.

    Args:
        matcher: Trained or untrained EnsembleMatcher.
        disease_scores: {disease_id: {model_name: score}}.

    Returns:
        List of RankResult sorted by ensemble score descending.
    """
    results: list[RankResult] = []

    for disease_id, scores in disease_scores.items():
        ensemble_score = matcher.combine_scores(scores)
        results.append(RankResult(disease_id=disease_id, score=ensemble_score))

    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results
