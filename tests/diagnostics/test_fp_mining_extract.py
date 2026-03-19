"""Tests for false positive extraction framework."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.diagnostics.fp_mining.extract import (
    ExtractionResult,
    OperatingPoint,
    extract_at_operating_points,
    extract_false_positives,
    find_threshold_at_specificity,
    get_feature_columns,
)


def make_predictions(n_neg: int = 200, n_pos: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create synthetic prediction data."""
    rng = np.random.default_rng(seed)
    neg_scores = rng.normal(0.3, 0.15, n_neg).clip(0, 1)
    pos_scores = rng.normal(0.7, 0.15, n_pos).clip(0, 1)

    scores = np.concatenate([neg_scores, pos_scores])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

    features = rng.standard_normal((n_neg + n_pos, 3))

    return pd.DataFrame({
        "sample_id": [f"S{i:04d}" for i in range(n_neg + n_pos)],
        "y_true": labels.astype(int),
        "y_score": scores,
        "feat_a": features[:, 0],
        "feat_b": features[:, 1],
        "feat_c": features[:, 2],
    })


class TestFindThresholdAtSpecificity:
    def test_basic_specificity_95(self) -> None:
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 1000 + [1] * 200)
        y_score = np.concatenate([
            rng.normal(0.3, 0.1, 1000),
            rng.normal(0.7, 0.1, 200),
        ])

        threshold, spec, sens = find_threshold_at_specificity(y_true, y_score, 0.95)
        assert 0.93 <= spec <= 0.97
        assert 0.0 < threshold < 1.0
        assert 0.0 <= sens <= 1.0

    def test_perfect_separation(self) -> None:
        y_true = np.array([0] * 100 + [1] * 100)
        y_score = np.concatenate([
            np.linspace(0.0, 0.4, 100),
            np.linspace(0.6, 1.0, 100),
        ])

        threshold, spec, sens = find_threshold_at_specificity(y_true, y_score, 0.90)
        assert spec >= 0.90
        assert sens > 0.5

    def test_no_negatives_raises(self) -> None:
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7])
        with pytest.raises(ValueError, match="both positive and negative"):
            find_threshold_at_specificity(y_true, y_score, 0.95)

    def test_no_positives_raises(self) -> None:
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="both positive and negative"):
            find_threshold_at_specificity(y_true, y_score, 0.95)


class TestExtractFalsePositives:
    def test_correct_classification(self) -> None:
        df = pd.DataFrame({
            "sample_id": ["A", "B", "C", "D"],
            "y_true": [0, 0, 1, 1],
            "y_score": [0.8, 0.2, 0.9, 0.1],
            "feat": [1.0, 2.0, 3.0, 4.0],
        })

        fp, tn, tp, fn = extract_false_positives(df, threshold=0.5)
        assert list(fp["sample_id"]) == ["A"]  # FP: predicted pos, true neg
        assert list(tn["sample_id"]) == ["B"]  # TN: predicted neg, true neg
        assert list(tp["sample_id"]) == ["C"]  # TP: predicted pos, true pos
        assert list(fn["sample_id"]) == ["D"]  # FN: predicted neg, true pos

    def test_all_negative_predictions(self) -> None:
        df = pd.DataFrame({
            "sample_id": ["A", "B"],
            "y_true": [0, 1],
            "y_score": [0.1, 0.2],
        })
        fp, tn, tp, fn = extract_false_positives(df, threshold=0.5)
        assert len(fp) == 0
        assert len(tn) == 1
        assert len(tp) == 0
        assert len(fn) == 1

    def test_preserves_features(self) -> None:
        df = pd.DataFrame({
            "sample_id": ["A"],
            "y_true": [0],
            "y_score": [0.9],
            "feat_x": [42.0],
        })
        fp, _, _, _ = extract_false_positives(df, threshold=0.5)
        assert "feat_x" in fp.columns
        assert fp.iloc[0]["feat_x"] == 42.0


class TestGetFeatureColumns:
    def test_excludes_metadata(self) -> None:
        df = pd.DataFrame({
            "sample_id": ["A"],
            "y_true": [0],
            "y_score": [0.5],
            "feat_a": [1.0],
            "feat_b": [2.0],
        })
        feats = get_feature_columns(df)
        assert feats == ["feat_a", "feat_b"]

    def test_custom_exclude(self) -> None:
        df = pd.DataFrame({
            "sample_id": ["A"],
            "y_true": [0],
            "y_score": [0.5],
            "feat_a": [1.0],
            "batch": ["X"],
        })
        feats = get_feature_columns(df, exclude={"batch"})
        assert feats == ["feat_a"]


class TestExtractionResult:
    def test_summary_computed(self) -> None:
        op = OperatingPoint("test", 0.5, 0.95, 0.80)
        fp = pd.DataFrame({"sample_id": ["A"]})
        tn = pd.DataFrame({"sample_id": ["B", "C"]})
        tp = pd.DataFrame({"sample_id": ["D"]})
        fn = pd.DataFrame({"sample_id": ["E"]})

        result = ExtractionResult("sepsis", op, fp, tn, tp, fn)
        assert result.summary["n_fp"] == 1
        assert result.summary["n_tn"] == 2
        assert result.summary["n_total"] == 5
        assert abs(result.summary["fp_rate"] - 1 / 3) < 1e-6


class MockSource:
    """Mock prediction source for testing."""

    domain = "test"

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def load_predictions(self) -> pd.DataFrame:
        return self._df


class TestExtractAtOperatingPoints:
    def test_multiple_operating_points(self) -> None:
        df = make_predictions()
        source = MockSource(df)
        results = extract_at_operating_points(source, specificities=[0.90, 0.95])
        assert len(results) == 2
        # Higher specificity -> fewer FPs
        assert results[1].summary["n_fp"] <= results[0].summary["n_fp"]

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        source = MockSource(df)
        with pytest.raises(ValueError, match="missing required columns"):
            extract_at_operating_points(source)

    def test_counts_are_consistent(self) -> None:
        df = make_predictions()
        source = MockSource(df)
        results = extract_at_operating_points(source, specificities=[0.95])
        r = results[0]
        total = len(r.false_positives) + len(r.true_negatives) + len(r.true_positives) + len(r.false_negatives)
        assert total == len(df)
