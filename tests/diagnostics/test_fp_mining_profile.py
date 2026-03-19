"""Tests for FP profiling module."""

import pandas as pd

from bioagentics.diagnostics.fp_mining.adapters.sepsis_adapter import MockSepsisAdapter
from bioagentics.diagnostics.fp_mining.extract import extract_at_operating_points
from bioagentics.diagnostics.fp_mining.profile import (
    compare_fp_vs_tn,
    compute_distribution_stats,
    profile_confidence_scores,
    run_profiling,
)


def _get_result():
    """Helper: extract at 90% specificity from mock sepsis data."""
    adapter = MockSepsisAdapter(n_admissions=500, seed=42)
    results = extract_at_operating_points(adapter, specificities=[0.90])
    return results[0]


class TestComputeDistributionStats:
    def test_basic_stats(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_distribution_stats(s)
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["n"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0


class TestCompareFpVsTn:
    def test_returns_per_feature_comparison(self) -> None:
        result = _get_result()
        comparison = compare_fp_vs_tn(result)
        assert len(comparison) > 0
        assert "feature" in comparison.columns
        assert "t_pvalue" in comparison.columns
        assert "cohens_d" in comparison.columns

    def test_sorted_by_pvalue(self) -> None:
        result = _get_result()
        comparison = compare_fp_vs_tn(result)
        if len(comparison) > 1:
            pvals = comparison["t_pvalue"].values
            assert all(pvals[i] <= pvals[i + 1] for i in range(len(pvals) - 1))


class TestProfileConfidenceScores:
    def test_all_groups_present(self) -> None:
        result = _get_result()
        profiles = profile_confidence_scores(result)
        assert "false_positives" in profiles
        assert "true_negatives" in profiles
        assert "true_positives" in profiles
        assert "false_negatives" in profiles

    def test_fp_scores_higher_than_tn(self) -> None:
        result = _get_result()
        profiles = profile_confidence_scores(result)
        # FPs have higher scores than TNs by definition (above threshold)
        assert profiles["false_positives"]["mean"] > profiles["true_negatives"]["mean"]


class TestRunProfiling:
    def test_full_pipeline(self, tmp_path) -> None:
        result = _get_result()
        output = run_profiling(result, output_dir=tmp_path)
        assert "comparison" in output
        assert "confidence" in output
        assert "summary" in output
        assert output["summary"]["n_fp"] > 0

    def test_saves_csv(self, tmp_path) -> None:
        result = _get_result()
        run_profiling(result, output_dir=tmp_path)
        csvs = list(tmp_path.glob("*.csv"))
        assert len(csvs) == 1
