"""Tests for the evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioagentics.diagnostics.rare_disease.evaluation import (
    CaseResult,
    EvalMetrics,
    BenchmarkCase,
    compare_models,
    compute_metrics,
    evaluate_case,
    evaluate_matcher,
    save_results,
)


def _make_rank_fn(rankings: dict[str, list[tuple[str, float]]]):
    """Create a mock rank function from pre-defined rankings."""

    def rank_fn(query_hpo_terms: list[str]) -> list[tuple[str, float]]:
        # Use first query term as key to look up ranking
        key = query_hpo_terms[0] if query_hpo_terms else ""
        return rankings.get(key, [])

    return rank_fn


# Pre-defined rankings for testing
MOCK_RANKINGS = {
    # Query term "HP:001" → disease D1 is rank 1
    "HP:001": [("D1", 0.9), ("D2", 0.7), ("D3", 0.5), ("D4", 0.3)],
    # Query term "HP:002" → disease D2 is rank 3
    "HP:002": [("D3", 0.8), ("D4", 0.6), ("D2", 0.4), ("D1", 0.2)],
    # Query term "HP:003" → disease D3 is rank 1
    "HP:003": [("D3", 0.95), ("D1", 0.5), ("D2", 0.3), ("D4", 0.1)],
}


@pytest.fixture
def rank_fn():
    return _make_rank_fn(MOCK_RANKINGS)


@pytest.fixture
def test_cases():
    return [
        BenchmarkCase(case_id="case1", query_hpo_terms=["HP:001"], true_disease_id="D1"),
        BenchmarkCase(case_id="case2", query_hpo_terms=["HP:002"], true_disease_id="D2"),
        BenchmarkCase(case_id="case3", query_hpo_terms=["HP:003"], true_disease_id="D3"),
    ]


class TestEvaluateCase:
    def test_correct_rank_1(self, rank_fn):
        case = BenchmarkCase(case_id="c1", query_hpo_terms=["HP:001"], true_disease_id="D1")
        result = evaluate_case(rank_fn, case)
        assert result.predicted_rank == 1

    def test_correct_rank_3(self, rank_fn):
        case = BenchmarkCase(case_id="c2", query_hpo_terms=["HP:002"], true_disease_id="D2")
        result = evaluate_case(rank_fn, case)
        assert result.predicted_rank == 3

    def test_disease_not_found(self, rank_fn):
        case = BenchmarkCase(case_id="c3", query_hpo_terms=["HP:001"], true_disease_id="D_MISSING")
        result = evaluate_case(rank_fn, case)
        assert result.predicted_rank == 0

    def test_top_predictions_correct(self, rank_fn):
        case = BenchmarkCase(case_id="c1", query_hpo_terms=["HP:001"], true_disease_id="D1")
        result = evaluate_case(rank_fn, case)
        assert result.top_predictions == ["D1", "D2", "D3", "D4"]

    def test_score_captured(self, rank_fn):
        case = BenchmarkCase(case_id="c1", query_hpo_terms=["HP:001"], true_disease_id="D1")
        result = evaluate_case(rank_fn, case)
        assert result.true_disease_score == 0.9


class TestComputeMetrics:
    def test_perfect_ranking(self):
        results = [
            CaseResult("c1", "D1", 1, ["D1"], 1.0),
            CaseResult("c2", "D2", 1, ["D2"], 1.0),
            CaseResult("c3", "D3", 1, ["D3"], 1.0),
        ]
        m = compute_metrics(results)
        assert m.top1_accuracy == pytest.approx(1.0)
        assert m.top5_accuracy == pytest.approx(1.0)
        assert m.top10_accuracy == pytest.approx(1.0)
        assert m.mrr == pytest.approx(1.0)

    def test_mixed_ranking(self):
        results = [
            CaseResult("c1", "D1", 1, ["D1"], 0.9),  # rank 1
            CaseResult("c2", "D2", 3, ["D3", "D4", "D2"], 0.4),  # rank 3
            CaseResult("c3", "D3", 8, ["D1"] * 7 + ["D3"], 0.1),  # rank 8
        ]
        m = compute_metrics(results)
        assert m.n_cases == 3
        assert m.top1_accuracy == pytest.approx(1 / 3)  # only c1
        assert m.top5_accuracy == pytest.approx(2 / 3)  # c1, c2
        assert m.top10_accuracy == pytest.approx(3 / 3)  # all
        # MRR = (1/1 + 1/3 + 1/8) / 3
        expected_mrr = (1.0 + 1 / 3 + 1 / 8) / 3
        assert m.mrr == pytest.approx(expected_mrr)

    def test_not_found_cases(self):
        results = [
            CaseResult("c1", "D1", 1, ["D1"], 0.9),
            CaseResult("c2", "D2", 0, [], 0.0),  # not found
        ]
        m = compute_metrics(results)
        assert m.top1_accuracy == pytest.approx(0.5)
        assert m.mrr == pytest.approx(0.5)  # (1/1 + 0) / 2

    def test_empty_results(self):
        m = compute_metrics([])
        assert m.n_cases == 0
        assert m.mrr == 0.0

    def test_median_rank(self):
        results = [
            CaseResult("c1", "D1", 1, [], 0),
            CaseResult("c2", "D2", 5, [], 0),
            CaseResult("c3", "D3", 10, [], 0),
        ]
        m = compute_metrics(results)
        assert m.median_rank == 5

    def test_mean_rank(self):
        results = [
            CaseResult("c1", "D1", 2, [], 0),
            CaseResult("c2", "D2", 4, [], 0),
            CaseResult("c3", "D3", 6, [], 0),
        ]
        m = compute_metrics(results)
        assert m.mean_rank == pytest.approx(4.0)


class TestEvaluateMatcher:
    def test_returns_metrics_and_results(self, rank_fn, test_cases):
        metrics, results = evaluate_matcher(rank_fn, test_cases, name="test")
        assert isinstance(metrics, EvalMetrics)
        assert len(results) == 3

    def test_metrics_computed_correctly(self, rank_fn, test_cases):
        metrics, _ = evaluate_matcher(rank_fn, test_cases)
        # case1: rank 1, case2: rank 3, case3: rank 1
        assert metrics.top1_accuracy == pytest.approx(2 / 3)
        assert metrics.top5_accuracy == pytest.approx(3 / 3)


class TestSaveResults:
    def test_saves_json(self, tmp_path: Path):
        metrics = EvalMetrics(n_cases=1, top1_accuracy=1.0, mrr=1.0)
        results = [CaseResult("c1", "D1", 1, ["D1"], 0.9)]
        path = save_results(metrics, results, "test_model", output_dir=tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["model"] == "test_model"
        assert data["metrics"]["top1_accuracy"] == 1.0
        assert len(data["cases"]) == 1


class TestCompareModels:
    def test_format_output(self):
        models = {
            "ic_resnik": EvalMetrics(n_cases=100, top1_accuracy=0.4, top5_accuracy=0.6,
                                     top10_accuracy=0.7, top20_accuracy=0.8, mrr=0.5,
                                     median_rank=3, mean_rank=5),
            "ic_lin": EvalMetrics(n_cases=100, top1_accuracy=0.35, top5_accuracy=0.55,
                                  top10_accuracy=0.65, top20_accuracy=0.75, mrr=0.45,
                                  median_rank=4, mean_rank=6),
        }
        table = compare_models(models)
        assert "ic_resnik" in table
        assert "ic_lin" in table
        assert "Top-1" in table
