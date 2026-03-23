"""Tests for the benchmark runner module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioagentics.diagnostics.rare_disease.benchmark_runner import (
    BenchmarkReport,
    BenchmarkResult,
    REFERENCE_TARGETS,
    make_ensemble_rank_fn,
    make_ic_rank_fn,
    run_benchmark,
)
from bioagentics.diagnostics.rare_disease.evaluation import (
    BenchmarkCase,
    EvalMetrics,
    RankFn,
)


def _mock_rank_fn(rankings: list[tuple[str, float]]) -> RankFn:
    """Create a mock RankFn that always returns the same rankings."""

    def fn(query_hpo_terms: list[str]) -> list[tuple[str, float]]:
        return rankings

    return fn


PERFECT_RANKINGS = [("D1", 0.9), ("D2", 0.7), ("D3", 0.5)]
POOR_RANKINGS = [("D3", 0.9), ("D2", 0.7), ("D1", 0.5)]

TEST_CASES = [
    BenchmarkCase(case_id="c1", query_hpo_terms=["HP:001"], true_disease_id="D1"),
    BenchmarkCase(case_id="c2", query_hpo_terms=["HP:002"], true_disease_id="D2"),
    BenchmarkCase(case_id="c3", query_hpo_terms=["HP:003"], true_disease_id="D3"),
]


class TestRunBenchmark:
    def test_single_matcher(self, tmp_path: Path):
        matchers = {"perfect": _mock_rank_fn(PERFECT_RANKINGS)}
        report = run_benchmark(
            matchers, TEST_CASES, output_dir=tmp_path, save=False
        )
        assert len(report.results) == 1
        assert report.results[0].matcher_name == "perfect"
        assert report.n_total_cases == 3

    def test_multiple_matchers(self, tmp_path: Path):
        matchers = {
            "perfect": _mock_rank_fn(PERFECT_RANKINGS),
            "poor": _mock_rank_fn(POOR_RANKINGS),
        }
        report = run_benchmark(
            matchers, TEST_CASES, output_dir=tmp_path, save=False
        )
        assert len(report.results) == 2

    def test_perfect_matcher_metrics(self, tmp_path: Path):
        matchers = {"perfect": _mock_rank_fn(PERFECT_RANKINGS)}
        report = run_benchmark(
            matchers, TEST_CASES, output_dir=tmp_path, save=False
        )
        m = report.results[0].metrics
        # D1 rank=1 for c1, D2 rank=2 for c2, D3 rank=3 for c3
        assert m.top1_accuracy == pytest.approx(1 / 3)  # only D1 is rank 1
        assert m.top5_accuracy == pytest.approx(1.0)  # all within top 5

    def test_saves_report_json(self, tmp_path: Path):
        matchers = {"test": _mock_rank_fn(PERFECT_RANKINGS)}
        report = run_benchmark(
            matchers, TEST_CASES,
            benchmark_name="test_bench",
            output_dir=tmp_path,
            save=True,
        )
        report_path = tmp_path / "test_bench_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["benchmark"] == "test_bench"
        assert len(data["results"]) == 1

    def test_saves_per_matcher_results(self, tmp_path: Path):
        matchers = {"mymodel": _mock_rank_fn(PERFECT_RANKINGS)}
        run_benchmark(
            matchers, TEST_CASES,
            benchmark_name="bench",
            output_dir=tmp_path,
            save=True,
        )
        eval_path = tmp_path / "bench_mymodel_eval.json"
        assert eval_path.exists()


class TestBenchmarkReport:
    def test_comparison_table(self):
        report = BenchmarkReport(
            results=[
                BenchmarkResult(
                    "model_a",
                    EvalMetrics(n_cases=10, top1_accuracy=0.5, top5_accuracy=0.7,
                                top10_accuracy=0.8, top20_accuracy=0.9, mrr=0.6,
                                median_rank=2, mean_rank=3),
                ),
                BenchmarkResult(
                    "model_b",
                    EvalMetrics(n_cases=10, top1_accuracy=0.3, top5_accuracy=0.5,
                                top10_accuracy=0.6, top20_accuracy=0.7, mrr=0.4,
                                median_rank=4, mean_rank=5),
                ),
            ],
            n_total_cases=10,
        )
        table = report.comparison_table()
        assert "model_a" in table
        assert "model_b" in table

    def test_to_dict(self):
        report = BenchmarkReport(
            benchmark_name="test",
            n_total_cases=5,
            results=[
                BenchmarkResult("m1", EvalMetrics(n_cases=5, mrr=0.5)),
            ],
        )
        d = report.to_dict()
        assert d["benchmark"] == "test"
        assert d["n_cases"] == 5
        assert len(d["results"]) == 1
        assert d["results"][0]["matcher"] == "m1"

    def test_to_dict_includes_reference_targets(self):
        report = BenchmarkReport(
            benchmark_name="test",
            n_total_cases=5,
            results=[BenchmarkResult("m1", EvalMetrics(n_cases=5))],
        )
        d = report.to_dict()
        assert "reference_targets" in d
        assert "DeepRare (HPO-only)" in d["reference_targets"]
        assert d["reference_targets"]["DeepRare (HPO-only)"]["recall_at_1"] == 0.644
        assert "PhenoBrain (standalone)" in d["reference_targets"]
        assert d["reference_targets"]["PhenoBrain (standalone)"]["top10_recall"] == 0.654

    def test_comparison_table_includes_references(self):
        report = BenchmarkReport(
            results=[
                BenchmarkResult("m1", EvalMetrics(n_cases=5, top1_accuracy=0.5)),
            ],
            n_total_cases=5,
        )
        table = report.comparison_table()
        assert "DeepRare" in table
        assert "PhenoBrain" in table
        assert "64.4%" in table


class TestMakeEnsembleRankFn:
    def test_combines_multiple_matchers(self):
        from bioagentics.diagnostics.rare_disease.ensemble_matcher import EnsembleMatcher

        matcher_fns = {
            "a": _mock_rank_fn([("D1", 0.8), ("D2", 0.2)]),
            "b": _mock_rank_fn([("D1", 0.6), ("D2", 0.4)]),
        }
        ensemble = EnsembleMatcher(model_names=["a", "b"])
        rank_fn = make_ensemble_rank_fn(ensemble, matcher_fns)

        results = rank_fn(["HP:001"])
        assert len(results) == 2
        # D1 should score higher than D2 (both matchers prefer D1)
        d1_score = next(s for d, s in results if d == "D1")
        d2_score = next(s for d, s in results if d == "D2")
        assert d1_score > d2_score


class TestMakeICRankFn:
    def test_adapter_returns_ranked_tuples(self):
        import networkx as nx
        from bioagentics.diagnostics.rare_disease.ic_matcher import ICScorer

        dag = nx.DiGraph()
        dag.add_edge("HP:001", "HP:000")
        dag.add_edge("HP:002", "HP:000")
        dag.add_edge("HP:003", "HP:000")

        scorer = ICScorer(hpo_dag=dag)
        annotations = {"D1": ["HP:001", "HP:002"], "D2": ["HP:003"]}
        scorer.compute_ic(annotations)

        rank_fn = make_ic_rank_fn(scorer, annotations, method="resnik")
        results = rank_fn(["HP:001"])
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
