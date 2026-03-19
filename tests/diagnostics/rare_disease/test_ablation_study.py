"""Tests for the ablation study module."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from bioagentics.diagnostics.rare_disease.ablation_study import (
    AblationConfig,
    AblationPoint,
    AblationReport,
    run_ablation,
    run_single_dimension_ablation,
)
from bioagentics.diagnostics.rare_disease.evaluation import (
    BenchmarkCase,
    EvalMetrics,
    RankFn,
)


def _build_test_dag() -> nx.DiGraph:
    """Build a minimal HPO DAG for testing."""
    g = nx.DiGraph()
    terms = {
        "HP:0000001": "All",
        "HP:0000118": "Phenotypic abnormality",
        "HP:0000707": "Neuro",
        "HP:0000152": "Head/neck",
        "HP:0000234": "Head",
        "HP:0002011": "CNS",
        "HP:0012443": "Brain",
        "HP:0001250": "Seizure",
        "HP:0001249": "Intellectual disability",
        "HP:0000252": "Microcephaly",
    }
    for tid, name in terms.items():
        g.add_node(tid, name=name, node_type="phenotype")

    edges = [
        ("HP:0000118", "HP:0000001"),
        ("HP:0000707", "HP:0000118"),
        ("HP:0000152", "HP:0000118"),
        ("HP:0000234", "HP:0000152"),
        ("HP:0002011", "HP:0000707"),
        ("HP:0012443", "HP:0002011"),
        ("HP:0001250", "HP:0000707"),
        ("HP:0001249", "HP:0000707"),
        ("HP:0000252", "HP:0000234"),
    ]
    for child, parent in edges:
        g.add_edge(child, parent, relation="is_a")
    return g


DISEASE_ANNOTATIONS = {
    "OMIM:100001": ["HP:0000707", "HP:0002011", "HP:0012443", "HP:0001250", "HP:0001249"],
    "OMIM:100002": ["HP:0000234", "HP:0000252", "HP:0001249", "HP:0001250"],
}


def _mock_rank_fn(disease_scores: dict[str, float]) -> RankFn:
    """Create a mock RankFn with fixed scores per disease."""
    ranked = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)

    def fn(query_hpo_terms: list[str]) -> list[tuple[str, float]]:
        return ranked

    return fn


class TestAblationConfig:
    def test_defaults(self):
        config = AblationConfig()
        assert 0.2 in config.completeness_levels
        assert 0.8 in config.completeness_levels
        assert 0.0 in config.noise_levels
        assert 0.20 in config.noise_levels


class TestRunAblation:
    def test_generates_points_for_all_combos(self, tmp_path: Path):
        dag = _build_test_dag()
        config = AblationConfig(
            completeness_levels=[0.4, 0.8],
            noise_levels=[0.0, 0.1],
        )
        matchers = {"mock": _mock_rank_fn({"OMIM:100001": 0.9, "OMIM:100002": 0.5})}

        report = run_ablation(
            matchers, DISEASE_ANNOTATIONS, dag,
            config=config, output_dir=tmp_path, save=False,
        )

        # 2 completeness x 2 noise x 1 matcher = 4 points
        assert len(report.points) == 4

    def test_multiple_matchers(self, tmp_path: Path):
        dag = _build_test_dag()
        config = AblationConfig(
            completeness_levels=[0.6],
            noise_levels=[0.0],
        )
        matchers = {
            "good": _mock_rank_fn({"OMIM:100001": 0.9, "OMIM:100002": 0.5}),
            "bad": _mock_rank_fn({"OMIM:100002": 0.9, "OMIM:100001": 0.5}),
        }

        report = run_ablation(
            matchers, DISEASE_ANNOTATIONS, dag,
            config=config, output_dir=tmp_path, save=False,
        )

        # 1 completeness x 1 noise x 2 matchers = 2 points
        assert len(report.points) == 2
        names = {p.matcher_name for p in report.points}
        assert names == {"good", "bad"}

    def test_saves_json_report(self, tmp_path: Path):
        dag = _build_test_dag()
        config = AblationConfig(
            completeness_levels=[0.5],
            noise_levels=[0.0],
        )
        matchers = {"test": _mock_rank_fn({"OMIM:100001": 0.9, "OMIM:100002": 0.5})}

        run_ablation(
            matchers, DISEASE_ANNOTATIONS, dag,
            config=config, output_dir=tmp_path, save=True,
        )

        report_path = tmp_path / "ablation_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "points" in data
        assert "config" in data

    def test_point_has_correct_metadata(self, tmp_path: Path):
        dag = _build_test_dag()
        config = AblationConfig(
            completeness_levels=[0.4],
            noise_levels=[0.1],
        )
        matchers = {"m1": _mock_rank_fn({"OMIM:100001": 0.9, "OMIM:100002": 0.5})}

        report = run_ablation(
            matchers, DISEASE_ANNOTATIONS, dag,
            config=config, output_dir=tmp_path, save=False,
        )

        assert len(report.points) == 1
        p = report.points[0]
        assert p.completeness == 0.4
        assert p.noise_level == 0.1
        assert p.matcher_name == "m1"
        assert p.n_cases > 0


class TestAblationReport:
    def test_get_metrics_grid(self):
        report = AblationReport(
            points=[
                AblationPoint(0.4, 0.0, "m1", EvalMetrics(top10_accuracy=0.6)),
                AblationPoint(0.4, 0.1, "m1", EvalMetrics(top10_accuracy=0.5)),
                AblationPoint(0.8, 0.0, "m1", EvalMetrics(top10_accuracy=0.8)),
                AblationPoint(0.8, 0.1, "m1", EvalMetrics(top10_accuracy=0.7)),
            ],
        )
        grid = report.get_metrics_grid("m1", "top10_accuracy")
        assert len(grid) == 2  # 2 completeness levels
        assert len(grid[0]) == 2  # 2 noise levels
        # Row 0 = completeness 0.4
        assert grid[0][0] == 0.6
        assert grid[0][1] == 0.5
        # Row 1 = completeness 0.8
        assert grid[1][0] == 0.8
        assert grid[1][1] == 0.7

    def test_format_grid(self):
        report = AblationReport(
            points=[
                AblationPoint(0.4, 0.0, "test", EvalMetrics(top10_accuracy=0.7)),
                AblationPoint(0.8, 0.0, "test", EvalMetrics(top10_accuracy=0.9)),
            ],
        )
        table = report.format_grid("test", "top10_accuracy")
        assert "test" in table
        assert "top10_accuracy" in table

    def test_to_dict(self):
        report = AblationReport(
            config=AblationConfig(completeness_levels=[0.5], noise_levels=[0.0]),
            matcher_names=["m1"],
            points=[
                AblationPoint(0.5, 0.0, "m1", EvalMetrics(mrr=0.6), n_cases=10),
            ],
        )
        d = report.to_dict()
        assert d["matcher_names"] == ["m1"]
        assert len(d["points"]) == 1
        assert d["points"][0]["completeness"] == 0.5
        assert d["points"][0]["metrics"]["mrr"] == 0.6


class TestRunSingleDimensionAblation:
    def test_stratifies_by_completeness(self):
        cases = [
            BenchmarkCase("c1", ["HP:001"], "D1", completeness=0.4),
            BenchmarkCase("c2", ["HP:002"], "D1", completeness=0.4),
            BenchmarkCase("c3", ["HP:003"], "D1", completeness=0.8),
        ]
        matchers = {"m1": _mock_rank_fn({"D1": 0.9, "D2": 0.5})}

        results = run_single_dimension_ablation(matchers, cases, "completeness")
        assert "m1" in results
        assert 0.4 in results["m1"]
        assert 0.8 in results["m1"]
        # 2 cases at 0.4, 1 case at 0.8
        assert results["m1"][0.4].n_cases == 2
        assert results["m1"][0.8].n_cases == 1

    def test_stratifies_by_noise(self):
        cases = [
            BenchmarkCase("c1", ["HP:001"], "D1", noise_level=0.0),
            BenchmarkCase("c2", ["HP:002"], "D1", noise_level=0.1),
        ]
        matchers = {"m1": _mock_rank_fn({"D1": 0.9})}

        results = run_single_dimension_ablation(matchers, cases, "noise_level")
        assert 0.0 in results["m1"]
        assert 0.1 in results["m1"]
