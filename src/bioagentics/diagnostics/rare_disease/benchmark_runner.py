"""Benchmark runner for evaluating all matchers on phenopacket cases.

Adapts each matcher (IC, freq-IC, node2vec, GAT, ensemble) to the common
RankFn protocol used by the evaluation harness. Runs all matchers against
a shared set of BenchmarkCases and produces a comparison table.

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.benchmark_runner
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from bioagentics.config import REPO_ROOT
from bioagentics.diagnostics.rare_disease.evaluation import (
    BenchmarkCase,
    EvalMetrics,
    RankFn,
    compare_models,
    evaluate_matcher,
    save_results,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "rare-disease-phenotype-matcher"


@dataclass
class BenchmarkResult:
    """Result from running one matcher on a benchmark set."""

    matcher_name: str
    metrics: EvalMetrics
    n_cases: int = 0


@dataclass
class BenchmarkReport:
    """Full benchmark report across all matchers."""

    results: list[BenchmarkResult] = field(default_factory=list)
    benchmark_name: str = "phenopacket_store"
    n_total_cases: int = 0

    def comparison_table(self) -> str:
        """Generate a formatted comparison table."""
        model_metrics = {r.matcher_name: r.metrics for r in self.results}
        return compare_models(model_metrics)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            "benchmark": self.benchmark_name,
            "n_cases": self.n_total_cases,
            "results": [
                {
                    "matcher": r.matcher_name,
                    "metrics": asdict(r.metrics),
                }
                for r in self.results
            ],
        }


def make_ic_rank_fn(
    scorer,
    disease_annotations: dict[str, list[str]],
    method: str = "resnik",
) -> RankFn:
    """Create a RankFn adapter for the IC matcher.

    Args:
        scorer: ICScorer instance with precomputed IC values.
        disease_annotations: {disease_id: [hpo_id, ...]}.
        method: "resnik" or "lin".
    """
    from bioagentics.diagnostics.rare_disease.ic_matcher import rank_diseases

    def rank_fn(query_hpo_terms: list[str]) -> list[tuple[str, float]]:
        results = rank_diseases(scorer, query_hpo_terms, disease_annotations, method)
        return [(r.disease_id, r.score) for r in results]

    return rank_fn


def make_freq_ic_rank_fn(
    matcher,
    disease_annotations: dict[str, list[str]],
    method: str = "resnik",
) -> RankFn:
    """Create a RankFn adapter for the frequency-weighted IC matcher.

    Args:
        matcher: FreqICMatcher instance.
        disease_annotations: {disease_id: [hpo_id, ...]}.
        method: "resnik" or "lin".
    """
    from bioagentics.diagnostics.rare_disease.freq_ic_matcher import rank_diseases

    def rank_fn(query_hpo_terms: list[str]) -> list[tuple[str, float]]:
        results = rank_diseases(matcher, query_hpo_terms, disease_annotations, method)
        return [(r.disease_id, r.score) for r in results]

    return rank_fn


def make_node2vec_rank_fn(
    matcher,
    disease_ids: list[str],
) -> RankFn:
    """Create a RankFn adapter for the node2vec matcher.

    Args:
        matcher: Node2VecMatcher instance.
        disease_ids: List of disease IDs to rank against.
    """
    from bioagentics.diagnostics.rare_disease.node2vec_matcher import rank_diseases

    def rank_fn(query_hpo_terms: list[str]) -> list[tuple[str, float]]:
        results = rank_diseases(matcher, query_hpo_terms, disease_ids)
        return [(r.disease_id, r.score) for r in results]

    return rank_fn


def make_gat_rank_fn(
    matcher,
    disease_ids: list[str],
) -> RankFn:
    """Create a RankFn adapter for the GAT matcher.

    Args:
        matcher: GATMatcher instance.
        disease_ids: List of disease IDs to rank against.
    """
    from bioagentics.diagnostics.rare_disease.gat_matcher import rank_diseases

    def rank_fn(query_hpo_terms: list[str]) -> list[tuple[str, float]]:
        results = rank_diseases(matcher, query_hpo_terms, disease_ids)
        return [(r.disease_id, r.score) for r in results]

    return rank_fn


def make_ensemble_rank_fn(
    ensemble_matcher,
    matchers: dict[str, RankFn],
) -> RankFn:
    """Create a RankFn adapter for the ensemble matcher.

    The ensemble collects scores from all individual matchers for each
    disease, then combines them.

    Args:
        ensemble_matcher: EnsembleMatcher instance.
        matchers: Dict mapping model name to its RankFn.
    """

    def rank_fn(query_hpo_terms: list[str]) -> list[tuple[str, float]]:
        # Collect per-disease scores from all matchers
        disease_scores: dict[str, dict[str, float]] = {}
        for name, matcher_fn in matchers.items():
            ranked = matcher_fn(query_hpo_terms)
            for disease_id, score in ranked:
                if disease_id not in disease_scores:
                    disease_scores[disease_id] = {}
                disease_scores[disease_id][name] = score

        # Combine via ensemble
        combined: list[tuple[str, float]] = []
        for disease_id, scores in disease_scores.items():
            ensemble_score = ensemble_matcher.combine_scores(scores)
            combined.append((disease_id, ensemble_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    return rank_fn


def run_benchmark(
    matchers: dict[str, RankFn],
    cases: list[BenchmarkCase],
    benchmark_name: str = "phenopacket_store",
    output_dir: Path | None = None,
    save: bool = True,
) -> BenchmarkReport:
    """Run all matchers against a set of benchmark cases.

    Args:
        matchers: Dict mapping matcher name to RankFn.
        cases: List of BenchmarkCase to evaluate.
        benchmark_name: Name for the benchmark set.
        output_dir: Directory for saving results.
        save: Whether to save per-matcher results to JSON.

    Returns:
        BenchmarkReport with comparison across all matchers.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    report = BenchmarkReport(
        benchmark_name=benchmark_name,
        n_total_cases=len(cases),
    )

    logger.info(
        "Running benchmark '%s' with %d cases across %d matchers",
        benchmark_name,
        len(cases),
        len(matchers),
    )

    for name, rank_fn in matchers.items():
        logger.info("Evaluating matcher: %s", name)
        metrics, results = evaluate_matcher(rank_fn, cases, name=name)
        report.results.append(
            BenchmarkResult(
                matcher_name=name,
                metrics=metrics,
                n_cases=len(results),
            )
        )

        if save:
            save_results(metrics, results, f"{benchmark_name}_{name}", output_dir)

    if save:
        report_path = output_dir / f"{benchmark_name}_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Saved benchmark report to %s", report_path)

    comparison = report.comparison_table()
    logger.info("Benchmark results:\n%s", comparison)

    return report
