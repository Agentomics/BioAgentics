"""Evaluation harness for phenotype matching models.

Runs any matcher against test cases and computes standard ranking metrics:
Top-1, Top-5, Top-10, Top-20 accuracy, Mean Reciprocal Rank (MRR), and
per-case rank of the correct diagnosis.

Supports both simulated and real benchmark cases via a common BenchmarkCase format.

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.evaluation
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "rare-disease-phenotype-matcher"


@dataclass
class BenchmarkCase:
    """A single evaluation test case."""

    case_id: str
    query_hpo_terms: list[str]
    true_disease_id: str
    completeness: float = 1.0  # fraction of disease terms included
    noise_level: float = 0.0  # fraction of noise terms added
    metadata: dict = field(default_factory=dict)


@dataclass
class CaseResult:
    """Evaluation result for a single test case."""

    case_id: str
    true_disease_id: str
    predicted_rank: int  # 1-based rank of true disease (0 = not found)
    top_predictions: list[str]  # top-10 predicted disease IDs
    true_disease_score: float = 0.0


@dataclass
class EvalMetrics:
    """Aggregate metrics across all test cases."""

    n_cases: int = 0
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    top10_accuracy: float = 0.0
    top20_accuracy: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    median_rank: float = 0.0
    mean_rank: float = 0.0


class Matcher(Protocol):
    """Protocol for phenotype matching models.

    Any matcher must implement a rank() method that takes query HPO terms
    and returns a list of (disease_id, score) tuples sorted by score descending.
    """

    def rank(self, query_hpo_terms: list[str]) -> list[tuple[str, float]]: ...


RankFn = Callable[[list[str]], list[tuple[str, float]]]


def evaluate_case(rank_fn: RankFn, case: BenchmarkCase) -> CaseResult:
    """Evaluate a single test case against a ranking function.

    Args:
        rank_fn: Function that takes query HPO terms and returns ranked
            (disease_id, score) pairs.
        case: Test case with query terms and true disease.

    Returns:
        CaseResult with the rank of the true disease.
    """
    ranked = rank_fn(case.query_hpo_terms)

    predicted_rank = 0
    true_disease_score = 0.0
    for i, (disease_id, score) in enumerate(ranked):
        if disease_id == case.true_disease_id:
            predicted_rank = i + 1
            true_disease_score = score
            break

    top_predictions = [d for d, _ in ranked[:10]]

    return CaseResult(
        case_id=case.case_id,
        true_disease_id=case.true_disease_id,
        predicted_rank=predicted_rank,
        top_predictions=top_predictions,
        true_disease_score=true_disease_score,
    )


def compute_metrics(results: list[CaseResult]) -> EvalMetrics:
    """Compute aggregate evaluation metrics from case results.

    Args:
        results: List of per-case evaluation results.

    Returns:
        EvalMetrics with Top-K accuracy, MRR, and rank statistics.
    """
    if not results:
        return EvalMetrics()

    n = len(results)
    ranks = [r.predicted_rank for r in results]

    # Filter out cases where disease was not found (rank=0)
    found_ranks = [r for r in ranks if r > 0]

    top1 = sum(1 for r in ranks if r == 1) / n
    top5 = sum(1 for r in ranks if 0 < r <= 5) / n
    top10 = sum(1 for r in ranks if 0 < r <= 10) / n
    top20 = sum(1 for r in ranks if 0 < r <= 20) / n

    # MRR: mean of 1/rank (0 for not-found cases)
    reciprocal_ranks = [1.0 / r if r > 0 else 0.0 for r in ranks]
    mrr = sum(reciprocal_ranks) / n

    # Rank statistics (only for found cases)
    if found_ranks:
        sorted_ranks = sorted(found_ranks)
        median_rank = sorted_ranks[len(sorted_ranks) // 2]
        mean_rank = sum(found_ranks) / len(found_ranks)
    else:
        median_rank = 0.0
        mean_rank = 0.0

    return EvalMetrics(
        n_cases=n,
        top1_accuracy=top1,
        top5_accuracy=top5,
        top10_accuracy=top10,
        top20_accuracy=top20,
        mrr=mrr,
        median_rank=median_rank,
        mean_rank=mean_rank,
    )


def evaluate_matcher(
    rank_fn: RankFn,
    test_cases: list[BenchmarkCase],
    name: str = "matcher",
) -> tuple[EvalMetrics, list[CaseResult]]:
    """Run full evaluation of a matcher on a set of test cases.

    Args:
        rank_fn: Ranking function for the matcher.
        test_cases: List of test cases.
        name: Name for logging.

    Returns:
        Tuple of (aggregate metrics, per-case results).
    """
    logger.info("Evaluating %s on %d cases", name, len(test_cases))

    results = [evaluate_case(rank_fn, case) for case in test_cases]
    metrics = compute_metrics(results)

    logger.info(
        "%s: Top-1=%.1f%% Top-5=%.1f%% Top-10=%.1f%% MRR=%.3f",
        name,
        metrics.top1_accuracy * 100,
        metrics.top5_accuracy * 100,
        metrics.top10_accuracy * 100,
        metrics.mrr,
    )

    return metrics, results


def save_results(
    metrics: EvalMetrics,
    results: list[CaseResult],
    name: str,
    output_dir: Path | None = None,
) -> Path:
    """Save evaluation results to JSON.

    Args:
        metrics: Aggregate metrics.
        results: Per-case results.
        name: Model name (used in filename).
        output_dir: Output directory. Defaults to OUTPUT_DIR.

    Returns:
        Path to saved file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "model": name,
        "metrics": asdict(metrics),
        "cases": [asdict(r) for r in results],
    }

    path = output_dir / f"{name}_eval.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved results to %s", path)
    return path


def compare_models(
    model_metrics: dict[str, EvalMetrics],
) -> str:
    """Generate a comparison table across models.

    Args:
        model_metrics: Dict mapping model name to metrics.

    Returns:
        Formatted comparison string.
    """
    header = f"{'Model':<25} {'Top-1':>7} {'Top-5':>7} {'Top-10':>7} {'Top-20':>7} {'MRR':>7} {'Med Rank':>9}"
    sep = "-" * len(header)
    lines = [header, sep]

    for name, m in sorted(model_metrics.items()):
        lines.append(
            f"{name:<25} {m.top1_accuracy:>6.1%} {m.top5_accuracy:>6.1%} "
            f"{m.top10_accuracy:>6.1%} {m.top20_accuracy:>6.1%} "
            f"{m.mrr:>6.3f} {m.median_rank:>9.0f}"
        )

    return "\n".join(lines)
