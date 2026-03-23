"""Ablation study for phenotype matching models.

Systematically evaluates matcher performance across varying phenotype
completeness levels and noise levels. Produces a metrics grid suitable
for heatmap visualization and performance degradation analysis.

Key analyses:
- Completeness ablation: How does accuracy degrade as fewer phenotype
  terms are available (20%, 40%, 60%, 80%)?
- Noise ablation: How robust are models to spurious HPO terms?
- Combined: Performance at each (completeness, noise) grid point.

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.ablation_study
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
    evaluate_matcher,
)
from bioagentics.diagnostics.rare_disease.patient_simulator import (
    SimulationConfig,
    generate_benchmark_cases,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "rare-disease-phenotype-matcher"

# External SOTA reference targets for contextualizing ablation results.
# Sources: DeepRare (Nature 2026, doi:10.1038/s41586-025-10097-9),
#          PhenoBrain (npj Digital Medicine, Jan 2025).
REFERENCE_TARGETS = {
    "DeepRare (HPO-only)": {"recall_at_1": 0.644},
    "DeepRare (multimodal)": {"recall_at_1": 0.706},
    "PhenoBrain (standalone)": {"top10_recall": 0.654},
    "PhenoBrain (human+computer)": {"top10_recall": 0.813},
}


@dataclass
class AblationConfig:
    """Configuration for the ablation study."""

    completeness_levels: list[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    noise_levels: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.20]
    )
    cases_per_disease: int = 1
    min_disease_terms: int = 3
    seed: int = 42


@dataclass
class AblationPoint:
    """Metrics for a single (completeness, noise) grid point."""

    completeness: float
    noise_level: float
    matcher_name: str
    metrics: EvalMetrics
    n_cases: int = 0


@dataclass
class AblationReport:
    """Full ablation study results."""

    points: list[AblationPoint] = field(default_factory=list)
    config: AblationConfig = field(default_factory=AblationConfig)
    matcher_names: list[str] = field(default_factory=list)

    def get_metrics_grid(
        self, matcher_name: str, metric: str = "top10_accuracy"
    ) -> list[list[float]]:
        """Extract a 2D grid of a specific metric for one matcher.

        Rows = completeness levels, Columns = noise levels.

        Args:
            matcher_name: Name of the matcher.
            metric: Metric field name from EvalMetrics.

        Returns:
            2D list of metric values.
        """
        comp_levels = sorted(set(p.completeness for p in self.points))
        noise_levels = sorted(set(p.noise_level for p in self.points))

        lookup: dict[tuple[float, float], float] = {}
        for p in self.points:
            if p.matcher_name == matcher_name:
                val = getattr(p.metrics, metric, 0.0)
                lookup[(p.completeness, p.noise_level)] = val

        grid = []
        for comp in comp_levels:
            row = [lookup.get((comp, noise), 0.0) for noise in noise_levels]
            grid.append(row)
        return grid

    def format_grid(
        self, matcher_name: str, metric: str = "top10_accuracy"
    ) -> str:
        """Format a metrics grid as a readable table.

        Args:
            matcher_name: Name of the matcher.
            metric: Metric field name from EvalMetrics.

        Returns:
            Formatted table string.
        """
        comp_levels = sorted(set(p.completeness for p in self.points))
        noise_levels = sorted(set(p.noise_level for p in self.points))
        grid = self.get_metrics_grid(matcher_name, metric)

        # Header
        header = f"{'Completeness':>13}"
        for noise in noise_levels:
            header += f"  noise={noise:.0%}".rjust(12)
        lines = [f"{matcher_name} — {metric}", header, "-" * len(header)]

        for comp, row in zip(comp_levels, grid):
            line = f"{comp:>12.0%} "
            for val in row:
                line += f"{val:>11.1%} "
            lines.append(line)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize report for JSON output."""
        return {
            "config": asdict(self.config),
            "matcher_names": self.matcher_names,
            "points": [
                {
                    "completeness": p.completeness,
                    "noise_level": p.noise_level,
                    "matcher": p.matcher_name,
                    "n_cases": p.n_cases,
                    "metrics": asdict(p.metrics),
                }
                for p in self.points
            ],
            "reference_targets": REFERENCE_TARGETS,
        }


def run_ablation(
    matchers: dict[str, RankFn],
    disease_annotations: dict[str, list[str]],
    hpo_dag,
    config: AblationConfig | None = None,
    output_dir: Path | None = None,
    save: bool = True,
) -> AblationReport:
    """Run the full ablation study across completeness and noise levels.

    For each (completeness, noise) combination, generates simulated patients
    from disease_annotations and evaluates all matchers.

    Args:
        matchers: Dict mapping matcher name to RankFn.
        disease_annotations: {disease_id: [hpo_id, ...]}.
        hpo_dag: HPO DAG (networkx DiGraph) for noise term generation.
        config: Ablation configuration.
        output_dir: Directory for saving results.
        save: Whether to save results to JSON.

    Returns:
        AblationReport with metrics at each grid point.
    """
    if config is None:
        config = AblationConfig()
    if output_dir is None:
        output_dir = OUTPUT_DIR

    report = AblationReport(
        config=config,
        matcher_names=list(matchers.keys()),
    )

    total_combos = len(config.completeness_levels) * len(config.noise_levels)
    logger.info(
        "Running ablation study: %d completeness x %d noise levels = %d combos, "
        "%d matchers",
        len(config.completeness_levels),
        len(config.noise_levels),
        total_combos,
        len(matchers),
    )

    combo_idx = 0
    for completeness in config.completeness_levels:
        for noise_level in config.noise_levels:
            combo_idx += 1
            logger.info(
                "Ablation %d/%d: completeness=%.0f%%, noise=%.0f%%",
                combo_idx,
                total_combos,
                completeness * 100,
                noise_level * 100,
            )

            # Generate simulated cases for this (completeness, noise) point
            sim_config = SimulationConfig(
                completeness_levels=[completeness],
                noise_levels=[noise_level],
                cases_per_combination=config.cases_per_disease,
                min_disease_terms=config.min_disease_terms,
                seed=config.seed,
            )
            cases = generate_benchmark_cases(
                disease_annotations, hpo_dag, sim_config
            )

            if not cases:
                logger.warning(
                    "No cases generated for completeness=%.0f%%, noise=%.0f%%",
                    completeness * 100,
                    noise_level * 100,
                )
                continue

            # Evaluate each matcher at this point
            for matcher_name, rank_fn in matchers.items():
                metrics, results = evaluate_matcher(
                    rank_fn,
                    cases,
                    name=f"{matcher_name}_c{completeness:.0%}_n{noise_level:.0%}",
                )
                report.points.append(
                    AblationPoint(
                        completeness=completeness,
                        noise_level=noise_level,
                        matcher_name=matcher_name,
                        metrics=metrics,
                        n_cases=len(results),
                    )
                )

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "ablation_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Saved ablation report to %s", report_path)

    # Log summary grids
    for matcher_name in matchers:
        logger.info("\n%s", report.format_grid(matcher_name, "top10_accuracy"))

    return report


def run_single_dimension_ablation(
    matchers: dict[str, RankFn],
    cases: list[BenchmarkCase],
    dimension: str,
) -> dict[str, dict[float, EvalMetrics]]:
    """Run ablation on pre-generated cases, stratifying by one dimension.

    Useful for analyzing phenopacket benchmark cases that already have
    varying completeness or noise levels set.

    Args:
        matchers: Dict mapping matcher name to RankFn.
        cases: Pre-generated BenchmarkCases with completeness/noise metadata.
        dimension: "completeness" or "noise_level" to stratify by.

    Returns:
        {matcher_name: {dimension_value: EvalMetrics}}.
    """
    # Group cases by dimension value
    groups: dict[float, list[BenchmarkCase]] = {}
    for case in cases:
        val = getattr(case, dimension, 0.0)
        groups.setdefault(val, []).append(case)

    results: dict[str, dict[float, EvalMetrics]] = {}

    for matcher_name, rank_fn in matchers.items():
        results[matcher_name] = {}
        for dim_val, group_cases in sorted(groups.items()):
            metrics, _ = evaluate_matcher(
                rank_fn,
                group_cases,
                name=f"{matcher_name}_{dimension}={dim_val}",
            )
            results[matcher_name][dim_val] = metrics

    return results
