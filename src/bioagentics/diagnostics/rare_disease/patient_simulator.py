"""Simulated patient profile generator for phenotype matching evaluation.

For each OMIM disease, generates simulated patient HPO profiles at varying
completeness levels (20-80% of annotated terms) with controlled noise
(5-20% random HPO terms added). Used for systematic evaluation of matcher
accuracy under realistic clinical conditions.

Output:
    List of BenchmarkCase objects compatible with the evaluation harness.

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.patient_simulator
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import networkx as nx

from bioagentics.diagnostics.rare_disease.evaluation import BenchmarkCase

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for patient simulation."""

    completeness_levels: list[float] = None  # type: ignore[assignment]
    noise_levels: list[float] = None  # type: ignore[assignment]
    cases_per_combination: int = 1
    min_disease_terms: int = 3  # skip diseases with fewer terms
    seed: int = 42

    def __post_init__(self):
        if self.completeness_levels is None:
            self.completeness_levels = [0.2, 0.4, 0.6, 0.8]
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.05, 0.10, 0.20]


def get_all_leaf_terms(hpo_dag: nx.DiGraph) -> list[str]:
    """Get all leaf HPO terms (no children) for use as noise terms."""
    return [n for n in hpo_dag.nodes() if hpo_dag.in_degree(n) == 0]


def get_non_root_terms(hpo_dag: nx.DiGraph) -> list[str]:
    """Get all non-root HPO phenotype terms for use as noise candidates."""
    root_id = "HP:0000001"
    return [n for n in hpo_dag.nodes() if n != root_id]


def simulate_patient(
    disease_terms: list[str],
    all_hpo_terms: list[str],
    completeness: float,
    noise_level: float,
    rng: random.Random,
) -> list[str]:
    """Generate a simulated patient HPO profile from a disease's term set.

    Args:
        disease_terms: HPO terms annotated to the disease.
        all_hpo_terms: All available HPO terms (for noise generation).
        completeness: Fraction of disease terms to include (0-1).
        noise_level: Fraction of noise terms to add relative to selected terms.
        rng: Random number generator.

    Returns:
        List of HPO term IDs representing the simulated patient profile.
    """
    # Sample disease terms at the given completeness
    n_select = max(1, int(len(disease_terms) * completeness))
    selected = rng.sample(disease_terms, min(n_select, len(disease_terms)))

    # Add noise terms (random HPO terms not in disease profile)
    disease_set = set(disease_terms)
    noise_candidates = [t for t in all_hpo_terms if t not in disease_set]

    n_noise = max(0, int(len(selected) * noise_level))
    if n_noise > 0 and noise_candidates:
        noise_terms = rng.sample(noise_candidates, min(n_noise, len(noise_candidates)))
        selected.extend(noise_terms)

    return selected


def generate_benchmark_cases(
    disease_annotations: dict[str, list[str]],
    hpo_dag: nx.DiGraph,
    config: SimulationConfig | None = None,
) -> list[BenchmarkCase]:
    """Generate simulated benchmark cases for all diseases.

    Args:
        disease_annotations: {disease_id: [hpo_id, ...]}.
        hpo_dag: HPO DAG for noise term selection.
        config: Simulation configuration.

    Returns:
        List of BenchmarkCase objects.
    """
    if config is None:
        config = SimulationConfig()

    rng = random.Random(config.seed)
    all_terms = get_non_root_terms(hpo_dag)
    cases: list[BenchmarkCase] = []

    eligible = {
        d: terms
        for d, terms in disease_annotations.items()
        if len(terms) >= config.min_disease_terms
    }

    logger.info(
        "Generating cases for %d diseases (min %d terms), "
        "%d completeness x %d noise levels x %d per combo",
        len(eligible),
        config.min_disease_terms,
        len(config.completeness_levels),
        len(config.noise_levels),
        config.cases_per_combination,
    )

    for disease_id, terms in eligible.items():
        for completeness in config.completeness_levels:
            for noise_level in config.noise_levels:
                for i in range(config.cases_per_combination):
                    query = simulate_patient(terms, all_terms, completeness, noise_level, rng)
                    case_id = f"{disease_id}_c{completeness:.0%}_n{noise_level:.0%}_{i}"
                    cases.append(
                        BenchmarkCase(
                            case_id=case_id,
                            query_hpo_terms=query,
                            true_disease_id=disease_id,
                            completeness=completeness,
                            noise_level=noise_level,
                        )
                    )

    logger.info("Generated %d benchmark cases", len(cases))
    return cases
