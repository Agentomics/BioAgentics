"""Frequency-weighted Information Content phenotype matching.

Extends the IC baseline (ic_matcher.py) by weighting phenotype contributions
using Orphanet frequency qualifiers. Obligate terms contribute more than
occasional terms. Excluded terms penalize the disease score.

Scoring formula:
  For each query term q, the best-match against disease D is:
    best(q, D) = max over d in D of (freq(d) * sim(q, d))
  where freq(d) is the Orphanet frequency weight for term d in disease D.

  Excluded term penalty:
    If a query term matches an excluded phenotype for D (sim > threshold),
    subtract a penalty proportional to the match strength.

  Disease score = (sum of best-match scores - penalties) / |query|

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.freq_ic_matcher
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bioagentics.diagnostics.rare_disease.ic_matcher import ICScorer, RankResult

logger = logging.getLogger(__name__)

# Default frequency weights when Orphanet data is unavailable
DEFAULT_FREQ = 0.50

# Penalty multiplier for excluded phenotypes
EXCLUDED_PENALTY = 0.5


@dataclass
class FreqAnnotation:
    """Frequency annotation for a disease-phenotype pair."""

    hpo_id: str
    frequency: float  # 0.0 (excluded) to 1.0 (obligate)
    category: str  # e.g. "obligate", "frequent", "excluded"


@dataclass
class FreqICMatcher:
    """Frequency-weighted IC matcher for disease ranking.

    Attributes:
        scorer: ICScorer with precomputed IC values and HPO DAG.
        disease_freq: Mapping {disease_id: {hpo_id: FreqAnnotation}}.
        excluded_terms: Mapping {disease_id: set of excluded HPO term IDs}.
    """

    scorer: ICScorer
    disease_freq: dict[str, dict[str, FreqAnnotation]] = field(default_factory=dict)
    excluded_terms: dict[str, set[str]] = field(default_factory=dict)

    def add_disease_frequencies(
        self,
        disease_id: str,
        annotations: list[dict],
    ) -> None:
        """Register frequency data for a disease from Orphanet annotations.

        Args:
            disease_id: Disease identifier (e.g. "OMIM:100300", "ORPHA:558").
            annotations: List of dicts with keys: hpo_id, frequency_category,
                frequency_value (from orphanet_parser.build_orphanet_hpo_map or
                omim_mapper.build_disease_hpo_map).
        """
        freq_map: dict[str, FreqAnnotation] = {}
        excluded: set[str] = set()

        for ann in annotations:
            hpo_id = ann.get("hpo_id", "")
            if not hpo_id:
                continue

            # Support both Orphanet-style and HPOA-style annotation dicts
            category = ann.get("frequency_category", ann.get("frequency_label", "unknown"))
            freq_val = ann.get("frequency_value", ann.get("frequency", DEFAULT_FREQ))

            if category == "excluded":
                excluded.add(hpo_id)
            else:
                freq_map[hpo_id] = FreqAnnotation(
                    hpo_id=hpo_id,
                    frequency=freq_val,
                    category=category,
                )

        self.disease_freq[disease_id] = freq_map
        if excluded:
            self.excluded_terms[disease_id] = excluded

    def _get_freq_weight(self, disease_id: str, hpo_id: str) -> float:
        """Get the frequency weight for a disease-phenotype pair."""
        if disease_id in self.disease_freq:
            ann = self.disease_freq[disease_id].get(hpo_id)
            if ann is not None:
                return ann.frequency
        return DEFAULT_FREQ

    def score_disease(
        self,
        query_hpo_terms: list[str],
        disease_id: str,
        disease_terms: list[str],
        method: str = "resnik",
    ) -> float:
        """Score a single disease against a query phenotype profile.

        Args:
            query_hpo_terms: Patient's HPO term IDs.
            disease_id: Disease identifier.
            disease_terms: HPO terms annotated to this disease.
            method: Similarity method ("resnik" or "lin").

        Returns:
            Frequency-weighted similarity score.
        """
        if not query_hpo_terms or not disease_terms:
            return 0.0

        sim_fn = (
            self.scorer.resnik_similarity
            if method == "resnik"
            else self.scorer.lin_similarity
        )

        excluded = self.excluded_terms.get(disease_id, set())
        total_score = 0.0
        penalty = 0.0

        for q_term in query_hpo_terms:
            # Find best frequency-weighted match across disease terms
            best_weighted = 0.0
            for d_term in disease_terms:
                sim = sim_fn(q_term, d_term)
                weight = self._get_freq_weight(disease_id, d_term)
                weighted = weight * sim
                if weighted > best_weighted:
                    best_weighted = weighted
            total_score += best_weighted

            # Check excluded terms: penalize if query matches excluded phenotype
            for ex_term in excluded:
                sim = sim_fn(q_term, ex_term)
                if sim > 0:
                    penalty += EXCLUDED_PENALTY * sim

        n = len(query_hpo_terms)
        return max(0.0, (total_score - penalty) / n)


def rank_diseases(
    matcher: FreqICMatcher,
    query_hpo_terms: list[str],
    disease_annotations: dict[str, list[str]],
    method: str = "resnik",
) -> list[RankResult]:
    """Rank diseases by frequency-weighted phenotype similarity.

    Args:
        matcher: FreqICMatcher with preloaded frequency data.
        query_hpo_terms: Patient's HPO term IDs.
        disease_annotations: {disease_id: [hpo_id, ...]}.
        method: "resnik" or "lin".

    Returns:
        List of RankResult sorted by score descending.
    """
    if method not in ("resnik", "lin"):
        raise ValueError(f"Unknown similarity method: {method}")

    results: list[RankResult] = []

    for disease_id, disease_terms in disease_annotations.items():
        score = matcher.score_disease(
            query_hpo_terms, disease_id, disease_terms, method
        )
        results.append(RankResult(disease_id=disease_id, score=score))

    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results
