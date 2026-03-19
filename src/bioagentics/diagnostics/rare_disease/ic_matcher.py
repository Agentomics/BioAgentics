"""Information Content (IC) based phenotype matching — baseline model.

Implements Resnik and Lin semantic similarity using the Most Informative
Common Ancestor (MICA) approach. This reproduces Phenomizer-style ranking
as the baseline against which graph-based models are compared.

IC for each HPO term is computed from annotation frequency:
  IC(t) = -log2(p(t))
where p(t) = (annotations of t and its descendants) / (total annotations).

Similarity between two terms:
  Resnik: sim(t1, t2) = IC(MICA(t1, t2))
  Lin:    sim(t1, t2) = 2 * IC(MICA) / (IC(t1) + IC(t2))

Disease ranking:
  Given a query set Q of HPO terms, score each disease D by the average
  best-match similarity across query terms.

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.ic_matcher
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class ICScorer:
    """Computes Information Content for HPO terms and similarity scores.

    Attributes:
        hpo_dag: HPO DAG with child→parent edges.
        ic: Mapping from HPO term ID to IC value.
        term_ancestors: Cached ancestors for each term (inclusive).
    """

    hpo_dag: nx.DiGraph
    ic: dict[str, float] = field(default_factory=dict)
    term_ancestors: dict[str, frozenset[str]] = field(default_factory=dict)

    def compute_ic(self, disease_annotations: dict[str, list[str]]) -> None:
        """Compute IC for all HPO terms from disease annotation frequencies.

        Args:
            disease_annotations: Mapping {disease_id: [hpo_id, ...]}.
                Each disease contributes its annotated terms AND their ancestors.
        """
        total_diseases = len(disease_annotations)
        if total_diseases == 0:
            return

        # Count how many diseases are annotated with each term (including ancestors)
        term_counts: Counter[str] = Counter()
        for disease_id, hpo_ids in disease_annotations.items():
            # Collect all terms including ancestors
            disease_terms: set[str] = set()
            for hpo_id in hpo_ids:
                disease_terms.update(self._get_ancestors(hpo_id))
            for term in disease_terms:
                term_counts[term] += 1

        # Compute IC: -log2(p(t)) where p(t) = count(t) / total
        for term, count in term_counts.items():
            p = count / total_diseases
            self.ic[term] = -math.log2(p) if p > 0 else 0.0

        logger.info(
            "Computed IC for %d terms from %d diseases",
            len(self.ic),
            total_diseases,
        )

    def _get_ancestors(self, term_id: str) -> frozenset[str]:
        """Get all ancestors of a term (inclusive), with caching."""
        if term_id in self.term_ancestors:
            return self.term_ancestors[term_id]

        if term_id not in self.hpo_dag:
            result = frozenset()
        else:
            # In our DAG, edges go child→parent, so descendants() gives ancestors
            result = frozenset({term_id} | nx.descendants(self.hpo_dag, term_id))

        self.term_ancestors[term_id] = result
        return result

    def mica(self, term1: str, term2: str) -> str | None:
        """Find the Most Informative Common Ancestor of two terms.

        Returns the common ancestor with the highest IC value.
        """
        ancestors1 = self._get_ancestors(term1)
        ancestors2 = self._get_ancestors(term2)
        common = ancestors1 & ancestors2

        if not common:
            return None

        # Break IC ties by preferring deeper (more specific) terms
        return max(common, key=lambda t: (self.ic.get(t, 0.0), len(self._get_ancestors(t))))

    def resnik_similarity(self, term1: str, term2: str) -> float:
        """Resnik similarity: IC of MICA."""
        mica_term = self.mica(term1, term2)
        if mica_term is None:
            return 0.0
        return self.ic.get(mica_term, 0.0)

    def lin_similarity(self, term1: str, term2: str) -> float:
        """Lin similarity: 2 * IC(MICA) / (IC(t1) + IC(t2))."""
        ic_mica = self.resnik_similarity(term1, term2)
        ic1 = self.ic.get(term1, 0.0)
        ic2 = self.ic.get(term2, 0.0)
        denom = ic1 + ic2
        if denom == 0:
            return 0.0
        return 2.0 * ic_mica / denom


@dataclass
class RankResult:
    """Result of ranking a single disease."""

    disease_id: str
    score: float
    rank: int = 0


def rank_diseases(
    scorer: ICScorer,
    query_hpo_terms: list[str],
    disease_annotations: dict[str, list[str]],
    method: str = "resnik",
) -> list[RankResult]:
    """Rank diseases by phenotype similarity to a query.

    For each query term, find the best-matching term in each disease's profile.
    Disease score = average of best-match similarities across query terms.

    Args:
        scorer: ICScorer with precomputed IC values.
        query_hpo_terms: Patient's HPO term IDs.
        disease_annotations: {disease_id: [hpo_id, ...]}.
        method: "resnik" or "lin".

    Returns:
        List of RankResult sorted by score descending.
    """
    if method == "resnik":
        sim_fn = scorer.resnik_similarity
    elif method == "lin":
        sim_fn = scorer.lin_similarity
    else:
        raise ValueError(f"Unknown similarity method: {method}")

    results: list[RankResult] = []

    for disease_id, disease_terms in disease_annotations.items():
        if not disease_terms:
            results.append(RankResult(disease_id=disease_id, score=0.0))
            continue

        # For each query term, find best match in disease profile
        query_scores: list[float] = []
        for q_term in query_hpo_terms:
            best = max(
                (sim_fn(q_term, d_term) for d_term in disease_terms),
                default=0.0,
            )
            query_scores.append(best)

        # Average best-match similarity
        score = sum(query_scores) / len(query_scores) if query_scores else 0.0
        results.append(RankResult(disease_id=disease_id, score=score))

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)

    # Assign ranks
    for i, r in enumerate(results):
        r.rank = i + 1

    return results
