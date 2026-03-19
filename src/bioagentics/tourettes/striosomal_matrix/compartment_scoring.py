"""Compartment scoring: map mesoscale molecular zones to striosome/matrix identity.

Scores each of the 6 mesoscale striatal zones on striosomal vs. matrix identity
using canonical marker gene expression:
  - Striosome markers: OPRM1 (mu-opioid receptor), TAC1 (substance P)
  - Matrix markers: CALB1 (calbindin), PENK (enkephalin)
  - Additional marker: SST (somatostatin, striosome-enriched in some contexts)

Input: gene expression matrix (zones x genes) from mesoscale striatum atlas.
Output: continuous striosome/matrix scores per zone.

Usage:
    from bioagentics.tourettes.striosomal_matrix.compartment_scoring import (
        compute_compartment_scores,
        score_genes_by_compartment,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Canonical compartment markers
STRIOSOME_MARKERS = {
    "OPRM1": "mu-opioid receptor — classical striosome marker",
    "TAC1": "tachykinin/substance P — striosome/D1-MSN marker",
}

MATRIX_MARKERS = {
    "CALB1": "calbindin — classical matrix marker",
    "PENK": "proenkephalin — matrix/D2-MSN marker",
}

ADDITIONAL_MARKERS = {
    "SST": "somatostatin — striosome-enriched in some studies",
}

ALL_MARKERS = {**STRIOSOME_MARKERS, **MATRIX_MARKERS, **ADDITIONAL_MARKERS}


@dataclass
class ZoneCompartmentScore:
    """Compartment identity score for a single molecular zone."""

    zone_id: str
    striosome_score: float  # mean z-scored striosome marker expression
    matrix_score: float     # mean z-scored matrix marker expression
    compartment_bias: float  # striosome - matrix (positive = striosome-biased)
    confidence: float       # based on marker availability and variance


def compute_compartment_scores(
    zone_expression: pd.DataFrame,
    striosome_markers: list[str] | None = None,
    matrix_markers: list[str] | None = None,
) -> pd.DataFrame:
    """Compute striosome/matrix compartment scores for each zone.

    Args:
        zone_expression: DataFrame with zones as rows, genes as columns.
            Values should be mean expression per zone (e.g., from
            pseudobulk aggregation of single-cell data).
        striosome_markers: Override striosome marker genes.
        matrix_markers: Override matrix marker genes.

    Returns:
        DataFrame with columns: zone_id, striosome_score, matrix_score,
        compartment_bias, confidence, striosome_markers_found, matrix_markers_found.
    """
    if striosome_markers is None:
        striosome_markers = list(STRIOSOME_MARKERS.keys())
    if matrix_markers is None:
        matrix_markers = list(MATRIX_MARKERS.keys())

    available_genes = set(zone_expression.columns)

    # Find available markers
    strio_avail = [g for g in striosome_markers if g in available_genes]
    matrix_avail = [g for g in matrix_markers if g in available_genes]

    if not strio_avail and not matrix_avail:
        raise ValueError(
            f"No compartment markers found in expression data. "
            f"Looked for striosome: {striosome_markers}, matrix: {matrix_markers}. "
            f"Available genes (sample): {sorted(available_genes)[:10]}"
        )

    # Z-score normalize across zones for each marker
    z_scored = zone_expression.apply(
        lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col * 0,
        axis=0,
    )

    rows = []
    for zone_id in zone_expression.index:
        strio_vals = [z_scored.loc[zone_id, g] for g in strio_avail] if strio_avail else [0.0]
        matrix_vals = [z_scored.loc[zone_id, g] for g in matrix_avail] if matrix_avail else [0.0]

        strio_score = float(np.mean(strio_vals))
        matrix_score = float(np.mean(matrix_vals))
        bias = strio_score - matrix_score

        # Confidence: higher when more markers available and consistent
        n_markers = len(strio_avail) + len(matrix_avail)
        max_markers = len(striosome_markers) + len(matrix_markers)
        marker_coverage = n_markers / max_markers if max_markers > 0 else 0

        all_vals = strio_vals + matrix_vals
        consistency = 1.0 - float(np.std(all_vals)) if len(all_vals) > 1 else 0.5
        confidence = marker_coverage * max(0, consistency)

        rows.append({
            "zone_id": str(zone_id),
            "striosome_score": strio_score,
            "matrix_score": matrix_score,
            "compartment_bias": bias,
            "confidence": confidence,
            "striosome_markers_found": ";".join(strio_avail),
            "matrix_markers_found": ";".join(matrix_avail),
        })

    return pd.DataFrame(rows)


def score_genes_by_compartment(
    zone_expression: pd.DataFrame,
    zone_scores: pd.DataFrame,
) -> dict[str, float]:
    """Score each gene by its compartment bias across zones.

    Computes a weighted average of each gene's expression across zones,
    weighted by the zone's compartment_bias score. Genes expressed more
    in striosome-biased zones get positive scores; matrix-biased get negative.

    Args:
        zone_expression: Zones x genes expression matrix.
        zone_scores: Output from compute_compartment_scores().

    Returns:
        Dict mapping gene_symbol -> compartment_bias_score.
    """
    # Align zone order
    zones = zone_scores["zone_id"].values
    weights = zone_scores["compartment_bias"].values

    # Normalize weights to sum to 0 (center them)
    weights = weights - np.mean(weights)
    w_norm = np.sum(np.abs(weights))
    if w_norm > 0:
        weights = weights / w_norm

    gene_scores: dict[str, float] = {}
    for gene in zone_expression.columns:
        if gene not in zone_expression.columns:
            continue
        # Z-score expression across zones
        vals = zone_expression.loc[zones, gene].values.astype(float)
        std = np.std(vals)
        if std > 0:
            z_vals = (vals - np.mean(vals)) / std
        else:
            z_vals = np.zeros_like(vals)

        gene_scores[gene] = float(np.dot(weights, z_vals))

    return gene_scores
