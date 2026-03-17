"""IL-23/Th17 pathway activity scoring for single-cell data.

Scores individual cells for IL-23/Th17 axis gene modules and computes
per-cell-type pathway activity summaries. Includes the FCGR1A (CD64)
module per literature_reviewer finding on guselkumab dual mechanism.

Usage:
    from bioagentics.scrna.pathway_scoring import score_il23_pathway
    adata, activity = score_il23_pathway(adata)
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

# IL-23/Th17 axis gene modules
IL23_PATHWAY_MODULES: dict[str, list[str]] = {
    # IL-23 receptor complex — marks cells that can respond to IL-23
    "il23_receptor": ["IL23R", "IL12RB1"],
    # Th17 transcription program — downstream effectors
    "th17_effector": ["RORC", "IL17A", "IL17F", "IL22", "CCR6", "AHR"],
    # IL-23 signaling cascade — JAK/STAT signaling
    "il23_signaling": ["STAT3", "JAK2", "TYK2", "SOCS3"],
    # IL-23 production (myeloid) — cells producing IL-23
    "il23_production": ["IL23A", "IL12B"],
    # FCGR1A/CD64 module — guselkumab dual mechanism (Bhol et al.)
    # FCGR1A on macrophages mediates guselkumab's Fab-mediated binding
    "fcgr1a_mechanism": ["FCGR1A", "FCGR1B", "FCGR2A", "FCGR3A"],
    # Combined broad IL-23/Th17 pathway
    "il23_th17_combined": [
        "IL23R", "IL12RB1", "RORC", "IL17A", "IL17F", "IL22",
        "CCR6", "STAT3", "JAK2", "TYK2", "IL23A", "IL12B", "FCGR1A",
    ],
}


@dataclass
class PathwayActivity:
    """Per-cell-type IL-23/Th17 pathway activity summary."""

    module_name: str
    cell_type: str
    mean_score: float
    median_score: float
    pct_positive: float  # % of cells with score > 0
    n_cells: int


def score_pathway_modules(
    adata: ad.AnnData,
    modules: dict[str, list[str]] | None = None,
    score_prefix: str = "il23_",
) -> ad.AnnData:
    """Score each cell for each IL-23/Th17 pathway module.

    Uses scanpy's score_genes with background-subtracted scoring.
    Stores scores in adata.obs['{score_prefix}{module_name}'].
    """
    if modules is None:
        modules = IL23_PATHWAY_MODULES

    for module_name, genes in modules.items():
        available = [g for g in genes if g in adata.var_names]
        col_name = f"{score_prefix}{module_name}"

        if len(available) < 1:
            print(f"  {module_name}: 0/{len(genes)} genes found — skipping")
            adata.obs[col_name] = 0.0
            continue

        sc.tl.score_genes(adata, gene_list=available, score_name=col_name)
        n_positive = (adata.obs[col_name] > 0).sum()
        print(
            f"  {module_name}: {len(available)}/{len(genes)} genes, "
            f"{n_positive:,} cells positive ({n_positive / adata.n_obs * 100:.1f}%)"
        )

    return adata


def compute_celltype_activity(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    score_prefix: str = "il23_",
) -> list[PathwayActivity]:
    """Compute per-cell-type pathway activity summaries.

    Returns a list of PathwayActivity records, one per (module, cell_type) pair.
    """
    score_cols = [c for c in adata.obs.columns if c.startswith(score_prefix)]
    if not score_cols:
        return []

    if cell_type_key not in adata.obs.columns:
        return []

    results = []
    for col in score_cols:
        module_name = col.replace(score_prefix, "")
        for ct, group in adata.obs.groupby(cell_type_key, observed=True):
            scores = group[col].values
            results.append(
                PathwayActivity(
                    module_name=module_name,
                    cell_type=str(ct),
                    mean_score=float(np.mean(scores)),
                    median_score=float(np.median(scores)),
                    pct_positive=float((scores > 0).sum() / len(scores) * 100),
                    n_cells=len(scores),
                )
            )

    return results


def activity_to_dataframe(activities: list[PathwayActivity]) -> pd.DataFrame:
    """Convert pathway activity records to a pivot-ready DataFrame."""
    if not activities:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "module": a.module_name,
            "cell_type": a.cell_type,
            "mean_score": a.mean_score,
            "median_score": a.median_score,
            "pct_positive": a.pct_positive,
            "n_cells": a.n_cells,
        }
        for a in activities
    ])
    return df


def rank_cell_types_by_pathway(
    activities: list[PathwayActivity],
    module: str = "il23_th17_combined",
    metric: str = "mean_score",
) -> pd.DataFrame:
    """Rank cell types by pathway activity for a specific module.

    Returns DataFrame sorted by the chosen metric (descending).
    """
    df = activity_to_dataframe(activities)
    if df.empty:
        return df

    subset = df[df["module"] == module].copy()
    return subset.sort_values(metric, ascending=False).reset_index(drop=True)


def score_il23_pathway(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    modules: dict[str, list[str]] | None = None,
) -> tuple[ad.AnnData, list[PathwayActivity]]:
    """Full IL-23/Th17 pathway scoring pipeline.

    Scores all cells for pathway modules and computes per-cell-type activity.

    Args:
        adata: Annotated, log-normalized AnnData.
        cell_type_key: Column in .obs with cell type labels.
        modules: Custom module dict, or None for default IL-23/Th17 modules.

    Returns:
        Tuple of (scored AnnData, list of PathwayActivity records).
    """
    print(f"Scoring IL-23/Th17 pathway modules on {adata.n_obs:,} cells...")

    # Check gene coverage
    all_genes = set()
    for genes in (modules or IL23_PATHWAY_MODULES).values():
        all_genes.update(genes)
    overlap = len(all_genes & set(adata.var_names))
    print(f"  Pathway gene overlap: {overlap}/{len(all_genes)}")

    # Score modules
    adata = score_pathway_modules(adata, modules=modules)

    # Compute per-cell-type activity
    activities = compute_celltype_activity(adata, cell_type_key=cell_type_key)

    if activities:
        # Print top cell types by combined score
        ranking = rank_cell_types_by_pathway(activities)
        if not ranking.empty:
            print(f"\nTop cell types by IL-23/Th17 combined score:")
            for _, row in ranking.head(10).iterrows():
                print(
                    f"  {row['cell_type']}: mean={row['mean_score']:.3f}, "
                    f"pct_pos={row['pct_positive']:.1f}%, n={row['n_cells']:,}"
                )

    return adata, activities
