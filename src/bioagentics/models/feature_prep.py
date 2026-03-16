"""Feature matrix preparation for elastic-net dependency prediction models.

Loads DepMap NSCLC expression and CRISPR dependency data, aligns cell lines
and gene IDs, selects high-variance expression features, and returns
aligned X (features) and Y (targets) matrices.

Usage:
    from bioagentics.models.feature_prep import prepare_features
    result = prepare_features(depmap_dir="data/depmap/25q3")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from bioagentics.data.gene_ids import load_depmap_matrix
from bioagentics.data.nsclc_depmap import annotate_nsclc_lines

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"

# KPL grouping strategies for sensitivity analysis
GROUPING_STRATEGIES = {
    "kl": {"KPL": "KL"},       # Primary: group KPL with KL (STK11-dominant)
    "kp": {"KPL": "KP"},       # Alternative: group KPL with KP
    "separate": {},             # Keep KPL as its own category
}


@dataclass
class FeatureResult:
    """Result of feature matrix preparation."""

    X: pd.DataFrame                # (n_lines x n_features) expression features
    Y: pd.DataFrame                # (n_lines x n_targets) dependency scores
    cell_line_meta: pd.DataFrame   # cell line annotations (subtypes, etc.)
    feature_genes: list[str] = field(default_factory=list)  # selected feature gene names
    target_genes: list[str] = field(default_factory=list)   # target gene names
    n_lines: int = 0
    n_features: int = 0
    n_targets: int = 0


def prepare_features(
    depmap_dir: str | Path = DEFAULT_DEPMAP_DIR,
    n_features: int = 6000,
    grouping: str = "kl",
) -> FeatureResult:
    """Prepare aligned feature and target matrices for elastic-net models.

    Parameters
    ----------
    depmap_dir : path
        Directory containing DepMap 25Q3 data files.
    n_features : int
        Number of most-variable expression genes to select as features.
    grouping : str
        KPL grouping strategy: "kl" (default, group with KL),
        "kp" (group with KP), or "separate" (keep KPL distinct).

    Returns
    -------
    FeatureResult with aligned X, Y, and metadata.
    """
    depmap_dir = Path(depmap_dir)

    if grouping not in GROUPING_STRATEGIES:
        raise ValueError(f"Unknown grouping '{grouping}', must be one of {list(GROUPING_STRATEGIES)}")

    # 1. Get annotated NSCLC cell lines
    nsclc_meta = annotate_nsclc_lines(depmap_dir)
    nsclc_ids = set(nsclc_meta.index)

    # 2. Load expression and CRISPR matrices (HUGO symbol columns)
    expr_path = depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    crispr_path = depmap_dir / "CRISPRGeneEffect.csv"

    expr = load_depmap_matrix(expr_path)
    crispr = load_depmap_matrix(crispr_path)

    # 3. Align cell lines: intersection of NSCLC lines, expression, and CRISPR
    common_lines = sorted(nsclc_ids & set(expr.index) & set(crispr.index))
    if not common_lines:
        raise ValueError("No NSCLC cell lines found in both expression and CRISPR datasets")

    expr = expr.loc[common_lines]
    crispr = crispr.loc[common_lines]
    meta = nsclc_meta.loc[common_lines].copy()

    # Apply KPL grouping
    remap = GROUPING_STRATEGIES[grouping]
    if remap:
        meta["molecular_subtype"] = meta["molecular_subtype"].replace(remap)

    # 4. Feature selection: top N most variable expression genes
    gene_var = expr.var(axis=0)
    gene_var = gene_var.dropna().sort_values(ascending=False)
    n_select = min(n_features, len(gene_var))
    selected_genes = gene_var.head(n_select).index.tolist()

    X = expr[selected_genes].copy()

    # 5. Target matrix: all CRISPR genes (dependency scores)
    # Drop any all-NaN columns
    crispr = crispr.dropna(axis=1, how="all")
    Y = crispr.copy()

    return FeatureResult(
        X=X,
        Y=Y,
        cell_line_meta=meta,
        feature_genes=selected_genes,
        target_genes=Y.columns.tolist(),
        n_lines=len(common_lines),
        n_features=len(selected_genes),
        n_targets=Y.shape[1],
    )
