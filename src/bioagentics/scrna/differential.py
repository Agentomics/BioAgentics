"""Differential expression analysis per cell type: inflamed vs non-inflamed.

Implements Wilcoxon rank-sum DE with FDR correction for each cell type
in the IL-23/Th17 atlas. Supports pseudobulk aggregation for datasets
with patient-level replicates.

Usage:
    from bioagentics.scrna.differential import run_de_per_celltype
    results = run_de_per_celltype(adata, condition_key="condition")
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests


@dataclass
class DEResult:
    """Differential expression result for one cell type."""

    cell_type: str
    n_cells_group1: int
    n_cells_group2: int
    n_genes_tested: int
    n_significant: int  # FDR < 0.05
    n_up: int  # logFC > 0 and significant
    n_down: int  # logFC < 0 and significant
    top_up_genes: list[str]
    top_down_genes: list[str]
    results_df: pd.DataFrame

    def summary(self) -> str:
        return (
            f"  {self.cell_type}: {self.n_cells_group1}v{self.n_cells_group2} cells, "
            f"{self.n_significant} DE genes (↑{self.n_up} ↓{self.n_down})"
        )


def wilcoxon_de(
    adata: ad.AnnData,
    group1_mask: np.ndarray,
    group2_mask: np.ndarray,
    min_cells: int = 10,
) -> pd.DataFrame:
    """Wilcoxon rank-sum test between two groups for all genes.

    Returns DataFrame with gene, logFC, pval, padj columns.
    """
    if group1_mask.sum() < min_cells or group2_mask.sum() < min_cells:
        return pd.DataFrame()

    X1 = adata[group1_mask].X
    X2 = adata[group2_mask].X

    # Convert to dense if sparse
    if hasattr(X1, "toarray"):
        X1 = X1.toarray()
    if hasattr(X2, "toarray"):
        X2 = X2.toarray()

    results = []
    for i, gene in enumerate(adata.var_names):
        g1 = X1[:, i].ravel()
        g2 = X2[:, i].ravel()

        # Skip genes with no expression
        if g1.sum() == 0 and g2.sum() == 0:
            continue

        mean1 = np.mean(g1)
        mean2 = np.mean(g2)
        logfc = mean1 - mean2  # Already log-space if log-normalized

        try:
            stat, pval = scipy_stats.ranksums(g1, g2)
        except ValueError:
            continue

        results.append({
            "gene": gene,
            "logFC": logfc,
            "mean_group1": mean1,
            "mean_group2": mean2,
            "pval": pval,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # FDR correction
    _, padj, _, _ = multipletests(df["pval"].values, method="fdr_bh")
    df["padj"] = padj

    return df.sort_values("pval").reset_index(drop=True)


def pseudobulk_de(
    adata: ad.AnnData,
    condition_key: str,
    patient_key: str,
    group1_label: str,
    group2_label: str,
) -> pd.DataFrame:
    """Pseudobulk differential expression aggregating by patient.

    Averages expression per patient within each condition, then runs
    Wilcoxon rank-sum on the patient-level pseudobulk profiles.
    """
    mask1 = adata.obs[condition_key] == group1_label
    mask2 = adata.obs[condition_key] == group2_label

    def _aggregate(sub_adata: ad.AnnData, key: str) -> pd.DataFrame:
        """Sum counts per patient, then normalize."""
        X = sub_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        patients = sub_adata.obs[key].values
        unique_patients = np.unique(patients)
        agg = np.zeros((len(unique_patients), X.shape[1]))
        for i, pat in enumerate(unique_patients):
            pat_mask = patients == pat
            agg[i] = X[pat_mask].sum(axis=0)
        # CPM-like normalization per patient
        totals = agg.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        agg = agg / totals * 1e6
        return pd.DataFrame(agg, index=unique_patients, columns=sub_adata.var_names)

    pb1 = _aggregate(adata[mask1], patient_key)
    pb2 = _aggregate(adata[mask2], patient_key)

    if len(pb1) < 3 or len(pb2) < 3:
        return pd.DataFrame()

    results = []
    for gene in adata.var_names:
        g1 = pb1[gene].values
        g2 = pb2[gene].values

        if g1.sum() == 0 and g2.sum() == 0:
            continue

        logfc = np.log2((g1.mean() + 1) / (g2.mean() + 1))
        try:
            stat, pval = scipy_stats.ranksums(g1, g2)
        except ValueError:
            continue

        results.append({"gene": gene, "logFC": logfc, "pval": pval})

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    _, padj, _, _ = multipletests(df["pval"].values, method="fdr_bh")
    df["padj"] = padj

    return df.sort_values("pval").reset_index(drop=True)


def run_de_per_celltype(
    adata: ad.AnnData,
    condition_key: str = "condition",
    cell_type_key: str = "cell_type",
    group1_label: str = "inflamed",
    group2_label: str = "non-inflamed",
    min_cells_per_group: int = 10,
    fdr_threshold: float = 0.05,
    patient_key: str | None = None,
) -> list[DEResult]:
    """Run differential expression for each cell type.

    Args:
        adata: Log-normalized AnnData with cell type and condition annotations.
        condition_key: Column in .obs with condition labels.
        cell_type_key: Column in .obs with cell type labels.
        group1_label: Label for group 1 (e.g., "inflamed").
        group2_label: Label for group 2 (e.g., "non-inflamed").
        min_cells_per_group: Minimum cells per group per cell type.
        fdr_threshold: FDR threshold for significance.
        patient_key: If provided, use pseudobulk DE aggregated by patient.

    Returns:
        List of DEResult, one per cell type with enough cells.
    """
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not in adata.obs")
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not in adata.obs")

    cell_types = [ct for ct in adata.obs[cell_type_key].unique() if ct != "Unassigned"]
    print(f"Running DE: {group1_label} vs {group2_label} across {len(cell_types)} cell types...")

    results = []
    for ct in sorted(cell_types):
        ct_mask = adata.obs[cell_type_key] == ct
        ct_adata = adata[ct_mask]

        g1_mask = ct_adata.obs[condition_key] == group1_label
        g2_mask = ct_adata.obs[condition_key] == group2_label

        n_g1 = g1_mask.sum()
        n_g2 = g2_mask.sum()

        if n_g1 < min_cells_per_group or n_g2 < min_cells_per_group:
            print(f"  {ct}: skipping ({n_g1}v{n_g2} cells, need {min_cells_per_group})")
            continue

        # Run DE
        if patient_key and patient_key in ct_adata.obs.columns:
            de_df = pseudobulk_de(ct_adata, condition_key, patient_key, group1_label, group2_label)
        else:
            de_df = wilcoxon_de(ct_adata, g1_mask.values, g2_mask.values)

        if de_df.empty:
            print(f"  {ct}: no testable genes")
            continue

        sig = de_df[de_df["padj"] < fdr_threshold]
        n_up = len(sig[sig["logFC"] > 0])
        n_down = len(sig[sig["logFC"] < 0])

        top_up = sig[sig["logFC"] > 0].head(5)["gene"].tolist()
        top_down = sig[sig["logFC"] < 0].head(5)["gene"].tolist()

        result = DEResult(
            cell_type=ct,
            n_cells_group1=int(n_g1),
            n_cells_group2=int(n_g2),
            n_genes_tested=len(de_df),
            n_significant=len(sig),
            n_up=n_up,
            n_down=n_down,
            top_up_genes=top_up,
            top_down_genes=top_down,
            results_df=de_df,
        )
        results.append(result)
        print(result.summary())

    print(f"\nDE complete: {len(results)} cell types tested")
    return results
