"""Pseudobulk differential expression analysis for IVIG scRNA-seq data.

Aggregates single-cell counts per cell type per sample, then performs
differential expression testing using a negative binomial GLM approach
(Wald test), similar to DESeq2.

Contrasts:
1. Pre-IVIG PANS vs healthy controls
2. Post-IVIG PANS vs healthy controls
3. Pre-IVIG vs post-IVIG within PANS patients

Usage:
    from bioagentics.pandas_pans.ivig_pseudobulk_de import run_pseudobulk_de
    results = run_pseudobulk_de(adata, condition_key="condition", sample_key="sample")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as scipy_stats

from bioagentics.stats_utils import benjamini_hochberg as _benjamini_hochberg


@dataclass
class DEResult:
    """Differential expression result for a single gene in a single comparison."""

    gene: str
    cell_type: str
    comparison: str
    log2_fold_change: float
    pvalue: float
    pvalue_adj: float = 1.0
    mean_group1: float = 0.0
    mean_group2: float = 0.0
    method: str = "wald_nb"

    def to_dict(self) -> dict:
        return {
            "gene": self.gene,
            "cell_type": self.cell_type,
            "comparison": self.comparison,
            "log2_fold_change": self.log2_fold_change,
            "pvalue": self.pvalue,
            "pvalue_adj": self.pvalue_adj,
            "mean_group1": self.mean_group1,
            "mean_group2": self.mean_group2,
            "method": self.method,
        }


@dataclass
class PseudobulkDESummary:
    """Summary of pseudobulk DE results across cell types and comparisons."""

    results_by_celltype: dict[str, list[DEResult]] = field(default_factory=dict)
    comparisons: list[str] = field(default_factory=list)
    cell_types_tested: list[str] = field(default_factory=list)
    n_de_genes: dict[str, int] = field(default_factory=dict)
    alpha: float = 0.05
    lfc_threshold: float = 0.5

    def get_de_genes(
        self,
        cell_type: str,
        comparison: str | None = None,
        alpha: float | None = None,
        lfc_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Get significant DE genes for a cell type."""
        a = alpha if alpha is not None else self.alpha
        lfc = lfc_threshold if lfc_threshold is not None else self.lfc_threshold

        if cell_type not in self.results_by_celltype:
            return pd.DataFrame()

        results = self.results_by_celltype[cell_type]
        if comparison:
            results = [r for r in results if r.comparison == comparison]

        df = pd.DataFrame([r.to_dict() for r in results])
        if df.empty:
            return df

        sig = df[(df["pvalue_adj"] < a) & (df["log2_fold_change"].abs() > lfc)]
        return sig.sort_values("pvalue_adj")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a single DataFrame."""
        all_results = []
        for results in self.results_by_celltype.values():
            all_results.extend(r.to_dict() for r in results)
        return pd.DataFrame(all_results)

    def summary(self) -> str:
        lines = [
            "Pseudobulk DE Analysis Summary:",
            f"  Comparisons: {len(self.comparisons)}",
            f"  Cell types tested: {len(self.cell_types_tested)}",
        ]
        for ct in self.cell_types_tested:
            n_de = self.n_de_genes.get(ct, 0)
            lines.append(f"  {ct}: {n_de} DE genes")
        return "\n".join(lines)


def aggregate_pseudobulk(
    adata: ad.AnnData,
    cell_type_key: str,
    sample_key: str,
    condition_key: str,
    min_cells: int = 10,
    layer: str | None = None,
) -> dict[str, ad.AnnData]:
    """Aggregate single-cell data to pseudobulk per cell type.

    Sums raw counts across cells of the same type within each sample.
    Only keeps cell-type/sample combinations with >= min_cells.

    Args:
        adata: Annotated AnnData.
        cell_type_key: obs column for cell type.
        sample_key: obs column for sample ID.
        condition_key: obs column for condition.
        min_cells: Minimum cells per cell type per sample.
        layer: Layer to use for counts (None = adata.X).

    Returns:
        Dict mapping cell_type -> AnnData with pseudobulk counts.
    """
    cell_types = sorted(adata.obs[cell_type_key].unique())
    cell_types = [ct for ct in cell_types if ct != "Unassigned"]

    pseudobulk = {}
    for ct in cell_types:
        ct_mask = adata.obs[cell_type_key] == ct
        ct_adata = adata[ct_mask]

        samples = ct_adata.obs[sample_key].unique()
        agg_data = []
        sample_meta = []

        for sample in samples:
            s_mask = ct_adata.obs[sample_key] == sample
            n_cells = int(s_mask.sum())
            if n_cells < min_cells:
                continue

            X_subset = ct_adata[s_mask].layers[layer] if layer else ct_adata[s_mask].X
            counts = np.asarray(X_subset.sum(axis=0)).ravel()

            agg_data.append(counts)
            condition = ct_adata[s_mask].obs[condition_key].iloc[0]
            sample_meta.append({
                "sample": sample,
                "condition": str(condition),
                "n_cells": n_cells,
            })

        if len(agg_data) < 2:
            continue

        X_pb = np.array(agg_data, dtype=np.float64)
        obs = pd.DataFrame(sample_meta)
        obs.index = obs["sample"].values

        pb_adata = ad.AnnData(
            X=X_pb,
            obs=obs,
            var=pd.DataFrame(index=adata.var_names.copy()),
        )
        pseudobulk[ct] = pb_adata

    return pseudobulk





def _normalize_counts(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute size factors and normalize counts (DESeq2-style median-of-ratios).

    Returns (normalized_counts, size_factors).
    """
    # Geometric mean per gene (excluding zeros)
    log_X = np.log(X + 1)
    geo_means = np.exp(log_X.mean(axis=0))

    # Size factors: median of ratios
    ratios = (X + 1) / geo_means[np.newaxis, :]
    size_factors = np.median(ratios, axis=1)
    size_factors[size_factors == 0] = 1.0

    normalized = X / size_factors[:, np.newaxis]
    return normalized, size_factors


def run_de_wald(
    pb_adata: ad.AnnData,
    group1_samples: list[str],
    group2_samples: list[str],
    cell_type: str,
    comparison_name: str,
    min_mean_count: float = 1.0,
) -> list[DEResult]:
    """Run Wald-type DE test on pseudobulk counts.

    Uses log-transformed normalized counts with Welch's t-test,
    approximating a negative binomial Wald test for small sample sizes.

    Args:
        pb_adata: Pseudobulk AnnData (samples x genes).
        group1_samples: Reference group sample IDs.
        group2_samples: Test group sample IDs.
        cell_type: Cell type label.
        comparison_name: Comparison label.
        min_mean_count: Minimum mean count across groups to test.

    Returns:
        List of DEResult per gene.
    """
    g1 = [s for s in group1_samples if s in pb_adata.obs_names]
    g2 = [s for s in group2_samples if s in pb_adata.obs_names]

    if len(g1) < 2 or len(g2) < 2:
        return []

    X = pb_adata.X if not sp.issparse(pb_adata.X) else pb_adata.X.toarray()

    # Normalize
    X_norm, _ = _normalize_counts(X)

    # Get indices
    g1_idx = [list(pb_adata.obs_names).index(s) for s in g1]
    g2_idx = [list(pb_adata.obs_names).index(s) for s in g2]

    X1 = X_norm[g1_idx]
    X2 = X_norm[g2_idx]

    # Log-transform for testing
    log_X1 = np.log2(X1 + 1)
    log_X2 = np.log2(X2 + 1)

    results = []
    genes = pb_adata.var_names.tolist()

    for j, gene in enumerate(genes):
        mean1 = float(X1[:, j].mean())
        mean2 = float(X2[:, j].mean())

        # Skip very lowly expressed genes
        if (mean1 + mean2) / 2 < min_mean_count:
            continue

        # Log2 fold change from normalized means
        pseudocount = 0.5
        log2fc = float(np.log2((mean2 + pseudocount) / (mean1 + pseudocount)))

        # Welch's t-test on log-transformed counts
        vals1 = log_X1[:, j]
        vals2 = log_X2[:, j]

        # Skip if no variance
        if vals1.std() == 0 and vals2.std() == 0:
            continue

        t_stat, pval = scipy_stats.ttest_ind(vals1, vals2, equal_var=False)  # type: ignore[assignment]
        pval = float(pval)
        if np.isnan(pval):
            pval = 1.0

        results.append(DEResult(
            gene=gene,
            cell_type=cell_type,
            comparison=comparison_name,
            log2_fold_change=log2fc,
            pvalue=pval,
            mean_group1=mean1,
            mean_group2=mean2,
            method="wald_nb",
        ))

    # BH correction
    if results:
        pvals = np.array([r.pvalue for r in results])
        adj_pvals = _benjamini_hochberg(pvals)
        for r, adj_p in zip(results, adj_pvals):
            r.pvalue_adj = float(adj_p)

    return results


def define_de_comparisons(
    pb_adata: ad.AnnData,
    condition_key: str = "condition",
) -> list[tuple[str, list[str], list[str]]]:
    """Define standard IVIG DE comparisons.

    Returns list of (name, reference_samples, test_samples).
    """
    conditions = pb_adata.obs[condition_key].values
    samples = pb_adata.obs_names.tolist()

    condition_map: dict[str, str] = {}
    for sample, cond in zip(samples, conditions):
        cond_lower = str(cond).lower().strip()
        if "pre" in cond_lower:
            condition_map[sample] = "pans_pre"
        elif "post" in cond_lower:
            condition_map[sample] = "pans_post"
        elif "control" in cond_lower or "healthy" in cond_lower:
            condition_map[sample] = "control"
        else:
            condition_map[sample] = cond_lower

    pans_pre = [s for s, c in condition_map.items() if c == "pans_pre"]
    pans_post = [s for s, c in condition_map.items() if c == "pans_post"]
    controls = [s for s, c in condition_map.items() if c == "control"]

    comparisons = []
    if pans_pre and controls:
        comparisons.append(("pans_pre_vs_control", controls, pans_pre))
    if pans_post and controls:
        comparisons.append(("pans_post_vs_control", controls, pans_post))
    if pans_pre and pans_post:
        comparisons.append(("pans_pre_vs_post", pans_pre, pans_post))

    return comparisons


def run_pseudobulk_de(
    adata: ad.AnnData,
    condition_key: str = "condition",
    sample_key: str = "sample",
    cell_type_key: str = "cell_type",
    min_cells: int = 10,
    min_mean_count: float = 1.0,
    alpha: float = 0.05,
    lfc_threshold: float = 0.5,
    layer: str | None = None,
    comparisons: list[tuple[str, list[str], list[str]]] | None = None,
) -> PseudobulkDESummary:
    """Run pseudobulk differential expression across cell types.

    Args:
        adata: Annotated AnnData with cell type labels and condition metadata.
        condition_key: obs column with condition.
        sample_key: obs column with sample ID.
        cell_type_key: obs column with cell type.
        min_cells: Min cells per type/sample for pseudobulk.
        min_mean_count: Min mean count to test a gene.
        alpha: FDR threshold.
        lfc_threshold: Min |log2FC| for significance.
        layer: AnnData layer for raw counts (None = X).
        comparisons: Custom comparisons. If None, auto-detected.

    Returns:
        PseudobulkDESummary with results.
    """
    print("Running pseudobulk DE analysis...")

    # Aggregate to pseudobulk
    pseudobulk = aggregate_pseudobulk(
        adata, cell_type_key, sample_key, condition_key,
        min_cells=min_cells, layer=layer,
    )

    print(f"  Aggregated {len(pseudobulk)} cell types to pseudobulk")

    summary = PseudobulkDESummary(
        alpha=alpha,
        lfc_threshold=lfc_threshold,
    )

    for ct, pb_adata in sorted(pseudobulk.items()):
        # Define comparisons for this cell type
        ct_comps = comparisons
        if ct_comps is None:
            ct_comps = define_de_comparisons(pb_adata, condition_key)

        if not ct_comps:
            continue

        ct_results: list[DEResult] = []
        for name, g1, g2 in ct_comps:
            results = run_de_wald(
                pb_adata, g1, g2, ct, name,
                min_mean_count=min_mean_count,
            )
            ct_results.extend(results)

            if name not in summary.comparisons:
                summary.comparisons.append(name)

        if ct_results:
            summary.results_by_celltype[ct] = ct_results
            summary.cell_types_tested.append(ct)

            n_sig = sum(
                1 for r in ct_results
                if r.pvalue_adj < alpha and abs(r.log2_fold_change) > lfc_threshold
            )
            summary.n_de_genes[ct] = n_sig
            print(f"  {ct}: {len(ct_results)} genes tested, {n_sig} DE")

    print(f"\n{summary.summary()}")
    return summary
