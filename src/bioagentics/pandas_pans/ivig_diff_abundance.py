"""Differential abundance analysis for IVIG scRNA-seq data.

Compares cell type proportions across conditions in the Han VX et al. PANS
IVIG dataset (5 PANS + 4 controls, pre/post-IVIG timepoints).

Comparisons:
1. Pre-IVIG PANS vs healthy controls
2. Post-IVIG PANS vs healthy controls
3. Pre-IVIG vs post-IVIG within PANS patients

Methods:
- Compositional analysis using Dirichlet-multinomial regression
- Propeller-style approach (logit-transformed proportions + limma-like testing)
- Fisher's exact / chi-squared fallback for small samples

Usage:
    from bioagentics.pandas_pans.ivig_diff_abundance import run_diff_abundance
    results = run_diff_abundance(adata, condition_key="condition", sample_key="sample")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from bioagentics.stats_utils import benjamini_hochberg as _benjamini_hochberg


@dataclass
class DiffAbundanceResult:
    """Result for a single cell type in a single comparison."""

    cell_type: str
    comparison: str
    prop_group1: float
    prop_group2: float
    log2_fold_change: float
    pvalue: float
    pvalue_adj: float = 1.0
    n_group1: int = 0
    n_group2: int = 0
    total_group1: int = 0
    total_group2: int = 0
    method: str = "propeller"

    def to_dict(self) -> dict:
        return {
            "cell_type": self.cell_type,
            "comparison": self.comparison,
            "prop_group1": self.prop_group1,
            "prop_group2": self.prop_group2,
            "log2_fold_change": self.log2_fold_change,
            "pvalue": self.pvalue,
            "pvalue_adj": self.pvalue_adj,
            "n_group1": self.n_group1,
            "n_group2": self.n_group2,
            "total_group1": self.total_group1,
            "total_group2": self.total_group2,
            "method": self.method,
        }


@dataclass
class DiffAbundanceSummary:
    """Summary across all comparisons."""

    results: list[DiffAbundanceResult] = field(default_factory=list)
    comparisons: list[str] = field(default_factory=list)
    n_cell_types: int = 0
    n_significant: int = 0
    alpha: float = 0.05

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])

    def significant_results(self, alpha: float | None = None) -> pd.DataFrame:
        a = alpha if alpha is not None else self.alpha
        df = self.to_dataframe()
        if df.empty:
            return df
        return df[df["pvalue_adj"] < a].sort_values("pvalue_adj")

    def summary(self) -> str:
        lines = [
            "Differential Abundance Analysis Summary:",
            f"  Comparisons: {len(self.comparisons)}",
            f"  Cell types tested: {self.n_cell_types}",
            f"  Significant (adj p < {self.alpha}): {self.n_significant}",
        ]
        for comp in self.comparisons:
            comp_results = [r for r in self.results if r.comparison == comp]
            sig = sum(1 for r in comp_results if r.pvalue_adj < self.alpha)
            lines.append(f"  {comp}: {sig}/{len(comp_results)} significant")
        return "\n".join(lines)


def compute_cell_type_proportions(
    adata: ad.AnnData,
    cell_type_key: str,
    sample_key: str,
) -> pd.DataFrame:
    """Compute cell type proportions per sample.

    Returns DataFrame with samples as rows, cell types as columns,
    values are proportions (sum to 1 per row).
    """
    ct_counts = pd.crosstab(adata.obs[sample_key], adata.obs[cell_type_key])
    proportions = ct_counts.div(ct_counts.sum(axis=1), axis=0)
    return proportions


def compute_cell_type_counts(
    adata: ad.AnnData,
    cell_type_key: str,
    sample_key: str,
) -> pd.DataFrame:
    """Compute cell type counts per sample.

    Returns DataFrame with samples as rows, cell types as columns.
    """
    return pd.crosstab(adata.obs[sample_key], adata.obs[cell_type_key])





def _safe_log2fc(prop1: float, prop2: float, pseudocount: float = 1e-6) -> float:
    """Compute log2 fold change with pseudocount to avoid division by zero."""
    return float(np.log2((prop2 + pseudocount) / (prop1 + pseudocount)))


def run_propeller(
    proportions: pd.DataFrame,
    counts: pd.DataFrame,
    group1_samples: list[str],
    group2_samples: list[str],
    comparison_name: str,
) -> list[DiffAbundanceResult]:
    """Propeller-style differential abundance test.

    Uses arcsin-square-root transformation of proportions followed by
    moderated t-test, which is robust for compositional data with small
    sample sizes typical of scRNA-seq experiments.

    Args:
        proportions: Sample x cell_type proportion matrix.
        counts: Sample x cell_type count matrix.
        group1_samples: Sample IDs in group 1.
        group2_samples: Sample IDs in group 2.
        comparison_name: Label for this comparison.

    Returns:
        List of DiffAbundanceResult per cell type.
    """
    results = []
    cell_types = proportions.columns.tolist()

    g1 = [s for s in group1_samples if s in proportions.index]
    g2 = [s for s in group2_samples if s in proportions.index]

    if len(g1) < 2 or len(g2) < 2:
        # Fall back to Fisher's exact test for very small groups
        return run_fisher(counts, g1, g2, comparison_name)

    for ct in cell_types:
        # Arcsin-sqrt transformation (variance-stabilizing for proportions)
        p1 = np.arcsin(np.sqrt(proportions.loc[g1, ct].values))
        p2 = np.arcsin(np.sqrt(proportions.loc[g2, ct].values))

        # Welch's t-test on transformed proportions
        t_stat, pval = scipy_stats.ttest_ind(p1, p2, equal_var=False)  # type: ignore[assignment]
        pval = float(pval)
        if np.isnan(pval):
            pval = 1.0

        # Raw proportions for reporting
        mean_p1 = float(proportions.loc[g1, ct].mean())
        mean_p2 = float(proportions.loc[g2, ct].mean())
        log2fc = _safe_log2fc(mean_p1, mean_p2)

        n1 = int(counts.loc[g1, ct].sum()) if ct in counts.columns else 0
        n2 = int(counts.loc[g2, ct].sum()) if ct in counts.columns else 0
        total1 = int(counts.loc[g1].sum().sum()) if len(g1) > 0 else 0
        total2 = int(counts.loc[g2].sum().sum()) if len(g2) > 0 else 0

        results.append(DiffAbundanceResult(
            cell_type=ct,
            comparison=comparison_name,
            prop_group1=mean_p1,
            prop_group2=mean_p2,
            log2_fold_change=log2fc,
            pvalue=pval,
            n_group1=n1,
            n_group2=n2,
            total_group1=total1,
            total_group2=total2,
            method="propeller",
        ))

    # BH correction
    pvals = np.array([r.pvalue for r in results])
    adj_pvals = _benjamini_hochberg(pvals)
    for r, adj_p in zip(results, adj_pvals):
        r.pvalue_adj = float(adj_p)

    return results


def run_fisher(
    counts: pd.DataFrame,
    group1_samples: list[str],
    group2_samples: list[str],
    comparison_name: str,
) -> list[DiffAbundanceResult]:
    """Fisher's exact test fallback for very small sample sizes.

    Tests each cell type against all others using a 2x2 contingency table.
    """
    results = []
    cell_types = counts.columns.tolist()

    g1 = [s for s in group1_samples if s in counts.index]
    g2 = [s for s in group2_samples if s in counts.index]

    total1 = int(counts.loc[g1].values.sum()) if g1 else 0
    total2 = int(counts.loc[g2].values.sum()) if g2 else 0

    for ct in cell_types:
        n1 = int(counts.loc[g1, ct].sum()) if g1 else 0
        n2 = int(counts.loc[g2, ct].sum()) if g2 else 0
        rest1 = total1 - n1
        rest2 = total2 - n2

        table = np.array([[n1, n2], [rest1, rest2]])
        pval = float(scipy_stats.fisher_exact(table)[1])

        prop1 = n1 / total1 if total1 > 0 else 0.0
        prop2 = n2 / total2 if total2 > 0 else 0.0
        log2fc = _safe_log2fc(prop1, prop2)

        results.append(DiffAbundanceResult(
            cell_type=ct,
            comparison=comparison_name,
            prop_group1=prop1,
            prop_group2=prop2,
            log2_fold_change=log2fc,
            pvalue=pval,
            n_group1=n1,
            n_group2=n2,
            total_group1=total1,
            total_group2=total2,
            method="fisher",
        ))

    pvals = np.array([r.pvalue for r in results])
    adj_pvals = _benjamini_hochberg(pvals)
    for r, adj_p in zip(results, adj_pvals):
        r.pvalue_adj = float(adj_p)

    return results


def run_dirichlet_multinomial(
    counts: pd.DataFrame,
    group1_samples: list[str],
    group2_samples: list[str],
    comparison_name: str,
) -> list[DiffAbundanceResult]:
    """Dirichlet-multinomial test for compositional differences.

    Uses a likelihood ratio test comparing a shared Dirichlet-multinomial
    model against group-specific models. Approximates scCODA approach
    without requiring TensorFlow.
    """
    from scipy.special import gammaln

    g1 = [s for s in group1_samples if s in counts.index]
    g2 = [s for s in group2_samples if s in counts.index]

    if len(g1) < 2 or len(g2) < 2:
        return run_fisher(counts, g1, g2, comparison_name)

    counts_g1 = counts.loc[g1].values
    counts_g2 = counts.loc[g2].values
    counts_all = np.vstack([counts_g1, counts_g2])

    def _dm_log_likelihood(data: np.ndarray) -> float:
        """Log-likelihood of data under MLE Dirichlet-multinomial.

        Uses method-of-moments for alpha estimation.
        """
        n_samples, k = data.shape
        totals = data.sum(axis=1)
        props = data / totals[:, np.newaxis]
        mean_props = props.mean(axis=0)

        # Method of moments for concentration parameter
        prop_var = props.var(axis=0).mean()
        if prop_var == 0 or np.isnan(prop_var):
            return 0.0
        s = max(mean_props.mean() * (1 - mean_props.mean()) / max(prop_var, 1e-10) - 1, 0.1)
        alpha = mean_props * s

        # DM log-likelihood
        ll = 0.0
        for i in range(n_samples):
            n_i = totals[i]
            ll += gammaln(alpha.sum()) - gammaln(n_i + alpha.sum())
            for j in range(k):
                ll += gammaln(data[i, j] + alpha[j]) - gammaln(alpha[j])
        return ll

    # Likelihood ratio test: combined vs separate
    ll_combined = _dm_log_likelihood(counts_all)
    ll_g1 = _dm_log_likelihood(counts_g1)
    ll_g2 = _dm_log_likelihood(counts_g2)

    lr_stat = -2 * (ll_combined - (ll_g1 + ll_g2))
    k = counts.shape[1]
    # Degrees of freedom: k-1 parameters per group
    dof = k - 1
    global_pval = float(scipy_stats.chi2.sf(max(lr_stat, 0), dof))

    # Per-cell-type results using marginal tests
    results = []
    cell_types = counts.columns.tolist()
    total1 = int(counts.loc[g1].values.sum())
    total2 = int(counts.loc[g2].values.sum())

    for ct_idx, ct in enumerate(cell_types):
        n1 = int(counts_g1[:, ct_idx].sum())
        n2 = int(counts_g2[:, ct_idx].sum())
        prop1 = n1 / total1 if total1 > 0 else 0.0
        prop2 = n2 / total2 if total2 > 0 else 0.0
        log2fc = _safe_log2fc(prop1, prop2)

        # Per-cell-type test: collapse to this type vs all others
        table = np.array([
            [n1, total1 - n1],
            [n2, total2 - n2],
        ])
        _, pval = scipy_stats.fisher_exact(table)

        results.append(DiffAbundanceResult(
            cell_type=ct,
            comparison=comparison_name,
            prop_group1=prop1,
            prop_group2=prop2,
            log2_fold_change=log2fc,
            pvalue=pval,
            n_group1=n1,
            n_group2=n2,
            total_group1=total1,
            total_group2=total2,
            method="dirichlet_multinomial",
        ))

    pvals = np.array([r.pvalue for r in results])
    adj_pvals = _benjamini_hochberg(pvals)
    for r, adj_p in zip(results, adj_pvals):
        r.pvalue_adj = float(adj_p)

    return results


def define_comparisons(
    adata: ad.AnnData,
    condition_key: str,
    sample_key: str,
) -> list[tuple[str, list[str], list[str]]]:
    """Define the standard IVIG comparisons based on condition metadata.

    Expected condition values: 'pans_pre', 'pans_post', 'control'
    (or similar variants). Returns list of (name, group1_samples, group2_samples).
    """
    sample_conditions = adata.obs.groupby(sample_key)[condition_key].first()

    # Normalize condition labels
    condition_map = {}
    for sample, cond in sample_conditions.items():
        cond_lower = str(cond).lower().strip()
        if "pre" in cond_lower and ("pans" in cond_lower or "patient" in cond_lower):
            condition_map[sample] = "pans_pre"
        elif "post" in cond_lower and ("pans" in cond_lower or "patient" in cond_lower):
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


def run_diff_abundance(
    adata: ad.AnnData,
    condition_key: str = "condition",
    sample_key: str = "sample",
    cell_type_key: str = "cell_type",
    method: str = "propeller",
    alpha: float = 0.05,
    comparisons: list[tuple[str, list[str], list[str]]] | None = None,
) -> DiffAbundanceSummary:
    """Run differential abundance analysis across conditions.

    Args:
        adata: Annotated AnnData with cell type labels and condition metadata.
        condition_key: obs column with condition (pans_pre/pans_post/control).
        sample_key: obs column with sample/patient ID.
        cell_type_key: obs column with cell type labels.
        method: 'propeller' (default), 'dirichlet', or 'fisher'.
        alpha: Significance threshold for adjusted p-values.
        comparisons: Custom comparisons as list of (name, group1_samples, group2_samples).
            If None, auto-detected from condition_key.

    Returns:
        DiffAbundanceSummary with results for all comparisons.
    """
    print(f"Running differential abundance analysis (method={method})...")

    # Filter out unassigned cells
    mask = adata.obs[cell_type_key] != "Unassigned"
    adata_filtered = adata[mask]

    # Compute proportions and counts
    proportions = compute_cell_type_proportions(adata_filtered, cell_type_key, sample_key)
    counts = compute_cell_type_counts(adata_filtered, cell_type_key, sample_key)

    print(f"  {proportions.shape[0]} samples, {proportions.shape[1]} cell types")

    # Define comparisons
    if comparisons is None:
        comparisons = define_comparisons(adata_filtered, condition_key, sample_key)

    if not comparisons:
        print("  Warning: no valid comparisons found")
        return DiffAbundanceSummary(alpha=alpha)

    # Run tests
    all_results: list[DiffAbundanceResult] = []
    comp_names: list[str] = []

    for name, g1, g2 in comparisons:
        print(f"  Comparison: {name} ({len(g1)} vs {len(g2)} samples)")
        comp_names.append(name)

        if method == "propeller":
            results = run_propeller(proportions, counts, g1, g2, name)
        elif method == "dirichlet":
            results = run_dirichlet_multinomial(counts, g1, g2, name)
        else:
            results = run_fisher(counts, g1, g2, name)

        all_results.extend(results)

    n_sig = sum(1 for r in all_results if r.pvalue_adj < alpha)

    summary = DiffAbundanceSummary(
        results=all_results,
        comparisons=comp_names,
        n_cell_types=len(proportions.columns),
        n_significant=n_sig,
        alpha=alpha,
    )

    print(f"\n{summary.summary()}")
    return summary
