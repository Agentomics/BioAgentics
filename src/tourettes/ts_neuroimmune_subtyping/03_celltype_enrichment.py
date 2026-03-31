"""Phase 2: Immune cell-type enrichment analysis.

Tests whether genes with higher immune cell-type specificity carry stronger
TS GWAS signal, using two complementary approaches:

1. MAGMA-style continuous association: linear regression of gene GWAS Z-scores
   on cell-type specificity scores, controlling for gene size (n_snps).
2. Top-decile competitive test: Wilcoxon rank-sum comparing GWAS Z-scores of
   top-10% most specific genes vs. the rest.

Tests both grouped cell types (9 types from DICE) and 25 individual subtypes.
Prioritizes Th17 and NK cells per research plan.

Reuses gene-level Z-scores from Phase 1b (19,141 genes).

Usage:
    uv run python -m src.tourettes.ts_neuroimmune_subtyping.03_celltype_enrichment
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "tourettes" / "ts-neuroimmune-subtyping"
IMMUNE_REF = DATA_DIR / "immune_references"
GENE_RESULTS = (
    ROOT
    / "output"
    / "tourettes"
    / "ts-neuroimmune-subtyping"
    / "phase1b_lymphocytic_validation"
    / "gene_results.tsv"
)
OUTPUT_DIR = (
    ROOT
    / "output"
    / "tourettes"
    / "ts-neuroimmune-subtyping"
    / "phase2_celltype_enrichment"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gene_results() -> pd.DataFrame:
    """Load gene-level GWAS results from Phase 1b."""
    df = pd.read_csv(GENE_RESULTS, sep="\t")
    logger.info("Loaded %d gene results from Phase 1b", len(df))
    return df


def load_specificity(path: Path) -> pd.DataFrame:
    """Load a DICE specificity matrix (genes x cell types)."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    logger.info("Loaded specificity matrix: %d genes x %d cell types", *df.shape)
    return df


def load_gene_sets(path: Path) -> dict[str, set[str]]:
    """Load MAGMA-format gene sets (top 10% specific genes per cell type)."""
    gene_sets: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                gene_sets[parts[0]] = set(parts[1].split())
    return gene_sets


# ---------------------------------------------------------------------------
# Analysis: MAGMA-style continuous cell-type association
# ---------------------------------------------------------------------------


def magma_celltype_association(
    gene_z: pd.DataFrame, specificity: pd.DataFrame
) -> pd.DataFrame:
    """MAGMA-style cell-type association: regress gene Z on specificity.

    For each cell type, fits: Z_gene ~ specificity_celltype + log(n_snps)
    Tests one-sided: does higher specificity predict higher Z? (beta > 0)

    Following Skene et al. 2018 (Nat Genetics) approach.
    """
    # Merge gene results with specificity on Ensembl gene ID
    merged = gene_z.merge(specificity, left_on="gene", right_index=True, how="inner")
    logger.info("Merged: %d genes with both GWAS results and specificity", len(merged))

    if len(merged) < 100:
        logger.error("Too few genes after merge (%d). Aborting.", len(merged))
        return pd.DataFrame()

    cell_types = specificity.columns.tolist()
    results = []

    # Covariate: log(n_snps) controls for gene size
    log_nsnps = np.log(merged["n_snps"].clip(lower=1).values)
    z_scores = merged["z_score"].values

    for ct in cell_types:
        spec = merged[ct].values

        # Skip if no variance in specificity
        if np.std(spec) < 1e-10:
            logger.warning("Skipping %s: no variance in specificity", ct)
            continue

        # Build design matrix: [intercept, specificity, log_nsnps]
        X = np.column_stack([np.ones(len(z_scores)), spec, log_nsnps])

        try:
            # OLS regression
            beta, residuals, rank, sv = np.linalg.lstsq(X, z_scores, rcond=None)

            # Compute standard errors
            y_hat = X @ beta
            resid = z_scores - y_hat
            n, p = X.shape
            dof = n - p
            mse = np.sum(resid**2) / dof
            cov_beta = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov_beta))

            # t-statistic for specificity coefficient (index 1)
            t_stat = beta[1] / se[1]
            # One-sided p-value: higher specificity → higher Z
            p_val = stats.t.sf(t_stat, dof)

            results.append(
                {
                    "cell_type": ct,
                    "beta": beta[1],
                    "se": se[1],
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "n_genes": len(z_scores),
                }
            )
        except np.linalg.LinAlgError:
            logger.warning("LinAlgError for %s, skipping", ct)

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("p_value")
        df["fdr"] = _fdr(df["p_value"].values)
        df["bonferroni"] = (df["p_value"] * len(df)).clip(upper=1.0)
    return df


# ---------------------------------------------------------------------------
# Analysis: Top-decile competitive test (Wilcoxon rank-sum)
# ---------------------------------------------------------------------------


def top_decile_competitive_test(
    gene_z: pd.DataFrame, gene_sets: dict[str, set[str]]
) -> pd.DataFrame:
    """Wilcoxon rank-sum test: top-10% specific genes vs rest."""
    all_genes = set(gene_z["gene"])
    results = []

    for ct, ct_genes in gene_sets.items():
        overlap = ct_genes & all_genes
        if len(overlap) < 10:
            logger.warning("Skipping %s: only %d overlapping genes", ct, len(overlap))
            continue

        in_set = gene_z[gene_z["gene"].isin(overlap)]["z_score"].values
        not_in_set = gene_z[~gene_z["gene"].isin(overlap)]["z_score"].values

        stat, p_two = stats.mannwhitneyu(in_set, not_in_set, alternative="greater")
        z_wilcox = stats.norm.isf(p_two)

        results.append(
            {
                "cell_type": ct,
                "n_genes_in_set": len(overlap),
                "n_genes_rest": len(not_in_set),
                "mean_z_in_set": float(np.mean(in_set)),
                "mean_z_rest": float(np.mean(not_in_set)),
                "wilcoxon_z": z_wilcox,
                "p_value": p_two,
            }
        )

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("p_value")
        df["fdr"] = _fdr(df["p_value"].values)
    return df


# ---------------------------------------------------------------------------
# Analysis: Conditional analysis
# ---------------------------------------------------------------------------


def conditional_analysis(
    gene_z: pd.DataFrame,
    specificity: pd.DataFrame,
    sig_cell_types: list[str],
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Test each cell type conditioning on all others that are nominally significant.

    For each significant cell type, include all other significant types as
    covariates to assess independent contributions.
    """
    if len(sig_cell_types) < 2:
        return pd.DataFrame()

    merged = gene_z.merge(specificity, left_on="gene", right_index=True, how="inner")
    log_nsnps = np.log(merged["n_snps"].clip(lower=1).values)
    z_scores = merged["z_score"].values

    results = []
    for target_ct in sig_cell_types:
        # Covariates: all other significant cell types + log_nsnps
        other_cts = [ct for ct in sig_cell_types if ct != target_ct]
        covar_cols = [merged[ct].values for ct in other_cts]
        X = np.column_stack(
            [np.ones(len(z_scores)), merged[target_ct].values, log_nsnps]
            + covar_cols
        )

        try:
            beta, _, _, _ = np.linalg.lstsq(X, z_scores, rcond=None)
            y_hat = X @ beta
            resid = z_scores - y_hat
            n, p = X.shape
            dof = n - p
            mse = np.sum(resid**2) / dof
            cov_beta = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov_beta))

            t_stat = beta[1] / se[1]
            p_val = stats.t.sf(t_stat, dof)

            results.append(
                {
                    "cell_type": target_ct,
                    "conditioned_on": ", ".join(other_cts),
                    "beta_conditional": beta[1],
                    "se_conditional": se[1],
                    "t_stat_conditional": t_stat,
                    "p_conditional": p_val,
                }
            )
        except np.linalg.LinAlgError:
            logger.warning("Conditional LinAlgError for %s", target_ct)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    ranked = np.argsort(pvals)
    fdr = np.empty(n)
    for i, idx in enumerate(ranked):
        fdr[idx] = pvals[idx] * n / (i + 1)
    # Enforce monotonicity
    fdr_sorted = fdr[np.argsort(pvals)]
    for i in range(len(fdr_sorted) - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    for i, idx in enumerate(np.argsort(pvals)):
        fdr[idx] = min(fdr_sorted[i], 1.0)
    return fdr


def format_report(
    grouped_continuous: pd.DataFrame,
    subtype_continuous: pd.DataFrame,
    grouped_competitive: pd.DataFrame,
    conditional_results: pd.DataFrame,
    n_genes: int,
) -> str:
    """Generate markdown report."""
    lines = [
        "# Phase 2: Immune Cell-Type Enrichment Analysis",
        "",
        f"**Gene-level results from Phase 1b**: {n_genes} genes",
        "**Methods**: MAGMA-style continuous association, Wilcoxon top-decile competitive test",
        "**Specificity reference**: DICE (Schmiedel et al. 2018, Cell)",
        "",
    ]

    # Grouped continuous results
    lines.append("## Grouped Cell Types (9 types) — Continuous Association")
    lines.append("")
    lines.append("| Cell Type | Beta | SE | t-stat | P-value | FDR | Bonferroni |")
    lines.append("|-----------|------|----|---------|---------|----|------------|")
    for _, row in grouped_continuous.iterrows():
        lines.append(
            f"| {row['cell_type']} | {row['beta']:.4f} | {row['se']:.4f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4e} | {row['fdr']:.3f} | "
            f"{row['bonferroni']:.3f} |"
        )
    lines.append("")

    # Top-decile competitive
    lines.append("## Grouped Cell Types — Top-Decile Competitive Test")
    lines.append("")
    lines.append("| Cell Type | N Genes | Mean Z (set) | Mean Z (rest) | Wilcoxon Z | P-value | FDR |")
    lines.append("|-----------|---------|-------------|--------------|-----------|---------|-----|")
    for _, row in grouped_competitive.iterrows():
        lines.append(
            f"| {row['cell_type']} | {row['n_genes_in_set']} | "
            f"{row['mean_z_in_set']:.3f} | {row['mean_z_rest']:.3f} | "
            f"{row['wilcoxon_z']:.3f} | {row['p_value']:.4e} | {row['fdr']:.3f} |"
        )
    lines.append("")

    # Subtype continuous
    lines.append("## Individual Subtypes (25 types) — Continuous Association")
    lines.append("")
    lines.append("| Cell Type | Beta | SE | t-stat | P-value | FDR |")
    lines.append("|-----------|------|----|---------|---------|----|")
    for _, row in subtype_continuous.iterrows():
        lines.append(
            f"| {row['cell_type']} | {row['beta']:.4f} | {row['se']:.4f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4e} | {row['fdr']:.3f} |"
        )
    lines.append("")

    # Method agreement
    grouped_sig = set(
        grouped_continuous.loc[grouped_continuous["p_value"] < 0.05, "cell_type"]
    )
    competitive_sig = set(
        grouped_competitive.loc[grouped_competitive["p_value"] < 0.05, "cell_type"]
    )
    both_sig = grouped_sig & competitive_sig

    lines.append("## Method Agreement")
    lines.append("")
    if both_sig:
        lines.append(
            f"Cell types significant (P<0.05) in both methods: **{', '.join(sorted(both_sig))}**"
        )
    else:
        lines.append("No cell types reached P<0.05 in both methods simultaneously.")
    lines.append("")

    # Conditional results
    if len(conditional_results) > 0:
        lines.append("## Conditional Analysis")
        lines.append("")
        lines.append("| Cell Type | Conditioned On | Beta | t-stat | P-conditional |")
        lines.append("|-----------|---------------|------|--------|--------------|")
        for _, row in conditional_results.iterrows():
            lines.append(
                f"| {row['cell_type']} | {row['conditioned_on']} | "
                f"{row['beta_conditional']:.4f} | {row['t_stat_conditional']:.3f} | "
                f"{row['p_conditional']:.4e} |"
            )
        lines.append("")

    # Priority cell types summary
    lines.append("## Priority Cell Types (per research plan)")
    lines.append("")
    for ct_name in ["Th17", "NK", "CD4_T"]:
        gc = grouped_continuous[grouped_continuous["cell_type"] == ct_name]
        if len(gc) > 0:
            row = gc.iloc[0]
            lines.append(
                f"- **{ct_name}**: beta={row['beta']:.4f}, "
                f"t={row['t_stat']:.3f}, P={row['p_value']:.4e}, FDR={row['fdr']:.3f}"
            )
    lines.append("")

    # Power note
    lines.append("## Power Note")
    lines.append("")
    lines.append(
        "The 2019 TS GWAS (N=14,307) has limited power for cell-type enrichment. "
        "Null results are expected and do not exclude immune cell-type involvement. "
        "Future re-analysis with the 2024 TSAICG GWAS (N=19,138) is planned."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load gene-level GWAS results
    gene_z = load_gene_results()
    n_genes = len(gene_z)

    # 2. Load specificity matrices
    grouped_spec = load_specificity(IMMUNE_REF / "dice_grouped_specificity.tsv")
    subtype_spec = load_specificity(IMMUNE_REF / "dice_specificity_matrix.tsv")

    # 3. Load top-10% gene sets for competitive tests
    gene_sets = load_gene_sets(IMMUNE_REF / "dice_magma_gene_sets.tsv")

    # 4. MAGMA-style continuous association — grouped cell types
    logger.info("=== Continuous association: grouped cell types ===")
    grouped_continuous = magma_celltype_association(gene_z, grouped_spec)
    logger.info("Grouped results:\n%s", grouped_continuous.to_string(index=False))

    # 5. MAGMA-style continuous association — individual subtypes
    logger.info("=== Continuous association: individual subtypes ===")
    subtype_continuous = magma_celltype_association(gene_z, subtype_spec)
    logger.info("Subtype results:\n%s", subtype_continuous.to_string(index=False))

    # 6. Top-decile competitive test — grouped gene sets
    logger.info("=== Top-decile competitive test ===")
    grouped_competitive = top_decile_competitive_test(gene_z, gene_sets)
    logger.info("Competitive results:\n%s", grouped_competitive.to_string(index=False))

    # 7. Conditional analysis on nominally significant cell types
    sig_threshold = 0.05
    sig_cts = grouped_continuous.loc[
        grouped_continuous["p_value"] < sig_threshold, "cell_type"
    ].tolist()
    logger.info("Nominally significant (P<%.2f): %s", sig_threshold, sig_cts)

    conditional_results = conditional_analysis(
        gene_z, grouped_spec, sig_cts, sig_threshold
    )
    if len(conditional_results) > 0:
        logger.info(
            "Conditional results:\n%s", conditional_results.to_string(index=False)
        )

    # 8. Save results
    grouped_continuous.to_csv(
        OUTPUT_DIR / "grouped_continuous_results.tsv", sep="\t", index=False
    )
    subtype_continuous.to_csv(
        OUTPUT_DIR / "subtype_continuous_results.tsv", sep="\t", index=False
    )
    grouped_competitive.to_csv(
        OUTPUT_DIR / "grouped_competitive_results.tsv", sep="\t", index=False
    )
    if len(conditional_results) > 0:
        conditional_results.to_csv(
            OUTPUT_DIR / "conditional_results.tsv", sep="\t", index=False
        )

    # 9. Generate report
    report = format_report(
        grouped_continuous,
        subtype_continuous,
        grouped_competitive,
        conditional_results,
        n_genes,
    )
    (OUTPUT_DIR / "phase2_report.md").write_text(report)

    # 10. Summary JSON
    summary = {
        "n_genes_tested": n_genes,
        "n_grouped_cell_types": len(grouped_continuous),
        "n_subtypes": len(subtype_continuous),
        "grouped_top_hit": (
            {
                "cell_type": grouped_continuous.iloc[0]["cell_type"],
                "p_value": float(grouped_continuous.iloc[0]["p_value"]),
                "fdr": float(grouped_continuous.iloc[0]["fdr"]),
            }
            if len(grouped_continuous) > 0
            else None
        ),
        "subtype_top_hit": (
            {
                "cell_type": subtype_continuous.iloc[0]["cell_type"],
                "p_value": float(subtype_continuous.iloc[0]["p_value"]),
                "fdr": float(subtype_continuous.iloc[0]["fdr"]),
            }
            if len(subtype_continuous) > 0
            else None
        ),
        "n_grouped_nominal_sig": int(
            (grouped_continuous["p_value"] < 0.05).sum()
            if len(grouped_continuous) > 0
            else 0
        ),
        "n_grouped_fdr_sig": int(
            (grouped_continuous["fdr"] < 0.05).sum()
            if len(grouped_continuous) > 0
            else 0
        ),
        "n_subtype_nominal_sig": int(
            (subtype_continuous["p_value"] < 0.05).sum()
            if len(subtype_continuous) > 0
            else 0
        ),
        "n_subtype_fdr_sig": int(
            (subtype_continuous["fdr"] < 0.05).sum()
            if len(subtype_continuous) > 0
            else 0
        ),
        "priority_cell_types": {},
    }
    for ct in ["Th17", "NK", "CD4_T"]:
        row = grouped_continuous[grouped_continuous["cell_type"] == ct]
        if len(row) > 0:
            r = row.iloc[0]
            summary["priority_cell_types"][ct] = {
                "beta": float(r["beta"]),
                "t_stat": float(r["t_stat"]),
                "p_value": float(r["p_value"]),
                "fdr": float(r["fdr"]),
            }

    with open(OUTPUT_DIR / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase 2 complete. Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
