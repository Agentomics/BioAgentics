"""Cepo + Cauchy combination for GWAS-to-single-cell integration.

Implements the Cepo cell-type-specific gene identification approach combined
with the Cauchy p-value combination test for aggregating GWAS-to-scRNA-seq
enrichment results across multiple methods, as recommended by the
benchmarking study (PMC12140538).

Three stages:
  1. Cepo: identify cell-type-specific differentially-expressed genes
     using weighted CV^2-based specificity (protein-coding only)
  2. MAGMA-GSEA: test GWAS enrichment of Cepo gene sets via competitive
     regression of gene Z-scores on set membership
  3. Cauchy combination: aggregate p-values across methods
     (MAGMA-celltype, Cepo-MAGMA) for maximum power

Usage:
    python -m bioagentics.tourettes.ts_gwas_functional_annotation.cepo_cauchy \
        --gene-results output/.../magma_results/gene_results.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.tourettes.ts_gwas_functional_annotation.config import (
    CELLTYPE_DIR,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CepoGeneScore:
    """Gene-level Cepo specificity score for a cell type."""

    gene: str
    cell_type: str
    cepo_score: float
    rank: int


@dataclass
class CepoCelltypeResult:
    """Cell-type enrichment result from Cepo + MAGMA-GSEA."""

    cell_type: str
    label: str
    n_cepo_genes: int
    n_genes_tested: int
    beta: float
    beta_se: float
    z_score: float
    p_value: float
    fdr_q: float = 1.0
    top_genes: list[str] = field(default_factory=list)


@dataclass
class CauchyCombinedResult:
    """Combined cell-type enrichment from Cauchy p-value combination."""

    cell_type: str
    label: str
    n_methods: int
    method_pvalues: dict[str, float]
    cauchy_stat: float
    combined_p: float
    fdr_q: float = 1.0


# ---------------------------------------------------------------------------
# Cepo: cell-type-specific gene identification
# ---------------------------------------------------------------------------


def compute_cepo_scores(
    expression_matrix: pd.DataFrame,
    cell_labels: pd.Series,
    min_cells: int = 10,
    top_n: int = 200,
) -> dict[str, list[CepoGeneScore]]:
    """Compute Cepo specificity scores for each cell type.

    Cepo identifies cell-type-specific genes by computing a weighted
    coefficient of variation (CV) metric that captures both expression
    magnitude and consistency within each cell type.

    Parameters
    ----------
    expression_matrix : pd.DataFrame
        Genes x cells expression matrix (log-normalized).
    cell_labels : pd.Series
        Cell-type label for each cell (index-aligned with matrix columns).
    min_cells : int
        Minimum cells per cell type.
    top_n : int
        Number of top Cepo genes to return per cell type.

    Returns
    -------
    Dict mapping cell_type -> list of CepoGeneScore (sorted by rank).
    """
    cell_types = cell_labels.value_counts()
    valid_types = cell_types[cell_types >= min_cells].index.tolist()

    if not valid_types:
        logger.warning("No cell types with >= %d cells", min_cells)
        return {}

    results: dict[str, list[CepoGeneScore]] = {}

    for ct in valid_types:
        ct_str = str(ct)
        ct_mask = cell_labels == ct
        other_mask = ~ct_mask

        ct_expr = expression_matrix.loc[:, ct_mask]
        other_expr = expression_matrix.loc[:, other_mask]

        # Cepo score: differential stability metric
        # Higher score = gene is specifically and consistently expressed in this type
        ct_mean = ct_expr.mean(axis=1)
        ct_var = ct_expr.var(axis=1, ddof=1).clip(lower=1e-10)
        other_mean = other_expr.mean(axis=1)
        other_var = other_expr.var(axis=1, ddof=1).clip(lower=1e-10)

        # Weighted CV^2 specificity: low CV in target + high CV in others
        ct_cv2 = ct_var / (ct_mean ** 2).clip(lower=1e-10)
        other_cv2 = other_var / (other_mean ** 2).clip(lower=1e-10)

        # Cepo score: fold-change in stability (inverse CV^2)
        # Genes with low within-type variation and high expression get high scores
        specificity = (1.0 / ct_cv2.clip(lower=1e-10)) - (1.0 / other_cv2.clip(lower=1e-10))

        # Weight by expression magnitude (require expression in target type)
        weight = ct_mean.clip(lower=0)
        cepo_scores = specificity * weight

        # Filter to expressed genes only
        expressed = ct_mean > 0.1
        cepo_scores = cepo_scores[expressed]

        # Rank and take top N
        ranked = cepo_scores.sort_values(ascending=False)
        top_genes = ranked.head(top_n)

        scores = [
            CepoGeneScore(
                gene=str(gene),
                cell_type=ct_str,
                cepo_score=float(score),
                rank=i + 1,
            )
            for i, (gene, score) in enumerate(top_genes.items())
        ]
        results[ct_str] = scores
        logger.debug("Cepo %s: %d genes scored, top score=%.3f",
                      ct_str, len(scores), scores[0].cepo_score if scores else 0)

    logger.info("Cepo: computed scores for %d cell types, top_n=%d",
                len(results), top_n)
    return results


def cepo_from_markers(
    markers: dict[str, list[str]],
    all_genes: list[str],
    top_n: int = 200,
) -> dict[str, list[CepoGeneScore]]:
    """Create Cepo-like scores from predefined marker gene lists.

    When scRNA-seq data is unavailable, use curated marker genes with
    synthetic specificity scores based on marker membership.

    Parameters
    ----------
    markers : dict
        Cell-type -> list of marker genes.
    all_genes : list
        All genes in the analysis (from MAGMA gene results).
    top_n : int
        Max genes per cell type.
    """
    all_gene_set = set(all_genes)
    results: dict[str, list[CepoGeneScore]] = {}

    for ct, gene_list in markers.items():
        overlapping = [g for g in gene_list if g in all_gene_set]
        scores = [
            CepoGeneScore(
                gene=g,
                cell_type=ct,
                cepo_score=1.0 / (i + 1),  # rank-based score
                rank=i + 1,
            )
            for i, g in enumerate(overlapping[:top_n])
        ]
        results[ct] = scores

    logger.info("Cepo (marker-based): %d cell types, %d-%d genes per type",
                len(results),
                min(len(v) for v in results.values()) if results else 0,
                max(len(v) for v in results.values()) if results else 0)
    return results


def filter_protein_coding(
    cepo_scores: dict[str, list[CepoGeneScore]],
    protein_coding_genes: set[str] | None = None,
) -> dict[str, list[CepoGeneScore]]:
    """Filter Cepo genes to protein-coding only.

    If no protein-coding gene list is provided, no filtering is applied.
    """
    if protein_coding_genes is None:
        return cepo_scores

    filtered = {}
    for ct, scores in cepo_scores.items():
        ct_filtered = [s for s in scores if s.gene in protein_coding_genes]
        # Re-rank
        for i, s in enumerate(ct_filtered):
            s.rank = i + 1
        filtered[ct] = ct_filtered

    logger.info("Protein-coding filter: retained %d/%d genes across types",
                sum(len(v) for v in filtered.values()),
                sum(len(v) for v in cepo_scores.values()))
    return filtered


# ---------------------------------------------------------------------------
# MAGMA-GSEA enrichment of Cepo gene sets
# ---------------------------------------------------------------------------


def cepo_magma_gsea(
    gene_df: pd.DataFrame,
    cepo_scores: dict[str, list[CepoGeneScore]],
    min_genes: int = 5,
    labels: dict[str, str] | None = None,
) -> list[CepoCelltypeResult]:
    """Test GWAS enrichment of Cepo gene sets via MAGMA competitive regression.

    For each cell type, test whether Cepo-identified genes have higher
    MAGMA Z-scores than background genes.

    Parameters
    ----------
    gene_df : pd.DataFrame
        MAGMA gene results with columns GENE, Z, P, and optionally N_SNPS.
    cepo_scores : dict
        Cepo scores per cell type from compute_cepo_scores.
    min_genes : int
        Minimum overlap between Cepo genes and tested genes.
    labels : dict | None
        Human-readable cell-type labels.
    """
    if gene_df.empty:
        return []

    genes = gene_df["GENE"].tolist()
    z_scores = gene_df["Z"].values.astype(float)
    gene_set = set(genes)
    n = len(genes)

    # Covariate: log(n_snps) if available
    covariates = None
    if "N_SNPS" in gene_df.columns:
        covariates = np.log1p(gene_df["N_SNPS"].values.astype(float))

    results = []
    for ct, scores in cepo_scores.items():
        cepo_genes = {s.gene for s in scores if s.gene in gene_set}
        n_overlap = len(cepo_genes)

        if n_overlap < min_genes:
            continue

        # Build weighted indicator: Cepo score as weight (or binary)
        cepo_lookup = {s.gene: s.cepo_score for s in scores}
        indicator = np.array([
            cepo_lookup.get(g, 0.0) for g in genes
        ])

        # Normalize indicator to [0, 1] range
        imax = indicator.max()
        if imax > 0:
            indicator = indicator / imax

        beta, se, z, p = _regression_test(z_scores, indicator, covariates)

        # Top contributing genes
        in_set = [(g, z_scores[i], cepo_lookup.get(g, 0))
                  for i, g in enumerate(genes) if g in cepo_genes]
        in_set.sort(key=lambda x: -x[1])
        top = [g for g, _, _ in in_set[:5]]

        results.append(CepoCelltypeResult(
            cell_type=ct,
            label=(labels or {}).get(ct, ct),
            n_cepo_genes=len(scores),
            n_genes_tested=n_overlap,
            beta=beta,
            beta_se=se,
            z_score=z,
            p_value=p,
            top_genes=top,
        ))

    _apply_fdr_cepo(results)
    logger.info("Cepo-MAGMA: %d cell types tested, %d at P < 0.05",
                len(results), sum(1 for r in results if r.p_value < 0.05))
    return results


def _regression_test(
    gene_z: np.ndarray,
    indicator: np.ndarray,
    covariates: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Competitive regression test: Z ~ indicator [+ covariates].

    Returns (beta, se, z_score, p_value).
    """
    n = len(gene_z)
    if n < 10:
        return 0.0, 1.0, 0.0, 1.0

    parts = [np.ones(n)]
    if covariates is not None:
        cov = covariates.reshape(n, -1) if covariates.ndim == 1 else covariates
        for col_idx in range(cov.shape[1]):
            col = cov[:, col_idx]
            if np.var(col) > 1e-10:
                parts.append(col)
    parts.append(indicator)

    X = np.column_stack(parts)
    test_idx = X.shape[1] - 1

    try:
        beta, _residuals, _, _ = np.linalg.lstsq(X, gene_z, rcond=None)
        beta_test = float(beta[test_idx])

        resid = gene_z - X @ beta
        df = max(n - X.shape[1], 1)
        mse = float(np.sum(resid ** 2)) / df

        XtX_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt(max(mse * XtX_inv[test_idx, test_idx], 1e-15)))
        z = beta_test / se if se > 0 else 0.0
        p = float(stats.norm.sf(z))  # one-sided enrichment
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 1.0, 0.0, 1.0

    return beta_test, se, z, p


# ---------------------------------------------------------------------------
# Cauchy p-value combination
# ---------------------------------------------------------------------------


def cauchy_combination(
    p_values: list[float],
    weights: list[float] | None = None,
) -> float:
    """Combine p-values using the Cauchy combination test.

    The Cauchy combination test (Liu & Xie, 2020) transforms p-values
    to Cauchy statistics: T_i = tan((0.5 - p_i) * pi), then combines
    as weighted sum T = sum(w_i * T_i). Under the null, T follows a
    standard Cauchy distribution.

    Robust to arbitrary correlation between tests.

    Parameters
    ----------
    p_values : list[float]
        P-values to combine.
    weights : list[float] | None
        Non-negative weights (default: equal). Will be normalized to sum to 1.

    Returns
    -------
    Combined p-value.
    """
    valid_p = [p for p in p_values if np.isfinite(p) and 0 < p <= 1]
    if not valid_p:
        return 1.0
    if len(valid_p) == 1:
        return valid_p[0]

    p_arr = np.array(valid_p)

    if weights is not None:
        w = np.array(weights[:len(valid_p)], dtype=float)
    else:
        w = np.ones(len(valid_p))

    # Normalize weights
    w_sum = w.sum()
    if w_sum <= 0:
        return 1.0
    w = w / w_sum

    # Cauchy transform: T_i = tan((0.5 - p_i) * pi)
    cauchy_stats = np.tan((0.5 - p_arr) * np.pi)

    # Weighted sum
    T = float(np.sum(w * cauchy_stats))

    # P-value from standard Cauchy CDF: P(T > t) = 0.5 - arctan(t)/pi
    combined_p = 0.5 - np.arctan(T) / np.pi

    return float(np.clip(combined_p, 0.0, 1.0))


def combine_celltype_methods(
    method_results: dict[str, dict[str, float]],
    cell_type_labels: dict[str, str] | None = None,
    method_weights: dict[str, float] | None = None,
) -> list[CauchyCombinedResult]:
    """Combine p-values across multiple cell-type enrichment methods.

    Parameters
    ----------
    method_results : dict
        Method name -> dict of cell_type -> p-value.
        E.g. {"magma_celltype": {"msn": 0.001, ...}, "cepo_magma": {"msn": 0.01, ...}}
    cell_type_labels : dict | None
        Human-readable labels for cell types.
    method_weights : dict | None
        Weights per method. Default: equal weights.

    Returns
    -------
    List of CauchyCombinedResult sorted by combined p-value.
    """
    # Collect all cell types across methods
    all_cell_types: set[str] = set()
    for pvals in method_results.values():
        all_cell_types.update(pvals.keys())

    methods = list(method_results.keys())
    labels = cell_type_labels or {}

    results = []
    for ct in sorted(all_cell_types):
        ct_pvals = {}
        p_list = []
        w_list = []

        for method in methods:
            if ct in method_results[method]:
                p = method_results[method][ct]
                ct_pvals[method] = p
                p_list.append(p)
                w = (method_weights or {}).get(method, 1.0)
                w_list.append(w)

        if not p_list:
            continue

        if len(p_list) == 1:
            combined_p = p_list[0]
            cauchy_stat = np.tan((0.5 - p_list[0]) * np.pi)
        else:
            combined_p = cauchy_combination(p_list, w_list)
            # Compute the stat for reporting
            w_arr = np.array(w_list)
            w_arr = w_arr / w_arr.sum()
            cauchy_stat = float(np.sum(
                w_arr * np.tan((0.5 - np.array(p_list)) * np.pi)
            ))

        results.append(CauchyCombinedResult(
            cell_type=ct,
            label=labels.get(ct, ct),
            n_methods=len(p_list),
            method_pvalues=ct_pvals,
            cauchy_stat=cauchy_stat,
            combined_p=combined_p,
        ))

    # Sort by combined p-value
    results.sort(key=lambda r: r.combined_p)

    # FDR correction
    n = len(results)
    for i, r in enumerate(results):
        r.fdr_q = min(r.combined_p * n / (i + 1), 1.0)
    for i in range(n - 2, -1, -1):
        results[i].fdr_q = min(results[i].fdr_q, results[i + 1].fdr_q)

    logger.info("Cauchy combination: %d cell types, %d methods, %d at P < 0.05",
                len(results), len(methods),
                sum(1 for r in results if r.combined_p < 0.05))
    return results


# ---------------------------------------------------------------------------
# FDR correction for Cepo results
# ---------------------------------------------------------------------------


def _apply_fdr_cepo(results: list[CepoCelltypeResult]) -> None:
    """Apply Benjamini-Hochberg FDR correction in-place."""
    if not results:
        return
    results.sort(key=lambda r: r.p_value)
    n = len(results)
    for i, r in enumerate(results):
        r.fdr_q = min(r.p_value * n / (i + 1), 1.0)
    for i in range(n - 2, -1, -1):
        results[i].fdr_q = min(results[i].fdr_q, results[i + 1].fdr_q)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_cepo_results(
    results: list[CepoCelltypeResult],
    output_path: Path,
) -> Path:
    """Write Cepo-MAGMA enrichment results to TSV."""
    rows = [{
        "CELL_TYPE": r.cell_type,
        "LABEL": r.label,
        "N_CEPO_GENES": r.n_cepo_genes,
        "N_GENES_TESTED": r.n_genes_tested,
        "BETA": r.beta,
        "BETA_SE": r.beta_se,
        "Z": r.z_score,
        "P": r.p_value,
        "FDR_Q": r.fdr_q,
        "TOP_GENES": ";".join(r.top_genes),
    } for r in results]

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote %d Cepo results to %s", len(rows), output_path)
    return output_path


def write_cauchy_results(
    results: list[CauchyCombinedResult],
    output_path: Path,
) -> Path:
    """Write Cauchy combined results to TSV."""
    rows = []
    for r in results:
        row = {
            "CELL_TYPE": r.cell_type,
            "LABEL": r.label,
            "N_METHODS": r.n_methods,
            "CAUCHY_STAT": r.cauchy_stat,
            "COMBINED_P": r.combined_p,
            "FDR_Q": r.fdr_q,
        }
        for method, p in sorted(r.method_pvalues.items()):
            row[f"P_{method}"] = p
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote %d Cauchy combined results to %s", len(rows), output_path)
    return output_path


def write_summary(
    cepo_results: list[CepoCelltypeResult],
    cauchy_results: list[CauchyCombinedResult],
    output_dir: Path,
) -> Path:
    """Write human-readable summary of Cepo + Cauchy analysis."""
    lines = ["# Cepo + Cauchy Combination Summary\n"]

    # Cepo-MAGMA section
    lines.append("## Cepo-MAGMA Enrichment\n")
    lines.append(f"- Cell types tested: {len(cepo_results)}")
    sig = sum(1 for r in cepo_results if r.fdr_q < 0.05)
    sug = sum(1 for r in cepo_results if r.fdr_q < 0.25)
    lines.append(f"- Significant (FDR < 0.05): {sig}")
    lines.append(f"- Suggestive (FDR < 0.25): {sug}")

    if cepo_results:
        lines.append(f"\n| Cell Type | N Tested | Beta | Z | P | FDR | Top Genes |")
        lines.append("|-----------|----------|------|---|---|-----|-----------|")
        for r in cepo_results:
            top = ", ".join(r.top_genes[:3])
            lines.append(
                f"| {r.label} | {r.n_genes_tested} | {r.beta:.3f} | "
                f"{r.z_score:.2f} | {r.p_value:.2e} | {r.fdr_q:.3f} | {top} |"
            )

    # Cauchy combination section
    lines.append("\n## Cauchy Combined Results\n")
    lines.append(f"- Cell types combined: {len(cauchy_results)}")
    sig_c = sum(1 for r in cauchy_results if r.fdr_q < 0.05)
    sug_c = sum(1 for r in cauchy_results if r.fdr_q < 0.25)
    lines.append(f"- Significant (FDR < 0.05): {sig_c}")
    lines.append(f"- Suggestive (FDR < 0.25): {sug_c}")

    if cauchy_results:
        lines.append(f"\n| Cell Type | N Methods | Cauchy Stat | Combined P | FDR |")
        lines.append("|-----------|-----------|-------------|------------|-----|")
        for r in cauchy_results:
            lines.append(
                f"| {r.label} | {r.n_methods} | {r.cauchy_stat:.2f} | "
                f"{r.combined_p:.2e} | {r.fdr_q:.3f} |"
            )

    lines.append("")
    path = output_dir / "cepo_cauchy_summary.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote summary to %s", path)
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_cepo_cauchy(
    gene_results_path: Path,
    expression_path: Path | None = None,
    cell_labels_path: Path | None = None,
    magma_celltype_path: Path | None = None,
    markers: dict[str, list[str]] | None = None,
    labels: dict[str, str] | None = None,
    protein_coding_path: Path | None = None,
    output_dir: Path | None = None,
    top_n: int = 200,
    min_genes: int = 5,
) -> tuple[list[CepoCelltypeResult], list[CauchyCombinedResult]]:
    """Run Cepo + Cauchy combination pipeline.

    Parameters
    ----------
    gene_results_path : Path
        TSV from MAGMA gene analysis (GENE, Z, P, N_SNPS).
    expression_path : Path | None
        scRNA-seq expression matrix (genes x cells). If None, uses markers.
    cell_labels_path : Path | None
        Cell-type labels for expression matrix columns.
    magma_celltype_path : Path | None
        Existing MAGMA-celltype results TSV for Cauchy combination.
    markers : dict | None
        Cell-type marker genes. Used when no scRNA-seq data available.
    labels : dict | None
        Human-readable cell-type labels.
    protein_coding_path : Path | None
        File with one protein-coding gene name per line.
    output_dir : Path | None
        Output directory. Defaults to celltype_enrichment dir.
    top_n : int
        Number of top Cepo genes per cell type.
    min_genes : int
        Minimum gene overlap for enrichment testing.

    Returns
    -------
    Tuple of (cepo_results, cauchy_combined_results).
    """
    ensure_dirs()
    if output_dir is None:
        output_dir = CELLTYPE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gene-level results
    from bioagentics.tourettes.ts_gwas_functional_annotation.celltype_enrichment import (
        load_gene_results,
    )
    gene_df = load_gene_results(gene_results_path)
    if gene_df.empty:
        logger.error("No gene results loaded — cannot run Cepo + Cauchy")
        return [], []

    all_genes = gene_df["GENE"].tolist()

    # Load protein-coding gene list if available
    protein_coding = None
    if protein_coding_path is not None and protein_coding_path.exists():
        protein_coding = set(protein_coding_path.read_text().strip().split("\n"))
        logger.info("Loaded %d protein-coding genes", len(protein_coding))

    # Step 1: Compute Cepo scores
    if expression_path is not None and cell_labels_path is not None:
        expr_df = pd.read_csv(expression_path, sep="\t", index_col=0)
        cell_labels = pd.read_csv(cell_labels_path, sep="\t", header=None).iloc[:, 0]
        cepo_scores = compute_cepo_scores(expr_df, cell_labels, top_n=top_n)
    else:
        # Use marker genes as fallback
        if markers is None:
            from bioagentics.tourettes.ts_gwas_functional_annotation.celltype_enrichment import (
                BRAIN_CELLTYPE_MARKERS,
            )
            markers = BRAIN_CELLTYPE_MARKERS
        cepo_scores = cepo_from_markers(markers, all_genes, top_n=top_n)

    # Filter to protein-coding genes
    cepo_scores = filter_protein_coding(cepo_scores, protein_coding)

    # Step 2: MAGMA-GSEA enrichment of Cepo gene sets
    cepo_results = cepo_magma_gsea(gene_df, cepo_scores, min_genes, labels)

    # Write Cepo results
    if cepo_results:
        write_cepo_results(cepo_results, output_dir / "cepo_magma.tsv")

    # Step 3: Cauchy combination across methods
    method_pvals: dict[str, dict[str, float]] = {}

    # Add Cepo-MAGMA p-values
    method_pvals["cepo_magma"] = {
        r.cell_type: r.p_value for r in cepo_results
    }

    # Load existing MAGMA-celltype results if available
    if magma_celltype_path is not None and magma_celltype_path.exists():
        magma_ct_df = pd.read_csv(magma_celltype_path, sep="\t")
        if {"CELL_TYPE", "P"}.issubset(magma_ct_df.columns):
            # Use marginal results only for combination
            if "ANALYSIS" in magma_ct_df.columns:
                magma_ct_df = magma_ct_df[magma_ct_df["ANALYSIS"] == "marginal"]
            method_pvals["magma_celltype"] = dict(
                zip(magma_ct_df["CELL_TYPE"], magma_ct_df["P"])
            )
            logger.info("Loaded %d MAGMA-celltype p-values",
                        len(method_pvals["magma_celltype"]))

    cauchy_results = combine_celltype_methods(
        method_pvals,
        cell_type_labels=labels,
    )

    # Write outputs
    if cauchy_results:
        write_cauchy_results(cauchy_results, output_dir / "cepo_cauchy.tsv")

    # Summary
    write_summary(cepo_results, cauchy_results, output_dir)

    return cepo_results, cauchy_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cepo + Cauchy combination for GWAS-to-single-cell integration."
    )
    parser.add_argument(
        "--gene-results", type=Path, required=True,
        help="Gene-level results TSV from MAGMA gene analysis.",
    )
    parser.add_argument(
        "--expression", type=Path, default=None,
        help="scRNA-seq expression matrix (genes x cells TSV).",
    )
    parser.add_argument(
        "--cell-labels", type=Path, default=None,
        help="Cell-type labels for expression matrix.",
    )
    parser.add_argument(
        "--magma-celltype", type=Path, default=None,
        help="Existing MAGMA-celltype results TSV.",
    )
    parser.add_argument(
        "--protein-coding", type=Path, default=None,
        help="Protein-coding gene list (one per line).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--top-n", type=int, default=200,
        help="Top Cepo genes per cell type (default: 200).",
    )
    parser.add_argument(
        "--min-genes", type=int, default=5,
        help="Minimum overlapping genes (default: 5).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.gene_results.exists():
        logger.error("Gene results file not found: %s", args.gene_results)
        sys.exit(1)

    run_cepo_cauchy(
        gene_results_path=args.gene_results,
        expression_path=args.expression,
        cell_labels_path=args.cell_labels,
        magma_celltype_path=args.magma_celltype,
        protein_coding_path=args.protein_coding,
        output_dir=args.output_dir,
        top_n=args.top_n,
        min_genes=args.min_genes,
    )


if __name__ == "__main__":
    main()
