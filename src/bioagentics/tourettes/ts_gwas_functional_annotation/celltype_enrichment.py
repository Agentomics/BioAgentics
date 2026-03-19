"""MAGMA cell-type enrichment analysis for TS GWAS.

Implements MAGMA-celltype methodology for testing enrichment of GWAS
signal in specific brain cell types using scRNA-seq reference data.

Two analysis modes:
  1. Marker-based: competitive regression per cell type (binary membership)
  2. Specificity-based: gene property test using continuous expression specificity

Both support conditional analysis to identify independently enriched cell types.

Usage:
    python -m bioagentics.tourettes.ts_gwas_functional_annotation.celltype_enrichment \
        --gene-results output/tourettes/ts-gwas-functional-annotation/magma_results/gene_results.tsv
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
# Brain cell-type marker gene definitions
# ---------------------------------------------------------------------------

BRAIN_CELLTYPE_MARKERS: dict[str, list[str]] = {
    # Striatal medium spiny neurons (general)
    "msn": [
        "DRD1", "DRD2", "PPP1R1B", "PENK", "TAC1",
        "ADORA2A", "GPR88", "FOXP1", "ISL1", "EBF1",
        "BCL11B", "FOXP2", "MEIS2", "RGS9", "ARPP21",
    ],
    # D1 MSNs (striatonigral / direct pathway)
    "d1_msn": [
        "DRD1", "TAC1", "PDYN", "ISL1", "EBF1",
        "FOXP1", "PPP1R1B", "GPR88", "RGS9", "SLC35D3",
    ],
    # D2 MSNs (striatopallidal / indirect pathway)
    "d2_msn": [
        "DRD2", "PENK", "ADORA2A", "GPR6", "SP9",
        "PPP1R1B", "GPR88", "FOXP1", "RGS9", "ARPP21",
    ],
    # GABAergic interneurons (striatal)
    "striatal_interneuron": [
        "SST", "PVALB", "NPY", "NOS1", "CALB1",
        "GAD1", "GAD2", "SLC32A1", "TH", "CALB2",
    ],
    # Cholinergic interneurons (striatal)
    "cholinergic_interneuron": [
        "CHAT", "SLC5A7", "SLC18A3", "CHRM2", "CHRNA4",
        "ACHE", "ISL1", "GBX2", "LHX8", "NKX2-1",
    ],
    # Cortical excitatory (glutamatergic) neurons
    "cortical_excitatory": [
        "SLC17A7", "CAMK2A", "NRGN", "SATB2", "CUX2",
        "RORB", "TLE4", "FEZF2", "BCL6", "FOXP2",
    ],
    # Cortical PV+ interneurons
    "cortical_pv_interneuron": [
        "PVALB", "GAD1", "GAD2", "SLC32A1", "ERBB4",
        "LHX6", "SOX6", "TAC1", "KCNC1", "KCNC2",
    ],
    # Cortical SST+ interneurons
    "cortical_sst_interneuron": [
        "SST", "GAD1", "GAD2", "SLC32A1", "NPY",
        "CORT", "LHX6", "SOX6", "ELFN1", "CHRNA2",
    ],
    # Cortical VIP+ interneurons
    "cortical_vip_interneuron": [
        "VIP", "GAD1", "GAD2", "SLC32A1", "CALB2",
        "CCK", "CRH", "ADARB2", "PROX1", "CNR1",
    ],
    # Astrocytes
    "astrocyte": [
        "GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1",
        "GJA1", "SOX9", "S100B", "GLUL", "NDRG2",
    ],
    # Microglia
    "microglia": [
        "CX3CR1", "AIF1", "ITGAM", "CSF1R", "TREM2",
        "P2RY12", "HEXB", "TMEM119", "CD68", "CD163",
    ],
    # Oligodendrocytes
    "oligodendrocyte": [
        "MBP", "PLP1", "MOG", "MAG", "OLIG2",
        "SOX10", "CNP", "MOBP", "CLDN11", "ERMN",
    ],
    # Oligodendrocyte precursor cells
    "opc": [
        "PDGFRA", "CSPG4", "OLIG1", "OLIG2", "SOX10",
        "GPR17", "PCDH15", "LHFPL3", "VCAN", "BCAN",
    ],
    # Dopaminergic neurons (substantia nigra / VTA)
    "dopaminergic": [
        "TH", "SLC6A3", "DDC", "SLC18A2", "NR4A2",
        "PITX3", "FOXA2", "EN1", "LMX1B", "KCNJ6",
    ],
}

# Human-readable labels for cell types
CELLTYPE_LABELS: dict[str, str] = {
    "msn": "Medium Spiny Neurons",
    "d1_msn": "D1 MSNs (Direct Pathway)",
    "d2_msn": "D2 MSNs (Indirect Pathway)",
    "striatal_interneuron": "Striatal GABAergic Interneurons",
    "cholinergic_interneuron": "Cholinergic Interneurons",
    "cortical_excitatory": "Cortical Excitatory Neurons",
    "cortical_pv_interneuron": "Cortical PV+ Interneurons",
    "cortical_sst_interneuron": "Cortical SST+ Interneurons",
    "cortical_vip_interneuron": "Cortical VIP+ Interneurons",
    "astrocyte": "Astrocytes",
    "microglia": "Microglia",
    "oligodendrocyte": "Oligodendrocytes",
    "opc": "OPCs",
    "dopaminergic": "Dopaminergic Neurons",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CelltypeResult:
    """Cell-type enrichment result from MAGMA-celltype analysis."""

    cell_type: str
    label: str
    analysis_type: str  # "marginal" or "conditional"
    n_marker_genes: int
    n_genes_tested: int
    beta: float
    beta_se: float
    z_score: float
    p_value: float
    fdr_q: float = 1.0
    top_genes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading gene-level results
# ---------------------------------------------------------------------------


def load_gene_results(path: Path) -> pd.DataFrame:
    """Load gene-level results from MAGMA gene analysis TSV.

    Expected columns: GENE, Z (z-score), P, N_SNPS.
    """
    if not path.exists():
        logger.warning("Gene results file not found: %s", path)
        return pd.DataFrame(columns=["GENE", "Z", "P", "N_SNPS"])

    df = pd.read_csv(path, sep="\t")
    required = {"GENE", "Z", "P"}
    if not required.issubset(df.columns):
        logger.error("Gene results missing columns: %s", required - set(df.columns))
        return pd.DataFrame(columns=["GENE", "Z", "P", "N_SNPS"])

    # Drop genes with missing Z-scores
    df = df.dropna(subset=["Z"])
    logger.info("Loaded %d gene results from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Loading scRNA-seq specificity data
# ---------------------------------------------------------------------------


def load_specificity_matrix(path: Path) -> pd.DataFrame | None:
    """Load cell-type specificity matrix from scRNA-seq reference.

    Expected format: TSV with GENE column + one column per cell type,
    values are specificity scores (0-1, proportion of total expression).
    """
    if not path.exists():
        logger.info("No specificity matrix at %s — using marker genes", path)
        return None

    df = pd.read_csv(path, sep="\t")
    if "GENE" not in df.columns:
        logger.warning("Specificity matrix missing GENE column")
        return None

    df = df.set_index("GENE")
    logger.info("Loaded specificity matrix: %d genes x %d cell types",
                len(df), len(df.columns))
    return df


def markers_to_specificity(
    markers: dict[str, list[str]],
    all_genes: list[str],
) -> pd.DataFrame:
    """Convert marker gene lists to a binary specificity matrix.

    Creates a genes x cell_types DataFrame where 1 = marker, 0 = not.
    """
    gene_set = set(all_genes)
    data = {}
    for ct, genes in markers.items():
        data[ct] = [1.0 if g in set(genes) else 0.0 for g in all_genes]
    return pd.DataFrame(data, index=all_genes)


# ---------------------------------------------------------------------------
# Marginal cell-type enrichment (one cell type at a time)
# ---------------------------------------------------------------------------


def _compute_enrichment(
    gene_z: np.ndarray,
    indicator: np.ndarray,
    covariates: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Run enrichment regression: Z ~ indicator + covariates.

    Returns (beta, se, z_score, p_value).
    """
    n = len(gene_z)
    if n < 10:
        return 0.0, 1.0, 0.0, 1.0

    # Build design matrix: intercept + covariates (filtered) + indicator
    parts = [np.ones(n)]
    if covariates is not None:
        cov = covariates.reshape(n, -1) if covariates.ndim == 1 else covariates
        for col_idx in range(cov.shape[1]):
            col = cov[:, col_idx]
            # Skip constant columns (zero variance) to avoid singularity
            if np.var(col) > 1e-10:
                parts.append(col)
    parts.append(indicator)  # test variable is last column

    X = np.column_stack(parts)
    test_idx = X.shape[1] - 1

    try:
        beta, residuals, _, _ = np.linalg.lstsq(X, gene_z, rcond=None)
        beta_test = float(beta[test_idx])

        # Compute standard error
        resid = gene_z - X @ beta
        df = max(n - X.shape[1], 1)
        mse = float(np.sum(resid ** 2)) / df

        XtX_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt(max(mse * XtX_inv[test_idx, test_idx], 1e-15)))
        z = beta_test / se if se > 0 else 0.0
        p = float(stats.norm.sf(z))  # one-sided (enrichment)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 1.0, 0.0, 1.0

    return beta_test, se, z, p


def marginal_celltype_analysis(
    gene_df: pd.DataFrame,
    specificity: pd.DataFrame,
    min_genes: int = 5,
) -> list[CelltypeResult]:
    """Test each cell type for GWAS enrichment independently (marginal).

    For each cell type, regress gene Z-scores on cell-type specificity
    (or marker indicator), controlling for gene size (log N_SNPS).
    """
    genes = gene_df["GENE"].tolist()
    z_scores = gene_df["Z"].values.astype(float)

    # Covariate: log(n_snps) if available
    covariates = None
    if "N_SNPS" in gene_df.columns:
        log_nsnps = np.log1p(gene_df["N_SNPS"].values.astype(float))
        covariates = log_nsnps.reshape(-1, 1)

    # Align specificity to gene order
    spec_genes = set(specificity.index)
    gene_mask = [g in spec_genes for g in genes]

    results = []
    for ct in specificity.columns:
        # Get specificity values aligned to gene_df
        ct_vals = np.array([
            float(specificity.loc[g, ct]) if g in spec_genes else 0.0
            for g in genes
        ])

        n_marker = int(np.sum(ct_vals > 0))
        n_overlap = int(np.sum(np.array(gene_mask) & (ct_vals > 0)))
        if n_overlap < min_genes:
            continue

        beta, se, z, p = _compute_enrichment(z_scores, ct_vals, covariates)

        # Top contributing genes (highest Z among markers)
        marker_idx = np.where(ct_vals > 0)[0]
        marker_z = [(genes[i], z_scores[i]) for i in marker_idx]
        marker_z.sort(key=lambda x: -x[1])
        top = [g for g, _ in marker_z[:5]]

        results.append(CelltypeResult(
            cell_type=ct,
            label=CELLTYPE_LABELS.get(ct) or ct,
            analysis_type="marginal",
            n_marker_genes=n_marker,
            n_genes_tested=n_overlap,
            beta=beta,
            beta_se=se,
            z_score=z,
            p_value=p,
            top_genes=top,
        ))

    _apply_fdr(results)
    logger.info("Marginal analysis: %d cell types tested, %d at P < 0.05",
                len(results), sum(1 for r in results if r.p_value < 0.05))
    return results


# ---------------------------------------------------------------------------
# Conditional cell-type enrichment
# ---------------------------------------------------------------------------


def conditional_celltype_analysis(
    gene_df: pd.DataFrame,
    specificity: pd.DataFrame,
    min_genes: int = 5,
) -> list[CelltypeResult]:
    """Test each cell type while conditioning on all other cell types.

    For cell type j: Z ~ specificity_j + sum(specificity_k, k != j) + covariates.
    This identifies cell types with independent enrichment signal.
    """
    genes = gene_df["GENE"].tolist()
    z_scores = gene_df["Z"].values.astype(float)

    # Build full specificity matrix aligned to gene_df
    spec_genes = set(specificity.index)
    ct_names = list(specificity.columns)
    spec_matrix = np.zeros((len(genes), len(ct_names)))
    for col_idx, ct in enumerate(ct_names):
        for row_idx, g in enumerate(genes):
            if g in spec_genes:
                spec_matrix[row_idx, col_idx] = float(specificity.loc[g, ct])

    # Add log(n_snps) covariate if available
    base_cov = None
    if "N_SNPS" in gene_df.columns:
        base_cov = np.log1p(gene_df["N_SNPS"].values.astype(float)).reshape(-1, 1)

    results = []
    for test_idx, ct in enumerate(ct_names):
        ct_vals = spec_matrix[:, test_idx]
        n_overlap = int(np.sum(ct_vals > 0))
        if n_overlap < min_genes:
            continue

        # Other cell types as conditioning covariates
        other_cols = [i for i in range(len(ct_names)) if i != test_idx]
        if other_cols:
            other_spec = spec_matrix[:, other_cols]
            # Remove columns with zero variance
            col_vars = np.var(other_spec, axis=0)
            keep = col_vars > 1e-10
            other_spec = other_spec[:, keep]
        else:
            other_spec = None

        # Combine covariates
        cov_parts = []
        if base_cov is not None:
            cov_parts.append(base_cov)
        if other_spec is not None and other_spec.shape[1] > 0:
            cov_parts.append(other_spec)

        covariates = np.hstack(cov_parts) if cov_parts else None

        beta, se, z, p = _compute_enrichment(z_scores, ct_vals, covariates)

        # Top genes
        marker_idx = np.where(ct_vals > 0)[0]
        marker_z = [(genes[i], z_scores[i]) for i in marker_idx]
        marker_z.sort(key=lambda x: -x[1])
        top = [g for g, _ in marker_z[:5]]

        n_marker = int(np.sum(ct_vals > 0))
        results.append(CelltypeResult(
            cell_type=ct,
            label=CELLTYPE_LABELS.get(ct) or ct,
            analysis_type="conditional",
            n_marker_genes=n_marker,
            n_genes_tested=n_overlap,
            beta=beta,
            beta_se=se,
            z_score=z,
            p_value=p,
            top_genes=top,
        ))

    _apply_fdr(results)
    logger.info("Conditional analysis: %d cell types tested, %d at P < 0.05",
                len(results), sum(1 for r in results if r.p_value < 0.05))
    return results


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------


def _apply_fdr(results: list[CelltypeResult]) -> None:
    """Apply Benjamini-Hochberg FDR correction in-place."""
    if not results:
        return
    results.sort(key=lambda r: r.p_value)
    n = len(results)
    for i, r in enumerate(results):
        r.fdr_q = min(r.p_value * n / (i + 1), 1.0)
    # Ensure monotonicity
    for i in range(n - 2, -1, -1):
        results[i].fdr_q = min(results[i].fdr_q, results[i + 1].fdr_q)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_celltype_results(
    results: list[CelltypeResult],
    output_path: Path,
) -> Path:
    """Write cell-type enrichment results to TSV."""
    rows = [{
        "CELL_TYPE": r.cell_type,
        "LABEL": r.label,
        "ANALYSIS": r.analysis_type,
        "N_MARKER_GENES": r.n_marker_genes,
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
    logger.info("Wrote %d cell-type results to %s", len(rows), output_path)
    return output_path


def write_summary(
    marginal: list[CelltypeResult],
    conditional: list[CelltypeResult],
    output_dir: Path,
) -> Path:
    """Write human-readable summary of cell-type enrichment analysis."""
    lines = ["# MAGMA Cell-Type Enrichment Summary\n"]

    for label, results in [("Marginal", marginal), ("Conditional", conditional)]:
        lines.append(f"## {label} Analysis\n")
        lines.append(f"- Cell types tested: {len(results)}")
        sig = sum(1 for r in results if r.fdr_q < 0.05)
        sug = sum(1 for r in results if r.fdr_q < 0.25)
        lines.append(f"- Significant (FDR < 0.05): {sig}")
        lines.append(f"- Suggestive (FDR < 0.25): {sug}")

        if results:
            lines.append(f"\n| Cell Type | N Tested | Beta | Z | P | FDR | Top Genes |")
            lines.append("|-----------|----------|------|---|---|-----|-----------|")
            for r in results:
                top = ", ".join(r.top_genes[:3])
                lines.append(
                    f"| {r.label} | {r.n_genes_tested} | {r.beta:.3f} | "
                    f"{r.z_score:.2f} | {r.p_value:.2e} | {r.fdr_q:.3f} | {top} |"
                )
        lines.append("")

    path = output_dir / "celltype_summary.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote summary to %s", path)
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_celltype_enrichment(
    gene_results_path: Path,
    specificity_path: Path | None = None,
    markers: dict[str, list[str]] | None = None,
    output_dir: Path | None = None,
    min_genes: int = 5,
) -> tuple[list[CelltypeResult], list[CelltypeResult]]:
    """Run MAGMA cell-type enrichment analysis.

    Parameters
    ----------
    gene_results_path : Path
        TSV from MAGMA gene analysis (GENE, Z, P, N_SNPS).
    specificity_path : Path | None
        scRNA-seq specificity matrix TSV. If None, uses marker genes.
    markers : dict | None
        Cell-type marker gene dict. Defaults to BRAIN_CELLTYPE_MARKERS.
    output_dir : Path | None
        Output directory. Defaults to celltype_enrichment dir.
    min_genes : int
        Minimum marker genes overlapping tested genes.

    Returns
    -------
    Tuple of (marginal_results, conditional_results).
    """
    ensure_dirs()
    if output_dir is None:
        output_dir = CELLTYPE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gene-level results
    gene_df = load_gene_results(gene_results_path)
    if gene_df.empty:
        logger.error("No gene results loaded — cannot run cell-type analysis")
        return [], []

    # Load or build specificity matrix
    specificity = None
    if specificity_path is not None:
        specificity = load_specificity_matrix(specificity_path)

    if specificity is None:
        if markers is None:
            markers = BRAIN_CELLTYPE_MARKERS
        specificity = markers_to_specificity(markers, gene_df["GENE"].tolist())
        logger.info("Using marker-gene approach: %d cell types, %d genes",
                    len(specificity.columns), len(specificity))

    # Run marginal analysis
    marginal = marginal_celltype_analysis(gene_df, specificity, min_genes)

    # Run conditional analysis
    conditional = conditional_celltype_analysis(gene_df, specificity, min_genes)

    # Write outputs
    if marginal:
        write_celltype_results(
            marginal, output_dir / "magma_celltype_marginal.tsv"
        )
    if conditional:
        write_celltype_results(
            conditional, output_dir / "magma_celltype_conditional.tsv"
        )

    # Combined output as requested by task spec
    all_results = marginal + conditional
    if all_results:
        write_celltype_results(all_results, output_dir / "magma_celltype.tsv")

    # Summary
    write_summary(marginal, conditional, output_dir)

    return marginal, conditional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MAGMA cell-type enrichment analysis for TS GWAS."
    )
    parser.add_argument(
        "--gene-results", type=Path, required=True,
        help="Gene-level results TSV from MAGMA gene analysis.",
    )
    parser.add_argument(
        "--specificity", type=Path, default=None,
        help="scRNA-seq specificity matrix TSV (optional).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for cell-type results.",
    )
    parser.add_argument(
        "--min-genes", type=int, default=5,
        help="Minimum marker genes overlapping tested genes (default: 5).",
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

    run_celltype_enrichment(
        gene_results_path=args.gene_results,
        specificity_path=args.specificity,
        output_dir=args.output_dir,
        min_genes=args.min_genes,
    )


if __name__ == "__main__":
    main()
