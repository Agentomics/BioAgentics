"""MAGMA gene-based and gene-set analysis for TS GWAS.

Pure-Python implementation of MAGMA-style gene analysis and competitive
gene-set enrichment testing. Designed for TS GWAS summary statistics with
support for external pathway collections (MSigDB, GO, KEGG, Reactome).

Steps:
  1. Map SNPs to genes (positional, using gene annotations + window)
  2. Gene-based analysis: combine SNP P-values per gene (top-SNP + Brown's)
  3. Gene-set analysis: competitive regression of gene Z-scores on set membership
  4. FDR correction across all tested gene sets

Usage:
    python -m bioagentics.tourettes.ts_gwas_functional_annotation.magma_analysis \
        --gwas data/tourettes/ts-gwas-functional-annotation/gwas/tsaicg_cleaned.tsv \
        --gene-annot data/tourettes/ts-gwas-functional-annotation/gene_annotations/genes.tsv
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
    GENE_ANNOT_DIR,
    GENE_SETS_DIR,
    GWAS_DIR,
    MAGMA_DIR,
    POSITIONAL_WINDOW_KB,
    TS_CANDIDATE_GENES,
    TS_DE_NOVO_GENES,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GeneResult:
    """Gene-level association from MAGMA-style gene analysis."""

    gene: str
    chr: int
    start: int
    stop: int
    n_snps: int
    top_snp: str
    top_snp_p: float
    stat: float
    p_value: float
    z_score: float
    is_candidate: bool = False
    is_de_novo: bool = False


@dataclass
class GeneSetResult:
    """Gene-set enrichment result from competitive analysis."""

    gene_set: str
    source: str  # msigdb_c2, msigdb_c5_go, kegg, reactome, builtin
    n_genes_defined: int
    n_genes_tested: int
    beta: float
    beta_se: float
    z_score: float
    p_value: float
    fdr_q: float = 1.0
    top_genes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in TS-relevant gene sets (used when no external GMT files available)
# ---------------------------------------------------------------------------

BUILTIN_GENE_SETS: dict[str, list[str]] = {
    "dopamine_signaling": [
        "DRD1", "DRD2", "DRD3", "DRD4", "DRD5", "TH", "DDC",
        "SLC6A3", "COMT", "MAOA", "MAOB", "DBH",
    ],
    "serotonin_signaling": [
        "HTR1A", "HTR1B", "HTR2A", "HTR2C", "HTR3A", "HTR4",
        "TPH1", "TPH2", "SLC6A4", "MAOA",
    ],
    "gaba_signaling": [
        "GABRA1", "GABRA2", "GABRB1", "GABRB2", "GABRG2",
        "GAD1", "GAD2", "SLC32A1", "SLC6A1", "ABAT",
    ],
    "glutamate_signaling": [
        "GRIN1", "GRIN2A", "GRIN2B", "GRIA1", "GRIA2", "GRM1", "GRM5",
        "SLC17A6", "SLC17A7", "SLC1A2", "SLC1A3", "GLUL",
    ],
    "synaptic_adhesion": [
        "NRXN1", "NRXN2", "NRXN3", "NLGN1", "NLGN2", "NLGN3",
        "CNTN6", "CNTNAP2", "LRRC4", "SLITRK1", "SLITRK5",
    ],
    "axon_guidance": [
        "SEMA3A", "SEMA6D", "NTN1", "NTN4", "DCC", "ROBO1", "ROBO2",
        "SLIT1", "SLIT2", "EPHB2", "EPHA4", "PLXNA2",
    ],
    "striatal_msn_markers": [
        "DRD1", "DRD2", "PPP1R1B", "PENK", "TAC1",
        "ADORA2A", "GPR88", "FOXP1", "ISL1", "EBF1",
    ],
    "interneuron_markers": [
        "GAD1", "GAD2", "SLC32A1", "SST", "PVALB", "VIP",
        "CALB1", "CALB2", "NPY", "CCK",
    ],
    "cortical_excitatory_markers": [
        "SLC17A7", "CAMK2A", "NRGN", "SATB2", "CUX2",
        "RORB", "TLE4", "FEZF2", "BCL6", "FOXP2",
    ],
    "neurodevelopment": [
        "BCL11B", "FOXP2", "TBR1", "SATB2", "DLX1", "DLX2",
        "NKX2-1", "LHX6", "ARX", "ASCL1", "NEUROD1", "PAX6",
    ],
}


# ---------------------------------------------------------------------------
# Gene-set loading
# ---------------------------------------------------------------------------


def load_gmt(path: Path) -> dict[str, list[str]]:
    """Load gene sets from GMT format file.

    GMT: gene_set_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
    """
    gene_sets: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                name = parts[0]
                genes = [g for g in parts[2:] if g]
                if genes:
                    gene_sets[name] = genes
    return gene_sets


def load_all_gene_sets(
    gene_sets_dir: Path | None = None,
) -> dict[str, tuple[str, list[str]]]:
    """Load gene sets from all available sources.

    Returns dict: gene_set_name -> (source, gene_list).
    """
    if gene_sets_dir is None:
        gene_sets_dir = GENE_SETS_DIR

    result: dict[str, tuple[str, list[str]]] = {}

    # Load built-in sets
    for name, genes in BUILTIN_GENE_SETS.items():
        result[name] = ("builtin", genes)

    if not gene_sets_dir.exists():
        logger.info("Gene sets dir not found: %s — using built-in sets only", gene_sets_dir)
        return result

    # Map filename patterns to source labels
    source_patterns = {
        "c2": "msigdb_c2",
        "c5": "msigdb_c5_go",
        "kegg": "kegg",
        "reactome": "reactome",
        "go_bp": "go_bp",
        "go_mf": "go_mf",
        "go_cc": "go_cc",
    }

    for gmt_file in sorted(gene_sets_dir.glob("*.gmt")):
        fname_lower = gmt_file.stem.lower()
        source = "custom"
        for pattern, src_label in source_patterns.items():
            if pattern in fname_lower:
                source = src_label
                break

        try:
            gmt_sets = load_gmt(gmt_file)
            for name, genes in gmt_sets.items():
                result[name] = (source, genes)
            logger.info("Loaded %d gene sets from %s (source: %s)",
                        len(gmt_sets), gmt_file.name, source)
        except Exception as e:
            logger.warning("Failed to load GMT file %s: %s", gmt_file, e)

    logger.info("Total gene sets loaded: %d", len(result))
    return result


# ---------------------------------------------------------------------------
# SNP-to-gene mapping for MAGMA
# ---------------------------------------------------------------------------


def map_snps_to_genes(
    gwas_df: pd.DataFrame,
    gene_annot: pd.DataFrame,
    window_kb: int = POSITIONAL_WINDOW_KB,
) -> dict[str, pd.DataFrame]:
    """Map SNPs to genes using positional annotation.

    Returns dict: gene_name -> DataFrame of SNPs within window.
    """
    if gene_annot.empty or not {"SNP", "CHR", "BP", "P"}.issubset(gwas_df.columns):
        return {}

    window = window_kb * 1000
    gene_snps: dict[str, pd.DataFrame] = {}

    for chr_val in gwas_df["CHR"].unique():
        chr_gwas = gwas_df[gwas_df["CHR"] == chr_val]
        chr_genes = gene_annot[gene_annot["CHR"] == chr_val]
        if chr_genes.empty:
            continue

        for _, gene in chr_genes.iterrows():
            gene_start = gene["START"] - window
            gene_stop = gene["STOP"] + window
            mask = (chr_gwas["BP"] >= gene_start) & (chr_gwas["BP"] <= gene_stop)
            snps = chr_gwas[mask]
            if len(snps) > 0:
                gene_snps[gene["GENE"]] = snps

    logger.info("Mapped SNPs to %d genes", len(gene_snps))
    return gene_snps


# ---------------------------------------------------------------------------
# Gene-based analysis
# ---------------------------------------------------------------------------


def gene_analysis(
    gwas_df: pd.DataFrame,
    gene_annot: pd.DataFrame,
    window_kb: int = POSITIONAL_WINDOW_KB,
) -> list[GeneResult]:
    """Compute gene-level association statistics.

    Uses top-SNP method with Bonferroni correction within gene, combined
    with Brown's method for correlated SNP P-values. Reports the more
    conservative of the two P-values.
    """
    gene_snps = map_snps_to_genes(gwas_df, gene_annot, window_kb)
    if not gene_snps:
        logger.warning("No SNP-to-gene mappings; check gene annotations and GWAS columns")
        return []

    # Build gene info lookup
    gene_info = {}
    for _, row in gene_annot.iterrows():
        gene_info[row["GENE"]] = {
            "chr": int(row["CHR"]),
            "start": int(row["START"]),
            "stop": int(row["STOP"]),
        }

    candidate_set = set(TS_CANDIDATE_GENES)
    de_novo_set = set(TS_DE_NOVO_GENES)
    results = []

    for gene_name, snps_df in gene_snps.items():
        p_values = snps_df["P"].values
        p_values = p_values[np.isfinite(p_values) & (p_values > 0) & (p_values <= 1)]
        if len(p_values) == 0:
            continue

        n_snps = len(p_values)
        min_p = float(np.min(p_values))
        top_snp_idx = np.argmin(snps_df["P"].values)
        top_snp = str(snps_df["SNP"].iloc[top_snp_idx])

        # Method 1: Top-SNP with Bonferroni correction
        bonf_p = min(min_p * n_snps, 1.0)

        # Method 2: Brown's combined statistic
        chi2_stat = float(-2 * np.sum(np.log(p_values)))
        # Conservative df for correlated SNPs: 2 * sqrt(n)
        effective_df = max(2 * np.sqrt(n_snps), 2)
        brown_p = float(stats.chi2.sf(chi2_stat, effective_df))

        # Use more conservative P-value
        gene_p = min(max(bonf_p, brown_p), 1.0)
        z = float(stats.norm.isf(gene_p)) if 0 < gene_p < 1 else 0.0

        info = gene_info.get(gene_name, {"chr": 0, "start": 0, "stop": 0})

        results.append(GeneResult(
            gene=gene_name,
            chr=info["chr"],
            start=info["start"],
            stop=info["stop"],
            n_snps=n_snps,
            top_snp=top_snp,
            top_snp_p=min_p,
            stat=chi2_stat,
            p_value=gene_p,
            z_score=z,
            is_candidate=gene_name in candidate_set,
            is_de_novo=gene_name in de_novo_set,
        ))

    results.sort(key=lambda r: r.p_value)
    logger.info("Gene analysis: %d genes tested, %d at P < 0.05, %d at P < 2.5e-6",
                len(results),
                sum(1 for r in results if r.p_value < 0.05),
                sum(1 for r in results if r.p_value < 2.5e-6))
    return results


# ---------------------------------------------------------------------------
# Gene-set enrichment (competitive test)
# ---------------------------------------------------------------------------


def gene_set_analysis(
    gene_results: list[GeneResult],
    gene_sets: dict[str, tuple[str, list[str]]],
    min_genes: int = 5,
    max_genes: int = 5000,
) -> list[GeneSetResult]:
    """Competitive gene-set enrichment analysis.

    Tests whether genes in a set have higher Z-scores than background
    using linear regression (MAGMA competitive test).

    Parameters
    ----------
    gene_results : list[GeneResult]
        Gene-level results from gene_analysis.
    gene_sets : dict
        Gene-set definitions: name -> (source, gene_list).
    min_genes : int
        Minimum genes in set that overlap tested genes (default: 5).
    max_genes : int
        Maximum gene-set size to test (default: 5000).
    """
    if not gene_results:
        return []

    gene_names = [r.gene for r in gene_results]
    gene_z = np.array([r.z_score for r in gene_results])
    gene_name_set = set(gene_names)
    n = len(gene_z)

    results = []
    for gs_name, (source, gs_genes) in gene_sets.items():
        if len(gs_genes) > max_genes:
            continue

        genes_in_data = [g for g in gs_genes if g in gene_name_set]
        n_in = len(genes_in_data)
        if n_in < min_genes:
            continue

        # Binary indicator for set membership
        member_set = set(genes_in_data)
        indicator = np.array([1.0 if g in member_set else 0.0 for g in gene_names])

        # Competitive regression: Z_g = beta_0 + beta_1 * I(g in set) + eps
        X = np.column_stack([np.ones(n), indicator])

        try:
            beta, residuals, _, _ = np.linalg.lstsq(X, gene_z, rcond=None)
            beta1 = float(beta[1])

            # Standard error
            if len(residuals) > 0 and residuals[0] > 0:
                mse = residuals[0] / max(n - 2, 1)
            else:
                resid = gene_z - X @ beta
                mse = float(np.sum(resid ** 2)) / max(n - 2, 1)

            XtX_inv = np.linalg.inv(X.T @ X)
            se = float(np.sqrt(max(mse * XtX_inv[1, 1], 1e-15)))

            z_score = beta1 / se if se > 0 else 0.0
            p_value = float(stats.norm.sf(z_score))  # one-sided enrichment
        except (np.linalg.LinAlgError, ValueError):
            beta1 = 0.0
            se = 1.0
            z_score = 0.0
            p_value = 1.0

        # Identify top contributing genes (highest Z in set)
        gene_z_in_set = [(g, gene_z[i]) for i, g in enumerate(gene_names) if g in member_set]
        gene_z_in_set.sort(key=lambda x: -x[1])
        top = [g for g, _ in gene_z_in_set[:5]]

        results.append(GeneSetResult(
            gene_set=gs_name,
            source=source,
            n_genes_defined=len(gs_genes),
            n_genes_tested=n_in,
            beta=beta1,
            beta_se=se,
            z_score=z_score,
            p_value=p_value,
            top_genes=top,
        ))

    # FDR correction (Benjamini-Hochberg)
    results.sort(key=lambda r: r.p_value)
    n_tests = len(results)
    for i, r in enumerate(results):
        r.fdr_q = min(r.p_value * n_tests / (i + 1), 1.0)

    # Ensure monotonicity
    for i in range(n_tests - 2, -1, -1):
        results[i].fdr_q = min(results[i].fdr_q, results[i + 1].fdr_q)

    logger.info("Gene-set analysis: %d sets tested, %d at FDR < 0.05, %d at FDR < 0.25",
                n_tests,
                sum(1 for r in results if r.fdr_q < 0.05),
                sum(1 for r in results if r.fdr_q < 0.25))
    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_gene_results(results: list[GeneResult], output_dir: Path) -> Path:
    """Write gene-level results to TSV."""
    rows = [{
        "GENE": r.gene,
        "CHR": r.chr,
        "START": r.start,
        "STOP": r.stop,
        "N_SNPS": r.n_snps,
        "TOP_SNP": r.top_snp,
        "TOP_SNP_P": r.top_snp_p,
        "STAT": r.stat,
        "P": r.p_value,
        "Z": r.z_score,
        "IS_CANDIDATE": r.is_candidate,
        "IS_DE_NOVO": r.is_de_novo,
    } for r in results]

    df = pd.DataFrame(rows)
    path = output_dir / "gene_results.tsv"
    df.to_csv(path, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote %d gene results to %s", len(rows), path)
    return path


def write_gene_set_results(results: list[GeneSetResult], output_dir: Path) -> Path:
    """Write gene-set enrichment results to TSV."""
    rows = [{
        "GENE_SET": r.gene_set,
        "SOURCE": r.source,
        "N_GENES_DEFINED": r.n_genes_defined,
        "N_GENES_TESTED": r.n_genes_tested,
        "BETA": r.beta,
        "BETA_SE": r.beta_se,
        "Z": r.z_score,
        "P": r.p_value,
        "FDR_Q": r.fdr_q,
        "TOP_GENES": ";".join(r.top_genes),
    } for r in results]

    df = pd.DataFrame(rows)
    path = output_dir / "gene_set_results.tsv"
    df.to_csv(path, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote %d gene-set results to %s", len(rows), path)
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_magma_analysis(
    gwas_path: Path,
    gene_annot_path: Path | None = None,
    gene_sets_dir: Path | None = None,
    output_dir: Path | None = None,
    window_kb: int = POSITIONAL_WINDOW_KB,
    min_genes: int = 5,
    max_genes: int = 5000,
) -> tuple[list[GeneResult], list[GeneSetResult]]:
    """Run full MAGMA-style gene and gene-set analysis pipeline.

    Parameters
    ----------
    gwas_path : Path
        Cleaned GWAS summary statistics (TSV with SNP, CHR, BP, P).
    gene_annot_path : Path | None
        Gene annotation file (GENE, CHR, START, STOP). Defaults to project dir.
    gene_sets_dir : Path | None
        Directory with GMT files. Defaults to project gene_sets dir.
    output_dir : Path | None
        Output directory. Defaults to magma_results dir.
    window_kb : int
        SNP-to-gene mapping window in kb.
    min_genes : int
        Minimum overlapping genes to test a gene set.
    max_genes : int
        Maximum gene-set size to test.

    Returns
    -------
    Tuple of (gene_results, gene_set_results).
    """
    ensure_dirs()
    if output_dir is None:
        output_dir = MAGMA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GWAS
    gwas_df = pd.read_csv(gwas_path, sep="\t")
    logger.info("Loaded GWAS: %d SNPs", len(gwas_df))

    # Load gene annotations
    if gene_annot_path is None:
        gene_annot_path = GENE_ANNOT_DIR / "genes.tsv"

    from bioagentics.tourettes.ts_gwas_functional_annotation.snp_to_gene import (
        load_gene_annotations,
    )
    gene_annot = load_gene_annotations(gene_annot_path)

    # Step 1: Gene-based analysis
    gene_results = gene_analysis(gwas_df, gene_annot, window_kb)
    if gene_results:
        write_gene_results(gene_results, output_dir)

    # Step 2: Load gene sets
    gene_sets = load_all_gene_sets(gene_sets_dir)

    # Step 3: Gene-set enrichment analysis
    gs_results = gene_set_analysis(gene_results, gene_sets, min_genes, max_genes)
    if gs_results:
        write_gene_set_results(gs_results, output_dir)

    # Step 4: Write summary report
    _write_summary(gene_results, gs_results, output_dir)

    return gene_results, gs_results


def _write_summary(
    gene_results: list[GeneResult],
    gs_results: list[GeneSetResult],
    output_dir: Path,
) -> None:
    """Write a concise summary of MAGMA analysis results."""
    lines = ["# MAGMA Analysis Summary\n"]

    lines.append(f"## Gene Analysis\n")
    lines.append(f"- Genes tested: {len(gene_results)}")
    sig_genes = [r for r in gene_results if r.p_value < 2.5e-6]
    sug_genes = [r for r in gene_results if r.p_value < 0.05]
    lines.append(f"- Genome-wide significant (P < 2.5e-6): {len(sig_genes)}")
    lines.append(f"- Nominally significant (P < 0.05): {len(sug_genes)}")

    if gene_results:
        lines.append(f"\n### Top 20 Genes\n")
        lines.append("| Gene | CHR | N_SNPs | Top SNP P | Gene P | Candidate |")
        lines.append("|------|-----|--------|-----------|--------|-----------|")
        for r in gene_results[:20]:
            flag = "GWAS" if r.is_candidate else ("DeNovo" if r.is_de_novo else "")
            lines.append(f"| {r.gene} | {r.chr} | {r.n_snps} | {r.top_snp_p:.2e} | {r.p_value:.2e} | {flag} |")

    lines.append(f"\n## Gene-Set Analysis\n")
    lines.append(f"- Gene sets tested: {len(gs_results)}")
    sig_gs = [r for r in gs_results if r.fdr_q < 0.05]
    sug_gs = [r for r in gs_results if r.fdr_q < 0.25]
    lines.append(f"- Significant (FDR < 0.05): {len(sig_gs)}")
    lines.append(f"- Suggestive (FDR < 0.25): {len(sug_gs)}")

    if gs_results:
        lines.append(f"\n### Top 20 Gene Sets\n")
        lines.append("| Gene Set | Source | N Tested | Z | P | FDR | Top Genes |")
        lines.append("|----------|--------|----------|---|---|-----|-----------|")
        for r in gs_results[:20]:
            top = ", ".join(r.top_genes[:3])
            lines.append(f"| {r.gene_set} | {r.source} | {r.n_genes_tested} | {r.z_score:.2f} | {r.p_value:.2e} | {r.fdr_q:.3f} | {top} |")

    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote summary to %s", path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MAGMA gene-based and gene-set analysis for TS GWAS."
    )
    parser.add_argument(
        "--gwas", type=Path, required=True,
        help="Cleaned GWAS summary statistics file (TSV).",
    )
    parser.add_argument(
        "--gene-annot", type=Path, default=None,
        help="Gene annotation file (TSV: GENE, CHR, START, STOP).",
    )
    parser.add_argument(
        "--gene-sets-dir", type=Path, default=None,
        help="Directory containing GMT gene-set files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--window-kb", type=int, default=POSITIONAL_WINDOW_KB,
        help=f"SNP-to-gene window in kb (default: {POSITIONAL_WINDOW_KB}).",
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

    if not args.gwas.exists():
        logger.error("GWAS file not found: %s", args.gwas)
        sys.exit(1)

    run_magma_analysis(
        gwas_path=args.gwas,
        gene_annot_path=args.gene_annot,
        gene_sets_dir=args.gene_sets_dir,
        output_dir=args.output_dir,
        window_kb=args.window_kb,
    )


if __name__ == "__main__":
    main()
