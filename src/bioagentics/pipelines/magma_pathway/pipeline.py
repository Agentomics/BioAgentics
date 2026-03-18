"""MAGMA gene-set analysis pipeline for pathway decomposition.

Implements MAGMA-style gene and gene-set analysis on factor-specific GWAS
to identify comorbidity-axis-specific pathways. For each PRS stratum:
  - Gene analysis: maps SNPs to genes, computes gene-level associations
  - Gene-set analysis: tests enrichment against curated pathway collections

Uses a pure-Python implementation of the core MAGMA methodology:
  - Gene analysis via Brown's method for combining correlated SNP P-values
  - Gene-set competitive enrichment via linear regression of gene Z on
    gene-set membership

Pathway collections tested:
  - Brain cell-type markers
  - CSTC circuit gene sets
  - Neurotransmitter pathway gene sets (glutamate, dopamine, serotonin, GABA)
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

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in gene-set definitions for TS-relevant pathways
# ---------------------------------------------------------------------------

CSTC_GENE_SETS = {
    "cstc_cortical": [
        "SLC17A7", "GRIN1", "GRIN2A", "GRIN2B", "GRM5", "DLG4",
        "CAMK2A", "NRGN", "SNAP25", "SYT1",
    ],
    "cstc_striatal": [
        "DRD1", "DRD2", "DARPP32", "PPP1R1B", "PENK", "TAC1",
        "ADORA2A", "SLC6A3", "TH", "COMT",
    ],
    "cstc_thalamic": [
        "GABRA1", "GABRB2", "SLC32A1", "GAD1", "GAD2",
        "CACNA1G", "SCN1A", "HCN1", "KCNA1", "KCNC1",
    ],
    "cstc_pallidal": [
        "GAD1", "GAD2", "SLC32A1", "GABRA1", "GPR88",
        "FOXP2", "DRD2", "OPRM1", "PENK", "NPY",
    ],
}

NEUROTRANSMITTER_GENE_SETS = {
    "glutamate_signaling": [
        "GRIN1", "GRIN2A", "GRIN2B", "GRIA1", "GRIA2", "GRM1", "GRM5",
        "SLC17A6", "SLC17A7", "SLC1A2", "SLC1A3", "GLUL",
    ],
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
}

BRAIN_CELL_TYPE_MARKERS = {
    "excitatory_neurons": [
        "SLC17A7", "CAMK2A", "NRGN", "SATB2", "CUX2",
        "RORB", "TLE4", "FEZF2", "BCL6", "FOXP2",
    ],
    "inhibitory_interneurons": [
        "GAD1", "GAD2", "SLC32A1", "SST", "PVALB", "VIP",
        "CALB1", "CALB2", "NPY", "CCK",
    ],
    "medium_spiny_neurons": [
        "DRD1", "DRD2", "PPP1R1B", "PENK", "TAC1",
        "ADORA2A", "GPR88", "FOXP1", "ISL1", "EBF1",
    ],
    "cholinergic_interneurons": [
        "CHAT", "SLC5A7", "SLC18A3", "CHRM2", "CHRNA4",
        "ACHE", "ISL1", "GBX2", "LHX8", "NKX2-1",
    ],
    "astrocytes": [
        "GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1",
        "GJA1", "SOX9", "S100B", "GLUL", "NDRG2",
    ],
    "microglia": [
        "CX3CR1", "AIF1", "ITGAM", "CSF1R", "TREM2",
        "P2RY12", "HEXB", "TMEM119", "CD68", "CD163",
    ],
    "oligodendrocytes": [
        "MBP", "PLP1", "MOG", "MAG", "OLIG2",
        "SOX10", "CNP", "MOBP", "CLDN11", "ERMN",
    ],
}

ALL_BUILTIN_GENE_SETS = {
    **CSTC_GENE_SETS,
    **NEUROTRANSMITTER_GENE_SETS,
    **BRAIN_CELL_TYPE_MARKERS,
}


@dataclass
class GeneResult:
    """Gene-level association result from MAGMA gene analysis."""

    gene: str
    n_snps: int
    stat: float  # combined test statistic
    p_value: float
    z_score: float  # z-transformed p-value


@dataclass
class GeneSetResult:
    """Gene-set enrichment result from MAGMA gene-set analysis."""

    gene_set: str
    category: str  # e.g., "cstc", "neurotransmitter", "cell_type"
    n_genes: int
    n_genes_in_data: int
    beta: float  # regression coefficient
    beta_se: float
    z_score: float
    p_value: float
    fdr_q: float = 1.0  # filled after FDR correction


@dataclass
class PathwayAnalysisResult:
    """Full pathway analysis result for one GWAS stratum."""

    stratum: str
    gene_results: list[GeneResult]
    gene_set_results: list[GeneSetResult]
    top_genes: list[GeneResult] = field(default_factory=list)
    top_pathways: list[GeneSetResult] = field(default_factory=list)


def load_gene_annotations(path: Path | None = None) -> pd.DataFrame:
    """Load SNP-to-gene mapping annotations.

    Expected format: TSV with columns GENE, CHR, START, STOP
    If no file provided, returns empty DataFrame (user must supply).
    """
    if path is None or not path.exists():
        return pd.DataFrame(columns=["GENE", "CHR", "START", "STOP"])

    df = pd.read_csv(path, sep="\t")
    col_map = {
        "SYMBOL": "GENE",
        "gene_symbol": "GENE",
        "gene_name": "GENE",
        "CHROMOSOME": "CHR",
        "chr": "CHR",
        "START_POS": "START",
        "start": "START",
        "STOP_POS": "STOP",
        "stop": "STOP",
        "end": "STOP",
        "END": "STOP",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    return df


def load_gene_sets(path: Path | None = None) -> dict[str, list[str]]:
    """Load gene-set definitions from GMT or TSV file.

    GMT format: gene_set_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
    TSV format: GENE_SET<TAB>GENE columns

    If no file provided, returns built-in TS-relevant gene sets.
    """
    if path is None or not path.exists():
        return dict(ALL_BUILTIN_GENE_SETS)

    gene_sets: dict[str, list[str]] = {}

    if path.suffix == ".gmt":
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    name = parts[0]
                    genes = [g for g in parts[2:] if g]
                    gene_sets[name] = genes
    else:
        df = pd.read_csv(path, sep="\t")
        if "GENE_SET" in df.columns and "GENE" in df.columns:
            for gs_name, group in df.groupby("GENE_SET"):
                gene_sets[str(gs_name)] = group["GENE"].tolist()

    return gene_sets


def map_snps_to_genes(
    gwas_df: pd.DataFrame,
    gene_annot: pd.DataFrame,
    window_kb: int = 10,
) -> dict[str, pd.DataFrame]:
    """Map SNPs to genes using positional annotation + window.

    Parameters
    ----------
    gwas_df : pd.DataFrame
        GWAS results with SNP, CHR, BP, P columns.
    gene_annot : pd.DataFrame
        Gene annotations with GENE, CHR, START, STOP columns.
    window_kb : int
        Window around gene boundaries for SNP assignment (default: 10kb).

    Returns
    -------
    Dict mapping gene names to their associated SNP rows from gwas_df.
    """
    if gene_annot.empty:
        return {}

    required_gwas = {"SNP", "CHR", "BP", "P"}
    if not required_gwas.issubset(gwas_df.columns):
        logger.warning("GWAS missing positional columns (CHR, BP) for SNP-to-gene mapping")
        return {}

    window = window_kb * 1000
    gene_snps: dict[str, pd.DataFrame] = {}

    for _, gene in gene_annot.iterrows():
        gene_chr = gene["CHR"]
        gene_start = gene["START"] - window
        gene_stop = gene["STOP"] + window

        mask = (
            (gwas_df["CHR"] == gene_chr)
            & (gwas_df["BP"] >= gene_start)
            & (gwas_df["BP"] <= gene_stop)
        )
        snps = gwas_df[mask]
        if len(snps) > 0:
            gene_snps[gene["GENE"]] = snps

    logger.info("Mapped SNPs to %d genes", len(gene_snps))
    return gene_snps


def gene_analysis_snp_wise(
    gwas_df: pd.DataFrame,
    gene_snps: dict[str, pd.DataFrame] | None = None,
    gene_annot: pd.DataFrame | None = None,
    window_kb: int = 10,
) -> list[GeneResult]:
    """Compute gene-level statistics by combining SNP P-values.

    Uses the top-SNP method (minimum P-value per gene, Bonferroni corrected
    for number of SNPs in gene) as a simple approximation of MAGMA's
    multi-SNP model. When no positional info is available, assigns SNPs
    to synthetic gene blocks.

    Parameters
    ----------
    gwas_df : pd.DataFrame
        GWAS with SNP, P columns. Optionally CHR, BP.
    gene_snps : dict | None
        Pre-computed SNP-to-gene mapping. If None, computed from gene_annot.
    gene_annot : pd.DataFrame | None
        Gene annotations for mapping.
    window_kb : int
        Window for SNP-to-gene mapping.

    Returns
    -------
    List of GeneResult sorted by P-value.
    """
    if gene_snps is None:
        if gene_annot is not None and not gene_annot.empty:
            gene_snps = map_snps_to_genes(gwas_df, gene_annot, window_kb)
        else:
            # Fallback: create synthetic gene blocks from consecutive SNPs
            gene_snps = _synthetic_gene_blocks(gwas_df)

    results = []
    for gene, snps_df in gene_snps.items():
        n_snps = len(snps_df)
        if n_snps == 0:
            continue

        p_values = snps_df["P"].values
        p_values = p_values[np.isfinite(p_values) & (p_values > 0)]
        if len(p_values) == 0:
            continue

        # Top-SNP method with Bonferroni correction within gene
        min_p = float(np.min(p_values))
        corrected_p = min(min_p * len(p_values), 1.0)

        # Also compute Brown's combined statistic (sum of -2*log(p))
        chi2_stat = -2 * np.sum(np.log(p_values))
        # Under null with independent SNPs: chi2 with 2*n_snps df
        # Use a conservative df = 2 * sqrt(n_snps) for correlated SNPs
        effective_df = max(2 * np.sqrt(len(p_values)), 2)
        combined_p = float(stats.chi2.sf(chi2_stat, effective_df))

        # Use the more conservative P-value
        gene_p = max(corrected_p, combined_p)
        gene_p = min(gene_p, 1.0)

        z = float(stats.norm.isf(gene_p)) if gene_p < 1.0 else 0.0

        results.append(GeneResult(
            gene=gene,
            n_snps=len(p_values),
            stat=chi2_stat,
            p_value=gene_p,
            z_score=z,
        ))

    results.sort(key=lambda r: r.p_value)
    logger.info("Gene analysis: %d genes tested", len(results))
    return results


def _synthetic_gene_blocks(
    gwas_df: pd.DataFrame,
    block_size: int = 20,
) -> dict[str, pd.DataFrame]:
    """Create synthetic gene blocks from consecutive SNPs.

    Fallback when no gene annotations are available.
    """
    gene_snps: dict[str, pd.DataFrame] = {}
    df = gwas_df.sort_values("P").reset_index(drop=True)

    for i in range(0, len(df), block_size):
        block = df.iloc[i : i + block_size]
        gene_name = f"BLOCK_{i // block_size + 1}"
        gene_snps[gene_name] = block

    return gene_snps


def gene_set_analysis(
    gene_results: list[GeneResult],
    gene_sets: dict[str, list[str]],
) -> list[GeneSetResult]:
    """Competitive gene-set enrichment analysis.

    Tests whether genes in a gene set have higher association statistics
    than genes outside the set, using linear regression of gene Z-scores
    on gene-set membership (MAGMA's competitive test).

    Parameters
    ----------
    gene_results : list[GeneResult]
        Gene-level results from gene_analysis.
    gene_sets : dict[str, list[str]]
        Gene-set definitions (name -> gene list).

    Returns
    -------
    List of GeneSetResult sorted by P-value, with FDR correction.
    """
    if not gene_results:
        return []

    # Build gene-level Z-score vector
    gene_names = [r.gene for r in gene_results]
    gene_z = np.array([r.z_score for r in gene_results])
    gene_name_set = set(gene_names)

    # Categorize gene sets
    categories = {}
    for gs_name in gene_sets:
        if gs_name in CSTC_GENE_SETS:
            categories[gs_name] = "cstc"
        elif gs_name in NEUROTRANSMITTER_GENE_SETS:
            categories[gs_name] = "neurotransmitter"
        elif gs_name in BRAIN_CELL_TYPE_MARKERS:
            categories[gs_name] = "cell_type"
        else:
            categories[gs_name] = "custom"

    results = []
    for gs_name, gs_genes in gene_sets.items():
        genes_in_data = [g for g in gs_genes if g in gene_name_set]
        n_in = len(genes_in_data)

        if n_in < 2:
            continue

        # Binary indicator: is gene in this set?
        indicator = np.array([1.0 if g in set(genes_in_data) else 0.0 for g in gene_names])

        # Competitive test: regress Z on indicator
        # Z_g = beta_0 + beta_1 * I(g in set) + epsilon
        n = len(gene_z)
        X = np.column_stack([np.ones(n), indicator])

        try:
            beta, residuals, _, _ = np.linalg.lstsq(X, gene_z, rcond=None)
            beta1 = beta[1]

            # Standard error from residuals
            if len(residuals) > 0:
                mse = residuals[0] / (n - 2)
            else:
                resid = gene_z - X @ beta
                mse = float(np.sum(resid**2)) / max(n - 2, 1)

            XtX_inv = np.linalg.inv(X.T @ X)
            se = np.sqrt(max(mse * XtX_inv[1, 1], 1e-15))

            z_score = beta1 / se if se > 0 else 0.0
            p_value = float(stats.norm.sf(z_score))  # one-sided (enrichment)
        except np.linalg.LinAlgError:
            beta1 = 0.0
            se = 1.0
            z_score = 0.0
            p_value = 1.0

        results.append(GeneSetResult(
            gene_set=gs_name,
            category=categories.get(gs_name, "custom"),
            n_genes=len(gs_genes),
            n_genes_in_data=n_in,
            beta=float(beta1),
            beta_se=float(se),
            z_score=float(z_score),
            p_value=float(p_value),
        ))

    # FDR correction (Benjamini-Hochberg)
    results.sort(key=lambda r: r.p_value)
    n_tests = len(results)
    for i, r in enumerate(results):
        rank = i + 1
        r.fdr_q = min(r.p_value * n_tests / rank, 1.0)

    # Ensure monotonicity of FDR
    for i in range(n_tests - 2, -1, -1):
        results[i].fdr_q = min(results[i].fdr_q, results[i + 1].fdr_q)

    logger.info("Gene-set analysis: %d sets tested", len(results))
    return results


def run_pathway_analysis(
    gwas_dir: Path,
    gene_annot_path: Path | None = None,
    gene_sets_path: Path | None = None,
    output_dir: Path | None = None,
    window_kb: int = 10,
    n_top_genes: int = 20,
    n_top_pathways: int = 10,
) -> list[PathwayAnalysisResult]:
    """Run full MAGMA-style pathway analysis on factor-specific GWAS.

    Parameters
    ----------
    gwas_dir : Path
        Directory containing factor-specific GWAS files.
    gene_annot_path : Path | None
        Gene annotation file.
    gene_sets_path : Path | None
        Gene-set definitions file. If None, uses built-in sets.
    output_dir : Path | None
        Output directory.
    window_kb : int
        SNP-to-gene mapping window.
    n_top_genes : int
        Number of top genes to report per stratum.
    n_top_pathways : int
        Number of top pathways to report per stratum.

    Returns
    -------
    List of PathwayAnalysisResult, one per stratum.
    """
    if output_dir is None:
        output_dir = REPO_ROOT / "output" / "tourettes" / "ts-comorbidity-genetic-architecture" / "phase4"
    output_dir.mkdir(parents=True, exist_ok=True)

    gene_annot = load_gene_annotations(gene_annot_path)
    gene_sets = load_gene_sets(gene_sets_path)
    logger.info("Loaded %d gene sets", len(gene_sets))

    # Discover GWAS files
    strata_files = {
        "compulsive": ["compulsive.tsv", "compulsive_factor.tsv"],
        "neurodevelopmental": ["neurodevelopmental.tsv", "neurodevelopmental_factor.tsv"],
        "ts_specific": ["ts_specific.tsv", "ts_residual.tsv", "residual.tsv"],
    }

    all_results = []

    for stratum, filenames in strata_files.items():
        gwas_path = None
        for fn in filenames:
            candidate = gwas_dir / fn
            if candidate.exists():
                gwas_path = candidate
                break

        if gwas_path is None:
            logger.warning("No GWAS file for stratum %s", stratum)
            continue

        logger.info("Processing stratum: %s (%s)", stratum, gwas_path)

        # Load GWAS
        gwas_df = _load_gwas(gwas_path)

        # Gene analysis
        gene_results = gene_analysis_snp_wise(
            gwas_df, gene_annot=gene_annot, window_kb=window_kb
        )

        # Gene-set analysis
        gs_results = gene_set_analysis(gene_results, gene_sets)

        # Top results
        top_genes = gene_results[:n_top_genes]
        top_pathways = gs_results[:n_top_pathways]

        result = PathwayAnalysisResult(
            stratum=stratum,
            gene_results=gene_results,
            gene_set_results=gs_results,
            top_genes=top_genes,
            top_pathways=top_pathways,
        )
        all_results.append(result)

        # Write stratum outputs
        _write_gene_results(gene_results, stratum, output_dir)
        _write_gene_set_results(gs_results, stratum, output_dir)

    # Write cross-stratum comparison
    if all_results:
        _write_pathway_comparison(all_results, output_dir)

    return all_results


def _load_gwas(path: Path) -> pd.DataFrame:
    """Load and standardize GWAS summary statistics."""
    sep = "\t" if path.suffix in (".tsv", ".tab") else None
    df = pd.read_csv(path, sep=sep, engine="python")

    col_map = {
        "RSID": "SNP", "rsid": "SNP", "MarkerName": "SNP",
        "EFFECT_ALLELE": "A1", "ALT": "A1",
        "OTHER_ALLELE": "A2", "REF": "A2",
        "EFFECT": "BETA", "B": "BETA",
        "STDERR": "SE",
        "PVAL": "P", "P_VALUE": "P", "PVALUE": "P",
        "ZSCORE": "Z",
        "NMISS": "N",
        "CHROMOSOME": "CHR", "chr": "CHR",
        "POSITION": "BP", "pos": "BP", "POS": "BP", "bp": "BP",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    if "Z" not in df.columns and "BETA" in df.columns and "SE" in df.columns:
        mask = df["SE"] > 0
        df.loc[mask, "Z"] = df.loc[mask, "BETA"] / df.loc[mask, "SE"]

    if "P" not in df.columns and "Z" in df.columns:
        df["P"] = 2 * stats.norm.sf(np.abs(df["Z"]))

    df = df.dropna(subset=["SNP", "P"])
    df = df[df["P"] > 0]
    df = df.drop_duplicates(subset=["SNP"], keep="first")

    return df.reset_index(drop=True)


def _write_gene_results(
    results: list[GeneResult],
    stratum: str,
    output_dir: Path,
) -> None:
    """Write gene-level results to TSV."""
    if not results:
        return

    rows = [{
        "gene": r.gene,
        "n_snps": r.n_snps,
        "stat": r.stat,
        "p_value": r.p_value,
        "z_score": r.z_score,
    } for r in results]

    df = pd.DataFrame(rows)
    path = output_dir / f"gene_results_{stratum}.tsv"
    df.to_csv(path, sep="\t", index=False, float_format="%.6f")
    logger.info("Wrote %d gene results to %s", len(rows), path)


def _write_gene_set_results(
    results: list[GeneSetResult],
    stratum: str,
    output_dir: Path,
) -> None:
    """Write gene-set enrichment results to TSV."""
    if not results:
        return

    rows = [{
        "gene_set": r.gene_set,
        "category": r.category,
        "n_genes": r.n_genes,
        "n_genes_in_data": r.n_genes_in_data,
        "beta": r.beta,
        "beta_se": r.beta_se,
        "z_score": r.z_score,
        "p_value": r.p_value,
        "fdr_q": r.fdr_q,
    } for r in results]

    df = pd.DataFrame(rows)
    path = output_dir / f"gene_set_results_{stratum}.tsv"
    df.to_csv(path, sep="\t", index=False, float_format="%.6f")
    logger.info("Wrote %d gene-set results to %s", len(rows), path)


def _write_pathway_comparison(
    results: list[PathwayAnalysisResult],
    output_dir: Path,
) -> None:
    """Write cross-stratum pathway comparison table."""
    # Collect all gene sets tested across strata
    all_gene_sets: set[str] = set()
    for r in results:
        for gs in r.gene_set_results:
            all_gene_sets.add(gs.gene_set)

    if not all_gene_sets:
        return

    # Build comparison table: gene_set x stratum (P-values)
    rows = []
    for gs_name in sorted(all_gene_sets):
        row: dict[str, object] = {"gene_set": gs_name}
        for r in results:
            gs_match = [gs for gs in r.gene_set_results if gs.gene_set == gs_name]
            if gs_match:
                row[f"p_{r.stratum}"] = gs_match[0].p_value
                row[f"z_{r.stratum}"] = gs_match[0].z_score
                row[f"fdr_{r.stratum}"] = gs_match[0].fdr_q
            else:
                row[f"p_{r.stratum}"] = np.nan
                row[f"z_{r.stratum}"] = np.nan
                row[f"fdr_{r.stratum}"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    path = output_dir / "pathway_comparison.tsv"
    df.to_csv(path, sep="\t", index=False, float_format="%.6f")
    logger.info("Wrote pathway comparison for %d gene sets to %s", len(rows), path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the MAGMA pathway analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="MAGMA gene-set analysis on factor-specific GWAS results."
    )
    parser.add_argument(
        "--gwas-dir",
        type=Path,
        required=True,
        help="Directory containing factor-specific GWAS files.",
    )
    parser.add_argument(
        "--gene-annot",
        type=Path,
        default=None,
        help="Gene annotation file (TSV: GENE, CHR, START, STOP).",
    )
    parser.add_argument(
        "--gene-sets-dir",
        type=Path,
        default=None,
        help="Gene-set definitions file (GMT or TSV). Uses built-in sets if not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "tourettes" / "ts-comorbidity-genetic-architecture" / "phase4",
        help="Output directory.",
    )
    parser.add_argument(
        "--window-kb",
        type=int,
        default=10,
        help="SNP-to-gene mapping window in kb (default: 10).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.gwas_dir.exists():
        logger.error("GWAS directory not found: %s", args.gwas_dir)
        sys.exit(1)

    run_pathway_analysis(
        gwas_dir=args.gwas_dir,
        gene_annot_path=args.gene_annot,
        gene_sets_path=args.gene_sets_dir,
        output_dir=args.output_dir,
        window_kb=args.window_kb,
    )


if __name__ == "__main__":
    main()
