"""Phase 1b: Lymphocytic enrichment method validation.

Validate lymphocytic gene-set enrichment in the 2019 PGC TS GWAS using two
independent competitive methods:
  1. MAGMA-style competitive regression (gene Z-scores ~ set membership)
  2. Wilcoxon rank-sum competitive test (non-parametric alternative to seismic)

Compares results across methods and reports convergence/divergence with the
original Translational Psychiatry 2020 finding of lymphocytic GWAS enrichment.

Gene sets tested:
  - DICE lymphocyte subtypes: CD4_T, CD8_T, B_cell, NK, Th17, Treg, Gamma_delta_T
  - Non-lymphocyte controls: Monocyte, DC
  - Aggregated: lymphocyte_union (union of all lymphocyte subtype sets)
  - MSigDB-derived: KEGG_HEMATOPOIETIC_CELL_LINEAGE, REACTOME_ADAPTIVE_IMMUNE_SYSTEM

Usage:
    uv run python -m src.tourettes.ts_neuroimmune_subtyping.02_lymphocytic_enrichment
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from io import StringIO
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
GWAS_PATH = DATA_DIR / "ts_gwas_2019" / "TS_Oct2018.gz"
IMMUNE_REF = DATA_DIR / "immune_references"
GENE_ANNOT_CACHE = DATA_DIR / "gene_annotations"
OUTPUT_DIR = ROOT / "output" / "tourettes" / "ts-neuroimmune-subtyping" / "phase1b_lymphocytic_validation"

WINDOW_KB = 10  # SNP-to-gene mapping window
MIN_GENES_PER_SET = 10  # Minimum overlapping genes to test

# ---------------------------------------------------------------------------
# Gene annotations (Ensembl GRCh37)
# ---------------------------------------------------------------------------

BIOMART_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    "<!DOCTYPE Query>"
    '<Query virtualSchemaName="default" formatter="TSV" header="1" '
    'uniqueRows="1" count="" datasetConfigVersion="0.6">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Filter name="biotype" value="protein_coding"/>'
    '<Attribute name="ensembl_gene_id"/>'
    '<Attribute name="chromosome_name"/>'
    '<Attribute name="start_position"/>'
    '<Attribute name="end_position"/>'
    '<Attribute name="external_gene_name"/>'
    "</Dataset></Query>"
)
BIOMART_ENDPOINT = "https://grch37.ensembl.org/biomart/martservice"


def download_gene_annotations(cache_dir: Path) -> pd.DataFrame:
    """Download Ensembl GRCh37 protein-coding gene annotations via BioMart."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "ensembl_grch37_genes.tsv"

    if cache_file.exists():
        logger.info("Loading cached gene annotations from %s", cache_file)
        df = pd.read_csv(cache_file, sep="\t")
        logger.info("  %d genes loaded", len(df))
        return df

    logger.info("Downloading gene annotations from Ensembl GRCh37 BioMart...")
    data = urllib.parse.urlencode({"query": BIOMART_XML}).encode("utf-8")
    req = urllib.request.Request(
        BIOMART_ENDPOINT, data=data,
        headers={"User-Agent": "BioAgentics/1.0", "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        text = resp.read().decode("utf-8")

    df = pd.read_csv(StringIO(text), sep="\t")
    # Rename columns to standard names
    df.columns = ["GENE", "CHR", "START", "STOP", "SYMBOL"]
    # Keep only standard chromosomes
    valid_chr = sorted([str(i) for i in range(1, 23)] + ["X", "Y"])
    df = df[df["CHR"].astype(str).isin(valid_chr)].copy()
    df["CHR"] = df["CHR"].astype(str)
    df = df.dropna(subset=["GENE", "CHR", "START", "STOP"])
    df = df.drop_duplicates(subset="GENE")
    df.to_csv(cache_file, sep="\t", index=False)
    logger.info("  Cached %d protein-coding genes to %s", len(df), cache_file)
    return df


# ---------------------------------------------------------------------------
# GWAS loading (memory-safe: chunked for 8GB machine)
# ---------------------------------------------------------------------------


def load_gwas(path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Load raw TS GWAS, keeping only essential columns."""
    logger.info("Loading GWAS from %s (chunked)...", path)
    chunks = []
    for chunk in pd.read_csv(
        path, sep=r"\s+", usecols=["SNP", "CHR", "BP", "P"],
        dtype={"SNP": str, "CHR": str, "BP": "int32", "P": "float32"},
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["P"])
        chunk = chunk[chunk["P"] > 0]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    logger.info("  Loaded %d SNPs", len(df))
    return df


# ---------------------------------------------------------------------------
# SNP-to-gene mapping
# ---------------------------------------------------------------------------


def map_snps_to_genes(
    gwas: pd.DataFrame, genes: pd.DataFrame, window_kb: int = WINDOW_KB
) -> dict[str, pd.DataFrame]:
    """Positional SNP-to-gene mapping within a window."""
    window = window_kb * 1000
    result: dict[str, pd.DataFrame] = {}
    valid_chr = set(gwas["CHR"].unique()) & set(genes["CHR"].unique())

    for chrom in sorted(valid_chr):
        chr_gwas = gwas[gwas["CHR"] == chrom]
        chr_genes = genes[genes["CHR"] == chrom]
        if chr_genes.empty or chr_gwas.empty:
            continue

        bp = chr_gwas["BP"].values
        for _, gene in chr_genes.iterrows():
            g_start = gene["START"] - window
            g_stop = gene["STOP"] + window
            mask = (bp >= g_start) & (bp <= g_stop)
            snps = chr_gwas[mask]
            if len(snps) > 0:
                result[gene["GENE"]] = snps

    logger.info("Mapped SNPs to %d genes", len(result))
    return result


# ---------------------------------------------------------------------------
# Gene-level analysis
# ---------------------------------------------------------------------------


@dataclass
class GeneResult:
    gene: str
    n_snps: int
    top_snp_p: float
    gene_p: float
    z_score: float


def gene_analysis(gene_snps: dict[str, pd.DataFrame]) -> list[GeneResult]:
    """Compute gene-level P-values using top-SNP + Brown's combined method."""
    results = []
    for gene_id, snps in gene_snps.items():
        pvals = snps["P"].values.astype(float)
        pvals = pvals[(pvals > 0) & (pvals <= 1) & np.isfinite(pvals)]
        if len(pvals) == 0:
            continue

        n = len(pvals)
        min_p = float(np.min(pvals))

        # Bonferroni-corrected top-SNP
        bonf_p = min(min_p * n, 1.0)

        # Brown's combined statistic (conservative df for correlated SNPs)
        chi2 = float(-2 * np.sum(np.log(pvals)))
        eff_df = max(2 * np.sqrt(n), 2)
        brown_p = float(stats.chi2.sf(chi2, eff_df))

        gene_p = min(max(bonf_p, brown_p), 1.0)
        z = float(stats.norm.isf(gene_p)) if 0 < gene_p < 1 else 0.0

        results.append(GeneResult(
            gene=gene_id, n_snps=n, top_snp_p=min_p, gene_p=gene_p, z_score=z,
        ))

    results.sort(key=lambda r: r.gene_p)
    n_sig = sum(1 for r in results if r.gene_p < 0.05)
    logger.info("Gene analysis: %d genes tested, %d nominally significant (P<0.05)", len(results), n_sig)
    return results


# ---------------------------------------------------------------------------
# Gene-set definitions
# ---------------------------------------------------------------------------

# KEGG Hematopoietic Cell Lineage (hsa04640) — standard gene symbols
KEGG_HEMATOPOIETIC = [
    "IL1A", "IL1B", "IL1R1", "IL1R2", "IL2", "IL2RA", "IL3", "IL3RA",
    "IL4", "IL4R", "IL5", "IL5RA", "IL6", "IL6R", "IL7", "IL7R", "IL9",
    "IL9R", "IL11", "IL11RA", "CSF1", "CSF1R", "CSF2", "CSF2RA", "CSF2RB",
    "CSF3", "CSF3R", "EPO", "EPOR", "TPO", "MPL", "KITLG", "KIT",
    "FLT3LG", "FLT3", "TNF", "TNFRSF1A", "TNFRSF1B", "CD1A", "CD1B",
    "CD1C", "CD1D", "CD1E", "CD2", "CD3D", "CD3E", "CD3G", "CD4", "CD5",
    "CD7", "CD8A", "CD8B", "CD9", "CD10", "MME", "CD19", "CD20", "MS4A1",
    "CD22", "CD24", "CD25", "CD33", "CD34", "CD36", "CD38", "CD41",
    "ITGA2B", "CD42A", "GP9", "CD42B", "GP1BA", "CD42C", "GP1BB",
    "CD42D", "GP5", "CD44", "CD45", "PTPRC", "CD49D", "ITGA4",
    "CD55", "CD59", "CD61", "ITGB3", "CD64", "FCGR1A", "CD71",
    "TFRC", "CD117", "CD123", "IL3RA", "CD135", "GYPA", "GYPB",
    "HLA-DRA", "HLA-DRB1", "CR1", "CR2", "ANPEP",
]

# REACTOME Adaptive Immune System (R-HSA-1280218) — key members
REACTOME_ADAPTIVE = [
    "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "CD247", "ZAP70",
    "LCK", "FYN", "LAT", "SLP76", "LCP2", "PLCG1", "PLCG2", "ITK",
    "CARD11", "BCL10", "MALT1", "PRKCQ", "NFKB1", "NFKB2", "RELA",
    "REL", "NFKBIA", "NFAT", "NFATC1", "NFATC2", "NFATC3", "NFATC4",
    "AP1", "FOS", "JUN", "JUNB", "JUND", "IL2", "IL2RA", "IL2RB",
    "IL2RG", "JAK1", "JAK3", "STAT5A", "STAT5B", "CD28", "CTLA4",
    "ICOS", "PD1", "PDCD1", "CD274", "PDCD1LG2", "BTLA", "HVEM",
    "TNFRSF14", "CD40LG", "CD40", "TNFRSF4", "TNFSF4", "CD19",
    "CD79A", "CD79B", "BLNK", "BTK", "SYK", "BCAR1", "CBL", "VAV1",
    "VAV2", "VAV3", "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2",
    "AKT1", "AKT2", "AKT3", "MTOR", "HLA-A", "HLA-B", "HLA-C",
    "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "HLA-DRA",
    "HLA-DRB1", "B2M", "TAP1", "TAP2", "TAPBP", "CALR", "CANX",
    "PDIA3", "PSME1", "PSME2", "RAG1", "RAG2", "DNTT", "DCLRE1C",
    "PRKDC", "LIG4", "XRCC4", "NHEJ1", "CD80", "CD86", "ICAM1",
    "LFA1", "ITGAL", "ITGB2", "GRAP2", "ADAP1", "GRB2", "SOS1",
    "SOS2", "RASGRP1", "HRAS", "KRAS", "NRAS", "RAF1", "BRAF",
    "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
]

# Lymphocyte-specific innate immunity genes (NK cell markers + signaling)
NK_INNATE_LYMPHOID = [
    "NCAM1", "CD56", "FCGR3A", "CD16", "NCR1", "NCR2", "NCR3",
    "KLRK1", "NKG2D", "KLRD1", "CD94", "KLRC1", "NKG2A", "KLRB1",
    "KIR2DL1", "KIR2DL2", "KIR2DL3", "KIR2DL4", "KIR2DS1",
    "KIR2DS2", "KIR2DS4", "KIR3DL1", "KIR3DL2", "KIR3DS1",
    "PRF1", "GZMA", "GZMB", "GZMH", "GZMK", "GNLY", "FASLG",
    "TRAIL", "TNFSF10", "IFNG", "TNF", "CSF2", "IL2", "IL15",
    "IL15RA", "IL21", "IL21R", "EOMES", "TBX21", "TBET",
    "SH2D1A", "SH2D1B", "FCER1G", "TYROBP",
]


def load_dice_gene_sets(immune_ref_dir: Path) -> dict[str, set[str]]:
    """Load DICE cell-type gene sets (top-10% specific genes, Ensembl IDs)."""
    path = immune_ref_dir / "dice_magma_gene_sets.tsv"
    gene_sets: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                name = parts[0]
                genes = set(parts[1].split())
                gene_sets[name] = genes
    logger.info("Loaded %d DICE gene sets (total genes: %s)",
                len(gene_sets), {k: len(v) for k, v in gene_sets.items()})
    return gene_sets


def build_gene_sets(
    gene_annot: pd.DataFrame, immune_ref_dir: Path
) -> dict[str, set[str]]:
    """Build all gene sets for testing (Ensembl gene IDs)."""
    # Symbol-to-Ensembl mapping
    sym2ens = {}
    for _, row in gene_annot.iterrows():
        sym = str(row.get("SYMBOL", "")).upper()
        if sym and sym != "NAN":
            sym2ens[sym] = row["GENE"]

    def symbols_to_ensembl(symbols: list[str]) -> set[str]:
        return {sym2ens[s.upper()] for s in symbols if s.upper() in sym2ens}

    # DICE cell-type sets (already Ensembl IDs)
    dice_sets = load_dice_gene_sets(immune_ref_dir)

    # MSigDB-derived sets (symbol → Ensembl)
    kegg_ens = symbols_to_ensembl(KEGG_HEMATOPOIETIC)
    reactome_ens = symbols_to_ensembl(REACTOME_ADAPTIVE)
    nk_ens = symbols_to_ensembl(NK_INNATE_LYMPHOID)

    # Lymphocyte subtypes from DICE
    lympho_types = ["CD4_T", "CD8_T", "B_cell", "NK", "Th17", "Treg", "Gamma_delta_T"]
    lympho_union = set()
    for lt in lympho_types:
        if lt in dice_sets:
            lympho_union |= dice_sets[lt]

    all_sets: dict[str, set[str]] = {}

    # DICE individual cell types
    for name, genes in dice_sets.items():
        all_sets[f"DICE_{name}"] = genes

    # MSigDB-derived
    all_sets["KEGG_HEMATOPOIETIC_CELL_LINEAGE"] = kegg_ens
    all_sets["REACTOME_ADAPTIVE_IMMUNE_SYSTEM"] = reactome_ens
    all_sets["NK_INNATE_LYMPHOID"] = nk_ens

    # Aggregated lymphocyte set
    all_sets["lymphocyte_union"] = lympho_union
    all_sets["lymphocyte_union_plus_msigdb"] = lympho_union | kegg_ens | reactome_ens | nk_ens

    for name, genes in all_sets.items():
        logger.info("  Gene set %-40s: %d genes", name, len(genes))

    return all_sets


# ---------------------------------------------------------------------------
# Method 1: MAGMA competitive regression
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentResult:
    gene_set: str
    method: str
    n_genes_tested: int
    n_genes_in_set: int
    beta: float
    se: float
    z_score: float
    p_value: float
    fdr_q: float = 1.0
    mean_z_in_set: float = 0.0
    mean_z_background: float = 0.0
    top_genes: list[str] = field(default_factory=list)


def magma_competitive_regression(
    gene_results: list[GeneResult],
    gene_sets: dict[str, set[str]],
    min_genes: int = MIN_GENES_PER_SET,
) -> list[EnrichmentResult]:
    """MAGMA-style competitive gene-set analysis via linear regression."""
    gene_ids = [r.gene for r in gene_results]
    gene_z = np.array([r.z_score for r in gene_results])
    gene_id_set = set(gene_ids)
    n = len(gene_z)

    results = []
    for gs_name, gs_genes in gene_sets.items():
        overlap = gs_genes & gene_id_set
        n_in = len(overlap)
        if n_in < min_genes:
            continue

        indicator = np.array([1.0 if g in overlap else 0.0 for g in gene_ids])
        X = np.column_stack([np.ones(n), indicator])

        try:
            beta, residuals, _, _ = np.linalg.lstsq(X, gene_z, rcond=None)
            beta1 = float(beta[1])

            resid = gene_z - X @ beta
            mse = float(np.sum(resid ** 2)) / max(n - 2, 1)
            XtX_inv = np.linalg.inv(X.T @ X)
            se = float(np.sqrt(max(mse * XtX_inv[1, 1], 1e-15)))

            z = beta1 / se if se > 0 else 0.0
            p = float(stats.norm.sf(z))  # one-sided
        except (np.linalg.LinAlgError, ValueError):
            beta1, se, z, p = 0.0, 1.0, 0.0, 1.0

        # Mean Z inside vs outside
        z_in = gene_z[indicator == 1]
        z_out = gene_z[indicator == 0]
        mean_in = float(np.mean(z_in)) if len(z_in) > 0 else 0.0
        mean_out = float(np.mean(z_out)) if len(z_out) > 0 else 0.0

        # Top genes
        gene_z_pairs = [(gene_ids[i], gene_z[i]) for i in range(n) if gene_ids[i] in overlap]
        gene_z_pairs.sort(key=lambda x: -x[1])
        top = [g for g, _ in gene_z_pairs[:5]]

        results.append(EnrichmentResult(
            gene_set=gs_name, method="MAGMA_competitive", n_genes_tested=n,
            n_genes_in_set=n_in, beta=beta1, se=se, z_score=z, p_value=p,
            mean_z_in_set=mean_in, mean_z_background=mean_out, top_genes=top,
        ))

    _fdr_correct(results)
    return results


# ---------------------------------------------------------------------------
# Method 2: Wilcoxon rank-sum competitive test
# ---------------------------------------------------------------------------


def wilcoxon_competitive_test(
    gene_results: list[GeneResult],
    gene_sets: dict[str, set[str]],
    min_genes: int = MIN_GENES_PER_SET,
) -> list[EnrichmentResult]:
    """Non-parametric competitive enrichment using Wilcoxon rank-sum.

    Tests whether gene-level Z-scores within a set are significantly higher
    than background genes (one-sided). This is a distribution-free alternative
    to the MAGMA regression, comparable to competitive approaches used in
    tools like seismic (Nat Commun 2025).
    """
    gene_ids = [r.gene for r in gene_results]
    gene_z = np.array([r.z_score for r in gene_results])
    gene_id_set = set(gene_ids)

    results = []
    for gs_name, gs_genes in gene_sets.items():
        overlap = gs_genes & gene_id_set
        n_in = len(overlap)
        if n_in < min_genes:
            continue

        mask = np.array([g in overlap for g in gene_ids])
        z_in = gene_z[mask]
        z_out = gene_z[~mask]

        mean_in = float(np.mean(z_in))
        mean_out = float(np.mean(z_out))

        # Wilcoxon rank-sum (one-sided: gene set > background)
        stat_val, p_two = stats.mannwhitneyu(z_in, z_out, alternative="greater")

        # Effect size: rank-biserial correlation
        n1, n2 = len(z_in), len(z_out)
        r_rb = 1 - (2 * stat_val) / (n1 * n2) if (n1 * n2) > 0 else 0.0

        # Top genes
        gene_z_pairs = [(gene_ids[i], gene_z[i]) for i in range(len(gene_ids)) if gene_ids[i] in overlap]
        gene_z_pairs.sort(key=lambda x: -x[1])
        top = [g for g, _ in gene_z_pairs[:5]]

        results.append(EnrichmentResult(
            gene_set=gs_name, method="Wilcoxon_competitive",
            n_genes_tested=len(gene_z), n_genes_in_set=n_in,
            beta=r_rb, se=0.0, z_score=float(stats.norm.isf(p_two)) if 0 < p_two < 1 else 0.0,
            p_value=float(p_two),
            mean_z_in_set=mean_in, mean_z_background=mean_out, top_genes=top,
        ))

    _fdr_correct(results)
    return results


def _fdr_correct(results: list[EnrichmentResult]) -> None:
    """Benjamini-Hochberg FDR correction in place."""
    results.sort(key=lambda r: r.p_value)
    n = len(results)
    for i, r in enumerate(results):
        r.fdr_q = min(r.p_value * n / (i + 1), 1.0)
    for i in range(n - 2, -1, -1):
        results[i].fdr_q = min(results[i].fdr_q, results[i + 1].fdr_q)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results(
    magma_results: list[EnrichmentResult],
    wilcoxon_results: list[EnrichmentResult],
    gene_results: list[GeneResult],
    output_dir: Path,
) -> None:
    """Write all results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined enrichment results
    rows = []
    for r in magma_results + wilcoxon_results:
        rows.append({
            "gene_set": r.gene_set, "method": r.method,
            "n_genes_in_set": r.n_genes_in_set, "n_genes_tested": r.n_genes_tested,
            "beta": r.beta, "se": r.se, "z_score": r.z_score,
            "p_value": r.p_value, "fdr_q": r.fdr_q,
            "mean_z_in_set": r.mean_z_in_set, "mean_z_background": r.mean_z_background,
            "top_genes": ";".join(r.top_genes),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "enrichment_results.tsv", sep="\t", index=False, float_format="%.6g")

    # Gene-level results
    gene_rows = [{"gene": r.gene, "n_snps": r.n_snps, "top_snp_p": r.top_snp_p,
                   "gene_p": r.gene_p, "z_score": r.z_score} for r in gene_results]
    pd.DataFrame(gene_rows).to_csv(output_dir / "gene_results.tsv", sep="\t", index=False, float_format="%.6g")

    # Method comparison pivot
    pivot_rows = []
    magma_map = {r.gene_set: r for r in magma_results}
    wilcox_map = {r.gene_set: r for r in wilcoxon_results}
    all_sets = set(magma_map.keys()) | set(wilcox_map.keys())
    for gs in sorted(all_sets):
        m = magma_map.get(gs)
        w = wilcox_map.get(gs)
        pivot_rows.append({
            "gene_set": gs,
            "n_genes": m.n_genes_in_set if m else (w.n_genes_in_set if w else 0),
            "magma_z": m.z_score if m else None,
            "magma_p": m.p_value if m else None,
            "magma_fdr": m.fdr_q if m else None,
            "wilcoxon_z": w.z_score if w else None,
            "wilcoxon_p": w.p_value if w else None,
            "wilcoxon_fdr": w.fdr_q if w else None,
            "both_nominal": (m and w and m.p_value < 0.05 and w.p_value < 0.05),
        })
    pivot_df = pd.DataFrame(pivot_rows).sort_values("magma_p", na_position="last")
    pivot_df.to_csv(output_dir / "method_comparison.tsv", sep="\t", index=False, float_format="%.6g")

    # JSON summary
    lympho_sets = [s for s in all_sets if "lympho" in s.lower() or "DICE_CD4" in s
                   or "DICE_CD8" in s or "DICE_B_cell" in s or "DICE_NK" in s
                   or "DICE_Th17" in s or "DICE_Treg" in s or "DICE_Gamma" in s
                   or "KEGG_HEMATOPOIETIC" in s or "REACTOME_ADAPTIVE" in s
                   or "NK_INNATE" in s]
    non_lympho_sets = [s for s in all_sets if s.startswith("DICE_Monocyte") or s.startswith("DICE_DC")]

    lympho_sig_magma = sum(1 for s in lympho_sets if s in magma_map and magma_map[s].p_value < 0.05)
    lympho_sig_wilcox = sum(1 for s in lympho_sets if s in wilcox_map and wilcox_map[s].p_value < 0.05)
    control_sig_magma = sum(1 for s in non_lympho_sets if s in magma_map and magma_map[s].p_value < 0.05)
    control_sig_wilcox = sum(1 for s in non_lympho_sets if s in wilcox_map and wilcox_map[s].p_value < 0.05)

    summary = {
        "phase": "Phase 1b: Lymphocytic Enrichment Method Validation",
        "gwas": "Yu et al. 2019 PGC TS GWAS (N=14,307)",
        "n_genes_tested": len(gene_results),
        "n_gene_sets_tested": len(all_sets),
        "methods": ["MAGMA_competitive_regression", "Wilcoxon_rank_sum_competitive"],
        "lymphocyte_sets_tested": len(lympho_sets),
        "control_sets_tested": len(non_lympho_sets),
        "lymphocyte_nominal_p005": {
            "magma": lympho_sig_magma,
            "wilcoxon": lympho_sig_wilcox,
        },
        "control_nominal_p005": {
            "magma": control_sig_magma,
            "wilcoxon": control_sig_wilcox,
        },
        "results_by_set": {
            gs: {
                "n_genes": pivot_rows[i]["n_genes"],
                "magma_p": pivot_rows[i]["magma_p"],
                "wilcoxon_p": pivot_rows[i]["wilcoxon_p"],
                "converge": pivot_rows[i]["both_nominal"],
            }
            for i, gs in enumerate(sorted(all_sets))
        },
        "interpretation": "",
        "power_note": (
            "2019 TS GWAS (N=14,307) has limited power for gene-set enrichment. "
            "Null results do not exclude immune involvement. "
            "Future replication with 2024 TSAICG GWAS (N=19,138) expected to improve power."
        ),
    }

    # Generate interpretation
    any_lympho_sig = lympho_sig_magma > 0 or lympho_sig_wilcox > 0
    if any_lympho_sig:
        summary["interpretation"] = (
            "At least one lymphocytic gene set shows nominal enrichment (P<0.05) in the 2019 TS GWAS. "
            "This provides partial method validation of the original Translational Psychiatry 2020 "
            "lymphocytic enrichment finding using independent competitive methods."
        )
    else:
        summary["interpretation"] = (
            "No lymphocytic gene sets reached nominal significance (P<0.05) in either method. "
            "This is consistent with limited statistical power at N=14,307 rather than absence of "
            "lymphocytic involvement. The original finding used a different statistical framework "
            "(set-based association) which may have different sensitivity characteristics."
        )

    with open(output_dir / "phase1b_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Results written to %s", output_dir)


def write_report(
    magma_results: list[EnrichmentResult],
    wilcoxon_results: list[EnrichmentResult],
    gene_results: list[GeneResult],
    output_dir: Path,
) -> None:
    """Write markdown report."""
    lines = ["# Phase 1b: Lymphocytic Enrichment Method Validation\n"]
    lines.append(f"**GWAS**: Yu et al. 2019 PGC TS (N=14,307; 4,819 cases + 9,488 controls)")
    lines.append(f"**Genes tested**: {len(gene_results)}")
    lines.append(f"**Methods**: MAGMA competitive regression, Wilcoxon rank-sum competitive test\n")

    # Categorize sets
    lympho_names = {"DICE_CD4_T", "DICE_CD8_T", "DICE_B_cell", "DICE_NK", "DICE_Th17",
                    "DICE_Treg", "DICE_Gamma_delta_T", "KEGG_HEMATOPOIETIC_CELL_LINEAGE",
                    "REACTOME_ADAPTIVE_IMMUNE_SYSTEM", "NK_INNATE_LYMPHOID",
                    "lymphocyte_union", "lymphocyte_union_plus_msigdb"}
    control_names = {"DICE_Monocyte", "DICE_DC"}

    magma_map = {r.gene_set: r for r in magma_results}
    wilcox_map = {r.gene_set: r for r in wilcoxon_results}
    all_names = set(magma_map.keys()) | set(wilcox_map.keys())

    for section, names, label in [
        ("Lymphocyte Gene Sets", lympho_names & all_names, "lymphocyte"),
        ("Non-Lymphocyte Controls", control_names & all_names, "control"),
    ]:
        lines.append(f"\n## {section}\n")
        lines.append("| Gene Set | N Genes | MAGMA Z | MAGMA P | MAGMA FDR | Wilcoxon Z | Wilcoxon P | Wilcoxon FDR |")
        lines.append("|----------|---------|---------|---------|-----------|------------|------------|--------------|")
        for gs in sorted(names):
            m = magma_map.get(gs)
            w = wilcox_map.get(gs)
            ng = m.n_genes_in_set if m else (w.n_genes_in_set if w else 0)
            mz = f"{m.z_score:.2f}" if m else "—"
            mp = f"{m.p_value:.3e}" if m else "—"
            mf = f"{m.fdr_q:.3f}" if m else "—"
            wz = f"{w.z_score:.2f}" if w else "—"
            wp = f"{w.p_value:.3e}" if w else "—"
            wf = f"{w.fdr_q:.3f}" if w else "—"
            lines.append(f"| {gs} | {ng} | {mz} | {mp} | {mf} | {wz} | {wp} | {wf} |")

    lines.append("\n## Method Agreement\n")
    both_sig = []
    for gs in sorted(all_names):
        m = magma_map.get(gs)
        w = wilcox_map.get(gs)
        if m and w and m.p_value < 0.05 and w.p_value < 0.05:
            both_sig.append(gs)
    if both_sig:
        lines.append(f"Gene sets significant (P<0.05) in both methods: **{', '.join(both_sig)}**\n")
    else:
        lines.append("No gene sets reached P<0.05 in both methods simultaneously.\n")

    lines.append("\n## Power Note\n")
    lines.append("The 2019 TS GWAS (N=14,307) has limited power for gene-set enrichment analyses. ")
    lines.append("Null results are expected and do not exclude lymphocytic involvement. ")
    lines.append("Future re-analysis with the 2024 TSAICG GWAS (N=19,138) is planned.\n")

    (output_dir / "phase1b_report.md").write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s/phase1b_report.md", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== Phase 1b: Lymphocytic Enrichment Method Validation ===")

    # Step 1: Gene annotations
    logger.info("Step 1: Loading gene annotations...")
    gene_annot = download_gene_annotations(GENE_ANNOT_CACHE)

    # Step 2: Load GWAS
    logger.info("Step 2: Loading TS GWAS...")
    gwas = load_gwas(GWAS_PATH)

    # Step 3: SNP-to-gene mapping
    logger.info("Step 3: Mapping SNPs to genes (window=%dkb)...", WINDOW_KB)
    gene_snps = map_snps_to_genes(gwas, gene_annot, WINDOW_KB)

    # Free GWAS memory
    del gwas

    # Step 4: Gene-level analysis
    logger.info("Step 4: Computing gene-level statistics...")
    gene_results = gene_analysis(gene_snps)
    del gene_snps

    if not gene_results:
        logger.error("No gene results computed. Check GWAS/annotation overlap.")
        sys.exit(1)

    # Step 5: Build gene sets
    logger.info("Step 5: Building gene sets...")
    gene_sets = build_gene_sets(gene_annot, IMMUNE_REF)

    # Step 6: MAGMA competitive regression
    logger.info("Step 6: Running MAGMA competitive regression...")
    magma_results = magma_competitive_regression(gene_results, gene_sets)

    # Step 7: Wilcoxon rank-sum competitive test
    logger.info("Step 7: Running Wilcoxon rank-sum competitive test...")
    wilcoxon_results = wilcoxon_competitive_test(gene_results, gene_sets)

    # Step 8: Write results
    logger.info("Step 8: Writing results...")
    write_results(magma_results, wilcoxon_results, gene_results, OUTPUT_DIR)
    write_report(magma_results, wilcoxon_results, gene_results, OUTPUT_DIR)

    # Print summary
    logger.info("=== RESULTS SUMMARY ===")
    for method_name, results in [("MAGMA", magma_results), ("Wilcoxon", wilcoxon_results)]:
        logger.info("--- %s ---", method_name)
        for r in results:
            sig = "*" if r.p_value < 0.05 else ""
            logger.info("  %-40s  n=%4d  z=%6.2f  p=%.3e  fdr=%.3f %s",
                        r.gene_set, r.n_genes_in_set, r.z_score, r.p_value, r.fdr_q, sig)

    logger.info("=== Phase 1b complete ===")


if __name__ == "__main__":
    main()
