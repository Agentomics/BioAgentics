"""Compute transcriptomic features for stricture risk prediction.

Loads RPKM expression data from the RISK cohort (GSE57945) and computes:
1. Individual candidate gene expression (8-gene panel, TL1A, creeping fat, fibrosis markers)
2. Fibrosis pathway gene set scores (TGFb, Wnt, collagen, MMP, EMT, YAP/TAZ)
3. Inflammatory fibroblast spatial module score (Kong et al. Nat Genet 2025)

Gene set scoring uses a rank-based z-score method (similar to ssGSEA):
  score = mean(z-scored ranks of genes in set)

NOTE: MSigDB gene sets (task #506) not yet downloaded. Using curated gene sets
from the literature until MSigDB data is available.

Usage:
    uv run python -m crohns.cd_stricture_risk_prediction.02_transcriptomic_features
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "crohns" / "cd-stricture-risk-prediction"
GSE57945_DIR = DATA_DIR / "risk-cohort" / "GSE57945"
OUTPUT_DIR = DATA_DIR / "processed"

# ── Priority candidate genes ──

# 8-gene stricture panel (IBD 2025, doi:10.1093/ibd/izaf026)
STRICTURE_PANEL_8 = ["LY96", "AKAP11", "SRM", "GREM1", "EHD2", "SERPINE1", "HDAC1", "FGF2"]

# TL1A pathway — therapeutically actionable (duvakitug, tulisokibart)
TL1A_PATHWAY = ["TNFSF15", "TNFRSF25"]

# Creeping fat fibroblast markers (CTHRC1/YAP-TAZ axis)
CREEPING_FAT = ["CTHRC1", "POSTN", "CYR61"]  # CYR61 = CCN1

# Additional fibrosis markers
FIBROSIS_MARKERS = ["CD38", "TNC", "CPA3"]

# All individual candidate genes
ALL_CANDIDATES = STRICTURE_PANEL_8 + TL1A_PATHWAY + CREEPING_FAT + FIBROSIS_MARKERS

# ── Curated fibrosis pathway gene sets ──
# Core members from literature — will be expanded with MSigDB when available

PATHWAY_GENE_SETS: dict[str, list[str]] = {
    "tgfb_signaling": [
        "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2",
        "SMAD2", "SMAD3", "SMAD4", "SMAD7",
        "ACVRL1", "INHBA", "BMP2", "BMP4", "GREM1",
    ],
    "wnt_signaling": [
        "WNT2", "WNT3A", "WNT5A", "WNT7B", "WNT11",
        "CTNNB1", "APC", "AXIN2", "LEF1", "TCF7L2",
        "FZD1", "FZD2", "LRP5", "LRP6", "DKK1",
    ],
    "collagen_ecm": [
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1",
        "COL6A1", "COL6A3", "FN1", "ELN", "LAMA1",
        "SPARC", "THBS1", "THBS2", "VCAN", "DCN",
    ],
    "mmp_activity": [
        "MMP1", "MMP2", "MMP3", "MMP7", "MMP9",
        "MMP12", "MMP13", "MMP14", "TIMP1", "TIMP2",
        "TIMP3", "ADAM17", "ADAMTS1",
    ],
    "emt": [
        "VIM", "CDH2", "SNAI1", "SNAI2", "TWIST1",
        "TWIST2", "ZEB1", "ZEB2", "FN1", "CDH1",
        "CLDN1", "TJP1", "ACTA2", "S100A4",
    ],
    "yap_taz_targets": [
        "CYR61", "CTGF", "ANKRD1", "AMOTL2", "AXL",
        "BIRC5", "CCND1", "EDN1", "LATS2", "SERPINE1",
        "TEAD1", "TEAD4", "WWTR1", "YAP1",
    ],
    # Kong et al. inflammatory fibroblast spatial module (Nat Genet 2025)
    "inflammatory_fibroblast_module": [
        "CXCL12", "CXCL14", "IL6", "IL11", "CCL2",
        "POSTN", "CTHRC1", "COL1A1", "COL3A1", "FN1",
        "FAP", "PDPN", "THY1", "ACTA2", "SERPINE1",
        "MMP2", "MMP14", "TIMP1", "GREM1",
    ],
}


def load_rpkm_chunked(rpkm_path: Path, sample_ids: list[str] | None = None) -> pd.DataFrame:
    """Load RPKM expression matrix, optionally subsetting to specific samples.

    Reads in chunks to stay within memory constraints (8GB machine).
    Returns genes x samples DataFrame with Gene Symbol as index.
    """
    # Read header to get column names
    with gzip.open(rpkm_path, "rt") as f:
        header = f.readline().rstrip().split("\t")

    # Determine which columns to read
    usecols = ["Gene Symbol"]
    if sample_ids:
        usecols += [s for s in sample_ids if s in header]
    else:
        usecols += header[2:]  # All sample columns

    chunks = []
    for chunk in pd.read_csv(
        rpkm_path,
        sep="\t",
        usecols=usecols,
        chunksize=5000,
    ):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Some genes appear multiple times (e.g. DDX11L1 on different chromosomes)
    # Aggregate by taking the max RPKM across duplicates
    df = df.groupby("Gene Symbol").max()

    print(f"Expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")
    return df


def gene_set_score(expr: pd.DataFrame, gene_set: list[str], set_name: str = "") -> pd.Series:
    """Compute gene set score using rank-based z-score method.

    For each sample:
    1. Rank all genes by expression
    2. Z-score the ranks
    3. Mean z-scored rank of genes in the set = score

    This is robust to outliers and doesn't require normalization assumptions.
    """
    # Find which genes from the set are present
    present = [g for g in gene_set if g in expr.index]
    if not present:
        if set_name:
            print(f"  WARNING: no genes from {set_name} found in expression data")
        return pd.Series(np.nan, index=expr.columns)

    coverage = len(present) / len(gene_set)
    if set_name:
        print(f"  {set_name}: {len(present)}/{len(gene_set)} genes found ({coverage:.0%})")

    # Rank genes per sample, then z-score the ranks
    ranked = expr.rank(axis=0)
    z_ranked = ranked.apply(stats.zscore, axis=0)

    # Score = mean z-scored rank of genes in set
    scores = z_ranked.loc[present].mean(axis=0)
    return scores


def extract_candidate_genes(expr: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """Extract expression values for individual candidate genes.

    Returns samples x genes DataFrame. Missing genes are filled with NaN.
    """
    present = [g for g in genes if g in expr.index]
    missing = [g for g in genes if g not in expr.index]
    if missing:
        print(f"  Missing candidate genes: {missing}")

    result = expr.loc[present].T.copy()
    for g in missing:
        result[g] = np.nan
    return result[genes]  # Ensure consistent column order


def build_transcriptomic_features(
    rpkm_path: Path,
    phenotype_path: Path,
) -> pd.DataFrame:
    """Build the full transcriptomic feature matrix.

    Loads expression for CD patients only, computes pathway scores and
    individual gene features.
    """
    # Load phenotype table to get CD patient risk IDs
    pheno = pd.read_csv(phenotype_path, sep="\t", index_col=0)
    cd_risk_ids = pheno["risk_id"].dropna().tolist()
    print(f"CD patients with risk IDs: {len(cd_risk_ids)}")

    # Load expression for CD patients
    expr = load_rpkm_chunked(rpkm_path, sample_ids=cd_risk_ids)

    # Log-transform RPKM: log2(RPKM + 1)
    expr_log = np.log2(expr + 1)
    print(f"Log2(RPKM+1) transformed")

    # ── Pathway scores ──
    print("\nComputing pathway scores...")
    pathway_scores = pd.DataFrame(index=expr.columns)
    for name, gene_set in PATHWAY_GENE_SETS.items():
        pathway_scores[f"pathway_{name}"] = gene_set_score(expr_log, gene_set, name)

    # ── Individual candidate genes ──
    print("\nExtracting candidate gene expression...")
    candidates = extract_candidate_genes(expr_log, ALL_CANDIDATES)
    candidates.columns = [f"gene_{g}" for g in candidates.columns]

    # ── Combine ──
    features = pd.concat([pathway_scores, candidates], axis=1)
    features.index.name = "risk_id"

    print(f"\nTotal transcriptomic features: {features.shape[1]}")
    return features


def main() -> None:
    rpkm_path = GSE57945_DIR / "supplementary" / "GSE57945_all_samples_RPKM.txt.gz"
    phenotype_path = OUTPUT_DIR / "phenotype_table.tsv"

    if not rpkm_path.exists():
        print(f"ERROR: RPKM file not found: {rpkm_path}", file=sys.stderr)
        sys.exit(1)
    if not phenotype_path.exists():
        print(f"ERROR: Phenotype table not found: {phenotype_path}", file=sys.stderr)
        print("Run 01_phenotype_extraction.py first.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = build_transcriptomic_features(rpkm_path, phenotype_path)

    out_path = OUTPUT_DIR / "transcriptomic_features.tsv"
    features.to_csv(out_path, sep="\t")
    print(f"\nTranscriptomic features saved: {out_path}")
    print(f"Shape: {features.shape}")

    # Report feature completeness
    null_pct = features.isnull().mean() * 100
    if null_pct.any():
        print(f"\nFeatures with missing values:")
        for col in null_pct[null_pct > 0].index:
            print(f"  {col}: {null_pct[col]:.0f}% missing")


if __name__ == "__main__":
    main()
