"""Compute transcriptomic features from GSE93624 expression data.

Adapts the feature engineering from 02_transcriptomic_features.py for the
GSE93624 dataset. Expression values are already TMM-normalized log-expression,
so no additional RPKM->log2 transform is needed.

Computes the same feature set:
  - 7 pathway scores (rank-based z-score method)
  - Individual candidate gene expression values
  - Filters to CD patients only

Usage:
    uv run python -m crohns.cd_stricture_risk_prediction.05_gse93624_transcriptomic_features
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "crohns" / "cd-stricture-risk-prediction"
OUTPUT_DIR = DATA_DIR / "processed"

# ── Gene sets (same as 02_transcriptomic_features.py) ──

STRICTURE_PANEL_8 = ["LY96", "AKAP11", "SRM", "GREM1", "EHD2", "SERPINE1", "HDAC1", "FGF2"]
TL1A_PATHWAY = ["TNFSF15", "TNFRSF25"]
CREEPING_FAT = ["CTHRC1", "POSTN", "CYR61"]
FIBROSIS_MARKERS = ["CD38", "TNC", "CPA3"]
ALL_CANDIDATES = STRICTURE_PANEL_8 + TL1A_PATHWAY + CREEPING_FAT + FIBROSIS_MARKERS

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
    "inflammatory_fibroblast_module": [
        "CXCL12", "CXCL14", "IL6", "IL11", "CCL2",
        "POSTN", "CTHRC1", "COL1A1", "COL3A1", "FN1",
        "FAP", "PDPN", "THY1", "ACTA2", "SERPINE1",
        "MMP2", "MMP14", "TIMP1", "GREM1",
    ],
}


def gene_set_score(expr: pd.DataFrame, gene_set: list[str], set_name: str = "") -> pd.Series:
    """Compute gene set score using rank-based z-score method."""
    present = [g for g in gene_set if g in expr.index]
    if not present:
        if set_name:
            print(f"  WARNING: no genes from {set_name} found")
        return pd.Series(np.nan, index=expr.columns)

    coverage = len(present) / len(gene_set)
    if set_name:
        print(f"  {set_name}: {len(present)}/{len(gene_set)} genes ({coverage:.0%})")

    ranked = expr.rank(axis=0)
    z_ranked = ranked.apply(stats.zscore, axis=0)
    return z_ranked.loc[present].mean(axis=0)


def extract_candidate_genes(expr: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """Extract expression for individual candidate genes."""
    present = [g for g in genes if g in expr.index]
    missing = [g for g in genes if g not in expr.index]
    if missing:
        print(f"  Missing candidate genes: {missing}")

    result = expr.loc[present].T.copy()
    for g in missing:
        result[g] = np.nan
    return result[genes]


def main() -> None:
    expr_path = OUTPUT_DIR / "gse93624_expression.tsv.gz"
    pheno_path = OUTPUT_DIR / "gse93624_phenotype.tsv"

    if not expr_path.exists():
        print(f"ERROR: Expression file not found: {expr_path}", file=sys.stderr)
        print("Run 04_gse93624_phenotype.py first.", file=sys.stderr)
        sys.exit(1)
    if not pheno_path.exists():
        print(f"ERROR: Phenotype file not found: {pheno_path}", file=sys.stderr)
        sys.exit(1)

    # Load phenotype to filter CD patients
    pheno = pd.read_csv(pheno_path, sep="\t", index_col=0)
    cd_gsm = pheno[pheno["is_cd"]].index.tolist()
    print(f"CD patients: {len(cd_gsm)}")

    # Load expression (genes x samples) — chunked for memory safety
    print("Loading expression data...")
    chunks = []
    for chunk in pd.read_csv(expr_path, sep="\t", index_col=0, chunksize=3000):
        chunks.append(chunk[cd_gsm])
    expr = pd.concat(chunks)
    print(f"Expression matrix (CD only): {expr.shape[0]} genes x {expr.shape[1]} samples")

    # Pathway scores
    print("\nComputing pathway scores...")
    pathway_scores = pd.DataFrame(index=expr.columns)
    for name, gene_set in PATHWAY_GENE_SETS.items():
        pathway_scores[f"pathway_{name}"] = gene_set_score(expr, gene_set, name)

    # Individual candidate genes
    print("\nExtracting candidate gene expression...")
    candidates = extract_candidate_genes(expr, ALL_CANDIDATES)
    candidates.columns = [f"gene_{g}" for g in candidates.columns]

    # Combine
    features = pd.concat([pathway_scores, candidates], axis=1)
    features.index.name = "gsm_id"

    print(f"\nTotal transcriptomic features: {features.shape[1]}")
    print(f"Samples: {features.shape[0]}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "gse93624_transcriptomic_features.tsv"
    features.to_csv(out_path, sep="\t")
    print(f"\nSaved: {out_path}")

    # Report completeness
    null_pct = features.isnull().mean() * 100
    non_null = null_pct[null_pct > 0]
    if len(non_null) > 0:
        print(f"\nFeatures with missing values:")
        for col in non_null.index:
            print(f"  {col}: {null_pct[col]:.0f}% missing")
    else:
        print("\nAll features complete (no missing values).")


if __name__ == "__main__":
    main()
