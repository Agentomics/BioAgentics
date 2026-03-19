"""GWAS summary statistics preprocessing and QC.

Cleans, standardizes, and quality-filters TSAICG GWAS meta-analysis
summary statistics for downstream functional annotation.

Usage:
    python -m bioagentics.tourettes.ts_gwas_functional_annotation.preprocess_gwas \
        --input data/tourettes/ts-gwas-functional-annotation/raw/tsaicg_gwas.tsv \
        --output data/tourettes/ts-gwas-functional-annotation/gwas/tsaicg_cleaned.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.tourettes.ts_gwas_functional_annotation.config import (
    GWAS_DIR,
    GWAS_QC,
    RAW_DIR,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# Column name mappings for common GWAS formats (PGC, METAL, PLINK, etc.)
_COLUMN_MAP = {
    # SNP identifiers
    "RSID": "SNP", "rsid": "SNP", "MarkerName": "SNP", "ID": "SNP",
    "SNPID": "SNP", "variant_id": "SNP", "rs_id": "SNP",
    # Chromosome
    "CHROMOSOME": "CHR", "chr": "CHR", "chromosome": "CHR",
    "Chromosome": "CHR", "#CHROM": "CHR", "hm_chrom": "CHR",
    # Position
    "POSITION": "BP", "pos": "BP", "POS": "BP", "bp": "BP",
    "base_pair_location": "BP", "hm_pos": "BP", "position": "BP",
    # Effect allele
    "EFFECT_ALLELE": "A1", "ALT": "A1", "Allele1": "A1",
    "effect_allele": "A1", "hm_effect_allele": "A1", "A1FREQ": "FRQ",
    # Other allele
    "OTHER_ALLELE": "A2", "REF": "A2", "Allele2": "A2",
    "other_allele": "A2", "hm_other_allele": "A2",
    # Effect size
    "EFFECT": "BETA", "B": "BETA", "Effect": "BETA", "beta": "BETA",
    "hm_beta": "BETA", "ODDS_RATIO": "OR", "or": "OR",
    # Standard error
    "STDERR": "SE", "StdErr": "SE", "se": "SE",
    "standard_error": "SE", "hm_se": "SE",
    # P-value
    "PVAL": "P", "P_VALUE": "P", "PVALUE": "P", "p_value": "P",
    "Pvalue": "P", "p-value": "P", "hm_p": "P",
    # Z-score
    "ZSCORE": "Z", "z_score": "Z", "Zscore": "Z",
    # Sample size
    "NMISS": "N", "N_total": "N", "n": "N", "TotalN": "N",
    "Neff": "N",
    # Frequency
    "MAF": "FRQ", "Freq1": "FRQ", "freq": "FRQ",
    "effect_allele_frequency": "FRQ", "EAF": "FRQ",
    # Imputation quality
    "INFO": "INFO", "info": "INFO", "IMPINFO": "INFO",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical GWAS format."""
    renamed = df.rename(
        columns={k: v for k, v in _COLUMN_MAP.items() if k in df.columns},
    )
    return renamed


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Z-scores and P-values from available columns."""
    # Z from BETA/SE
    if "Z" not in df.columns and "BETA" in df.columns and "SE" in df.columns:
        mask = df["SE"] > 0
        df.loc[mask, "Z"] = df.loc[mask, "BETA"] / df.loc[mask, "SE"]

    # P from Z
    if "P" not in df.columns and "Z" in df.columns:
        df["P"] = 2 * stats.norm.sf(np.abs(df["Z"]))

    # BETA from OR
    if "BETA" not in df.columns and "OR" in df.columns:
        mask = df["OR"] > 0
        df.loc[mask, "BETA"] = np.log(df.loc[mask, "OR"])

    # Z from P and BETA sign (for studies reporting only P)
    if "Z" not in df.columns and "P" in df.columns:
        valid = (df["P"] > 0) & (df["P"] < 1)
        df.loc[valid, "Z"] = stats.norm.isf(df.loc[valid, "P"] / 2)
        if "BETA" in df.columns:
            neg = df["BETA"] < 0
            df.loc[valid & neg, "Z"] = -df.loc[valid & neg, "Z"]

    return df


def clean_chromosome(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize chromosome column to integer (1-22, X=23, Y=24)."""
    if "CHR" not in df.columns:
        return df

    df["CHR"] = df["CHR"].astype(str).str.replace("chr", "", case=False)

    chr_map = {"X": "23", "Y": "24", "MT": "25", "M": "25"}
    df["CHR"] = df["CHR"].replace(chr_map)

    df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce")
    df = df.dropna(subset=["CHR"])
    df["CHR"] = df["CHR"].astype(int)

    # Keep autosomes only for standard GWAS analysis
    df = df[df["CHR"].between(1, 22)]

    return df


def apply_qc_filters(
    df: pd.DataFrame,
    min_maf: float | None = None,
    min_info: float | None = None,
    max_abs_beta: float | None = None,
) -> pd.DataFrame:
    """Apply quality control filters to GWAS summary statistics."""
    n_start = len(df)
    qc = GWAS_QC

    if min_maf is None:
        min_maf = qc["min_maf"]
    if min_info is None:
        min_info = qc["min_info"]
    if max_abs_beta is None:
        max_abs_beta = qc["max_abs_beta"]

    # Remove missing P-values
    df = df.dropna(subset=["P"])
    df = df[(df["P"] > 0) & (df["P"] <= 1.0)]

    # MAF filter
    if "FRQ" in df.columns:
        maf = df["FRQ"].copy()
        maf = np.where(maf > 0.5, 1 - maf, maf)
        df = df[maf >= min_maf]

    # Imputation info filter
    if "INFO" in df.columns:
        df = df[df["INFO"] >= min_info]

    # Extreme effect size filter
    if "BETA" in df.columns:
        df = df[df["BETA"].abs() <= max_abs_beta]

    # Remove duplicate SNPs (keep lowest P-value)
    if "SNP" in df.columns:
        df = df.sort_values("P").drop_duplicates(subset=["SNP"], keep="first")

    n_end = len(df)
    logger.info("QC: %d -> %d SNPs (removed %d)", n_start, n_end, n_start - n_end)

    return df.reset_index(drop=True)


def compute_qc_summary(df: pd.DataFrame) -> dict:
    """Compute QC summary statistics for a cleaned GWAS dataset."""
    summary = {
        "n_snps": len(df),
        "n_chromosomes": int(df["CHR"].nunique()) if "CHR" in df.columns else 0,
    }

    if "P" in df.columns:
        summary["n_gw_sig"] = int((df["P"] < 5e-8).sum())
        summary["n_suggestive"] = int((df["P"] < 1e-5).sum())
        summary["lambda_gc"] = float(_genomic_inflation(df["P"]))

    if "FRQ" in df.columns:
        maf = df["FRQ"].copy()
        maf = np.where(maf > 0.5, 1 - maf, maf)
        summary["median_maf"] = float(np.median(maf))

    if "INFO" in df.columns:
        summary["median_info"] = float(df["INFO"].median())

    if "N" in df.columns:
        summary["median_n"] = float(df["N"].median())

    return summary


def _genomic_inflation(p_values: pd.Series) -> float:
    """Compute genomic inflation factor (lambda_GC)."""
    p = p_values.dropna()
    p = p[(p > 0) & (p < 1)]
    if len(p) == 0:
        return 1.0

    chi2_obs = stats.chi2.isf(p, df=1)
    lambda_gc = float(np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1))
    return lambda_gc


def preprocess_gwas(
    input_path: Path,
    output_path: Path | None = None,
    min_maf: float | None = None,
    min_info: float | None = None,
) -> pd.DataFrame:
    """Full preprocessing pipeline for GWAS summary statistics.

    Parameters
    ----------
    input_path : Path
        Raw GWAS summary statistics file (TSV or CSV).
    output_path : Path | None
        Output path for cleaned data. If None, writes to GWAS_DIR.
    min_maf : float | None
        Minimum minor allele frequency filter.
    min_info : float | None
        Minimum imputation info score filter.

    Returns
    -------
    Cleaned and QC'd GWAS DataFrame.
    """
    logger.info("Loading GWAS: %s", input_path)

    sep = "\t" if input_path.suffix in (".tsv", ".tab") else None
    df = pd.read_csv(input_path, sep=sep, engine="python")
    logger.info("Raw: %d SNPs, %d columns", len(df), len(df.columns))

    # Standardize
    df = standardize_columns(df)
    df = clean_chromosome(df)
    df = compute_derived_fields(df)

    # QC
    df = apply_qc_filters(df, min_maf=min_maf, min_info=min_info)

    # Summary
    qc_summary = compute_qc_summary(df)
    logger.info("QC summary: %s", qc_summary)

    # Write output
    if output_path is None:
        ensure_dirs()
        output_path = GWAS_DIR / "tsaicg_cleaned.tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote cleaned GWAS to %s", output_path)

    return df


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess TSAICG GWAS summary statistics."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Raw GWAS summary statistics file.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for cleaned data.",
    )
    parser.add_argument(
        "--min-maf", type=float, default=None,
        help="Minimum MAF filter.",
    )
    parser.add_argument(
        "--min-info", type=float, default=None,
        help="Minimum imputation INFO filter.",
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

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    preprocess_gwas(args.input, args.output, args.min_maf, args.min_info)


if __name__ == "__main__":
    main()
