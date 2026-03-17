"""LDSC genetic correlation pipeline.

Computes pairwise genetic correlations between Tourette syndrome and
psychiatric comorbidities using LD score regression (Bulik-Sullivan et al. 2015).

Implements the core bivariate LDSC regression from first principles so the
pipeline is self-contained, testable with synthetic data, and does not depend
on the legacy Python-2 ldsc package.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

# Disorders from the Nature 2025 cross-disorder study to correlate with TS
PSYCHIATRIC_DISORDERS = [
    "TS",
    "OCD",
    "ADHD",
    "ASD",
    "MDD",
    "anxiety",
    "schizophrenia",
    "bipolar",
    "anorexia",
    "PTSD",
    "alcohol_use_disorder",
    "opioid_use_disorder",
    "cannabis_use_disorder",
    "problematic_internet_use",
]

# Required columns in munged summary statistics
REQUIRED_COLS = {"SNP", "A1", "A2", "Z", "N"}


@dataclass
class LDSCResult:
    """Result of a single bivariate LDSC regression."""

    trait1: str
    trait2: str
    rg: float
    rg_se: float
    p_value: float
    h2_trait1: float
    h2_trait1_se: float
    h2_trait2: float
    h2_trait2_se: float
    gcov_int: float
    n_snps: int


def load_sumstats(path: Path) -> pd.DataFrame:
    """Load GWAS summary statistics from a tab/space-delimited file.

    Accepts standard formats with columns: SNP, A1, A2, and one of
    Z, BETA+SE, or OR+SE for effect sizes, plus P and N.
    """
    name = path.name.lower()
    if name.endswith((".tsv", ".gz", ".sumstats")):
        sep = "\t"
    else:
        sep = r"\s+"
    df = pd.read_csv(path, sep=sep, comment="#", na_values=[".", "NA", ""], engine="python" if sep == r"\s+" else "c")
    df.columns = df.columns.str.strip().str.upper()

    # Rename common variants
    col_map = {
        "RSID": "SNP",
        "MARKERNAME": "SNP",
        "EFFECT_ALLELE": "A1",
        "OTHER_ALLELE": "A2",
        "ALLELE1": "A1",
        "ALLELE2": "A2",
        "ZSCORE": "Z",
        "BETA": "BETA",
        "STDERR": "SE",
        "PVALUE": "P",
        "P_VALUE": "P",
        "PVAL": "P",
        "SAMPLESIZE": "N",
        "NEFF": "N",
        "N_EFF": "N",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    return df


def munge_sumstats(df: pd.DataFrame, trait_name: str | None = None) -> pd.DataFrame:
    """Munge summary statistics into a standardized format for LDSC.

    Converts BETA/SE to Z-scores if Z is not present, applies QC filters,
    and retains only required columns.
    """
    df = df.copy()

    # Derive Z from BETA/SE if needed
    if "Z" not in df.columns:
        if "BETA" in df.columns and "SE" in df.columns:
            df["Z"] = df["BETA"] / df["SE"]
        elif "OR" in df.columns and "SE" in df.columns:
            df["Z"] = np.log(df["OR"]) / df["SE"]
        elif "P" in df.columns and "BETA" in df.columns:
            df["Z"] = np.sign(df["BETA"]) * np.abs(stats.norm.ppf(df["P"] / 2))
        else:
            raise ValueError(
                f"Cannot derive Z-scores for {trait_name}: need Z, BETA+SE, OR+SE, or BETA+P"
            )

    # Ensure N column
    if "N" not in df.columns:
        raise ValueError(f"Missing sample size (N) column for {trait_name}")

    # QC filters
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {trait_name}: {missing}")

    df = df[list(REQUIRED_COLS)].copy()
    df["A1"] = df["A1"].str.upper()
    df["A2"] = df["A2"].str.upper()

    n_before = len(df)
    df.dropna(subset=["SNP", "Z", "N"], inplace=True)

    # Remove extreme Z-scores (|Z| > 37 ~ p < 1e-300, likely errors)
    df = df[df["Z"].abs() <= 37]

    # Remove duplicates
    df.drop_duplicates(subset="SNP", keep="first", inplace=True)

    n_after = len(df)
    if trait_name:
        logger.info(
            "%s: %d -> %d SNPs after munging (removed %d)",
            trait_name,
            n_before,
            n_after,
            n_before - n_after,
        )

    return df.reset_index(drop=True)


def load_ld_scores(path: Path) -> pd.DataFrame:
    """Load LD scores from a file or directory of per-chromosome files.

    Expects columns: SNP, L2 (the LD score).
    If path is a directory, reads all .l2.ldscore[.gz] files and concatenates.
    """
    if path.is_dir():
        frames = []
        for f in sorted(path.glob("*.l2.ldscore*")):
            frames.append(pd.read_csv(f, sep="\t"))
        if not frames:
            raise FileNotFoundError(f"No .l2.ldscore files found in {path}")
        ldscore = pd.concat(frames, ignore_index=True)
    else:
        ldscore = pd.read_csv(path, sep="\t")

    ldscore.columns = ldscore.columns.str.strip().str.upper()

    # Standardize column names
    if "LDSCORE" in ldscore.columns and "L2" not in ldscore.columns:
        ldscore.rename(columns={"LDSCORE": "L2"}, inplace=True)

    if "L2" not in ldscore.columns:
        raise ValueError(f"LD score file must contain L2 column, got: {list(ldscore.columns)}")

    ldscore = ldscore[["SNP", "L2"]].drop_duplicates(subset=["SNP"])
    logger.info("Loaded %d LD scores from %s", len(ldscore), path)
    return ldscore


def _univariate_ldsc(z_sq: np.ndarray, ld: np.ndarray, n: np.ndarray, m: int) -> tuple[float, float, float, float]:
    """Univariate LDSC regression: E[chi2] = N * h2 * l_j / M + 1 + Na.

    Uses weighted least squares with LD-score-based weights (1/l_j^2).

    Returns: (h2, h2_se, intercept, intercept_se)
    """
    x = (n * ld / m).reshape(-1, 1)
    y = z_sq
    ones = np.ones((len(y), 1))
    X = np.hstack([x, ones])

    # Weights: inverse of variance of chi2, approximated as 1/l_j^2
    w = 1.0 / np.maximum(ld**2, 1.0)
    W = np.diag(w)

    # WLS: beta = (X'WX)^{-1} X'Wy
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan

    h2 = float(beta[0])
    intercept = float(beta[1])

    # Standard errors via sandwich estimator
    residuals = y - X @ beta
    meat = X.T @ W @ np.diag(residuals**2) @ W @ X
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return h2, np.nan, intercept, np.nan

    cov = bread @ meat @ bread
    h2_se = float(np.sqrt(max(cov[0, 0], 0)))
    intercept_se = float(np.sqrt(max(cov[1, 1], 0)))

    return h2, h2_se, intercept, intercept_se


def _bivariate_ldsc(
    z1: np.ndarray,
    z2: np.ndarray,
    ld: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    m: int,
) -> tuple[float, float, float]:
    """Bivariate LDSC: E[z1*z2] = sqrt(N1*N2) * rg_cov * l_j / M + intercept.

    Returns: (genetic_covariance, gcov_se, intercept)
    """
    z_product = z1 * z2
    sqrt_n = np.sqrt(n1 * n2)
    x = (sqrt_n * ld / m).reshape(-1, 1)
    y = z_product
    ones = np.ones((len(y), 1))
    X = np.hstack([x, ones])

    w = 1.0 / np.maximum(ld**2, 1.0)
    W = np.diag(w)

    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    gcov = float(beta[0])
    intercept = float(beta[1])

    residuals = y - X @ beta
    meat = X.T @ W @ np.diag(residuals**2) @ W @ X
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return gcov, np.nan, intercept

    cov = bread @ meat @ bread
    gcov_se = float(np.sqrt(max(cov[0, 0], 0)))

    return gcov, gcov_se, intercept


def ldsc_regression(
    sumstats1: pd.DataFrame,
    sumstats2: pd.DataFrame,
    ld_scores: pd.DataFrame,
    trait1_name: str = "trait1",
    trait2_name: str = "trait2",
) -> LDSCResult:
    """Run bivariate LDSC regression between two traits.

    Parameters
    ----------
    sumstats1, sumstats2 : pd.DataFrame
        Munged summary statistics with columns SNP, Z, N.
    ld_scores : pd.DataFrame
        LD scores with columns SNP, L2.
    trait1_name, trait2_name : str
        Labels for the two traits.

    Returns
    -------
    LDSCResult with genetic correlation, standard errors, and heritabilities.
    """
    # Merge on shared SNPs
    merged = (
        sumstats1[["SNP", "Z", "N"]]
        .merge(sumstats2[["SNP", "Z", "N"]], on="SNP", suffixes=("_1", "_2"))
        .merge(ld_scores[["SNP", "L2"]], on="SNP")
    )

    n_snps = len(merged)
    if n_snps < 200:
        logger.warning(
            "Only %d shared SNPs between %s and %s — results unreliable",
            n_snps,
            trait1_name,
            trait2_name,
        )

    z1 = merged["Z_1"].values
    z2 = merged["Z_2"].values
    ld = merged["L2"].values
    n1 = merged["N_1"].values
    n2 = merged["N_2"].values
    m = n_snps  # number of SNPs in regression

    # Univariate h2 for each trait
    h2_1, h2_1_se, _, _ = _univariate_ldsc(z1**2, ld, n1, m)
    h2_2, h2_2_se, _, _ = _univariate_ldsc(z2**2, ld, n2, m)

    # Bivariate genetic covariance
    gcov, gcov_se, gcov_int = _bivariate_ldsc(z1, z2, ld, n1, n2, m)

    # Genetic correlation: rg = gcov / sqrt(h2_1 * h2_2)
    if h2_1 > 0 and h2_2 > 0:
        rg = gcov / np.sqrt(h2_1 * h2_2)
        # Delta method SE for rg
        rg_se = abs(rg) * np.sqrt(
            (gcov_se / gcov) ** 2 + 0.25 * ((h2_1_se / h2_1) ** 2 + (h2_2_se / h2_2) ** 2)
            if gcov != 0
            else float("inf")
        )
    else:
        rg = np.nan
        rg_se = np.nan

    # P-value for rg (Wald test)
    if np.isfinite(rg) and np.isfinite(rg_se) and rg_se > 0:
        p_value = float(2 * stats.norm.sf(abs(rg / rg_se)))
    else:
        p_value = np.nan

    return LDSCResult(
        trait1=trait1_name,
        trait2=trait2_name,
        rg=rg,
        rg_se=rg_se,
        p_value=p_value,
        h2_trait1=h2_1,
        h2_trait1_se=h2_1_se,
        h2_trait2=h2_2,
        h2_trait2_se=h2_2_se,
        gcov_int=gcov_int,
        n_snps=n_snps,
    )


def compute_genetic_correlation_matrix(
    sumstats_dir: Path,
    ld_scores: pd.DataFrame,
    reference_trait: str = "TS",
) -> list[LDSCResult]:
    """Compute genetic correlations between a reference trait and all others.

    Loads all .sumstats.gz / .sumstats / .tsv / .txt files from sumstats_dir,
    munges them, and runs pairwise LDSC against the reference trait.

    Parameters
    ----------
    sumstats_dir : Path
        Directory containing summary statistics files named <TRAIT>.sumstats[.gz].
    ld_scores : pd.DataFrame
        LD scores.
    reference_trait : str
        Name of the reference trait (default: TS for Tourette syndrome).

    Returns
    -------
    List of LDSCResult, one per pairwise comparison.
    """
    # Discover summary stats files
    suffixes = (".sumstats.gz", ".sumstats", ".tsv", ".txt", ".csv")
    files: dict[str, Path] = {}
    for f in sumstats_dir.iterdir():
        if f.is_file():
            name = f.name
            for suf in suffixes:
                if name.endswith(suf):
                    trait = name[: -len(suf)]
                    files[trait] = f
                    break

    if reference_trait not in files:
        raise FileNotFoundError(
            f"Reference trait '{reference_trait}' not found in {sumstats_dir}. "
            f"Available: {sorted(files.keys())}"
        )

    logger.info("Found %d traits: %s", len(files), sorted(files.keys()))

    # Load and munge reference trait
    ref_df = munge_sumstats(load_sumstats(files[reference_trait]), reference_trait)

    results = []
    for trait_name, path in sorted(files.items()):
        if trait_name == reference_trait:
            continue

        logger.info("Computing rg(%s, %s)...", reference_trait, trait_name)
        trait_df = munge_sumstats(load_sumstats(path), trait_name)
        result = ldsc_regression(ref_df, trait_df, ld_scores, reference_trait, trait_name)
        results.append(result)

        logger.info(
            "  rg = %.4f (SE = %.4f, p = %.2e, n_snps = %d)",
            result.rg,
            result.rg_se,
            result.p_value,
            result.n_snps,
        )

    return results


def results_to_dataframe(results: list[LDSCResult]) -> pd.DataFrame:
    """Convert a list of LDSCResult to a tidy DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "trait1": r.trait1,
                "trait2": r.trait2,
                "rg": r.rg,
                "rg_se": r.rg_se,
                "p_value": r.p_value,
                "h2_trait1": r.h2_trait1,
                "h2_trait1_se": r.h2_trait1_se,
                "h2_trait2": r.h2_trait2,
                "h2_trait2_se": r.h2_trait2_se,
                "gcov_intercept": r.gcov_int,
                "n_snps": r.n_snps,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the LDSC genetic correlation pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute pairwise LDSC genetic correlations between TS and psychiatric disorders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing GWAS summary statistics files.",
    )
    parser.add_argument(
        "--ld-scores",
        type=Path,
        default=None,
        help="Path to LD scores file or directory. Defaults to <input-dir>/ld_scores/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "tourettes" / "ts-comorbidity-genetic-architecture" / "phase1",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--reference-trait",
        type=str,
        default="TS",
        help="Reference trait name (default: TS).",
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

    input_dir = args.input_dir
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # Load LD scores
    ld_path = args.ld_scores or input_dir / "ld_scores"
    if not ld_path.exists():
        logger.error("LD scores not found at %s", ld_path)
        sys.exit(1)
    ld_scores = load_ld_scores(ld_path)

    # Run pipeline
    results = compute_genetic_correlation_matrix(input_dir, ld_scores, args.reference_trait)

    if not results:
        logger.warning("No pairwise comparisons computed.")
        sys.exit(0)

    # Write output
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)
    out_path = output_dir / "genetic_correlations.tsv"
    df.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    logger.info("Wrote %d correlations to %s", len(df), out_path)

    # Also write a symmetric matrix form for visualization
    traits = [results[0].trait1] + [r.trait2 for r in results]
    n = len(traits)
    rg_matrix = pd.DataFrame(np.eye(n), index=traits, columns=traits)
    for r in results:
        rg_matrix.loc[r.trait1, r.trait2] = r.rg
        rg_matrix.loc[r.trait2, r.trait1] = r.rg

    matrix_path = output_dir / "rg_matrix.tsv"
    rg_matrix.to_csv(matrix_path, sep="\t", float_format="%.6f")
    logger.info("Wrote correlation matrix to %s", matrix_path)


if __name__ == "__main__":
    main()
