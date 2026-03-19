"""FP profiling module for feature distributions and demographics.

Compares false positive vs true negative feature distributions, confidence
scores, and demographic characteristics. Outputs summary statistics and
statistical tests to identify systematic differences.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.diagnostics.fp_mining.extract import ExtractionResult, get_feature_columns

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/diagnostics/false-positive-biomarker-mining/profiles")


def compute_distribution_stats(series: pd.Series) -> dict:
    """Compute summary statistics for a numeric series."""
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "median": float(series.median()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
        "n": int(len(series)),
    }


def compare_fp_vs_tn(
    result: ExtractionResult,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compare feature distributions between FP and TN groups.

    Runs t-tests and KS-tests for each feature to identify systematic
    differences between false positives and true negatives.

    Args:
        result: ExtractionResult from extract module.
        feature_cols: Features to compare. If None, auto-detected.

    Returns:
        DataFrame with per-feature comparison statistics.
    """
    fp = result.false_positives
    tn = result.true_negatives

    if len(fp) == 0 or len(tn) == 0:
        logger.warning("Cannot compare FP vs TN: one group is empty")
        return pd.DataFrame()

    if feature_cols is None:
        feature_cols = get_feature_columns(fp)

    rows = []
    for col in feature_cols:
        if col not in fp.columns or col not in tn.columns:
            continue

        fp_vals = fp[col].dropna()
        tn_vals = tn[col].dropna()

        if len(fp_vals) < 2 or len(tn_vals) < 2:
            continue

        fp_stats = compute_distribution_stats(fp_vals)
        tn_stats = compute_distribution_stats(tn_vals)

        # Welch's t-test (unequal variances)
        t_stat, t_pval = stats.ttest_ind(fp_vals, tn_vals, equal_var=False)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(fp_vals, tn_vals)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (fp_vals.std() ** 2 + tn_vals.std() ** 2) / 2
        )
        cohens_d = (fp_stats["mean"] - tn_stats["mean"]) / max(pooled_std, 1e-10)

        rows.append({
            "feature": col,
            "fp_mean": fp_stats["mean"],
            "fp_std": fp_stats["std"],
            "fp_median": fp_stats["median"],
            "fp_n": fp_stats["n"],
            "tn_mean": tn_stats["mean"],
            "tn_std": tn_stats["std"],
            "tn_median": tn_stats["median"],
            "tn_n": tn_stats["n"],
            "mean_diff": fp_stats["mean"] - tn_stats["mean"],
            "cohens_d": cohens_d,
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pval),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
        })

    comparison = pd.DataFrame(rows)
    if len(comparison) > 0:
        comparison = comparison.sort_values("t_pvalue").reset_index(drop=True)

    return comparison


def profile_confidence_scores(result: ExtractionResult) -> dict:
    """Profile prediction confidence score distributions across groups.

    Returns:
        Dict with distribution stats for FP, TN, TP, FN groups.
    """
    groups = {
        "false_positives": result.false_positives,
        "true_negatives": result.true_negatives,
        "true_positives": result.true_positives,
        "false_negatives": result.false_negatives,
    }

    profiles = {}
    for name, df in groups.items():
        if len(df) > 0 and "y_score" in df.columns:
            profiles[name] = compute_distribution_stats(df["y_score"])
        else:
            profiles[name] = {"n": 0}

    return profiles


def profile_demographics(
    result: ExtractionResult,
    demographic_cols: list[str] | None = None,
) -> dict:
    """Profile demographic characteristics of FP vs TN groups.

    Args:
        result: ExtractionResult.
        demographic_cols: Columns to profile (e.g., ['age', 'sex']).
            If None, auto-detects common demographic column names.

    Returns:
        Dict with demographic breakdowns for FP and TN groups.
    """
    if demographic_cols is None:
        demographic_cols = []
        for col in ["age", "sex", "gender", "ethnicity", "race", "stage", "stage_numeric"]:
            if col in result.false_positives.columns:
                demographic_cols.append(col)

    if not demographic_cols:
        return {}

    profiles = {}
    for col in demographic_cols:
        fp_vals = result.false_positives[col].dropna() if col in result.false_positives.columns else pd.Series(dtype=object)
        tn_vals = result.true_negatives[col].dropna() if col in result.true_negatives.columns else pd.Series(dtype=object)

        if pd.api.types.is_numeric_dtype(fp_vals) and len(fp_vals) > 0 and len(tn_vals) > 0:
            profiles[col] = {
                "fp": compute_distribution_stats(fp_vals),
                "tn": compute_distribution_stats(tn_vals),
            }
        elif len(fp_vals) > 0:
            profiles[col] = {
                "fp": fp_vals.value_counts(normalize=True).to_dict(),
                "tn": tn_vals.value_counts(normalize=True).to_dict() if len(tn_vals) > 0 else {},
            }

    return profiles


def run_profiling(
    result: ExtractionResult,
    output_dir: Path | None = None,
) -> dict:
    """Run full profiling pipeline on an extraction result.

    Args:
        result: ExtractionResult to profile.
        output_dir: Directory to save outputs. Defaults to OUTPUT_DIR.

    Returns:
        Dict with keys: comparison, confidence, demographics, summary.
    """
    save_dir = output_dir or OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    domain = result.domain
    op_name = result.operating_point.name

    # Feature comparison
    comparison = compare_fp_vs_tn(result)
    if len(comparison) > 0:
        comparison.to_csv(
            save_dir / f"{domain}_{op_name}_fp_vs_tn.csv",
            index=False,
        )

    # Confidence score profiles
    confidence = profile_confidence_scores(result)

    # Demographics
    demographics = profile_demographics(result)

    # Count significant features
    n_significant = 0
    if len(comparison) > 0:
        n_significant = int((comparison["t_pvalue"] < 0.05).sum())

    summary = {
        "domain": domain,
        "operating_point": op_name,
        "n_features_compared": len(comparison),
        "n_significant_features": n_significant,
        "n_fp": len(result.false_positives),
        "n_tn": len(result.true_negatives),
    }

    logger.info(
        "%s @ %s: %d/%d features significantly different (p<0.05)",
        domain,
        op_name,
        n_significant,
        len(comparison),
    )

    return {
        "comparison": comparison,
        "confidence": confidence,
        "demographics": demographics,
        "summary": summary,
    }
