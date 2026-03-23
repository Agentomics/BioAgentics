"""Build composite innate deficiency score per patient.

Task 106: Combine lectin complement expression, cGAS-STING pathway expression,
innate/adaptive ratio, and variant burden information into a single per-patient
deficiency score.

Components (z-scored, then weighted):
  1. Innate/adaptive ratio (lower = more deficient, inverted)
  2. Lectin complement module expression (lower = more deficient, inverted)
  3. cGAS-STING pathway expression (higher = more activated from variants)
  4. ISG expression from bulk (higher = type I IFN signature from cGAS-STING)

The score represents how "innate-deficient" a patient appears: higher score =
more evidence of innate immune deficiency or dysregulation.

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.composite_deficiency_score
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from bioagentics.config import DATA_DIR, REPO_ROOT
from bioagentics.pandas_pans.innate_immunity_modules import CGAS_STING_GENES

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"
BULK_PATH = DATA_DIR / "pandas_pans" / "ivig-mechanism-single-cell-analysis" / "bulk" / "GSE278678_pans_ivig_counts.csv.gz"

# Weights for composite score components
WEIGHTS = {
    "innate_adaptive_ratio_inv": 0.30,   # Lower ratio = more deficient
    "lectin_complement_inv": 0.25,       # Lower lectin = more deficient
    "cgas_sting_activation": 0.20,       # Higher = cGAS-STING activated
    "isg_signature": 0.25,              # Higher = type I IFN signature
}

# ISGs — functional readout of cGAS-STING activation
ISG_GENES = ["MX1", "ISG15", "IFIT1", "OAS1", "IFITM1", "IFI44L"]


def _assign_condition(sample: str) -> str:
    if sample.startswith("Control"):
        return "control"
    elif sample.startswith("PreIVIG"):
        return "pre"
    elif sample.startswith("PostIVIG"):
        return "post"
    elif sample.startswith("Secondpost"):
        return "second_post"
    return "unknown"


def _zscore(values: np.ndarray) -> np.ndarray:
    """Z-score normalization. Returns zeros if std is zero."""
    std = np.std(values, ddof=1)
    if std == 0 or np.isnan(std):
        return np.zeros_like(values, dtype=float)
    return (values - np.mean(values)) / std


def compute_bulk_score() -> pd.DataFrame:
    """Compute composite innate deficiency score from bulk RNA-seq.

    Uses innate_adaptive_ratio_bulk.csv for ratios and module sums,
    and extracts ISG expression from the raw bulk counts.
    """
    # Load ratio data
    ratio_path = OUTPUT_DIR / "innate_adaptive_ratio_bulk.csv"
    if not ratio_path.exists():
        logger.warning("Bulk ratio file not found")
        return pd.DataFrame()

    ratios = pd.read_csv(ratio_path)

    # Extract ISG expression from bulk counts
    isg_sums = {}
    if BULK_PATH.exists():
        bulk = pd.read_csv(BULK_PATH)
        sample_cols = [c for c in bulk.columns if c not in ("Unnamed: 0", "ENTREZID", "SYMBOL")]
        for sample in sample_cols:
            isg_rows = bulk[bulk["SYMBOL"].isin(ISG_GENES)]
            isg_sums[sample] = float(isg_rows[sample].sum()) if not isg_rows.empty else 0.0

    # Build per-sample feature matrix
    rows: list[dict] = []
    for _, row in ratios.iterrows():
        sample = row["sample"]
        rows.append({
            "sample": sample,
            "condition": row["condition"],
            "innate_adaptive_ratio": row["ratio"],
            "lectin_complement": row["innate_lectin_complement"],
            "cgas_sting": row["innate_cgas_sting_pathway"],
            "isg_sum": isg_sums.get(sample, np.nan),
        })

    df = pd.DataFrame(rows)

    # Z-score each component across all samples
    df["z_ratio_inv"] = -_zscore(df["innate_adaptive_ratio"].values)  # Inverted: lower ratio = higher score
    df["z_lectin_inv"] = -_zscore(df["lectin_complement"].values)      # Inverted: lower expression = higher score
    df["z_cgas_sting"] = _zscore(df["cgas_sting"].values)              # Higher activation = higher score
    df["z_isg"] = _zscore(df["isg_sum"].values)                        # Higher ISG = higher score

    # Composite weighted score
    df["deficiency_score"] = (
        WEIGHTS["innate_adaptive_ratio_inv"] * df["z_ratio_inv"]
        + WEIGHTS["lectin_complement_inv"] * df["z_lectin_inv"]
        + WEIGHTS["cgas_sting_activation"] * df["z_cgas_sting"]
        + WEIGHTS["isg_signature"] * df["z_isg"]
    )

    return df.sort_values("deficiency_score", ascending=False)


def compute_scrna_score() -> pd.DataFrame:
    """Compute composite innate deficiency score from scRNA-seq pseudobulk."""
    ratio_path = OUTPUT_DIR / "innate_adaptive_ratio_scrna.csv"
    if not ratio_path.exists():
        logger.warning("scRNA ratio file not found")
        return pd.DataFrame()

    ratios = pd.read_csv(ratio_path)

    # Build per-patient feature matrix (no ISG from scRNA — use cGAS-STING module as proxy)
    rows: list[dict] = []
    for _, row in ratios.iterrows():
        rows.append({
            "patient": row["patient"],
            "condition": row["condition"],
            "innate_adaptive_ratio": row["ratio"],
            "lectin_complement": row["innate_lectin_complement"],
            "cgas_sting": row["innate_cgas_sting_pathway"],
        })

    df = pd.DataFrame(rows)

    # Z-score components
    df["z_ratio_inv"] = -_zscore(df["innate_adaptive_ratio"].values)
    df["z_lectin_inv"] = -_zscore(df["lectin_complement"].values)
    df["z_cgas_sting"] = _zscore(df["cgas_sting"].values)

    # Composite (3-component, re-weighted)
    w_ratio = 0.40
    w_lectin = 0.30
    w_cgas = 0.30
    df["deficiency_score"] = (
        w_ratio * df["z_ratio_inv"]
        + w_lectin * df["z_lectin_inv"]
        + w_cgas * df["z_cgas_sting"]
    )

    return df.sort_values("deficiency_score", ascending=False)


def score_group_comparison(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compare deficiency scores between conditions."""
    results: list[dict] = []

    conditions = df["condition"].unique()
    if "pre" in conditions and "control" in conditions:
        pre = df[df["condition"] == "pre"]["deficiency_score"].values
        ctrl = df[df["condition"] == "control"]["deficiency_score"].values

        if len(pre) >= 2 and len(ctrl) >= 2:
            u_stat, p_val = sp_stats.mannwhitneyu(pre, ctrl, alternative="two-sided")
            results.append({
                "data_source": label,
                "comparison": "pre_vs_control",
                "mean_pans": float(np.mean(pre)),
                "mean_control": float(np.mean(ctrl)),
                "diff": float(np.mean(pre) - np.mean(ctrl)),
                "U_statistic": float(u_stat),
                "p_value": float(p_val),
                "direction": "pans_more_deficient" if np.mean(pre) > np.mean(ctrl) else "control_more_deficient",
            })

    if "post" in conditions and "pre" in conditions:
        post = df[df["condition"] == "post"]["deficiency_score"].values
        pre = df[df["condition"] == "pre"]["deficiency_score"].values

        if len(post) >= 2 and len(pre) >= 2:
            u_stat, p_val = sp_stats.mannwhitneyu(post, pre, alternative="two-sided")
            results.append({
                "data_source": label,
                "comparison": "post_vs_pre",
                "mean_pans": float(np.mean(post)),
                "mean_control": float(np.mean(pre)),
                "diff": float(np.mean(post) - np.mean(pre)),
                "U_statistic": float(u_stat),
                "p_value": float(p_val),
                "direction": "post_more_deficient" if np.mean(post) > np.mean(pre) else "post_less_deficient",
            })

    return pd.DataFrame(results)


def run_composite_score() -> dict[str, Path]:
    """Run composite innate deficiency score pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # 1. Bulk scores
    logger.info("=== Bulk composite innate deficiency score ===")
    bulk_scores = compute_bulk_score()
    if not bulk_scores.empty:
        path = OUTPUT_DIR / "composite_deficiency_score_bulk.csv"
        bulk_scores.to_csv(path, index=False)
        outputs["bulk_scores"] = path

        # Per-condition summary
        summary = bulk_scores.groupby("condition").agg(
            mean_score=("deficiency_score", "mean"),
            std_score=("deficiency_score", "std"),
            n=("sample", "count"),
        ).reset_index()
        logger.info("Bulk score summary:\n%s", summary.to_string(index=False))

        # Group comparison
        bulk_comp = score_group_comparison(bulk_scores, "bulk")
        if not bulk_comp.empty:
            logger.info("Bulk group comparison:\n%s", bulk_comp.to_string(index=False))

    # 2. scRNA scores
    logger.info("\n=== scRNA composite innate deficiency score ===")
    scrna_scores = compute_scrna_score()
    if not scrna_scores.empty:
        path = OUTPUT_DIR / "composite_deficiency_score_scrna.csv"
        scrna_scores.to_csv(path, index=False)
        outputs["scrna_scores"] = path

        summary = scrna_scores.groupby("condition").agg(
            mean_score=("deficiency_score", "mean"),
            std_score=("deficiency_score", "std"),
            n=("patient", "count"),
        ).reset_index()
        logger.info("scRNA score summary:\n%s", summary.to_string(index=False))

        scrna_comp = score_group_comparison(scrna_scores, "scrna")
        if not scrna_comp.empty:
            logger.info("scRNA group comparison:\n%s", scrna_comp.to_string(index=False))

    # 3. Combined comparison table
    logger.info("\n=== Combined comparison ===")
    comparisons = []
    if not bulk_scores.empty:
        comparisons.append(score_group_comparison(bulk_scores, "bulk"))
    if not scrna_scores.empty:
        comparisons.append(score_group_comparison(scrna_scores, "scrna"))
    if comparisons:
        combined = pd.concat(comparisons, ignore_index=True)
        path = OUTPUT_DIR / "composite_score_comparison.csv"
        combined.to_csv(path, index=False)
        outputs["comparison"] = path
        logger.info("\n%s", combined.to_string(index=False))

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_composite_score()
