"""Phase 3 (partial) — Innate deficiency score vs IVIG transcriptional response.

Task 107 (partial): Test whether pre-treatment innate deficiency score predicts
IVIG transcriptional response magnitude. Full task requires clinical outcome
metadata (severity, flare frequency) which is not yet available.

What we CAN test with available data:
  - Pre-treatment deficiency score vs magnitude of transcriptional change post-IVIG
  - Hypothesis: patients with higher innate deficiency show greater transcriptional
    response to IVIG (because IVIG provides passive antibodies that compensate for
    the innate clearance failure, producing a larger observable shift)

Data sources:
  - Bulk RNA-seq: GSE278678 (9 pre-IVIG, 9 post-IVIG paired samples)
  - scRNA-seq: Han VX (5 patients pre/post, 4 controls)
  - Composite deficiency scores from Task 106 outputs

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.deficiency_ivig_response
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _extract_patient_id(sample: str) -> str:
    """Extract numeric patient ID from sample names like 'PreIVIG3', 'PostIVIG3'."""
    for prefix in ("PreIVIG", "PostIVIG", "SecondpostIVIG", "Control"):
        if sample.startswith(prefix):
            return sample[len(prefix):]
    return sample


def load_bulk_scores() -> pd.DataFrame:
    """Load bulk composite deficiency scores."""
    path = OUTPUT_DIR / "composite_deficiency_score_bulk.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_scrna_scores() -> pd.DataFrame:
    """Load scRNA composite deficiency scores."""
    path = OUTPUT_DIR / "composite_deficiency_score_scrna.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# IVIG response computation
# ---------------------------------------------------------------------------


def compute_bulk_ivig_response(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute per-patient IVIG transcriptional response from bulk data.

    For each patient with both pre and post samples, compute:
    - Change in deficiency score (post - pre)
    - Change in each component (ratio, lectin, cGAS-STING, ISG)
    """
    if scores.empty:
        return pd.DataFrame()

    pre = scores[scores["condition"] == "pre"].copy()
    post = scores[scores["condition"] == "post"].copy()

    pre["patient_id"] = pre["sample"].apply(_extract_patient_id)
    post["patient_id"] = post["sample"].apply(_extract_patient_id)

    merged = pre.merge(post, on="patient_id", suffixes=("_pre", "_post"))
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in merged.iterrows():
        pid = row["patient_id"]
        pre_score = row["deficiency_score_pre"]
        post_score = row["deficiency_score_post"]
        delta = post_score - pre_score

        r: dict = {
            "patient_id": pid,
            "pre_deficiency_score": pre_score,
            "post_deficiency_score": post_score,
            "delta_deficiency_score": delta,
            "abs_delta_deficiency_score": abs(delta),
            "direction": "improved" if delta < 0 else "worsened",
        }

        # Component-level changes
        for component in ["innate_adaptive_ratio", "lectin_complement", "cgas_sting"]:
            if f"{component}_pre" in row and f"{component}_post" in row:
                r[f"delta_{component}"] = (
                    row[f"{component}_post"] - row[f"{component}_pre"]
                )

        if "isg_sum_pre" in row and "isg_sum_post" in row:
            r["delta_isg_sum"] = row["isg_sum_post"] - row["isg_sum_pre"]

        rows.append(r)

    return pd.DataFrame(rows).sort_values("pre_deficiency_score", ascending=False)


def compute_scrna_ivig_response(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute per-patient IVIG response from scRNA data."""
    if scores.empty:
        return pd.DataFrame()

    pre = scores[scores["condition"] == "pre"].copy()
    post = scores[scores["condition"] == "post"].copy()

    # Patient column in scRNA data
    patient_col = "patient" if "patient" in pre.columns else "sample"

    merged = pre.merge(post, on=patient_col, suffixes=("_pre", "_post"))
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in merged.iterrows():
        pid = row[patient_col]
        pre_score = row["deficiency_score_pre"]
        post_score = row["deficiency_score_post"]
        delta = post_score - pre_score

        rows.append({
            "patient": pid,
            "pre_deficiency_score": pre_score,
            "post_deficiency_score": post_score,
            "delta_deficiency_score": delta,
            "abs_delta_deficiency_score": abs(delta),
            "direction": "improved" if delta < 0 else "worsened",
        })

    return pd.DataFrame(rows).sort_values("pre_deficiency_score", ascending=False)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def correlate_deficiency_response(
    response_df: pd.DataFrame,
    label: str,
) -> dict:
    """Test correlation between pre-treatment deficiency score and IVIG response."""
    if response_df.empty or len(response_df) < 3:
        return {"data_source": label, "n": len(response_df), "error": "insufficient_data"}

    pre_scores = response_df["pre_deficiency_score"].values
    abs_deltas = response_df["abs_delta_deficiency_score"].values

    n = len(pre_scores)
    result: dict = {"data_source": label, "n": n}

    # Spearman (rank-based, robust to small n)
    if n >= 3:
        rho, p_val = sp_stats.spearmanr(pre_scores, abs_deltas)
        result["spearman_rho"] = float(rho) if not np.isnan(rho) else None
        result["spearman_p"] = float(p_val) if not np.isnan(p_val) else None

    # Pearson
    if n >= 3:
        r, p_val = sp_stats.pearsonr(pre_scores, abs_deltas)
        result["pearson_r"] = float(r) if not np.isnan(r) else None
        result["pearson_p"] = float(p_val) if not np.isnan(p_val) else None

    # Direction test: signed delta
    signed_deltas = response_df["delta_deficiency_score"].values
    if n >= 3:
        rho, p_val = sp_stats.spearmanr(pre_scores, signed_deltas)
        result["signed_spearman_rho"] = float(rho) if not np.isnan(rho) else None
        result["signed_spearman_p"] = float(p_val) if not np.isnan(p_val) else None

    # Median split: high vs low deficiency IVIG response
    median_score = np.median(pre_scores)
    high = response_df[response_df["pre_deficiency_score"] >= median_score]["abs_delta_deficiency_score"]
    low = response_df[response_df["pre_deficiency_score"] < median_score]["abs_delta_deficiency_score"]
    if len(high) >= 2 and len(low) >= 2:
        u_stat, p_val = sp_stats.mannwhitneyu(high, low, alternative="two-sided")
        result["median_split_U"] = float(u_stat)
        result["median_split_p"] = float(p_val)
        result["high_deficiency_mean_response"] = float(high.mean())
        result["low_deficiency_mean_response"] = float(low.mean())

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_deficiency_ivig_response() -> dict[str, Path]:
    """Run deficiency score vs IVIG response analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # 1. Bulk analysis
    logger.info("=== Bulk: deficiency score vs IVIG response ===")
    bulk_scores = load_bulk_scores()
    bulk_response = compute_bulk_ivig_response(bulk_scores)
    if not bulk_response.empty:
        path = OUTPUT_DIR / "deficiency_ivig_response_bulk.csv"
        bulk_response.to_csv(path, index=False)
        outputs["bulk_response"] = path
        logger.info("Bulk IVIG response (n=%d patients):", len(bulk_response))
        logger.info("\n%s", bulk_response[
            ["patient_id", "pre_deficiency_score", "delta_deficiency_score",
             "abs_delta_deficiency_score", "direction"]
        ].to_string(index=False))

        bulk_corr = correlate_deficiency_response(bulk_response, "bulk")
        logger.info("\nBulk correlation: %s", {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in bulk_corr.items()
        })

    # 2. scRNA analysis
    logger.info("\n=== scRNA: deficiency score vs IVIG response ===")
    scrna_scores = load_scrna_scores()
    scrna_response = compute_scrna_ivig_response(scrna_scores)
    if not scrna_response.empty:
        path = OUTPUT_DIR / "deficiency_ivig_response_scrna.csv"
        scrna_response.to_csv(path, index=False)
        outputs["scrna_response"] = path
        logger.info("scRNA IVIG response (n=%d patients):", len(scrna_response))
        logger.info("\n%s", scrna_response.to_string(index=False))

        scrna_corr = correlate_deficiency_response(scrna_response, "scrna")
        logger.info("\nscRNA correlation: %s", {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in scrna_corr.items()
        })

    # 3. Combined summary
    logger.info("\n=== Summary ===")
    summary: dict = {
        "analysis": "deficiency_score_vs_ivig_response",
        "note": "Partial task 107 — clinical outcome metadata not yet available",
        "available_tests": ["deficiency_score_vs_transcriptional_response"],
        "unavailable_tests": [
            "deficiency_score_vs_disease_severity",
            "deficiency_score_vs_flare_frequency",
            "deficiency_score_vs_clinical_ivig_response",
        ],
    }

    correlations = []
    if not bulk_response.empty:
        summary["bulk_n_patients"] = len(bulk_response)
        correlations.append(correlate_deficiency_response(bulk_response, "bulk"))
    if not scrna_response.empty:
        summary["scrna_n_patients"] = len(scrna_response)
        correlations.append(correlate_deficiency_response(scrna_response, "scrna"))

    summary["correlations"] = correlations

    path = OUTPUT_DIR / "deficiency_ivig_response_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    outputs["summary"] = path

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_deficiency_ivig_response()
