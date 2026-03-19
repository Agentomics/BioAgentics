"""Classify cytokines as innate vs adaptive and compute flare directional changes.

Re-analyzes existing cytokine-network-flare-prediction meta-analysis results,
classifying each cytokine by immune arm and computing per-category flare
direction and magnitude.

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.cytokine_classification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.pandas_pans.innate_immunity_modules import CYTOKINE_CLASSIFICATION

logger = logging.getLogger(__name__)

CYTOKINE_OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "cytokine-network-flare-prediction"
OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"

# Map protein names (from meta-analysis) to classification
# Uses CYTOKINE_CLASSIFICATION from innate_immunity_modules
_PROTEIN_TO_CATEGORY: dict[str, str] = {}
for _cat, _info in CYTOKINE_CLASSIFICATION.items():
    for _prot in _info["proteins"]:
        _PROTEIN_TO_CATEGORY[_prot] = _cat

# Additional mappings for analyte names that differ from standard protein names
_ANALYTE_OVERRIDES: dict[str, str] = {
    "S100B": "innate",      # Neuronal damage marker, released via innate BBB disruption
    "TGF-β1": "regulatory",
}


def classify_cytokines(
    meta_path: Path | None = None,
) -> pd.DataFrame:
    """Classify meta-analysis cytokine results by immune arm.

    Returns DataFrame with columns:
        analyte, category, pooled_g, p_value, direction, significant,
        k (number of studies)
    """
    if meta_path is None:
        meta_path = CYTOKINE_OUTPUT_DIR / "meta_analysis_summary.csv"

    if not meta_path.exists():
        logger.warning("Meta-analysis file not found: %s", meta_path)
        return pd.DataFrame()

    meta = pd.read_csv(meta_path)

    # Classify each analyte
    categories = []
    for analyte in meta["analyte"]:
        if analyte in _ANALYTE_OVERRIDES:
            categories.append(_ANALYTE_OVERRIDES[analyte])
        elif analyte in _PROTEIN_TO_CATEGORY:
            categories.append(_PROTEIN_TO_CATEGORY[analyte])
        else:
            categories.append("unclassified")

    meta["category"] = categories

    return meta.sort_values(by=["category", "analyte"])


def compute_category_summary(classified: pd.DataFrame) -> pd.DataFrame:
    """Compute per-category summary of flare directional changes.

    Returns DataFrame with columns:
        category, n_cytokines, mean_effect_size, median_effect_size,
        n_up, n_down, n_ns, dominant_direction
    """
    results = []
    for cat in sorted(classified["category"].unique()):
        cat_df = classified[classified["category"] == cat]
        sig = cat_df[cat_df["significant"]]

        n_up = int((sig["direction"] == "up").sum())
        n_down = int((sig["direction"] == "down").sum())
        n_ns = int((~cat_df["significant"]).sum())

        if n_up > n_down:
            dominant = "up"
        elif n_down > n_up:
            dominant = "down"
        else:
            dominant = "mixed"

        results.append({
            "category": cat,
            "n_cytokines": len(cat_df),
            "mean_effect_size": float(cat_df["pooled_g"].mean()),
            "median_effect_size": float(cat_df["pooled_g"].median()),
            "mean_abs_effect": float(cat_df["pooled_g"].abs().mean()),
            "n_up": n_up,
            "n_down": n_down,
            "n_ns": n_ns,
            "dominant_direction": dominant,
            "cytokines_up": ", ".join(sig[sig["direction"] == "up"]["analyte"].tolist()),
            "cytokines_down": ", ".join(sig[sig["direction"] == "down"]["analyte"].tolist()),
        })

    return pd.DataFrame(results)


def compute_flare_profile(
    extracted_path: Path | None = None,
) -> pd.DataFrame:
    """Compute classified flare vs remission cytokine profile from raw extracted data.

    Uses extracted_cytokines.csv which has per-study measurements.

    Returns DataFrame with columns:
        analyte, category, condition, mean_value, std_value, n_studies
    """
    if extracted_path is None:
        extracted_path = (
            REPO_ROOT / "data" / "pandas_pans"
            / "cytokine-network-flare-prediction" / "extracted_cytokines.csv"
        )

    if not extracted_path.exists():
        logger.warning("Extracted cytokines not found: %s", extracted_path)
        return pd.DataFrame()

    raw = pd.read_csv(extracted_path)

    # Convert mean_or_median to numeric
    raw["value"] = pd.to_numeric(raw["mean_or_median"], errors="coerce")
    raw = raw.dropna(subset=["value"])

    # Classify analytes
    raw["category"] = raw["analyte_name"].map(
        lambda x: _ANALYTE_OVERRIDES.get(x, _PROTEIN_TO_CATEGORY.get(x, "unclassified"))
    )

    # Aggregate per analyte × condition
    summary = (
        raw.groupby(["analyte_name", "category", "condition"])
        .agg(
            mean_value=("value", "mean"),
            std_value=("value", "std"),
            n_studies=("study_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"analyte_name": "analyte"})
    )

    return summary.sort_values(by=["category", "analyte", "condition"])


def run_classification() -> dict[str, Path]:
    """Run full cytokine classification pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # 1. Classify meta-analysis results
    logger.info("=== Classifying cytokines ===")
    classified = classify_cytokines()
    if not classified.empty:
        path = OUTPUT_DIR / "cytokine_classified_meta.csv"
        classified.to_csv(path, index=False)
        outputs["classified_meta"] = path
        logger.info("Classified %d cytokines", len(classified))
        logger.info("\n%s", classified[["analyte", "category", "pooled_g", "direction", "significant"]].to_string(index=False))

    # 2. Per-category summary
    logger.info("\n=== Category summary ===")
    cat_summary = compute_category_summary(classified)
    if not cat_summary.empty:
        path = OUTPUT_DIR / "cytokine_category_summary.csv"
        cat_summary.to_csv(path, index=False)
        outputs["category_summary"] = path
        logger.info("\n%s", cat_summary[["category", "n_cytokines", "mean_effect_size", "dominant_direction", "cytokines_up", "cytokines_down"]].to_string(index=False))

    # 3. Raw flare profile
    logger.info("\n=== Flare profile from raw data ===")
    profile = compute_flare_profile()
    if not profile.empty:
        path = OUTPUT_DIR / "cytokine_flare_profile.csv"
        profile.to_csv(path, index=False)
        outputs["flare_profile"] = path
        logger.info("Saved flare profile: %d entries", len(profile))

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_classification()
