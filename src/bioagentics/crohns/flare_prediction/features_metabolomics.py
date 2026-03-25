"""Compute metabolomic trajectory features from untargeted metabolomics.

Features computed per classification window:
- Sliding-window slopes for known metabolite classes (bile acids, SCFAs, tryptophan, acylcarnitines)
- Rate-of-change for top variable metabolites
- Window-level summary stats (mean, std, slope, range)

Usage::

    from bioagentics.crohns.flare_prediction.features_metabolomics import (
        compute_metabolomic_features,
    )

    features = compute_metabolomic_features(metabolomics_df, windows)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)

# Known metabolite class prefixes for targeted feature engineering
METABOLITE_CLASSES = {
    "bile_acid": ["bile", "cholate", "deoxycholate", "chenodeoxycholate", "ursodeoxycholate", "lithocholate"],
    "scfa": ["butyrate", "propionate", "acetate", "valerate", "caproate"],
    "tryptophan": ["tryptophan", "kynurenine", "indole", "serotonin", "melatonin"],
    "acylcarnitine": ["carnitine", "acylcarnitine"],
}

# Validated pre-flare metabolite candidates (RD-approved, journal #1950).
# Levhar PMID 40650893: urate, 3-HB, acetoacetate, trehalose, mesaconic acid.
CANDIDATE_METABOLITES: dict[str, list[str]] = {
    "urate": ["urate", "uric_acid", "uric acid"],
    "3hb": ["3-hydroxybutyrate", "3_hydroxybutyrate", "beta-hydroxybutyrate", "bhb", "3-hb"],
    "acetoacetate": ["acetoacetate", "acetoacetic"],
    "trehalose": ["trehalose"],
    "mesaconic_acid": ["mesaconate", "mesaconic"],
}


def _slope(values: list[float]) -> float:
    """Compute linear slope. Returns 0 if < 2 values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    result = linregress(x, values)
    return float(result.slope)


def _find_candidate_columns(
    df_columns: list[str], meta_cols: set[str]
) -> dict[str, list[str]]:
    """Match candidate metabolite names to actual DataFrame columns."""
    candidates: dict[str, list[str]] = {}
    feature_cols = [c for c in df_columns if c not in meta_cols]
    for candidate, keywords in CANDIDATE_METABOLITES.items():
        matched = []
        for col in feature_cols:
            col_lower = col.lower()
            if any(kw.lower() in col_lower for kw in keywords):
                matched.append(col)
        candidates[candidate] = matched
    return candidates


def _classify_metabolite(name: str) -> str | None:
    """Return the metabolite class for a column name, or None."""
    name_lower = name.lower()
    for cls, keywords in METABOLITE_CLASSES.items():
        if any(kw in name_lower for kw in keywords):
            return cls
    return None


def _top_variable_metabolites(
    df: pd.DataFrame, meta_cols: set[str], top_n: int = 200
) -> list[str]:
    """Return the top_n most variable metabolite columns by CV."""
    feature_cols = [c for c in df.columns if c not in meta_cols]
    if not feature_cols:
        return []
    stds = df[feature_cols].std()
    means = df[feature_cols].mean().abs()
    # CV = std / |mean|, avoid division by zero
    cv = stds / means.replace(0, np.nan)
    cv = cv.dropna().sort_values(ascending=False)
    return cv.head(top_n).index.tolist()


def compute_metabolomic_features(
    metabolomics: pd.DataFrame,
    windows: list[Window],
    top_n_variable: int = 200,
) -> pd.DataFrame:
    """Compute metabolomic trajectory features for each classification window.

    Parameters
    ----------
    metabolomics:
        Untargeted metabolomics table indexed by (subject_id, visit_num)
        with a ``date`` column.
    windows:
        List of classification windows.
    top_n_variable:
        Number of top-variable metabolites to include slope features for.

    Returns
    -------
    DataFrame indexed by window index with metabolomic feature columns.
    """
    met_reset = metabolomics.reset_index()
    met_reset["date"] = pd.to_datetime(met_reset["date"])

    meta_cols = {"subject_id", "visit_num", "date", "diagnosis"}
    feature_cols = [c for c in met_reset.columns if c not in meta_cols]

    # Identify top variable metabolites across the full dataset
    top_var = _top_variable_metabolites(met_reset, meta_cols, top_n=top_n_variable)

    # Classify metabolites into known classes
    class_cols: dict[str, list[str]] = {cls: [] for cls in METABOLITE_CLASSES}
    for col in feature_cols:
        cls = _classify_metabolite(col)
        if cls:
            class_cols[cls].append(col)

    # Match named candidate metabolites to actual columns
    candidate_cols = _find_candidate_columns(met_reset.columns.tolist(), meta_cols)

    feature_rows: list[dict[str, float]] = []

    for window in windows:
        mask = (
            (met_reset["subject_id"] == window.subject_id)
            & (met_reset["date"] >= window.window_start)
            & (met_reset["date"] <= window.window_end)
        )
        samples = met_reset.loc[mask].sort_values("date")
        features: dict[str, float] = {}

        if len(samples) == 0:
            feature_rows.append(features)
            continue

        values = samples[feature_cols].values

        # Window-level summary stats for all metabolites
        features["met_mean_global"] = float(np.nanmean(values))
        features["met_std_global"] = float(np.nanstd(values))

        # Per-class slopes
        for cls, cols in class_cols.items():
            if not cols:
                continue
            class_means = samples[cols].mean(axis=1).tolist()
            features[f"met_slope__{cls}"] = _slope(class_means)
            features[f"met_mean__{cls}"] = float(np.nanmean(samples[cols].values))

        # Top-variable metabolite slopes
        for col in top_var:
            if col in samples.columns:
                col_values = samples[col].tolist()
                features[f"met_slope__{col}"] = _slope(col_values)
                features[f"met_mean__{col}"] = float(np.nanmean(samples[col].values))
                features[f"met_std__{col}"] = float(np.nanstd(samples[col].values))
                features[f"met_range__{col}"] = float(
                    np.nanmax(samples[col].values) - np.nanmin(samples[col].values)
                )

        # Named candidate metabolite features (Levhar PMID 40650893)
        for cand_name, cand_cols in candidate_cols.items():
            if not cand_cols:
                features[f"met_cand_slope__{cand_name}"] = np.nan
                features[f"met_cand_mean__{cand_name}"] = np.nan
                features[f"met_cand_std__{cand_name}"] = np.nan
                features[f"met_cand_range__{cand_name}"] = np.nan
                continue
            cand_mean_series = samples[cand_cols].mean(axis=1).tolist()
            cand_values = samples[cand_cols].values
            features[f"met_cand_slope__{cand_name}"] = _slope(cand_mean_series)
            features[f"met_cand_mean__{cand_name}"] = float(np.nanmean(cand_values))
            features[f"met_cand_std__{cand_name}"] = float(np.nanstd(cand_values))
            features[f"met_cand_range__{cand_name}"] = float(
                np.nanmax(cand_values) - np.nanmin(cand_values)
            )

        feature_rows.append(features)

    result = pd.DataFrame(feature_rows, index=range(len(windows)))
    result.index.name = "instance_id"
    logger.info("Computed %d metabolomic features for %d instances", result.shape[1], len(windows))
    return result
