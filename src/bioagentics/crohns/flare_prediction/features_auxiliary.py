"""Compute pathway dynamics and auxiliary omic features.

Features from HUMAnN pathways, host transcriptomics, and serology:
- Pathway abundance change rates (oxidative stress, butyrate, sulfur metabolism)
- Host transcriptomic inflammatory module scores (TNF, IL-17, IFN-gamma)
- Serologic marker trajectory slopes (ASCA, anti-CBir1, anti-OmpC)

Usage::

    from bioagentics.crohns.flare_prediction.features_auxiliary import (
        compute_auxiliary_features,
    )

    features = compute_auxiliary_features(data, windows)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)

# Key pathways to track specifically
KEY_PATHWAYS = {
    "oxidative_stress": ["oxidative", "reactive_oxygen", "glutathione"],
    "butyrate_production": ["butyrate", "butanoate", "butyryl"],
    "sulfur_metabolism": ["sulfur", "sulfate", "cysteine_methionine"],
    "succinate_oxphos": [
        "succinate",
        "succinyl",
        "oxidative_phosphorylation",
        "oxphos",
        "electron_transfer",
        "aerobic_respiration",
        "TCA",
        "citric_acid",
        "fumarate",
    ],
}

# Inflammatory gene modules for host transcriptomics
INFLAMMATORY_MODULES = {
    "tnf": ["TNF", "TNFRSF1A", "TNFRSF1B", "TRADD", "RIPK1", "NFKB1"],
    "il17": ["IL17A", "IL17F", "IL17RA", "IL17RC", "RORC", "IL23R"],
    "ifng": ["IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11"],
}

# Serologic markers
SERO_MARKERS = ["ASCA_IgA", "ASCA_IgG", "anti_CBir1", "anti_OmpC"]


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    result = linregress(x, values)
    return float(result.slope)


def _match_columns(df_columns: list[str], keywords: list[str]) -> list[str]:
    """Find columns matching any keyword (case-insensitive substring)."""
    matched = []
    for col in df_columns:
        col_lower = col.lower()
        if any(kw.lower() in col_lower for kw in keywords):
            matched.append(col)
    return matched


def _compute_pathway_features(
    samples: pd.DataFrame, meta_cols: set[str], max_pathway_slopes: int = 50,
) -> dict[str, float]:
    """Compute pathway dynamics features for a window's samples."""
    features: dict[str, float] = {}
    feature_cols = [c for c in samples.columns if c not in meta_cols]

    if len(samples) == 0 or not feature_cols:
        return features

    # Per-key-pathway group slopes
    for group_name, keywords in KEY_PATHWAYS.items():
        matched = _match_columns(feature_cols, keywords)
        if matched:
            group_means = samples[matched].mean(axis=1).tolist()
            features[f"pw_slope__{group_name}"] = _slope(group_means)
            features[f"pw_mean__{group_name}"] = float(np.nanmean(samples[matched].values))
        else:
            features[f"pw_slope__{group_name}"] = np.nan
            features[f"pw_mean__{group_name}"] = np.nan

    # Global pathway stats
    all_values = samples[feature_cols].values
    features["pw_mean_global"] = float(np.nanmean(all_values))
    features["pw_std_global"] = float(np.nanstd(all_values))

    # Per-pathway slopes (capped to top N most variable)
    if len(feature_cols) > max_pathway_slopes:
        stds = samples[feature_cols].std()
        slope_cols = stds.sort_values(ascending=False).head(max_pathway_slopes).index.tolist()
    else:
        slope_cols = feature_cols
    for col in slope_cols:
        features[f"pw_slope__{col}"] = _slope(samples[col].tolist())

    return features


def _compute_transcriptomic_features(
    samples: pd.DataFrame, meta_cols: set[str]
) -> dict[str, float]:
    """Compute host inflammatory module scores."""
    features: dict[str, float] = {}
    feature_cols = [c for c in samples.columns if c not in meta_cols]

    if len(samples) == 0 or not feature_cols:
        return features

    for module_name, genes in INFLAMMATORY_MODULES.items():
        matched = _match_columns(feature_cols, genes)
        if matched:
            # Z-score then mean across genes for module score
            module_vals = samples[matched].values
            with np.errstate(divide="ignore", invalid="ignore"):
                zscored = (module_vals - np.nanmean(module_vals, axis=0)) / np.nanstd(
                    module_vals, axis=0
                )
            zscored = np.nan_to_num(zscored, nan=0.0)
            module_scores = zscored.mean(axis=1).tolist()
            features[f"tx_module__{module_name}_mean"] = float(np.mean(module_scores))
            features[f"tx_module__{module_name}_slope"] = _slope(module_scores)
        else:
            features[f"tx_module__{module_name}_mean"] = np.nan
            features[f"tx_module__{module_name}_slope"] = np.nan

    return features


def _compute_serology_features(
    samples: pd.DataFrame,
) -> dict[str, float]:
    """Compute serologic marker trajectory slopes."""
    features: dict[str, float] = {}

    for marker in SERO_MARKERS:
        if marker in samples.columns and len(samples) > 0:
            values = samples[marker].dropna().tolist()
            features[f"sero_slope__{marker}"] = _slope(values)
            features[f"sero_mean__{marker}"] = float(np.nanmean(values)) if values else np.nan
        else:
            features[f"sero_slope__{marker}"] = np.nan
            features[f"sero_mean__{marker}"] = np.nan

    return features


def _get_window_samples(
    df: pd.DataFrame, window: Window
) -> pd.DataFrame:
    """Extract samples within a window for a given layer."""
    df_reset = df.reset_index()
    df_reset["date"] = pd.to_datetime(df_reset["date"])
    mask = (
        (df_reset["subject_id"] == window.subject_id)
        & (df_reset["date"] >= window.window_start)
        & (df_reset["date"] <= window.window_end)
    )
    return df_reset.loc[mask].sort_values("date")


def compute_auxiliary_features(
    data: dict[str, pd.DataFrame | None],
    windows: list[Window],
) -> pd.DataFrame:
    """Compute auxiliary features from pathways, transcriptomics, and serology.

    Parameters
    ----------
    data:
        Dict of omic DataFrames from ``HMP2DataLoader.load_all()``.
    windows:
        List of classification windows.

    Returns
    -------
    DataFrame indexed by window index with auxiliary feature columns.
    NaN for unavailable layers.
    """
    pathways_df = data.get("pathways")
    tx_df = data.get("transcriptomics")
    sero_df = data.get("serology")

    meta_cols = {"subject_id", "visit_num", "date", "diagnosis"}
    feature_rows: list[dict[str, float]] = []

    for window in windows:
        features: dict[str, float] = {}

        # Pathway features
        if pathways_df is not None:
            pw_samples = _get_window_samples(pathways_df, window)
            features.update(_compute_pathway_features(pw_samples, meta_cols))

        # Transcriptomic features
        if tx_df is not None:
            tx_samples = _get_window_samples(tx_df, window)
            features.update(_compute_transcriptomic_features(tx_samples, meta_cols))

        # Serology features
        if sero_df is not None:
            sero_samples = _get_window_samples(sero_df, window)
            features.update(_compute_serology_features(sero_samples))

        feature_rows.append(features)

    result = pd.DataFrame(feature_rows, index=range(len(windows)))
    result.index.name = "instance_id"
    logger.info("Computed %d auxiliary features for %d instances", result.shape[1], len(windows))
    return result
