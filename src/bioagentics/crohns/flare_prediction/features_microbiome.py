"""Compute microbiome temporal features from MetaPhlAn species profiles.

Features computed per classification window:
- Bray-Curtis dissimilarity between consecutive samples
- Shannon diversity trajectory slopes
- Species-level abundance change rates
- Community volatility index

Usage::

    from bioagentics.crohns.flare_prediction.features_microbiome import (
        compute_microbiome_features,
    )

    features = compute_microbiome_features(species_df, windows)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import braycurtis
from scipy.stats import linregress

from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)


def _shannon_diversity(abundances: np.ndarray) -> float:
    """Compute Shannon diversity index for a single sample."""
    p = abundances[abundances > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def _bray_curtis_consecutive(samples: np.ndarray) -> list[float]:
    """Compute Bray-Curtis dissimilarity between consecutive samples."""
    dists = []
    for i in range(1, len(samples)):
        d = braycurtis(samples[i - 1], samples[i])
        if np.isfinite(d):
            dists.append(float(d))
    return dists


def _slope(values: list[float]) -> float:
    """Compute linear slope over a series. Returns 0 if < 2 values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    result = linregress(x, values)
    return float(result.slope)


def compute_microbiome_features(
    species: pd.DataFrame,
    windows: list[Window],
    max_species_slopes: int = 50,
) -> pd.DataFrame:
    """Compute microbiome temporal features for each classification window.

    Parameters
    ----------
    species:
        MetaPhlAn species abundance table indexed by (subject_id, visit_num)
        with a ``date`` column.
    windows:
        List of classification windows.

    Returns
    -------
    DataFrame indexed by window index with microbiome feature columns.
    """
    species_reset = species.reset_index()
    species_reset["date"] = pd.to_datetime(species_reset["date"])

    # Identify abundance columns (exclude metadata)
    meta_cols = {"subject_id", "visit_num", "date", "diagnosis"}
    abundance_cols = [c for c in species_reset.columns if c not in meta_cols]

    # Cap per-species slopes to the top N most variable species by CV
    if len(abundance_cols) > max_species_slopes:
        stds = species_reset[abundance_cols].std()
        means = species_reset[abundance_cols].mean().abs()
        cv = stds / means.replace(0, np.nan)
        cv = cv.dropna().sort_values(ascending=False)
        slope_cols = cv.head(max_species_slopes).index.tolist()
        logger.info(
            "Capping per-species slopes: %d -> %d (top by CV)",
            len(abundance_cols), len(slope_cols),
        )
    else:
        slope_cols = abundance_cols

    feature_rows: list[dict[str, float]] = []

    for idx, window in enumerate(windows):
        mask = (
            (species_reset["subject_id"] == window.subject_id)
            & (species_reset["date"] >= window.window_start)
            & (species_reset["date"] <= window.window_end)
        )
        samples = species_reset.loc[mask].sort_values("date")
        abundances = samples[abundance_cols].values

        features: dict[str, float] = {}

        if len(abundances) == 0:
            # No samples in window — fill with NaN
            features["mb_shannon_mean"] = np.nan
            features["mb_shannon_slope"] = np.nan
            features["mb_bc_mean"] = np.nan
            features["mb_bc_max"] = np.nan
            features["mb_volatility_index"] = np.nan
            for col in slope_cols:
                features[f"mb_slope__{col}"] = np.nan
            feature_rows.append(features)
            continue

        # Shannon diversity per sample
        diversities = [_shannon_diversity(row) for row in abundances]
        features["mb_shannon_mean"] = float(np.mean(diversities))
        features["mb_shannon_slope"] = _slope(diversities)

        # Bray-Curtis between consecutive samples
        bc_dists = _bray_curtis_consecutive(abundances)
        features["mb_bc_mean"] = float(np.mean(bc_dists)) if bc_dists else 0.0
        features["mb_bc_max"] = float(np.max(bc_dists)) if bc_dists else 0.0
        features["mb_volatility_index"] = float(np.mean(bc_dists)) if bc_dists else 0.0

        # Per-species abundance slopes within window (capped to top variable species)
        for col in slope_cols:
            col_idx = abundance_cols.index(col)
            values = abundances[:, col_idx].tolist()
            features[f"mb_slope__{col}"] = _slope(values)

        feature_rows.append(features)

    result = pd.DataFrame(feature_rows, index=range(len(windows)))
    result.index.name = "instance_id"
    logger.info("Computed %d microbiome features for %d instances", result.shape[1], len(windows))
    return result
