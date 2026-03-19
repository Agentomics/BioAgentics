"""Feature engineering module for sepsis prediction.

From hourly extracted vitals/labs, computes:
- Delta values (1h, 6h, 12h differences)
- Rolling statistics (mean, std, slope over 6h windows)
- Missingness indicators per feature

Output: flat feature matrix per time-step.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.diagnostics.sepsis.config import (
    ALL_FEATURES,
    DATA_DIR,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

DELTA_WINDOWS = [1, 6, 12]
ROLLING_WINDOW = 6


def compute_deltas(
    group: pd.DataFrame, features: list[str], windows: list[int] | None = None
) -> pd.DataFrame:
    """Compute delta values (current - value N hours ago) for each feature.

    Parameters
    ----------
    group : Single-admission hourly DataFrame sorted by hours_in.
    features : Feature column names to compute deltas for.
    windows : List of hour deltas (default: [1, 6, 12]).

    Returns
    -------
    DataFrame with delta columns named {feature}_delta_{window}h.
    """
    if windows is None:
        windows = DELTA_WINDOWS
    result = pd.DataFrame(index=group.index)
    for feat in features:
        if feat not in group.columns:
            continue
        col = group[feat]
        for w in windows:
            result[f"{feat}_delta_{w}h"] = col - col.shift(w)
    return result


def compute_rolling_stats(
    group: pd.DataFrame, features: list[str], window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """Compute rolling mean, std, and slope over a window for each feature.

    Parameters
    ----------
    group : Single-admission hourly DataFrame sorted by hours_in.
    features : Feature column names.
    window : Rolling window size in hours (default: 6).

    Returns
    -------
    DataFrame with columns {feature}_roll_mean, {feature}_roll_std, {feature}_roll_slope.
    """
    result = pd.DataFrame(index=group.index)
    for feat in features:
        if feat not in group.columns:
            continue
        col = group[feat]
        rolling = col.rolling(window=window, min_periods=1)
        result[f"{feat}_roll_mean"] = rolling.mean()
        result[f"{feat}_roll_std"] = rolling.std()

        # Slope: linear regression coefficient over window
        # Use rolling apply with a simple least-squares slope
        def _slope(arr: np.ndarray) -> float:
            n = len(arr)
            valid = ~np.isnan(arr)
            if valid.sum() < 2:
                return np.nan
            x = np.arange(n)[valid]
            y = arr[valid]
            x_mean = x.mean()
            y_mean = y.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom == 0:
                return 0.0
            return float(((x - x_mean) * (y - y_mean)).sum() / denom)

        result[f"{feat}_roll_slope"] = col.rolling(
            window=window, min_periods=2
        ).apply(_slope, raw=True)

    return result


def compute_missingness(
    group: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Compute missingness indicators for each feature.

    For each feature, generates:
    - {feature}_missing: 1 if NaN at this time-step, 0 otherwise
    - {feature}_hours_since_obs: hours since last non-NaN observation

    Parameters
    ----------
    group : Single-admission hourly DataFrame sorted by hours_in.
    features : Feature column names.
    """
    result = pd.DataFrame(index=group.index)
    for feat in features:
        if feat not in group.columns:
            result[f"{feat}_missing"] = 1
            result[f"{feat}_hours_since_obs"] = np.nan
            continue

        is_missing = group[feat].isna().astype(int)
        result[f"{feat}_missing"] = is_missing

        # Hours since last observation
        observed_mask = ~group[feat].isna()
        last_obs_idx = observed_mask.where(observed_mask).ffill()
        # Compute distance in rows (hours)
        cumcount = np.arange(len(group))
        last_obs_pos = pd.Series(cumcount, index=group.index)
        last_obs_pos[~observed_mask] = np.nan
        last_obs_pos = last_obs_pos.ffill()
        result[f"{feat}_hours_since_obs"] = cumcount - last_obs_pos.values

    return result


def engineer_features_single(
    group: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute all engineered features for a single admission.

    Parameters
    ----------
    group : Hourly DataFrame for one admission, sorted by hours_in.
    features : Feature names to engineer (default: ALL_FEATURES from config).

    Returns
    -------
    DataFrame with original + engineered feature columns.
    """
    if features is None:
        features = ALL_FEATURES

    available = [f for f in features if f in group.columns]
    group = group.sort_values("hours_in")

    deltas = compute_deltas(group, available)
    rolling = compute_rolling_stats(group, available)
    missing = compute_missingness(group, features)  # Pass all features for missingness

    return pd.concat([group, deltas, rolling, missing], axis=1)


def engineer_features(
    hourly_features: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute engineered features for all admissions.

    Parameters
    ----------
    hourly_features : Full hourly feature matrix with subject_id, hadm_id, hours_in.
    features : Feature names to engineer (default: ALL_FEATURES from config).

    Returns
    -------
    DataFrame with all original and engineered columns.
    """
    if features is None:
        features = ALL_FEATURES

    result_frames = []
    grouped = hourly_features.groupby(["subject_id", "hadm_id"])
    total = grouped.ngroups
    for i, ((sid, hid), group) in enumerate(grouped):
        if (i + 1) % 1000 == 0:
            logger.info("Engineering features: %d/%d admissions", i + 1, total)
        enriched = engineer_features_single(group, features)
        result_frames.append(enriched)

    if not result_frames:
        return pd.DataFrame()

    return pd.concat(result_frames, ignore_index=True)


def run_feature_engineering(
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Run feature engineering on extracted hourly features.

    Reads hourly_features.parquet, outputs engineered_features.parquet.
    """
    output_path = output_dir / "engineered_features.parquet"
    if output_path.exists() and not force:
        logger.info("Loading cached engineered features from %s", output_path)
        return pd.read_parquet(output_path)

    input_path = output_dir / "hourly_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Run data extraction first: {input_path} not found"
        )
    hourly_features = pd.read_parquet(input_path)

    result = engineer_features(hourly_features)
    logger.info("Engineered features: %d rows, %d columns", *result.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info("Saved to %s", output_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute engineered features")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_feature_engineering(args.output_dir, force=args.force)

    print(f"\n=== Feature Engineering Complete ===")
    print(f"Rows: {len(result)}")
    print(f"Total columns: {len(result.columns)}")
    n_eng = len([c for c in result.columns if "_delta_" in c or "_roll_" in c or "_missing" in c])
    print(f"Engineered columns: {n_eng}")


if __name__ == "__main__":
    main()
