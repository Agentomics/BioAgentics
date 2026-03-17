"""Single-marker baseline models for benchmarking flare prediction.

Baselines:
- Calprotectin-only: logistic regression on fecal calprotectin trajectory
- CRP-only: logistic regression on CRP trajectory features
- Diversity-only: logistic regression on Shannon diversity trajectory

All baselines use the same LOPO-CV framework as the primary classifier.

Usage::

    from bioagentics.crohns.flare_prediction.baselines import (
        run_all_baselines, compare_models,
    )

    baseline_results = run_all_baselines(features, windows)
    comparison = compare_models(primary_metrics, baseline_results)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from bioagentics.crohns.flare_prediction.classifier import (
    lopo_cv,
    evaluate_results,
    CVResults,
)
from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    result = linregress(x, values)
    return float(result.slope)


def extract_single_marker_features(
    marker_name: str,
    marker_data: pd.DataFrame | None,
    windows: list[Window],
) -> pd.DataFrame:
    """Extract trajectory features for a single clinical marker.

    Computes: mean, slope, last value, max value, range within each window.

    Parameters
    ----------
    marker_name:
        Name prefix for feature columns.
    marker_data:
        DataFrame with 'subject_id', 'date', and marker value column(s).
    windows:
        Classification windows.

    Returns
    -------
    DataFrame with single-marker features per instance.
    """
    feature_rows = []

    if marker_data is None or marker_data.empty:
        for _ in windows:
            feature_rows.append({
                f"{marker_name}_mean": np.nan,
                f"{marker_name}_slope": np.nan,
                f"{marker_name}_last": np.nan,
                f"{marker_name}_max": np.nan,
                f"{marker_name}_range": np.nan,
            })
        return pd.DataFrame(feature_rows, index=range(len(windows)))

    df = marker_data.reset_index() if hasattr(marker_data.index, 'names') else marker_data.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Find the marker value column (not metadata)
    meta_cols = {"subject_id", "visit_num", "date", "diagnosis"}
    value_cols = [c for c in df.columns if c not in meta_cols]

    for window in windows:
        mask = (
            (df["subject_id"] == window.subject_id)
            & (df["date"] >= window.window_start)
            & (df["date"] <= window.window_end)
        )
        samples = df.loc[mask].sort_values("date")

        row: dict[str, float] = {}
        if len(samples) == 0 or not value_cols:
            row[f"{marker_name}_mean"] = np.nan
            row[f"{marker_name}_slope"] = np.nan
            row[f"{marker_name}_last"] = np.nan
            row[f"{marker_name}_max"] = np.nan
            row[f"{marker_name}_range"] = np.nan
        else:
            # Use first value column or mean across all
            vals = samples[value_cols].mean(axis=1).tolist()
            row[f"{marker_name}_mean"] = float(np.nanmean(vals))
            row[f"{marker_name}_slope"] = _slope(vals)
            row[f"{marker_name}_last"] = float(vals[-1])
            row[f"{marker_name}_max"] = float(np.nanmax(vals))
            row[f"{marker_name}_range"] = float(np.nanmax(vals) - np.nanmin(vals))

        feature_rows.append(row)

    result = pd.DataFrame(feature_rows, index=range(len(windows)))
    result.index.name = "instance_id"
    return result


def run_baseline(
    marker_name: str,
    marker_features: pd.DataFrame,
    windows: list[Window],
) -> dict:
    """Run a single-marker baseline with LOPO-CV.

    Returns
    -------
    Dict with model name and evaluation metrics.
    """
    # Skip if all features are NaN
    if marker_features.isna().all().all():
        logger.warning("Baseline %s: all features are NaN, skipping", marker_name)
        return {"model": f"baseline_{marker_name}", "auc": np.nan}

    results = lopo_cv(marker_features, windows, model_type="logistic", calibrate=False)
    metrics = evaluate_results(results)
    metrics["model"] = f"baseline_{marker_name}"
    return metrics


def run_all_baselines(
    data: dict[str, pd.DataFrame | None],
    windows: list[Window],
    diversity_features: pd.DataFrame | None = None,
) -> list[dict]:
    """Run all single-marker baselines.

    Parameters
    ----------
    data:
        Dict of omic DataFrames from data loader.
    windows:
        Classification windows.
    diversity_features:
        Pre-computed Shannon diversity features (from microbiome module).
        If None, extracts from species data.

    Returns
    -------
    List of metrics dicts, one per baseline.
    """
    results = []

    # Calprotectin baseline
    calprotectin = data.get("calprotectin")
    cal_features = extract_single_marker_features("calprotectin", calprotectin, windows)
    results.append(run_baseline("calprotectin", cal_features, windows))

    # CRP baseline
    crp = data.get("crp")
    crp_features = extract_single_marker_features("crp", crp, windows)
    results.append(run_baseline("crp", crp_features, windows))

    # Shannon diversity baseline
    if diversity_features is not None:
        div_cols = [c for c in diversity_features.columns if "shannon" in c.lower()]
        if div_cols:
            div_df = diversity_features[div_cols]
        else:
            div_df = diversity_features.iloc[:, :5]
        results.append(run_baseline("diversity", div_df, windows))
    else:
        # Extract from species if available
        species = data.get("species")
        div_features = extract_single_marker_features("diversity", species, windows)
        results.append(run_baseline("diversity", div_features, windows))

    logger.info("Ran %d baselines", len(results))
    return results


def compare_models(
    primary_metrics: dict,
    baseline_metrics: list[dict],
) -> pd.DataFrame:
    """Build comparison table of primary model vs baselines.

    Parameters
    ----------
    primary_metrics:
        Metrics dict from the primary XGBoost model.
    baseline_metrics:
        List of metrics dicts from baselines.

    Returns
    -------
    DataFrame with one row per model and metric columns.
    """
    all_metrics = [primary_metrics] + baseline_metrics

    rows = []
    for m in all_metrics:
        rows.append({
            "model": m.get("model", "unknown"),
            "auc": m.get("auc", np.nan),
            "sensitivity": m.get("sensitivity", np.nan),
            "specificity": m.get("specificity", np.nan),
            "ppv": m.get("ppv", np.nan),
            "npv": m.get("npv", np.nan),
        })

    comparison = pd.DataFrame(rows)

    # Compute AUC difference vs calprotectin baseline
    cal_auc = None
    for m in baseline_metrics:
        if "calprotectin" in m.get("model", ""):
            cal_auc = m.get("auc")
            break

    if cal_auc is not None and np.isfinite(cal_auc):
        comparison["auc_delta_vs_calprotectin"] = comparison["auc"] - cal_auc
    else:
        comparison["auc_delta_vs_calprotectin"] = np.nan

    return comparison


def save_baseline_results(
    comparison: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save baseline comparison table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    logger.info("Saved baseline comparison to %s", output_dir)
