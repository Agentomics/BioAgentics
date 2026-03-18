"""Visualization data generation for CD flare prediction results.

Generates plot-ready data (CSV/JSON) for:
1. Patient-level risk trajectory plots
2. Representative patient case studies (TP, FN, TN)
3. Feature trajectory heatmaps (top SHAP features over time)
4. Model comparison ROC curves
5. Calibration plots

No images are generated — all output is tabular data for reproducibility.

Usage::

    from bioagentics.crohns.flare_prediction.visualizations import (
        generate_all_visualizations,
    )

    generate_all_visualizations(cv_results, windows, shap_result, output_dir)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.crohns.flare_prediction.classifier import CVResults, evaluate_results
from bioagentics.crohns.flare_prediction.clinical_utility import SHAPResult
from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)


def patient_risk_trajectories(
    cv_results: CVResults,
    windows: list[Window],
) -> pd.DataFrame:
    """Generate patient-level risk trajectory data.

    For each patient, outputs predicted flare probability at each timepoint
    alongside actual labels.

    Returns
    -------
    DataFrame: patient_id, timepoint_idx, window_start, window_end,
    y_true, y_prob, label.
    """
    rows = []
    for fold in cv_results.folds:
        # Find windows for this patient
        patient_windows = [w for w in windows if w.subject_id == fold.patient_id]
        for i, (yt, yp) in enumerate(zip(fold.y_true, fold.y_prob)):
            w = patient_windows[i] if i < len(patient_windows) else None
            rows.append(
                {
                    "patient_id": fold.patient_id,
                    "timepoint_idx": i,
                    "window_start": str(w.window_start) if w else "",
                    "window_end": str(w.window_end) if w else "",
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "label": "pre_flare" if yt == 1 else "stable",
                }
            )

    return pd.DataFrame(rows)


def select_representative_patients(
    cv_results: CVResults,
    windows: list[Window],
    threshold: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Select representative patients for case study visualization.

    Categories:
    - true_positive: model correctly predicted flare (highest mean prob for pre-flare)
    - false_negative: model missed a flare (lowest mean prob for pre-flare)
    - true_negative: model correctly identified stable period

    Returns
    -------
    Dict mapping category to patient trajectory DataFrame.
    """
    trajectories = patient_risk_trajectories(cv_results, windows)
    representatives: dict[str, pd.DataFrame] = {}

    # Score each patient fold
    patient_scores = []
    for fold in cv_results.folds:
        pre_flare_mask = fold.y_true == 1
        stable_mask = fold.y_true == 0

        mean_prob_preflare = (
            float(fold.y_prob[pre_flare_mask].mean()) if pre_flare_mask.any() else None
        )
        mean_prob_stable = (
            float(fold.y_prob[stable_mask].mean()) if stable_mask.any() else None
        )

        tp_score = mean_prob_preflare if mean_prob_preflare is not None else -1
        fn_score = (
            1.0 - mean_prob_preflare if mean_prob_preflare is not None else -1
        )
        tn_score = (
            1.0 - mean_prob_stable if mean_prob_stable is not None else -1
        )

        patient_scores.append(
            {
                "patient_id": fold.patient_id,
                "tp_score": tp_score,
                "fn_score": fn_score,
                "tn_score": tn_score,
                "has_pre_flare": pre_flare_mask.any(),
                "has_stable": stable_mask.any(),
            }
        )

    scores_df = pd.DataFrame(patient_scores)

    # Best true positive
    tp_candidates = scores_df[scores_df["has_pre_flare"]]
    if not tp_candidates.empty:
        best_tp = tp_candidates.loc[tp_candidates["tp_score"].idxmax(), "patient_id"]
        representatives["true_positive"] = trajectories[
            trajectories["patient_id"] == best_tp
        ]

    # Best false negative (pre-flare patient with lowest prediction)
    fn_candidates = scores_df[scores_df["has_pre_flare"]]
    if not fn_candidates.empty:
        best_fn = fn_candidates.loc[fn_candidates["fn_score"].idxmax(), "patient_id"]
        representatives["false_negative"] = trajectories[
            trajectories["patient_id"] == best_fn
        ]

    # Best true negative
    tn_candidates = scores_df[scores_df["has_stable"]]
    if not tn_candidates.empty:
        best_tn = tn_candidates.loc[tn_candidates["tn_score"].idxmax(), "patient_id"]
        representatives["true_negative"] = trajectories[
            trajectories["patient_id"] == best_tn
        ]

    return representatives


def feature_trajectory_heatmap_data(
    shap_result: SHAPResult,
    features: pd.DataFrame,
    windows: list[Window],
    top_n: int = 10,
) -> pd.DataFrame:
    """Generate data for feature trajectory heatmaps.

    Shows top SHAP features changing over time per patient,
    highlighting pre-flare shifts.

    Parameters
    ----------
    shap_result:
        SHAP analysis results.
    features:
        Feature matrix.
    windows:
        Classification windows.
    top_n:
        Number of top features to include.

    Returns
    -------
    Long-form DataFrame: patient_id, timepoint, feature, value, shap_value, label.
    """
    ranking = shap_result.feature_ranking
    top_features = ranking.head(top_n)["feature"].tolist()

    # Filter to features that exist in the feature matrix
    available = [f for f in top_features if f in features.columns]
    if not available:
        return pd.DataFrame()

    rows = []
    for idx, w in enumerate(windows):
        for feat in available:
            feat_idx = shap_result.feature_names.index(feat) if feat in shap_result.feature_names else None
            shap_val = (
                float(shap_result.shap_values[idx, feat_idx])
                if feat_idx is not None and idx < shap_result.shap_values.shape[0]
                else np.nan
            )
            rows.append(
                {
                    "patient_id": w.subject_id,
                    "timepoint_idx": idx,
                    "window_start": str(w.window_start),
                    "feature": feat,
                    "value": float(features.iloc[idx][feat]) if feat in features.columns else np.nan,
                    "shap_value": shap_val,
                    "label": w.label,
                }
            )

    return pd.DataFrame(rows)


def model_comparison_roc_data(
    model_results: dict[str, CVResults],
) -> pd.DataFrame:
    """Generate ROC curve data for model comparison.

    Parameters
    ----------
    model_results:
        Dict mapping model name to CVResults.

    Returns
    -------
    Long-form DataFrame: model, fpr, tpr.
    """
    rows = []
    for model_name, results in model_results.items():
        metrics = evaluate_results(results)
        fpr = metrics.get("roc_fpr", [])
        tpr = metrics.get("roc_tpr", [])
        auc = metrics.get("auc", np.nan)

        for f, t in zip(fpr, tpr):
            rows.append(
                {
                    "model": model_name,
                    "fpr": float(f),
                    "tpr": float(t),
                    "auc": float(auc) if np.isfinite(auc) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def calibration_plot_data(
    model_results: dict[str, CVResults],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Generate calibration plot data for models.

    Parameters
    ----------
    model_results:
        Dict mapping model name to CVResults.
    n_bins:
        Number of calibration bins.

    Returns
    -------
    DataFrame: model, predicted_mean, observed_frac, count.
    """
    rows = []
    for model_name, results in model_results.items():
        metrics = evaluate_results(results)
        cal = metrics.get("calibration", {})

        pred_means = cal.get("predicted_means", [])
        obs_fracs = cal.get("observed_fracs", [])
        counts = cal.get("counts", [])

        for pm, of, c in zip(pred_means, obs_fracs, counts):
            rows.append(
                {
                    "model": model_name,
                    "predicted_mean": float(pm),
                    "observed_frac": float(of),
                    "count": int(c),
                }
            )

    return pd.DataFrame(rows)


def generate_all_visualizations(
    cv_results: CVResults,
    windows: list[Window],
    shap_result: SHAPResult | None = None,
    features: pd.DataFrame | None = None,
    additional_models: dict[str, CVResults] | None = None,
    output_dir: str | Path = "output/crohns/cd-flare-longitudinal-prediction",
) -> dict[str, pd.DataFrame]:
    """Generate all visualization data and save to output directory.

    Parameters
    ----------
    cv_results:
        Primary model LOPO-CV results.
    windows:
        Classification windows.
    shap_result:
        Optional SHAP analysis results for feature heatmaps.
    features:
        Optional feature matrix for heatmap values.
    additional_models:
        Optional dict of additional model results for comparison.
    output_dir:
        Output directory path.

    Returns
    -------
    Dict of generated DataFrames keyed by name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, pd.DataFrame] = {}

    # 1. Patient risk trajectories
    trajectories = patient_risk_trajectories(cv_results, windows)
    trajectories.to_csv(output_dir / "viz_patient_risk_trajectories.csv", index=False)
    generated["patient_risk_trajectories"] = trajectories

    # 2. Representative patients
    representatives = select_representative_patients(cv_results, windows)
    for category, df in representatives.items():
        df.to_csv(output_dir / f"viz_representative_{category}.csv", index=False)
        generated[f"representative_{category}"] = df

    # 3. Feature trajectory heatmap
    if shap_result is not None and features is not None:
        heatmap = feature_trajectory_heatmap_data(shap_result, features, windows)
        if not heatmap.empty:
            heatmap.to_csv(output_dir / "viz_feature_heatmap.csv", index=False)
            generated["feature_heatmap"] = heatmap

    # 4. Model comparison ROC
    all_models: dict[str, CVResults] = {cv_results.model_name: cv_results}
    if additional_models:
        all_models.update(additional_models)

    roc_data = model_comparison_roc_data(all_models)
    if not roc_data.empty:
        roc_data.to_csv(output_dir / "viz_roc_comparison.csv", index=False)
        generated["roc_comparison"] = roc_data

    # 5. Calibration
    cal_data = calibration_plot_data(all_models)
    if not cal_data.empty:
        cal_data.to_csv(output_dir / "viz_calibration.csv", index=False)
        generated["calibration"] = cal_data

    logger.info(
        "Generated %d visualization datasets to %s", len(generated), output_dir
    )
    return generated
