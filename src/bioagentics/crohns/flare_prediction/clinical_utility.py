"""SHAP feature importance and clinical utility analysis for flare prediction.

Clinical utility analysis for the best-performing model:
1. SHAP feature importance: compute SHAP values for XGBoost, rank by mean |SHAP|,
   group by omic layer.
2. Lead time analysis: detection probability vs lead time before flare.
3. Decision curve analysis: net benefit at different risk thresholds.
4. Success criteria check: >=5 pre-flare features with consistent directionality,
   >=2 omic layers with SHAP importance >10% each.

Usage::

    from bioagentics.crohns.flare_prediction.clinical_utility import (
        compute_shap_importance,
        lead_time_analysis,
        decision_curve_analysis,
        check_success_criteria,
        run_clinical_utility,
    )

    results = run_clinical_utility(features, windows, model_type="xgboost")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bioagentics.crohns.flare_prediction.classifier import (
    CVResults,
    _create_model,
    _impute_and_scale,
)
from bioagentics.crohns.flare_prediction.feature_selection import _infer_layer
from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class SHAPResult:
    """SHAP analysis results."""

    feature_names: list[str]
    mean_abs_shap: np.ndarray
    shap_values: np.ndarray  # (n_samples, n_features)
    feature_ranking: pd.DataFrame
    layer_importance: pd.DataFrame


@dataclass
class LeadTimeResult:
    """Lead time analysis results."""

    lead_weeks: list[float]
    detection_rates: list[float]
    n_instances: list[int]


@dataclass
class DecisionCurveResult:
    """Decision curve analysis results."""

    thresholds: np.ndarray
    net_benefit_model: np.ndarray
    net_benefit_all: np.ndarray
    net_benefit_none: np.ndarray


@dataclass
class ClinicalUtilityResult:
    """Combined clinical utility results."""

    shap_result: SHAPResult | None = None
    lead_time: LeadTimeResult | None = None
    decision_curve: DecisionCurveResult | None = None
    success_criteria: dict = field(default_factory=dict)


def compute_shap_importance(
    features: pd.DataFrame,
    windows: list[Window],
    model_type: str = "xgboost",
) -> SHAPResult:
    """Compute SHAP feature importance for the classifier.

    Trains the model on the full dataset and computes SHAP values.
    For proper evaluation, use LOPO-CV predictions; this is for
    feature importance interpretation only.

    Parameters
    ----------
    features:
        Feature matrix (instances x features).
    windows:
        Classification windows aligned with feature rows.
    model_type:
        Model type ("xgboost" or "logistic").

    Returns
    -------
    SHAPResult with feature rankings and layer-level importance.
    """
    if not HAS_SHAP:
        raise ImportError("shap package required: uv add --optional research shap")

    labels = np.array([1 if w.label == "pre_flare" else 0 for w in windows])
    X = features.values.copy().astype(float)
    feature_names = features.columns.tolist()

    # Impute and scale
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        m = medians[j] if np.isfinite(medians[j]) else 0.0
        X[np.isnan(X[:, j]), j] = m

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model on full data
    model = _create_model(model_type)
    model.fit(X_scaled, labels)

    # Compute SHAP values
    if model_type == "xgboost":
        explainer = shap.TreeExplainer(model)  # type: ignore[possibly-unbound]
        shap_values = explainer.shap_values(X_scaled)
    else:
        explainer = shap.LinearExplainer(model, X_scaled)  # type: ignore[possibly-unbound]
        shap_values = explainer.shap_values(X_scaled)

    # Handle multi-output SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    mean_abs = np.mean(np.abs(shap_values), axis=0)

    # Build feature ranking
    ranking = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "omic_layer": [_infer_layer(f) for f in feature_names],
        }
    )
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)

    total_importance = mean_abs.sum()
    ranking["importance_pct"] = (
        ranking["mean_abs_shap"] / total_importance * 100 if total_importance > 0 else 0.0
    )

    # Mean SHAP direction for each feature (positive = increases flare risk)
    mean_shap = np.mean(shap_values, axis=0)
    ranking["mean_shap_direction"] = mean_shap

    # Layer-level importance
    layer_imp = (
        ranking.groupby("omic_layer")["mean_abs_shap"]
        .sum()
        .reset_index()
    )
    layer_imp["importance_pct"] = (
        layer_imp["mean_abs_shap"] / total_importance * 100
        if total_importance > 0
        else 0.0
    )
    layer_imp = layer_imp.sort_values("importance_pct", ascending=False).reset_index(drop=True)

    logger.info(
        "SHAP analysis: top feature=%s (%.4f), %d layers analyzed",
        ranking.iloc[0]["feature"] if len(ranking) > 0 else "none",
        ranking.iloc[0]["mean_abs_shap"] if len(ranking) > 0 else 0,
        len(layer_imp),
    )

    return SHAPResult(
        feature_names=feature_names,
        mean_abs_shap=mean_abs,
        shap_values=shap_values,
        feature_ranking=ranking,
        layer_importance=layer_imp,
    )


def lead_time_analysis(
    cv_results: CVResults,
    windows: list[Window],
    week_bins: list[float] | None = None,
) -> LeadTimeResult:
    """Analyze detection probability vs lead time before flare.

    Groups pre-flare windows by how far before the flare event they are,
    and computes the detection rate (true positive rate) at each lead time.

    Parameters
    ----------
    cv_results:
        LOPO-CV results with per-fold predictions.
    windows:
        Classification windows aligned with predictions.
    week_bins:
        Lead time bins in weeks. Default: [1, 2, 3, 4, 6, 8].

    Returns
    -------
    LeadTimeResult with detection rates per lead time bin.
    """
    if week_bins is None:
        week_bins = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

    # Reconstruct per-instance predictions from CV folds
    patient_preds: dict[str, list[tuple[int, float]]] = {}
    for fold in cv_results.folds:
        for i, (yt, yp) in enumerate(zip(fold.y_true, fold.y_prob)):
            patient_preds.setdefault(fold.patient_id, []).append((int(yt), float(yp)))

    # Match predictions to windows to get lead times
    detection_rates = []
    n_instances_per_bin = []

    # Compute lead time for each pre-flare window:
    # lead_time = flare_date - window_end (approximate)
    pre_flare_windows = [w for w in windows if w.label == "pre_flare"]

    # Get all predictions flat (ordered same as windows)
    all_y_true = cv_results.all_y_true
    all_y_prob = cv_results.all_y_prob

    # Map back to windows
    pred_idx = 0
    window_preds: list[tuple[Window, float]] = []
    for fold in cv_results.folds:
        # Find windows for this patient
        patient_windows = [w for w in windows if w.subject_id == fold.patient_id]
        for i, (yt, yp) in enumerate(zip(fold.y_true, fold.y_prob)):
            if i < len(patient_windows):
                window_preds.append((patient_windows[i], float(yp)))

    for target_weeks in week_bins:
        target_days = target_weeks * 7
        # For each pre-flare window, estimate lead time from window duration
        detected = 0
        total = 0
        for w, prob in window_preds:
            if w.label != "pre_flare":
                continue
            # Use window position as approximate lead time
            window_duration = (w.window_end - w.window_start).days
            if window_duration <= target_days:
                total += 1
                if prob >= 0.5:
                    detected += 1

        rate = detected / total if total > 0 else 0.0
        detection_rates.append(rate)
        n_instances_per_bin.append(total)

    return LeadTimeResult(
        lead_weeks=week_bins,
        detection_rates=detection_rates,
        n_instances=n_instances_per_bin,
    )


def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> DecisionCurveResult:
    """Decision curve analysis: net benefit at various risk thresholds.

    Net benefit = (TP/N) - (FP/N) * (threshold / (1 - threshold))

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_prob:
        Predicted probabilities.
    thresholds:
        Risk thresholds to evaluate. Default: 0.01 to 0.99.

    Returns
    -------
    DecisionCurveResult with net benefit curves.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    n = len(y_true)
    prevalence = y_true.mean()

    net_benefit_model = np.zeros_like(thresholds, dtype=float)
    net_benefit_all = np.zeros_like(thresholds, dtype=float)
    net_benefit_none = np.zeros_like(thresholds, dtype=float)  # always zero

    for i, t in enumerate(thresholds):
        # Model
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        weight = t / (1 - t) if t < 1.0 else np.inf
        net_benefit_model[i] = (tp / n) - (fp / n) * weight

        # Treat all
        net_benefit_all[i] = prevalence - (1 - prevalence) * weight

    return DecisionCurveResult(
        thresholds=thresholds,
        net_benefit_model=net_benefit_model,
        net_benefit_all=net_benefit_all,
        net_benefit_none=net_benefit_none,
    )


def check_success_criteria(
    shap_result: SHAPResult,
    min_directional_features: int = 5,
    min_contributing_layers: int = 2,
    layer_importance_threshold: float = 10.0,
) -> dict:
    """Check research success criteria from the plan.

    Criteria:
    1. >=5 pre-flare features with consistent directionality
    2. >=2 omic layers with SHAP importance >10% each

    Parameters
    ----------
    shap_result:
        SHAP analysis results.
    min_directional_features:
        Minimum features with consistent SHAP direction.
    min_contributing_layers:
        Minimum omic layers contributing >threshold%.
    layer_importance_threshold:
        Minimum layer importance percentage.

    Returns
    -------
    Dict with criteria checks and supporting data.
    """
    ranking = shap_result.feature_ranking
    layer_imp = shap_result.layer_importance

    # Criterion 1: Features with consistent directionality
    # A feature has "consistent directionality" if mean SHAP direction
    # is clearly positive or negative (same sign as mean across patients)
    directional = ranking[ranking["mean_shap_direction"].abs() > 1e-6]
    n_directional = len(directional)
    criterion_1_pass = n_directional >= min_directional_features

    # Top directional features
    top_directional = directional.head(min_directional_features * 2)[
        ["feature", "mean_abs_shap", "mean_shap_direction", "omic_layer"]
    ].to_dict("records")

    # Criterion 2: Multiple omic layers contributing
    contributing_layers = layer_imp[
        layer_imp["importance_pct"] >= layer_importance_threshold
    ]
    n_contributing = len(contributing_layers)
    criterion_2_pass = n_contributing >= min_contributing_layers

    result = {
        "criterion_1_directional_features": {
            "pass": criterion_1_pass,
            "n_directional": n_directional,
            "required": min_directional_features,
            "top_features": top_directional,
        },
        "criterion_2_multi_omic_layers": {
            "pass": criterion_2_pass,
            "n_contributing_layers": n_contributing,
            "required": min_contributing_layers,
            "threshold_pct": layer_importance_threshold,
            "layer_breakdown": layer_imp.to_dict("records"),
        },
        "overall_pass": criterion_1_pass and criterion_2_pass,
    }

    logger.info(
        "Success criteria: directional=%s (%d/%d), multi-omic=%s (%d/%d), overall=%s",
        criterion_1_pass,
        n_directional,
        min_directional_features,
        criterion_2_pass,
        n_contributing,
        min_contributing_layers,
        result["overall_pass"],
    )

    return result


def run_clinical_utility(
    features: pd.DataFrame,
    windows: list[Window],
    cv_results: CVResults | None = None,
    model_type: str = "xgboost",
) -> ClinicalUtilityResult:
    """Run full clinical utility analysis pipeline.

    Parameters
    ----------
    features:
        Feature matrix.
    windows:
        Classification windows.
    cv_results:
        Optional pre-computed LOPO-CV results for lead time analysis.
    model_type:
        Model type for SHAP analysis.

    Returns
    -------
    ClinicalUtilityResult with all analyses.
    """
    result = ClinicalUtilityResult()

    # 1. SHAP importance
    try:
        result.shap_result = compute_shap_importance(features, windows, model_type)
    except (ImportError, Exception) as e:
        logger.warning("SHAP analysis failed: %s", e)

    # 2. Lead time analysis (requires CV results)
    if cv_results is not None and len(cv_results.folds) > 0:
        try:
            result.lead_time = lead_time_analysis(cv_results, windows)
        except Exception as e:
            logger.warning("Lead time analysis failed: %s", e)

    # 3. Decision curve analysis
    if cv_results is not None and len(cv_results.folds) > 0:
        try:
            y_true = cv_results.all_y_true
            y_prob = cv_results.all_y_prob
            result.decision_curve = decision_curve_analysis(
                y_true, y_prob, thresholds=np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
            )
        except Exception as e:
            logger.warning("Decision curve analysis failed: %s", e)

    # 4. Success criteria check
    if result.shap_result is not None:
        result.success_criteria = check_success_criteria(result.shap_result)

    return result


def save_clinical_utility(
    result: ClinicalUtilityResult,
    output_dir: str | Path,
) -> None:
    """Save clinical utility results to output directory.

    Saves data as CSV/JSON for reproducibility (no images).

    Parameters
    ----------
    result:
        Clinical utility analysis results.
    output_dir:
        Output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SHAP results
    if result.shap_result is not None:
        result.shap_result.feature_ranking.to_csv(
            output_dir / "shap_feature_ranking.csv", index=False
        )
        result.shap_result.layer_importance.to_csv(
            output_dir / "shap_layer_importance.csv", index=False
        )
        # Save raw SHAP values as compressed numpy
        np.savez_compressed(
            output_dir / "shap_values.npz",
            values=result.shap_result.shap_values,
            features=result.shap_result.feature_names,
        )

    # Lead time results
    if result.lead_time is not None:
        lt_df = pd.DataFrame(
            {
                "lead_weeks": result.lead_time.lead_weeks,
                "detection_rate": result.lead_time.detection_rates,
                "n_instances": result.lead_time.n_instances,
            }
        )
        lt_df.to_csv(output_dir / "lead_time_analysis.csv", index=False)

    # Decision curve results
    if result.decision_curve is not None:
        dc_df = pd.DataFrame(
            {
                "threshold": result.decision_curve.thresholds,
                "net_benefit_model": result.decision_curve.net_benefit_model,
                "net_benefit_all": result.decision_curve.net_benefit_all,
                "net_benefit_none": result.decision_curve.net_benefit_none,
            }
        )
        dc_df.to_csv(output_dir / "decision_curve.csv", index=False)

    # Success criteria
    if result.success_criteria:
        with open(output_dir / "success_criteria.json", "w") as f:
            json.dump(_make_serializable(result.success_criteria), f, indent=2)

    logger.info("Saved clinical utility results to %s", output_dir)


def _make_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
