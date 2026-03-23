"""Alarm burden analysis for sepsis early warning models.

Computes false positive rates at clinically relevant sensitivity
thresholds (80%, 90%, 95%) to assess alarm fatigue risk. Includes
conformal prediction comparison, per-patient-hour alarm rates,
and clinical setting operating characteristic tables.

Targets:
- FPR <2:1 at 80% sensitivity
- COMPOSER benchmark: false alarm rate <= 0.01/patient-hour
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from bioagentics.diagnostics.sepsis.config import RESULTS_DIR

logger = logging.getLogger(__name__)

SENSITIVITY_THRESHOLDS = [0.80, 0.90, 0.95]

# Clinical setting parameters (sensitivity targets per venue)
CLINICAL_SETTINGS = {
    "icu": {
        "sensitivity_target": 0.95,
        "description": "ICU: high sensitivity, tolerate more alarms",
    },
    "ward": {
        "sensitivity_target": 0.90,
        "description": "Ward: balanced sensitivity/specificity",
    },
    "ed": {
        "sensitivity_target": 0.80,
        "description": "ED: minimize alarm burden, higher threshold",
    },
}

# COMPOSER benchmark: 0.0087 false alarms per patient-hour
COMPOSER_FAR_TARGET = 0.01  # false alarms per patient-hour


def find_threshold_at_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float,
) -> dict:
    """Find the decision threshold that achieves a target sensitivity.

    Parameters
    ----------
    y_true : Binary labels.
    y_prob : Predicted probabilities.
    target_sensitivity : Target recall/sensitivity (e.g. 0.80).

    Returns
    -------
    Dictionary with threshold, achieved sensitivity, specificity, FPR,
    PPV, false_alarm_ratio (FP/TP), and counts.
    """
    _, tpr_curve, thresholds = roc_curve(y_true, y_prob)

    # Find the threshold closest to but >= target sensitivity
    valid = tpr_curve >= target_sensitivity
    if not valid.any():
        # Cannot reach target — use lowest threshold
        idx = len(thresholds) - 1
    else:
        # Among valid points, pick the one with highest threshold (lowest FPR)
        valid_indices = np.where(valid)[0]
        # fpr, tpr, thresholds all have the same length from roc_curve
        idx = valid_indices[0]

    # Compute metrics at this operating point
    if idx < len(thresholds):
        threshold = float(thresholds[idx])
    else:
        threshold = 0.0

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    false_alarm_ratio = fp / tp if tp > 0 else float("inf")

    return {
        "target_sensitivity": target_sensitivity,
        "threshold": threshold,
        "achieved_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "ppv": float(ppv),
        "false_alarm_ratio": float(false_alarm_ratio),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_alarms": tp + fp,
        "n_true_alarms": tp,
    }


def evaluate_alarm_burden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict:
    """Evaluate alarm burden at multiple sensitivity thresholds.

    Parameters
    ----------
    y_true : Binary labels.
    y_prob : Predicted probabilities.
    thresholds : Sensitivity thresholds to evaluate (default: [0.80, 0.90, 0.95]).

    Returns
    -------
    Dictionary with per-threshold results and overall assessment.
    """
    if thresholds is None:
        thresholds = SENSITIVITY_THRESHOLDS

    results: dict[str, object] = {}
    threshold_results = []

    for target_sens in thresholds:
        metrics = find_threshold_at_sensitivity(y_true, y_prob, target_sens)
        threshold_results.append(metrics)

        logger.info(
            "  Sensitivity=%.0f%%: threshold=%.3f, FPR=%.3f, "
            "false_alarm_ratio=%.1f:1, PPV=%.3f",
            target_sens * 100,
            metrics["threshold"],
            metrics["fpr"],
            metrics["false_alarm_ratio"],
            metrics["ppv"],
        )

    results["threshold_analysis"] = threshold_results

    # Check target: FPR <2:1 at 80% sensitivity
    sens80 = next(
        (r for r in threshold_results if r["target_sensitivity"] == 0.80), None
    )
    if sens80:
        results["target_met"] = sens80["false_alarm_ratio"] < 2.0
        results["false_alarm_ratio_at_80pct"] = sens80["false_alarm_ratio"]
    else:
        results["target_met"] = False

    results["n_samples"] = int(len(y_true))
    results["n_positive"] = int(y_true.sum())
    results["prevalence"] = float(y_true.mean())

    return results


def compute_per_patient_hour_rate(
    n_false_alarms: int,
    n_patients: int,
    hours_monitored_per_patient: float = 48.0,
) -> float:
    """Compute false alarm rate in per-patient-hour units.

    Parameters
    ----------
    n_false_alarms : Total false alarms.
    n_patients : Number of patients monitored.
    hours_monitored_per_patient : Average monitoring duration per patient.
        Default 48h (typical ICU stay length).

    Returns
    -------
    False alarms per patient-hour.
    """
    total_hours = n_patients * hours_monitored_per_patient
    return n_false_alarms / max(total_hours, 1.0)


def generate_operating_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    hours_monitored: float = 48.0,
) -> list[dict]:
    """Generate operating characteristic table for clinical settings.

    Computes alarm metrics at ICU/Ward/ED sensitivity targets.

    Parameters
    ----------
    y_true : Binary labels.
    y_prob : Predicted probabilities.
    hours_monitored : Average hours monitored per patient.

    Returns
    -------
    List of dicts, one per clinical setting, with alarm metrics.
    """
    table = []
    for setting, params in CLINICAL_SETTINGS.items():
        metrics = find_threshold_at_sensitivity(
            y_true, y_prob, params["sensitivity_target"]
        )
        far_per_ph = compute_per_patient_hour_rate(
            metrics["fp"], len(y_true), hours_monitored
        )
        table.append({
            "setting": setting,
            "description": params["description"],
            "sensitivity_target": params["sensitivity_target"],
            "threshold": metrics["threshold"],
            "achieved_sensitivity": metrics["achieved_sensitivity"],
            "specificity": metrics["specificity"],
            "ppv": metrics["ppv"],
            "false_alarm_ratio": metrics["false_alarm_ratio"],
            "false_alarms_per_patient_hour": far_per_ph,
            "meets_composer_target": far_per_ph <= COMPOSER_FAR_TARGET,
            "n_alarms": metrics["n_alarms"],
        })
    return table


def compare_with_conformal(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float = 0.10,
    n_folds: int = 5,
) -> dict:
    """Compare standard thresholding vs conformal prediction alarm burden.

    Parameters
    ----------
    y_true : Binary labels.
    y_prob : Predicted probabilities.
    alpha : Conformal significance level.
    n_folds : CV folds for conformal evaluation.

    Returns
    -------
    Dictionary with standard and conformal alarm metrics side by side.
    """
    from bioagentics.diagnostics.sepsis.calibration.conformal import (
        evaluate_conformal,
    )

    # Standard: at 80% sensitivity
    standard = find_threshold_at_sensitivity(y_true, y_prob, 0.80)
    conformal = evaluate_conformal(y_true, y_prob, alpha=alpha, n_folds=n_folds)

    return {
        "standard": {
            "false_alarms": standard["fp"],
            "true_alarms": standard["tp"],
            "false_alarm_ratio": standard["false_alarm_ratio"],
            "sensitivity": standard["achieved_sensitivity"],
        },
        "conformal": {
            "false_alarms": conformal["conformal_false_alarms"],
            "true_alarms": conformal["conformal_true_alarms"],
            "coverage": conformal["mean_coverage"],
            "avg_set_size": conformal["mean_set_size"],
        },
        "false_alarm_reduction": conformal["false_alarm_reduction"],
        "conformal_reduces_alarms": conformal["false_alarm_reduction"] > 0,
    }


def flag_subclinical_candidates(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    confidence_cutoff: float = 0.7,
) -> dict:
    """Flag high-confidence false positives as potential subclinical infections.

    Per COMPOSER finding: 62% of false positives had genuine bacterial
    infections. High-confidence FPs may represent subclinical sepsis.

    Parameters
    ----------
    y_true : Binary labels (0=no sepsis, 1=sepsis).
    y_prob : Predicted probabilities.
    threshold : Decision threshold for alarm.
    confidence_cutoff : Minimum probability to flag as subclinical candidate.

    Returns
    -------
    Dictionary with counts and indices of flagged cases.
    """
    y_pred = (y_prob >= threshold).astype(int)
    fp_mask = (y_pred == 1) & (y_true == 0)
    high_conf_fp = fp_mask & (y_prob >= confidence_cutoff)

    return {
        "total_false_positives": int(fp_mask.sum()),
        "high_confidence_fps": int(high_conf_fp.sum()),
        "fraction_high_confidence": float(
            high_conf_fp.sum() / max(fp_mask.sum(), 1)
        ),
        "subclinical_candidate_indices": np.where(high_conf_fp)[0].tolist(),
        "confidence_cutoff": confidence_cutoff,
        "note": (
            "Per COMPOSER: 62% of FPs had bacterial infections. "
            "High-confidence FPs may represent subclinical sepsis."
        ),
    }


def run_alarm_burden(
    datasets: dict[int, dict[str, np.ndarray]],
    model_probs: dict[int, dict[str, np.ndarray]] | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict[int, dict]:
    """Run alarm burden analysis on all lookahead windows.

    Parameters
    ----------
    datasets : Output of generate_datasets.
    model_probs : Pre-computed predictions {lookahead: {"y_true", "y_prob"}}.
        If None, uses cross-validated LR predictions.
    results_dir : Output directory.

    Returns
    -------
    Dictionary keyed by lookahead with alarm burden results.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from bioagentics.diagnostics.sepsis.config import OUTER_CV_FOLDS, RANDOM_STATE

    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[int, dict] = {}

    for lh in sorted(datasets.keys()):
        data = datasets[lh]

        if model_probs and lh in model_probs:
            y_true = model_probs[lh]["y_true"]
            y_prob = model_probs[lh]["y_prob"]
        else:
            X = np.vstack([data["X_train"], data["X_test"]])
            y = np.concatenate([data["y_train"], data["y_test"]])

            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_clean = scaler.fit_transform(imp.fit_transform(X))

            cv = StratifiedKFold(
                n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
            )
            y_prob = np.zeros(len(y))
            for train_idx, test_idx in cv.split(X_clean, y):
                lr = LogisticRegression(
                    C=0.1, solver="saga", max_iter=2000, random_state=RANDOM_STATE
                )
                lr.fit(X_clean[train_idx], y[train_idx])
                y_prob[test_idx] = lr.predict_proba(X_clean[test_idx])[:, 1]
            y_true = y

        logger.info("=== Alarm burden: %dh lookahead (%d samples) ===", lh, len(y_true))
        metrics = evaluate_alarm_burden(y_true, y_prob)
        metrics["lookahead_hours"] = lh
        metrics["operating_table"] = generate_operating_table(y_true, y_prob)
        metrics["conformal_comparison"] = compare_with_conformal(y_true, y_prob)
        metrics["subclinical_candidates"] = flag_subclinical_candidates(y_true, y_prob)
        all_results[lh] = metrics

        out_path = results_dir / f"alarm_burden_{lh}h.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved %s", out_path)

    # Summary
    summary = {
        str(lh): {
            "target_met": r.get("target_met", False),
            "false_alarm_ratio_at_80pct": r.get("false_alarm_ratio_at_80pct"),
        }
        for lh, r in all_results.items()
    }
    summary_path = results_dir / "alarm_burden_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return all_results
