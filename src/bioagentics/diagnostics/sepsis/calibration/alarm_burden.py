"""Alarm burden analysis for sepsis early warning models.

Computes false positive rates at clinically relevant sensitivity
thresholds (80%, 90%, 95%) to assess alarm fatigue risk.
Target: FPR <2:1 (i.e., <2 false alarms per true alarm) at 80% sensitivity.
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
        # tpr_curve and fpr_curve have len(thresholds)+1 entries
        # thresholds[i] corresponds to tpr_curve[i+1], fpr_curve[i+1]
        # We want the first valid index (lowest FPR that meets sensitivity)
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
