"""Probability calibration for sepsis early warning models.

Implements Platt scaling (logistic) and isotonic regression calibration,
expected calibration error (ECE), and calibration curve computation.
Uses cross-validated calibration to avoid information leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from bioagentics.diagnostics.sepsis.config import (
    OUTER_CV_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute expected calibration error.

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    y_true : Binary ground truth labels.
    y_prob : Predicted probabilities.
    n_bins : Number of equal-width bins.

    Returns
    -------
    ECE value (lower is better, <0.05 target).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (count / n) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration curve data points.

    Returns
    -------
    Dictionary with fraction_of_positives, mean_predicted_value,
    bin_counts for plotting.
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return {
        "fraction_of_positives": fraction_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
        "n_bins_used": len(fraction_pos),
    }


def calibrate_cv(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "sigmoid",
    n_folds: int = OUTER_CV_FOLDS,
) -> np.ndarray:
    """Cross-validated probability calibration.

    Fits calibration on (n_folds - 1) folds, transforms on held-out fold.
    Returns calibrated probabilities for all samples (no leakage).

    Parameters
    ----------
    y_true : Binary labels.
    y_prob : Uncalibrated predicted probabilities.
    method : "sigmoid" (Platt scaling) or "isotonic".
    n_folds : Number of CV folds.

    Returns
    -------
    Calibrated probability array (same length as input).
    """
    calibrated = np.zeros_like(y_prob)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, test_idx in cv.split(y_prob.reshape(-1, 1), y_true):
        # Fit a simple LR on train probs -> calibrated probs
        if method == "sigmoid":
            cal = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
            cal.fit(y_prob[train_idx].reshape(-1, 1), y_true[train_idx])
            calibrated[test_idx] = cal.predict_proba(
                y_prob[test_idx].reshape(-1, 1)
            )[:, 1]
        elif method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(y_prob[train_idx], y_true[train_idx])
            calibrated[test_idx] = cal.predict(y_prob[test_idx])
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    return calibrated


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    n_folds: int = OUTER_CV_FOLDS,
) -> dict:
    """Evaluate calibration before/after Platt and isotonic methods.

    Returns
    -------
    Dictionary with ECE and calibration curve data for uncalibrated,
    Platt-scaled, and isotonic-calibrated predictions.
    """
    results = {}

    # Uncalibrated
    ece_uncal = compute_ece(y_true, y_prob, n_bins)
    curve_uncal = calibration_curve_data(y_true, y_prob, n_bins)
    results["uncalibrated"] = {
        "ece": ece_uncal,
        "calibration_curve": curve_uncal,
    }
    logger.info("Uncalibrated ECE: %.4f", ece_uncal)

    # Platt scaling (sigmoid)
    prob_platt = calibrate_cv(y_true, y_prob, method="sigmoid", n_folds=n_folds)
    ece_platt = compute_ece(y_true, prob_platt, n_bins)
    curve_platt = calibration_curve_data(y_true, prob_platt, n_bins)
    results["platt_scaling"] = {
        "ece": ece_platt,
        "calibration_curve": curve_platt,
    }
    logger.info("Platt scaling ECE: %.4f", ece_platt)

    # Isotonic regression
    prob_iso = calibrate_cv(y_true, y_prob, method="isotonic", n_folds=n_folds)
    ece_iso = compute_ece(y_true, prob_iso, n_bins)
    curve_iso = calibration_curve_data(y_true, prob_iso, n_bins)
    results["isotonic"] = {
        "ece": ece_iso,
        "calibration_curve": curve_iso,
    }
    logger.info("Isotonic ECE: %.4f", ece_iso)

    # Best method
    best = min(results.items(), key=lambda kv: kv[1]["ece"])
    results["best_method"] = best[0]
    results["best_ece"] = best[1]["ece"]
    results["meets_target"] = best[1]["ece"] < 0.05

    return results


def run_calibration(
    datasets: dict[int, dict[str, np.ndarray]],
    model_probs: dict[int, dict[str, np.ndarray]] | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict[int, dict]:
    """Run calibration analysis on all lookahead windows.

    Parameters
    ----------
    datasets : Output of generate_datasets, keyed by lookahead.
    model_probs : Pre-computed model predictions keyed by
        {lookahead: {"y_true": ..., "y_prob": ...}}. If None, uses
        a simple LR to generate predictions for calibration analysis.
    results_dir : Where to save JSON results.

    Returns
    -------
    Dictionary keyed by lookahead with calibration results.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[int, dict] = {}

    for lh in sorted(datasets.keys()):
        data = datasets[lh]

        if model_probs and lh in model_probs:
            y_true = model_probs[lh]["y_true"]
            y_prob = model_probs[lh]["y_prob"]
        else:
            # Generate predictions via cross-validated LR
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler

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

        logger.info("=== Calibration: %dh lookahead (%d samples) ===", lh, len(y_true))
        metrics = evaluate_calibration(y_true, y_prob)
        metrics["lookahead_hours"] = lh
        metrics["n_samples"] = len(y_true)
        all_results[lh] = metrics

        out_path = results_dir / f"calibration_{lh}h.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved %s", out_path)

    # Summary
    summary = {
        str(lh): {
            "best_method": r["best_method"],
            "best_ece": r["best_ece"],
            "uncalibrated_ece": r["uncalibrated"]["ece"],
            "meets_target": r["meets_target"],
        }
        for lh, r in all_results.items()
    }
    summary_path = results_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return all_results
