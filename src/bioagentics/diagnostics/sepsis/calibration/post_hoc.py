"""Post-hoc calibration baselines with conformal comparison.

Applies Platt scaling and isotonic regression, computes ECE and MCE
before and after calibration, generates calibration curves, and
compares against conformal prediction approach.

Builds on the core calibration utilities in calibration.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from bioagentics.diagnostics.sepsis.calibration.calibration import (
    calibrate_cv,
    calibration_curve_data,
    compute_ece,
)
from bioagentics.diagnostics.sepsis.calibration.conformal import (
    evaluate_conformal,
)
from bioagentics.diagnostics.sepsis.config import (
    OUTER_CV_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute maximum calibration error.

    MCE = max_b |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    y_true : Binary ground truth labels.
    y_prob : Predicted probabilities.
    n_bins : Number of equal-width bins.

    Returns
    -------
    MCE value (lower is better).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    max_err = 0.0

    for i in range(n_bins):
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        max_err = max(max_err, abs(avg_accuracy - avg_confidence))

    return float(max_err)


def evaluate_post_hoc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    n_folds: int = OUTER_CV_FOLDS,
    alpha: float = 0.10,
) -> dict:
    """Evaluate post-hoc calibration methods and compare with conformal.

    Computes ECE, MCE, and calibration curves for uncalibrated, Platt,
    and isotonic methods. Also runs conformal prediction for comparison.

    Parameters
    ----------
    y_true : Binary labels.
    y_prob : Uncalibrated predicted probabilities.
    n_bins : Number of bins for ECE/MCE.
    n_folds : Number of CV folds for calibration.
    alpha : Significance level for conformal prediction comparison.

    Returns
    -------
    Dictionary with ECE, MCE, calibration curves per method,
    conformal comparison, and best method selection.
    """
    results = {}

    # Uncalibrated
    ece_uncal = compute_ece(y_true, y_prob, n_bins)
    mce_uncal = compute_mce(y_true, y_prob, n_bins)
    curve_uncal = calibration_curve_data(y_true, y_prob, n_bins)
    results["uncalibrated"] = {
        "ece": ece_uncal,
        "mce": mce_uncal,
        "calibration_curve": curve_uncal,
    }
    logger.info("Uncalibrated ECE=%.4f MCE=%.4f", ece_uncal, mce_uncal)

    # Platt scaling
    prob_platt = calibrate_cv(y_true, y_prob, method="sigmoid", n_folds=n_folds)
    ece_platt = compute_ece(y_true, prob_platt, n_bins)
    mce_platt = compute_mce(y_true, prob_platt, n_bins)
    curve_platt = calibration_curve_data(y_true, prob_platt, n_bins)
    results["platt_scaling"] = {
        "ece": ece_platt,
        "mce": mce_platt,
        "calibration_curve": curve_platt,
    }
    logger.info("Platt ECE=%.4f MCE=%.4f", ece_platt, mce_platt)

    # Isotonic regression
    prob_iso = calibrate_cv(y_true, y_prob, method="isotonic", n_folds=n_folds)
    ece_iso = compute_ece(y_true, prob_iso, n_bins)
    mce_iso = compute_mce(y_true, prob_iso, n_bins)
    curve_iso = calibration_curve_data(y_true, prob_iso, n_bins)
    results["isotonic"] = {
        "ece": ece_iso,
        "mce": mce_iso,
        "calibration_curve": curve_iso,
    }
    logger.info("Isotonic ECE=%.4f MCE=%.4f", ece_iso, mce_iso)

    # Conformal prediction comparison
    conformal_results = evaluate_conformal(
        y_true, y_prob, alpha=alpha, n_folds=n_folds
    )
    results["conformal"] = {
        "coverage": conformal_results["mean_coverage"],
        "avg_set_size": conformal_results["mean_set_size"],
        "false_alarm_reduction": conformal_results["false_alarm_reduction"],
        "alpha": alpha,
    }
    logger.info(
        "Conformal: coverage=%.3f, FP reduction=%.1f%%",
        conformal_results["mean_coverage"],
        conformal_results["false_alarm_reduction"] * 100,
    )

    # Best post-hoc method by ECE
    post_hoc_methods = {
        k: v for k, v in results.items() if k in ("uncalibrated", "platt_scaling", "isotonic")
    }
    best = min(post_hoc_methods.items(), key=lambda kv: kv[1]["ece"])
    results["best_method"] = best[0]
    results["best_ece"] = best[1]["ece"]
    results["best_mce"] = best[1]["mce"]
    results["meets_ece_target"] = best[1]["ece"] < 0.05

    return results


def run_post_hoc(
    datasets: dict[int, dict[str, np.ndarray]],
    model_probs: dict[int, dict[str, np.ndarray]] | None = None,
    alpha: float = 0.10,
    results_dir: Path = RESULTS_DIR,
) -> dict[int, dict]:
    """Run post-hoc calibration analysis on all lookahead windows.

    Parameters
    ----------
    datasets : Output of generate_datasets, keyed by lookahead.
    model_probs : Pre-computed model predictions keyed by
        {lookahead: {"y_true": ..., "y_prob": ...}}.
    alpha : Significance level for conformal comparison.
    results_dir : Where to save JSON results.

    Returns
    -------
    Dictionary keyed by lookahead with post-hoc calibration results.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

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

        logger.info(
            "=== Post-hoc calibration: %dh lookahead (%d samples) ===",
            lh, len(y_true),
        )
        metrics = evaluate_post_hoc(y_true, y_prob, alpha=alpha)
        metrics["lookahead_hours"] = lh
        metrics["n_samples"] = len(y_true)
        all_results[lh] = metrics

        out_path = results_dir / f"post_hoc_{lh}h.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved %s", out_path)

    # Summary
    summary = {
        str(lh): {
            "best_method": r["best_method"],
            "best_ece": r["best_ece"],
            "best_mce": r["best_mce"],
            "conformal_fp_reduction": r["conformal"]["false_alarm_reduction"],
            "meets_ece_target": r["meets_ece_target"],
        }
        for lh, r in all_results.items()
    }
    summary_path = results_dir / "post_hoc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return all_results
