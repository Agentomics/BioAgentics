"""Conformal prediction framework for uncertainty-aware sepsis alerts.

Implements split conformal and Mondrian conformal prediction on top of
ensemble predictions. Produces prediction sets with guaranteed coverage
and evaluates alarm reduction vs. standard thresholding.

References:
- Computers in Biology and Medicine 2026 (conformal prediction for sepsis)
- COMPOSER-LLM (npj Digital Medicine): conformal prediction as calibration layer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from bioagentics.diagnostics.sepsis.config import (
    OUTER_CV_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def compute_nonconformity_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> np.ndarray:
    """Compute nonconformity scores for conformal prediction.

    Uses the heuristic nonconformity measure: 1 - p(y_true) where
    p(y_true) is the predicted probability of the true class.

    Parameters
    ----------
    y_true : Binary ground truth labels.
    y_prob : Predicted probability of class 1.

    Returns
    -------
    Nonconformity scores (higher = less conforming).
    """
    p_true = np.where(y_true == 1, y_prob, 1 - y_prob)
    return 1.0 - p_true


def split_conformal_calibrate(
    cal_scores: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Compute conformal quantile threshold from calibration scores.

    Parameters
    ----------
    cal_scores : Nonconformity scores from calibration set.
    alpha : Significance level (1 - desired coverage). Default 0.10
        gives 90% coverage guarantee.

    Returns
    -------
    Quantile threshold q_hat. A test point is included in the
    prediction set if its nonconformity score <= q_hat.
    """
    n = len(cal_scores)
    # Finite-sample correction: ceil((n+1)*(1-alpha)) / n
    quantile_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_scores, quantile_level))


def conformal_prediction_sets(
    y_prob: np.ndarray,
    q_hat: float,
) -> dict[str, np.ndarray]:
    """Generate prediction sets using conformal threshold.

    For binary classification, produces a prediction set per sample
    indicating which classes are included.

    Parameters
    ----------
    y_prob : Predicted probability of class 1 on test data.
    q_hat : Conformal quantile threshold.

    Returns
    -------
    Dictionary with:
        - include_0: boolean array, True if class 0 in prediction set
        - include_1: boolean array, True if class 1 in prediction set
        - set_size: int array, size of each prediction set (0, 1, or 2)
        - predicted_label: int array, deterministic prediction
          (1 if only class 1 in set, 0 if only class 0, -1 if ambiguous)
    """
    # Nonconformity for class 0: 1 - (1 - p) = p
    score_0 = y_prob
    # Nonconformity for class 1: 1 - p
    score_1 = 1.0 - y_prob

    include_0 = score_0 <= q_hat
    include_1 = score_1 <= q_hat

    set_size = include_0.astype(int) + include_1.astype(int)

    # Deterministic label: if exactly one class, use it; else ambiguous (-1)
    predicted_label = np.full(len(y_prob), -1, dtype=int)
    predicted_label[(include_1) & (~include_0)] = 1
    predicted_label[(include_0) & (~include_1)] = 0

    return {
        "include_0": include_0,
        "include_1": include_1,
        "set_size": set_size,
        "predicted_label": predicted_label,
    }


def mondrian_conformal_calibrate(
    cal_scores: np.ndarray,
    cal_groups: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Group-conditional conformal calibration (Mondrian conformal).

    Computes separate quantile thresholds per group for fair coverage
    across demographic subgroups.

    Parameters
    ----------
    cal_scores : Nonconformity scores from calibration set.
    cal_groups : Group labels for each calibration sample.
    alpha : Significance level.

    Returns
    -------
    Dictionary mapping group label -> quantile threshold.
    """
    thresholds = {}
    for group in np.unique(cal_groups):
        mask = cal_groups == group
        group_scores = cal_scores[mask]
        if len(group_scores) < 2:
            # Not enough data — use global threshold
            thresholds[str(group)] = split_conformal_calibrate(cal_scores, alpha)
        else:
            thresholds[str(group)] = split_conformal_calibrate(group_scores, alpha)
    return thresholds


def mondrian_prediction_sets(
    y_prob: np.ndarray,
    groups: np.ndarray,
    group_thresholds: dict[str, float],
    fallback_q: float,
) -> dict[str, np.ndarray]:
    """Generate prediction sets with group-specific thresholds.

    Parameters
    ----------
    y_prob : Predicted probabilities.
    groups : Group labels for each test sample.
    group_thresholds : Per-group conformal thresholds.
    fallback_q : Threshold for groups not in calibration set.

    Returns
    -------
    Same structure as conformal_prediction_sets.
    """
    include_0 = np.zeros(len(y_prob), dtype=bool)
    include_1 = np.zeros(len(y_prob), dtype=bool)

    for group in np.unique(groups):
        mask = groups == group
        q = group_thresholds.get(str(group), fallback_q)
        score_0 = y_prob[mask]
        score_1 = 1.0 - y_prob[mask]
        include_0[mask] = score_0 <= q
        include_1[mask] = score_1 <= q

    set_size = include_0.astype(int) + include_1.astype(int)
    predicted_label = np.full(len(y_prob), -1, dtype=int)
    predicted_label[(include_1) & (~include_0)] = 1
    predicted_label[(include_0) & (~include_1)] = 0

    return {
        "include_0": include_0,
        "include_1": include_1,
        "set_size": set_size,
        "predicted_label": predicted_label,
    }


def evaluate_conformal(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float = 0.10,
    n_folds: int = OUTER_CV_FOLDS,
) -> dict:
    """Cross-validated evaluation of split conformal prediction.

    Uses CV to avoid data leakage: calibrate on train fold, evaluate
    on test fold.

    Parameters
    ----------
    y_true : Binary labels.
    y_prob : Predicted probabilities.
    alpha : Significance level.
    n_folds : Number of CV folds.

    Returns
    -------
    Dictionary with coverage, average set size, alarm metrics,
    and comparison to standard thresholding.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    all_coverage = []
    all_set_sizes = []
    all_alarm_results = []

    for cal_idx, test_idx in cv.split(y_prob.reshape(-1, 1), y_true):
        cal_scores = compute_nonconformity_scores(y_true[cal_idx], y_prob[cal_idx])
        q_hat = split_conformal_calibrate(cal_scores, alpha)

        psets = conformal_prediction_sets(y_prob[test_idx], q_hat)
        y_test = y_true[test_idx]

        # Coverage: fraction where true label is in the prediction set
        covered = np.where(
            y_test == 1, psets["include_1"], psets["include_0"]
        )
        coverage = float(covered.mean())
        avg_set_size = float(psets["set_size"].mean())

        # Alarm analysis: compare conformal vs standard 0.5 threshold
        standard_alarms = y_prob[test_idx] >= 0.5
        conformal_alarms = psets["include_1"]

        std_fp = int(((standard_alarms) & (y_test == 0)).sum())
        std_tp = int(((standard_alarms) & (y_test == 1)).sum())
        conf_fp = int(((conformal_alarms) & (y_test == 0)).sum())
        conf_tp = int(((conformal_alarms) & (y_test == 1)).sum())

        alarm_results = {
            "standard_fp": std_fp,
            "standard_tp": std_tp,
            "conformal_fp": conf_fp,
            "conformal_tp": conf_tp,
            "standard_far": std_fp / max(std_tp, 1),
            "conformal_far": conf_fp / max(conf_tp, 1),
        }

        all_coverage.append(coverage)
        all_set_sizes.append(avg_set_size)
        all_alarm_results.append(alarm_results)

    # Aggregate
    mean_coverage = float(np.mean(all_coverage))
    mean_set_size = float(np.mean(all_set_sizes))

    std_fp_total = sum(r["standard_fp"] for r in all_alarm_results)
    std_tp_total = sum(r["standard_tp"] for r in all_alarm_results)
    conf_fp_total = sum(r["conformal_fp"] for r in all_alarm_results)
    conf_tp_total = sum(r["conformal_tp"] for r in all_alarm_results)

    fp_reduction = 1 - (conf_fp_total / max(std_fp_total, 1))

    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "mean_coverage": mean_coverage,
        "coverage_per_fold": all_coverage,
        "mean_set_size": mean_set_size,
        "set_sizes_per_fold": all_set_sizes,
        "standard_false_alarms": std_fp_total,
        "standard_true_alarms": std_tp_total,
        "conformal_false_alarms": conf_fp_total,
        "conformal_true_alarms": conf_tp_total,
        "false_alarm_reduction": fp_reduction,
        "meets_50pct_reduction_target": fp_reduction >= 0.50,
        "coverage_valid": mean_coverage >= (1 - alpha) - 0.02,
    }


def evaluate_mondrian(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.10,
    n_folds: int = OUTER_CV_FOLDS,
) -> dict:
    """Cross-validated evaluation of Mondrian conformal prediction.

    Parameters
    ----------
    y_true : Binary labels.
    y_prob : Predicted probabilities.
    groups : Group labels for fairness-aware conformal prediction.
    alpha : Significance level.
    n_folds : Number of CV folds.

    Returns
    -------
    Dictionary with per-group coverage and overall metrics.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    group_coverage: dict[str, list[float]] = {}
    group_set_sizes: dict[str, list[float]] = {}
    overall_coverage = []

    for cal_idx, test_idx in cv.split(y_prob.reshape(-1, 1), y_true):
        cal_scores = compute_nonconformity_scores(y_true[cal_idx], y_prob[cal_idx])
        fallback_q = split_conformal_calibrate(cal_scores, alpha)
        group_q = mondrian_conformal_calibrate(
            cal_scores, groups[cal_idx], alpha
        )

        psets = mondrian_prediction_sets(
            y_prob[test_idx], groups[test_idx], group_q, fallback_q
        )
        y_test = y_true[test_idx]

        covered = np.where(y_test == 1, psets["include_1"], psets["include_0"])
        overall_coverage.append(float(covered.mean()))

        for g in np.unique(groups[test_idx]):
            g_str = str(g)
            mask = groups[test_idx] == g
            g_covered = covered[mask]
            g_sizes = psets["set_size"][mask]

            if g_str not in group_coverage:
                group_coverage[g_str] = []
                group_set_sizes[g_str] = []
            group_coverage[g_str].append(float(g_covered.mean()))
            group_set_sizes[g_str].append(float(g_sizes.mean()))

    per_group = {}
    for g in group_coverage:
        per_group[g] = {
            "mean_coverage": float(np.mean(group_coverage[g])),
            "mean_set_size": float(np.mean(group_set_sizes[g])),
        }

    coverages = [v["mean_coverage"] for v in per_group.values()]
    max_disparity = max(coverages) - min(coverages) if coverages else 0.0

    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "overall_mean_coverage": float(np.mean(overall_coverage)),
        "per_group": per_group,
        "max_coverage_disparity": max_disparity,
        "fair_coverage": max_disparity < 0.05,
    }


def run_conformal(
    datasets: dict[int, dict[str, np.ndarray]],
    model_probs: dict[int, dict[str, np.ndarray]] | None = None,
    alpha: float = 0.10,
    results_dir: Path = RESULTS_DIR,
) -> dict[int, dict]:
    """Run conformal prediction analysis on all lookahead windows.

    Parameters
    ----------
    datasets : Output of generate_datasets, keyed by lookahead.
    model_probs : Pre-computed model predictions keyed by
        {lookahead: {"y_true": ..., "y_prob": ...}}. If None,
        generates predictions via LR for analysis.
    alpha : Significance level for conformal prediction.
    results_dir : Where to save JSON results.

    Returns
    -------
    Dictionary keyed by lookahead with conformal evaluation results.
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
            # Generate predictions via cross-validated LR
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
            "=== Conformal: %dh lookahead (%d samples) ===", lh, len(y_true)
        )

        # Split conformal evaluation
        metrics = evaluate_conformal(y_true, y_prob, alpha=alpha)
        metrics["lookahead_hours"] = lh
        metrics["n_samples"] = len(y_true)
        all_results[lh] = metrics

        out_path = results_dir / f"conformal_{lh}h.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(
            "Saved %s (coverage=%.3f, FP reduction=%.1f%%)",
            out_path,
            metrics["mean_coverage"],
            metrics["false_alarm_reduction"] * 100,
        )

    # Summary
    summary = {
        str(lh): {
            "coverage": r["mean_coverage"],
            "avg_set_size": r["mean_set_size"],
            "false_alarm_reduction": r["false_alarm_reduction"],
            "meets_target": r["meets_50pct_reduction_target"],
        }
        for lh, r in all_results.items()
    }
    summary_path = results_dir / "conformal_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return all_results
