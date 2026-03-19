"""Logistic regression baseline for sepsis early warning.

L1-regularized logistic regression with nested cross-validation
(5-fold outer, 3-fold inner for hyperparameter C). Reports AUROC,
AUPRC, sensitivity, specificity at each prediction lookahead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bioagentics.diagnostics.sepsis.config import (
    INNER_CV_FOLDS,
    OUTER_CV_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# Suppress convergence warnings during inner CV search
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def _build_pipeline(C: float) -> Pipeline:
    """Build impute -> scale -> LR pipeline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    l1_ratio=1.0,
                    solver="saga",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _select_best_C(
    X: np.ndarray,
    y: np.ndarray,
    n_inner_folds: int = INNER_CV_FOLDS,
) -> float:
    """Inner CV loop to select best regularisation strength C."""
    inner_cv = StratifiedKFold(
        n_splits=n_inner_folds, shuffle=True, random_state=RANDOM_STATE
    )
    best_C = C_GRID[0]
    best_auc = -1.0

    for C in C_GRID:
        aucs = []
        for train_idx, val_idx in inner_cv.split(X, y):
            pipe = _build_pipeline(C)
            pipe.fit(X[train_idx], y[train_idx])
            y_prob = pipe.predict_proba(X[val_idx])[:, 1]
            try:
                aucs.append(roc_auc_score(y[val_idx], y_prob))
            except ValueError:
                aucs.append(0.5)
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_C = C

    logger.info("Inner CV best C=%.4g (AUROC=%.4f)", best_C, best_auc)
    return best_C


def evaluate_lr_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    n_outer_folds: int = OUTER_CV_FOLDS,
    n_inner_folds: int = INNER_CV_FOLDS,
) -> dict:
    """Run nested CV for logistic regression and return metrics.

    Returns
    -------
    Dictionary with keys: auroc, auprc, sensitivity, specificity,
    fold_results, best_Cs, coef_importance (if feature_names provided).
    """
    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=RANDOM_STATE
    )

    fold_results = []
    all_best_Cs = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for C selection
        best_C = _select_best_C(X_train, y_train, n_inner_folds)
        all_best_Cs.append(best_C)

        # Retrain on full outer-train with best C
        pipe = _build_pipeline(best_C)
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        try:
            auroc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auroc = 0.5
        try:
            auprc = float(average_precision_score(y_test, y_prob))
        except ValueError:
            auprc = 0.0

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        fold_results.append(
            {
                "fold": fold_i,
                "auroc": auroc,
                "auprc": auprc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "best_C": best_C,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_pos_test": int(y_test.sum()),
            }
        )
        logger.info(
            "Fold %d: AUROC=%.4f AUPRC=%.4f sens=%.4f spec=%.4f C=%.4g",
            fold_i,
            auroc,
            auprc,
            sensitivity,
            specificity,
            best_C,
        )

    # Aggregate metrics
    aurocs = [f["auroc"] for f in fold_results]
    auprcs = [f["auprc"] for f in fold_results]
    sens = [f["sensitivity"] for f in fold_results]
    specs = [f["specificity"] for f in fold_results]

    result = {
        "model": "logistic_regression_l1",
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "sensitivity_mean": float(np.mean(sens)),
        "sensitivity_std": float(np.std(sens)),
        "specificity_mean": float(np.mean(specs)),
        "specificity_std": float(np.std(specs)),
        "best_Cs": all_best_Cs,
        "fold_results": fold_results,
    }

    # Feature importance from coefficients (train on full data with median C)
    if feature_names is not None:
        median_C = float(np.median(all_best_Cs))
        full_pipe = _build_pipeline(median_C)
        full_pipe.fit(X, y)
        coefs = full_pipe.named_steps["lr"].coef_[0]
        importance = sorted(
            zip(feature_names, coefs.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        result["coef_importance"] = [
            {"feature": f, "coef": c} for f, c in importance[:30]
        ]

    return result


def run_lr_baseline(
    datasets: dict[int, dict[str, np.ndarray]],
    results_dir: Path = RESULTS_DIR,
) -> dict[int, dict]:
    """Run LR baseline on all lookahead windows and save results.

    Parameters
    ----------
    datasets : Output of generate_datasets — keyed by lookahead hour.
    results_dir : Where to save JSON results.

    Returns
    -------
    Dictionary keyed by lookahead with metric dicts.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[int, dict] = {}

    for lh in sorted(datasets.keys()):
        data = datasets[lh]
        X = np.vstack([data["X_train"], data["X_test"]])
        y = np.concatenate([data["y_train"], data["y_test"]])
        feature_names = (
            list(data["feature_names"])
            if "feature_names" in data
            else None
        )

        logger.info("=== LR baseline: %dh lookahead (%d samples) ===", lh, len(y))
        metrics = evaluate_lr_nested_cv(X, y, feature_names=feature_names)
        metrics["lookahead_hours"] = lh
        metrics["n_samples"] = len(y)
        metrics["n_positive"] = int(y.sum())
        all_results[lh] = metrics

        # Save per-lookahead result
        out_path = results_dir / f"lr_baseline_{lh}h.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved %s", out_path)

    # Save summary
    summary = {
        lh: {
            "auroc": r["auroc_mean"],
            "auprc": r["auprc_mean"],
            "sensitivity": r["sensitivity_mean"],
            "specificity": r["specificity_mean"],
        }
        for lh, r in all_results.items()
    }
    summary_path = results_dir / "lr_baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    return all_results
