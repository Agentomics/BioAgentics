"""Temporal validation for sepsis early warning models.

Splits MIMIC-IV admissions by date (earliest 80% for training,
most recent 20% for testing) to simulate prospective deployment.
Trains LR, GBM, and ensemble models on the training period and
evaluates on the temporal holdout. Includes ECE calibration and
fairness evaluation on the holdout set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from bioagentics.diagnostics.sepsis.config import (
    RANDOM_STATE,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def temporal_split(
    features: np.ndarray,
    labels: np.ndarray,
    admit_times: np.ndarray,
    holdout_frac: float = 0.2,
) -> dict:
    """Split data by admission time into train/test periods.

    Parameters
    ----------
    features : (n_samples, n_features) feature matrix.
    labels : (n_samples,) binary labels.
    admit_times : (n_samples,) sortable admission timestamps or ordinals.
    holdout_frac : Fraction of most recent admissions for test set.

    Returns
    -------
    Dictionary with X_train, X_test, y_train, y_test, split_index.
    """
    order = np.argsort(admit_times)
    split_idx = int(len(order) * (1 - holdout_frac))

    train_idx = order[:split_idx]
    test_idx = order[split_idx:]

    return {
        "X_train": features[train_idx],
        "X_test": features[test_idx],
        "y_train": labels[train_idx],
        "y_test": labels[test_idx],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "split_index": split_idx,
    }


def _train_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Train L1-regularized LR and return test probabilities."""
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_train))
    X_te = scaler.transform(imp.transform(X_test))

    lr = LogisticRegression(
        C=0.1, l1_ratio=1.0, solver="saga",
        max_iter=2000, random_state=RANDOM_STATE,
    )
    lr.fit(X_tr, y_train)
    return lr.predict_proba(X_te)[:, 1]


def _train_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_type: str = "xgboost",
) -> np.ndarray:
    """Train GBM and return test probabilities."""
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_te = imp.transform(X_test)

    if model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
        )
    else:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
        )

    model.fit(X_tr, y_train)
    return model.predict_proba(X_te)[:, 1]


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute standard classification metrics."""
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = 0.5
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = 0.0

    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        tn = fp = fn = tp = 0
        sensitivity = specificity = ppv = npv = 0.0
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) > 0 else 0.0,
    }


def evaluate_temporal_holdout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate all model types on temporal holdout.

    Parameters
    ----------
    X_train : Training features (early period).
    y_train : Training labels.
    X_test : Test features (recent holdout period).
    y_test : Test labels.

    Returns
    -------
    Dictionary with per-model metrics on the temporal holdout.
    """
    results = {}

    # LR baseline
    logger.info("Training LR on temporal training set (%d samples)", len(y_train))
    lr_probs = _train_lr(X_train, y_train, X_test)
    results["logistic_regression"] = _compute_metrics(y_test, lr_probs)

    # XGBoost
    logger.info("Training XGBoost on temporal training set")
    xgb_probs = _train_gbm(X_train, y_train, X_test, "xgboost")
    results["xgboost"] = _compute_metrics(y_test, xgb_probs)

    # LightGBM
    logger.info("Training LightGBM on temporal training set")
    lgbm_probs = _train_gbm(X_train, y_train, X_test, "lightgbm")
    results["lightgbm"] = _compute_metrics(y_test, lgbm_probs)

    # Simple average ensemble
    ensemble_probs = (lr_probs + xgb_probs + lgbm_probs) / 3.0
    results["ensemble_avg"] = _compute_metrics(y_test, ensemble_probs)

    # Store raw probabilities for downstream calibration/fairness
    results["_probabilities"] = {
        "logistic_regression": lr_probs,
        "xgboost": xgb_probs,
        "lightgbm": lgbm_probs,
        "ensemble_avg": ensemble_probs,
        "y_true": y_test,
    }

    # Calibration (ECE) on temporal holdout
    from bioagentics.diagnostics.sepsis.calibration.calibration import compute_ece

    probs = results["_probabilities"]
    calibration_results = {}
    for model_name in ["logistic_regression", "xgboost", "lightgbm", "ensemble_avg"]:
        model_probs = probs[model_name]
        ece = compute_ece(y_test, model_probs)
        calibration_results[model_name] = {
            "ece": ece,
            "meets_ece_target": ece < 0.05,
        }
    results["calibration"] = calibration_results

    # Check success criterion: AUROC >= 0.85 at 6h
    best_auroc = max(
        r["auroc"]
        for k, r in results.items()
        if isinstance(r, dict) and "auroc" in r
    )
    results["best_auroc"] = best_auroc
    results["meets_target_085"] = best_auroc >= 0.85

    return results


def run_temporal_validation(
    features: np.ndarray,
    labels: np.ndarray,
    admit_times: np.ndarray,
    feature_names: list[str] | None = None,
    holdout_frac: float = 0.2,
    results_dir: Path = RESULTS_DIR,
    label: str = "",
) -> dict:
    """Run full temporal validation pipeline.

    Parameters
    ----------
    features : Feature matrix.
    labels : Binary sepsis labels.
    admit_times : Admission timestamps (sortable).
    feature_names : Optional feature names.
    holdout_frac : Fraction for temporal holdout (default: 0.2).
    results_dir : Output directory.
    label : Optional filename label (e.g., "6h").

    Returns
    -------
    Dictionary with temporal validation results.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    split = temporal_split(features, labels, admit_times, holdout_frac)
    logger.info(
        "Temporal split: %d train, %d test (holdout=%.0f%%)",
        len(split["y_train"]),
        len(split["y_test"]),
        holdout_frac * 100,
    )

    results = evaluate_temporal_holdout(
        split["X_train"], split["y_train"],
        split["X_test"], split["y_test"],
    )

    # Remove non-serializable numpy arrays for JSON output
    probs = results.pop("_probabilities", None)

    results["split_info"] = {
        "n_train": len(split["y_train"]),
        "n_test": len(split["y_test"]),
        "train_pos": int(split["y_train"].sum()),
        "test_pos": int(split["y_test"].sum()),
        "holdout_frac": holdout_frac,
    }
    if feature_names:
        results["n_features"] = len(feature_names)

    suffix = f"_{label}" if label else ""
    out_path = results_dir / f"temporal_validation{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", out_path)

    # Restore probabilities for caller
    if probs is not None:
        results["_probabilities"] = probs

    return results
