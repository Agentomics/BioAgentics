"""Sliding-window XGBoost classifier with Leave-One-Patient-Out CV.

Primary predictive model for CD flare prediction:
- XGBoost (primary) and logistic regression (interpretable baseline)
- LOPO-CV to avoid temporal data leakage
- Platt scaling for probability calibration
- Per-fold and aggregate metrics

Usage::

    from bioagentics.crohns.flare_prediction.classifier import (
        lopo_cv, evaluate_results,
    )

    results = lopo_cv(features, labels, patient_ids)
    metrics = evaluate_results(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class CVFold:
    """Results from a single LOPO-CV fold."""
    patient_id: str
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray
    n_instances: int


@dataclass
class CVResults:
    """Aggregate LOPO-CV results."""
    model_name: str
    folds: list[CVFold] = field(default_factory=list)

    @property
    def all_y_true(self) -> np.ndarray:
        return np.concatenate([f.y_true for f in self.folds])

    @property
    def all_y_prob(self) -> np.ndarray:
        return np.concatenate([f.y_prob for f in self.folds])

    @property
    def all_y_pred(self) -> np.ndarray:
        return np.concatenate([f.y_pred for f in self.folds])


def _impute_and_scale(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaN with training medians and standardize."""
    medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        m = medians[j] if np.isfinite(medians[j]) else 0.0
        X_train[np.isnan(X_train[:, j]), j] = m
        X_test[np.isnan(X_test[:, j]), j] = m

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def lopo_cv(
    features: pd.DataFrame,
    windows: list[Window],
    model_type: str = "xgboost",
    calibrate: bool = True,
) -> CVResults:
    """Run Leave-One-Patient-Out cross-validation.

    Parameters
    ----------
    features:
        Feature matrix (instances x features).
    windows:
        Classification windows aligned with feature rows.
    model_type:
        "xgboost" (primary) or "logistic" (interpretable baseline).
    calibrate:
        Whether to apply Platt scaling for probability calibration.

    Returns
    -------
    CVResults with per-fold and aggregate predictions.
    """
    patient_ids = np.array([w.subject_id for w in windows])
    labels = np.array([1 if w.label == "pre_flare" else 0 for w in windows])
    X = features.values.copy().astype(float)

    unique_patients = sorted(set(patient_ids))
    results = CVResults(model_name=model_type)

    for held_out in unique_patients:
        train_mask = patient_ids != held_out
        test_mask = patient_ids == held_out

        X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
        y_train, y_test = labels[train_mask], labels[test_mask]

        # Skip fold if test set has only one class or is empty
        if len(y_test) == 0 or len(np.unique(y_train)) < 2:
            continue

        X_train, X_test = _impute_and_scale(X_train, X_test)

        model = _create_model(model_type)

        if calibrate and len(np.unique(y_train)) >= 2 and len(y_train) >= 5:
            cal_model = CalibratedClassifierCV(model, cv=min(3, len(y_train)), method="sigmoid")
            try:
                cal_model.fit(X_train, y_train)
                y_prob = cal_model.predict_proba(X_test)[:, 1]
            except (ValueError, IndexError):
                model.fit(X_train, y_train)
                y_prob = _get_probabilities(model, X_test)
        else:
            model.fit(X_train, y_train)
            y_prob = _get_probabilities(model, X_test)

        y_pred = (y_prob >= 0.5).astype(int)

        results.folds.append(CVFold(
            patient_id=held_out,
            y_true=y_test,
            y_prob=y_prob,
            y_pred=y_pred,
            n_instances=len(y_test),
        ))

    logger.info(
        "LOPO-CV (%s): %d folds, %d total instances",
        model_type, len(results.folds), sum(f.n_instances for f in results.folds),
    )
    return results


def _create_model(model_type: str):
    """Create a classifier instance."""
    if model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed")
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
        )
    elif model_type == "logistic":
        return LogisticRegression(
            solver="saga", C=1.0, l1_ratio=0.5,
            max_iter=1000, random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _get_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Get predicted probabilities from a fitted model."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 else probs
    elif hasattr(model, "decision_function"):
        return expit(model.decision_function(X))
    else:
        return model.predict(X).astype(float)


def evaluate_results(results: CVResults, threshold: float = 0.5) -> dict:
    """Compute aggregate metrics from LOPO-CV results.

    Returns
    -------
    Dict with AUC, sensitivity, specificity, PPV, NPV, and per-fold AUCs.
    """
    y_true = results.all_y_true
    y_prob = results.all_y_prob
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict = {"model": results.model_name, "threshold": threshold}

    # AUC
    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["roc_fpr"] = fpr.tolist()
        metrics["roc_tpr"] = tpr.tolist()
    else:
        metrics["auc"] = np.nan

    # Confusion matrix metrics
    if len(np.unique(y_true)) >= 2 and len(np.unique(y_pred)) >= 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tn"] = int(tn)

    # Per-fold AUCs
    fold_aucs = []
    for fold in results.folds:
        if len(np.unique(fold.y_true)) >= 2:
            fold_aucs.append(float(roc_auc_score(fold.y_true, fold.y_prob)))
    metrics["fold_aucs"] = fold_aucs
    metrics["mean_fold_auc"] = float(np.mean(fold_aucs)) if fold_aucs else np.nan

    # Calibration data
    metrics["calibration"] = _calibration_data(y_true, y_prob)

    return metrics


def _calibration_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Compute calibration curve data."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_true_fracs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(float(y_prob[mask].mean()))
            bin_true_fracs.append(float(y_true[mask].mean()))
            bin_counts.append(int(mask.sum()))

    return {
        "predicted_means": bin_means,
        "observed_fracs": bin_true_fracs,
        "counts": bin_counts,
    }


def save_cv_results(
    results: CVResults,
    metrics: dict,
    output_dir: str | Path,
) -> None:
    """Save CV results and metrics to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-patient predictions
    rows = []
    for fold in results.folds:
        for i in range(len(fold.y_true)):
            rows.append({
                "patient_id": fold.patient_id,
                "y_true": int(fold.y_true[i]),
                "y_prob": float(fold.y_prob[i]),
                "y_pred": int(fold.y_pred[i]),
            })
    pd.DataFrame(rows).to_csv(
        output_dir / f"cv_predictions_{results.model_name}.csv", index=False
    )

    # Summary metrics
    summary = {k: v for k, v in metrics.items() if k not in ("roc_fpr", "roc_tpr", "calibration")}
    pd.Series(summary).to_csv(output_dir / f"cv_metrics_{results.model_name}.csv")

    logger.info("Saved CV results to %s", output_dir)
