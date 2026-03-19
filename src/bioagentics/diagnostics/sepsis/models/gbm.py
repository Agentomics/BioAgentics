"""Gradient boosting models (XGBoost + LightGBM) for sepsis early warning.

Nested cross-validation (5-fold outer, 3-fold inner) with hyperparameter
tuning over n_estimators, max_depth, and learning_rate. Reports AUROC,
AUPRC at each prediction lookahead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bioagentics.diagnostics.sepsis.config import (
    INNER_CV_FOLDS,
    OUTER_CV_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# Hyperparameter grids (kept small for 8GB RAM constraint)
PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.01},
]


def _make_xgb_model(params: dict) -> object:
    """Create an XGBoost classifier with given params."""
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


def _make_lgbm_model(params: dict) -> object:
    """Create a LightGBM classifier with given params."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )


def _impute(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute NaNs (fit on train, transform both)."""
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    return X_train_imp, X_test_imp


def _inner_cv_select(
    X: np.ndarray,
    y: np.ndarray,
    model_fn,
    n_inner_folds: int = INNER_CV_FOLDS,
) -> dict:
    """Inner CV to select best hyperparameters."""
    inner_cv = StratifiedKFold(
        n_splits=n_inner_folds, shuffle=True, random_state=RANDOM_STATE
    )
    best_params = PARAM_GRID[0]
    best_auc = -1.0

    for params in PARAM_GRID:
        aucs = []
        for train_idx, val_idx in inner_cv.split(X, y):
            X_tr, X_val = _impute(X[train_idx], X[val_idx])
            model = model_fn(params)
            model.fit(X_tr, y[train_idx])
            y_prob = model.predict_proba(X_val)[:, 1]
            try:
                aucs.append(roc_auc_score(y[val_idx], y_prob))
            except ValueError:
                aucs.append(0.5)
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    logger.info("Inner CV best params: %s (AUROC=%.4f)", best_params, best_auc)
    return best_params


def evaluate_gbm_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "xgboost",
    feature_names: list[str] | None = None,
    n_outer_folds: int = OUTER_CV_FOLDS,
    n_inner_folds: int = INNER_CV_FOLDS,
) -> dict:
    """Run nested CV for a gradient boosting model.

    Parameters
    ----------
    X : Feature matrix (n_samples, n_features).
    y : Binary labels.
    model_name : "xgboost" or "lightgbm".
    feature_names : Optional feature names for importance.
    n_outer_folds : Number of outer CV folds.
    n_inner_folds : Number of inner CV folds.

    Returns
    -------
    Dictionary with auroc/auprc means, fold results, feature importance.
    """
    model_fn = _make_xgb_model if model_name == "xgboost" else _make_lgbm_model
    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=RANDOM_STATE
    )

    fold_results = []
    all_importances = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for param selection
        best_params = _inner_cv_select(X_train, y_train, model_fn, n_inner_folds)

        # Retrain on full outer-train
        X_train_imp, X_test_imp = _impute(X_train, X_test)
        model = model_fn(best_params)
        model.fit(X_train_imp, y_train)

        y_prob = model.predict_proba(X_test_imp)[:, 1]
        try:
            auroc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auroc = 0.5
        try:
            auprc = float(average_precision_score(y_test, y_prob))
        except ValueError:
            auprc = 0.0

        fold_results.append(
            {
                "fold": fold_i,
                "auroc": auroc,
                "auprc": auprc,
                "best_params": best_params,
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
        )

        # Collect feature importance
        if hasattr(model, "feature_importances_"):
            all_importances.append(model.feature_importances_)

        logger.info(
            "Fold %d: AUROC=%.4f AUPRC=%.4f params=%s",
            fold_i, auroc, auprc, best_params,
        )

    aurocs = [f["auroc"] for f in fold_results]
    auprcs = [f["auprc"] for f in fold_results]

    result = {
        "model": model_name,
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "fold_results": fold_results,
    }

    # Average feature importance across folds
    if all_importances and feature_names is not None:
        mean_imp = np.mean(all_importances, axis=0)
        importance = sorted(
            zip(feature_names, mean_imp.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        result["feature_importance"] = [
            {"feature": f, "importance": v} for f, v in importance[:30]
        ]

    return result


def run_gbm_models(
    datasets: dict[int, dict[str, np.ndarray]],
    results_dir: Path = RESULTS_DIR,
) -> dict[str, dict[int, dict]]:
    """Run XGBoost and LightGBM on all lookahead windows.

    Returns
    -------
    {"xgboost": {lh: metrics}, "lightgbm": {lh: metrics}}
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[int, dict]] = {}

    for model_name in ("xgboost", "lightgbm"):
        model_results: dict[int, dict] = {}
        for lh in sorted(datasets.keys()):
            data = datasets[lh]
            X = np.vstack([data["X_train"], data["X_test"]])
            y = np.concatenate([data["y_train"], data["y_test"]])
            feature_names = (
                list(data["feature_names"]) if "feature_names" in data else None
            )

            logger.info("=== %s: %dh lookahead (%d samples) ===", model_name, lh, len(y))
            metrics = evaluate_gbm_nested_cv(
                X, y, model_name=model_name, feature_names=feature_names,
            )
            metrics["lookahead_hours"] = lh
            metrics["n_samples"] = len(y)
            metrics["n_positive"] = int(y.sum())
            model_results[lh] = metrics

            out_path = results_dir / f"{model_name}_{lh}h.json"
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Saved %s", out_path)

        all_results[model_name] = model_results

    # Save comparison summary
    summary = {}
    for model_name, model_results in all_results.items():
        summary[model_name] = {
            str(lh): {"auroc": r["auroc_mean"], "auprc": r["auprc_mean"]}
            for lh, r in model_results.items()
        }
    summary_path = results_dir / "gbm_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return all_results
