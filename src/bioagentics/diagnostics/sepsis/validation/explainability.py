"""SHAP-based explainability for sepsis early warning models.

Computes SHAP feature importance values using TreeExplainer for
GBM models, analyzes feature importance stability across CV folds,
and generates summary data for visualization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import shap
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from bioagentics.diagnostics.sepsis.config import (
    OUTER_CV_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# Limit background samples for KernelExplainer to conserve memory
MAX_BACKGROUND_SAMPLES = 100


def compute_shap_tree(
    model,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 500,
) -> dict:
    """Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model : Fitted tree-based model (XGBoost or LightGBM).
    X : Feature matrix (already imputed).
    feature_names : Optional feature names.
    max_samples : Max samples to explain (for memory safety on 8GB).

    Returns
    -------
    Dictionary with shap_values array, mean absolute SHAP per feature,
    and top feature rankings.
    """
    if len(X) > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    ranked = sorted(
        zip(feature_names, mean_abs_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "shap_values": shap_values,
        "mean_abs_shap": mean_abs_shap,
        "feature_ranking": [
            {"feature": f, "mean_abs_shap": v} for f, v in ranked
        ],
        "n_samples_explained": len(X_sample),
    }


def feature_importance_stability(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    n_folds: int = OUTER_CV_FOLDS,
    max_samples_per_fold: int = 300,
) -> dict:
    """Analyze SHAP feature importance stability across CV folds.

    Trains XGBoost on each fold and computes SHAP values. Reports
    rank correlation between folds and identifies consistently
    important features.

    Parameters
    ----------
    X : Feature matrix (may contain NaN).
    y : Binary labels.
    feature_names : Feature names.
    n_folds : Number of CV folds.
    max_samples_per_fold : Max samples per fold for SHAP.

    Returns
    -------
    Dictionary with per-fold rankings, rank correlations,
    and stable top features.
    """
    import xgboost as xgb

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    imp = SimpleImputer(strategy="median")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    fold_rankings: list[list[str]] = []
    fold_mean_shap: list[np.ndarray] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train = imp.fit_transform(X[train_idx])
        X_test = imp.transform(X[test_idx])

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
        )
        model.fit(X_train, y[train_idx])

        result = compute_shap_tree(
            model, X_test,
            feature_names=feature_names,
            max_samples=max_samples_per_fold,
        )
        fold_rankings.append(
            [r["feature"] for r in result["feature_ranking"]]
        )
        fold_mean_shap.append(result["mean_abs_shap"])
        logger.info(
            "Fold %d: top-3 features = %s",
            fold_i,
            [r["feature"] for r in result["feature_ranking"][:3]],
        )

    # Compute rank correlations between all fold pairs
    n_features = len(feature_names)
    fold_ranks = np.zeros((n_folds, n_features))
    for i, ranking in enumerate(fold_rankings):
        rank_map = {f: r for r, f in enumerate(ranking)}
        for j, feat in enumerate(feature_names):
            fold_ranks[i, j] = rank_map.get(feat, n_features)

    # Spearman rank correlation between all pairs
    from scipy.stats import spearmanr
    correlations = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            rho, _ = spearmanr(fold_ranks[i], fold_ranks[j])
            correlations.append(float(rho))

    # Stable features: top-10 in every fold
    top_k = min(10, n_features)
    top_sets = [set(r[:top_k]) for r in fold_rankings]
    stable_top = sorted(set.intersection(*top_sets)) if top_sets else []

    # Mean importance across folds
    mean_importance = np.mean(fold_mean_shap, axis=0)
    std_importance = np.std(fold_mean_shap, axis=0)
    overall_ranking = sorted(
        zip(feature_names, mean_importance.tolist(), std_importance.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "n_folds": n_folds,
        "rank_correlations": correlations,
        "mean_rank_correlation": float(np.mean(correlations)) if correlations else 0.0,
        "stable_top_features": stable_top,
        "n_stable_top": len(stable_top),
        "overall_ranking": [
            {"feature": f, "mean_abs_shap": m, "std_abs_shap": s}
            for f, m, s in overall_ranking
        ],
        "fold_top3": [r[:3] for r in fold_rankings],
    }


def run_explainability(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    results_dir: Path = RESULTS_DIR,
    label: str = "",
) -> dict:
    """Run full SHAP explainability analysis.

    Parameters
    ----------
    X : Feature matrix.
    y : Binary labels.
    feature_names : Feature names.
    results_dir : Output directory.
    label : Optional filename label (e.g., "6h").

    Returns
    -------
    Dictionary with SHAP results and stability analysis.
    """
    import xgboost as xgb

    results_dir.mkdir(parents=True, exist_ok=True)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Train a single model on full data for overall SHAP
    imp = SimpleImputer(strategy="median")
    X_clean = imp.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
    )
    model.fit(X_clean, y)

    logger.info("Computing SHAP values (%d samples, %d features)", len(X), X.shape[1])
    shap_result = compute_shap_tree(
        model, X_clean, feature_names=feature_names,
    )

    # Stability analysis
    logger.info("Running feature importance stability analysis")
    stability = feature_importance_stability(X, y, feature_names=feature_names)

    # Combine results (exclude raw shap_values from JSON)
    results = {
        "feature_ranking": shap_result["feature_ranking"],
        "n_samples_explained": shap_result["n_samples_explained"],
        "stability": {
            k: v for k, v in stability.items()
            if k != "fold_mean_shap"
        },
        "n_features": len(feature_names),
        "n_samples": len(y),
    }

    suffix = f"_{label}" if label else ""
    out_path = results_dir / f"explainability{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", out_path)

    return results
