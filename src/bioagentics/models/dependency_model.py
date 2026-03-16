"""ElasticNet dependency prediction model training and selection.

Trains per-gene elastic-net models predicting CRISPR dependency scores
from expression features. Filters to predictable genes (CV r > threshold).

Usage:
    from bioagentics.models.dependency_model import train_all_models
    results = train_all_models(X, Y, n_folds=5, min_r=0.3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Results from training elastic-net dependency models."""

    metrics: pd.DataFrame          # per-gene: cv_r, rmse, alpha, l1_ratio
    predictable_genes: list[str] = field(default_factory=list)  # genes with CV r > min_r
    models: dict = field(default_factory=dict)  # gene -> trained ElasticNetCV
    n_total: int = 0
    n_predictable: int = 0
    min_r: float = 0.3


def _train_single_gene(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    random_state: int,
) -> dict:
    """Train ElasticNetCV for a single target gene and compute CV metrics.

    Returns dict with cv_r, rmse, alpha, l1_ratio, model.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Collect out-of-fold predictions for CV correlation
    oof_pred = np.full(len(y), np.nan)
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            alphas=50,
            cv=3,  # inner CV for hyperparameter selection
            n_jobs=1,  # parallelism at outer level instead
            max_iter=5000,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        oof_pred[val_idx] = model.predict(X_val)

    # Compute CV Pearson r from out-of-fold predictions
    valid_mask = ~np.isnan(oof_pred) & ~np.isnan(y)
    if valid_mask.sum() < 5:
        return {"cv_r": 0.0, "rmse": np.inf, "alpha": np.nan, "l1_ratio": np.nan, "model": None}

    r, _ = pearsonr(y[valid_mask], oof_pred[valid_mask])
    rmse = np.sqrt(np.mean((y[valid_mask] - oof_pred[valid_mask]) ** 2))

    # Refit on all data for the final model
    final_model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=50,
        cv=n_folds,
        n_jobs=1,
        max_iter=5000,
        random_state=random_state,
    )
    final_model.fit(X, y)

    return {
        "cv_r": float(r),
        "rmse": float(rmse),
        "alpha": float(final_model.alpha_),
        "l1_ratio": float(final_model.l1_ratio_),
        "model": final_model,
    }


def train_all_models(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_folds: int = 5,
    min_r: float = 0.3,
    random_state: int = 42,
) -> ModelResults:
    """Train elastic-net models for all target genes.

    Parameters
    ----------
    X : DataFrame (n_samples x n_features)
        Expression feature matrix.
    Y : DataFrame (n_samples x n_targets)
        CRISPR dependency score matrix.
    n_folds : int
        Number of outer CV folds for evaluation.
    min_r : float
        Minimum CV Pearson r to consider a gene predictable.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    ModelResults with per-gene metrics, filtered gene list, and trained models.
    """
    X_arr = X.values
    target_genes = Y.columns.tolist()
    n_genes = len(target_genes)

    logger.info("Training elastic-net models for %d target genes...", n_genes)

    rows = []
    models = {}
    for i, gene in enumerate(target_genes):
        if (i + 1) % 500 == 0 or i == 0:
            logger.info("  Progress: %d / %d genes", i + 1, n_genes)

        y = Y[gene].values
        # Skip genes with no variance in dependency scores
        if np.nanstd(y) < 1e-10:
            rows.append({"gene": gene, "cv_r": 0.0, "rmse": np.inf,
                         "alpha": np.nan, "l1_ratio": np.nan})
            continue

        # Drop samples with NaN dependency
        valid = ~np.isnan(y)
        if valid.sum() < n_folds + 2:
            rows.append({"gene": gene, "cv_r": 0.0, "rmse": np.inf,
                         "alpha": np.nan, "l1_ratio": np.nan})
            continue

        result = _train_single_gene(X_arr[valid], y[valid], n_folds, random_state)
        rows.append({
            "gene": gene,
            "cv_r": result["cv_r"],
            "rmse": result["rmse"],
            "alpha": result["alpha"],
            "l1_ratio": result["l1_ratio"],
        })
        if result["model"] is not None:
            models[gene] = result["model"]

    metrics = pd.DataFrame(rows).set_index("gene")
    predictable = metrics[metrics["cv_r"] > min_r].index.tolist()

    logger.info(
        "Training complete. %d / %d genes predictable (CV r > %.2f)",
        len(predictable), n_genes, min_r,
    )

    return ModelResults(
        metrics=metrics,
        predictable_genes=predictable,
        models={g: models[g] for g in predictable if g in models},
        n_total=n_genes,
        n_predictable=len(predictable),
        min_r=min_r,
    )


def save_results(results: ModelResults, results_dir: str | Path) -> None:
    """Save model results to disk.

    Saves:
    - metrics.csv: per-gene CV metrics
    - predictable_genes.txt: filtered gene list
    - models/: directory of joblib-serialized models
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results.metrics.to_csv(results_dir / "gene_metrics.csv")

    with open(results_dir / "predictable_genes.txt", "w") as f:
        for gene in results.predictable_genes:
            f.write(gene + "\n")

    models_dir = results_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for gene, model in results.models.items():
        joblib.dump(model, models_dir / f"{gene}.joblib")

    logger.info("Saved results to %s", results_dir)


def load_models(results_dir: str | Path) -> dict:
    """Load trained models from disk."""
    models_dir = Path(results_dir) / "models"
    models = {}
    for path in sorted(models_dir.glob("*.joblib")):
        gene = path.stem
        models[gene] = joblib.load(path)
    return models
