"""Apply trained elastic-net models to predict TCGA patient dependencies.

Loads trained models and TCGA expression data, aligns features,
and predicts per-patient dependency scores for all predictable genes.

Usage:
    from bioagentics.models.tcga_prediction import predict_tcga_dependencies
    dep_matrix = predict_tcga_dependencies(models, feature_genes, tcga_expr)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def predict_tcga_dependencies(
    models: dict,
    feature_genes: list[str],
    tcga_expression: pd.DataFrame,
    max_missing_frac: float = 0.20,
) -> pd.DataFrame:
    """Predict dependency scores for TCGA patients using trained models.

    Parameters
    ----------
    models : dict
        Gene name -> trained sklearn model (from dependency_model.train_all_models).
    feature_genes : list[str]
        Ordered list of feature gene names used during training.
    tcga_expression : DataFrame
        TCGA expression matrix (patients x genes) with HUGO symbol columns.
    max_missing_frac : float
        Maximum fraction of missing features before raising an error.

    Returns
    -------
    DataFrame (patients x predictable_genes) of predicted dependency scores.
    """
    # Align TCGA features to training feature set
    available = set(tcga_expression.columns)
    missing = [g for g in feature_genes if g not in available]
    missing_frac = len(missing) / len(feature_genes)

    if missing_frac > max_missing_frac:
        raise ValueError(
            f"{len(missing)}/{len(feature_genes)} ({missing_frac:.1%}) features missing "
            f"in TCGA data, exceeds {max_missing_frac:.0%} threshold"
        )

    if missing_frac > 0.05:
        logger.warning(
            "%d/%d (%.1f%%) features missing in TCGA data — filling with zeros",
            len(missing), len(feature_genes), missing_frac * 100,
        )

    # Build aligned feature matrix, fill missing with 0
    X = tcga_expression.reindex(columns=feature_genes, fill_value=0.0)

    # Predict for each gene model
    predictions = {}
    for gene, model in models.items():
        predictions[gene] = model.predict(X.values)

    dep_matrix = pd.DataFrame(predictions, index=tcga_expression.index)
    dep_matrix.index.name = "patient_id"

    logger.info(
        "Predicted dependencies for %d patients x %d genes",
        dep_matrix.shape[0], dep_matrix.shape[1],
    )

    return dep_matrix


def save_predictions(
    dep_matrix: pd.DataFrame,
    results_dir: str | Path,
    patient_meta: pd.DataFrame | None = None,
) -> Path:
    """Save predicted dependency matrix to CSV.

    Parameters
    ----------
    dep_matrix : DataFrame
        Predicted dependencies (patients x genes).
    results_dir : path
        Output directory.
    patient_meta : DataFrame, optional
        Patient metadata (subtypes, etc.) to save alongside.

    Returns
    -------
    Path to saved predictions file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "tcga_predicted_dependencies.csv"
    dep_matrix.to_csv(out_path)
    logger.info("Saved predictions to %s", out_path)

    if patient_meta is not None:
        meta_path = results_dir / "tcga_patient_metadata.csv"
        patient_meta.to_csv(meta_path, index=False)

    return out_path
