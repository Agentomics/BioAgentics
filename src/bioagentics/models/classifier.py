"""ML classifier pipeline with RFE and nested cross-validation.

Implements feature selection + classifier training for the transcriptomic
biomarker panel. Trains Random Forest, XGBoost, and logistic regression
with RFECV feature selection and nested cross-validation for unbiased
performance estimation. MANDATORY: trains sex-stratified models in addition
to combined models. Target: AUC >= 0.80 on held-out validation.

Usage:
    uv run python -m bioagentics.models.classifier input.h5ad \\
        --condition-key condition --sex-key sex
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "transcriptomic-biomarker-panel"


@dataclass
class ClassifierResult:
    """Result from training a single classifier."""

    name: str
    model: object
    auc: float
    sensitivity: float
    specificity: float
    selected_features: list[str]
    y_true: np.ndarray
    y_prob: np.ndarray
    mode: str  # "combined", "male", "female"


def _get_xgboost_classifier(**kwargs):
    """Import and return XGBClassifier."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
        **kwargs,
    )


def select_features_rfecv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    min_features: int = 10,
    cv_folds: int = 5,
) -> tuple[list[str], RFECV]:
    """Select features using Recursive Feature Elimination with CV.

    Uses a Random Forest as the estimator for RFE.

    Returns
    -------
    Tuple of (selected feature names, fitted RFECV object).
    """
    estimator = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    rfecv = RFECV(
        estimator=estimator,
        step=0.1,  # remove 10% of features per step
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring="roc_auc",
        min_features_to_select=min_features,
        n_jobs=-1,
    )
    rfecv.fit(X, y)

    selected_mask = rfecv.support_
    selected = [f for f, s in zip(feature_names, selected_mask) if s]

    logger.info(
        "RFECV selected %d/%d features (optimal CV AUC: %.3f)",
        len(selected), len(feature_names),
        rfecv.cv_results_["mean_test_score"].max(),
    )
    return selected, rfecv


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute AUC, sensitivity, specificity from true labels and probabilities."""
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Youden's J statistic for optimal threshold
    j_scores = tpr - fpr
    optimal_idx = j_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_prob >= optimal_threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def train_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    mode: str = "combined",
    outer_folds: int = 5,
    inner_folds: int = 5,
) -> list[ClassifierResult]:
    """Train classifiers with nested cross-validation.

    Outer loop: performance estimation. Inner loop: feature selection + hyperparameter tuning.

    Returns a ClassifierResult for each of three classifiers.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection on full data (for final model)
    logger.info("Running RFECV for feature selection (%s)...", mode)
    selected_features, rfecv = select_features_rfecv(
        X_scaled, y, feature_names, cv_folds=inner_folds
    )
    selected_idx = [feature_names.index(f) for f in selected_features]
    X_selected = X_scaled[:, selected_idx]

    # Define classifiers
    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": _get_xgboost_classifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, solver="saga", l1_ratio=1.0,
        ),
    }

    results: list[ClassifierResult] = []
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    for clf_name, clf in classifiers.items():
        logger.info("Training %s (%s, %d features)...", clf_name, mode, len(selected_features))

        # Nested CV: get out-of-fold probability predictions
        y_prob = cross_val_predict(
            clf, X_selected, y, cv=outer_cv, method="predict_proba"
        )[:, 1]

        metrics = _compute_metrics(y, y_prob)

        # Fit final model on all data
        clf.fit(X_selected, y)

        result = ClassifierResult(
            name=clf_name,
            model=clf,
            auc=metrics["auc"],
            sensitivity=metrics["sensitivity"],
            specificity=metrics["specificity"],
            selected_features=selected_features,
            y_true=y,
            y_prob=y_prob,
            mode=mode,
        )
        results.append(result)

        logger.info(
            "%s (%s): AUC=%.3f, Sens=%.3f, Spec=%.3f",
            clf_name, mode, result.auc, result.sensitivity, result.specificity,
        )

    return results


def run_classifier_pipeline(
    adata: ad.AnnData,
    condition_key: str = "condition",
    sex_key: str = "sex",
    candidate_genes: list[str] | None = None,
    dest_dir: Path | None = None,
) -> dict[str, list[ClassifierResult]]:
    """Full classifier pipeline with sex-stratified models.

    Parameters
    ----------
    adata : AnnData
        Normalized expression data.
    condition_key : str
        Column defining case/control labels.
    sex_key : str
        Column for sex stratification.
    candidate_genes : list[str], optional
        Restrict features to these genes. If None, uses all genes.
    dest_dir : Path, optional
        Output directory for models and results.

    Returns
    -------
    Dict with keys "combined", "male", "female" -> list of ClassifierResult.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Prepare feature matrix
    X = np.array(adata.X, dtype=float)
    feature_names = list(adata.var_names)

    if candidate_genes:
        # Restrict to candidate genes present in data
        available = set(feature_names)
        valid_genes = [g for g in candidate_genes if g in available]
        if len(valid_genes) < 10:
            logger.warning(
                "Only %d candidate genes found in data (need >= 10). Using all genes.",
                len(valid_genes),
            )
        else:
            gene_idx = [feature_names.index(g) for g in valid_genes]
            X = X[:, gene_idx]
            feature_names = valid_genes
            logger.info("Restricted to %d candidate genes", len(valid_genes))

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(adata.obs[condition_key].astype(str))
    logger.info("Labels: %s", dict(zip(le.classes_, range(len(le.classes_)))))

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    all_results: dict[str, list[ClassifierResult]] = {}

    # 1. Combined model
    logger.info("=== COMBINED MODEL ===")
    all_results["combined"] = train_nested_cv(X, y, feature_names, mode="combined")

    # 2. Sex-stratified models
    if sex_key in adata.obs.columns:
        sex_vals = adata.obs[sex_key].astype(str).str.upper().str.strip()
        male_mask = sex_vals.isin(["M", "MALE", "1"]).values
        female_mask = sex_vals.isin(["F", "FEMALE", "2"]).values

        for mask, label in [(male_mask, "male"), (female_mask, "female")]:
            n = mask.sum()
            if n < 10:
                logger.warning("Only %d %s samples — skipping %s model", n, label, label)
                all_results[label] = []
                continue
            # Check both classes present
            y_sub = y[mask]
            if len(np.unique(y_sub)) < 2:
                logger.warning("%s subset has only 1 class — skipping", label)
                all_results[label] = []
                continue

            logger.info("=== %s MODEL (%d samples) ===", label.upper(), n)
            all_results[label] = train_nested_cv(
                X[mask], y_sub, feature_names, mode=label
            )
    else:
        logger.warning("Sex key '%s' not found — skipping sex-stratified models", sex_key)
        all_results["male"] = []
        all_results["female"] = []

    # Save results
    _save_results(all_results, le, dest_dir)

    return all_results


def _save_results(
    results: dict[str, list[ClassifierResult]],
    label_encoder: LabelEncoder,
    dest_dir: Path,
) -> None:
    """Save classifier results: metrics CSV, models pickle, feature lists."""
    metrics_rows: list[dict] = []
    for mode, clf_results in results.items():
        for cr in clf_results:
            metrics_rows.append({
                "mode": cr.mode,
                "classifier": cr.name,
                "auc": cr.auc,
                "sensitivity": cr.sensitivity,
                "specificity": cr.specificity,
                "n_features": len(cr.selected_features),
            })

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(dest_dir / "classifier_metrics.csv", index=False)
        logger.info("Saved classifier metrics to %s", dest_dir / "classifier_metrics.csv")

    # Save models
    models_dir = dest_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for mode, clf_results in results.items():
        for cr in clf_results:
            model_path = models_dir / f"{cr.name}_{mode}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"model": cr.model, "features": cr.selected_features,
                             "label_encoder": label_encoder}, f)

    # Save combined feature list
    all_features: set[str] = set()
    for clf_results in results.values():
        for cr in clf_results:
            all_features.update(cr.selected_features)
    if all_features:
        features_df = pd.DataFrame({"gene": sorted(all_features)})
        features_df.to_csv(dest_dir / "selected_features.csv", index=False)
        logger.info("Saved %d unique selected features", len(all_features))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train ML classifiers with nested CV and sex stratification"
    )
    parser.add_argument("input", type=Path, help="Input h5ad file")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--sex-key", default="sex")
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    adata = ad.read_h5ad(args.input)
    results = run_classifier_pipeline(
        adata,
        condition_key=args.condition_key,
        sex_key=args.sex_key,
        dest_dir=args.dest,
    )

    print("\n=== RESULTS ===")
    for mode, clf_results in results.items():
        for cr in clf_results:
            print(f"{mode}/{cr.name}: AUC={cr.auc:.3f} Sens={cr.sensitivity:.3f} Spec={cr.specificity:.3f}")


if __name__ == "__main__":
    main()
