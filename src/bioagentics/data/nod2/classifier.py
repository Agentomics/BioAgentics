"""Ensemble classifier for NOD2 variant functional impact prediction.

Trains gradient boosting + logistic regression ensemble with nested
cross-validation for 3-class (GOF/neutral/LOF) prediction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/crohns/nod2-variant-functional-impact")
DATA_DIR = Path("data/crohns/nod2-variant-functional-impact")

# Feature columns for the classifier
STRUCTURE_FEATURES = ["plddt", "rsasa", "active_site_distance"]
PREDICTOR_FEATURES = [
    "cadd_phred", "revel", "polyphen2_hdiv", "polyphen2_hvar",
    "sift", "phylop_100way", "gerp_rs", "phastcons_100way", "alphamissense",
]
VARMETER2_FEATURES = ["nsasa", "mutation_energy", "grantham_distance"]
GIRDIN_FEATURES = ["girdin_interface_distance", "disrupts_girdin_domain", "ripk2_interface_distance"]
DOMAIN_FEATURES = [
    "domain_CARD1", "domain_CARD2", "domain_NACHT",
    "domain_WH", "domain_LRR", "domain_linker",
]

ALL_FEATURES = (
    STRUCTURE_FEATURES + PREDICTOR_FEATURES +
    VARMETER2_FEATURES + GIRDIN_FEATURES + DOMAIN_FEATURES
)


def merge_features(data_dir: Path | None = None) -> pd.DataFrame:
    """Merge all feature tables with training set labels.

    Returns DataFrame with features and functional_class label.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Load training set
    training_df = pd.read_csv(data_dir / "nod2_training_set.tsv", sep="\t")
    logger.info("Training set: %d variants", len(training_df))

    # Load structure features
    struct_df = pd.read_csv(data_dir / "nod2_structure_features.tsv", sep="\t")

    # Load predictor scores
    pred_df = pd.read_csv(data_dir / "nod2_predictor_scores.tsv", sep="\t")

    # Load VarMeter2 features
    vm2_df = pd.read_csv(data_dir / "nod2_varmeter2_features.tsv", sep="\t")

    # Load girdin features
    girdin_df = pd.read_csv(data_dir / "nod2_girdin_features.tsv", sep="\t")

    # Parse residue position from hgvs_p in training set
    from bioagentics.data.nod2.varmeter2 import _parse_protein_change

    training_df["residue_pos"] = training_df["hgvs_p"].apply(
        lambda x: _parse_protein_change(str(x))[1]
        if _parse_protein_change(str(x)) is not None
        else _extract_fs_pos(str(x))
    )

    # Merge structure features by residue position
    merged = training_df.merge(
        struct_df[["residue_pos", "plddt", "rsasa", "active_site_distance", "domain"]],
        on="residue_pos",
        how="left",
    )

    # Merge predictor scores by genomic position
    merged = merged.merge(
        pred_df,
        on=["chrom", "pos", "ref", "alt"],
        how="left",
        suffixes=("", "_pred"),
    )

    # Merge VarMeter2 features by genomic coordinates (variant-specific)
    vm2_merge_keys = ["chrom", "pos", "ref", "alt"]
    vm2_cols = vm2_merge_keys + [c for c in VARMETER2_FEATURES if c in vm2_df.columns]
    vm2_cols = [c for c in vm2_cols if c in vm2_df.columns]
    merged = merged.merge(
        vm2_df[vm2_cols].drop_duplicates(subset=vm2_merge_keys),
        on=vm2_merge_keys,
        how="left",
    )

    # Merge girdin features
    girdin_cols = ["residue_pos"] + [c for c in GIRDIN_FEATURES if c in girdin_df.columns]
    merged = merged.merge(
        girdin_df[girdin_cols].drop_duplicates("residue_pos"),
        on="residue_pos",
        how="left",
    )

    # One-hot encode domain
    if "domain" in merged.columns:
        domain_dummies = pd.get_dummies(merged["domain"], prefix="domain")
        for expected_col in DOMAIN_FEATURES:
            if expected_col not in domain_dummies.columns:
                domain_dummies[expected_col] = 0
        merged = pd.concat([merged, domain_dummies[DOMAIN_FEATURES]], axis=1)

    # Convert boolean columns
    if "disrupts_girdin_domain" in merged.columns:
        merged["disrupts_girdin_domain"] = merged["disrupts_girdin_domain"].astype(float)

    logger.info("Merged feature matrix: %d variants, %d features available",
                len(merged), sum(1 for f in ALL_FEATURES if f in merged.columns))

    return merged


def _extract_fs_pos(hgvs_p: str) -> int | None:
    """Extract position from frameshift notation."""
    import re
    m = re.search(r"(\d+)fs", hgvs_p)
    if m:
        return int(m.group(1))
    return None


def train_ensemble(
    merged_df: pd.DataFrame,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
) -> dict:
    """Train ensemble classifier with nested cross-validation.

    Args:
        merged_df: Feature matrix with functional_class labels.
        n_outer_folds: Outer CV folds for evaluation.
        n_inner_folds: Inner CV folds for hyperparameter tuning.

    Returns:
        Dict with trained model, CV results, and feature importances.
    """
    # Prepare features and labels
    available_features = [f for f in ALL_FEATURES if f in merged_df.columns]
    logger.info("Using %d features: %s", len(available_features), available_features)

    X = merged_df[available_features].copy()
    y = merged_df["functional_class"].copy()

    # Handle missing values — fill with median
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_.tolist()
    logger.info("Classes: %s", class_names)

    # Compute class weights for imbalanced data
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_encoded)

    # Define models
    gb_params = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "min_samples_leaf": [2, 5],
    }

    lr_params = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0],  # equivalent to l2 penalty
    }

    # Nested CV
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    feature_importances_list = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_encoded)):
        logger.info("Training fold %d/%d...", fold_idx + 1, n_outer_folds)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        sw_train = sample_weights[train_idx]

        # Inner CV for gradient boosting hyperparameter tuning
        gb_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params,
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=1,
        )
        gb_search.fit(X_train, y_train, sample_weight=sw_train)
        best_gb = gb_search.best_estimator_

        # Inner CV for logistic regression
        lr_search = GridSearchCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            {"lr__" + k: v for k, v in lr_params.items()},
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=1,
        )
        lr_search.fit(X_train, y_train)
        best_lr = lr_search.best_estimator_

        # Build ensemble (soft voting)
        ensemble = VotingClassifier(
            estimators=[("gb", best_gb), ("lr", best_lr)],
            voting="soft",
        )
        ensemble.fit(X_train, y_train)

        # Evaluate on test fold
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

        # Per-fold metrics
        fold_result = {
            "fold": fold_idx + 1,
            "gb_best_params": gb_search.best_params_,
            "lr_best_params": {k.replace("lr__", ""): v for k, v in lr_search.best_params_.items()},
            "test_size": len(y_test),
        }
        fold_results.append(fold_result)

        # Feature importances from gradient boosting
        feature_importances_list.append(best_gb.feature_importances_)

    # Aggregate results
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    # Classification report
    report = classification_report(
        all_y_true, all_y_pred,
        target_names=class_names,
        output_dict=True,
    )

    # AUC (one-vs-rest)
    try:
        auc_ovr = roc_auc_score(
            all_y_true, all_y_proba,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc_ovr = float("nan")

    # Mean feature importances
    mean_importances = np.mean(feature_importances_list, axis=0)
    feature_importance_df = pd.DataFrame({
        "feature": available_features,
        "importance": mean_importances,
    }).sort_values("importance", ascending=False)

    # Train final model on all data
    logger.info("Training final model on all data...")
    final_gb = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    final_gb.fit(X, y_encoded, sample_weight=sample_weights)

    final_lr = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        {"lr__" + k: v for k, v in lr_params.items()},
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    final_lr.fit(X, y_encoded)

    final_ensemble = VotingClassifier(
        estimators=[("gb", final_gb.best_estimator_), ("lr", final_lr.best_estimator_)],
        voting="soft",
    )
    final_ensemble.fit(X, y_encoded)

    results = {
        "model": final_ensemble,
        "label_encoder": le,
        "features": available_features,
        "cv_results": {
            "n_folds": n_outer_folds,
            "macro_auc": float(auc_ovr),
            "classification_report": report,
            "fold_details": fold_results,
        },
        "feature_importance": feature_importance_df,
    }

    logger.info("Nested CV AUC (macro OvR): %.3f", auc_ovr)
    logger.info("Top features:\n%s", feature_importance_df.head(10).to_string())

    return results


def train_and_save(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Main pipeline: merge features, train, evaluate, save."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge features
    merged = merge_features(data_dir)

    # Train
    results = train_ensemble(merged)

    # Save model
    model_path = output_dir / "model.pkl"
    joblib.dump({
        "model": results["model"],
        "label_encoder": results["label_encoder"],
        "features": results["features"],
    }, model_path)
    logger.info("Saved model to %s", model_path)

    # Save CV results
    cv_path = output_dir / "cv_results.json"
    cv_data = results["cv_results"].copy()
    # Convert numpy types for JSON serialization
    cv_data = _make_json_serializable(cv_data)
    with open(cv_path, "w") as f:
        json.dump(cv_data, f, indent=2)
    logger.info("Saved CV results to %s", cv_path)

    # Save feature importances
    fi_path = output_dir / "feature_importance.tsv"
    results["feature_importance"].to_csv(fi_path, sep="\t", index=False)
    logger.info("Saved feature importances to %s", fi_path)

    return results


def _make_json_serializable(obj):
    """Recursively convert numpy types to Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
