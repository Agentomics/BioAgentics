"""Classifier training with leave-one-study-out cross-validation.

Trains elastic net logistic regression, random forest, and XGBoost classifiers
to predict anti-TNF response using the selected gene signature.

Usage:
    uv run python -m bioagentics.data.anti_tnf.classifier
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_BATCH_DIR = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "batch_correction"
DEFAULT_FS_DIR = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "feature_selection"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "classifier"


def load_data(
    batch_dir: Path,
    fs_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load expression data restricted to selected features.

    Returns (X: samples x genes, y: labels, study: study labels).
    """
    expr = pd.read_csv(batch_dir / "expression_combat.csv", index_col=0)
    metadata = pd.read_csv(batch_dir / "metadata.csv")
    ranked = pd.read_csv(fs_dir / "ranked_genes.csv")

    # Get selected genes
    signature = ranked[ranked["selected"]]["gene"].tolist()
    available = [g for g in signature if g in expr.index]
    logger.info("Using %d/%d signature genes", len(available), len(signature))

    # Build X, y, study
    meta = metadata.set_index("sample_id").loc[expr.columns]
    X = expr.loc[available].T
    y = (meta["response_status"] == "non_responder").astype(int)
    y.name = "non_responder"
    study = meta["study"]

    return X, y, study


MODELS = {
    "elastic_net": {
        "estimator": LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000, random_state=42,
        ),
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.9],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [2, 5],
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(
            random_state=42, eval_metric="logloss", use_label_encoder=False,
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
        },
    },
}


def loso_cv(
    X: pd.DataFrame,
    y: pd.Series,
    study: pd.Series,
    model_name: str,
) -> dict:
    """Leave-one-study-out cross-validation with inner hyperparameter tuning.

    Returns dict with per-study and aggregate metrics plus predictions.
    """
    cfg = MODELS[model_name]
    studies = study.unique()
    scaler = StandardScaler()

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_studies = []
    per_study_metrics = []

    for test_study in studies:
        test_mask = study == test_study
        train_mask = ~test_mask

        X_train = X[train_mask].values
        y_train = y[train_mask].values
        X_test = X[test_mask].values
        y_test = y[test_mask].values

        # Scale
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y_train))), shuffle=True, random_state=42)
        grid = GridSearchCV(
            cfg["estimator"], cfg["param_grid"],
            cv=inner_cv, scoring="roc_auc", n_jobs=-1, refit=True,
        )
        grid.fit(X_train_s, y_train)

        # Predict
        y_prob = grid.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Per-study metrics
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan

        sensitivity = recall_score(y_test, y_pred, zero_division=0)
        specificity = recall_score(1 - y_test, 1 - y_pred, zero_division=0)

        per_study_metrics.append({
            "study": test_study,
            "n_samples": len(y_test),
            "n_responder": (y_test == 0).sum(),
            "n_non_responder": (y_test == 1).sum(),
            "auc": auc,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "best_params": str(grid.best_params_),
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        all_studies.extend([test_study] * len(y_test))

        logger.info(
            "  %s held out: AUC=%.3f, BA=%.3f, Sens=%.3f, Spec=%.3f (n=%d)",
            test_study, auc, balanced_accuracy_score(y_test, y_pred),
            sensitivity, specificity, len(y_test),
        )

    # Aggregate metrics
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    try:
        agg_auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        agg_auc = np.nan

    return {
        "model": model_name,
        "per_study": pd.DataFrame(per_study_metrics),
        "aggregate": {
            "auc": agg_auc,
            "balanced_accuracy": balanced_accuracy_score(all_y_true, all_y_pred),
            "sensitivity": recall_score(all_y_true, all_y_pred, zero_division=0),
            "specificity": recall_score(1 - all_y_true, 1 - all_y_pred, zero_division=0),
            "accuracy": accuracy_score(all_y_true, all_y_pred),
            "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
        },
        "y_true": all_y_true,
        "y_prob": all_y_prob,
        "y_pred": all_y_pred,
        "studies": np.array(all_studies),
    }


def plot_roc_curves(results: dict[str, dict], output_dir: Path) -> None:
    """Generate ROC curves for all models (per-study and aggregate)."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    study_colors = {"GSE16879": "#1f77b4", "GSE12251": "#ff7f0e", "GSE73661": "#2ca02c"}

    for ax, (model_name, res) in zip(axes, results.items()):
        # Per-study ROC
        for study_name in np.unique(res["studies"]):
            mask = res["studies"] == study_name
            y_t = res["y_true"][mask]
            y_p = res["y_prob"][mask]
            if len(np.unique(y_t)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_t, y_p)
            auc = roc_auc_score(y_t, y_p)
            ax.plot(fpr, tpr, color=study_colors.get(study_name, "gray"),
                    label=f"{study_name} (AUC={auc:.2f})", alpha=0.7)

        # Aggregate ROC
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, "k-", linewidth=2, label=f"Aggregate (AUC={res['aggregate']['auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name}")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("LOSO-CV ROC Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curves")


def plot_calibration(results: dict[str, dict], output_dir: Path) -> None:
    """Generate calibration plots for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"elastic_net": "#1f77b4", "random_forest": "#ff7f0e", "xgboost": "#2ca02c"}

    for model_name, res in results.items():
        prob_true, prob_pred = calibration_curve(res["y_true"], res["y_prob"], n_bins=8)
        ax.plot(prob_pred, prob_true, "o-", color=colors.get(model_name, "gray"),
                label=model_name)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_plot.png", dpi=150)
    plt.close(fig)
    logger.info("Saved calibration plot")


def run_shap_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    best_model_name: str,
    output_dir: Path,
) -> None:
    """Run SHAP feature importance on the best model trained on all data."""
    cfg = MODELS[best_model_name]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # Train on all data with default params for SHAP visualization
    model = cfg["estimator"].__class__(**cfg["estimator"].get_params())
    model.fit(X_scaled, y.values)

    # SHAP
    if best_model_name == "elastic_net":
        explainer = shap.LinearExplainer(model, X_scaled)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_scaled)

    # For binary classification, some explainers return list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=X.columns.tolist(),
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary plot")

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    mean_abs.to_csv(output_dir / "shap_importance.csv", header=["mean_abs_shap"])
    logger.info("Saved SHAP importance values")


def run_classifier_pipeline(
    batch_dir: Path = DEFAULT_BATCH_DIR,
    fs_dir: Path = DEFAULT_FS_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, dict]:
    """Run the full classifier training pipeline.

    Returns dict of model_name -> results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, study = load_data(batch_dir, fs_dir)
    logger.info("Loaded: %d samples x %d features, %d studies", X.shape[0], X.shape[1], study.nunique())
    logger.info("Class balance: %d R, %d NR", (y == 0).sum(), (y == 1).sum())

    # Train all models with LOSO-CV
    results = {}
    all_metrics = []

    for model_name in MODELS:
        logger.info("Training %s...", model_name)
        res = loso_cv(X, y, study, model_name)
        results[model_name] = res

        agg = res["aggregate"]
        logger.info(
            "%s aggregate: AUC=%.3f, BA=%.3f, Sens=%.3f, Spec=%.3f",
            model_name, agg["auc"], agg["balanced_accuracy"],
            agg["sensitivity"], agg["specificity"],
        )

        # Save per-study metrics
        res["per_study"].to_csv(output_dir / f"{model_name}_per_study.csv", index=False)
        all_metrics.append({"model": model_name, **agg})

    # Save aggregate comparison
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    logger.info("Saved aggregate metrics")

    # Plots
    plot_roc_curves(results, output_dir)
    plot_calibration(results, output_dir)

    # SHAP on best model
    best_model = metrics_df.loc[metrics_df["auc"].idxmax(), "model"]
    logger.info("Best model by AUC: %s (%.3f)", best_model, metrics_df["auc"].max())
    run_shap_analysis(X, y, best_model, output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train anti-TNF response classifiers with LOSO-CV"
    )
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--fs-dir", type=Path, default=DEFAULT_FS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_classifier_pipeline(
        batch_dir=args.batch_dir, fs_dir=args.fs_dir, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
