"""Flare prediction model from cytokine profiles for PANDAS/PANS.

Implements logistic regression baseline, random forest with feature importance,
and optional time-series features. Uses scikit-learn with leave-one-study-out
cross-validation.

Usage::

    from bioagentics.cytokine_flare_predictor import FlarePredictor

    predictor = FlarePredictor(dataset)
    results = predictor.run()
    predictor.plot_roc(results, output_path="roc.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT
from bioagentics.cytokine_extraction import CytokineDataset

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "cytokine-network-flare-prediction"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Results from a single model evaluation."""

    model_name: str
    auc_score: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    accuracy: float = 0.0
    y_true: list[int] = field(default_factory=list)
    y_prob: list[float] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)
    cv_folds: int = 0


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_feature_matrix(dataset: CytokineDataset) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Convert cytokine dataset to a feature matrix for classification.

    Pivots the data so that each row is (study_id, condition) and columns
    are analyte values. Returns features (X), labels (y), and study IDs
    for leave-one-study-out CV.

    Labels: 1 = flare, 0 = remission/healthy_control/baseline.
    """
    df = dataset.to_dataframe()

    # Build wide-format: one row per (study_id, condition), one column per analyte
    pivot = df.pivot_table(
        index=["study_id", "condition"],
        columns="analyte_name",
        values="mean_or_median",
        aggfunc="mean",
    ).reset_index()

    # Binary label: flare vs non-flare
    pivot["label"] = (pivot["condition"] == "flare").astype(int)

    feature_cols = [c for c in pivot.columns if c not in ("study_id", "condition", "label")]
    X = pivot[feature_cols].fillna(0)
    y = pivot["label"]
    study_ids = pivot["study_id"]

    logger.info("Feature matrix: %d samples × %d features", X.shape[0], X.shape[1])
    return X, y, study_ids


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def leave_one_study_out_cv(
    X: pd.DataFrame,
    y: pd.Series,
    study_ids: pd.Series,
    model,
) -> PredictionResult:
    """Leave-one-study-out cross-validation.

    For each unique study, hold it out as test set and train on the rest.
    Aggregates predictions across all folds.
    """
    all_y_true = []
    all_y_prob = []
    unique_studies = study_ids.unique()

    for test_study in unique_studies:
        test_mask = study_ids == test_study
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(y_train.unique()) < 2 or len(X_test) == 0:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(probs.tolist())

    return _compute_metrics(all_y_true, all_y_prob, model.__class__.__name__, len(unique_studies))


def _compute_metrics(
    y_true: list[int],
    y_prob: list[float],
    model_name: str,
    n_folds: int,
) -> PredictionResult:
    """Compute AUC, sensitivity, specificity from aggregated predictions."""
    result = PredictionResult(model_name=model_name, cv_folds=n_folds)
    result.y_true = y_true
    result.y_prob = y_prob

    if not y_true or len(set(y_true)) < 2:
        return result

    y_t = np.array(y_true)
    y_p = np.array(y_prob)

    fpr, tpr, thresholds = roc_curve(y_t, y_p)
    result.auc_score = float(auc(fpr, tpr))

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_thresh = thresholds[optimal_idx]

    y_pred = (y_p >= optimal_thresh).astype(int)
    tp = np.sum((y_pred == 1) & (y_t == 1))
    tn = np.sum((y_pred == 0) & (y_t == 0))
    fp = np.sum((y_pred == 1) & (y_t == 0))
    fn = np.sum((y_pred == 0) & (y_t == 1))

    result.sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    result.specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    result.accuracy = float((tp + tn) / len(y_t)) if len(y_t) > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------


class FlarePredictor:
    """Flare prediction pipeline from cytokine profiles."""

    def __init__(self, dataset: CytokineDataset) -> None:
        self.dataset = dataset
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.study_ids: pd.Series | None = None

    def run(self) -> list[PredictionResult]:
        """Run all prediction models and return results."""
        self.X, self.y, self.study_ids = build_feature_matrix(self.dataset)

        if self.X.shape[0] < 4 or len(self.y.unique()) < 2:
            logger.warning("Insufficient data for prediction: %d samples", self.X.shape[0])
            return []

        results = []

        # Logistic regression baseline
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr_result = leave_one_study_out_cv(self.X, self.y, self.study_ids, lr)
        # Get feature importances from full model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        lr.fit(X_scaled, self.y)
        lr_result.feature_importances = dict(zip(self.X.columns, np.abs(lr.coef_[0])))
        results.append(lr_result)

        # Random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf_result = leave_one_study_out_cv(self.X, self.y, self.study_ids, rf)
        rf.fit(X_scaled, self.y)
        rf_result.feature_importances = dict(zip(self.X.columns, rf.feature_importances_))
        results.append(rf_result)

        for r in results:
            logger.info(
                "%s: AUC=%.3f, Sens=%.3f, Spec=%.3f, k=%d folds",
                r.model_name, r.auc_score, r.sensitivity, r.specificity, r.cv_folds,
            )

        return results

    @staticmethod
    def plot_roc(
        results: list[PredictionResult],
        output_path: Path | str | None = None,
    ) -> Path | None:
        """Generate ROC curves for all models."""
        if not results:
            return None

        if output_path is None:
            output_path = OUTPUT_DIR / "flare_roc_curves.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 7))
        colors = ["steelblue", "firebrick", "forestgreen"]

        for r, color in zip(results, colors):
            if not r.y_true or len(set(r.y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(r.y_true, r.y_prob)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{r.model_name} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Flare Prediction — ROC Curves (LOSO-CV)")
        ax.legend(loc="lower right")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved ROC curves: %s", output_path)
        return output_path

    @staticmethod
    def plot_feature_importance(
        result: PredictionResult,
        top_n: int = 15,
        output_path: Path | str | None = None,
    ) -> Path | None:
        """Plot feature importance bar chart."""
        if not result.feature_importances:
            return None

        if output_path is None:
            output_path = OUTPUT_DIR / f"feature_importance_{result.model_name}.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sorted_feats = sorted(result.feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [f[0] for f in sorted_feats]
        values = [f[1] for f in sorted_feats]

        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.35)))
        ax.barh(range(len(names)), values, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {result.model_name}")
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved feature importance: %s", output_path)
        return output_path


def results_summary(results: list[PredictionResult]) -> pd.DataFrame:
    """Convert prediction results to a summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "model": r.model_name,
            "auc": r.auc_score,
            "sensitivity": r.sensitivity,
            "specificity": r.specificity,
            "accuracy": r.accuracy,
            "cv_folds": r.cv_folds,
        })
    return pd.DataFrame(rows)
