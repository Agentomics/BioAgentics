"""External validation and Cunningham Panel comparison for biomarker panel.

Applies trained classifiers to a held-out validation dataset and compares
the gene signature with Cunningham Panel components.

Usage:
    uv run python -m bioagentics.models.biomarker_validation \\
        --model models/RandomForest_combined.pkl \\
        --validation validation.h5ad \\
        --condition-key condition --dest output/
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
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_sets import CUNNINGHAM_PANEL_GENES

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "transcriptomic-biomarker-panel"


@dataclass
class ValidationResult:
    """Result from validating a classifier on held-out data."""

    classifier_name: str
    mode: str
    auc: float
    sensitivity: float
    specificity: float
    n_samples: int
    n_features_used: int
    y_true: np.ndarray
    y_prob: np.ndarray
    confusion: np.ndarray


def load_trained_model(model_path: Path) -> dict:
    """Load a trained classifier from pickle.

    Returns dict with keys: model, features, label_encoder.
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "model" not in data:
        raise ValueError(f"Invalid model file: {model_path}")

    logger.info(
        "Loaded model from %s (%d features)",
        model_path.name, len(data.get("features", [])),
    )
    return data


def validate_on_holdout(
    model_data: dict,
    adata: ad.AnnData,
    condition_key: str = "condition",
    classifier_name: str = "unknown",
    mode: str = "combined",
) -> ValidationResult:
    """Apply a trained classifier to a held-out validation dataset.

    Parameters
    ----------
    model_data : dict
        From load_trained_model: model, features, label_encoder.
    adata : AnnData
        Held-out validation data (normalized, same preprocessing).
    condition_key : str
        Label column in adata.obs.
    classifier_name : str
        Name for reporting.
    mode : str
        "combined", "male", or "female".

    Returns
    -------
    ValidationResult with metrics and predictions.
    """
    model = model_data["model"]
    features = model_data["features"]
    le = model_data.get("label_encoder")

    if condition_key not in adata.obs.columns:
        raise ValueError(f"'{condition_key}' not in validation data obs columns")

    # Align features
    available = set(adata.var_names)
    valid_features = [f for f in features if f in available]
    missing = set(features) - available

    if len(valid_features) < len(features) * 0.5:
        raise ValueError(
            f"Only {len(valid_features)}/{len(features)} features found in validation data"
        )

    if missing:
        logger.warning(
            "%d/%d features missing in validation data — filling with zeros",
            len(missing), len(features),
        )

    # Build feature matrix with all expected features
    var_names_list = list(adata.var_names)
    X = np.zeros((adata.n_obs, len(features)), dtype=np.float32)
    for i, feat in enumerate(features):
        if feat in available:
            col_idx = var_names_list.index(feat)
            X[:, i] = np.array(adata.X[:, col_idx]).ravel()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Encode labels
    if le is not None:
        y = le.transform(adata.obs[condition_key].astype(str))
    else:
        le_new = LabelEncoder()
        y = le_new.fit_transform(adata.obs[condition_key].astype(str))

    # Predict
    y_prob = model.predict_proba(X)[:, 1]
    metrics = _compute_metrics(y, y_prob)

    cm = confusion_matrix(y, (y_prob >= 0.5).astype(int))

    result = ValidationResult(
        classifier_name=classifier_name,
        mode=mode,
        auc=metrics["auc"],
        sensitivity=metrics["sensitivity"],
        specificity=metrics["specificity"],
        n_samples=len(y),
        n_features_used=len(valid_features),
        y_true=y,
        y_prob=y_prob,
        confusion=cm,
    )

    logger.info(
        "Validation %s (%s): AUC=%.3f, Sens=%.3f, Spec=%.3f (%d samples)",
        classifier_name, mode, result.auc, result.sensitivity, result.specificity,
        result.n_samples,
    )
    return result


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute AUC, sensitivity, specificity."""
    if len(np.unique(y_true)) < 2:
        return {"auc": 0.0, "sensitivity": 0.0, "specificity": 0.0}

    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = j_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_prob >= optimal_threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    return {
        "auc": auc,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


def compare_with_cunningham_genes(
    selected_features: list[str],
) -> pd.DataFrame:
    """Compare classifier features with Cunningham Panel genes.

    Returns DataFrame identifying overlap and novel genes.
    """
    cunningham = set(CUNNINGHAM_PANEL_GENES)
    selected = set(selected_features)

    overlap = sorted(cunningham & selected)
    novel = sorted(selected - cunningham)
    cunningham_only = sorted(cunningham - selected)

    rows = []
    for gene in sorted(cunningham | selected):
        rows.append({
            "gene": gene,
            "in_classifier": gene in selected,
            "in_cunningham": gene in cunningham,
            "category": (
                "both" if gene in overlap
                else "classifier_only" if gene in novel
                else "cunningham_only"
            ),
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Cunningham comparison: %d overlap, %d novel, %d cunningham-only",
        len(overlap), len(novel), len(cunningham_only),
    )
    return df


def plot_roc_curves(
    results: list[ValidationResult],
    title: str = "Validation ROC Curves",
    save_path: Path | None = None,
) -> None:
    """Plot ROC curves for multiple classifiers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    for vr in results:
        if len(np.unique(vr.y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(vr.y_true, vr.y_prob)
        label = f"{vr.classifier_name} ({vr.mode}) AUC={vr.auc:.3f}"
        ax.plot(fpr, tpr, linewidth=1.5, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved ROC curves: %s", save_path)
    plt.close(fig)


def plot_confusion_matrix(
    vr: ValidationResult,
    save_path: Path | None = None,
) -> None:
    """Plot confusion matrix for a validation result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = vr.confusion

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{vr.classifier_name} ({vr.mode})")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Control", "Case"])
    ax.set_yticklabels(["Control", "Case"])

    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved confusion matrix: %s", save_path)
    plt.close(fig)


def validation_pipeline(
    model_paths: list[Path],
    validation_adata: ad.AnnData,
    condition_key: str = "condition",
    dest_dir: Path | None = None,
) -> dict[str, list[ValidationResult] | pd.DataFrame]:
    """Full validation pipeline.

    Parameters
    ----------
    model_paths : list[Path]
        Paths to trained model pickle files.
    validation_adata : AnnData
        Held-out validation dataset.
    condition_key : str
        Label column.
    dest_dir : Path, optional
        Output directory.

    Returns
    -------
    Dict with keys: results, metrics, cunningham_comparison.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[ValidationResult] = []
    all_features: set[str] = set()

    for model_path in model_paths:
        model_data = load_trained_model(model_path)
        features = model_data.get("features", [])
        all_features.update(features)

        # Infer classifier name and mode from filename
        stem = model_path.stem
        parts = stem.rsplit("_", 1)
        clf_name = parts[0] if len(parts) == 2 else stem
        mode = parts[1] if len(parts) == 2 else "combined"

        vr = validate_on_holdout(
            model_data, validation_adata, condition_key,
            classifier_name=clf_name, mode=mode,
        )
        all_results.append(vr)

        plot_confusion_matrix(
            vr,
            save_path=dest_dir / f"confusion_{clf_name}_{mode}.png",
        )

    # Metrics summary
    metrics_rows = [
        {
            "classifier": vr.classifier_name,
            "mode": vr.mode,
            "auc": vr.auc,
            "sensitivity": vr.sensitivity,
            "specificity": vr.specificity,
            "n_samples": vr.n_samples,
            "n_features": vr.n_features_used,
        }
        for vr in all_results
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(dest_dir / "validation_metrics.csv", index=False)

    # ROC curves
    plot_roc_curves(
        all_results,
        title="Held-out Validation ROC Curves",
        save_path=dest_dir / "validation_roc.png",
    )

    # Cunningham comparison
    cunningham_df = compare_with_cunningham_genes(sorted(all_features))
    cunningham_df.to_csv(dest_dir / "cunningham_gene_comparison.csv", index=False)

    # Success criteria check
    best_auc = max(vr.auc for vr in all_results) if all_results else 0
    target_met = best_auc >= 0.80
    logger.info(
        "Validation complete: best AUC=%.3f (%s target AUC >= 0.80)",
        best_auc, "MEETS" if target_met else "BELOW",
    )

    return {
        "results": all_results,
        "metrics": metrics_df,
        "cunningham_comparison": cunningham_df,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate classifiers on held-out data"
    )
    parser.add_argument("--model", type=Path, action="append", required=True,
                        help="Path to trained model pickle (can repeat)")
    parser.add_argument("--validation", type=Path, required=True,
                        help="Held-out validation h5ad file")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    val_adata = ad.read_h5ad(args.validation)
    results = validation_pipeline(
        model_paths=args.model,
        validation_adata=val_adata,
        condition_key=args.condition_key,
        dest_dir=args.dest,
    )

    print("\n=== VALIDATION RESULTS ===")
    print(results["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
