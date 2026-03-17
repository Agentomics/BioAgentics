"""Per-class AUROC evaluation with head/body/tail stratification.

Consistent evaluation module used across all CXR rare disease experiments.
Computes per-class AUROC, group-stratified means, and generates
performance comparison plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from bioagentics.cxr_rare.config import (
    BODY_CLASSES,
    HEAD_CLASSES,
    LABEL_NAMES,
    OUTPUT_DIR,
    TAIL_CLASSES,
)

logger = logging.getLogger(__name__)

_BIN_COLORS = {"head": "#2196F3", "body": "#FF9800", "tail": "#F44336"}


def compute_per_class_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute AUROC for each class independently.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape (N, C).
    y_score : np.ndarray
        Predicted scores/probabilities, shape (N, C).
    label_names : list[str], optional
        Class names. Defaults to CXR-LT label set.

    Returns
    -------
    dict[str, float]
        Per-class AUROC. NaN for classes with no positive samples.
    """
    label_names = label_names or LABEL_NAMES
    n_classes = y_true.shape[1]
    aurocs: dict[str, float] = {}
    for i in range(min(n_classes, len(label_names))):
        name = label_names[i]
        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == len(y_true):
            aurocs[name] = float("nan")
            logger.warning("Class '%s' has no variation — AUROC undefined", name)
        else:
            aurocs[name] = float(roc_auc_score(y_true[:, i], y_score[:, i]))
    return aurocs


def stratified_auroc(
    per_class: dict[str, float],
    head_classes: list[str] | None = None,
    body_classes: list[str] | None = None,
    tail_classes: list[str] | None = None,
) -> dict[str, float]:
    """Compute macro-AUROC and per-bin mean AUROC.

    Returns
    -------
    dict with keys: macro_auroc, head_mean, body_mean, tail_mean
    """
    head = head_classes or HEAD_CLASSES
    body = body_classes or BODY_CLASSES
    tail = tail_classes or TAIL_CLASSES

    def _mean(names: list[str]) -> float:
        vals = [per_class[n] for n in names if n in per_class and not np.isnan(per_class[n])]
        return float(np.mean(vals)) if vals else float("nan")

    all_valid = [v for v in per_class.values() if not np.isnan(v)]
    return {
        "macro_auroc": float(np.mean(all_valid)) if all_valid else float("nan"),
        "head_mean": _mean(head),
        "body_mean": _mean(body),
        "tail_mean": _mean(tail),
    }


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str] | None = None,
    head_classes: list[str] | None = None,
    body_classes: list[str] | None = None,
    tail_classes: list[str] | None = None,
) -> dict:
    """Full evaluation: per-class AUROC + stratified summary.

    Returns
    -------
    dict with keys: per_class_auroc, summary
    """
    per_class = compute_per_class_auroc(y_true, y_score, label_names)
    summary = stratified_auroc(per_class, head_classes, body_classes, tail_classes)
    return {"per_class_auroc": per_class, "summary": summary}


def _get_class_bin(
    name: str,
    head_classes: list[str],
    body_classes: list[str],
    tail_classes: list[str],
) -> str:
    if name in head_classes:
        return "head"
    if name in body_classes:
        return "body"
    if name in tail_classes:
        return "tail"
    return "unknown"


def plot_per_class_auroc(
    per_class: dict[str, float],
    output_path: Path,
    title: str = "Per-Class AUROC",
    head_classes: list[str] | None = None,
    body_classes: list[str] | None = None,
    tail_classes: list[str] | None = None,
) -> Path:
    """Bar chart of per-class AUROC, colored by head/body/tail."""
    head = head_classes or HEAD_CLASSES
    body = body_classes or BODY_CLASSES
    tail = tail_classes or TAIL_CLASSES

    names = list(per_class.keys())
    values = [per_class[n] for n in names]
    colors = [_BIN_COLORS.get(_get_class_bin(n, head, body, tail), "#999") for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(names)), values, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_BIN_COLORS["head"], label="Head"),
        Patch(facecolor=_BIN_COLORS["body"], label="Body"),
        Patch(facecolor=_BIN_COLORS["tail"], label="Tail"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved AUROC plot: %s", output_path)
    return output_path


def save_results(
    results: dict,
    output_dir: Path,
    experiment_name: str = "experiment",
) -> Path:
    """Save evaluation results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"results_{experiment_name}.json"

    # Convert NaN to null for JSON
    def _clean(obj: object) -> object:
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    logger.info("Saved results: %s", path)
    return path


def evaluate_and_save(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
    experiment_name: str = "experiment",
    label_names: list[str] | None = None,
) -> dict:
    """Evaluate, save JSON results, and generate plots."""
    results = evaluate(y_true, y_score, label_names)

    save_results(results, output_dir, experiment_name)
    plot_per_class_auroc(
        results["per_class_auroc"],
        output_dir / "figures" / f"auroc_{experiment_name}.png",
        title=f"Per-Class AUROC — {experiment_name}",
    )

    summary = results["summary"]
    logger.info(
        "Evaluation [%s]: macro=%.4f, head=%.4f, body=%.4f, tail=%.4f",
        experiment_name,
        summary["macro_auroc"],
        summary["head_mean"],
        summary["body_mean"],
        summary["tail_mean"],
    )
    return results
