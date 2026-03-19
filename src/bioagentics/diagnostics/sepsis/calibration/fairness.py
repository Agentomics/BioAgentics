"""Demographic fairness analysis for sepsis early warning models.

Computes AUROC, sensitivity, specificity, and PPV stratified by
age group, sex, and race/ethnicity. Reports maximum subgroup disparity
to verify no subgroup AUROC differs by >0.05.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from bioagentics.diagnostics.sepsis.config import RESULTS_DIR

logger = logging.getLogger(__name__)

# Default age bins for stratification
AGE_BINS = [(18, 45), (45, 65), (65, 80), (80, 200)]
AGE_BIN_LABELS = ["18-44", "45-64", "65-79", "80+"]


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Compute classification metrics per subgroup.

    Parameters
    ----------
    y_true : Binary ground truth labels.
    y_prob : Predicted probabilities.
    groups : Categorical group assignments (same length as y_true).
    threshold : Decision threshold for sens/spec/PPV.

    Returns
    -------
    Dictionary keyed by group value with AUROC, sensitivity,
    specificity, PPV, and sample count.
    """
    unique_groups = np.unique(groups)
    results: dict[str, dict] = {}

    for g in unique_groups:
        mask = groups == g
        n = int(mask.sum())
        n_pos = int(y_true[mask].sum())

        if n < 10 or n_pos < 2 or (n - n_pos) < 2:
            logger.warning(
                "Subgroup '%s' has insufficient samples (%d total, %d pos) — skipping",
                g, n, n_pos,
            )
            continue

        yt = y_true[mask]
        yp = y_prob[mask]

        try:
            auroc = float(roc_auc_score(yt, yp))
        except ValueError:
            auroc = float("nan")

        try:
            auprc = float(average_precision_score(yt, yp))
        except ValueError:
            auprc = float("nan")

        y_pred = (yp >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        results[str(g)] = {
            "auroc": auroc,
            "auprc": auprc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "n_samples": n,
            "n_positive": n_pos,
            "prevalence": float(n_pos / n),
        }

    return results


def compute_fairness_disparity(subgroup_metrics: dict[str, dict]) -> dict:
    """Compute max disparity in AUROC across subgroups.

    Returns
    -------
    Dictionary with max_auroc_disparity, min/max group names,
    and whether the <0.05 target is met.
    """
    aurocs = {
        g: m["auroc"]
        for g, m in subgroup_metrics.items()
        if not np.isnan(m["auroc"])
    }
    if len(aurocs) < 2:
        return {
            "max_auroc_disparity": 0.0,
            "meets_target": True,
            "n_groups_evaluated": len(aurocs),
        }

    min_group = min(aurocs, key=lambda g: aurocs[g])
    max_group = max(aurocs, key=lambda g: aurocs[g])
    disparity = aurocs[max_group] - aurocs[min_group]

    return {
        "max_auroc_disparity": float(disparity),
        "min_auroc_group": min_group,
        "min_auroc_value": float(aurocs[min_group]),
        "max_auroc_group": max_group,
        "max_auroc_value": float(aurocs[max_group]),
        "meets_target": disparity < 0.05,
        "n_groups_evaluated": len(aurocs),
    }


def assign_age_groups(
    ages: np.ndarray,
    bins: list[tuple[int, int]] | None = None,
    labels: list[str] | None = None,
) -> np.ndarray:
    """Assign age values to categorical groups.

    Parameters
    ----------
    ages : Array of age values.
    bins : List of (low, high) tuples.
    labels : Label for each bin.

    Returns
    -------
    Array of group label strings.
    """
    if bins is None:
        bins = AGE_BINS
    if labels is None:
        labels = AGE_BIN_LABELS

    groups = np.full(len(ages), "unknown", dtype=object)
    for (low, high), label in zip(bins, labels):
        mask = (ages >= low) & (ages < high)
        groups[mask] = label
    return groups


def run_fairness_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    demographics: dict[str, np.ndarray],
    results_dir: Path = RESULTS_DIR,
    label: str = "",
) -> dict:
    """Run full fairness analysis across demographic axes.

    Parameters
    ----------
    y_true : Binary labels.
    y_prob : Predicted probabilities.
    demographics : Dictionary with keys like "age", "sex", "ethnicity",
        each mapping to an array of group values aligned with y_true.
    results_dir : Output directory.
    label : Optional label for output filename (e.g., "6h").

    Returns
    -------
    Dictionary with per-axis subgroup metrics and disparity results.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    for axis_name, groups in demographics.items():
        if axis_name == "age":
            groups = assign_age_groups(groups.astype(float))

        logger.info("Fairness analysis: %s (%d unique groups)", axis_name, len(np.unique(groups)))
        subgroup = compute_subgroup_metrics(y_true, y_prob, groups)
        disparity = compute_fairness_disparity(subgroup)

        results[axis_name] = {
            "subgroup_metrics": subgroup,
            "disparity": disparity,
        }
        logger.info(
            "  %s: max AUROC disparity = %.4f (meets target: %s)",
            axis_name,
            disparity["max_auroc_disparity"],
            disparity["meets_target"],
        )

    # Overall fairness summary
    all_meet = all(
        r["disparity"]["meets_target"] for r in results.values()
    )
    results["overall_meets_target"] = all_meet

    suffix = f"_{label}" if label else ""
    out_path = results_dir / f"fairness{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", out_path)

    return results
