"""Leave-one-cancer-out cross-cancer generalization evaluation.

Trains MIL models on a subset of cancer types and evaluates on each held-out
cancer type to assess whether the MSI morphological signature transfers
across tissue origins.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .training import (
    TrainConfig,
    load_labels_and_features,
    train_single_fold,
)

logger = logging.getLogger(__name__)

# Default high-prevalence MSI cancer types
DEFAULT_CANCER_TYPES = ["COAD", "READ", "UCEC", "STAD"]


@dataclass
class LOCOResult:
    """Result from a single leave-one-cancer-out experiment."""

    held_out_cancer: str
    train_cancers: list[str]
    n_train: int
    n_eval: int
    n_eval_msi_h: int
    n_eval_mss: int
    auroc: float
    auprc: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float


def leave_one_cancer_out(
    labels_path: str | Path,
    features_dir: str | Path,
    cancer_types: list[str] | None = None,
    config: TrainConfig | None = None,
    output_dir: str | Path | None = None,
) -> list[LOCOResult]:
    """Run leave-one-cancer-out evaluation.

    For each cancer type, trains on all other types and evaluates on the
    held-out type. This measures cross-cancer generalization of MSI
    morphological features.

    Args:
        labels_path: Path to labels CSV with case_id, msi_status, cancer_type.
        features_dir: Directory with per-slide HDF5 feature files.
        cancer_types: Cancer types to include. Defaults to COAD, READ, UCEC, STAD.
        config: Training configuration. Uses defaults if None.
        output_dir: Where to save results. If None, results are not saved.

    Returns:
        List of LOCOResult, one per held-out cancer type.
    """
    if cancer_types is None:
        cancer_types = DEFAULT_CANCER_TYPES
    if config is None:
        config = TrainConfig()

    # Load all data
    all_slide_ids, all_labels, cancer_type_dict = load_labels_and_features(
        labels_path, features_dir, cancer_types=cancer_types,
    )

    results = []

    for held_out in cancer_types:
        train_cancers = [c for c in cancer_types if c != held_out]
        logger.info(
            f"LOCO: holding out {held_out}, training on {train_cancers}"
        )

        # Split slides by cancer type
        train_ids = [
            sid for sid in all_slide_ids if cancer_type_dict[sid] != held_out
        ]
        eval_ids = [
            sid for sid in all_slide_ids if cancer_type_dict[sid] == held_out
        ]

        if not eval_ids:
            logger.warning(f"  No slides for held-out cancer {held_out}, skipping")
            continue

        if not train_ids:
            logger.warning(f"  No training slides when holding out {held_out}, skipping")
            continue

        n_eval_msi = sum(1 for sid in eval_ids if all_labels[sid] == 1)
        n_eval_mss = len(eval_ids) - n_eval_msi

        if n_eval_msi == 0 or n_eval_mss == 0:
            logger.warning(
                f"  Held-out {held_out} has only one class "
                f"(MSI-H={n_eval_msi}, MSS={n_eval_mss}), "
                f"AUROC will be undefined"
            )

        # Train on all other cancers, evaluate on held-out
        checkpoint_dir = (
            Path(output_dir) / f"loco_{held_out}" if output_dir else None
        )
        metrics, _ = train_single_fold(
            train_ids, eval_ids, all_labels, features_dir, config,
            checkpoint_dir=checkpoint_dir,
        )

        result = LOCOResult(
            held_out_cancer=held_out,
            train_cancers=train_cancers,
            n_train=len(train_ids),
            n_eval=len(eval_ids),
            n_eval_msi_h=n_eval_msi,
            n_eval_mss=n_eval_mss,
            auroc=metrics.get("auroc", float("nan")),
            auprc=metrics.get("auprc", float("nan")),
            balanced_accuracy=metrics.get("balanced_accuracy", 0.0),
            sensitivity=metrics.get("sensitivity", 0.0),
            specificity=metrics.get("specificity", 0.0),
        )
        results.append(result)

        logger.info(
            f"  {held_out}: AUROC={result.auroc:.4f}, "
            f"AUPRC={result.auprc:.4f}, "
            f"Sens={result.sensitivity:.3f}, "
            f"Spec={result.specificity:.3f} "
            f"(n_train={result.n_train}, n_eval={result.n_eval})"
        )

    # Save results
    if output_dir and results:
        _save_loco_results(results, Path(output_dir), config)

    return results


def pairwise_generalization(
    labels_path: str | Path,
    features_dir: str | Path,
    cancer_types: list[str] | None = None,
    config: TrainConfig | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build a full cross-cancer generalization matrix.

    Trains on each single cancer type and evaluates on every other type,
    producing an NxN matrix of AUROC scores.

    Args:
        labels_path: Path to labels CSV.
        features_dir: Directory with HDF5 feature files.
        cancer_types: Cancer types to include.
        config: Training configuration.
        output_dir: Where to save the matrix.

    Returns:
        DataFrame with train types as rows and eval types as columns.
    """
    if cancer_types is None:
        cancer_types = DEFAULT_CANCER_TYPES
    if config is None:
        config = TrainConfig()

    all_slide_ids, all_labels, cancer_type_dict = load_labels_and_features(
        labels_path, features_dir, cancer_types=cancer_types,
    )

    # Group slides by cancer type
    slides_by_cancer: dict[str, list[str]] = {}
    for sid in all_slide_ids:
        ct = cancer_type_dict[sid]
        slides_by_cancer.setdefault(ct, []).append(sid)

    # Build NxN matrix
    matrix = pd.DataFrame(
        np.nan, index=cancer_types, columns=cancer_types, dtype=float,
    )
    matrix.index.name = "train_cancer"
    matrix.columns.name = "eval_cancer"

    for train_cancer in cancer_types:
        train_ids = slides_by_cancer.get(train_cancer, [])
        if not train_ids:
            logger.warning(f"No slides for train cancer {train_cancer}")
            continue

        for eval_cancer in cancer_types:
            eval_ids = slides_by_cancer.get(eval_cancer, [])
            if not eval_ids:
                continue

            logger.info(f"Pairwise: train={train_cancer} -> eval={eval_cancer}")

            checkpoint_dir = (
                Path(output_dir) / f"pair_{train_cancer}_{eval_cancer}"
                if output_dir
                else None
            )
            metrics, _ = train_single_fold(
                train_ids, eval_ids, all_labels, features_dir, config,
                checkpoint_dir=checkpoint_dir,
            )

            matrix.loc[train_cancer, eval_cancer] = metrics.get(
                "auroc", float("nan")
            )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(out / "generalization_matrix.csv")
        logger.info(f"Generalization matrix saved to {out / 'generalization_matrix.csv'}")

    return matrix


def _save_loco_results(
    results: list[LOCOResult],
    output_dir: Path,
    config: TrainConfig,
) -> None:
    """Save LOCO results to CSV and JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-cancer results CSV
    rows = []
    for r in results:
        rows.append({
            "held_out_cancer": r.held_out_cancer,
            "train_cancers": ",".join(r.train_cancers),
            "n_train": r.n_train,
            "n_eval": r.n_eval,
            "n_eval_msi_h": r.n_eval_msi_h,
            "n_eval_mss": r.n_eval_mss,
            "auroc": r.auroc,
            "auprc": r.auprc,
            "balanced_accuracy": r.balanced_accuracy,
            "sensitivity": r.sensitivity,
            "specificity": r.specificity,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "loco_results.csv", index=False)

    # Summary JSON
    summary = {
        "model": config.model_name,
        "n_cancer_types": len(results),
        "cancer_types_evaluated": [r.held_out_cancer for r in results],
    }
    for metric in ["auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity"]:
        values = [
            getattr(r, metric) for r in results
            if not np.isnan(getattr(r, metric))
        ]
        if values:
            summary[f"mean_{metric}"] = round(float(np.mean(values)), 4)
            summary[f"std_{metric}"] = round(float(np.std(values)), 4)
            summary[f"min_{metric}"] = round(float(np.min(values)), 4)
            summary[f"max_{metric}"] = round(float(np.max(values)), 4)

    with open(output_dir / "loco_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"LOCO results saved to {output_dir}")
