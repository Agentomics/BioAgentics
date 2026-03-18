"""Cross-population robustness evaluation for DR screening models.

Tests models on held-out external datasets and stratifies by:
  - Dataset of origin (population)
  - Image quality score
  - DR grade distribution

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.evaluation \\
        --model-path output/.../best_model.pt \\
        --splits data/.../splits.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bioagentics.diagnostics.retinal_dr_screening.augmentation import DRDataset
from bioagentics.diagnostics.retinal_dr_screening.config import (
    BATCH_SIZE,
    DATA_DIR,
    NUM_WORKERS,
    RESULTS_DIR,
    TRAIN_IMAGE_SIZE,
)
from bioagentics.diagnostics.retinal_dr_screening.training import (
    compute_metrics,
    create_model,
    evaluate,
)

logger = logging.getLogger(__name__)


def evaluate_by_dataset(
    model: nn.Module,
    splits_csv: Path,
    split: str,
    device: torch.device,
    image_size: int = TRAIN_IMAGE_SIZE,
) -> dict[str, dict]:
    """Evaluate model performance stratified by dataset source.

    Returns dict mapping dataset_name → metrics dict.
    """
    import pandas as pd

    df = pd.read_csv(splits_csv)
    split_df = df[df["split"] == split]
    datasets = split_df["dataset_source"].unique()

    results = {}
    for ds_name in datasets:
        # Create a temporary CSV with only this dataset
        ds_df = split_df[split_df["dataset_source"] == ds_name]
        if len(ds_df) < 5:
            logger.warning("Too few samples for %s (%d), skipping", ds_name, len(ds_df))
            continue

        # Create dataset directly
        ds = DRDataset.__new__(DRDataset)
        ds.data = ds_df.reset_index(drop=True)
        from bioagentics.diagnostics.retinal_dr_screening.augmentation import get_val_transform
        ds.transform = get_val_transform(image_size)

        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        criterion = nn.CrossEntropyLoss()

        _, _, labels, preds, probs = evaluate(model, loader, criterion, device)
        metrics = compute_metrics(labels, preds, probs)
        metrics["n_samples"] = len(ds_df)
        results[ds_name] = metrics

        logger.info(
            "%s (%d): acc=%.4f qwk=%.4f auc=%.4f",
            ds_name, len(ds_df), metrics["accuracy"], metrics["qwk"],
            metrics["auc_referable"],
        )

    return results


def evaluate_by_quality(
    model: nn.Module,
    splits_csv: Path,
    split: str,
    device: torch.device,
    image_size: int = TRAIN_IMAGE_SIZE,
    n_bins: int = 3,
) -> dict[str, dict]:
    """Evaluate model performance stratified by image quality.

    Bins images into quality terciles (low/medium/high) based on sharpness.
    """
    import pandas as pd
    from bioagentics.diagnostics.retinal_dr_screening.augmentation import get_val_transform

    df = pd.read_csv(splits_csv)
    split_df = df[df["split"] == split]

    if "quality_sharpness" not in split_df.columns:
        logger.warning("No quality scores in splits CSV — skipping quality-stratified eval")
        return {}

    # Bin by quality terciles
    split_df = split_df.copy()
    split_df["quality_bin"] = pd.qcut(
        split_df["quality_sharpness"], q=n_bins, labels=["low", "medium", "high"]
    )

    results = {}
    for bin_name in ["low", "medium", "high"]:
        bin_df = split_df[split_df["quality_bin"] == bin_name]
        if len(bin_df) < 5:
            continue

        ds = DRDataset.__new__(DRDataset)
        ds.data = bin_df.reset_index(drop=True)
        ds.transform = get_val_transform(image_size)

        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        criterion = nn.CrossEntropyLoss()
        _, _, labels, preds, probs = evaluate(model, loader, criterion, device)
        metrics = compute_metrics(labels, preds, probs)
        metrics["n_samples"] = len(bin_df)
        results[bin_name] = metrics

    return results


def cross_population_report(
    model_path: Path,
    splits_csv: Path,
    model_name: str = "mobilenetv3_small_100",
    image_size: int = TRAIN_IMAGE_SIZE,
) -> dict:
    """Generate full cross-population robustness report.

    Tests on test set and external_val set, stratified by dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(model_name, pretrained=False)
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    report = {}

    for split in ["test", "external_val"]:
        logger.info("Evaluating %s split...", split)
        by_dataset = evaluate_by_dataset(model, splits_csv, split, device, image_size)
        by_quality = evaluate_by_quality(model, splits_csv, split, device, image_size)

        report[split] = {
            "by_dataset": by_dataset,
            "by_quality": by_quality,
        }

        # Check AUC drop
        if by_dataset:
            aucs = [m["auc_referable"] for m in by_dataset.values()]
            max_drop = max(aucs) - min(aucs) if len(aucs) > 1 else 0
            report[split]["auc_range"] = round(max_drop, 4)
            report[split]["auc_mean"] = round(float(np.mean(aucs)), 4)

    # Save report
    results_dir = RESULTS_DIR / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "cross_population_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Cross-population report saved")
    return report


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Cross-population robustness evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=DATA_DIR / "splits.csv")
    parser.add_argument("--model-name", default="mobilenetv3_small_100")
    parser.add_argument("--image-size", type=int, default=TRAIN_IMAGE_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = cross_population_report(
        args.model_path, args.splits, args.model_name, args.image_size,
    )

    for split, data in report.items():
        print(f"\n{split}:")
        for ds_name, metrics in data.get("by_dataset", {}).items():
            print(f"  {ds_name}: acc={metrics['accuracy']:.4f} "
                  f"qwk={metrics['qwk']:.4f} auc={metrics['auc_referable']:.4f}")


if __name__ == "__main__":
    main()
