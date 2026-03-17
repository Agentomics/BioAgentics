"""Cross-institutional train/eval pipeline (MIMIC ↔ MIDRC transfer).

Evaluates domain shift impact by training on one institution and
testing on another. Outputs per-class AUROC comparison tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from bioagentics.cxr_rare.config import (
    CROSS_INST_DIR,
    DEFAULT_TRAIN_CONFIG,
    NUM_CLASSES,
    TrainConfig,
)
from bioagentics.cxr_rare.evaluation.metrics import evaluate, evaluate_and_save
from bioagentics.cxr_rare.training.trainer import CXRTrainer

logger = logging.getLogger(__name__)


def run_cross_institutional_experiments(
    model_factory: callable,
    dataset_a_train: Dataset,
    dataset_a_val: Dataset,
    dataset_b_train: Dataset,
    dataset_b_val: Dataset,
    name_a: str = "MIMIC",
    name_b: str = "MIDRC",
    config: TrainConfig | None = None,
    output_dir: Path = CROSS_INST_DIR,
) -> dict[str, dict]:
    """Run cross-institutional transfer experiments.

    Experiments:
      1. Train on A, eval on A (intra-institutional baseline)
      2. Train on A, eval on B (cross-institutional)
      3. Train on B, eval on A (cross-institutional)
      4. Train on combined, eval on A and B

    Parameters
    ----------
    model_factory : callable
        Function returning a fresh model instance.
    dataset_a_train, dataset_a_val : Dataset
        Institution A datasets.
    dataset_b_train, dataset_b_val : Dataset
        Institution B datasets.

    Returns
    -------
    dict mapping experiment name to evaluation results.
    """
    config = config or DEFAULT_TRAIN_CONFIG
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Experiment 1: A → A
    logger.info("=== Experiment: %s → %s ===", name_a, name_a)
    model = model_factory()
    trainer = CXRTrainer(
        model=model, train_dataset=dataset_a_train, val_dataset=dataset_a_val,
        config=config, output_dir=output_dir,
        experiment_name=f"{name_a}_to_{name_a}",
    )
    trainer.train()
    trainer.load_best_checkpoint()
    results[f"{name_a}_to_{name_a}"] = _eval_model(model, dataset_a_val, trainer.device, config)

    # Experiment 2: A → B
    logger.info("=== Experiment: %s → %s ===", name_a, name_b)
    results[f"{name_a}_to_{name_b}"] = _eval_model(model, dataset_b_val, trainer.device, config)

    # Experiment 3: B → A
    logger.info("=== Experiment: %s → %s ===", name_b, name_a)
    model = model_factory()
    trainer = CXRTrainer(
        model=model, train_dataset=dataset_b_train, val_dataset=dataset_b_val,
        config=config, output_dir=output_dir,
        experiment_name=f"{name_b}_to_{name_a}",
    )
    trainer.train()
    trainer.load_best_checkpoint()
    results[f"{name_b}_to_{name_a}"] = _eval_model(model, dataset_a_val, trainer.device, config)

    # Experiment 4: Combined → each
    logger.info("=== Experiment: Combined → %s/%s ===", name_a, name_b)
    combined_train = ConcatDataset([dataset_a_train, dataset_b_train])
    combined_val = ConcatDataset([dataset_a_val, dataset_b_val])
    model = model_factory()
    trainer = CXRTrainer(
        model=model, train_dataset=combined_train, val_dataset=combined_val,
        config=config, output_dir=output_dir,
        experiment_name="combined",
    )
    trainer.train()
    trainer.load_best_checkpoint()
    results[f"combined_to_{name_a}"] = _eval_model(model, dataset_a_val, trainer.device, config)
    results[f"combined_to_{name_b}"] = _eval_model(model, dataset_b_val, trainer.device, config)

    # Save comparison table
    _save_comparison_table(results, output_dir)
    return results


@torch.no_grad()
def _eval_model(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    config: TrainConfig,
) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    all_labels, all_scores = [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch["labels"].numpy())

    if not all_labels:
        return {}
    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)
    return evaluate(y_true, y_score)


def _save_comparison_table(results: dict[str, dict], output_dir: Path) -> None:
    """Save macro-AUROC comparison table."""
    rows = []
    for name, res in results.items():
        if res and "summary" in res:
            rows.append({
                "experiment": name,
                "macro_auroc": res["summary"]["macro_auroc"],
                "head_mean": res["summary"]["head_mean"],
                "body_mean": res["summary"]["body_mean"],
                "tail_mean": res["summary"]["tail_mean"],
            })
    df = pd.DataFrame(rows)
    path = output_dir / "cross_institutional_comparison.csv"
    df.to_csv(path, index=False)
    logger.info("Saved comparison table: %s", path)
