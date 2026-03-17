"""Decoupled training: representation learning + classifier re-balancing.

Two-stage training pipeline for long-tail classification:
  Stage 1: Train full model with standard BCE/instance-balanced sampling
  Stage 2: Freeze backbone, re-train classifier with:
    (a) cRT — classifier re-training with class-balanced sampling
    (b) tau-normalization — weight norm adjustment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from bioagentics.cxr_rare.config import OUTPUT_DIR, TrainConfig
from bioagentics.cxr_rare.training.trainer import CXRTrainer, TrainResult

logger = logging.getLogger(__name__)


@dataclass
class DecoupledConfig:
    """Configuration for decoupled training."""
    stage1_config: TrainConfig = field(default_factory=TrainConfig)
    stage2_epochs: int = 10
    stage2_lr: float = 0.01
    tau: float = 1.0  # tau-normalization temperature
    rebalance_method: str = "crt"  # "crt" or "tau_norm"


def make_class_balanced_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Create a sampler that balances class frequencies.

    For multi-label, weight each sample inversely proportional to
    the average frequency of its positive labels.
    """
    all_labels: list[np.ndarray] = []
    for i in range(len(dataset)):  # type: ignore[arg-type]
        sample = dataset[i]
        all_labels.append(sample["labels"].numpy() if hasattr(sample["labels"], "numpy") else np.array(sample["labels"]))

    label_matrix = np.stack(all_labels)
    # Per-class frequency
    class_freq = label_matrix.mean(axis=0) + 1e-8
    # Per-sample weight: mean of inverse frequencies of active labels
    sample_weights = []
    for i in range(len(label_matrix)):
        active = label_matrix[i] > 0
        if active.any():
            weight = (1.0 / class_freq[active]).mean()
        else:
            weight = 1.0
        sample_weights.append(weight)

    weights = torch.tensor(sample_weights, dtype=torch.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classification head.

    Expects models with a 'features' (or 'backbone') and 'head' (or 'fc') attribute.
    """
    # Try common patterns
    for name, param in model.named_parameters():
        if "head" in name or "fc" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Frozen backbone: %d/%d params trainable", trainable, total)


def tau_normalize(model: nn.Module, tau: float = 1.0) -> None:
    """Apply tau-normalization to classifier weights.

    Scales each class weight by ||w_i||^(tau-1), normalizing
    the effective magnitude across classes.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("head" in name or "fc" in name or "classifier" in name):
            with torch.no_grad():
                w = module.weight
                norms = w.norm(dim=1, keepdim=True)
                norms = norms.clamp(min=1e-8)
                module.weight.copy_(w * (norms ** (tau - 1)))
            logger.info("Applied tau-normalization (tau=%.2f) to %s", tau, name)
            return
    logger.warning("No classifier layer found for tau-normalization")


class DecoupledTrainer:
    """Two-stage decoupled training pipeline.

    Parameters
    ----------
    model : nn.Module
        Classification model with 'features' and 'head' attributes.
    train_dataset : Dataset
        Training dataset.
    val_dataset : Dataset
        Validation dataset.
    config : DecoupledConfig
        Decoupled training configuration.
    output_dir : Path
        Output directory.
    experiment_name : str
        Name prefix for outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: DecoupledConfig = DecoupledConfig(),
        output_dir: Path = OUTPUT_DIR,
        experiment_name: str = "decoupled",
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

    def train(self) -> TrainResult:
        """Run both stages of decoupled training."""
        # Stage 1: standard training (instance-balanced)
        logger.info("=== Stage 1: Representation learning (instance-balanced) ===")
        stage1_trainer = CXRTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            loss_fn=nn.BCEWithLogitsLoss(),
            config=self.config.stage1_config,
            output_dir=self.output_dir,
            experiment_name=f"{self.experiment_name}_stage1",
        )
        stage1_result = stage1_trainer.train()
        stage1_trainer.load_best_checkpoint()

        # Stage 2: re-balance classifier
        if self.config.rebalance_method == "crt":
            return self._stage2_crt(stage1_result)
        elif self.config.rebalance_method == "tau_norm":
            return self._stage2_tau_norm(stage1_result)
        else:
            raise ValueError(f"Unknown rebalance method: {self.config.rebalance_method}")

    def _stage2_crt(self, stage1_result: TrainResult) -> TrainResult:
        """Stage 2: classifier re-training with class-balanced sampling."""
        logger.info("=== Stage 2: cRT (class-balanced re-training) ===")
        freeze_backbone(self.model)

        sampler = make_class_balanced_sampler(self.train_dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.stage2_lr,
            momentum=0.9,
        )

        config2 = TrainConfig(
            epochs=self.config.stage2_epochs,
            batch_size=self.config.stage1_config.batch_size,
            num_workers=self.config.stage1_config.num_workers,
            learning_rate=self.config.stage2_lr,
            early_stopping_patience=self.config.stage2_epochs,  # no early stopping
            mixed_precision=self.config.stage1_config.mixed_precision,
        )

        trainer = CXRTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            loss_fn=nn.BCEWithLogitsLoss(),
            config=config2,
            output_dir=self.output_dir,
            experiment_name=f"{self.experiment_name}_crt",
            optimizer=optimizer,
        )
        # Replace train loader with balanced sampler
        trainer.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config2.batch_size,
            sampler=sampler,
            num_workers=config2.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        stage2_result = trainer.train()
        stage2_result.train_losses = stage1_result.train_losses + stage2_result.train_losses
        return stage2_result

    def _stage2_tau_norm(self, stage1_result: TrainResult) -> TrainResult:
        """Stage 2: tau-normalization (no additional training)."""
        logger.info("=== Stage 2: tau-normalization (tau=%.2f) ===", self.config.tau)
        tau_normalize(self.model, self.config.tau)

        # Re-evaluate with adjusted weights
        from bioagentics.cxr_rare.evaluation.metrics import evaluate
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.stage1_config.batch_size,
            shuffle=False,
            num_workers=self.config.stage1_config.num_workers,
        )

        all_labels, all_scores = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                logits = self.model(images)
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(batch["labels"].numpy())

        if all_labels:
            y_true = np.concatenate(all_labels)
            y_score = np.concatenate(all_scores)
            val_result = evaluate(y_true, y_score)
            stage1_result.val_metrics.append(val_result)
            stage1_result.best_metric = val_result["summary"]["macro_auroc"]

        return stage1_result
