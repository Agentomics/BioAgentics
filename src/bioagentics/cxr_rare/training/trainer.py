"""Configurable training loop for CXR multi-label classification.

Shared backbone for all experiments: accepts model, dataset, loss,
optimizer. Supports mixed-precision, per-epoch AUROC logging, and
best-model checkpointing by macro-AUROC.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bioagentics.cxr_rare.config import DEFAULT_TRAIN_CONFIG, OUTPUT_DIR, TrainConfig
from bioagentics.cxr_rare.evaluation.metrics import evaluate

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of a training run."""
    best_epoch: int = 0
    best_metric: float = 0.0
    train_losses: list[float] = field(default_factory=list)
    val_metrics: list[dict] = field(default_factory=list)
    total_time_s: float = 0.0


class CXRTrainer:
    """Configurable trainer for CXR multi-label classification.

    Parameters
    ----------
    model : nn.Module
        Classification model (outputs raw logits).
    train_dataset : Dataset
        Training dataset (returns dict with 'image' and 'labels').
    val_dataset : Dataset
        Validation dataset.
    loss_fn : nn.Module
        Loss function (takes logits, targets).
    config : TrainConfig
        Training hyperparameters.
    output_dir : Path
        Directory for checkpoints and logs.
    experiment_name : str
        Name for checkpoint files and logs.
    optimizer : torch.optim.Optimizer, optional
        If None, AdamW with config lr/wd is used.
    scheduler : optional
        LR scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        loss_fn: nn.Module | None = None,
        config: TrainConfig = DEFAULT_TRAIN_CONFIG,
        output_dir: Path = OUTPUT_DIR,
        experiment_name: str = "experiment",
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: object | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

        # Device selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = scheduler

        # pin_memory only benefits CUDA transfers
        use_pin_memory = self.device.type == "cuda"

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory,
        )

        # Mixed-precision only supported on CUDA
        self.scaler = torch.amp.GradScaler("cuda") if config.mixed_precision and self.device.type == "cuda" else None

    def train(self) -> TrainResult:
        """Run full training loop with early stopping."""
        result = TrainResult()
        best_metric = 0.0
        patience_counter = 0
        start_time = time.time()

        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(epoch)
            result.train_losses.append(train_loss)

            val_result = self._validate()
            result.val_metrics.append(val_result)

            metric = val_result["summary"]["macro_auroc"]
            if np.isnan(metric):
                metric = 0.0

            logger.info(
                "Epoch %d/%d — train_loss=%.4f, macro_auroc=%.4f, "
                "head=%.4f, body=%.4f, tail=%.4f",
                epoch, self.config.epochs, train_loss,
                metric,
                val_result["summary"]["head_mean"],
                val_result["summary"]["body_mean"],
                val_result["summary"]["tail_mean"],
            )

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    self.scheduler.step()  # type: ignore[union-attr]

            if metric > best_metric:
                best_metric = metric
                result.best_epoch = epoch
                result.best_metric = metric
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    ckpt_dir / f"{self.experiment_name}_best.pt",
                )
                logger.info("  New best model saved (macro_auroc=%.4f)", metric)
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        result.total_time_s = time.time() - start_time
        logger.info(
            "Training complete: best_epoch=%d, best_macro_auroc=%.4f, time=%.1fs",
            result.best_epoch, result.best_metric, result.total_time_s,
        )
        return result

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> dict:
        """Run validation and compute metrics."""
        self.model.eval()
        all_labels: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["labels"]

            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.numpy())

        if not all_labels:
            return {"per_class_auroc": {}, "summary": {
                "macro_auroc": float("nan"),
                "head_mean": float("nan"),
                "body_mean": float("nan"),
                "tail_mean": float("nan"),
            }}

        y_true = np.concatenate(all_labels, axis=0)
        y_score = np.concatenate(all_scores, axis=0)
        return evaluate(y_true, y_score)

    def load_best_checkpoint(self) -> None:
        """Load the best checkpoint saved during training."""
        ckpt_path = self.output_dir / "checkpoints" / f"{self.experiment_name}_best.pt"
        if ckpt_path.exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
            logger.info("Loaded best checkpoint: %s", ckpt_path)
        else:
            logger.warning("No checkpoint found: %s", ckpt_path)
