"""DenseNet-121 baseline for CXR long-tail multi-label classification.

ImageNet-pretrained DenseNet-121 with multi-label sigmoid output head.
Primary baseline for comparison with long-tail classification methods.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from bioagentics.cxr_rare.config import (
    BASELINES_DIR,
    DEFAULT_TRAIN_CONFIG,
    MIMIC_CXR_DIR,
    NUM_CLASSES,
    TrainConfig,
)
from bioagentics.cxr_rare.datasets.mimic_cxr import MIMICCXRDataset
from bioagentics.cxr_rare.evaluation.metrics import evaluate_and_save
from bioagentics.cxr_rare.training.trainer import CXRTrainer

logger = logging.getLogger(__name__)


class DenseNet121CXR(nn.Module):
    """DenseNet-121 with multi-label classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes (multi-label).
    pretrained : bool
        Use ImageNet-pretrained weights.
    dropout : float
        Dropout before classifier head.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.features = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (N, num_classes)."""
        features = self.features(x)
        return self.head(features)


def train_densenet_baseline(
    data_dir: Path = MIMIC_CXR_DIR,
    output_dir: Path = BASELINES_DIR,
    config: TrainConfig | None = None,
) -> dict:
    """Train DenseNet-121 baseline with standard BCE.

    Parameters
    ----------
    data_dir : Path
        MIMIC-CXR data directory.
    output_dir : Path
        Output directory for results.
    config : TrainConfig, optional
        Training config override.

    Returns
    -------
    dict
        Evaluation results on test set.
    """
    config = config or DEFAULT_TRAIN_CONFIG
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets from %s", data_dir)
    train_ds = MIMICCXRDataset(data_dir=data_dir, split="train")
    val_ds = MIMICCXRDataset(data_dir=data_dir, split="validate")
    test_ds = MIMICCXRDataset(data_dir=data_dir, split="test")

    if len(train_ds) == 0:
        logger.warning("Training dataset is empty — data may not be downloaded yet.")
        return {}

    model = DenseNet121CXR(num_classes=NUM_CLASSES, pretrained=True, dropout=0.2)
    loss_fn = nn.BCEWithLogitsLoss()

    trainer = CXRTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=loss_fn,
        config=config,
        output_dir=output_dir,
        experiment_name="densenet121_bce_baseline",
    )

    result = trainer.train()
    logger.info(
        "Training done: best_epoch=%d, best_macro_auroc=%.4f",
        result.best_epoch, result.best_metric,
    )

    # Evaluate on test set with best checkpoint
    trainer.load_best_checkpoint()

    import numpy as np
    all_labels, all_scores = [], []
    device = trainer.device
    model.eval()
    with torch.no_grad():
        for batch in trainer.val_loader.__class__(
            test_ds, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
        ):
            images = batch["image"].to(device)
            logits = model(images)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    if all_labels:
        y_true = np.concatenate(all_labels)
        y_score = np.concatenate(all_scores)
        return evaluate_and_save(y_true, y_score, output_dir, "densenet121_bce_baseline")

    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet-121 CXR baseline")
    parser.add_argument("--data-dir", type=Path, default=MIMIC_CXR_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASELINES_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    train_densenet_baseline(args.data_dir, args.output_dir, config)
