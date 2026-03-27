"""ConvNeXt-Large baseline for CXR long-tail multi-label classification.

ImageNet-pretrained ConvNeXt-Large with multi-label sigmoid output head.
Per CXR-LT 2026 results, ConvNeXt-Large is the best single-model CNN
backbone (mAP 0.5220), providing a stronger baseline than DenseNet-121
or ResNet-50.
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


class ConvNeXtLargeCXR(nn.Module):
    """ConvNeXt-Large with multi-label classification head.

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
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_large(weights=weights)

        # ConvNeXt uses backbone.classifier = Sequential(LayerNorm, Flatten, Linear)
        # Extract feature dim from the original classifier's Linear layer
        in_features = backbone.classifier[2].in_features  # type: ignore[index]
        backbone.classifier = nn.Identity()

        self.features = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (N, num_classes)."""
        features = self.features(x)
        if features.dim() > 2:
            features = features.mean(dim=list(range(2, features.dim())))
        return self.head(features)


def train_convnext_baseline(
    data_dir: Path = MIMIC_CXR_DIR,
    output_dir: Path = BASELINES_DIR,
    config: TrainConfig | None = None,
) -> dict:
    """Train ConvNeXt-Large baseline with standard BCE."""
    config = config or DEFAULT_TRAIN_CONFIG
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MIMICCXRDataset(data_dir=data_dir, split="train")
    val_ds = MIMICCXRDataset(data_dir=data_dir, split="validate")
    test_ds = MIMICCXRDataset(data_dir=data_dir, split="test")

    if len(train_ds) == 0:
        logger.warning("Training dataset is empty — data may not be downloaded yet.")
        return {}

    model = ConvNeXtLargeCXR(num_classes=NUM_CLASSES, pretrained=True, dropout=0.2)
    loss_fn = nn.BCEWithLogitsLoss()

    trainer = CXRTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=loss_fn,
        config=config,
        output_dir=output_dir,
        experiment_name="convnext_large_bce_baseline",
    )

    result = trainer.train()
    trainer.load_best_checkpoint()

    import numpy as np
    all_labels, all_scores = [], []
    device = trainer.device
    model.eval()
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(
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
        return evaluate_and_save(y_true, y_score, output_dir, "convnext_large_bce_baseline")

    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Large CXR baseline")
    parser.add_argument("--data-dir", type=Path, default=MIMIC_CXR_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASELINES_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    train_convnext_baseline(args.data_dir, args.output_dir, config)
