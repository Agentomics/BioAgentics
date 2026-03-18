"""Training loop for DR screening models.

Features:
  1. Model factory via timm (EfficientNet-B0, MobileNetV3-Small, EfficientNet-B4, etc.)
  2. Cross-entropy loss with optional class weighting for DR grade imbalance
  3. AdamW optimizer with cosine annealing LR schedule
  4. Mixed-precision training (AMP)
  5. Early stopping on validation quadratic weighted kappa (QWK)
  6. Epoch logging: train/val loss, accuracy, QWK, AUC
  7. Checkpoint saving (best val QWK)

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.training \\
        --model efficientnet_b0 \\
        --splits data/diagnostics/smartphone-retinal-dr-screening/splits.csv \\
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bioagentics.diagnostics.retinal_dr_screening.augmentation import DRDataset
from bioagentics.diagnostics.retinal_dr_screening.config import (
    BATCH_SIZE,
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    REFERABLE_THRESHOLD,
    RESULTS_DIR,
    TRAIN_IMAGE_SIZE,
    WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)


# ── Model factory ──


def create_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Create a timm model with the correct number of output classes.

    Args:
        model_name: timm model name (e.g. efficientnet_b0, mobilenetv3_small_100).
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    logger.info(
        "Created model: %s (pretrained=%s, num_classes=%d, params=%.1fM)",
        model_name,
        pretrained,
        num_classes,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model


# ── Training config ──


@dataclass
class TrainConfig:
    model_name: str = "efficientnet_b0"
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    image_size: int = TRAIN_IMAGE_SIZE
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOPPING_PATIENCE
    use_amp: bool = True
    use_class_weights: bool = True
    device: str = ""
    seed: int = 42


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    val_qwk: float
    val_auc_referable: float
    lr: float
    time_s: float


@dataclass
class TrainResult:
    best_epoch: int
    best_val_qwk: float
    best_val_auc: float
    test_metrics: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


# ── Metrics ──


def compute_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
) -> dict:
    """Compute comprehensive classification metrics.

    Returns:
        Dict with accuracy, qwk, auc_referable, per-class sensitivity/specificity.
    """
    accuracy = float(np.mean(all_labels == all_preds))
    qwk = float(cohen_kappa_score(all_labels, all_preds, weights="quadratic"))

    # Referable DR AUC (grade >= REFERABLE_THRESHOLD)
    binary_labels = (all_labels >= REFERABLE_THRESHOLD).astype(int)
    referable_probs = all_probs[:, REFERABLE_THRESHOLD:].sum(axis=1)

    try:
        auc_referable = float(roc_auc_score(binary_labels, referable_probs))
    except ValueError:
        auc_referable = 0.0

    # Per-class sensitivity and specificity
    num_classes = all_probs.shape[1]
    per_class = {}
    for c in range(num_classes):
        tp = int(((all_labels == c) & (all_preds == c)).sum())
        fn = int(((all_labels == c) & (all_preds != c)).sum())
        fp = int(((all_labels != c) & (all_preds == c)).sum())
        tn = int(((all_labels != c) & (all_preds != c)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class[f"class_{c}_sensitivity"] = round(sensitivity, 4)
        per_class[f"class_{c}_specificity"] = round(specificity, 4)

    # Referable DR sensitivity/specificity
    ref_preds = (referable_probs >= 0.5).astype(int)
    ref_tp = int(((binary_labels == 1) & (ref_preds == 1)).sum())
    ref_fn = int(((binary_labels == 1) & (ref_preds == 0)).sum())
    ref_fp = int(((binary_labels == 0) & (ref_preds == 1)).sum())
    ref_tn = int(((binary_labels == 0) & (ref_preds == 0)).sum())

    return {
        "accuracy": round(accuracy, 4),
        "qwk": round(qwk, 4),
        "auc_referable": round(auc_referable, 4),
        "referable_sensitivity": round(ref_tp / (ref_tp + ref_fn) if (ref_tp + ref_fn) > 0 else 0.0, 4),
        "referable_specificity": round(ref_tn / (ref_tn + ref_fp) if (ref_tn + ref_fp) > 0 else 0.0, 4),
        **per_class,
    }


def compute_class_weights(dataset: DRDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced DR grades."""
    labels = [int(dataset.data.iloc[i]["dr_grade"]) for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    # Avoid division by zero
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES  # normalize to sum = NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


# ── Training loop ──


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (avg_loss, accuracy, all_labels, all_preds, all_probs)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def train(
    splits_csv: Path,
    config: TrainConfig | None = None,
    output_dir: Path | None = None,
) -> TrainResult:
    """Full training pipeline.

    Args:
        splits_csv: Path to the splits CSV.
        config: Training configuration.
        output_dir: Directory for checkpoints and logs.

    Returns:
        TrainResult with best metrics and history.
    """
    if config is None:
        config = TrainConfig()
    if output_dir is None:
        output_dir = MODEL_DIR / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if config.device:
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Datasets
    train_ds = DRDataset(
        splits_csv, split="train", transform="train",
        image_size=config.image_size, gradable_only=True,
    )
    val_ds = DRDataset(
        splits_csv, split="val", transform="val",
        image_size=config.image_size, gradable_only=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    # Model
    model = create_model(config.model_name, config.num_classes, config.pretrained)
    model = model.to(device)

    # Loss with optional class weights
    if config.use_class_weights and len(train_ds) > 0:
        weights = compute_class_weights(train_ds).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info("Class weights: %s", weights.cpu().tolist())
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    # AMP
    scaler = GradScaler() if config.use_amp and device.type == "cuda" else None

    # Training loop
    best_qwk = -1.0
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
        )

        val_loss, val_acc, val_labels, val_preds, val_probs = evaluate(
            model, val_loader, criterion, device,
        )

        metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_qwk = metrics["qwk"]
        val_auc = metrics["auc_referable"]

        scheduler.step()
        elapsed = time.time() - t0

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_qwk": round(val_qwk, 4),
            "val_auc_referable": round(val_auc, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "time_s": round(elapsed, 1),
        }
        history.append(epoch_data)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.4f "
            "val_qwk=%.4f val_auc=%.4f lr=%.6f (%.1fs)",
            epoch, config.max_epochs, train_loss, val_loss, val_acc,
            val_qwk, val_auc, scheduler.get_last_lr()[0], elapsed,
        )

        # Best model checkpoint
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": val_qwk,
                "val_auc": val_auc,
                "config": asdict(config),
            }, output_dir / "best_model.pt")
            logger.info("  → New best model (QWK=%.4f)", val_qwk)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, config.patience)
                break

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    result = TrainResult(
        best_epoch=best_epoch,
        best_val_qwk=best_qwk,
        best_val_auc=best_auc,
        history=history,
    )

    logger.info(
        "Training complete. Best epoch=%d, QWK=%.4f, AUC=%.4f",
        best_epoch, best_qwk, best_auc,
    )
    return result


def evaluate_test_set(
    model_path: Path,
    splits_csv: Path,
    split: str = "test",
    config: TrainConfig | None = None,
) -> dict:
    """Load best model and evaluate on test or external_val set.

    Returns:
        Dict of metrics including accuracy, QWK, AUC, per-class stats.
    """
    if config is None:
        config = TrainConfig()

    checkpoint = torch.load(model_path, weights_only=False)

    if "config" in checkpoint:
        saved_config = checkpoint["config"]
        config.model_name = saved_config.get("model_name", config.model_name)
        config.num_classes = saved_config.get("num_classes", config.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(config.model_name, config.num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    test_ds = DRDataset(
        splits_csv, split=split, transform="val",
        image_size=config.image_size, gradable_only=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, labels, preds, probs = evaluate(model, test_loader, criterion, device)

    metrics = compute_metrics(labels, preds, probs)
    metrics["test_loss"] = round(test_loss, 4)
    metrics["n_samples"] = len(labels)
    metrics["split"] = split

    logger.info(
        "%s set: acc=%.4f, qwk=%.4f, auc=%.4f, ref_sens=%.4f, ref_spec=%.4f",
        split, metrics["accuracy"], metrics["qwk"], metrics["auc_referable"],
        metrics["referable_sensitivity"], metrics["referable_specificity"],
    )

    # Save results
    results_dir = RESULTS_DIR / config.model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Train DR screening model")
    parser.add_argument(
        "--model", default="efficientnet_b0", help="timm model name",
    )
    parser.add_argument(
        "--splits", type=Path, default=DATA_DIR / "splits.csv",
    )
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--image-size", type=int, default=TRAIN_IMAGE_SIZE)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = TrainConfig(
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        use_amp=not args.no_amp,
        device=args.device,
    )

    if args.evaluate:
        model_path = MODEL_DIR / args.model / "best_model.pt"
        for split in ["test", "external_val"]:
            evaluate_test_set(model_path, args.splits, split=split, config=config)
    else:
        result = train(args.splits, config)
        print(f"\nBest epoch: {result.best_epoch}")
        print(f"Best val QWK: {result.best_val_qwk:.4f}")
        print(f"Best val AUC: {result.best_val_auc:.4f}")


if __name__ == "__main__":
    main()
