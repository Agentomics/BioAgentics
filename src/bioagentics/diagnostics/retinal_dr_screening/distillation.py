"""Knowledge distillation: EfficientNet-B4 teacher → MobileNetV3-Small student.

Implements:
  1. Frozen teacher forward pass for soft labels
  2. Combined loss = alpha * CE(student, hard) + (1-alpha) * KL(student_soft, teacher_soft)
  3. Temperature-scaled softmax for soft label generation
  4. Hyperparameter search over alpha and T

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.distillation \\
        --teacher-checkpoint output/diagnostics/.../efficientnet_b4/best_model.pt \\
        --splits data/diagnostics/smartphone-retinal-dr-screening/splits.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bioagentics.diagnostics.retinal_dr_screening.augmentation import DRDataset
from bioagentics.diagnostics.retinal_dr_screening.config import (
    BATCH_SIZE,
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    KD_ALPHA,
    KD_TEMPERATURE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    TRAIN_IMAGE_SIZE,
    WEIGHT_DECAY,
)
from bioagentics.diagnostics.retinal_dr_screening.training import (
    compute_metrics,
    create_model,
    evaluate,
)

logger = logging.getLogger(__name__)


@dataclass
class DistillConfig:
    teacher_model: str = "efficientnet_b4"
    student_model: str = "mobilenetv3_small_100"
    num_classes: int = NUM_CLASSES
    image_size: int = TRAIN_IMAGE_SIZE
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOPPING_PATIENCE
    temperature: float = KD_TEMPERATURE
    alpha: float = KD_ALPHA  # weight for distillation loss
    device: str = ""
    seed: int = 42


class DistillationLoss(nn.Module):
    """Combined hard-label CE + soft-label KL divergence loss."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute distillation loss.

        Args:
            student_logits: Raw logits from student model.
            teacher_logits: Raw logits from teacher model.
            hard_labels: Ground truth integer labels.

        Returns:
            Tuple of (total_loss, ce_component, kd_component).
        """
        # Hard label loss
        ce = self.ce_loss(student_logits, hard_labels)

        # Soft label KL divergence
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        kd = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        total = self.alpha * kd + (1 - self.alpha) * ce
        return total, ce, kd


@torch.no_grad()
def get_teacher_logits(
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[torch.Tensor]:
    """Pre-compute teacher logits for the entire dataset."""
    teacher.eval()
    all_logits = []
    for batch in loader:
        images = batch["image"].to(device)
        logits = teacher(images)
        all_logits.append(logits.cpu())
    return all_logits


def distill(
    splits_csv: Path,
    teacher_checkpoint: Path,
    config: DistillConfig | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run knowledge distillation training.

    Args:
        splits_csv: Path to splits CSV.
        teacher_checkpoint: Path to trained teacher model checkpoint.
        config: Distillation configuration.
        output_dir: Output directory for student model and logs.

    Returns:
        Dict with best metrics and training history.
    """
    if config is None:
        config = DistillConfig()
    if output_dir is None:
        output_dir = MODEL_DIR / f"{config.student_model}_distilled"
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

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load teacher
    teacher = create_model(config.teacher_model, config.num_classes, pretrained=False)
    checkpoint = torch.load(teacher_checkpoint, weights_only=False, map_location=device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    logger.info("Teacher loaded: %s", config.teacher_model)

    # Student
    student = create_model(config.student_model, config.num_classes, pretrained=True)
    student = student.to(device)

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

    # Loss, optimizer, scheduler
    criterion = DistillationLoss(config.temperature, config.alpha)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    # Training loop
    best_qwk = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()
        student.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        n_samples = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

            loss, ce, kd = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_ce += ce.item() * bs
            total_kd += kd.item() * bs
            n_samples += bs

        train_loss = total_loss / n_samples
        train_ce = total_ce / n_samples
        train_kd = total_kd / n_samples

        # Validation
        val_loss, val_acc, val_labels, val_preds, val_probs = evaluate(
            student, val_loader, val_criterion, device,
        )
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_qwk = metrics["qwk"]

        scheduler.step()
        elapsed = time.time() - t0

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_ce": round(train_ce, 4),
            "train_kd": round(train_kd, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_qwk": round(val_qwk, 4),
            "val_auc_referable": round(metrics["auc_referable"], 4),
            "time_s": round(elapsed, 1),
        }
        history.append(epoch_data)

        logger.info(
            "Epoch %d/%d — loss=%.4f (ce=%.4f kd=%.4f) val_qwk=%.4f (%.1fs)",
            epoch, config.max_epochs, train_loss, train_ce, train_kd,
            val_qwk, elapsed,
        )

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": val_qwk,
                "config": asdict(config),
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Save logs
    with open(output_dir / "distill_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "distill_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info("Distillation complete. Best epoch=%d, QWK=%.4f", best_epoch, best_qwk)
    return {
        "best_epoch": best_epoch,
        "best_val_qwk": best_qwk,
        "history": history,
    }


def search_hyperparameters(
    splits_csv: Path,
    teacher_checkpoint: Path,
    alphas: list[float] | None = None,
    temperatures: list[float] | None = None,
    max_epochs: int = 10,
    device: str = "",
) -> list[dict]:
    """Grid search over alpha and temperature.

    Args:
        splits_csv: Path to splits CSV.
        teacher_checkpoint: Path to teacher checkpoint.
        alphas: List of alpha values to try.
        temperatures: List of temperature values to try.
        max_epochs: Epochs per trial (reduced for speed).
        device: Device string.

    Returns:
        List of dicts with config and best_val_qwk for each trial.
    """
    if alphas is None:
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    if temperatures is None:
        temperatures = [2.0, 4.0, 8.0, 16.0]

    results = []
    for alpha in alphas:
        for temp in temperatures:
            config = DistillConfig(
                alpha=alpha,
                temperature=temp,
                max_epochs=max_epochs,
                patience=max_epochs,  # No early stopping during search
                device=device,
            )
            logger.info("Trial: alpha=%.1f, T=%.1f", alpha, temp)
            output_dir = MODEL_DIR / f"distill_search/a{alpha}_t{temp}"
            result = distill(splits_csv, teacher_checkpoint, config, output_dir)
            results.append({
                "alpha": alpha,
                "temperature": temp,
                "best_val_qwk": result["best_val_qwk"],
                "best_epoch": result["best_epoch"],
            })
            logger.info("  → QWK=%.4f", result["best_val_qwk"])

    # Sort by best QWK
    results.sort(key=lambda x: x["best_val_qwk"], reverse=True)
    logger.info("Best: alpha=%.1f, T=%.1f, QWK=%.4f",
                results[0]["alpha"], results[0]["temperature"], results[0]["best_val_qwk"])
    return results


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation for DR screening")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=DATA_DIR / "splits.csv")
    parser.add_argument("--alpha", type=float, default=KD_ALPHA)
    parser.add_argument("--temperature", type=float, default=KD_TEMPERATURE)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--search", action="store_true", help="Run hyperparameter search")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.search:
        results = search_hyperparameters(
            args.splits, args.teacher_checkpoint,
            max_epochs=min(10, args.epochs), device=args.device,
        )
        print("\nSearch results:")
        for r in results:
            print(f"  alpha={r['alpha']:.1f} T={r['temperature']:.1f} → QWK={r['best_val_qwk']:.4f}")
    else:
        config = DistillConfig(
            alpha=args.alpha, temperature=args.temperature,
            max_epochs=args.epochs, device=args.device,
        )
        result = distill(args.splits, args.teacher_checkpoint, config)
        print(f"\nBest epoch: {result['best_epoch']}, QWK: {result['best_val_qwk']:.4f}")


if __name__ == "__main__":
    main()
