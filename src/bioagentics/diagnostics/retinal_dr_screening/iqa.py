"""Image Quality Assessment (IQA) pre-filter for smartphone DR screening.

Lightweight MobileNetV3-Small model that classifies fundus images as
gradable or ungradable before DR inference. Detects:
  - Blur / motion artifacts
  - Poor exposure (over/under-exposed)
  - Insufficient field-of-view / partial occlusion

Architecture: MobileNetV3-Small backbone → global avg pool → binary head.
Target: <100ms additional latency on mobile hardware.

Usage (training):
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.iqa \\
        --splits data/diagnostics/smartphone-retinal-dr-screening/splits.csv \\
        --epochs 30

Usage (inference):
    from bioagentics.diagnostics.retinal_dr_screening.iqa import IQAPredictor
    predictor = IQAPredictor("path/to/iqa_model.pt")
    result = predictor.assess(image_bgr)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bioagentics.diagnostics.retinal_dr_screening.config import (
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOBILE_IMAGE_SIZE,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)


# ── Quality failure categories ──

QUALITY_ISSUES = {
    "blur": "Image too blurry — hold camera steady and ensure focus lock",
    "dark": "Image too dark — increase room lighting or use flash",
    "bright": "Image overexposed — reduce lighting or move away from light source",
    "field_of_view": "Retina not fully visible — reposition lens adapter",
    "occlusion": "Image partially blocked — clean lens and remove obstructions",
}


# ── IQA Model ──


class IQAModel(nn.Module):
    """MobileNetV3-Small with binary quality classification head.

    Outputs two values: [ungradable_logit, gradable_logit].
    An additional quality_score (0-1) is derived from the gradable probability.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=0,  # remove classifier, keep features
        )
        # Probe actual feature dim (num_features can be stale in some timm versions)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feature_dim = self.backbone(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, 2): [ungradable, gradable]."""
        features = self.backbone(x)
        return self.head(features)


# ── Smartphone failure augmentations ──


def apply_motion_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Simulate motion blur from hand shake."""
    kernel = np.zeros((kernel_size, kernel_size))
    # Random angle
    angle = np.random.uniform(0, 180)
    radian = np.deg2rad(angle)
    cx, cy = kernel_size // 2, kernel_size // 2
    for i in range(kernel_size):
        x = int(cx + (i - cx) * np.cos(radian))
        y = int(cy + (i - cy) * np.sin(radian))
        x = np.clip(x, 0, kernel_size - 1)
        y = np.clip(y, 0, kernel_size - 1)
        kernel[y, x] = 1
    kernel = kernel / kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def apply_glare(image: np.ndarray, intensity: float = 0.6) -> np.ndarray:
    """Simulate lens glare / specular reflection."""
    h, w = image.shape[:2]
    # Random glare center
    cx = np.random.randint(w // 4, 3 * w // 4)
    cy = np.random.randint(h // 4, 3 * h // 4)
    radius = np.random.randint(min(h, w) // 8, min(h, w) // 3)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
    mask = np.clip(1.0 - dist / radius, 0, 1)
    mask = (mask * intensity * 255).astype(np.uint8)
    mask = np.stack([mask] * 3, axis=-1)

    return cv2.add(image, mask)


def apply_partial_occlusion(image: np.ndarray, coverage: float = 0.3) -> np.ndarray:
    """Simulate partial lens occlusion (finger, eyelid, etc.)."""
    h, w = image.shape[:2]
    mask = np.ones_like(image)

    # Random edge occlusion
    side = np.random.choice(["top", "bottom", "left", "right"])
    extent = int(max(h, w) * coverage)

    if side == "top":
        mask[:extent, :] = 0
    elif side == "bottom":
        mask[h - extent :, :] = 0
    elif side == "left":
        mask[:, :extent] = 0
    else:
        mask[:, w - extent :] = 0

    return (image * mask).astype(np.uint8)


def apply_underexposure(image: np.ndarray, factor: float = 0.3) -> np.ndarray:
    """Simulate underexposed (too dark) capture."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_overexposure(image: np.ndarray, factor: float = 1.8) -> np.ndarray:
    """Simulate overexposed (too bright) capture."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_defocus_blur(image: np.ndarray, radius: int = 7) -> np.ndarray:
    """Simulate out-of-focus capture (disk blur)."""
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(image, (ksize, ksize), radius)


# ── Dataset ──


@dataclass
class IQATrainConfig:
    """Training configuration for IQA model."""

    image_size: int = MOBILE_IMAGE_SIZE
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_epochs: int = 30
    patience: int = 5
    pretrained: bool = True
    dropout: float = 0.2
    # Probability of applying degradation to gradable images (creates synthetic ungradable)
    degradation_prob: float = 0.5
    device: str = ""
    seed: int = 42


class IQADataset(Dataset):
    """Dataset for IQA training.

    Uses the is_gradable column from splits CSV. Additionally applies
    synthetic degradations to gradable images to create more ungradable
    training samples (simulating smartphone capture failures).
    """

    def __init__(
        self,
        splits_csv: Path,
        split: str = "train",
        image_size: int = MOBILE_IMAGE_SIZE,
        degradation_prob: float = 0.5,
        augment: bool = True,
    ) -> None:
        import pandas as pd

        df = pd.read_csv(splits_csv)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.degradation_prob = degradation_prob if augment else 0.0
        self.augment = augment
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def _normalize(self, image: np.ndarray) -> torch.Tensor:
        """Resize, convert to RGB float tensor, normalize."""
        image = cv2.resize(image, (self.image_size, self.image_size))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for c in range(3):
            rgb[:, :, c] = (rgb[:, :, c] - self.mean[c]) / self.std[c]
        return torch.from_numpy(rgb.transpose(2, 0, 1))  # CHW

    def _apply_random_degradation(self, image: np.ndarray) -> np.ndarray:
        """Apply a random smartphone failure degradation."""
        degradation = np.random.choice([
            "motion_blur",
            "defocus",
            "underexposure",
            "overexposure",
            "glare",
            "occlusion",
        ])

        if degradation == "motion_blur":
            ks = np.random.randint(15, 35)
            return apply_motion_blur(image, kernel_size=ks)
        elif degradation == "defocus":
            r = np.random.randint(7, 15)
            return apply_defocus_blur(image, radius=r)
        elif degradation == "underexposure":
            f = np.random.uniform(0.15, 0.35)
            return apply_underexposure(image, factor=f)
        elif degradation == "overexposure":
            f = np.random.uniform(1.6, 2.5)
            return apply_overexposure(image, factor=f)
        elif degradation == "glare":
            i = np.random.uniform(0.4, 0.8)
            return apply_glare(image, intensity=i)
        else:  # occlusion
            c = np.random.uniform(0.2, 0.45)
            return apply_partial_occlusion(image, coverage=c)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        image = cv2.imread(str(row["image_path"]))
        if image is None:
            # Return a black image as fallback
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        is_gradable = bool(row.get("is_gradable", True))

        # For training: randomly degrade gradable images to create synthetic ungradable
        if self.augment and is_gradable and np.random.random() < self.degradation_prob:
            image = self._apply_random_degradation(image)
            is_gradable = False

        # Standard augmentation for training
        if self.augment:
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
            if np.random.random() > 0.5:
                image = cv2.flip(image, 0)

        tensor = self._normalize(image)
        label = 1 if is_gradable else 0

        return {"image": tensor, "label": label}


# ── Training ──


def train_iqa(
    splits_csv: Path,
    config: IQATrainConfig | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Train the IQA pre-filter model.

    Returns dict with best metrics.
    """
    if config is None:
        config = IQATrainConfig()
    if output_dir is None:
        output_dir = MODEL_DIR / "iqa"
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
    logger.info("IQA training on device: %s", device)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Datasets
    train_ds = IQADataset(
        splits_csv, split="train",
        image_size=config.image_size,
        degradation_prob=config.degradation_prob,
        augment=True,
    )
    val_ds = IQADataset(
        splits_csv, split="val",
        image_size=config.image_size,
        degradation_prob=0.0,
        augment=False,
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
    model = IQAModel(pretrained=config.pretrained, dropout=config.dropout)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("IQA model: %.1fM parameters", param_count)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs,
    )

    best_acc = 0.0
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_tn = val_fn = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(1)

                val_loss += loss.item() * images.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

                # Binary metrics (1=gradable, 0=ungradable)
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        scheduler.step()
        elapsed = time.time() - t0

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        sensitivity = val_tp / max(val_tp + val_fn, 1)
        specificity = val_tn / max(val_tn + val_fp, 1)

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(train_loss / max(train_total, 1), 4),
            "val_loss": round(val_loss / max(val_total, 1), 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "time_s": round(elapsed, 1),
        }
        history.append(epoch_data)

        logger.info(
            "Epoch %d/%d — val_acc=%.4f sens=%.4f spec=%.4f (%.1fs)",
            epoch, config.max_epochs, val_acc, sensitivity, specificity, elapsed,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "config": asdict(config),
            }, output_dir / "iqa_best.pt")
            logger.info("  → New best IQA model (acc=%.4f)", val_acc)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Save history
    with open(output_dir / "iqa_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "iqa_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    result = {
        "best_epoch": history[-1]["epoch"] if history else 0,
        "best_val_acc": best_acc,
        "history": history,
    }
    logger.info("IQA training complete. Best val_acc=%.4f", best_acc)
    return result


# ── Inference ──


@dataclass
class IQAResult:
    """Result from IQA assessment of a single image."""

    is_gradable: bool
    quality_score: float  # 0.0 (worst) to 1.0 (best)
    issues: list[str]  # list of detected quality issue keys
    guidance: list[str]  # human-readable recapture guidance messages
    latency_ms: float  # inference time


class IQAPredictor:
    """Inference wrapper for the IQA model.

    Loads a trained checkpoint and provides image quality assessment
    with recapture guidance.
    """

    def __init__(
        self,
        model_path: Path | str,
        device: str | None = None,
        quality_threshold: float = 0.5,
        image_size: int = MOBILE_IMAGE_SIZE,
    ) -> None:
        self.image_size = image_size
        self.quality_threshold = quality_threshold
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

        # Device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load model
        self.model = IQAModel(pretrained=False)
        checkpoint = torch.load(str(model_path), weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess BGR image to model input tensor."""
        resized = cv2.resize(image, (self.image_size, self.image_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for c in range(3):
            rgb[:, :, c] = (rgb[:, :, c] - self.mean[c]) / self.std[c]
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def _detect_issues(self, image: np.ndarray) -> list[str]:
        """Detect specific quality issues using heuristics on the raw image.

        Returns list of issue keys from QUALITY_ISSUES.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        issues = []

        # Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50.0:
            issues.append("blur")

        # Brightness (too dark or too bright)
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 30.0:
            issues.append("dark")
        elif mean_brightness > 220.0:
            issues.append("bright")

        # Field of view check: bright pixels should cover substantial area
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(thresh > 0) / thresh.size
        if bright_ratio < 0.3:
            issues.append("field_of_view")

        # Occlusion: check if a large edge region is uniformly dark
        h, w = gray.shape
        edge_regions = [
            gray[:h // 5, :],       # top
            gray[4 * h // 5:, :],   # bottom
            gray[:, :w // 5],       # left
            gray[:, 4 * w // 5:],   # right
        ]
        for region in edge_regions:
            if region.size > 0 and float(np.mean(region)) < 10.0:
                if "field_of_view" not in issues:
                    issues.append("occlusion")
                break

        return issues

    @torch.no_grad()
    def assess(self, image: np.ndarray) -> IQAResult:
        """Assess image quality and provide recapture guidance.

        Args:
            image: BGR uint8 image (any size).

        Returns:
            IQAResult with gradability decision, quality score, and guidance.
        """
        t0 = time.perf_counter()

        # Model inference
        tensor = self._preprocess(image)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        quality_score = float(probs[1])  # probability of gradable class

        # Issue detection
        issues = self._detect_issues(image)

        # Decision
        is_gradable = quality_score >= self.quality_threshold and len(issues) == 0

        # Build guidance messages
        guidance = [QUALITY_ISSUES[issue] for issue in issues]
        if not is_gradable and not guidance:
            guidance = ["Image quality insufficient — please recapture with better conditions"]

        latency_ms = (time.perf_counter() - t0) * 1000

        return IQAResult(
            is_gradable=is_gradable,
            quality_score=round(quality_score, 4),
            issues=issues,
            guidance=guidance,
            latency_ms=round(latency_ms, 1),
        )

    @torch.no_grad()
    def assess_batch(self, images: list[np.ndarray]) -> list[IQAResult]:
        """Assess a batch of images."""
        return [self.assess(img) for img in images]


# ── Combined inference pipeline ──


@dataclass
class DRScreeningResult:
    """Combined DR screening result with quality metadata."""

    # Quality assessment
    is_gradable: bool
    quality_score: float
    quality_issues: list[str]
    recapture_guidance: list[str]

    # DR prediction (only valid if is_gradable=True)
    dr_grade: int | None
    dr_grade_label: str | None
    dr_probabilities: list[float] | None
    is_referable: bool | None
    confidence: float | None


def run_screening_pipeline(
    image: np.ndarray,
    iqa_predictor: IQAPredictor,
    dr_model: nn.Module,
    device: torch.device | None = None,
    image_size: int = MOBILE_IMAGE_SIZE,
) -> DRScreeningResult:
    """Run the full screening pipeline: IQA check → DR grading.

    If IQA fails, returns result with quality guidance and no DR prediction.
    If IQA passes, runs DR model and returns full result.
    """
    from bioagentics.diagnostics.retinal_dr_screening.config import DR_CLASSES, REFERABLE_THRESHOLD

    # Step 1: Quality check
    iqa_result = iqa_predictor.assess(image)

    if not iqa_result.is_gradable:
        return DRScreeningResult(
            is_gradable=False,
            quality_score=iqa_result.quality_score,
            quality_issues=iqa_result.issues,
            recapture_guidance=iqa_result.guidance,
            dr_grade=None,
            dr_grade_label=None,
            dr_probabilities=None,
            is_referable=None,
            confidence=None,
        )

    # Step 2: DR inference
    if device is None:
        device = iqa_predictor.device

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    resized = cv2.resize(image, (image_size, image_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for c in range(3):
        rgb[:, :, c] = (rgb[:, :, c] - mean[c]) / std[c]
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    dr_model.eval()
    with torch.no_grad():
        logits = dr_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    dr_grade = int(probs.argmax())
    dr_probs = [round(float(p), 4) for p in probs]
    confidence = float(probs.max())
    is_referable = dr_grade >= REFERABLE_THRESHOLD

    return DRScreeningResult(
        is_gradable=True,
        quality_score=iqa_result.quality_score,
        quality_issues=iqa_result.issues,
        recapture_guidance=[],
        dr_grade=dr_grade,
        dr_grade_label=DR_CLASSES.get(dr_grade, f"Grade {dr_grade}"),
        dr_probabilities=dr_probs,
        is_referable=is_referable,
        confidence=round(confidence, 4),
    )


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Train IQA pre-filter model")
    parser.add_argument(
        "--splits", type=Path, default=DATA_DIR / "splits.csv",
        help="Path to splits CSV",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=MOBILE_IMAGE_SIZE)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = IQATrainConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        device=args.device,
    )

    result = train_iqa(args.splits, config)
    print(f"\nBest val accuracy: {result['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
