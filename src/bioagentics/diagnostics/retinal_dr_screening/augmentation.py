"""Training data augmentation and PyTorch Dataset for DR screening.

Augmentations (albumentations-based):
  1. Random horizontal/vertical flips
  2. Random rotation (0-360, fundus has no canonical orientation)
  3. Color jitter (brightness, contrast, saturation, hue)
  4. Gaussian blur (simulates phone camera noise/focus variation)
  5. Random resized crop
  6. CutMix/MixUp regularization (applied at batch level)

Usage:
    # Dataset
    from bioagentics.diagnostics.retinal_dr_screening.augmentation import DRDataset
    ds = DRDataset(splits_csv, split="train", transform="train")

    # Visualization
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.augmentation \\
        --splits data/diagnostics/smartphone-retinal-dr-screening/splits.csv \\
        --n 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from bioagentics.diagnostics.retinal_dr_screening.config import (
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOBILE_IMAGE_SIZE,
    TRAIN_IMAGE_SIZE,
)

logger = logging.getLogger(__name__)


# ── Augmentation transforms ──


def get_train_transform(image_size: int = TRAIN_IMAGE_SIZE) -> A.Compose:
    """Training augmentation pipeline for fundus images."""
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = TRAIN_IMAGE_SIZE) -> A.Compose:
    """Validation/test transform (deterministic, no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_mobile_transform(image_size: int = MOBILE_IMAGE_SIZE) -> A.Compose:
    """Mobile inference transform."""
    return get_val_transform(image_size)


# ── PyTorch Dataset ──


class DRDataset(Dataset):
    """PyTorch Dataset for DR screening images.

    Loads images from the splits CSV and applies augmentations on-the-fly.

    Args:
        splits_csv: Path to the splits CSV file.
        split: One of 'train', 'val', 'test', 'external_val'.
        transform: 'train' for training augmentations, 'val' for deterministic,
                   or an albumentations Compose object.
        image_size: Target image size.
        gradable_only: If True, filter to only gradable images.
    """

    def __init__(
        self,
        splits_csv: str | Path,
        split: str = "train",
        transform: str | A.Compose = "train",
        image_size: int = TRAIN_IMAGE_SIZE,
        gradable_only: bool = True,
    ):
        import pandas as pd

        df = pd.read_csv(splits_csv)
        self.data = df[df["split"] == split].reset_index(drop=True)

        if gradable_only and "is_gradable" in self.data.columns:
            self.data = self.data[self.data["is_gradable"]].reset_index(drop=True)

        if isinstance(transform, str):
            if transform == "train":
                self.transform = get_train_transform(image_size)
            else:
                self.transform = get_val_transform(image_size)
        else:
            self.transform = transform

        logger.info(
            "DRDataset(%s): %d images, image_size=%d",
            split,
            len(self.data),
            image_size,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        image_path = row["image_path"]

        # Load image (BGR → RGB)
        image = cv2.imread(str(image_path))
        if image is None:
            # Return a black image placeholder if file is missing
            logger.warning("Could not read image: %s", image_path)
            image = np.zeros(
                (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = int(row["dr_grade"])

        augmented = self.transform(image=image)
        image_tensor = augmented["image"]

        return {
            "image": image_tensor,
            "label": label,
            "dataset_source": row["dataset_source"],
        }


# ── Batch-level augmentation (CutMix / MixUp) ──


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix to a batch of images.

    Args:
        images: Batch tensor (B, C, H, W).
        labels: Label tensor (B,).
        alpha: Beta distribution parameter.

    Returns:
        Mixed images, labels_a, labels_b, lambda value.
    """
    lam = np.random.default_rng().beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    _, _, h, w = images.shape
    cut_h = int(h * np.sqrt(1 - lam))
    cut_w = int(w * np.sqrt(1 - lam))

    rng = np.random.default_rng()
    cx = int(rng.integers(0, w))
    cy = int(rng.integers(0, h))

    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda for actual area
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)

    return mixed, labels, labels[index], lam


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp to a batch of images.

    Args:
        images: Batch tensor (B, C, H, W).
        labels: Label tensor (B,).
        alpha: Beta distribution parameter.

    Returns:
        Mixed images, labels_a, labels_b, lambda value.
    """
    lam = np.random.default_rng().beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam


def cutmix_mixup_criterion(
    criterion: torch.nn.Module,
    outputs: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for CutMix/MixUp mixed samples."""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


# ── Visualization ──


def visualize_augmentations(
    splits_csv: Path,
    n_samples: int = 8,
    output_path: Path | None = None,
):
    """Generate a grid of augmented samples for visual sanity checking."""
    import matplotlib.pyplot as plt

    dataset = DRDataset(splits_csv, split="train", transform="train", gradable_only=False)

    if len(dataset) == 0:
        logger.warning("No images to visualize")
        return

    n_samples = min(n_samples, len(dataset))
    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    for i in range(n_samples):
        sample = dataset[i]
        img = sample["image"]

        # Denormalize for visualization
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_vis = img * std + mean
        img_vis = img_vis.clamp(0, 1).permute(1, 2, 0).numpy()

        # Show original (row 0) and augmented (row 1) — both are augmented
        # since dataset applies transform; show two different augmentations
        axes[0, i].imshow(img_vis)
        axes[0, i].set_title(f"Grade {sample['label']}")
        axes[0, i].axis("off")

        # Second augmented version
        sample2 = dataset[i]
        img2 = sample2["image"]
        img2_vis = img2 * std + mean
        img2_vis = img2_vis.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[1, i].imshow(img2_vis)
        axes[1, i].axis("off")

    plt.suptitle("Training Augmentations (two random augmentations per image)")
    plt.tight_layout()

    if output_path is None:
        from bioagentics.diagnostics.retinal_dr_screening.config import FIGURES_DIR

        output_path = FIGURES_DIR / "augmentation_samples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved augmentation visualization to %s", output_path)


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Visualize DR augmentations")
    parser.add_argument(
        "--splits",
        type=Path,
        default=DATA_DIR / "splits.csv",
        help="Path to splits CSV",
    )
    parser.add_argument("--n", type=int, default=8, help="Number of samples")
    parser.add_argument("--output", type=Path, default=None, help="Output image path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    visualize_augmentations(args.splits, n_samples=args.n, output_path=args.output)


if __name__ == "__main__":
    main()
