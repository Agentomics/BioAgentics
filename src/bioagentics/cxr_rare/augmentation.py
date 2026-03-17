"""Tail-class data augmentation for long-tail CXR classification.

Augmentation strategies that preferentially target rare findings:
  - TailBiasedCutMix: CutMix that preferentially mixes tail-class images
  - MosaicAugmentation: Combines 4 images with tail-class bias
  - ClassBalancedSampler: Oversamples rare findings
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from bioagentics.cxr_rare.config import TAIL_CLASSES, LABEL_TO_INDEX

logger = logging.getLogger(__name__)


class ClassBalancedSampler(Sampler[int]):
    """Sampler that oversamples images containing tail-class labels.

    Parameters
    ----------
    dataset : Dataset
        Dataset returning dict with 'labels' key.
    tail_classes : list[str]
        Names of tail classes to oversample.
    oversample_factor : float
        Factor by which to increase tail-class sample probability.
    """

    def __init__(
        self,
        dataset: Dataset,
        tail_classes: list[str] | None = None,
        oversample_factor: float = 3.0,
    ) -> None:
        self.dataset = dataset
        self.n = len(dataset)  # type: ignore[arg-type]
        tail = tail_classes or TAIL_CLASSES
        tail_indices = {LABEL_TO_INDEX[c] for c in tail if c in LABEL_TO_INDEX}

        weights = np.ones(self.n, dtype=np.float64)
        for i in range(self.n):
            sample = dataset[i]
            labels = sample["labels"]
            if hasattr(labels, "numpy"):
                labels = labels.numpy()
            for idx in tail_indices:
                if idx < len(labels) and labels[idx] > 0:
                    weights[i] = oversample_factor
                    break

        self._weights = weights / weights.sum()

    def __iter__(self):
        indices = np.random.choice(self.n, size=self.n, replace=True, p=self._weights)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.n


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    tail_bias: bool = True,
    tail_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply CutMix augmentation to a batch, biased toward tail classes.

    Parameters
    ----------
    images : Tensor, shape (N, C, H, W)
    labels : Tensor, shape (N, num_classes)
    alpha : float
        Beta distribution parameter for mixing ratio.
    tail_bias : bool
        If True, preferentially select tail-class samples as mix partners.
    tail_indices : list[int], optional
        Label indices of tail classes.

    Returns
    -------
    mixed_images, mixed_labels : Tensors
    """
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    if tail_bias and tail_indices is not None:
        # Prefer mixing with samples that have tail-class labels
        tail_mask = labels[:, tail_indices].sum(dim=1) > 0
        if tail_mask.any():
            tail_idx = torch.where(tail_mask)[0]
            # Randomly select from tail samples
            perm = tail_idx[torch.randint(len(tail_idx), (batch_size,))]
        else:
            perm = torch.randperm(batch_size)
    else:
        perm = torch.randperm(batch_size)

    # Random box
    _, _, h, w = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = np.random.randint(h)
    cx = np.random.randint(w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

    # Mix labels proportionally
    area_ratio = (y2 - y1) * (x2 - x1) / (h * w)
    mixed_labels = labels * (1 - area_ratio) + labels[perm] * area_ratio

    return mixed_images, mixed_labels


def mosaic_augmentation(
    images: list[torch.Tensor],
    labels: list[torch.Tensor],
    output_size: int = 224,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine 4 images into a mosaic, mixing labels proportionally.

    Parameters
    ----------
    images : list of 4 Tensors, each (C, H, W)
    labels : list of 4 Tensors, each (num_classes,)
    output_size : int
        Output image dimension.

    Returns
    -------
    mosaic_image (C, output_size, output_size), combined_labels
    """
    assert len(images) == 4 and len(labels) == 4

    c = images[0].size(0)
    mosaic = torch.zeros(c, output_size, output_size, dtype=images[0].dtype)
    combined_labels = torch.zeros_like(labels[0])

    mid_y = output_size // 2
    mid_x = output_size // 2

    regions = [
        (0, 0, mid_y, mid_x),         # top-left
        (0, mid_x, mid_y, output_size),  # top-right
        (mid_y, 0, output_size, mid_x),  # bottom-left
        (mid_y, mid_x, output_size, output_size),  # bottom-right
    ]

    total_area = output_size * output_size
    for i, (y1, x1, y2, x2) in enumerate(regions):
        img = images[i]
        # Resize to fit region
        rh, rw = y2 - y1, x2 - x1
        resized = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False
        ).squeeze(0)
        mosaic[:, y1:y2, x1:x2] = resized
        area_frac = (rh * rw) / total_area
        combined_labels += labels[i] * area_frac

    return mosaic, combined_labels


class TailBiasedMosaicSampler:
    """Creates mosaic samples biased toward tail-class images.

    Parameters
    ----------
    dataset : Dataset
        Source dataset.
    tail_classes : list[str], optional
        Tail class names.
    """

    def __init__(
        self,
        dataset: Dataset,
        tail_classes: list[str] | None = None,
    ) -> None:
        self.dataset = dataset
        tail = tail_classes or TAIL_CLASSES
        tail_idx = {LABEL_TO_INDEX[c] for c in tail if c in LABEL_TO_INDEX}

        self._tail_samples: list[int] = []
        self._all_samples: list[int] = list(range(len(dataset)))  # type: ignore[arg-type]

        for i in self._all_samples:
            sample = dataset[i]
            labels = sample["labels"]
            if hasattr(labels, "numpy"):
                labels = labels.numpy()
            for idx in tail_idx:
                if idx < len(labels) and labels[idx] > 0:
                    self._tail_samples.append(i)
                    break

    def sample_mosaic(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample 4 images with tail-class bias and create a mosaic."""
        indices = []
        # At least 1 tail-class image if available
        if self._tail_samples:
            indices.append(np.random.choice(self._tail_samples))
        # Fill remaining with random samples
        while len(indices) < 4:
            indices.append(np.random.choice(self._all_samples))
        np.random.shuffle(indices)

        images = []
        labels = []
        for idx in indices:
            sample = self.dataset[idx]
            images.append(sample["image"])
            labels.append(sample["labels"])

        return mosaic_augmentation(images, labels)
