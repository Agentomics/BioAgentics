"""Prototype-based classification with cosine classifier.

Class prototype learning for long-tail recognition:
  - Compute class-mean feature prototypes from training set
  - Cosine similarity classifier head instead of linear FC
  - Learnable temperature scaling
Works with frozen or fine-tuned backbones.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bioagentics.cxr_rare.config import NUM_CLASSES

logger = logging.getLogger(__name__)


class CosineClassifier(nn.Module):
    """Cosine similarity classifier with learnable temperature.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    num_classes : int
        Number of output classes.
    temperature : float
        Initial temperature for scaling cosine similarities.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = NUM_CLASSES,
        temperature: float = 16.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity * temperature as logits.

        Parameters
        ----------
        x : Tensor, shape (N, D)
            Feature vectors.

        Returns
        -------
        Tensor, shape (N, C)
            Logits (cosine similarity scaled by temperature).
        """
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return self.temperature * (x_norm @ w_norm.t())


class PrototypeCXRModel(nn.Module):
    """Backbone + cosine classifier model for prototype-based classification.

    Parameters
    ----------
    backbone : nn.Module
        Feature extractor (output shape: (N, feature_dim)).
    feature_dim : int
        Output dimension of the backbone.
    num_classes : int
        Number of classes.
    temperature : float
        Initial cosine classifier temperature.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = NUM_CLASSES,
        temperature: float = 16.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = CosineClassifier(feature_dim, num_classes, temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)


@torch.no_grad()
def compute_class_prototypes(
    model: PrototypeCXRModel,
    dataset: Dataset,
    num_classes: int = NUM_CLASSES,
    batch_size: int = 64,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute class-mean feature prototypes from training data.

    Parameters
    ----------
    model : PrototypeCXRModel
        Model with backbone and classifier.
    dataset : Dataset
        Training dataset.
    num_classes : int
        Number of classes.
    batch_size : int
        Batch size for feature extraction.
    num_workers : int
        DataLoader workers.
    device : torch.device, optional
        Compute device.

    Returns
    -------
    Tensor, shape (num_classes, feature_dim)
        Class-mean prototypes.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feature_sums: torch.Tensor | None = None
    class_counts = torch.zeros(num_classes, device=device)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        features = model.extract_features(images)

        if feature_sums is None:
            feature_sums = torch.zeros(num_classes, features.size(1), device=device)

        # Accumulate features per class
        for c in range(num_classes):
            mask = labels[:, c] > 0
            if mask.any():
                feature_sums[c] += features[mask].sum(dim=0)
                class_counts[c] += mask.sum()

    assert feature_sums is not None, "Empty dataset"
    # Mean prototype
    class_counts = class_counts.clamp(min=1)
    prototypes = feature_sums / class_counts.unsqueeze(1)

    logger.info(
        "Computed %d prototypes (avg samples/class: %.1f)",
        num_classes, class_counts.mean().item(),
    )
    return prototypes


def initialize_classifier_from_prototypes(
    model: PrototypeCXRModel,
    prototypes: torch.Tensor,
) -> None:
    """Initialize cosine classifier weights from class prototypes."""
    with torch.no_grad():
        model.classifier.weight.copy_(prototypes)
    logger.info("Initialized classifier from prototypes")
