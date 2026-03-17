"""Class-balanced loss functions for long-tail multi-label classification.

Drop-in replacements for BCE in the shared training pipeline:
  - FocalLoss: Focal loss with configurable gamma
  - ClassBalancedFocalLoss: CB-focal with effective number weighting (Cui et al. 2019)
  - AsymmetricLoss: ASL for multi-label with separate pos/neg focusing
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-label focal loss.

    Reduces the contribution of well-classified examples,
    focusing training on hard/misclassified samples.

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 is equivalent to BCE.
    reduction : str
        'mean', 'sum', or 'none'.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor, shape (N, C)
            Raw model output (before sigmoid).
        targets : Tensor, shape (N, C)
            Binary ground truth labels.
        """
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = probability of correct class
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss (Cui et al. 2019).

    Weights each class by the inverse effective number of samples:
      E_n = (1 - beta^n) / (1 - beta)
    where n is the sample count and beta = (N-1)/N for total N.

    Parameters
    ----------
    class_counts : list[int]
        Per-class positive sample counts.
    beta : float, optional
        Effective number hyperparameter. Default: (N-1)/N.
    gamma : float
        Focal loss focusing parameter.
    reduction : str
        'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        class_counts: list[int],
        beta: float | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        counts = torch.tensor(class_counts, dtype=torch.float32)
        total = counts.sum()
        if beta is None:
            beta = (total - 1.0) / total if total > 1 else 0.999

        # Effective number per class
        effective_num = (1.0 - beta ** counts) / (1.0 - beta)
        weights = 1.0 / (effective_num + 1e-8)
        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        # Apply class weights (broadcast across batch)
        loss = self.weights.unsqueeze(0) * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification.

    Applies different focusing parameters for positive and negative
    samples, with probability shifting to suppress easy negatives.

    Parameters
    ----------
    gamma_pos : float
        Focusing parameter for positive samples.
    gamma_neg : float
        Focusing parameter for negative samples (typically higher).
    clip : float
        Probability shifting threshold for negatives.
    reduction : str
        'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Probability shifting for negatives
        probs_neg = (probs + self.clip).clamp(max=1.0)

        # Separate positive and negative log-probabilities
        log_pos = torch.log(probs.clamp(min=1e-8))
        log_neg = torch.log((1 - probs_neg).clamp(min=1e-8))

        # Asymmetric focusing
        pos_loss = targets * (1 - probs) ** self.gamma_pos * (-log_pos)
        neg_loss = (1 - targets) * probs_neg ** self.gamma_neg * (-log_neg)
        loss = pos_loss + neg_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
