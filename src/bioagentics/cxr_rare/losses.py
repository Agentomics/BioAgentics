"""Class-balanced loss functions for long-tail multi-label classification.

Drop-in replacements for BCE in the shared training pipeline:
  - FocalLoss: Focal loss with configurable gamma
  - ClassBalancedFocalLoss: CB-focal with effective number weighting (Cui et al. 2019)
  - AsymmetricLoss: ASL for multi-label with separate pos/neg focusing
  - LDAMLoss: Label-Distribution-Aware Margin loss (Cao et al. 2019)
  - LDAMDRW: LDAM with Deferred Re-Weighting schedule
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


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin (LDAM) loss for multi-label classification.

    Enforces larger classification margins for rare classes:
      margin_j = C / n_j^(1/4)
    where n_j is the sample count for class j and C is a scaling constant.

    For multi-label: applies margin to the logit of each positive class
    independently, then computes BCE.

    Reference: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware
    Margin Loss", NeurIPS 2019.

    Parameters
    ----------
    class_counts : list[int]
        Per-class positive sample counts.
    max_margin : float
        Maximum margin C. Margins are scaled as C / n_j^(1/4).
    class_weights : torch.Tensor, optional
        Per-class weights for the loss. If None, uniform weights.
    reduction : str
        'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        class_counts: list[int],
        max_margin: float = 0.5,
        class_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

        # Compute per-class margins: C / n_j^(1/4)
        counts = torch.tensor(class_counts, dtype=torch.float32).clamp(min=1)
        margins = max_margin / counts.pow(0.25)
        self.register_buffer("margins", margins)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor, shape (N, C)
            Raw model output (before sigmoid).
        targets : Tensor, shape (N, C)
            Binary ground truth labels.
        """
        # Shift logits for positive classes: subtract margin
        # For negative classes: add margin (encourages separation)
        margin_shift = self.margins.unsqueeze(0)  # (1, C)
        adjusted_logits = logits - targets * margin_shift + (1 - targets) * margin_shift

        loss = F.binary_cross_entropy_with_logits(
            adjusted_logits, targets, reduction="none"
        )

        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LDAMDRW(nn.Module):
    """LDAM with Deferred Re-Weighting (DRW) schedule.

    During training, switches from uniform weighting to class-balanced
    weighting at a configurable epoch. This lets the model learn good
    representations first (uniform) before adjusting for class imbalance.

    Per Sulake (arXiv:2603.02294), LDAM-DRW achieved 0.3950 mAP on
    CXR-LT 2026 (5th/68 teams).

    Usage:
        loss_fn = LDAMDRW(class_counts, drw_start_epoch=5)
        for epoch in range(epochs):
            loss_fn.set_epoch(epoch)
            ...  # normal training loop

    Parameters
    ----------
    class_counts : list[int]
        Per-class positive sample counts.
    max_margin : float
        Maximum LDAM margin.
    drw_start_epoch : int
        Epoch at which to switch from uniform to class-balanced weights.
    beta : float, optional
        Effective number beta for class-balanced weights.
        Default: (N-1)/N where N is total samples.
    reduction : str
        'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        class_counts: list[int],
        max_margin: float = 0.5,
        drw_start_epoch: int = 5,
        beta: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.drw_start_epoch = drw_start_epoch
        self.reduction = reduction
        self._current_epoch = 0

        # Compute class-balanced weights for DRW phase
        counts = torch.tensor(class_counts, dtype=torch.float32).clamp(min=1)
        total = counts.sum().item()
        if beta is None:
            beta = (total - 1.0) / total if total > 1 else 0.999

        effective_num = (1.0 - beta ** counts) / (1.0 - beta)
        cb_weights = 1.0 / (effective_num + 1e-8)
        cb_weights = cb_weights / cb_weights.mean()  # normalize to mean=1
        self.register_buffer("cb_weights", cb_weights)

        # Initialize LDAM with uniform weights (pre-DRW phase)
        self.ldam = LDAMLoss(
            class_counts=class_counts,
            max_margin=max_margin,
            class_weights=None,
            reduction=reduction,
        )

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for DRW scheduling."""
        self._current_epoch = epoch
        if epoch >= self.drw_start_epoch:
            self.ldam.class_weights = self.cb_weights  # type: ignore[assignment]
        else:
            self.ldam.class_weights = None

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def is_reweighting(self) -> bool:
        """Whether DRW class-balanced weighting is active."""
        return self._current_epoch >= self.drw_start_epoch

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ldam(logits, targets)
