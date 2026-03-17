"""Domain adaptation for cross-institutional CXR generalization.

Methods:
  - Gradient reversal layer (adversarial domain discriminator)
  - Batch normalization alignment (separate BN stats per institution)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.autograd import Function

from bioagentics.cxr_rare.config import NUM_CLASSES

logger = logging.getLogger(__name__)


class GradientReversalFunction(Function):
    """Gradient reversal layer for adversarial domain adaptation."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """Gradient reversal wrapper module."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    """Adversarial domain discriminator head.

    Parameters
    ----------
    in_features : int
        Feature dimension from backbone.
    hidden_dim : int
        Hidden layer dimension.
    num_domains : int
        Number of institutions/domains.
    alpha : float
        Gradient reversal scaling factor.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_domains: int = 2,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.grl = GradientReversal(alpha)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns domain logits after gradient reversal."""
        return self.net(self.grl(features))


class DomainAdaptiveModel(nn.Module):
    """Model with both task classifier and domain discriminator.

    Parameters
    ----------
    backbone : nn.Module
        Feature extractor.
    feature_dim : int
        Backbone output dimension.
    num_classes : int
        Number of task classes.
    num_domains : int
        Number of institutions.
    domain_alpha : float
        Gradient reversal strength.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = NUM_CLASSES,
        num_domains: int = 2,
        domain_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.domain_head = DomainDiscriminator(feature_dim, num_domains=num_domains, alpha=domain_alpha)

    def forward(
        self, x: torch.Tensor, return_domain: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        task_logits = self.classifier(features)
        if return_domain:
            domain_logits = self.domain_head(features)
            return task_logits, domain_logits
        return task_logits


class InstitutionBatchNorm(nn.Module):
    """Batch normalization with separate stats per institution.

    Maintains independent running mean/var for each institution,
    selected at forward time by a domain index.

    Parameters
    ----------
    num_features : int
        Number of features (channels).
    num_domains : int
        Number of institutions.
    """

    def __init__(self, num_features: int, num_domains: int = 2) -> None:
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(num_domains)])
        self._domain_idx = 0

    def set_domain(self, domain_idx: int) -> None:
        """Set active domain for forward pass."""
        self._domain_idx = domain_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bns[self._domain_idx](x)


def replace_bn_with_institution_bn(
    model: nn.Module,
    num_domains: int = 2,
) -> nn.Module:
    """Replace all BatchNorm2d layers with InstitutionBatchNorm.

    Returns the modified model.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], InstitutionBatchNorm(module.num_features, num_domains))
            count += 1

    logger.info("Replaced %d BatchNorm2d layers with InstitutionBatchNorm", count)
    return model


def set_institution_bn_domain(model: nn.Module, domain_idx: int) -> None:
    """Set the active domain for all InstitutionBatchNorm layers."""
    for module in model.modules():
        if isinstance(module, InstitutionBatchNorm):
            module.set_domain(domain_idx)
