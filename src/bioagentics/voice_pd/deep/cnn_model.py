"""MobileNetV2-based CNN for mel-spectrogram PD classification.

Lightweight architecture suitable for mobile deployment. Uses pretrained
MobileNetV2 backbone with a custom binary classification head.
"""

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class SpectrogramCNN(nn.Module):
    """MobileNetV2-based binary classifier on mel-spectrograms.

    Architecture:
        MobileNetV2 feature extractor (pretrained on ImageNet) ->
        Adaptive average pool -> Dropout -> Linear(1280 -> 1)

    Input: (batch, 3, n_mels, time_frames) tensor in [0, 1].
    Output: (batch, 1) logits.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        # Use the feature layers only (drop the original classifier)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(pretrained: bool = True, dropout: float = 0.3) -> SpectrogramCNN:
    """Factory function to create the spectrogram CNN."""
    model = SpectrogramCNN(pretrained=pretrained, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("SpectrogramCNN: %.2fM parameters", n_params / 1e6)
    return model
