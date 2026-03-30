"""Grad-CAM visualization for composite spectrogram CNN.

Generates class-activation heatmaps showing which vowel regions in the
composite spectrogram contribute most to PD detection.  Literature predicts
vowel /u/ will show the strongest activation.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", IJCV 2020.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bioagentics.voice_pd.config import N_MELS
from bioagentics.voice_pd.deep.composite_spectrogram import VOWELS

log = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM for a CNN with a ``features`` attribute (MobileNetV2-style).

    Usage::

        cam = GradCAM(model)
        heatmap = cam(input_tensor)          # (H, W) in [0, 1]
        vowel_scores = cam.vowel_contributions(heatmap, n_mels=128)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or self._last_conv_layer()
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        self._hook_handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _last_conv_layer(self) -> torch.nn.Module:
        """Return the last convolutional block in model.features."""
        return self.model.features[-1]

    def _save_activation(self, _module: torch.nn.Module, _input: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module: torch.nn.Module, _grad_in: tuple, grad_out: tuple) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap for binary PD classification.

        Args:
            input_tensor: Shape (1, 3, H, W) — single composite spectrogram.

        Returns:
            2D numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        self.model.zero_grad()
        logits.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input spatial size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def release(self) -> None:
        """Remove hooks to allow garbage collection."""
        for h in self._hook_handles:
            h.remove()


def vowel_contributions(
    heatmap: np.ndarray,
    n_mels: int = N_MELS,
) -> dict[str, float]:
    """Compute mean Grad-CAM activation per vowel strip.

    The composite spectrogram stacks 5 vowels vertically, each n_mels
    rows tall.  This function computes the mean activation in each strip.

    Args:
        heatmap: 2D array of shape (n_mels * 5, time_frames).
        n_mels: Height of one vowel strip.

    Returns:
        Dict mapping vowel label to mean activation score.
    """
    scores = {}
    for i, vowel in enumerate(VOWELS):
        strip = heatmap[i * n_mels : (i + 1) * n_mels, :]
        scores[vowel] = float(strip.mean())
    return scores


def save_heatmap_overlay(
    composite: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    n_mels: int = N_MELS,
    alpha: float = 0.5,
) -> None:
    """Save a heatmap overlay on the composite spectrogram as PNG.

    Args:
        composite: 2D spectrogram array (n_mels * 5, time_frames) in [0, 1].
        heatmap: Grad-CAM heatmap of same shape, values in [0, 1].
        output_path: Path to save the PNG.
        n_mels: Mel bins per vowel (used for drawing separator lines).
        alpha: Overlay blend weight (0 = spectrogram only, 1 = heatmap only).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(composite, aspect="auto", origin="lower", cmap="gray")
    ax.imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=alpha)

    # Draw horizontal lines between vowel strips
    for i in range(1, 5):
        ax.axhline(y=i * n_mels - 0.5, color="white", linewidth=0.8, linestyle="--")

    # Label vowels
    for i, vowel in enumerate(VOWELS):
        ax.text(
            -0.02, (i * n_mels + n_mels / 2) / (n_mels * 5),
            f"/{vowel}/",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=10, color="white",
            fontweight="bold",
        )

    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel frequency bins (stacked vowels)")
    ax.set_title("Grad-CAM: Vowel Contributions to PD Detection")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap overlay saved to %s", output_path)
