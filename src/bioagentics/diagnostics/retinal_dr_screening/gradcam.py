"""Grad-CAM interpretability for DR screening models.

Generates class activation maps from the final convolutional layer,
highlighting regions the model attends to for DR grading decisions.

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.gradcam \\
        --model-path output/.../best_model.pt \\
        --image path/to/fundus.jpg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bioagentics.diagnostics.retinal_dr_screening.config import (
    DATA_DIR,
    DR_CLASSES,
    FIGURES_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOBILE_IMAGE_SIZE,
)
from bioagentics.diagnostics.retinal_dr_screening.training import create_model

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM implementation for CNN models.

    Computes gradient-weighted class activation maps from a target
    convolutional layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # Disable in-place operations to avoid conflicts with backward hooks
        self._disable_inplace(model)

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    @staticmethod
    def _disable_inplace(model: nn.Module):
        """Disable in-place operations in the model for hook compatibility."""
        for module in model.modules():
            if hasattr(module, "inplace"):
                module.inplace = False

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W).
            target_class: Class index to visualize. If None, uses predicted class.

        Returns:
            Heatmap as numpy array (H, W), values in [0, 1].
        """
        self.model.eval()
        # Clone input to avoid in-place operation issues with hooks
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP of gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only positive contributions

        # Resize to input size
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


def find_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Find the last convolutional layer for Grad-CAM.

    Handles common timm architectures.
    """
    # Try common patterns
    if hasattr(model, "conv_head"):
        # EfficientNet
        return model.conv_head
    if hasattr(model, "features"):
        # MobileNet / EfficientNet
        return model.features[-1]
    if hasattr(model, "layer4"):
        # ResNet
        return model.layer4[-1]

    # Fallback: find last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError(f"Could not find convolutional layer in {model_name}")

    return last_conv


def preprocess_image_for_gradcam(
    image_path: Path,
    image_size: int = MOBILE_IMAGE_SIZE,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess image for Grad-CAM.

    Returns (tensor for model input, original RGB image for overlay).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original, (image_size, image_size))

    # Normalize
    tensor = resized.astype(np.float32) / 255.0
    for c in range(3):
        tensor[:, :, c] = (tensor[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)
    return tensor, resized


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image.

    Args:
        image: RGB image, uint8.
        heatmap: Heatmap, float [0, 1], same spatial dims as image.
        alpha: Overlay transparency.

    Returns:
        Overlay image, uint8 RGB.
    """
    # Resize heatmap to match image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * image.astype(np.float32) + alpha * heatmap_color.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def generate_gradcam_for_image(
    model_path: Path,
    image_path: Path,
    model_name: str = "mobilenetv3_small_100",
    image_size: int = MOBILE_IMAGE_SIZE,
    output_path: Path | None = None,
) -> dict:
    """Generate Grad-CAM visualization for a single image.

    Returns dict with predicted class, confidence, and output path.
    """
    device = torch.device("cpu")  # Grad-CAM works best on CPU for hooks

    model = create_model(model_name, pretrained=False)
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = find_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    input_tensor, original = preprocess_image_for_gradcam(image_path, image_size)
    input_tensor = input_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    # Generate heatmap for predicted class
    heatmap = gradcam.generate(input_tensor, target_class=pred_class)
    overlay = overlay_heatmap(original, heatmap)

    gradcam.remove_hooks()

    if output_path is None:
        output_path = FIGURES_DIR / f"gradcam_{image_path.stem}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save side-by-side: original | heatmap | overlay
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Pred: {DR_CLASSES[pred_class]} ({confidence:.1%})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(
        "Grad-CAM saved: %s → %s (conf=%.1f%%)",
        image_path.name, DR_CLASSES[pred_class], confidence * 100,
    )

    return {
        "image": str(image_path),
        "predicted_class": pred_class,
        "predicted_label": DR_CLASSES[pred_class],
        "confidence": round(confidence, 4),
        "output_path": str(output_path),
    }


def batch_gradcam(
    model_path: Path,
    splits_csv: Path,
    split: str = "test",
    model_name: str = "mobilenetv3_small_100",
    image_size: int = MOBILE_IMAGE_SIZE,
    n_samples: int = 20,
    output_dir: Path | None = None,
) -> list[dict]:
    """Generate Grad-CAM for a batch of images from the dataset."""
    import pandas as pd

    if output_dir is None:
        output_dir = FIGURES_DIR / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(splits_csv)
    split_df = df[df["split"] == split]

    # Sample evenly across grades
    samples = split_df.groupby("dr_grade").apply(
        lambda x: x.sample(min(n_samples // 5, len(x)), random_state=42),
        include_groups=False,
    ).reset_index(drop=True)

    results = []
    for _, row in samples.iterrows():
        try:
            result = generate_gradcam_for_image(
                model_path,
                Path(row["image_path"]),
                model_name,
                image_size,
                output_dir / f"gradcam_{Path(row['image_path']).stem}.png",
            )
            results.append(result)
        except Exception as e:
            logger.warning("Grad-CAM failed for %s: %s", row["image_path"], e)

    logger.info("Generated %d Grad-CAM visualizations", len(results))
    return results


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for DR screening")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="mobilenetv3_small_100")
    parser.add_argument("--image", type=Path, help="Single image path")
    parser.add_argument("--splits", type=Path, help="Splits CSV for batch mode")
    parser.add_argument("--n", type=int, default=20, help="Batch sample count")
    parser.add_argument("--image-size", type=int, default=MOBILE_IMAGE_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.image:
        result = generate_gradcam_for_image(
            args.model_path, args.image, args.model_name, args.image_size,
        )
        print(f"Prediction: {result['predicted_label']} ({result['confidence']:.1%})")
        print(f"Saved to: {result['output_path']}")
    elif args.splits:
        results = batch_gradcam(
            args.model_path, args.splits, model_name=args.model_name,
            image_size=args.image_size, n_samples=args.n,
        )
        print(f"Generated {len(results)} Grad-CAM visualizations")
    else:
        parser.error("Provide --image or --splits")


if __name__ == "__main__":
    main()
