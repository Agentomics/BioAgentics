"""Tests for Grad-CAM interpretability module."""

import cv2
import numpy as np
import torch

from bioagentics.diagnostics.retinal_dr_screening.gradcam import (
    GradCAM,
    find_target_layer,
    overlay_heatmap,
    preprocess_image_for_gradcam,
)
from bioagentics.diagnostics.retinal_dr_screening.training import create_model


def test_find_target_layer_mobilenet():
    model = create_model("mobilenetv3_small_100", num_classes=5, pretrained=False)
    layer = find_target_layer(model, "mobilenetv3_small_100")
    assert isinstance(layer, torch.nn.Module)


def test_find_target_layer_efficientnet():
    model = create_model("efficientnet_b0", num_classes=5, pretrained=False)
    layer = find_target_layer(model, "efficientnet_b0")
    assert isinstance(layer, torch.nn.Module)


def test_gradcam_generates_heatmap():
    model = create_model("mobilenetv3_small_100", num_classes=5, pretrained=False)
    target_layer = find_target_layer(model, "mobilenetv3_small_100")

    gradcam = GradCAM(model, target_layer)
    input_tensor = torch.randn(1, 3, 64, 64)

    heatmap = gradcam.generate(input_tensor, target_class=0)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1.0

    gradcam.remove_hooks()


def test_gradcam_predicted_class():
    """When target_class=None, uses predicted class."""
    model = create_model("mobilenetv3_small_100", num_classes=5, pretrained=False)
    target_layer = find_target_layer(model, "mobilenetv3_small_100")

    gradcam = GradCAM(model, target_layer)
    input_tensor = torch.randn(1, 3, 64, 64)

    heatmap = gradcam.generate(input_tensor, target_class=None)
    assert heatmap.shape == (64, 64)

    gradcam.remove_hooks()


def test_overlay_heatmap():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    heatmap = np.random.rand(100, 100).astype(np.float32)

    overlay = overlay_heatmap(image, heatmap, alpha=0.4)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_overlay_heatmap_resize():
    """Overlay should handle different heatmap size."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    heatmap = np.random.rand(50, 50).astype(np.float32)

    overlay = overlay_heatmap(image, heatmap)
    assert overlay.shape == image.shape


def test_preprocess_image_for_gradcam(tmp_path):
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)

    tensor, original = preprocess_image_for_gradcam(img_path, image_size=64)

    assert tensor.shape == (1, 3, 64, 64)
    assert original.shape == (64, 64, 3)
    assert tensor.dtype == torch.float32
