"""Attention heatmap visualization for MIL MSI classifiers.

Generates attention heatmaps from trained MIL models overlaid on WSI
thumbnails. Extracts attention weights from ABMIL/CLAM/TransMIL and
maps them back to tile coordinates.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .mil_models import MILOutput

logger = logging.getLogger(__name__)

# Colormap: blue (low) -> yellow (mid) -> red (high)
_HEATMAP_COLORS = np.array([
    [0, 0, 255],    # low attention: blue
    [0, 255, 255],  # low-mid: cyan
    [0, 255, 0],    # mid: green
    [255, 255, 0],  # mid-high: yellow
    [255, 0, 0],    # high attention: red
], dtype=np.uint8)


def _attention_to_rgb(attention: np.ndarray) -> np.ndarray:
    """Map normalized attention values [0,1] to RGB colors.

    Args:
        attention: Array of attention values in [0, 1].

    Returns:
        Array of shape (*attention.shape, 3) with RGB values.
    """
    # Clamp and scale to colormap index
    att = np.clip(attention, 0, 1)
    n_colors = len(_HEATMAP_COLORS)
    idx = att * (n_colors - 1)
    lower = np.floor(idx).astype(int)
    upper = np.minimum(lower + 1, n_colors - 1)
    frac = (idx - lower)[..., np.newaxis]

    # Linear interpolation between color stops
    colors = (
        _HEATMAP_COLORS[lower] * (1 - frac)
        + _HEATMAP_COLORS[upper] * frac
    )
    return colors.astype(np.uint8)


@torch.no_grad()
def extract_attention(
    model: torch.nn.Module,
    features_path: str | Path,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract attention weights from a trained MIL model.

    Args:
        model: Trained MIL model (ABMIL/CLAM/TransMIL).
        features_path: Path to HDF5 file with patch features.
        device: Inference device.

    Returns:
        (attention_weights, prediction_probs) where attention_weights is
        shape (N_patches,) and prediction_probs is shape (n_classes,).
    """
    model.eval()

    with h5py.File(features_path, "r") as f:
        features = f["features"][:]

    features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    output: MILOutput = model(features_t)

    attention = output.attention_weights.squeeze(0).cpu().numpy()
    probs = torch.softmax(output.logits, dim=1).squeeze(0).cpu().numpy()

    return attention, probs


def generate_heatmap(
    attention: np.ndarray,
    tile_coords: pd.DataFrame,
    slide_dimensions: tuple[int, int],
    thumbnail_size: int = 2048,
    alpha: float = 0.5,
    thumbnail_image: Image.Image | None = None,
) -> Image.Image:
    """Generate an attention heatmap image.

    Maps attention weights to tile locations and renders a heatmap at
    thumbnail resolution, optionally overlaid on a WSI thumbnail.

    Args:
        attention: Attention weights, shape (N_patches,).
        tile_coords: DataFrame with columns x, y (level-0 coordinates)
            and tile_size.
        slide_dimensions: (width, height) of the slide at level 0.
        thumbnail_size: Max dimension of the output heatmap image.
        alpha: Opacity of the heatmap overlay (0=transparent, 1=opaque).
        thumbnail_image: Optional WSI thumbnail to overlay on.

    Returns:
        PIL Image of the attention heatmap.
    """
    slide_w, slide_h = slide_dimensions

    # Compute thumbnail scale
    scale = thumbnail_size / max(slide_w, slide_h)
    thumb_w = int(slide_w * scale)
    thumb_h = int(slide_h * scale)

    # Create heatmap canvas
    heatmap = np.zeros((thumb_h, thumb_w, 3), dtype=np.float64)
    counts = np.zeros((thumb_h, thumb_w), dtype=np.float64)

    # Normalize attention to [0, 1]
    att_min = attention.min()
    att_max = attention.max()
    if att_max > att_min:
        att_norm = (attention - att_min) / (att_max - att_min)
    else:
        att_norm = np.zeros_like(attention)

    # Map each tile's attention to thumbnail coordinates
    n_mapped = 0
    for i in range(min(len(attention), len(tile_coords))):
        row = tile_coords.iloc[i]
        x = int(row["x"] * scale)
        y = int(row["y"] * scale)
        tile_size = int(row.get("tile_size", 256) * scale)
        tile_size = max(tile_size, 1)

        x_end = min(x + tile_size, thumb_w)
        y_end = min(y + tile_size, thumb_h)

        if x < thumb_w and y < thumb_h:
            color = _attention_to_rgb(np.array([att_norm[i]]))[0].astype(np.float64)
            heatmap[y:y_end, x:x_end] += color
            counts[y:y_end, x:x_end] += 1
            n_mapped += 1

    # Average overlapping regions
    mask = counts > 0
    heatmap[mask] /= counts[mask, np.newaxis]

    # Convert to uint8
    heatmap_img = Image.fromarray(heatmap.astype(np.uint8))

    # Overlay on thumbnail if provided
    if thumbnail_image is not None:
        thumb = thumbnail_image.resize((thumb_w, thumb_h)).convert("RGB")
        thumb_arr = np.array(thumb, dtype=np.float64)
        heat_arr = np.array(heatmap_img, dtype=np.float64)

        # Only overlay where we have heatmap data
        blended = thumb_arr.copy()
        blended[mask] = (1 - alpha) * thumb_arr[mask] + alpha * heat_arr[mask]
        heatmap_img = Image.fromarray(blended.astype(np.uint8))

    logger.info(f"Generated heatmap: {thumb_w}x{thumb_h}, {n_mapped} tiles mapped")
    return heatmap_img


def generate_slide_heatmap(
    model: torch.nn.Module,
    features_path: str | Path,
    tile_coords_path: str | Path,
    slide_dimensions: tuple[int, int],
    output_path: str | Path,
    thumbnail_image: Image.Image | None = None,
    thumbnail_size: int = 2048,
    alpha: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Generate and save a complete attention heatmap for one slide.

    Args:
        model: Trained MIL model.
        features_path: Path to HDF5 feature file.
        tile_coords_path: Path to tile coordinates CSV.
        slide_dimensions: (width, height) at level 0.
        output_path: Where to save the heatmap image.
        thumbnail_image: Optional WSI thumbnail for overlay.
        thumbnail_size: Max dimension of output image.
        alpha: Heatmap overlay opacity.
        device: Inference device.

    Returns:
        Dict with prediction probabilities and attention statistics.
    """
    # Extract attention
    attention, probs = extract_attention(model, features_path, device)

    # Load tile coordinates
    tile_coords = pd.read_csv(tile_coords_path)

    # Generate heatmap
    heatmap = generate_heatmap(
        attention, tile_coords, slide_dimensions,
        thumbnail_size=thumbnail_size,
        alpha=alpha,
        thumbnail_image=thumbnail_image,
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap.save(output_path)

    result = {
        "slide": Path(features_path).stem,
        "prediction_msi_h_prob": float(probs[1]) if len(probs) > 1 else float(probs[0]),
        "prediction_class": "MSI-H" if (probs.argmax() == 1) else "MSS",
        "n_patches": len(attention),
        "attention_mean": float(attention.mean()),
        "attention_std": float(attention.std()),
        "attention_max": float(attention.max()),
        "output_path": str(output_path),
    }

    logger.info(
        f"Saved heatmap for {result['slide']}: "
        f"pred={result['prediction_class']} "
        f"(p={result['prediction_msi_h_prob']:.3f})"
    )
    return result
