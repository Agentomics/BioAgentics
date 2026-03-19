"""Calibration analysis for MSI classification models.

Computes Expected Calibration Error (ECE), generates reliability diagrams,
and performs temperature scaling for post-hoc calibration.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .training import (
    SlideFeatureDataset,
    collate_variable_length,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a model."""

    ece: float
    mce: float  # Maximum Calibration Error
    n_bins: int
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]
    temperature: float = 1.0  # post-calibration temperature


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Expected Calibration Error (ECE) and related metrics.

    ECE measures the gap between predicted confidence and actual accuracy,
    weighted by the proportion of samples in each bin.

    Args:
        probs: Predicted probabilities for the positive class, shape (N,).
        labels: True binary labels, shape (N,).
        n_bins: Number of confidence bins.

    Returns:
        CalibrationMetrics with ECE, MCE, and per-bin statistics.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Include upper boundary in last bin
        if i == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)

        count = mask.sum()
        bin_counts.append(int(count))

        if count > 0:
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        bin_accuracies.append(float(bin_acc))
        bin_confidences.append(float(bin_conf))

    # Weighted ECE
    total = len(probs)
    ece = sum(
        (count / total) * abs(acc - conf)
        for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences)
        if count > 0
    )

    # Maximum Calibration Error
    mce = max(
        abs(acc - conf)
        for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences)
        if count > 0
    ) if any(c > 0 for c in bin_counts) else 0.0

    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        n_bins=n_bins,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


class TemperatureScaling(nn.Module):
    """Temperature scaling for post-hoc calibration.

    Learns a single temperature parameter to scale logits before softmax.
    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature


def fit_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 100,
) -> float:
    """Fit temperature scaling on validation logits.

    Args:
        logits: Raw model logits, shape (N, n_classes).
        labels: True labels, shape (N,).
        lr: Learning rate for optimization.
        max_iter: Maximum optimization steps.

    Returns:
        Optimal temperature value.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    temp_model = TemperatureScaling()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = temp_model(logits_t)
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    optimal_temp = temp_model.temperature.item()
    logger.info(f"Optimal temperature: {optimal_temp:.4f}")
    return optimal_temp


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    slide_ids: list[str],
    labels: dict[str, int],
    features_dir: str | Path,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect model predictions for calibration analysis.

    Args:
        model: Trained MIL model.
        slide_ids: Slide IDs to evaluate.
        labels: Label dict mapping slide_id to 0/1.
        features_dir: Directory with HDF5 feature files.
        device: Device for inference.

    Returns:
        (logits, probs, true_labels) as numpy arrays.
    """
    model.eval()
    dataset = SlideFeatureDataset(slide_ids, labels, features_dir)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_variable_length,
    )

    all_logits = []
    all_probs = []
    all_labels = []

    for features, batch_labels, _ in loader:
        features = features.to(device)
        output = model(features)
        logits = output.logits.cpu().numpy()
        probs = torch.softmax(output.logits, dim=1)[:, 1].cpu().numpy()

        all_logits.append(logits)
        all_probs.extend(probs)
        all_labels.extend(batch_labels.numpy())

    return (
        np.vstack(all_logits),
        np.array(all_probs),
        np.array(all_labels),
    )


def calibrate_model(
    model: nn.Module,
    val_ids: list[str],
    test_ids: list[str],
    labels: dict[str, int],
    features_dir: str | Path,
    device: str = "cpu",
    n_bins: int = 10,
    output_dir: str | Path | None = None,
) -> dict:
    """Run full calibration analysis: ECE before/after temperature scaling.

    Args:
        model: Trained MIL model.
        val_ids: Validation slide IDs (for fitting temperature).
        test_ids: Test slide IDs (for evaluating calibration).
        labels: Label dict.
        features_dir: Directory with HDF5 feature files.
        device: Device for inference.
        n_bins: Number of bins for ECE computation.
        output_dir: Where to save calibration results.

    Returns:
        Dict with pre/post calibration metrics and optimal temperature.
    """
    # Collect predictions on validation set for temperature fitting
    val_logits, _val_probs, val_labels = collect_predictions(
        model, val_ids, labels, features_dir, device,
    )

    # Collect predictions on test set
    test_logits, test_probs, test_labels = collect_predictions(
        model, test_ids, labels, features_dir, device,
    )

    # Pre-calibration ECE on test set
    pre_cal = expected_calibration_error(test_probs, test_labels, n_bins)

    # Fit temperature on validation set
    optimal_temp = fit_temperature_scaling(val_logits, val_labels)

    # Apply temperature scaling to test logits
    scaled_logits = test_logits / optimal_temp
    scaled_probs = _softmax(scaled_logits)[:, 1]

    # Post-calibration ECE on test set
    post_cal = expected_calibration_error(scaled_probs, test_labels, n_bins)
    post_cal.temperature = optimal_temp

    results = {
        "pre_calibration": {
            "ece": pre_cal.ece,
            "mce": pre_cal.mce,
            "bin_accuracies": pre_cal.bin_accuracies,
            "bin_confidences": pre_cal.bin_confidences,
            "bin_counts": pre_cal.bin_counts,
        },
        "post_calibration": {
            "ece": post_cal.ece,
            "mce": post_cal.mce,
            "temperature": optimal_temp,
            "bin_accuracies": post_cal.bin_accuracies,
            "bin_confidences": post_cal.bin_confidences,
            "bin_counts": post_cal.bin_counts,
        },
        "ece_reduction": pre_cal.ece - post_cal.ece,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "calibration_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Calibration results saved to {out}")

    logger.info(
        f"ECE: {pre_cal.ece:.4f} -> {post_cal.ece:.4f} "
        f"(temp={optimal_temp:.4f})"
    )

    return results


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over last axis."""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)
