"""Post-hoc calibration for CXR classification.

Temperature scaling with optional per-institution parameters.
Computes reliability diagrams and expected calibration error (ECE).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from bioagentics.cxr_rare.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits."""
    return logits / temperature


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def find_optimal_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Find optimal temperature on validation set via NLL minimization.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C)
        Raw model logits.
    labels : np.ndarray, shape (N, C)
        Binary ground truth.

    Returns
    -------
    float
        Optimal temperature.
    """
    def nll(t: float) -> float:
        scaled = sigmoid(temperature_scale(logits, t))
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        loss = -(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled))
        return float(loss.mean())

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    logger.info("Optimal temperature: %.4f (NLL: %.4f)", result.x, result.fun)
    return float(result.x)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute expected calibration error (ECE).

    Parameters
    ----------
    probs : np.ndarray
        Predicted probabilities (flattened across samples and classes).
    labels : np.ndarray
        Binary labels (same shape as probs).

    Returns
    -------
    float
        ECE value.
    """
    probs_flat = probs.ravel()
    labels_flat = labels.ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs_flat > bin_edges[i]) & (probs_flat <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs_flat[mask].mean()
        bin_acc = labels_flat[mask].mean()
        ece += mask.sum() / len(probs_flat) * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
) -> Path:
    """Generate reliability diagram."""
    probs_flat = probs.ravel()
    labels_flat = labels.ravel()
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_confs = []
    bin_accs = []
    bin_counts = []
    for i in range(n_bins):
        mask = (probs_flat > bin_edges[i]) & (probs_flat <= bin_edges[i + 1])
        if mask.sum() == 0:
            bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accs.append(0)
            bin_counts.append(0)
        else:
            bin_confs.append(probs_flat[mask].mean())
            bin_accs.append(labels_flat[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability plot
    ax1.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.7, color="#2196F3", edgecolor="black")
    ax1.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Count histogram
    ax2.bar(bin_confs, bin_counts, width=1.0 / n_bins, alpha=0.7, color="#FF9800", edgecolor="black")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved reliability diagram: %s", output_path)
    return output_path


def calibrate_and_evaluate(
    logits: np.ndarray,
    labels: np.ndarray,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
    experiment_name: str = "calibration",
    institution: str | None = None,
) -> dict:
    """Calibrate with temperature scaling and evaluate.

    Parameters
    ----------
    logits : np.ndarray
        Calibration set logits (used to find temperature).
    labels : np.ndarray
        Calibration set labels.
    val_logits : np.ndarray
        Validation/test logits to calibrate.
    val_labels : np.ndarray
        Validation/test labels.
    institution : str, optional
        Institution name for per-institution calibration.

    Returns
    -------
    dict with temperature, ECE before/after, and file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{institution}" if institution else ""

    # Before calibration
    probs_before = sigmoid(val_logits)
    ece_before = expected_calibration_error(probs_before, val_labels)

    # Find optimal temperature
    temperature = find_optimal_temperature(logits, labels)

    # After calibration
    probs_after = sigmoid(temperature_scale(val_logits, temperature))
    ece_after = expected_calibration_error(probs_after, val_labels)

    logger.info(
        "Calibration%s: T=%.4f, ECE before=%.4f, ECE after=%.4f",
        f" ({institution})" if institution else "", temperature, ece_before, ece_after,
    )

    # Reliability diagrams
    reliability_diagram(
        probs_before, val_labels,
        fig_dir / f"reliability_before{suffix}.png",
        title=f"Before Calibration{' (' + institution + ')' if institution else ''}",
    )
    reliability_diagram(
        probs_after, val_labels,
        fig_dir / f"reliability_after{suffix}.png",
        title=f"After Calibration (T={temperature:.2f}){' (' + institution + ')' if institution else ''}",
    )

    return {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "institution": institution,
    }
