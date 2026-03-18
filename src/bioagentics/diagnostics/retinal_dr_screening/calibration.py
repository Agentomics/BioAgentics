"""Post-hoc calibration and ECE evaluation for DR screening models.

Implements:
  1. Temperature scaling — single scalar T learned on validation set
  2. Platt scaling — per-class logistic regression on logits
  3. Expected Calibration Error (ECE) and reliability diagrams
  4. Clinical operating point analysis for referable DR threshold

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.calibration \\
        --model-path output/.../best_model.pt \\
        --splits data/.../splits.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from bioagentics.diagnostics.retinal_dr_screening.augmentation import DRDataset
from bioagentics.diagnostics.retinal_dr_screening.config import (
    BATCH_SIZE,
    DATA_DIR,
    ECE_N_BINS,
    NUM_WORKERS,
    REFERABLE_THRESHOLD,
    RESULTS_DIR,
    TRAIN_IMAGE_SIZE,
)
from bioagentics.diagnostics.retinal_dr_screening.training import (
    create_model,
    evaluate,
)

logger = logging.getLogger(__name__)


# ── Temperature scaling ──


def learn_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Learn optimal temperature T on validation logits via NLL minimization.

    Args:
        logits: Raw model logits, shape (N, C).
        labels: Ground truth labels, shape (N,).

    Returns:
        Optimal temperature scalar.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    def nll_at_temp(T: float) -> float:
        scaled = logits_t / T
        log_probs = torch.nn.functional.log_softmax(scaled, dim=1)
        nll = torch.nn.functional.nll_loss(log_probs, labels_t)
        return nll.item()

    result = minimize_scalar(nll_at_temp, bounds=(0.1, 10.0), method="bounded")
    optimal_T = float(result.x)
    logger.info("Optimal temperature: %.4f (NLL: %.4f → %.4f)",
                optimal_T, nll_at_temp(1.0), result.fun)
    return optimal_T


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return calibrated probabilities."""
    scaled = logits / temperature
    # Softmax
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


# ── Platt scaling ──


def learn_platt_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    """Learn Platt scaling (per-class logistic regression on logits).

    Args:
        logits: Raw model logits, shape (N, C).
        labels: Ground truth labels, shape (N,).

    Returns:
        Fitted LogisticRegression model.
    """
    platt = LogisticRegression(max_iter=1000, solver="lbfgs")
    platt.fit(logits, labels)
    logger.info("Platt scaling fitted on %d samples", len(labels))
    return platt


def apply_platt_scaling(
    logits: np.ndarray,
    platt_model: LogisticRegression,
) -> np.ndarray:
    """Apply Platt scaling to get calibrated probabilities."""
    return platt_model.predict_proba(logits)


# ── ECE and reliability diagram ──


@dataclass
class CalibrationResult:
    """Calibration evaluation results."""

    ece: float
    mce: float  # maximum calibration error
    bin_confidences: list[float]
    bin_accuracies: list[float]
    bin_counts: list[int]
    n_bins: int


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> CalibrationResult:
    """Compute Expected Calibration Error (ECE) and reliability diagram data.

    Args:
        probs: Predicted probabilities, shape (N, C).
        labels: Ground truth labels, shape (N,).
        n_bins: Number of confidence bins.

    Returns:
        CalibrationResult with ECE, MCE, and per-bin data.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(in_bin.sum())
        bin_counts.append(count)

        if count > 0:
            avg_conf = float(confidences[in_bin].mean())
            avg_acc = float(correct[in_bin].mean())
            bin_confidences.append(round(avg_conf, 4))
            bin_accuracies.append(round(avg_acc, 4))

            gap = abs(avg_acc - avg_conf)
            ece += gap * count
            mce = max(mce, gap)
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    ece /= len(labels)

    return CalibrationResult(
        ece=round(ece, 4),
        mce=round(mce, 4),
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        n_bins=n_bins,
    )


def plot_reliability_diagram(
    result: CalibrationResult,
    title: str = "Reliability Diagram",
    output_path: Path | None = None,
):
    """Plot reliability diagram from calibration results."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    bin_centers = [(i + 0.5) / result.n_bins for i in range(result.n_bins)]
    width = 1.0 / result.n_bins

    # Bars for accuracy
    ax.bar(bin_centers, result.bin_accuracies, width=width, alpha=0.6,
           color="steelblue", edgecolor="navy", label="Accuracy")

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={result.ece:.4f}, MCE={result.mce:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Reliability diagram saved to %s", output_path)
    else:
        plt.close()

    return fig


# ── Clinical operating point ──


def find_clinical_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_sensitivity: float = 0.95,
) -> dict:
    """Find the referable DR probability threshold achieving target sensitivity.

    Args:
        probs: Predicted probabilities, shape (N, C).
        labels: Ground truth labels, shape (N,).
        target_sensitivity: Minimum sensitivity for referable DR.

    Returns:
        Dict with threshold, sensitivity, specificity, and PPV/NPV.
    """
    binary_labels = (labels >= REFERABLE_THRESHOLD).astype(int)
    referable_probs = probs[:, REFERABLE_THRESHOLD:].sum(axis=1)

    # Search thresholds
    thresholds = np.linspace(0.01, 0.99, 200)
    best = {"threshold": 0.5, "sensitivity": 0.0, "specificity": 0.0}

    for t in thresholds:
        preds = (referable_probs >= t).astype(int)
        tp = ((binary_labels == 1) & (preds == 1)).sum()
        fn = ((binary_labels == 1) & (preds == 0)).sum()
        fp = ((binary_labels == 0) & (preds == 1)).sum()
        tn = ((binary_labels == 0) & (preds == 0)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if sens >= target_sensitivity:
            if spec > best.get("specificity", 0.0):
                best = {
                    "threshold": round(float(t), 4),
                    "sensitivity": round(float(sens), 4),
                    "specificity": round(float(spec), 4),
                    "ppv": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4),
                    "npv": round(float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0, 4),
                }

    return best


# ── Full calibration pipeline ──


def calibrate_model(
    model_path: Path,
    splits_csv: Path,
    model_name: str = "mobilenetv3_small_100",
    image_size: int = TRAIN_IMAGE_SIZE,
) -> dict:
    """Run full calibration pipeline on a trained model.

    Returns dict with temperature, ECE before/after, clinical thresholds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_model(model_name, pretrained=False)
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Get val and test logits
    val_ds = DRDataset(splits_csv, split="val", transform="val",
                       image_size=image_size, gradable_only=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    criterion = nn.CrossEntropyLoss()
    _, _, val_labels, _, val_probs = evaluate(model, val_loader, criterion, device)

    # Get raw logits from val set
    model.eval()
    val_logits_list = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            logits = model(images)
            val_logits_list.append(logits.cpu().numpy())
    val_logits = np.concatenate(val_logits_list)

    # ECE before calibration
    ece_before = compute_ece(val_probs, val_labels)

    # Temperature scaling
    temperature = learn_temperature(val_logits, val_labels)
    cal_probs_temp = apply_temperature(val_logits, temperature)
    ece_after_temp = compute_ece(cal_probs_temp, val_labels)

    # Platt scaling
    platt = learn_platt_scaling(val_logits, val_labels)
    cal_probs_platt = apply_platt_scaling(val_logits, platt)
    ece_after_platt = compute_ece(cal_probs_platt, val_labels)

    # Clinical threshold
    clinical = find_clinical_threshold(cal_probs_temp, val_labels)

    results = {
        "temperature": temperature,
        "ece_before": ece_before.ece,
        "ece_after_temperature": ece_after_temp.ece,
        "ece_after_platt": ece_after_platt.ece,
        "mce_before": ece_before.mce,
        "mce_after_temperature": ece_after_temp.mce,
        "clinical_threshold": clinical,
    }

    # Save
    results_dir = RESULTS_DIR / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("ECE: %.4f → %.4f (temp) / %.4f (platt)",
                ece_before.ece, ece_after_temp.ece, ece_after_platt.ece)
    return results


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Calibrate DR screening model")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=DATA_DIR / "splits.csv")
    parser.add_argument("--model-name", default="mobilenetv3_small_100")
    parser.add_argument("--image-size", type=int, default=TRAIN_IMAGE_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = calibrate_model(args.model_path, args.splits, args.model_name, args.image_size)

    print(f"\nCalibration results:")
    print(f"  Temperature: {results['temperature']:.4f}")
    print(f"  ECE: {results['ece_before']:.4f} → {results['ece_after_temperature']:.4f}")
    print(f"  Clinical threshold: {results['clinical_threshold']}")


if __name__ == "__main__":
    main()
