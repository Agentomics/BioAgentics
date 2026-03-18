"""GRU temporal model for variable-length time-series flare prediction.

Handles variable-length multi-omic time-series from HMP2 (uneven sampling).
Uses the same LOPO-CV framework as XGBoost for fair comparison.

Usage::

    from bioagentics.crohns.flare_prediction.gru_model import (
        build_sequences,
        gru_lopo_cv,
    )

    sequences = build_sequences(features, windows)
    results = gru_lopo_cv(sequences)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import os

import numpy as np
import pandas as pd

# Fix xgboost/PyTorch OpenMP conflict on macOS: when both libraries load
# their own copy of libomp, the process segfaults during backward pass.
# Setting OMP_NUM_THREADS=1 before import avoids the conflict.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from bioagentics.crohns.flare_prediction.classifier import (
    CVFold,
    CVResults,
)
from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)


@dataclass
class PatientSequence:
    """A patient's temporal sequence of feature vectors."""

    patient_id: str
    features: np.ndarray  # (seq_len, n_features)
    labels: np.ndarray  # (seq_len,) — 1=pre_flare, 0=stable
    timestamps: list[pd.Timestamp]
    seq_len: int


def build_sequences(
    features: pd.DataFrame,
    windows: list[Window],
) -> list[PatientSequence]:
    """Build per-patient temporal sequences from windowed features.

    Groups windows by patient and orders by time to create sequences
    suitable for GRU input.

    Parameters
    ----------
    features:
        Feature matrix (instances x features), aligned with windows.
    windows:
        Classification windows.

    Returns
    -------
    List of PatientSequence, one per patient.
    """
    # Group windows by patient, ordered by time
    patient_data: dict[str, list[tuple[int, Window]]] = {}
    for idx, w in enumerate(windows):
        patient_data.setdefault(w.subject_id, []).append((idx, w))

    sequences = []
    for patient_id, entries in sorted(patient_data.items()):
        # Sort by window start time
        entries.sort(key=lambda x: x[1].window_start)

        idxs = [e[0] for e in entries]
        X = features.iloc[idxs].values.astype(float)
        y = np.array([1 if e[1].label == "pre_flare" else 0 for e in entries])
        timestamps = [e[1].window_start for e in entries]

        # Impute NaN with column medians
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            m = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
            X[mask, j] = m

        sequences.append(
            PatientSequence(
                patient_id=patient_id,
                features=X,
                labels=y,
                timestamps=timestamps,
                seq_len=len(entries),
            )
        )

    logger.info(
        "Built %d patient sequences (total %d timepoints)",
        len(sequences),
        sum(s.seq_len for s in sequences),
    )
    return sequences


class FlareGRU(nn.Module):
    """GRU model for per-timepoint flare prediction.

    Processes one sequence at a time (no packing needed for single sequences).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on a single sequence.

        Parameters
        ----------
        x:
            Single sequence (1, seq_len, n_features).

        Returns
        -------
        Per-timepoint logits (seq_len,).
        """
        output, _ = self.gru(x)
        out = self.dropout(output)
        logits = self.fc(out).squeeze(-1).squeeze(0)  # (seq_len,)
        return logits


def _standardize_sequences(
    train_seqs: list[PatientSequence],
    test_seqs: list[PatientSequence],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Standardize features using training set statistics."""
    # Stack all training features
    all_train = np.vstack([s.features for s in train_seqs])
    mean = np.nanmean(all_train, axis=0)
    std = np.nanstd(all_train, axis=0)
    std[std < 1e-8] = 1.0

    train_scaled = [(s.features - mean) / std for s in train_seqs]
    test_scaled = [(s.features - mean) / std for s in test_seqs]
    return train_scaled, test_scaled


def _train_epoch(
    model: FlareGRU,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_X: list[np.ndarray],
    train_y: list[np.ndarray],
) -> float:
    """Train one epoch (per-sequence gradient accumulation), returns average loss."""
    model.train()
    total_loss = 0.0

    optimizer.zero_grad()

    for x, y in zip(train_X, train_y):
        t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, feat)
        target = torch.tensor(y, dtype=torch.float32)

        logits = model(t)
        loss = criterion(logits, target) / len(train_X)
        loss.backward()
        total_loss += float(loss.item())

    optimizer.step()
    return total_loss


def _predict(
    model: FlareGRU,
    test_X: list[np.ndarray],
) -> list[np.ndarray]:
    """Predict probabilities for test sequences."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for x in test_X:
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, features)
            logits = model(t)
            probs = torch.sigmoid(logits)
            predictions.append(probs.numpy())

    return predictions


def gru_lopo_cv(
    sequences: list[PatientSequence],
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.2,
    lr: float = 0.001,
    n_epochs: int = 50,
    patience: int = 10,
    random_state: int = 42,
) -> CVResults:
    """Run Leave-One-Patient-Out CV with GRU model.

    Parameters
    ----------
    sequences:
        Per-patient temporal sequences.
    hidden_size:
        GRU hidden layer size.
    num_layers:
        Number of GRU layers.
    dropout:
        Dropout rate.
    lr:
        Learning rate.
    n_epochs:
        Maximum training epochs.
    patience:
        Early stopping patience (epochs without improvement).
    random_state:
        Random seed.

    Returns
    -------
    CVResults with per-fold predictions.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required: uv add --optional research torch")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if len(sequences) == 0:
        return CVResults(model_name="gru")

    n_features = sequences[0].features.shape[1]
    results = CVResults(model_name="gru")

    for held_out_idx, held_out in enumerate(sequences):
        train_seqs = [s for i, s in enumerate(sequences) if i != held_out_idx]

        # Need at least 2 training patients with both classes
        train_labels = np.concatenate([s.labels for s in train_seqs])
        if len(np.unique(train_labels)) < 2 or len(train_seqs) < 2:
            continue

        # Standardize
        train_scaled, test_scaled = _standardize_sequences(train_seqs, [held_out])
        train_y = [s.labels for s in train_seqs]

        # Create model
        model = FlareGRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # Train with early stopping
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            loss = _train_epoch(model, optimizer, criterion, train_scaled, train_y)

            if loss < best_loss - 1e-4:
                best_loss = loss
                epochs_without_improvement = 0
                # Save best model state
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        # Load best model
        if best_state:  # type: ignore[possibly-unbound]
            model.load_state_dict(best_state)  # type: ignore[possibly-unbound]

        # Predict
        preds = _predict(model, test_scaled)
        y_prob = preds[0]
        y_true = held_out.labels
        y_pred = (y_prob >= 0.5).astype(int)

        results.folds.append(
            CVFold(
                patient_id=held_out.patient_id,
                y_true=y_true,
                y_prob=y_prob,
                y_pred=y_pred,
                n_instances=len(y_true),
            )
        )

    logger.info(
        "GRU LOPO-CV: %d folds, %d total instances",
        len(results.folds),
        sum(f.n_instances for f in results.folds),
    )
    return results


def compare_gru_vs_xgboost(
    gru_metrics: dict,
    xgb_metrics: dict,
) -> pd.DataFrame:
    """Compare GRU vs XGBoost performance.

    Returns
    -------
    DataFrame with side-by-side metrics.
    """
    rows = []
    for metrics in [xgb_metrics, gru_metrics]:
        rows.append(
            {
                "model": metrics.get("model", "unknown"),
                "auc": metrics.get("auc", np.nan),
                "sensitivity": metrics.get("sensitivity", np.nan),
                "specificity": metrics.get("specificity", np.nan),
                "ppv": metrics.get("ppv", np.nan),
                "npv": metrics.get("npv", np.nan),
                "mean_fold_auc": metrics.get("mean_fold_auc", np.nan),
            }
        )
    return pd.DataFrame(rows)


def save_gru_results(
    results: CVResults,
    metrics: dict,
    comparison: pd.DataFrame | None,
    output_dir: str | Path,
) -> None:
    """Save GRU results to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-patient predictions
    rows = []
    for fold in results.folds:
        for i in range(len(fold.y_true)):
            rows.append(
                {
                    "patient_id": fold.patient_id,
                    "timepoint": i,
                    "y_true": int(fold.y_true[i]),
                    "y_prob": float(fold.y_prob[i]),
                    "y_pred": int(fold.y_pred[i]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "gru_cv_predictions.csv", index=False)

    # Metrics summary
    summary = {
        k: v
        for k, v in metrics.items()
        if k not in ("roc_fpr", "roc_tpr", "calibration", "fold_aucs")
    }
    pd.Series(summary).to_csv(output_dir / "gru_cv_metrics.csv")

    if comparison is not None:
        comparison.to_csv(output_dir / "gru_vs_xgboost.csv", index=False)

    logger.info("Saved GRU results to %s", output_dir)
