"""Training pipeline for deep spectrogram CNN.

Cross-validated training loop with AUC evaluation, matching the interface
of the classical model for downstream ensemble fusion.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from bioagentics.voice_pd.config import MODELS_DIR, TARGET_AUC
from bioagentics.voice_pd.deep.cnn_model import build_model

log = logging.getLogger(__name__)

# Memory-safe defaults (8GB machine)
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-4


class SpectrogramDataset(Dataset):
    """Dataset of precomputed mel-spectrogram tensors with binary labels.

    Args:
        spectrograms: Array of shape (N, 3, n_mels, time_frames).
        labels: Array of shape (N,) with values 0 or 1.
    """

    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray):
        self.spectrograms = spectrograms.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.spectrograms[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict probabilities on a dataset. Returns (y_true, y_prob)."""
    model.eval()
    all_probs = []
    all_labels = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)
        all_labels.append(y_batch.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def train_deep_model(
    spectrograms: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path | None = None,
    n_splits: int = 5,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    pretrained: bool = True,
) -> dict:
    """Train spectrogram CNN with stratified cross-validation.

    Args:
        spectrograms: Array (N, 3, n_mels, time_frames).
        labels: Binary labels (1=PD, 0=healthy).
        output_dir: Directory to save model and results.
        n_splits: Number of CV folds.
        epochs: Training epochs per fold.
        batch_size: Mini-batch size (keep small for 8GB RAM).
        lr: Learning rate.
        pretrained: Use ImageNet-pretrained MobileNetV2 backbone.

    Returns:
        Dict with mean_auc, std_auc, fold_aucs, best_model_state.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # 8GB RAM — stay on CPU to avoid GPU OOM
    dataset = SpectrogramDataset(spectrograms, labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs: list[float] = []
    best_auc = 0.0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(spectrograms, labels)):
        log.info("Fold %d/%d", fold + 1, n_splits)

        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx.tolist()),
            batch_size=batch_size, shuffle=False,
        )

        model = build_model(pretrained=pretrained).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            if (epoch + 1) % 5 == 0:
                log.info("  Epoch %d/%d loss=%.4f", epoch + 1, epochs, loss)

        y_true, y_prob = _evaluate(model, val_loader, device)
        auc = float(roc_auc_score(y_true, y_prob))
        fold_aucs.append(auc)
        log.info("Fold %d AUC=%.4f", fold + 1, auc)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Free memory between folds
        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    results = {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "fold_aucs": fold_aucs,
        "target_auc": TARGET_AUC,
        "meets_target": mean_auc >= TARGET_AUC,
        "n_samples": len(labels),
        "n_pd": int(labels.sum()),
        "n_healthy": int((1 - labels).sum()),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }

    # Save results JSON
    results_path = output_dir / "deep_cnn_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Mean AUC: %.4f +/- %.4f (target: %.2f)", mean_auc, std_auc, TARGET_AUC)

    # Save best model weights
    if best_state is not None:
        model_path = output_dir / "deep_cnn_model.pt"
        torch.save(best_state, model_path)
        log.info("Best model saved to %s", model_path)

    return {**results, "best_model_state": best_state}
