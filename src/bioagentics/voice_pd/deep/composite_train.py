"""Training pipeline for composite spectrogram CNN.

Trains MobileNetV2 on composite (5-vowel stacked) spectrograms with
ImageNet transfer learning, stratified cross-validation, class balancing,
and comprehensive metrics (AUROC, sensitivity, specificity).
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from bioagentics.voice_pd.config import MODELS_DIR, TARGET_AUC

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-4


class CompositeSpectrogramDataset(Dataset):
    """Dataset of composite spectrogram tensors with binary labels.

    Args:
        spectrograms: Array of shape (N, 3, n_mels * 5, time_frames).
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


class CompositeCNN(nn.Module):
    """MobileNetV2-based classifier for composite (5-vowel) spectrograms.

    Input: (batch, 3, n_mels * 5, time_frames) — taller than single-vowel.
    Output: (batch, 1) logits.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)

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


def build_composite_model(
    pretrained: bool = True, dropout: float = 0.3,
) -> CompositeCNN:
    """Factory function for the composite spectrogram CNN."""
    model = CompositeCNN(pretrained=pretrained, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("CompositeCNN: %.2fM parameters", n_params / 1e6)
    return model


def _make_balanced_sampler(labels: np.ndarray, indices: np.ndarray) -> WeightedRandomSampler:
    """Create a weighted sampler for class-balanced training."""
    subset_labels = labels[indices]
    class_counts = np.bincount(subset_labels.astype(int), minlength=2)
    weights = 1.0 / np.maximum(class_counts, 1).astype(float)
    sample_weights = weights[subset_labels.astype(int)]
    return WeightedRandomSampler(
        sample_weights.tolist(), len(sample_weights), replacement=True,
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
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
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)
        all_labels.append(y_batch.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute AUROC, sensitivity, specificity at threshold 0.5."""
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "auroc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def train_composite_model(
    spectrograms: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path | None = None,
    n_splits: int = 5,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    pretrained: bool = True,
) -> dict:
    """Train composite spectrogram CNN with stratified CV and class balancing.

    Args:
        spectrograms: Array (N, 3, n_mels * 5, time_frames).
        labels: Binary labels (1=PD, 0=healthy).
        output_dir: Save directory.
        n_splits: CV folds.
        epochs: Training epochs per fold.
        batch_size: Mini-batch size (keep small for 8GB RAM).
        lr: Learning rate.
        pretrained: Use ImageNet-pretrained backbone.

    Returns:
        Dict with mean_auc, std_auc, fold metrics, best_model_state.
    """
    from sklearn.model_selection import StratifiedKFold

    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # 8GB RAM constraint
    dataset = CompositeSpectrogramDataset(spectrograms, labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []
    best_auc = 0.0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(spectrograms, labels)):
        log.info("Fold %d/%d", fold + 1, n_splits)

        sampler = _make_balanced_sampler(labels, train_idx)
        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()),
            batch_size=batch_size,
            sampler=sampler,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
        )

        model = build_composite_model(pretrained=pretrained).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            if (epoch + 1) % 5 == 0:
                log.info("  Epoch %d/%d loss=%.4f", epoch + 1, epochs, loss)

        y_true, y_prob = _evaluate(model, val_loader, device)
        metrics = _compute_metrics(y_true, y_prob)
        fold_metrics.append(metrics)
        log.info("Fold %d AUROC=%.4f sens=%.3f spec=%.3f",
                 fold + 1, metrics["auroc"], metrics["sensitivity"], metrics["specificity"])

        if metrics["auroc"] > best_auc:
            best_auc = metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aucs = [m["auroc"] for m in fold_metrics]
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    results = {
        "model": "CompositeCNN_MobileNetV2",
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "fold_metrics": fold_metrics,
        "target_auc": TARGET_AUC,
        "meets_target": mean_auc >= TARGET_AUC,
        "n_samples": len(labels),
        "n_pd": int(labels.sum()),
        "n_healthy": int((1 - labels).sum()),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "class_balanced": True,
    }

    results_path = output_dir / "composite_cnn_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Composite CNN — Mean AUROC: %.4f +/- %.4f", mean_auc, std_auc)

    if best_state is not None:
        model_path = output_dir / "composite_cnn_model.pt"
        torch.save(best_state, model_path)
        log.info("Best composite model saved to %s", model_path)

    return {**results, "best_model_state": best_state}
