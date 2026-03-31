"""Wav2Vec 2.0 + classical acoustic feature fusion for PD detection.

Concatenates wav2vec 2.0 embeddings (frozen or contrastive fine-tuned)
with classical acoustic features (jitter, shimmer, HNR, MFCCs) and
trains a lightweight MLP classifier on the fused representation.

Evaluates whether foundation model + classical fusion improves over
either modality alone.

Ref: RD decision journal #2510.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from bioagentics.voice_pd.config import MODELS_DIR
from bioagentics.voice_pd.deep.wav2vec_features import EMBEDDING_DIM

log = logging.getLogger(__name__)

# Memory-safe defaults (8GB machine)
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN = 256
DEFAULT_DROPOUT = 0.3


class FusionMLP(nn.Module):
    """Lightweight MLP classifier on concatenated wav2vec + classical features.

    Architecture:
        Concat(wav2vec_768d, classical_Nd) ->
        Linear -> ReLU -> Dropout ->
        Linear -> ReLU -> Dropout ->
        Linear(1) -> sigmoid
    """

    def __init__(
        self,
        wav2vec_dim: int = EMBEDDING_DIM,
        classical_dim: int = 100,
        hidden_dim: int = DEFAULT_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        input_dim = wav2vec_dim + classical_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, wav2vec_features: torch.Tensor, classical_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            wav2vec_features: (batch, 768) wav2vec embeddings.
            classical_features: (batch, N) classical features.

        Returns:
            (batch, 1) logits.
        """
        fused = torch.cat([wav2vec_features, classical_features], dim=1)
        return self.net(fused)


class FusionDataset(Dataset):
    """Dataset combining wav2vec embeddings and classical features."""

    def __init__(
        self,
        wav2vec_emb: np.ndarray,
        classical_feats: np.ndarray,
        labels: np.ndarray,
    ):
        self.wav2vec_emb = wav2vec_emb.astype(np.float32)
        self.classical_feats = classical_feats.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.wav2vec_emb[idx]),
            torch.from_numpy(self.classical_feats[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def _train_epoch(
    model: FusionMLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    losses = []
    for w2v, classical, labels in loader:
        w2v, classical, labels = w2v.to(device), classical.to(device), labels.to(device)
        logits = model(w2v, classical).squeeze(-1)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def _eval_auc(
    model: FusionMLP,
    loader: DataLoader,
    device: str,
) -> float:
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for w2v, classical, labels in loader:
            w2v, classical = w2v.to(device), classical.to(device)
            logits = model(w2v, classical).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    if len(np.unique(all_labels)) < 2:
        return 0.0
    return float(roc_auc_score(all_labels, all_probs))


def train_fusion_model(
    wav2vec_emb: np.ndarray,
    classical_feats: np.ndarray,
    labels: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    hidden_dim: int = DEFAULT_HIDDEN,
    output_dir: str | Path | None = None,
    val_fraction: float = 0.15,
) -> dict:
    """Train fusion MLP on wav2vec + classical features.

    Args:
        wav2vec_emb: (N, 768) wav2vec embeddings.
        classical_feats: (N, D) classical acoustic features.
        labels: (N,) binary labels (0=healthy, 1=PD).
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        hidden_dim: Hidden layer size.
        output_dir: Where to save model and results.
        val_fraction: Fraction for validation split.

    Returns:
        Dict with AUC, training history, and checkpoint path.
    """
    if output_dir is None:
        output_dir = MODELS_DIR / "wav2vec_fusion"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    train_idx, val_idx = next(splitter.split(wav2vec_emb, labels))

    dataset = FusionDataset(wav2vec_emb, classical_feats, labels)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
    )

    classical_dim = classical_feats.shape[1]
    model = FusionMLP(
        wav2vec_dim=EMBEDDING_DIM,
        classical_dim=classical_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_auc": []}
    best_val_auc = 0.0

    for epoch in range(epochs):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_auc = _eval_auc(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0:
            log.info(
                "Epoch %d/%d — loss=%.4f, val_auc=%.4f",
                epoch + 1, epochs, train_loss, val_auc,
            )

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Fusion training complete. Best val AUC=%.4f", best_val_auc)
    return {
        "best_val_auc": best_val_auc,
        "checkpoint_dir": str(output_dir),
        "history": history,
    }


def compare_modalities(
    wav2vec_emb: np.ndarray,
    classical_feats: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Compare wav2vec-only, classical-only, and fused models.

    Uses cross-validated logistic regression for fair comparison.

    Args:
        wav2vec_emb: (N, 768) wav2vec embeddings.
        classical_feats: (N, D) classical features.
        labels: (N,) binary labels.
        n_splits: Number of CV folds.

    Returns:
        Dict with AUC stats for each modality.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    fused = np.hstack([wav2vec_emb, classical_feats])
    modalities = {
        "wav2vec_only": wav2vec_emb,
        "classical_only": classical_feats,
        "fused": fused,
    }

    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, X in modalities.items():
        aucs = []
        for train_idx, test_idx in skf.split(X, labels):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=1)
            clf.fit(X_train, labels[train_idx])
            probs = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(labels[test_idx], probs))

        results[name] = {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "auc_folds": [float(a) for a in aucs],
        }
        log.info("%s: AUC=%.4f ± %.4f", name, results[name]["auc_mean"], results[name]["auc_std"])

    # Check if fusion improves over individual modalities
    fusion_auc = results["fused"]["auc_mean"]
    best_single = max(results["wav2vec_only"]["auc_mean"], results["classical_only"]["auc_mean"])
    results["fusion_improvement"] = fusion_auc - best_single
    log.info("Fusion improvement over best single modality: %.4f", results["fusion_improvement"])

    return results
