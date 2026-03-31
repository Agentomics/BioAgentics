"""Supervised contrastive fine-tuning for Wav2Vec 2.0 PD detection.

Implements SupConLoss (Khosla et al. 2020) to fine-tune wav2vec 2.0
embeddings so that same-class (PD or healthy) representations are pulled
together while different-class representations are pushed apart.

Ref: RD decision journal #2510.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from bioagentics.voice_pd.config import MODELS_DIR, SAMPLE_RATE
from bioagentics.voice_pd.deep.wav2vec_features import (
    EMBEDDING_DIM,
    load_wav2vec_model,
)
from bioagentics.voice_pd.utils import load_audio, pad_or_trim

log = logging.getLogger(__name__)

# Memory-safe defaults (8GB machine)
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-5
DEFAULT_PROJ_DIM = 128
TEMPERATURE = 0.07
MAX_AUDIO_SEC = 10.0


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    Pulls embeddings of the same class together and pushes different
    classes apart in the projection space.
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            features: L2-normalized projections, shape (batch, proj_dim).
            labels: Class labels, shape (batch,).

        Returns:
            Scalar loss.
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Log-sum-exp stability
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        pos_count = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

        # Only compute loss for samples that have at least one positive pair
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -mean_log_prob[valid].mean()
        return loss


class Wav2VecContrastiveModel(nn.Module):
    """Wav2Vec 2.0 with a projection head for contrastive learning.

    Architecture:
        Frozen/fine-tunable Wav2Vec 2.0 encoder ->
        Average pool over time ->
        Linear projection head (768 -> proj_dim) ->
        L2 normalize
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        proj_dim: int = DEFAULT_PROJ_DIM,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proj_head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, proj_dim),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_values: Raw waveform tensor, shape (batch, samples).

        Returns:
            L2-normalized projections, shape (batch, proj_dim).
        """
        outputs = self.encoder(input_values)
        hidden = outputs.last_hidden_state  # (B, T, 768)
        pooled = hidden.mean(dim=1)  # (B, 768)
        projected = self.proj_head(pooled)  # (B, proj_dim)
        return nn.functional.normalize(projected, dim=1)

    def extract_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract 768-d embeddings (before projection) for downstream use.

        Args:
            input_values: Raw waveform tensor, shape (batch, samples).

        Returns:
            Embeddings of shape (batch, 768).
        """
        with torch.no_grad():
            outputs = self.encoder(input_values)
            hidden = outputs.last_hidden_state
            return hidden.mean(dim=1)


class AudioDataset(Dataset):
    """Dataset loading raw audio waveforms for contrastive training."""

    def __init__(
        self,
        audio_paths: list[str | Path],
        labels: np.ndarray,
        max_sec: float = MAX_AUDIO_SEC,
    ):
        self.audio_paths = [Path(p) for p in audio_paths]
        self.labels = labels.astype(np.float32)
        self.max_samples = int(max_sec * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        y, _ = load_audio(self.audio_paths[idx])
        y = pad_or_trim(y, min(len(y), self.max_samples))
        # Pad short recordings to max_samples for batching
        if len(y) < self.max_samples:
            y = pad_or_trim(y, self.max_samples)
        return (
            torch.from_numpy(y),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def train_contrastive(
    audio_paths: list[str | Path],
    labels: np.ndarray,
    model_name: str = "facebook/wav2vec2-base",
    freeze_encoder: bool = False,
    proj_dim: int = DEFAULT_PROJ_DIM,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    output_dir: str | Path | None = None,
    val_fraction: float = 0.15,
) -> dict:
    """Train Wav2Vec 2.0 with supervised contrastive loss.

    Args:
        audio_paths: List of paths to WAV files.
        labels: Binary labels (0=healthy, 1=PD).
        model_name: HuggingFace model name.
        freeze_encoder: If True, freeze wav2vec encoder (linear probe only).
        proj_dim: Projection head output dimension.
        epochs: Training epochs.
        batch_size: Batch size (keep small for 8GB RAM).
        lr: Learning rate.
        output_dir: Where to save checkpoint.
        val_fraction: Fraction for validation split.

    Returns:
        Dict with training history and checkpoint path.
    """
    if output_dir is None:
        output_dir = MODELS_DIR / "wav2vec_contrastive"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    # Stratified train/val split
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    train_idx, val_idx = next(splitter.split(audio_paths, labels))

    full_dataset = AudioDataset(audio_paths, labels)
    train_loader = DataLoader(
        Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False,
    )

    model = Wav2VecContrastiveModel(model_name, proj_dim, freeze_encoder).to(device)
    criterion = SupConLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for waveforms, batch_labels in train_loader:
            waveforms = waveforms.to(device)
            batch_labels = batch_labels.to(device)

            projections = model(waveforms)
            loss = criterion(projections, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for waveforms, batch_labels in val_loader:
                waveforms = waveforms.to(device)
                batch_labels = batch_labels.to(device)
                projections = model(waveforms)
                loss = criterion(projections, batch_labels)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        log.info(
            "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f",
            epoch + 1, epochs, avg_train, avg_val,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Save final checkpoint and history
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Contrastive training complete. Best val_loss=%.4f", best_val_loss)
    return {
        "best_val_loss": best_val_loss,
        "checkpoint_dir": str(output_dir),
        "history": history,
    }


def evaluate_contrastive_vs_frozen(
    audio_paths: list[str | Path],
    labels: np.ndarray,
    checkpoint_path: str | Path | None = None,
    model_name: str = "facebook/wav2vec2-base",
) -> dict:
    """Compare frozen linear probe vs contrastive fine-tuned embeddings.

    Extracts embeddings with both approaches and trains a simple logistic
    regression classifier on each, reporting AUC for comparison.

    Args:
        audio_paths: List of WAV file paths.
        labels: Binary labels.
        checkpoint_path: Path to fine-tuned model checkpoint (.pt file).
        model_name: HuggingFace model name.

    Returns:
        Dict with AUC for frozen and fine-tuned approaches.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    device = "cpu"

    # Extract frozen embeddings
    log.info("Extracting frozen wav2vec embeddings...")
    from bioagentics.voice_pd.deep.wav2vec_features import (
        batch_extract_embeddings,
    )

    frozen_emb = batch_extract_embeddings(audio_paths, model_name)

    # Extract fine-tuned embeddings
    finetuned_emb = np.zeros_like(frozen_emb)
    if checkpoint_path is not None:
        log.info("Extracting fine-tuned contrastive embeddings...")
        model = Wav2VecContrastiveModel(model_name)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        model.to(device)

        dataset = AudioDataset(audio_paths, labels)
        loader = DataLoader(dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

        idx = 0
        with torch.no_grad():
            for waveforms, _ in loader:
                emb = model.extract_embeddings(waveforms.to(device))
                batch_n = emb.shape[0]
                finetuned_emb[idx : idx + batch_n] = emb.cpu().numpy()
                idx += batch_n

    # Cross-validated AUC comparison
    results = {}
    for name, X in [("frozen", frozen_emb), ("contrastive_finetuned", finetuned_emb)]:
        if X.sum() == 0:
            results[name] = {"auc": None, "note": "no embeddings available"}
            continue

        aucs = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X, labels):
            clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=1)
            clf.fit(X[train_idx], labels[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(labels[test_idx], probs))

        results[name] = {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "auc_folds": [float(a) for a in aucs],
        }

    log.info("Frozen AUC: %s", results.get("frozen", {}).get("auc_mean"))
    log.info("Fine-tuned AUC: %s", results.get("contrastive_finetuned", {}).get("auc_mean"))
    return results
