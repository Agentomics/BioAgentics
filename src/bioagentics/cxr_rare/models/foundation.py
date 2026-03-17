"""Foundation model feature extraction for CXR classification.

Extracts features from pretrained medical foundation models (BiomedCLIP,
CXR-FM) and trains lightweight balanced classifiers on frozen features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from bioagentics.cxr_rare.config import NUM_CLASSES, OUTPUT_DIR, TrainConfig
from bioagentics.cxr_rare.training.trainer import CXRTrainer

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Simple linear classifier on frozen features."""

    def __init__(self, in_features: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPProbe(nn.Module):
    """Two-layer MLP classifier on frozen features."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 512,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_biomedclip_encoder(device: torch.device | None = None) -> tuple[nn.Module, int]:
    """Load BiomedCLIP image encoder.

    Attempts to load from open_clip or huggingface transformers.
    Returns (encoder, feature_dim).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        model = model.visual
        model.to(device)
        model.eval()
        feature_dim = 768  # ViT-B/16
        logger.info("Loaded BiomedCLIP via open_clip (feature_dim=%d)", feature_dim)
        return model, feature_dim
    except (ImportError, Exception) as e:
        logger.warning("open_clip BiomedCLIP not available: %s", e)

    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        vision_model = model.vision_model  # type: ignore[attr-defined]
        vision_model.to(device)
        vision_model.eval()
        feature_dim = 768
        logger.info("Loaded BiomedCLIP via transformers (feature_dim=%d)", feature_dim)
        return vision_model, feature_dim
    except (ImportError, Exception) as e:
        logger.warning("transformers BiomedCLIP not available: %s", e)

    raise RuntimeError(
        "Could not load BiomedCLIP. Install open_clip or transformers: "
        "uv add open_clip_torch  OR  uv add transformers"
    )


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature vectors for entire dataset.

    Returns (features, labels) as numpy arrays.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device)
        features = encoder(images)
        # Handle different output formats
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state[:, 0]
        if features.dim() > 2:
            features = features.mean(dim=list(range(1, features.dim() - 1)))
        all_features.append(features.cpu().numpy())
        all_labels.append(batch["labels"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def cache_features(
    features: np.ndarray,
    labels: np.ndarray,
    cache_path: Path,
) -> None:
    """Save extracted features to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=features, labels=labels)
    logger.info("Cached features: %s (shape=%s)", cache_path, features.shape)


def load_cached_features(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cached features from disk."""
    data = np.load(cache_path)
    return data["features"], data["labels"]


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
    num_classes: int = NUM_CLASSES,
    config: TrainConfig | None = None,
) -> dict:
    """Train a linear probe classifier on frozen features.

    Returns evaluation results.
    """
    from bioagentics.cxr_rare.evaluation.metrics import evaluate_and_save

    config = config or TrainConfig(
        epochs=50, batch_size=256, learning_rate=0.01,
        num_workers=0, mixed_precision=False,
    )

    train_ds = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_features).float(),
        torch.from_numpy(val_labels).float(),
    )

    # Wrap TensorDataset to return dict format
    class _DictWrapper(Dataset):
        def __init__(self, ds: TensorDataset) -> None:
            self.ds = ds
        def __len__(self) -> int:
            return len(self.ds)
        def __getitem__(self, idx: int) -> dict:
            feat, lbl = self.ds[idx]
            return {"image": feat, "labels": lbl}

    model = LinearProbe(features.shape[1], num_classes)
    trainer = CXRTrainer(
        model=model,
        train_dataset=_DictWrapper(train_ds),
        val_dataset=_DictWrapper(val_ds),
        loss_fn=nn.BCEWithLogitsLoss(),
        config=config,
        output_dir=output_dir,
        experiment_name="linear_probe",
    )
    trainer.train()
    trainer.load_best_checkpoint()

    # Evaluate
    model.eval()
    device = trainer.device
    with torch.no_grad():
        all_scores = torch.sigmoid(model(torch.from_numpy(val_features).float().to(device))).cpu().numpy()

    return evaluate_and_save(val_labels, all_scores, output_dir, "linear_probe")
