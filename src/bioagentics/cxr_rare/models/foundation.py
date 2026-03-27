"""Foundation model feature extraction and zero-shot classification for CXR.

Extracts features from pretrained medical foundation models (BiomedCLIP,
CXR-FM) and trains lightweight balanced classifiers on frozen features.
Also provides zero-shot classification via BiomedCLIP text-image similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from bioagentics.cxr_rare.config import LABEL_NAMES, NUM_CLASSES, OUTPUT_DIR, TrainConfig
from bioagentics.cxr_rare.training.trainer import CXRTrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text prompt templates for zero-shot CXR classification
# ---------------------------------------------------------------------------
# Multiple prompts per condition improve zero-shot accuracy via ensembling.
CXR_PROMPT_TEMPLATES: list[str] = [
    "A chest X-ray showing {}.",
    "Chest radiograph with findings of {}.",
    "An anteroposterior chest X-ray demonstrating {}.",
    "Frontal chest X-ray with evidence of {}.",
    "Radiological image showing {} in the thorax.",
]

# Condition names phrased for natural language prompts (maps label → phrase).
# Keys match LABEL_NAMES from config; values are radiologist-style descriptions.
LABEL_PROMPT_PHRASES: dict[str, str] = {
    "No Finding": "no abnormal findings",
    "Support Devices": "support devices such as tubes and lines",
    "Atelectasis": "atelectasis",
    "Pleural Effusion": "pleural effusion",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "lung opacity",
    "Edema": "pulmonary edema",
    "Consolidation": "consolidation",
    "Pneumonia": "pneumonia",
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinum",
    "Lung Lesion": "a lung lesion",
    "Fracture": "a rib fracture",
    "Pleural Other": "pleural abnormality",
    "Calcification of the Aorta": "aortic calcification",
    "Pneumothorax": "pneumothorax",
    "Subcutaneous Emphysema": "subcutaneous emphysema",
    "Tortuous Aorta": "a tortuous aorta",
    "Pneumomediastinum": "pneumomediastinum",
    "Pneumoperitoneum": "pneumoperitoneum",
    "Hernia": "a diaphragmatic hernia",
    "Mass": "a pulmonary mass",
    "Nodule": "a pulmonary nodule",
    "Fibrosis": "pulmonary fibrosis",
    "Emphysema": "emphysema",
    "Foreign Body": "a foreign body",
    "Interstitial Lung Disease": "interstitial lung disease",
}


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


def load_biomedclip_model(
    device: torch.device | None = None,
) -> tuple:
    """Load the full BiomedCLIP model (vision + text encoders) and tokenizer.

    Returns (model, tokenizer, feature_dim).
    The model object exposes both vision and text encoding.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        model.to(device)
        model.eval()
        feature_dim = 768
        logger.info("Loaded full BiomedCLIP via open_clip (feature_dim=%d)", feature_dim)
        return model, tokenizer, feature_dim
    except (ImportError, Exception) as e:
        logger.warning("open_clip BiomedCLIP not available: %s", e)

    raise RuntimeError(
        "Could not load full BiomedCLIP model. Install open_clip: "
        "uv add --optional research open_clip_torch"
    )


def _build_prompt_texts(
    conditions: list[str],
    templates: list[str] | None = None,
    label_phrases: dict[str, str] | None = None,
) -> list[list[str]]:
    """Build prompt text lists for each condition.

    Returns a list of length len(conditions), where each element is a list of
    prompt strings (one per template).
    """
    templates = templates or CXR_PROMPT_TEMPLATES
    phrases = label_phrases or LABEL_PROMPT_PHRASES

    all_prompts: list[list[str]] = []
    for cond in conditions:
        phrase = phrases.get(cond, cond.lower())
        prompts = [t.format(phrase) for t in templates]
        all_prompts.append(prompts)
    return all_prompts


@torch.no_grad()
def encode_text_prompts(
    model: nn.Module,
    tokenizer: object,
    conditions: list[str],
    templates: list[str] | None = None,
    label_phrases: dict[str, str] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode condition text prompts into averaged, normalized embeddings.

    For each condition, encodes all template variations and averages them
    to produce a single text embedding per condition.

    Returns tensor of shape (n_conditions, feature_dim), L2-normalized.
    """
    device = device or next(model.parameters()).device
    prompt_lists = _build_prompt_texts(conditions, templates, label_phrases)

    class_embeddings: list[torch.Tensor] = []
    for prompts in prompt_lists:
        tokens = tokenizer(prompts)  # type: ignore[operator]
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(device)
        else:
            tokens = torch.tensor(tokens).to(device)

        text_features = model.encode_text(tokens)
        # Average across templates, then normalize
        mean_emb = text_features.mean(dim=0)
        mean_emb = F.normalize(mean_emb, dim=-1)
        class_embeddings.append(mean_emb)

    return torch.stack(class_embeddings)  # (n_conditions, D)


@torch.no_grad()
def zero_shot_classify(
    model: nn.Module,
    tokenizer: object,
    dataset: Dataset,
    conditions: list[str] | None = None,
    templates: list[str] | None = None,
    label_phrases: dict[str, str] | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Zero-shot CXR classification using BiomedCLIP text-image similarity.

    For each image, computes cosine similarity against text embeddings of
    all conditions. This enables detecting conditions not in the training
    set, which has high clinical value for rare disease detection.

    Parameters
    ----------
    model : nn.Module
        Full BiomedCLIP model with encode_image and encode_text methods.
    tokenizer : object
        BiomedCLIP tokenizer (from open_clip.get_tokenizer).
    dataset : Dataset
        Image dataset returning dict with 'image' key.
    conditions : list[str], optional
        Condition names to classify. Defaults to LABEL_NAMES (26 CXR-LT classes).
    templates : list[str], optional
        Prompt templates. Defaults to CXR_PROMPT_TEMPLATES.
    label_phrases : dict[str, str], optional
        Condition→phrase mapping. Defaults to LABEL_PROMPT_PHRASES.
    batch_size : int
        Batch size for image encoding.
    num_workers : int
        DataLoader workers.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    scores : np.ndarray, shape (n_images, n_conditions)
        Cosine similarity scores (range [-1, 1]).
    labels : np.ndarray, shape (n_images, n_labels) or empty
        Ground truth labels if available in dataset, else empty array.
    condition_names : list[str]
        Ordered list of condition names corresponding to score columns.
    """
    device = device or next(model.parameters()).device
    conditions = conditions or list(LABEL_NAMES)

    # Encode text prompts
    text_embeddings = encode_text_prompts(
        model, tokenizer, conditions, templates, label_phrases, device
    )  # (n_conditions, D)

    # Encode images and compute similarities
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        # Cosine similarity: (batch, D) @ (D, n_conditions) → (batch, n_conditions)
        similarity = image_features @ text_embeddings.T
        all_scores.append(similarity.cpu().numpy())

        if "labels" in batch:
            all_labels.append(batch["labels"].numpy())

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    logger.info(
        "Zero-shot classification: %d images × %d conditions, "
        "score range [%.3f, %.3f]",
        scores.shape[0], scores.shape[1], scores.min(), scores.max(),
    )
    return scores, labels, conditions


def zero_shot_evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    condition_names: list[str],
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Evaluate zero-shot classification scores against ground truth.

    Only evaluates conditions that exist in the ground-truth label space
    (matched by index in LABEL_NAMES).

    Returns evaluation metrics dict.
    """
    from bioagentics.cxr_rare.evaluation.metrics import evaluate_and_save

    # Map zero-shot condition columns to ground truth label columns
    gt_indices: list[int] = []
    score_indices: list[int] = []
    matched_names: list[str] = []

    label_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
    for sc_idx, cond in enumerate(condition_names):
        if cond in label_to_idx:
            gt_indices.append(label_to_idx[cond])
            score_indices.append(sc_idx)
            matched_names.append(cond)

    if not gt_indices:
        logger.warning("No conditions matched ground truth labels")
        return {}

    # Extract matched columns
    matched_labels = labels[:, gt_indices]
    matched_scores = scores[:, score_indices]

    logger.info(
        "Evaluating %d/%d conditions matched to ground truth",
        len(matched_names), len(condition_names),
    )

    return evaluate_and_save(
        matched_labels, matched_scores, output_dir, "zero_shot_biomedclip"
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
