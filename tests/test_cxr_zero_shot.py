"""Tests for zero-shot CXR classification via BiomedCLIP prompts."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from bioagentics.cxr_rare.config import LABEL_NAMES, NUM_CLASSES
from bioagentics.cxr_rare.models.foundation import (
    CXR_PROMPT_TEMPLATES,
    LABEL_PROMPT_PHRASES,
    _build_prompt_texts,
    encode_text_prompts,
    zero_shot_classify,
    zero_shot_evaluate,
)


# ---------------------------------------------------------------------------
# Mock BiomedCLIP model and tokenizer for unit tests
# ---------------------------------------------------------------------------
class _MockBiomedCLIP(nn.Module):
    """Fake BiomedCLIP with encode_image / encode_text returning random vectors."""

    def __init__(self, feature_dim: int = 768) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # Need a parameter so next(model.parameters()) works
        self._dummy = nn.Linear(1, 1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # Deterministic per-image features based on pixel mean
        torch.manual_seed(0)
        return torch.randn(images.shape[0], self.feature_dim)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.randn(tokens.shape[0], self.feature_dim)


class _MockTokenizer:
    """Fake tokenizer returning integer token tensors."""

    def __call__(self, texts: list[str]) -> torch.Tensor:
        # Return dummy token IDs (shape: n_texts x 10)
        return torch.randint(0, 1000, (len(texts), 10))


class _FakeImageDataset(Dataset):
    """Minimal image dataset for testing."""

    def __init__(self, n_images: int = 16, n_labels: int = NUM_CLASSES) -> None:
        self.n_images = n_images
        self.n_labels = n_labels

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int) -> dict:
        torch.manual_seed(idx)
        return {
            "image": torch.randn(3, 224, 224),
            "labels": (torch.rand(self.n_labels) > 0.8).float(),
        }


# ---------------------------------------------------------------------------
# Tests: prompt building
# ---------------------------------------------------------------------------
class TestPromptBuilding:
    def test_default_templates_cover_all_labels(self) -> None:
        """Every CXR-LT label has a corresponding prompt phrase."""
        for label in LABEL_NAMES:
            assert label in LABEL_PROMPT_PHRASES, f"Missing phrase for '{label}'"

    def test_build_prompt_texts_default(self) -> None:
        conditions = ["Atelectasis", "Pneumothorax"]
        prompts = _build_prompt_texts(conditions)
        assert len(prompts) == 2
        assert len(prompts[0]) == len(CXR_PROMPT_TEMPLATES)
        assert "atelectasis" in prompts[0][0].lower()

    def test_build_prompt_texts_custom_template(self) -> None:
        conditions = ["Fibrosis"]
        custom = ["CXR with {}."]
        prompts = _build_prompt_texts(conditions, templates=custom)
        assert len(prompts[0]) == 1
        assert "fibrosis" in prompts[0][0].lower()

    def test_build_prompt_texts_unknown_condition(self) -> None:
        """Unknown conditions use the condition name lowercased."""
        conditions = ["Sarcoidosis"]
        prompts = _build_prompt_texts(conditions)
        assert all("sarcoidosis" in p for p in prompts[0])

    def test_num_templates(self) -> None:
        assert len(CXR_PROMPT_TEMPLATES) >= 3


# ---------------------------------------------------------------------------
# Tests: text encoding
# ---------------------------------------------------------------------------
class TestEncodeTextPrompts:
    def test_output_shape(self) -> None:
        model = _MockBiomedCLIP(feature_dim=128)
        tokenizer = _MockTokenizer()
        conditions = ["Atelectasis", "Pneumothorax", "Edema"]
        embeddings = encode_text_prompts(model, tokenizer, conditions)
        assert embeddings.shape == (3, 128)

    def test_embeddings_are_normalized(self) -> None:
        model = _MockBiomedCLIP(feature_dim=64)
        tokenizer = _MockTokenizer()
        conditions = ["Atelectasis"]
        embeddings = encode_text_prompts(model, tokenizer, conditions)
        norms = torch.norm(embeddings, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_all_labels_encode(self) -> None:
        model = _MockBiomedCLIP()
        tokenizer = _MockTokenizer()
        embeddings = encode_text_prompts(model, tokenizer, list(LABEL_NAMES))
        assert embeddings.shape[0] == NUM_CLASSES


# ---------------------------------------------------------------------------
# Tests: zero-shot classification
# ---------------------------------------------------------------------------
class TestZeroShotClassify:
    def test_output_shapes(self) -> None:
        model = _MockBiomedCLIP(feature_dim=128)
        tokenizer = _MockTokenizer()
        ds = _FakeImageDataset(n_images=8, n_labels=5)
        conditions = ["Atelectasis", "Pneumothorax", "Edema"]

        scores, labels, cond_names = zero_shot_classify(
            model, tokenizer, ds,
            conditions=conditions,
            batch_size=4, num_workers=0,
        )
        assert scores.shape == (8, 3)
        assert labels.shape == (8, 5)
        assert cond_names == conditions

    def test_scores_are_cosine_range(self) -> None:
        model = _MockBiomedCLIP(feature_dim=64)
        tokenizer = _MockTokenizer()
        ds = _FakeImageDataset(n_images=4)

        scores, _, _ = zero_shot_classify(
            model, tokenizer, ds,
            conditions=["Atelectasis"],
            batch_size=4, num_workers=0,
        )
        # Cosine similarity should be in [-1, 1]
        assert scores.min() >= -1.01
        assert scores.max() <= 1.01

    def test_default_conditions_are_label_names(self) -> None:
        model = _MockBiomedCLIP(feature_dim=64)
        tokenizer = _MockTokenizer()
        ds = _FakeImageDataset(n_images=2)

        _, _, cond_names = zero_shot_classify(
            model, tokenizer, ds,
            batch_size=2, num_workers=0,
        )
        assert cond_names == list(LABEL_NAMES)

    def test_unseen_conditions(self) -> None:
        """Can classify conditions not in the original label set."""
        model = _MockBiomedCLIP(feature_dim=64)
        tokenizer = _MockTokenizer()
        ds = _FakeImageDataset(n_images=4)
        unseen = ["Sarcoidosis", "Mesothelioma", "Aortic Dissection"]

        scores, _, cond_names = zero_shot_classify(
            model, tokenizer, ds,
            conditions=unseen,
            batch_size=4, num_workers=0,
        )
        assert scores.shape == (4, 3)
        assert cond_names == unseen


# ---------------------------------------------------------------------------
# Tests: zero-shot evaluation
# ---------------------------------------------------------------------------
class TestZeroShotEvaluate:
    def test_matched_evaluation(self, tmp_path) -> None:
        """Evaluation matches score columns to ground truth labels."""
        n_images = 50
        np.random.seed(42)
        # Scores for 3 conditions (2 are in LABEL_NAMES)
        conditions = ["Atelectasis", "Pneumothorax", "Sarcoidosis"]
        scores = np.random.rand(n_images, 3)
        labels = np.zeros((n_images, NUM_CLASSES))
        # Set some positive labels for matched conditions
        labels[:10, 2] = 1.0  # Atelectasis (index 2)
        labels[5:15, 14] = 1.0  # Pneumothorax (index 14)

        result = zero_shot_evaluate(scores, labels, conditions, output_dir=tmp_path)
        assert isinstance(result, dict)
        assert "per_class_auroc" in result or "summary" in result
