"""Tests for NIH ChestX-ray14 HuggingFace streaming dataset."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from bioagentics.cxr_rare.config import LABEL_NAMES, LABEL_TO_INDEX
from bioagentics.cxr_rare.datasets.nih_chestxray14_hf import (
    NIHChestXray14HFDataset,
    _parse_hf_labels,
)


def _make_fake_example(
    labels: list[str] | None = None,
    size: tuple[int, int] = (64, 64),
) -> dict:
    """Create a fake HF dataset example."""
    img = Image.fromarray(
        np.random.randint(0, 256, (*size, 3), dtype=np.uint8), mode="RGB"
    )
    return {
        "image": img,
        "labels": labels if labels is not None else [],
        "Patient Age": 55,
        "Patient Gender": "M",
        "View Position": "PA",
        "Patient ID": 1234,
    }


def _make_fake_stream(examples: list[dict]):
    """Create a fake HF streaming dataset (an iterable)."""

    class FakeStream:
        def __init__(self, data):
            self._data = data

        def __iter__(self):
            return iter(self._data)

    return FakeStream(examples)


# --- _parse_hf_labels tests ---


def test_parse_hf_labels_list() -> None:
    assert _parse_hf_labels(["Infiltration", "Effusion"]) == [
        "Infiltration",
        "Effusion",
    ]


def test_parse_hf_labels_filters_no_finding() -> None:
    assert _parse_hf_labels(["No Finding"]) == []


def test_parse_hf_labels_mixed() -> None:
    result = _parse_hf_labels(["No Finding", "Atelectasis"])
    assert result == ["Atelectasis"]


def test_parse_hf_labels_pipe_string() -> None:
    result = _parse_hf_labels("Infiltration|Effusion")
    assert result == ["Infiltration", "Effusion"]


def test_parse_hf_labels_none() -> None:
    assert _parse_hf_labels(None) == []


def test_parse_hf_labels_empty_list() -> None:
    assert _parse_hf_labels([]) == []


# --- NIHChestXray14HFDataset tests ---


@pytest.fixture
def fake_examples() -> list[dict]:
    return [
        _make_fake_example(["Infiltration", "Effusion"]),
        _make_fake_example(["Atelectasis"]),
        _make_fake_example(["No Finding"]),
        _make_fake_example(["Pneumonia", "Consolidation"]),
        _make_fake_example([]),
    ]


def test_dataset_yields_correct_format(fake_examples: list[dict]) -> None:
    ds = NIHChestXray14HFDataset(split="test", shuffle_buffer_size=0)
    stream = _make_fake_stream(fake_examples)

    with patch.object(ds, "_load_stream", return_value=stream):
        samples = list(ds)

    assert len(samples) == 5
    for sample in samples:
        assert "image" in sample
        assert "labels" in sample
        assert sample["image"].shape == (3, 224, 224)
        assert sample["labels"].shape == (len(LABEL_NAMES),)
        assert sample["image"].dtype == torch.float32
        assert sample["labels"].dtype == torch.float32


def test_label_mapping_infiltration_effusion(fake_examples: list[dict]) -> None:
    ds = NIHChestXray14HFDataset(split="test", shuffle_buffer_size=0)
    stream = _make_fake_stream([fake_examples[0]])

    with patch.object(ds, "_load_stream", return_value=stream):
        sample = next(iter(ds))

    labels = sample["labels"].numpy()
    # Infiltration -> "Lung Opacity", Effusion -> "Pleural Effusion"
    assert labels[LABEL_TO_INDEX["Lung Opacity"]] == 1.0
    assert labels[LABEL_TO_INDEX["Pleural Effusion"]] == 1.0
    # Other classes should be 0
    assert labels[LABEL_TO_INDEX["No Finding"]] == 0.0


def test_label_mapping_no_finding(fake_examples: list[dict]) -> None:
    ds = NIHChestXray14HFDataset(split="test", shuffle_buffer_size=0)
    stream = _make_fake_stream([fake_examples[2]])  # ["No Finding"]

    with patch.object(ds, "_load_stream", return_value=stream):
        sample = next(iter(ds))

    labels = sample["labels"].numpy()
    assert labels[LABEL_TO_INDEX["No Finding"]] == 1.0
    # All pathology labels should be 0
    assert labels.sum() == 1.0


def test_label_mapping_empty_labels(fake_examples: list[dict]) -> None:
    ds = NIHChestXray14HFDataset(split="test", shuffle_buffer_size=0)
    stream = _make_fake_stream([fake_examples[4]])  # []

    with patch.object(ds, "_load_stream", return_value=stream):
        sample = next(iter(ds))

    labels = sample["labels"].numpy()
    # Empty labels -> "No Finding"
    assert labels[LABEL_TO_INDEX["No Finding"]] == 1.0


def test_labels_are_binary(fake_examples: list[dict]) -> None:
    ds = NIHChestXray14HFDataset(split="test", shuffle_buffer_size=0)
    stream = _make_fake_stream(fake_examples)

    with patch.object(ds, "_load_stream", return_value=stream):
        for sample in ds:
            labels = sample["labels"].numpy()
            assert set(np.unique(labels)).issubset({0.0, 1.0})


def test_shuffle_buffer(fake_examples: list[dict]) -> None:
    ds = NIHChestXray14HFDataset(
        split="train", shuffle_buffer_size=3, seed=42
    )
    stream = _make_fake_stream(fake_examples)

    with patch.object(ds, "_load_stream", return_value=stream):
        samples = list(ds)

    # All samples should be yielded (just in different order)
    assert len(samples) == 5


def test_no_shuffle_for_eval() -> None:
    ds = NIHChestXray14HFDataset(split="test", shuffle_buffer_size=1000)
    # Even though shuffle_buffer_size=1000 was passed, split="test"
    # should force it to 0
    assert ds.shuffle_buffer_size == 0


def test_custom_transform(fake_examples: list[dict]) -> None:
    from torchvision import transforms

    custom_transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ]
    )
    ds = NIHChestXray14HFDataset(
        split="test", transform=custom_transform, shuffle_buffer_size=0
    )
    stream = _make_fake_stream([fake_examples[0]])

    with patch.object(ds, "_load_stream", return_value=stream):
        sample = next(iter(ds))

    assert sample["image"].shape == (3, 128, 128)


def test_grayscale_image_converted_to_rgb() -> None:
    gray_img = Image.fromarray(
        np.random.randint(0, 256, (64, 64), dtype=np.uint8), mode="L"
    )
    example = {"image": gray_img, "labels": ["Atelectasis"]}
    ds = NIHChestXray14HFDataset(split="test", shuffle_buffer_size=0)
    stream = _make_fake_stream([example])

    with patch.object(ds, "_load_stream", return_value=stream):
        sample = next(iter(ds))

    assert sample["image"].shape == (3, 224, 224)


def test_dataloader_compatibility(fake_examples: list[dict]) -> None:
    """Verify the dataset works with PyTorch DataLoader."""
    ds = NIHChestXray14HFDataset(split="test", shuffle_buffer_size=0)
    stream = _make_fake_stream(fake_examples)

    with patch.object(ds, "_load_stream", return_value=stream):
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        batch = next(iter(loader))

    assert batch["image"].shape == (2, 3, 224, 224)
    assert batch["labels"].shape == (2, len(LABEL_NAMES))
