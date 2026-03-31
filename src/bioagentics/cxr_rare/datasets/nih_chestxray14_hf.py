"""HuggingFace streaming Dataset for NIH ChestX-ray14.

Streams from BahaaEldin0/NIH-Chest-Xray-14 on HuggingFace Hub (Parquet
format) to avoid downloading the full 42GB dataset to disk.  Maps the
14 NIH labels to the 26-class CXR-LT label space.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from bioagentics.cxr_rare.config import LABEL_NAMES, NIH_TO_CXRLT
from bioagentics.cxr_rare.datasets.mimic_cxr import (
    default_eval_transform,
    default_train_transform,
)
from bioagentics.cxr_rare.datasets.nih_chestxray14 import nih_to_cxrlt_vector

logger = logging.getLogger(__name__)

HF_DATASET_ID = "BahaaEldin0/NIH-Chest-Xray-14"

# Known NIH-14 label strings (used for validation)
NIH_14_LABELS = frozenset(NIH_TO_CXRLT.keys())


def _parse_hf_labels(raw_labels: object) -> list[str]:
    """Extract NIH label strings from the HF example's label field.

    The HF dataset stores labels as a list of strings like
    ``["Infiltration", "Effusion"]``.  ``"No Finding"`` is filtered out
    so ``nih_to_cxrlt_vector`` correctly sets the No-Finding flag when
    the resulting list is empty.
    """
    if raw_labels is None:
        return []
    if isinstance(raw_labels, str):
        # Pipe-separated fallback (matches original NIH CSV format)
        parts = [s.strip() for s in raw_labels.split("|") if s.strip()]
    elif isinstance(raw_labels, (list, tuple)):
        parts = list(raw_labels)
    else:
        return []
    return [lbl for lbl in parts if lbl != "No Finding"]


class NIHChestXray14HFDataset(IterableDataset):
    """Streaming PyTorch IterableDataset for NIH ChestX-ray14 via HuggingFace.

    Streams images and labels from the ``BahaaEldin0/NIH-Chest-Xray-14``
    dataset without downloading the full 42 GB to disk.  Each yielded
    sample is a dict ``{"image": Tensor(3,224,224), "labels": Tensor(26)}``
    matching the interface expected by :class:`CXRTrainer`.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    transform : callable, optional
        Torchvision transform applied to the PIL image.  Defaults to the
        project's standard train or eval transform.
    shuffle_buffer_size : int
        Number of examples held in memory for approximate shuffling
        during training.  Set to 0 to disable (always disabled for
        non-train splits).
    seed : int
        Random seed for the shuffle buffer.
    label_column : str
        Name of the label column in the HF dataset.
    image_column : str
        Name of the image column in the HF dataset.
    """

    def __init__(
        self,
        split: str = "train",
        transform: object | None = None,
        shuffle_buffer_size: int = 1000,
        seed: int = 42,
        label_column: str = "labels",
        image_column: str = "image",
    ) -> None:
        super().__init__()
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size if split == "train" else 0
        self.seed = seed
        self.label_column = label_column
        self.image_column = image_column

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = default_train_transform()
        else:
            self.transform = default_eval_transform()

    def _load_stream(self):
        """Return a HuggingFace streaming dataset iterator."""
        from datasets import load_dataset

        return load_dataset(HF_DATASET_ID, split=self.split, streaming=True)

    def _process_example(self, example: dict) -> dict[str, torch.Tensor]:
        """Convert a single HF example to the trainer-expected format."""
        image = example[self.image_column].convert("RGB")
        raw_labels = example.get(self.label_column, [])
        nih_labels = _parse_hf_labels(raw_labels)
        label_vec = nih_to_cxrlt_vector(nih_labels)

        return {
            "image": self.transform(image),
            "labels": torch.from_numpy(label_vec),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        stream = self._load_stream()

        if self.shuffle_buffer_size > 0:
            yield from self._shuffled_iter(stream)
        else:
            for example in stream:
                yield self._process_example(example)

    def _shuffled_iter(self, stream) -> Iterator[dict[str, torch.Tensor]]:
        """Yield examples with approximate shuffling via a finite buffer."""
        buf: list[dict] = []
        rng = random.Random(self.seed)

        for example in stream:
            buf.append(example)
            if len(buf) >= self.shuffle_buffer_size:
                idx = rng.randint(0, len(buf) - 1)
                yield self._process_example(buf.pop(idx))

        # Flush remaining buffer in random order
        rng.shuffle(buf)
        for example in buf:
            yield self._process_example(example)

    def estimate_class_counts(self, n_samples: int = 5000) -> dict[str, int]:
        """Estimate per-class counts by sampling the stream.

        Iterates through up to *n_samples* examples and counts label
        occurrences, then extrapolates to the known dataset sizes
        (86 524 train / 25 596 test).

        Returns a dict of ``{class_name: estimated_count}``.
        """
        known_totals = {"train": 86_524, "test": 25_596}
        total_expected = known_totals.get(self.split, n_samples)

        counts = np.zeros(len(LABEL_NAMES), dtype=np.float64)
        seen = 0

        stream = self._load_stream()
        for example in stream:
            raw_labels = example.get(self.label_column, [])
            nih_labels = _parse_hf_labels(raw_labels)
            vec = nih_to_cxrlt_vector(nih_labels)
            counts += vec
            seen += 1
            if seen >= n_samples:
                break

        if seen == 0:
            return {name: 0 for name in LABEL_NAMES}

        scale = total_expected / seen
        estimated = (counts * scale).astype(int)
        return {name: int(estimated[i]) for i, name in enumerate(LABEL_NAMES)}
