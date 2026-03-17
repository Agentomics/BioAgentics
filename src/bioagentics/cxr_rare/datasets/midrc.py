"""PyTorch Dataset for CXR-LT 2026 MIDRC component.

Loads ~70K multi-institutional chest X-rays with the same CXR-LT
label space as the MIMIC-CXR loader. Includes institution metadata
for cross-institutional analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from bioagentics.cxr_rare.config import (
    DEFAULT_SPLIT,
    LABEL_NAMES,
    MIDRC_DIR,
    SplitConfig,
)
from bioagentics.cxr_rare.datasets.mimic_cxr import (
    _load_image,
    default_eval_transform,
    default_train_transform,
)

logger = logging.getLogger(__name__)

_LABELS_FILE = "labels.csv"
_METADATA_FILE = "metadata.csv"
_IMAGE_ID_COL = "image_id"
_INSTITUTION_COL = "institution"


class MIDRCDataset(Dataset):
    """PyTorch Dataset for CXR-LT 2026 MIDRC images.

    Parameters
    ----------
    data_dir : Path
        Root directory containing images and CSV files.
    split : str
        One of 'train', 'validate', 'test'.
    transform : transforms.Compose, optional
        Image transform pipeline.
    split_config : SplitConfig, optional
        Split configuration.
    label_names : list[str], optional
        Label columns. Defaults to CXR-LT set.
    """

    def __init__(
        self,
        data_dir: Path = MIDRC_DIR,
        split: str = "train",
        transform: transforms.Compose | None = None,
        split_config: SplitConfig = DEFAULT_SPLIT,
        label_names: list[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.label_names = label_names or LABEL_NAMES
        self.num_classes = len(self.label_names)
        self._label_to_idx = {n: i for i, n in enumerate(self.label_names)}

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = default_train_transform()
        else:
            self.transform = default_eval_transform()

        self._records = self._load_records(split_config)
        logger.info(
            "MIDRCDataset: split=%s, samples=%d, classes=%d",
            split, len(self._records), self.num_classes,
        )

    def _load_records(self, split_config: SplitConfig) -> list[dict]:
        """Load and filter records for the requested split."""
        labels_path = self.data_dir / _LABELS_FILE
        if not labels_path.exists():
            logger.warning(
                "Labels file not found: %s — dataset will be empty. "
                "Credentialed access may be pending.",
                labels_path,
            )
            return []

        labels_df = pd.read_csv(labels_path)

        # Load institution metadata if available
        meta_path = self.data_dir / _METADATA_FILE
        institution_map: dict[str, str] = {}
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path)
            if _IMAGE_ID_COL in meta_df.columns and _INSTITUTION_COL in meta_df.columns:
                institution_map = dict(zip(meta_df[_IMAGE_ID_COL], meta_df[_INSTITUTION_COL]))

        # Apply split filter
        if split_config.use_official_split and split_config.split_column in labels_df.columns:
            split_label = {
                "train": split_config.train_label,
                "validate": split_config.val_label,
                "val": split_config.val_label,
                "test": split_config.test_label,
            }[self.split]
            labels_df = labels_df[labels_df[split_config.split_column] == split_label]

        # Build records
        records = []
        files_dir = self.data_dir / "files"
        for _, row in labels_df.iterrows():
            image_id = str(row.get(_IMAGE_ID_COL, ""))
            image_path = self._find_image(files_dir, image_id)
            if image_path is None:
                continue

            label_vec = np.zeros(self.num_classes, dtype=np.float32)
            for lbl in self.label_names:
                if lbl in row and row[lbl] == 1:
                    label_vec[self._label_to_idx[lbl]] = 1.0

            records.append({
                "image_path": image_path,
                "labels": label_vec,
                "image_id": image_id,
                "institution": institution_map.get(image_id, "unknown"),
            })
        return records

    @staticmethod
    def _find_image(files_dir: Path, image_id: str) -> Path | None:
        """Find the image file for a given image ID."""
        for ext in (".jpg", ".png", ".dcm"):
            candidate = files_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self._records[idx]
        image = _load_image(record["image_path"])
        image_tensor = self.transform(image)
        label_tensor = torch.from_numpy(record["labels"])
        return {
            "image": image_tensor,
            "labels": label_tensor,
            "institution": record["institution"],
        }

    def get_class_counts(self) -> dict[str, int]:
        """Per-class positive sample counts."""
        if not self._records:
            return {name: 0 for name in self.label_names}
        all_labels = np.stack([r["labels"] for r in self._records])
        counts = all_labels.sum(axis=0).astype(int)
        return {name: int(counts[i]) for i, name in enumerate(self.label_names)}

    def get_institutions(self) -> list[str]:
        """List unique institutions in the dataset."""
        return sorted(set(r["institution"] for r in self._records))
