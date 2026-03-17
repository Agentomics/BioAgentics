"""PyTorch Dataset for NIH ChestX-ray14.

Open-access dataset (112K CXRs, 14 findings) for supplementary training.
Maps 14 NIH labels to the CXR-LT label space.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from bioagentics.cxr_rare.config import (
    LABEL_NAMES,
    NIH_CHESTXRAY14_DIR,
    NIH_TO_CXRLT,
)
from bioagentics.cxr_rare.datasets.mimic_cxr import (
    _load_image,
    default_eval_transform,
    default_train_transform,
)

logger = logging.getLogger(__name__)

_LABELS_FILE = "Data_Entry_2017_v2020.csv"
_TRAIN_LIST = "train_val_list.txt"
_TEST_LIST = "test_list.txt"
_IMAGE_COL = "Image Index"
_LABEL_COL = "Finding Labels"


def _parse_nih_labels(label_str: str) -> list[str]:
    """Parse pipe-separated NIH label string."""
    if pd.isna(label_str) or label_str.strip() == "No Finding":
        return []
    return [s.strip() for s in label_str.split("|") if s.strip()]


def nih_to_cxrlt_vector(
    nih_labels: list[str],
    mapping: dict[str, str | None] | None = None,
) -> np.ndarray:
    """Convert NIH label list to CXR-LT binary label vector."""
    mapping = mapping or NIH_TO_CXRLT
    label_to_idx = {n: i for i, n in enumerate(LABEL_NAMES)}
    vec = np.zeros(len(LABEL_NAMES), dtype=np.float32)
    for lbl in nih_labels:
        cxrlt = mapping.get(lbl)
        if cxrlt and cxrlt in label_to_idx:
            vec[label_to_idx[cxrlt]] = 1.0
    if not nih_labels:
        # "No Finding"
        idx = label_to_idx.get("No Finding")
        if idx is not None:
            vec[idx] = 1.0
    return vec


class NIHChestXray14Dataset(Dataset):
    """PyTorch Dataset for NIH ChestX-ray14.

    Parameters
    ----------
    data_dir : Path
        Root directory containing images/ and CSVs.
    split : str
        'train' or 'test'. Uses official split files.
    transform : optional
        Image transform pipeline.
    """

    def __init__(
        self,
        data_dir: Path = NIH_CHESTXRAY14_DIR,
        split: str = "train",
        transform: object | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = default_train_transform()
        else:
            self.transform = default_eval_transform()

        self._records = self._load_records()
        logger.info("NIHChestXray14: split=%s, samples=%d", split, len(self._records))

    def _load_records(self) -> list[dict]:
        labels_path = self.data_dir / _LABELS_FILE
        if not labels_path.exists():
            logger.warning("Labels file not found: %s — dataset empty.", labels_path)
            return []

        df = pd.read_csv(labels_path)

        # Filter by official split
        list_file = _TRAIN_LIST if self.split == "train" else _TEST_LIST
        list_path = self.data_dir / list_file
        if list_path.exists():
            split_images = set(list_path.read_text().strip().split("\n"))
            df = df[df[_IMAGE_COL].isin(split_images)]

        images_dir = self.data_dir / "images"
        records = []
        for _, row in df.iterrows():
            image_name = str(row[_IMAGE_COL])
            # NIH images may be in numbered subdirs (images_001/ etc.) or flat
            image_path = images_dir / image_name
            if not image_path.exists():
                # Search subdirectories
                found = False
                for subdir in sorted(images_dir.iterdir()) if images_dir.exists() else []:
                    candidate = subdir / image_name
                    if candidate.exists():
                        image_path = candidate
                        found = True
                        break
                if not found:
                    continue

            nih_labels = _parse_nih_labels(str(row[_LABEL_COL]))
            label_vec = nih_to_cxrlt_vector(nih_labels)

            records.append({
                "image_path": image_path,
                "labels": label_vec,
                "image_name": image_name,
            })
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self._records[idx]
        image = _load_image(record["image_path"])
        return {
            "image": self.transform(image),
            "labels": torch.from_numpy(record["labels"]),
        }

    def get_class_counts(self) -> dict[str, int]:
        if not self._records:
            return {name: 0 for name in LABEL_NAMES}
        all_labels = np.stack([r["labels"] for r in self._records])
        counts = all_labels.sum(axis=0).astype(int)
        return {name: int(counts[i]) for i, name in enumerate(LABEL_NAMES)}
