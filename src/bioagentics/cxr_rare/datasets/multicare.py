"""PyTorch Dataset and downloader for MultiCaRe dataset.

Open-access dataset from PubMed case reports (130K+ images, 140+ categories).
Supplements rare cases for tail-class augmentation. Maps MultiCaRe categories
to CXR-LT label space where possible.
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
    LABEL_TO_INDEX,
    MULTICARE_DIR,
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
_CATEGORY_COL = "category"

# Mapping from MultiCaRe's 140+ categories to CXR-LT labels.
# Only thoracic/CXR-relevant categories are mapped; others are ignored.
MULTICARE_TO_CXRLT: dict[str, str] = {
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pleural_effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "emphysema": "Emphysema",
    "fibrosis": "Fibrosis",
    "hernia": "Hernia",
    "mass": "Mass",
    "nodule": "Nodule",
    "fracture": "Fracture",
    "foreign_body": "Foreign Body",
    "interstitial_lung_disease": "Interstitial Lung Disease",
    "subcutaneous_emphysema": "Subcutaneous Emphysema",
    "pneumomediastinum": "Pneumomediastinum",
    "pneumoperitoneum": "Pneumoperitoneum",
}


def map_categories_to_cxrlt(
    categories: list[str],
    mapping: dict[str, str] | None = None,
) -> np.ndarray:
    """Convert MultiCaRe category list to CXR-LT binary label vector."""
    mapping = mapping or MULTICARE_TO_CXRLT
    label_vec = np.zeros(len(LABEL_NAMES), dtype=np.float32)
    for cat in categories:
        cat_lower = cat.lower().strip().replace(" ", "_")
        cxrlt_name = mapping.get(cat_lower)
        if cxrlt_name and cxrlt_name in LABEL_TO_INDEX:
            label_vec[LABEL_TO_INDEX[cxrlt_name]] = 1.0
    return label_vec


class MultiCaReDataset(Dataset):
    """PyTorch Dataset for MultiCaRe case report images.

    Parameters
    ----------
    data_dir : Path
        Root directory containing images/ and labels.csv.
    split : str
        'train', 'validate', or 'test'.
    transform : optional
        Image transform pipeline.
    """

    def __init__(
        self,
        data_dir: Path = MULTICARE_DIR,
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
        logger.info("MultiCaReDataset: split=%s, samples=%d", split, len(self._records))

    def _load_records(self) -> list[dict]:
        labels_path = self.data_dir / _LABELS_FILE
        if not labels_path.exists():
            logger.warning("Labels file not found: %s — dataset empty.", labels_path)
            return []

        df = pd.read_csv(labels_path)

        # Filter by split if column exists
        if "split" in df.columns:
            split_map = {"train": "train", "validate": "val", "val": "val", "test": "test"}
            df = df[df["split"] == split_map.get(self.split, self.split)]

        images_dir = self.data_dir / "images"
        records = []
        for _, row in df.iterrows():
            image_id = str(row.get(_IMAGE_ID_COL, ""))
            # Find image
            image_path = None
            for ext in (".jpg", ".png"):
                candidate = images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue

            # Parse categories (comma-separated or single)
            raw_cats = str(row.get(_CATEGORY_COL, ""))
            categories = [c.strip() for c in raw_cats.split(",") if c.strip()]
            label_vec = map_categories_to_cxrlt(categories)

            records.append({
                "image_path": image_path,
                "labels": label_vec,
                "image_id": image_id,
                "categories": categories,
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
