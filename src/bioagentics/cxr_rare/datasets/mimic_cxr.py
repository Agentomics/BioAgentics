"""PyTorch Dataset for CXR-LT 2026 MIMIC-CXR component.

Loads chest X-ray images with multi-label annotations from the CXR-LT
long-tail label space. Supports DICOM, PNG, and JPEG image formats.
Designed to work when data arrives (credentialed access pending).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from bioagentics.cxr_rare.config import (
    DEFAULT_IMAGE_CONFIG,
    DEFAULT_SPLIT,
    LABEL_NAMES,
    MIMIC_CXR_DIR,
    ImageConfig,
    SplitConfig,
)

logger = logging.getLogger(__name__)

# Expected CSV column names
_LABELS_FILE = "cxr-lt-2026-labels.csv"
_SPLIT_FILE = "mimic-cxr-2.0.0-split.csv"
_DICOM_ID_COL = "dicom_id"
_STUDY_ID_COL = "study_id"
_SUBJECT_ID_COL = "subject_id"


def _try_load_dicom(path: Path) -> Image.Image:
    """Load a DICOM file and convert to PIL Image."""
    try:
        import pydicom
    except ImportError as e:
        raise ImportError("pydicom is required for DICOM loading: uv add pydicom") from e
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    # Normalize to 0-255
    if arr.max() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def _load_image(path: Path) -> Image.Image:
    """Load an image file (DICOM, PNG, or JPEG) as PIL Image."""
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        img = _try_load_dicom(path)
    else:
        img = Image.open(path)
        if img.mode == "I" or img.mode == "I;16":
            arr = np.array(img, dtype=np.float32)
            if arr.max() > 0:
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
            img = Image.fromarray(arr.astype(np.uint8), mode="L")
    # Convert grayscale to 3-channel for pretrained models
    return img.convert("RGB")


def default_train_transform(cfg: ImageConfig = DEFAULT_IMAGE_CONFIG) -> transforms.Compose:
    """Standard training augmentation pipeline."""
    return transforms.Compose([
        transforms.Resize(cfg.image_size + 32),
        transforms.RandomCrop(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])


def default_eval_transform(cfg: ImageConfig = DEFAULT_IMAGE_CONFIG) -> transforms.Compose:
    """Standard evaluation transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(cfg.image_size + 32),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])


class MIMICCXRDataset(Dataset):
    """PyTorch Dataset for CXR-LT 2026 MIMIC-CXR images.

    Parameters
    ----------
    data_dir : Path
        Root directory containing images and CSV files.
    split : str
        One of 'train', 'validate', 'test'.
    transform : transforms.Compose, optional
        Image transform pipeline. Defaults to train or eval transform.
    split_config : SplitConfig, optional
        Split configuration.
    label_names : list[str], optional
        Label columns to use. Defaults to full CXR-LT label set.
    """

    def __init__(
        self,
        data_dir: Path = MIMIC_CXR_DIR,
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

        # Load annotations and split
        self._records = self._load_records(split_config)
        logger.info(
            "MIMICCXRDataset: split=%s, samples=%d, classes=%d",
            split, len(self._records), self.num_classes,
        )

    def _load_records(self, split_config: SplitConfig) -> list[dict]:
        """Load and filter records for the requested split."""
        labels_path = self.data_dir / _LABELS_FILE
        if not labels_path.exists():
            logger.warning(
                "Labels file not found: %s — dataset will be empty. "
                "Data download may be pending.",
                labels_path,
            )
            return []

        labels_df = pd.read_csv(labels_path)

        # Apply split filter
        if split_config.use_official_split:
            split_path = self.data_dir / _SPLIT_FILE
            if split_path.exists():
                splits_df = pd.read_csv(split_path)
                split_label = {
                    "train": split_config.train_label,
                    "validate": split_config.val_label,
                    "val": split_config.val_label,
                    "test": split_config.test_label,
                }[self.split]
                split_ids = list(
                    splits_df.loc[
                        splits_df[split_config.split_column] == split_label,
                        _DICOM_ID_COL,
                    ]
                )
                labels_df = labels_df[labels_df[_DICOM_ID_COL].isin(split_ids)]
            else:
                logger.warning("Split file not found: %s", split_path)

        # Build record list
        records = []
        for _, row in labels_df.iterrows():
            image_path = self._resolve_image_path(row)
            if image_path is None:
                continue
            label_vec = np.zeros(self.num_classes, dtype=np.float32)
            for lbl in self.label_names:
                if lbl in row and row[lbl] == 1:
                    label_vec[self._label_to_idx[lbl]] = 1.0
            records.append({
                "image_path": image_path,
                "labels": label_vec,
                "dicom_id": row.get(_DICOM_ID_COL, ""),
            })
        return records

    def _resolve_image_path(self, row: pd.Series) -> Path | None:
        """Find the image file for a record, checking multiple formats."""
        dicom_id = row.get(_DICOM_ID_COL, "")
        subject_id = str(row.get(_SUBJECT_ID_COL, ""))
        study_id = str(row.get(_STUDY_ID_COL, ""))

        files_dir = self.data_dir / "files"

        # Try MIMIC-CXR directory structure: files/p{subject}/s{study}/{dicom_id}.{ext}
        prefix = f"p{subject_id[:2]}" if subject_id else ""
        for ext in (".jpg", ".png", ".dcm"):
            candidate = files_dir / prefix / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}{ext}"
            if candidate.exists():
                return candidate

        # Flat directory fallback
        for ext in (".jpg", ".png", ".dcm"):
            candidate = files_dir / f"{dicom_id}{ext}"
            if candidate.exists():
                return candidate

        return None

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self._records[idx]
        image = _load_image(record["image_path"])
        image_tensor = self.transform(image)
        label_tensor = torch.from_numpy(record["labels"])
        return {"image": image_tensor, "labels": label_tensor}

    def get_class_counts(self) -> dict[str, int]:
        """Return per-class positive sample counts."""
        if not self._records:
            return {name: 0 for name in self.label_names}
        all_labels = np.stack([r["labels"] for r in self._records])
        counts = all_labels.sum(axis=0).astype(int)
        return {name: int(counts[i]) for i, name in enumerate(self.label_names)}
