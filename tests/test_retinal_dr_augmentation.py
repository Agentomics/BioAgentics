"""Tests for DR screening augmentation and dataset module."""

import cv2
import numpy as np
import pandas as pd
import torch

from bioagentics.diagnostics.retinal_dr_screening.augmentation import (
    DRDataset,
    cutmix_batch,
    cutmix_mixup_criterion,
    get_mobile_transform,
    get_train_transform,
    get_val_transform,
    mixup_batch,
)


def _make_splits_csv(tmp_path, n_images: int = 10, image_size: int = 64):
    """Create a mock splits CSV with actual image files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    records = []
    for i in range(n_images):
        # Create a colorful test image
        img = np.random.default_rng(i).integers(50, 200, (image_size, image_size, 3), dtype=np.uint8)
        img_path = img_dir / f"img_{i}.png"
        cv2.imwrite(str(img_path), img)

        split = "train" if i < 7 else ("val" if i < 9 else "test")
        records.append({
            "image_path": str(img_path),
            "dr_grade": i % 5,
            "dataset_source": "test",
            "original_filename": f"img_{i}.png",
            "split": split,
            "is_gradable": True,
        })

    df = pd.DataFrame(records)
    csv_path = tmp_path / "splits.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_get_train_transform():
    transform = get_train_transform(image_size=64)
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    result = transform(image=img)
    assert "image" in result
    assert isinstance(result["image"], torch.Tensor)
    assert result["image"].shape == (3, 64, 64)


def test_get_val_transform():
    transform = get_val_transform(image_size=64)
    img = np.random.default_rng(0).integers(0, 256, (100, 80, 3), dtype=np.uint8)
    result = transform(image=img)
    assert result["image"].shape == (3, 64, 64)


def test_get_mobile_transform():
    transform = get_mobile_transform(image_size=32)
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    result = transform(image=img)
    assert result["image"].shape == (3, 32, 32)


def test_train_transform_augments():
    """Two passes through train transform should produce different results."""
    transform = get_train_transform(image_size=64)
    img = np.random.default_rng(0).integers(50, 200, (64, 64, 3), dtype=np.uint8)
    r1 = transform(image=img)["image"]
    r2 = transform(image=img)["image"]
    # Very unlikely to be identical with random augmentation
    assert not torch.equal(r1, r2)


def test_val_transform_deterministic():
    """Val transform should produce identical results."""
    transform = get_val_transform(image_size=64)
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    r1 = transform(image=img)["image"]
    r2 = transform(image=img)["image"]
    assert torch.equal(r1, r2)


def test_dr_dataset_train(tmp_path):
    csv_path = _make_splits_csv(tmp_path)
    ds = DRDataset(csv_path, split="train", transform="train", image_size=64)
    assert len(ds) == 7  # 7 train images

    sample = ds[0]
    assert "image" in sample
    assert "label" in sample
    assert "dataset_source" in sample
    assert sample["image"].shape == (3, 64, 64)
    assert 0 <= sample["label"] <= 4


def test_dr_dataset_val(tmp_path):
    csv_path = _make_splits_csv(tmp_path)
    ds = DRDataset(csv_path, split="val", transform="val", image_size=64)
    assert len(ds) == 2


def test_dr_dataset_gradable_filter(tmp_path):
    csv_path = _make_splits_csv(tmp_path)
    # Modify CSV to mark some as ungradable
    df = pd.read_csv(csv_path)
    df.loc[0, "is_gradable"] = False
    df.to_csv(csv_path, index=False)

    ds_all = DRDataset(csv_path, split="train", transform="val", image_size=64, gradable_only=False)
    ds_gradable = DRDataset(csv_path, split="train", transform="val", image_size=64, gradable_only=True)
    assert len(ds_gradable) < len(ds_all)


def test_cutmix_batch():
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    mixed, labels_a, labels_b, lam = cutmix_batch(images, labels)

    assert mixed.shape == images.shape
    assert len(labels_a) == 4
    assert len(labels_b) == 4
    assert 0 <= lam <= 1


def test_mixup_batch():
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    mixed, labels_a, labels_b, lam = mixup_batch(images, labels)

    assert mixed.shape == images.shape
    assert len(labels_a) == 4
    assert len(labels_b) == 4
    assert 0 <= lam <= 1


def test_cutmix_mixup_criterion():
    criterion = torch.nn.CrossEntropyLoss()
    outputs = torch.randn(4, 5)  # 4 samples, 5 classes
    labels_a = torch.tensor([0, 1, 2, 3])
    labels_b = torch.tensor([1, 2, 3, 4])
    loss = cutmix_mixup_criterion(criterion, outputs, labels_a, labels_b, lam=0.7)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0
