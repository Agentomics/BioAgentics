"""Tests for DR screening training module."""

import json

import cv2
import numpy as np
import pandas as pd
import torch

from bioagentics.diagnostics.retinal_dr_screening.training import (
    TrainConfig,
    compute_class_weights,
    compute_metrics,
    create_model,
    evaluate,
    train,
    train_one_epoch,
)


def _make_splits_csv(tmp_path, n_train=20, n_val=6, n_test=4, image_size=32):
    """Create mock splits CSV with actual images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    records = []
    total = n_train + n_val + n_test
    for i in range(total):
        img = np.random.default_rng(i).integers(50, 200, (image_size, image_size, 3), dtype=np.uint8)
        img_path = img_dir / f"img_{i}.png"
        cv2.imwrite(str(img_path), img)

        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

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


def test_create_model():
    model = create_model("efficientnet_b0", num_classes=5, pretrained=False)
    assert isinstance(model, torch.nn.Module)
    # Check output shape
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


def test_create_model_mobilenet():
    model = create_model("mobilenetv3_small_100", num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


def test_compute_metrics():
    labels = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    preds = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 3])  # one wrong
    probs = np.eye(5)[preds] * 0.8 + 0.04  # high confidence on predictions

    metrics = compute_metrics(labels, preds, probs)

    assert "accuracy" in metrics
    assert "qwk" in metrics
    assert "auc_referable" in metrics
    assert "referable_sensitivity" in metrics
    assert "referable_specificity" in metrics
    assert metrics["accuracy"] == 0.9  # 9/10 correct
    assert metrics["qwk"] > 0.8  # should be high


def test_compute_metrics_perfect():
    labels = np.array([0, 1, 2, 3, 4])
    preds = np.array([0, 1, 2, 3, 4])
    probs = np.eye(5)

    metrics = compute_metrics(labels, preds, probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["qwk"] == 1.0


def test_compute_class_weights(tmp_path):
    csv_path = _make_splits_csv(tmp_path)
    from bioagentics.diagnostics.retinal_dr_screening.augmentation import DRDataset

    ds = DRDataset(csv_path, split="train", transform="val", image_size=32)
    weights = compute_class_weights(ds)

    assert weights.shape == (5,)
    assert all(w > 0 for w in weights)
    # Sum should equal NUM_CLASSES (normalized)
    assert abs(weights.sum().item() - 5.0) < 0.1


def test_train_one_epoch():
    model = create_model("efficientnet_b0", num_classes=5, pretrained=False)
    device = torch.device("cpu")
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create a simple data loader with random data
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])
    dataset = torch.utils.data.TensorDataset(images, labels)

    # Wrap in dict-style loader
    class DictLoader:
        def __init__(self, images, labels):
            self.data = list(zip(
                images.split(2),
                labels.split(2),
            ))

        def __iter__(self):
            for imgs, lbls in self.data:
                yield {"image": imgs, "label": lbls, "dataset_source": ["test"] * len(lbls)}

        def __len__(self):
            return len(self.data)

    loader = DictLoader(images, labels)
    loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)

    assert isinstance(loss, float)
    assert loss > 0
    assert 0 <= acc <= 1


def test_evaluate():
    model = create_model("efficientnet_b0", num_classes=5, pretrained=False)
    device = torch.device("cpu")
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    class DictLoader:
        def __init__(self, images, labels):
            self.data = [(images, labels)]

        def __iter__(self):
            for imgs, lbls in self.data:
                yield {"image": imgs, "label": lbls, "dataset_source": ["test"] * len(lbls)}

        def __len__(self):
            return len(self.data)

    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])
    loader = DictLoader(images, labels)

    loss, acc, all_labels, all_preds, all_probs = evaluate(model, loader, criterion, device)

    assert isinstance(loss, float)
    assert len(all_labels) == 4
    assert len(all_preds) == 4
    assert all_probs.shape == (4, 5)


def test_full_training_pipeline(tmp_path):
    """End-to-end training pipeline with mock data."""
    csv_path = _make_splits_csv(tmp_path, n_train=10, n_val=5, n_test=5, image_size=32)
    output_dir = tmp_path / "models"

    config = TrainConfig(
        model_name="efficientnet_b0",
        pretrained=False,
        image_size=32,
        batch_size=4,
        num_workers=0,
        max_epochs=2,
        patience=10,
        use_amp=False,
        device="cpu",
    )

    result = train(csv_path, config, output_dir=output_dir)

    assert result.best_epoch > 0
    assert -1 < result.best_val_qwk <= 1
    assert len(result.history) == 2
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "train_config.json").exists()

    # Check history format
    with open(output_dir / "training_history.json") as f:
        history = json.load(f)
    assert len(history) == 2
    assert "train_loss" in history[0]
    assert "val_qwk" in history[0]


def test_train_config_defaults():
    config = TrainConfig()
    assert config.model_name == "efficientnet_b0"
    assert config.num_classes == 5
    assert config.use_amp is True
