"""Tests for DR screening knowledge distillation module."""

import cv2
import numpy as np
import pandas as pd
import torch

from bioagentics.diagnostics.retinal_dr_screening.distillation import (
    DistillConfig,
    DistillationLoss,
    distill,
)
from bioagentics.diagnostics.retinal_dr_screening.training import create_model


def _make_splits_and_teacher(tmp_path, n_train=10, n_val=5):
    """Create mock splits CSV, images, and a teacher checkpoint."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    records = []
    for i in range(n_train + n_val):
        img = np.random.default_rng(i).integers(50, 200, (32, 32, 3), dtype=np.uint8)
        img_path = img_dir / f"img_{i}.png"
        cv2.imwrite(str(img_path), img)
        split = "train" if i < n_train else "val"
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

    # Create teacher checkpoint
    teacher = create_model("efficientnet_b4", num_classes=5, pretrained=False)
    teacher_path = tmp_path / "teacher.pt"
    torch.save({
        "epoch": 10,
        "model_state_dict": teacher.state_dict(),
        "optimizer_state_dict": {},
        "val_qwk": 0.85,
        "config": {"model_name": "efficientnet_b4", "num_classes": 5},
    }, teacher_path)

    return csv_path, teacher_path


def test_distillation_loss():
    criterion = DistillationLoss(temperature=4.0, alpha=0.7)

    student_logits = torch.randn(4, 5)
    teacher_logits = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])

    total, ce, kd = criterion(student_logits, teacher_logits, labels)

    assert total.ndim == 0  # scalar
    assert ce.ndim == 0
    assert kd.ndim == 0
    assert total.item() > 0


def test_distillation_loss_matching_logits():
    """When student matches teacher, KD loss should be near zero."""
    criterion = DistillationLoss(temperature=4.0, alpha=0.7)

    logits = torch.tensor([[2.0, 0.5, -1.0, 0.0, -0.5]])
    labels = torch.tensor([0])

    _, _, kd = criterion(logits, logits, labels)
    assert kd.item() < 0.01  # KL divergence should be near zero


def test_distillation_loss_alpha():
    """Higher alpha should weight KD loss more."""
    student_logits = torch.randn(4, 5)
    teacher_logits = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])

    loss_low_alpha = DistillationLoss(temperature=4.0, alpha=0.1)
    loss_high_alpha = DistillationLoss(temperature=4.0, alpha=0.9)

    total_low, _, _ = loss_low_alpha(student_logits, teacher_logits, labels)
    total_high, _, _ = loss_high_alpha(student_logits, teacher_logits, labels)

    # They should produce different loss values
    assert total_low.item() != total_high.item()


def test_distill_config_defaults():
    config = DistillConfig()
    assert config.teacher_model == "efficientnet_b4"
    assert config.student_model == "mobilenetv3_small_100"
    assert config.temperature == 4.0
    assert config.alpha == 0.7


def test_distill_pipeline(tmp_path):
    """End-to-end distillation pipeline with mock data."""
    csv_path, teacher_path = _make_splits_and_teacher(tmp_path)
    output_dir = tmp_path / "output"

    config = DistillConfig(
        image_size=32,
        batch_size=4,
        num_workers=0,
        max_epochs=2,
        patience=10,
        device="cpu",
    )

    result = distill(csv_path, teacher_path, config, output_dir)

    assert result["best_epoch"] > 0
    assert -1 < result["best_val_qwk"] <= 1
    assert len(result["history"]) == 2
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "distill_history.json").exists()

    # Check history has distillation-specific fields
    assert "train_ce" in result["history"][0]
    assert "train_kd" in result["history"][0]
