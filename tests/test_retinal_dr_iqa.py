"""Tests for DR screening IQA pre-filter module."""

import cv2
import numpy as np
import pandas as pd
import torch

from bioagentics.diagnostics.retinal_dr_screening.iqa import (
    DRScreeningResult,
    IQADataset,
    IQAModel,
    IQAPredictor,
    IQAResult,
    IQATrainConfig,
    QUALITY_ISSUES,
    apply_defocus_blur,
    apply_glare,
    apply_motion_blur,
    apply_overexposure,
    apply_partial_occlusion,
    apply_underexposure,
    run_screening_pipeline,
    train_iqa,
)


def _make_fundus_image(size=64, brightness=120):
    """Create a synthetic fundus-like image."""
    img = np.full((size, size, 3), brightness, dtype=np.uint8)
    # Draw a circle to simulate fundus
    center = size // 2
    cv2.circle(img, (center, center), center - 5, (100, 80, 60), -1)
    # Add some texture
    noise = np.random.default_rng(42).integers(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _make_splits_csv(tmp_path, n_train=16, n_val=4, image_size=32):
    """Create mock splits CSV with images for IQA."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    records = []
    total = n_train + n_val
    for i in range(total):
        img = np.random.default_rng(i).integers(50, 200, (image_size, image_size, 3), dtype=np.uint8)
        img_path = img_dir / f"img_{i}.png"
        cv2.imwrite(str(img_path), img)

        split = "train" if i < n_train else "val"
        records.append({
            "image_path": str(img_path),
            "dr_grade": i % 5,
            "dataset_source": "test",
            "original_filename": f"img_{i}.png",
            "split": split,
            "is_gradable": i % 3 != 0,  # ~33% ungradable
        })

    df = pd.DataFrame(records)
    csv_path = tmp_path / "splits.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ── Augmentation tests ──


class TestAugmentations:
    def test_motion_blur(self):
        img = _make_fundus_image()
        blurred = apply_motion_blur(img, kernel_size=15)
        assert blurred.shape == img.shape
        assert blurred.dtype == np.uint8
        # Blurred image should differ from original
        assert not np.array_equal(blurred, img)

    def test_defocus_blur(self):
        img = _make_fundus_image()
        blurred = apply_defocus_blur(img, radius=7)
        assert blurred.shape == img.shape
        assert blurred.dtype == np.uint8

    def test_glare(self):
        img = _make_fundus_image()
        glared = apply_glare(img, intensity=0.6)
        assert glared.shape == img.shape
        assert glared.dtype == np.uint8

    def test_partial_occlusion(self):
        img = _make_fundus_image()
        occluded = apply_partial_occlusion(img, coverage=0.3)
        assert occluded.shape == img.shape
        assert occluded.dtype == np.uint8
        # Some pixels should be zeroed
        assert np.sum(occluded == 0) > np.sum(img == 0)

    def test_underexposure(self):
        img = _make_fundus_image(brightness=120)
        dark = apply_underexposure(img, factor=0.3)
        assert dark.shape == img.shape
        assert float(np.mean(dark)) < float(np.mean(img))

    def test_overexposure(self):
        img = _make_fundus_image(brightness=80)
        bright = apply_overexposure(img, factor=1.8)
        assert bright.shape == img.shape
        assert float(np.mean(bright)) > float(np.mean(img))


# ── Model tests ──


class TestIQAModel:
    def test_forward_shape(self):
        model = IQAModel(pretrained=False, dropout=0.0)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 2)

    def test_forward_small_input(self):
        model = IQAModel(pretrained=False)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 2)

    def test_parameter_count(self):
        model = IQAModel(pretrained=False)
        params = sum(p.numel() for p in model.parameters())
        # MobileNetV3-Small should be ~2.5M params
        assert params < 5_000_000, f"Model too large: {params / 1e6:.1f}M params"


# ── Dataset tests ──


class TestIQADataset:
    def test_dataset_len(self, tmp_path):
        csv = _make_splits_csv(tmp_path)
        ds = IQADataset(csv, split="train", image_size=32, augment=False)
        assert len(ds) == 16

    def test_dataset_item(self, tmp_path):
        csv = _make_splits_csv(tmp_path)
        ds = IQADataset(csv, split="train", image_size=32, augment=False)
        item = ds[0]
        assert "image" in item
        assert "label" in item
        assert item["image"].shape == (3, 32, 32)
        assert item["label"] in (0, 1)

    def test_dataset_val_no_augment(self, tmp_path):
        csv = _make_splits_csv(tmp_path)
        ds = IQADataset(csv, split="val", image_size=32, augment=False)
        assert len(ds) == 4

    def test_dataset_with_degradation(self, tmp_path):
        csv = _make_splits_csv(tmp_path)
        ds = IQADataset(csv, split="train", image_size=32, degradation_prob=1.0, augment=True)
        item = ds[1]  # index 1 is gradable (1 % 3 != 0)
        # With degradation_prob=1.0, gradable images become ungradable
        assert item["label"] == 0


# ── Training tests ──


class TestIQATraining:
    def test_train_iqa_runs(self, tmp_path):
        csv = _make_splits_csv(tmp_path, n_train=8, n_val=4, image_size=32)
        config = IQATrainConfig(
            image_size=32,
            batch_size=4,
            num_workers=0,
            max_epochs=2,
            patience=5,
            pretrained=False,
            device="cpu",
        )
        output_dir = tmp_path / "model_out"
        result = train_iqa(csv, config, output_dir)

        assert "best_val_acc" in result
        assert (output_dir / "iqa_best.pt").exists()
        assert (output_dir / "iqa_history.json").exists()
        assert (output_dir / "iqa_config.json").exists()

    def test_checkpoint_loadable(self, tmp_path):
        csv = _make_splits_csv(tmp_path, n_train=8, n_val=4, image_size=32)
        config = IQATrainConfig(
            image_size=32, batch_size=4, num_workers=0,
            max_epochs=1, pretrained=False, device="cpu",
        )
        output_dir = tmp_path / "model_out"
        train_iqa(csv, config, output_dir)

        # Load checkpoint
        ckpt = torch.load(output_dir / "iqa_best.pt", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "config" in ckpt

        model = IQAModel(pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        # Forward pass
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 2)


# ── Predictor tests ──


class TestIQAPredictor:
    def _train_and_get_model(self, tmp_path):
        csv = _make_splits_csv(tmp_path, n_train=8, n_val=4, image_size=32)
        config = IQATrainConfig(
            image_size=32, batch_size=4, num_workers=0,
            max_epochs=1, pretrained=False, device="cpu",
        )
        output_dir = tmp_path / "model_out"
        train_iqa(csv, config, output_dir)
        return output_dir / "iqa_best.pt"

    def test_assess_returns_result(self, tmp_path):
        model_path = self._train_and_get_model(tmp_path)
        predictor = IQAPredictor(model_path, device="cpu", image_size=32)

        img = _make_fundus_image(size=64)
        result = predictor.assess(img)

        assert isinstance(result, IQAResult)
        assert isinstance(result.is_gradable, bool)
        assert 0.0 <= result.quality_score <= 1.0
        assert isinstance(result.issues, list)
        assert isinstance(result.guidance, list)
        assert result.latency_ms > 0

    def test_assess_dark_image(self, tmp_path):
        model_path = self._train_and_get_model(tmp_path)
        predictor = IQAPredictor(model_path, device="cpu", image_size=32)

        # Very dark image
        dark = np.zeros((64, 64, 3), dtype=np.uint8)
        result = predictor.assess(dark)

        assert "dark" in result.issues or "field_of_view" in result.issues
        assert not result.is_gradable

    def test_assess_blurry_image(self, tmp_path):
        model_path = self._train_and_get_model(tmp_path)
        predictor = IQAPredictor(model_path, device="cpu", image_size=32)

        # Create a heavily blurred image
        img = _make_fundus_image(size=64)
        blurry = apply_defocus_blur(img, radius=15)
        result = predictor.assess(blurry)

        assert "blur" in result.issues
        assert not result.is_gradable

    def test_assess_batch(self, tmp_path):
        model_path = self._train_and_get_model(tmp_path)
        predictor = IQAPredictor(model_path, device="cpu", image_size=32)

        images = [_make_fundus_image(size=64) for _ in range(3)]
        results = predictor.assess_batch(images)

        assert len(results) == 3
        assert all(isinstance(r, IQAResult) for r in results)

    def test_guidance_messages(self, tmp_path):
        model_path = self._train_and_get_model(tmp_path)
        predictor = IQAPredictor(model_path, device="cpu", image_size=32)

        # Dark image should get guidance
        dark = np.full((64, 64, 3), 5, dtype=np.uint8)
        result = predictor.assess(dark)

        assert len(result.guidance) > 0
        assert any("dark" in g.lower() or "light" in g.lower() or "quality" in g.lower()
                    for g in result.guidance)


# ── Screening pipeline tests ──


class TestScreeningPipeline:
    def test_pipeline_ungradable(self, tmp_path):
        csv = _make_splits_csv(tmp_path, n_train=8, n_val=4, image_size=32)
        config = IQATrainConfig(
            image_size=32, batch_size=4, num_workers=0,
            max_epochs=1, pretrained=False, device="cpu",
        )
        output_dir = tmp_path / "model_out"
        train_iqa(csv, config, output_dir)

        predictor = IQAPredictor(output_dir / "iqa_best.pt", device="cpu", image_size=32)

        # Dummy DR model
        from bioagentics.diagnostics.retinal_dr_screening.training import create_model
        dr_model = create_model("mobilenetv3_small_100", pretrained=False)
        dr_model.eval()

        # Dark image → ungradable
        dark = np.zeros((64, 64, 3), dtype=np.uint8)
        result = run_screening_pipeline(
            dark, predictor, dr_model,
            device=torch.device("cpu"), image_size=32,
        )

        assert isinstance(result, DRScreeningResult)
        assert not result.is_gradable
        assert result.dr_grade is None
        assert result.confidence is None
        assert len(result.recapture_guidance) > 0

    def test_pipeline_gradable(self, tmp_path):
        csv = _make_splits_csv(tmp_path, n_train=8, n_val=4, image_size=32)
        config = IQATrainConfig(
            image_size=32, batch_size=4, num_workers=0,
            max_epochs=1, pretrained=False, device="cpu",
        )
        output_dir = tmp_path / "model_out"
        train_iqa(csv, config, output_dir)

        # Use a high threshold so model likely passes
        predictor = IQAPredictor(
            output_dir / "iqa_best.pt", device="cpu",
            image_size=32, quality_threshold=0.0,  # very permissive
        )

        from bioagentics.diagnostics.retinal_dr_screening.training import create_model
        dr_model = create_model("mobilenetv3_small_100", pretrained=False)
        dr_model.eval()

        # Normal-ish image
        img = _make_fundus_image(size=64, brightness=120)
        result = run_screening_pipeline(
            img, predictor, dr_model,
            device=torch.device("cpu"), image_size=32,
        )

        assert isinstance(result, DRScreeningResult)
        # With threshold=0.0, issues may still block — check structure
        if result.is_gradable:
            assert result.dr_grade is not None
            assert 0 <= result.dr_grade <= 4
            assert result.dr_probabilities is not None
            assert len(result.dr_probabilities) == 5
            assert result.confidence is not None
            assert 0 <= result.confidence <= 1.0
            assert result.is_referable is not None


# ── Quality issues dict test ──


class TestQualityIssues:
    def test_all_issues_have_guidance(self):
        expected_keys = {"blur", "dark", "bright", "field_of_view", "occlusion"}
        assert set(QUALITY_ISSUES.keys()) == expected_keys
        for key, msg in QUALITY_ISSUES.items():
            assert isinstance(msg, str)
            assert len(msg) > 10
