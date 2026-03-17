"""Tests for DR screening config module."""

from bioagentics.diagnostics.retinal_dr_screening.config import (
    DATA_DIR,
    DATASETS,
    DR_CLASSES,
    ECE_N_BINS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    KD_ALPHA,
    KD_TEMPERATURE,
    MAX_INFERENCE_LATENCY_MS,
    MOBILE_IMAGE_SIZE,
    MODEL_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    REFERABLE_THRESHOLD,
    TRAIN_IMAGE_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def test_dr_classes():
    assert NUM_CLASSES == 5
    assert DR_CLASSES[0] == "No DR"
    assert DR_CLASSES[4] == "Proliferative DR"


def test_referable_threshold():
    assert REFERABLE_THRESHOLD == 2


def test_dataset_registry():
    assert len(DATASETS) == 5
    assert "eyepacs" in DATASETS
    assert "aptos2019" in DATASETS
    assert "idrid" in DATASETS
    assert DATASETS["idrid"].get("has_lesion_annotations") is True


def test_split_ratios_sum_to_one():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9


def test_image_sizes():
    assert TRAIN_IMAGE_SIZE == 512
    assert MOBILE_IMAGE_SIZE == 224


def test_imagenet_normalization():
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD) == 3


def test_paths_under_repo():
    assert "smartphone-retinal-dr-screening" in str(DATA_DIR)
    assert "smartphone-retinal-dr-screening" in str(OUTPUT_DIR)
    assert "models" in str(MODEL_DIR)


def test_training_defaults():
    assert KD_TEMPERATURE > 1.0
    assert 0.0 < KD_ALPHA < 1.0
    assert ECE_N_BINS > 0
    assert MAX_INFERENCE_LATENCY_MS == 500
