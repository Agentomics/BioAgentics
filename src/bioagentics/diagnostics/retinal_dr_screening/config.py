"""Shared configuration for the DR screening pipeline."""

from __future__ import annotations

from bioagentics.config import REPO_ROOT

# ── Paths ──
DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "smartphone-retinal-dr-screening"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "smartphone-retinal-dr-screening"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ── DR severity labels (International Clinical DR Severity Scale) ──
DR_CLASSES: dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}
NUM_CLASSES = len(DR_CLASSES)

# Referable DR = grade >= 2 (moderate NPDR or worse)
REFERABLE_THRESHOLD = 2

# ── Image settings ──
TRAIN_IMAGE_SIZE = 512  # pixels (square crop for training)
MOBILE_IMAGE_SIZE = 224  # pixels (square crop for mobile inference)

# ImageNet normalization (used with pretrained backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ── Dataset registry ──
DATASETS: dict[str, dict] = {
    "eyepacs": {
        "name": "EyePACS / Kaggle DR Detection",
        "source": "kaggle:diabetic-retinopathy-detection",
        "expected_images": 88_702,
        "population": "US multi-ethnic",
    },
    "aptos2019": {
        "name": "APTOS 2019 Blindness Detection",
        "source": "kaggle:aptos2019-blindness-detection",
        "expected_images": 5_590,
        "population": "South Asian (India)",
    },
    "idrid": {
        "name": "Indian Diabetic Retinopathy Image Dataset",
        "source": "ieee:idrid",
        "expected_images": 516,
        "population": "South Asian (India)",
        "has_lesion_annotations": True,
    },
    "messidor2": {
        "name": "Messidor-2",
        "source": "adcis:messidor-2",
        "expected_images": 1_748,
        "population": "European (France)",
    },
    "odir5k": {
        "name": "ODIR-5K",
        "source": "odir2019",
        "expected_images": 5_000,
        "population": "East Asian (China)",
    },
}

# ── Splits ──
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ── Training defaults ──
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

# ── Knowledge distillation ──
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.7  # weight for distillation loss vs. hard-label loss

# ── Calibration ──
ECE_N_BINS = 15

# ── Mobile deployment targets ──
MAX_INFERENCE_LATENCY_MS = 500
MAX_MODEL_SIZE_MB = 20
