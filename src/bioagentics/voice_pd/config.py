"""Configuration constants for voice-biomarkers-parkinsons project."""

from pathlib import Path

# ── Repository paths ──
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "voice-biomarkers-parkinsons"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "voice-biomarkers-parkinsons"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
EVAL_DIR = OUTPUT_DIR / "evaluation"

# ── Audio parameters ──
SAMPLE_RATE = 16_000  # 16 kHz
N_CHANNELS = 1  # mono
AUDIO_FORMAT = "wav"

# ── Spectrogram parameters ──
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
FIXED_DURATION_SEC = 5.0  # sustained vowel clip length

# ── MFCC parameters ──
N_MFCC = 13  # 13 coefficients + delta + delta-delta = 78 features

# ── Dataset names ──
DATASETS = {
    "mpower": {
        "name": "mPower Study",
        "raw_dir": RAW_DIR / "mpower",
        "description": "Sage Bionetworks smartphone PD recordings (~16K participants)",
    },
    "uci_speech": {
        "name": "UCI Parkinson's Speech Dataset",
        "raw_dir": RAW_DIR / "uci_speech",
        "description": "1,040 recordings from 40 subjects (20 PD, 20 healthy)",
    },
    "uci_telemonitoring": {
        "name": "UCI Parkinson's Telemonitoring Dataset",
        "raw_dir": RAW_DIR / "uci_telemonitoring",
        "description": "5,875 recordings from 42 early-stage PD patients with UPDRS scores",
    },
    "pcgita": {
        "name": "PC-GITA (Colombian Spanish)",
        "raw_dir": RAW_DIR / "pcgita",
        "description": "100 subjects (50 PD, 50 control), Spanish speech tasks",
    },
    "italian": {
        "name": "Italian Parkinson's Voice and Speech",
        "raw_dir": RAW_DIR / "italian",
        "description": "65 subjects, multiple speech tasks, professionally recorded",
    },
}

# ── Quality tiers ──
QUALITY_TIERS = ("high", "medium", "low")

# ── Model targets ──
TARGET_AUC = 0.85  # held-out mPower test set
CROSS_DATASET_AUC = 0.80  # train mPower, test UCI/PC-GITA
ENSEMBLE_IMPROVEMENT = 0.03  # ensemble must beat single-modality by >=3% AUC
MOBILE_INFERENCE_SEC = 2.0  # max inference time on mobile
