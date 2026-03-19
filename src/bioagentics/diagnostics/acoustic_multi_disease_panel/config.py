"""Configuration for acoustic multi-disease screening panel."""

from pathlib import Path

# ── Repository paths ──
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "acoustic-multi-disease-panel"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "acoustic-multi-disease-panel"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
EVAL_DIR = OUTPUT_DIR / "evaluation"

# ── Audio parameters (shared with voice_pd) ──
SAMPLE_RATE = 16_000
N_CHANNELS = 1
AUDIO_FORMAT = "wav"

# ── Spectrogram parameters ──
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# ── MFCC parameters ──
N_MFCC = 13

# ── Recording types in the protocol ──
RECORDING_TYPES = ("sustained_vowel", "cough", "counting", "reading_passage")

# ── Target conditions ──
CONDITIONS = ("parkinsons", "respiratory", "mci")

# ── Datasets ──
DATASETS = {
    # Parkinson's
    "mpower": {
        "name": "mPower Study",
        "condition": "parkinsons",
        "recording_types": ["sustained_vowel"],
        "raw_dir": RAW_DIR / "mpower",
    },
    "uci_telemonitoring": {
        "name": "UCI Parkinson's Telemonitoring",
        "condition": "parkinsons",
        "recording_types": ["sustained_vowel"],
        "raw_dir": RAW_DIR / "uci_telemonitoring",
    },
    "pcgita": {
        "name": "PC-GITA",
        "condition": "parkinsons",
        "recording_types": ["sustained_vowel", "reading_passage"],
        "raw_dir": RAW_DIR / "pcgita",
    },
    # Respiratory
    "coughvid": {
        "name": "COUGHVID",
        "condition": "respiratory",
        "recording_types": ["cough"],
        "raw_dir": RAW_DIR / "coughvid",
    },
    "zambia_tb": {
        "name": "Zambia TB Cough Study",
        "condition": "respiratory",
        "recording_types": ["cough"],
        "raw_dir": RAW_DIR / "zambia_tb",
    },
    # MCI / Dementia
    "adress2020": {
        "name": "ADReSS Challenge 2020",
        "condition": "mci",
        "recording_types": ["reading_passage"],
        "raw_dir": RAW_DIR / "adress2020",
    },
    "adresso2021": {
        "name": "ADReSSo 2021",
        "condition": "mci",
        "recording_types": ["reading_passage"],
        "raw_dir": RAW_DIR / "adresso2021",
    },
}

# ── Success thresholds ──
TARGET_AUROC = 0.80
MAX_DEGRADATION = 0.03  # max AUROC drop vs single-disease baseline
