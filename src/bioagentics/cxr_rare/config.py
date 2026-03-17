"""Configuration for CXR long-tail rare disease detection.

Paths, CXR-LT label taxonomy, head/body/tail class boundaries,
dataset metadata, and train/val/test split conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

from bioagentics.config import REPO_ROOT

# --- Paths ---
PROJECT = "cxr-rare-disease-detection"
DATA_DIR = REPO_ROOT / "data" / PROJECT
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / PROJECT

# Per-dataset data directories
MIMIC_CXR_DIR = DATA_DIR / "mimic-cxr"
MIDRC_DIR = DATA_DIR / "midrc"
MULTICARE_DIR = DATA_DIR / "multicare"
NIH_CHESTXRAY14_DIR = DATA_DIR / "nih-chestxray14"
CHEXPERT_DIR = DATA_DIR / "chexpert"

# Output subdirectories
BASELINES_DIR = OUTPUT_DIR / "baselines"
LONGTAIL_DIR = OUTPUT_DIR / "longtail"
CROSS_INST_DIR = OUTPUT_DIR / "cross_institutional"
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_dirs() -> None:
    """Create all project directories."""
    for d in [
        DATA_DIR,
        MIMIC_CXR_DIR,
        MIDRC_DIR,
        MULTICARE_DIR,
        NIH_CHESTXRAY14_DIR,
        CHEXPERT_DIR,
        OUTPUT_DIR,
        BASELINES_DIR,
        LONGTAIL_DIR,
        CROSS_INST_DIR,
        FIGURES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# --- CXR-LT Label Taxonomy ---
# 26 thoracic findings from CXR-LT 2026 (MIMIC-CXR + MIDRC label space).
# Ordered roughly by expected prevalence (descending).
LABEL_NAMES: list[str] = [
    # Head classes (high prevalence, >5% of dataset)
    "No Finding",
    "Support Devices",
    "Atelectasis",
    "Pleural Effusion",
    "Cardiomegaly",
    "Lung Opacity",
    "Edema",
    # Body classes (moderate prevalence, 1-5%)
    "Consolidation",
    "Pneumonia",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Fracture",
    "Pleural Other",
    "Calcification of the Aorta",
    # Tail classes (rare, <1%)
    "Pneumothorax",
    "Subcutaneous Emphysema",
    "Tortuous Aorta",
    "Pneumomediastinum",
    "Pneumoperitoneum",
    "Hernia",
    "Mass",
    "Nodule",
    "Fibrosis",
    "Emphysema",
    "Foreign Body",
    "Interstitial Lung Disease",
]

NUM_CLASSES = len(LABEL_NAMES)
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABEL_NAMES)}
INDEX_TO_LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}

# --- Head / Body / Tail Boundaries ---
# Thresholds are sample counts; classes with fewer samples than the boundary
# are assigned to that bin. These are configurable defaults based on
# typical CXR-LT distributions.
HEAD_THRESHOLD = 10_000  # >= this count → head
TAIL_THRESHOLD = 1_000   # < this count → tail; between tail and head → body

# Default bin assignments (expected, refined after label profiling)
HEAD_CLASSES: list[str] = LABEL_NAMES[:7]
BODY_CLASSES: list[str] = LABEL_NAMES[7:14]
TAIL_CLASSES: list[str] = LABEL_NAMES[14:]


def classify_by_frequency(
    class_counts: dict[str, int],
    head_thresh: int = HEAD_THRESHOLD,
    tail_thresh: int = TAIL_THRESHOLD,
) -> dict[str, str]:
    """Assign each class to head/body/tail based on sample count."""
    bins: dict[str, str] = {}
    for name, count in class_counts.items():
        if count >= head_thresh:
            bins[name] = "head"
        elif count < tail_thresh:
            bins[name] = "tail"
        else:
            bins[name] = "body"
    return bins


# --- Dataset Metadata ---
@dataclass(frozen=True)
class DatasetInfo:
    name: str
    description: str
    size: str
    access: str
    label_count: int
    data_dir_name: str

DATASETS: dict[str, DatasetInfo] = {
    "mimic-cxr": DatasetInfo(
        name="CXR-LT 2026 (MIMIC-CXR)",
        description="Multi-label, 26+ findings, long-tail distribution",
        size="~230,000 CXRs",
        access="PhysioNet (credentialed)",
        label_count=NUM_CLASSES,
        data_dir_name="mimic-cxr",
    ),
    "midrc": DatasetInfo(
        name="CXR-LT 2026 (MIDRC)",
        description="Multi-institutional, distribution shift source",
        size="~70,000 CXRs",
        access="MIDRC portal",
        label_count=NUM_CLASSES,
        data_dir_name="midrc",
    ),
    "multicare": DatasetInfo(
        name="MultiCaRe",
        description="140+ category taxonomy from PubMed case reports",
        size="130,791 images, 93,816 cases",
        access="Open access",
        label_count=140,
        data_dir_name="multicare",
    ),
    "nih-chestxray14": DatasetInfo(
        name="NIH ChestX-ray14",
        description="14 findings, NLP-extracted labels",
        size="112,120 CXRs",
        access="Open access",
        label_count=14,
        data_dir_name="nih-chestxray14",
    ),
    "chexpert": DatasetInfo(
        name="CheXpert",
        description="14 findings with uncertainty labels",
        size="224,316 CXRs",
        access="Stanford (application)",
        label_count=14,
        data_dir_name="chexpert",
    ),
}

# --- NIH ChestX-ray14 Label Mapping ---
# Maps NIH 14 labels to the CXR-LT label space.
NIH_TO_CXRLT: dict[str, str | None] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Pleural Effusion",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Hernia": "Hernia",
    "Infiltration": "Lung Opacity",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pleural_Thickening": "Pleural Other",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}

# --- Train / Val / Test Split Conventions ---
@dataclass
class SplitConfig:
    """Train/val/test split configuration."""
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    # For MIMIC-CXR: use official patient-level splits
    use_official_split: bool = True
    split_column: str = "split"
    train_label: str = "train"
    val_label: str = "validate"
    test_label: str = "test"


DEFAULT_SPLIT = SplitConfig()

# --- Image Preprocessing ---
@dataclass
class ImageConfig:
    """Default image preprocessing parameters."""
    image_size: int = 224
    # ImageNet normalization (standard for pretrained models)
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    # CXR-specific: single-channel images replicated to 3 channels
    input_channels: int = 3


DEFAULT_IMAGE_CONFIG = ImageConfig()

# --- Training Defaults ---
@dataclass
class TrainConfig:
    """Default training hyperparameters."""
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    early_stopping_patience: int = 5
    mixed_precision: bool = True
    checkpoint_metric: str = "macro_auroc"


DEFAULT_TRAIN_CONFIG = TrainConfig()
