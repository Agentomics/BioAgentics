"""Unified dataset loader for multi-disease acoustic screening.

Normalizes mPower, UCI PD Telemonitoring, COUGHVID, and ADReSS datasets
into a common schema with participant-aware train/val/test splits to
prevent data leakage.

Common record schema:
    {audio_path, participant_id, condition, recording_type, dataset, metadata}

Registry pattern: add new datasets by subclassing DatasetLoader and
registering via @register_loader.
"""

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from bioagentics.diagnostics.acoustic_multi_disease_panel.config import (
    DATA_DIR,
    DATASETS,
)

log = logging.getLogger(__name__)

# Common record type
Record = dict[str, str]

# Registry of dataset loaders
_LOADER_REGISTRY: dict[str, type["DatasetLoader"]] = {}


def register_loader(name: str):
    """Decorator to register a dataset loader class."""

    def decorator(cls):
        _LOADER_REGISTRY[name] = cls
        return cls

    return decorator


class DatasetLoader(ABC):
    """Base class for dataset-specific loaders."""

    def __init__(self, raw_dir: str | Path, condition: str, recording_types: list[str]):
        self.raw_dir = Path(raw_dir)
        self.condition = condition
        self.recording_types = recording_types

    @abstractmethod
    def load(self) -> list[Record]:
        """Load dataset and return list of normalized records.

        Each record must have keys:
            audio_path, participant_id, condition, recording_type, dataset
        """

    def is_available(self) -> bool:
        """Check if dataset directory exists."""
        return self.raw_dir.is_dir()


@register_loader("mpower")
class MPowerLoader(DatasetLoader):
    """mPower Parkinson's study (Synapse format).

    Expects: raw_dir/audio/*.wav or raw_dir/*.m4a
    Participant IDs extracted from filename pattern: {healthCode}_{recordId}.wav
    """

    def load(self) -> list[Record]:
        records: list[Record] = []
        audio_dir = self.raw_dir
        # Try subdirectory patterns
        for subdir in [audio_dir, audio_dir / "audio", audio_dir / "voice"]:
            if not subdir.is_dir():
                continue
            for f in sorted(subdir.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".wav", ".m4a", ".mp3"):
                    parts = f.stem.split("_", 1)
                    pid = parts[0] if parts else f.stem
                    records.append({
                        "audio_path": str(f),
                        "participant_id": pid,
                        "condition": self.condition,
                        "recording_type": "sustained_vowel",
                        "dataset": "mpower",
                    })
        return records


@register_loader("uci_telemonitoring")
class UCIPDLoader(DatasetLoader):
    """UCI Parkinson's Telemonitoring Dataset.

    Expects: raw_dir/ with CSV metadata and audio files.
    CSV has subject# column for participant IDs.
    """

    def load(self) -> list[Record]:
        records: list[Record] = []
        # Look for metadata CSV
        csv_files = list(self.raw_dir.glob("*.csv")) + list(self.raw_dir.glob("*.data"))
        if csv_files:
            return self._load_from_csv(csv_files[0])
        # Fallback: scan for audio files
        for f in sorted(self.raw_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".wav", ".mp3"):
                pid = f.stem.split("_")[0] if "_" in f.stem else f.stem
                records.append({
                    "audio_path": str(f),
                    "participant_id": pid,
                    "condition": self.condition,
                    "recording_type": "sustained_vowel",
                    "dataset": "uci_telemonitoring",
                })
        return records

    def _load_from_csv(self, csv_path: Path) -> list[Record]:
        records: list[Record] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("subject#", row.get("subject", "unknown"))
                # UCI dataset may not have audio files (features only)
                records.append({
                    "audio_path": str(csv_path),
                    "participant_id": str(pid),
                    "condition": self.condition,
                    "recording_type": "sustained_vowel",
                    "dataset": "uci_telemonitoring",
                })
        return records


@register_loader("coughvid")
class COUGHVIDLoader(DatasetLoader):
    """COUGHVID crowdsourced cough dataset.

    Expects: raw_dir/ with .webm or .ogg files and metadata.csv.
    """

    def load(self) -> list[Record]:
        records: list[Record] = []
        # Try metadata CSV first
        meta_path = self.raw_dir / "metadata.csv"
        if meta_path.exists():
            return self._load_from_metadata(meta_path)
        # Fallback: scan audio files
        for f in sorted(self.raw_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".webm", ".ogg", ".wav", ".mp3"):
                records.append({
                    "audio_path": str(f),
                    "participant_id": f.stem,
                    "condition": self.condition,
                    "recording_type": "cough",
                    "dataset": "coughvid",
                })
        return records

    def _load_from_metadata(self, meta_path: Path) -> list[Record]:
        records: list[Record] = []
        with open(meta_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uuid = row.get("uuid", row.get("id", ""))
                if not uuid:
                    continue
                # Find matching audio file
                for ext in (".webm", ".ogg", ".wav"):
                    audio = self.raw_dir / f"{uuid}{ext}"
                    if audio.exists():
                        records.append({
                            "audio_path": str(audio),
                            "participant_id": uuid,
                            "condition": self.condition,
                            "recording_type": "cough",
                            "dataset": "coughvid",
                        })
                        break
        return records


@register_loader("adress2020")
class ADReSSLoader(DatasetLoader):
    """ADReSS Challenge 2020 (DementiaBank).

    Expects: raw_dir/ with subdirectories for AD and control groups.
    """

    def load(self) -> list[Record]:
        records: list[Record] = []
        for subdir in [self.raw_dir, self.raw_dir / "Full_wave_enhanced_audio"]:
            if not subdir.is_dir():
                continue
            for group_dir in sorted(subdir.iterdir()):
                if not group_dir.is_dir():
                    continue
                for f in sorted(group_dir.rglob("*")):
                    if f.is_file() and f.suffix.lower() in (".wav", ".mp3"):
                        records.append({
                            "audio_path": str(f),
                            "participant_id": f.stem,
                            "condition": self.condition,
                            "recording_type": "reading_passage",
                            "dataset": "adress2020",
                        })
        # Fallback: flat directory
        if not records:
            for f in sorted(self.raw_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".wav", ".mp3"):
                    records.append({
                        "audio_path": str(f),
                        "participant_id": f.stem,
                        "condition": self.condition,
                        "recording_type": "reading_passage",
                        "dataset": "adress2020",
                    })
        return records


@register_loader("adresso2021")
class ADReSSoLoader(DatasetLoader):
    """ADReSSo 2021 — same structure as ADReSS 2020."""

    def load(self) -> list[Record]:
        # Reuse ADReSS loader pattern
        loader = ADReSSLoader(self.raw_dir, self.condition, self.recording_types)
        records = loader.load()
        for r in records:
            r["dataset"] = "adresso2021"
        return records


@register_loader("pcgita")
class PCGITALoader(DatasetLoader):
    """PC-GITA Parkinson's speech dataset."""

    def load(self) -> list[Record]:
        records: list[Record] = []
        for f in sorted(self.raw_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".wav", ".mp3"):
                pid = f.stem.split("_")[0] if "_" in f.stem else f.stem
                # Infer recording type from filename or parent dir
                fname_lower = f.stem.lower()
                rec_type = "reading_passage"
                if "vowel" in fname_lower or "aaa" in fname_lower:
                    rec_type = "sustained_vowel"
                records.append({
                    "audio_path": str(f),
                    "participant_id": pid,
                    "condition": self.condition,
                    "recording_type": rec_type,
                    "dataset": "pcgita",
                })
        return records


@register_loader("zambia_tb")
class ZambiaTBLoader(DatasetLoader):
    """Zambia TB cough study."""

    def load(self) -> list[Record]:
        records: list[Record] = []
        for f in sorted(self.raw_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".wav", ".mp3", ".ogg"):
                pid = f.stem.split("_")[0] if "_" in f.stem else f.stem
                records.append({
                    "audio_path": str(f),
                    "participant_id": pid,
                    "condition": self.condition,
                    "recording_type": "cough",
                    "dataset": "zambia_tb",
                })
        return records


# ── Unified loader ──


def load_all_datasets(
    datasets: dict | None = None,
) -> list[Record]:
    """Load all configured datasets into a unified record list.

    Args:
        datasets: Override the default DATASETS config dict.

    Returns:
        List of normalized records from all available datasets.
    """
    if datasets is None:
        datasets = DATASETS

    all_records: list[Record] = []

    for ds_key, ds_info in datasets.items():
        loader_cls = _LOADER_REGISTRY.get(ds_key)
        if loader_cls is None:
            log.warning("No loader registered for dataset: %s", ds_key)
            continue

        loader = loader_cls(
            raw_dir=ds_info["raw_dir"],
            condition=ds_info["condition"],
            recording_types=ds_info["recording_types"],
        )

        if not loader.is_available():
            log.info("Skipping %s: directory not found (%s)", ds_key, ds_info["raw_dir"])
            continue

        records = loader.load()
        all_records.extend(records)
        log.info("Loaded %d records from %s", len(records), ds_key)

    log.info("Total: %d records from %d datasets", len(all_records), len(datasets))
    return all_records


# ── Train/val/test splitting ──


def participant_stratified_split(
    records: list[Record],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, list[Record]]:
    """Split records into train/val/test by participant ID.

    Ensures no participant appears in multiple splits (prevents data leakage).
    Stratifies by condition to maintain class balance across splits.

    Args:
        records: List of record dicts with 'participant_id' and 'condition'.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to record lists.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(random_state)

    # Group participants by condition
    pid_condition: dict[str, str] = {}
    for r in records:
        pid = r["participant_id"]
        if pid not in pid_condition:
            pid_condition[pid] = r["condition"]

    condition_pids: dict[str, list[str]] = {}
    for pid, cond in pid_condition.items():
        condition_pids.setdefault(cond, []).append(pid)

    train_pids: set[str] = set()
    val_pids: set[str] = set()
    test_pids: set[str] = set()

    # Stratified split per condition
    for cond, pids in condition_pids.items():
        pids_arr = np.array(pids)
        rng.shuffle(pids_arr)
        n = len(pids_arr)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0
        # Test gets the remainder
        train_pids.update(pids_arr[:n_train])
        val_pids.update(pids_arr[n_train : n_train + n_val])
        test_pids.update(pids_arr[n_train + n_val :])

    # Assign records to splits
    splits: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
    for r in records:
        pid = r["participant_id"]
        if pid in train_pids:
            splits["train"].append(r)
        elif pid in val_pids:
            splits["val"].append(r)
        else:
            splits["test"].append(r)

    for split_name, split_records in splits.items():
        cond_counts = {}
        for r in split_records:
            cond_counts[r["condition"]] = cond_counts.get(r["condition"], 0) + 1
        log.info(
            "Split %s: %d records, conditions: %s",
            split_name, len(split_records), cond_counts,
        )

    return splits


def save_splits(
    splits: dict[str, list[Record]],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Save train/val/test splits as CSV files.

    Args:
        splits: Dict from participant_stratified_split.
        output_dir: Output directory. Defaults to DATA_DIR/splits/.

    Returns:
        Dict mapping split name to output file path.
    """
    if output_dir is None:
        output_dir = DATA_DIR / "splits"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["audio_path", "participant_id", "condition", "recording_type", "dataset"]
    paths: dict[str, Path] = {}

    for split_name, records in splits.items():
        out_path = output_dir / f"{split_name}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        paths[split_name] = out_path
        log.info("Saved %s split: %d records to %s", split_name, len(records), out_path)

    return paths


def load_split(path: str | Path) -> list[Record]:
    """Load a split CSV file."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
