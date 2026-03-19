"""Loader for UCI Parkinson's Speech Dataset.

Expected structure under raw_dir:
  train_data.txt / test_data.txt   (pre-extracted features, 26 features per row)
  -- OR --
  audio/          (directory of WAV files, if available)
  metadata.csv    (optional, subject-level metadata)

The UCI Speech dataset contains 1,040 recordings from 40 subjects (20 PD, 20 healthy).
Multiple phonation types: sustained vowels, words, short sentences.

Reference: Sakar et al. (2013), UCI ML Repository ID 301.
"""

import csv
import logging
from pathlib import Path

from bioagentics.voice_pd.data_loaders.base import BaseDatasetLoader
from bioagentics.voice_pd.preprocessing import SUPPORTED_EXTENSIONS

log = logging.getLogger(__name__)


class UCISpeechLoader(BaseDatasetLoader):
    """Load UCI Parkinson's Speech Dataset."""

    name = "uci_speech"

    def load_manifest(self) -> list[dict]:
        """Build manifest from audio files or pre-extracted feature files."""
        manifest: list[dict] = []

        # Try audio directory first
        audio_dir = self.raw_dir / "audio"
        if audio_dir.is_dir():
            manifest = self._load_from_audio(audio_dir)
        else:
            # Fall back to scanning raw_dir for audio files
            manifest = self._load_from_audio(self.raw_dir)

        # If feature text files exist, load pre-extracted features
        if not manifest:
            manifest = self._load_from_feature_files()

        return manifest

    def _load_from_audio(self, audio_dir: Path) -> list[dict]:
        """Scan for audio files and infer metadata from filenames."""
        manifest: list[dict] = []
        audio_files = sorted(
            f for f in audio_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            return manifest

        # Try to load subject metadata if available
        subject_info = self._load_subject_metadata()

        for audio_file in audio_files:
            recording_id = audio_file.stem
            # UCI naming: subjectID_taskType_repetition (common convention)
            parts = recording_id.split("_")
            subject_id = parts[0] if parts else recording_id

            pd_status = "unknown"
            if subject_id in subject_info:
                pd_status = subject_info[subject_id].get("pd_status", "unknown")

            manifest.append({
                "recording_id": recording_id,
                "audio_path": str(audio_file),
                "dataset": self.name,
                "pd_status": pd_status,
                "subject_id": subject_id,
            })

        log.info("UCI Speech: found %d audio files", len(manifest))
        return manifest

    def _load_from_feature_files(self) -> list[dict]:
        """Load from pre-extracted feature text files (no audio)."""
        manifest: list[dict] = []

        for fname in ("train_data.txt", "test_data.txt"):
            fpath = self.raw_dir / fname
            if not fpath.exists():
                continue

            split = "train" if "train" in fname else "test"
            with open(fpath) as f:
                for i, line in enumerate(f):
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue
                    # Last column is typically the label (1=PD, 0=healthy)
                    label = parts[-1].strip()
                    pd_status = "pd" if label == "1" else "healthy"

                    manifest.append({
                        "recording_id": f"{split}_{i}",
                        "audio_path": "",  # no audio file, features only
                        "dataset": self.name,
                        "pd_status": pd_status,
                        "subject_id": f"subj_{i // 26}",  # 26 recordings per subject
                        "split": split,
                        "preextracted": True,
                    })

        if manifest:
            log.info("UCI Speech: loaded %d rows from feature files", len(manifest))
        return manifest

    def _load_subject_metadata(self) -> dict[str, dict]:
        """Load optional subject metadata CSV."""
        meta_path = self.raw_dir / "metadata.csv"
        if not meta_path.exists():
            return {}

        info: dict[str, dict] = {}
        with open(meta_path, newline="") as f:
            for row in csv.DictReader(f):
                sid = row.get("subject_id", "")
                if sid:
                    info[sid] = {
                        "pd_status": row.get("pd_status", "unknown"),
                        "age": row.get("age"),
                        "sex": row.get("sex"),
                    }
        return info
