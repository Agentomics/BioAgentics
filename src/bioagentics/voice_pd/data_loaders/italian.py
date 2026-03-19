"""Loader for Italian Parkinson's Voice and Speech dataset.

Expected structure under raw_dir:
  audio/          (WAV files, professionally recorded)
  metadata.csv    (subject info: ID, diagnosis, age, sex)

65 subjects, multiple speech tasks, professionally recorded.
"""

import csv
import logging

from bioagentics.voice_pd.data_loaders.base import BaseDatasetLoader
from bioagentics.voice_pd.preprocessing import SUPPORTED_EXTENSIONS

log = logging.getLogger(__name__)


class ItalianLoader(BaseDatasetLoader):
    """Load Italian Parkinson's Voice and Speech dataset."""

    name = "italian"

    def load_manifest(self) -> list[dict]:
        """Build manifest from audio files and metadata."""
        subject_info = self._load_metadata()

        audio_dir = self.raw_dir / "audio"
        scan_dir = audio_dir if audio_dir.is_dir() else self.raw_dir

        audio_files = sorted(
            f for f in scan_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            log.warning("Italian: no audio files found in %s", scan_dir)
            return []

        manifest: list[dict] = []
        for audio_file in audio_files:
            recording_id = audio_file.stem
            parts = recording_id.split("_")
            subject_id = parts[0] if parts else recording_id
            info = subject_info.get(subject_id, {})

            manifest.append({
                "recording_id": recording_id,
                "audio_path": str(audio_file),
                "dataset": self.name,
                "pd_status": info.get("pd_status", "unknown"),
                "subject_id": subject_id,
                "language": "it",
                "age": info.get("age"),
                "sex": info.get("sex"),
            })

        log.info("Italian: found %d recordings", len(manifest))
        return manifest

    def _load_metadata(self) -> dict[str, dict]:
        """Load subject metadata CSV."""
        info: dict[str, dict] = {}
        for fname in ("metadata.csv", "subjects.csv", "participants.csv"):
            fpath = self.raw_dir / fname
            if not fpath.exists():
                continue
            with open(fpath, newline="") as f:
                for row in csv.DictReader(f):
                    sid = row.get("subject_id", row.get("id", ""))
                    if not sid:
                        continue
                    diag = row.get("diagnosis", row.get("group", "")).lower()
                    if "pd" in diag or "parkinson" in diag:
                        pd_status = "pd"
                    elif "control" in diag or "hc" in diag or "healthy" in diag:
                        pd_status = "healthy"
                    else:
                        pd_status = "unknown"
                    info[sid] = {
                        "pd_status": pd_status,
                        "age": row.get("age"),
                        "sex": row.get("sex", row.get("gender")),
                    }
            log.info("Italian: loaded metadata for %d subjects", len(info))
            break
        return info
