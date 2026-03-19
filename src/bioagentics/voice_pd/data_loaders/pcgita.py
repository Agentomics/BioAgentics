"""Loader for PC-GITA dataset (Colombian Spanish).

Expected structure under raw_dir:
  audio/          (WAV files organized by task/subject)
  metadata.csv    (subject ID, diagnosis, age, sex, UPDRS if available)

100 subjects (50 PD, 50 control), multiple speech tasks in Spanish.
Critical for cross-language evaluation of voice biomarkers.

Reference: Orozco-Arroyave et al. (2014).
"""

import csv
import logging

from bioagentics.voice_pd.data_loaders.base import BaseDatasetLoader
from bioagentics.voice_pd.preprocessing import SUPPORTED_EXTENSIONS

log = logging.getLogger(__name__)


class PCGITALoader(BaseDatasetLoader):
    """Load PC-GITA Colombian Spanish PD voice dataset."""

    name = "pcgita"

    def load_manifest(self) -> list[dict]:
        """Build manifest from audio files and metadata."""
        subject_info = self._load_metadata()

        # Find audio files
        audio_dir = self.raw_dir / "audio"
        scan_dir = audio_dir if audio_dir.is_dir() else self.raw_dir

        audio_files = sorted(
            f for f in scan_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            log.warning("PC-GITA: no audio files found in %s", scan_dir)
            return []

        manifest: list[dict] = []
        for audio_file in audio_files:
            recording_id = audio_file.stem
            # PC-GITA naming conventions vary; try to extract subject ID
            subject_id = self._extract_subject_id(recording_id)
            info = subject_info.get(subject_id, {})

            manifest.append({
                "recording_id": recording_id,
                "audio_path": str(audio_file),
                "dataset": self.name,
                "pd_status": info.get("pd_status", "unknown"),
                "subject_id": subject_id,
                "language": "es",
                "age": info.get("age"),
                "sex": info.get("sex"),
            })

        log.info("PC-GITA: found %d recordings", len(manifest))
        return manifest

    def _extract_subject_id(self, recording_id: str) -> str:
        """Extract subject ID from filename. Handles common naming patterns."""
        # Common: AVPEPUDEA0001_vowel_a_1 -> AVPEPUDEA0001
        # Or: PD01_vowel -> PD01, HC01_vowel -> HC01
        parts = recording_id.split("_")
        return parts[0] if parts else recording_id

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
                    # Normalize diagnosis
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
            log.info("PC-GITA: loaded metadata for %d subjects", len(info))
            break
        return info
