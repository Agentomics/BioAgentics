"""Loader for mPower Study (Sage Bionetworks / Synapse).

Expected structure under raw_dir:
  voice/              (directory of m4a or wav audio recordings)
  demographics.csv    (subject demographics: healthCode, age, gender, diagnosis)
  voice_activity.csv  (recording metadata: recordId, healthCode, medTimepoint, etc.)

~16,000 participants, smartphone-recorded sustained vowel phonation.
Self-reported PD diagnosis and medication status.

Data access: https://www.synapse.org/ (Sage Bionetworks, syn4993293)
"""

import csv
import logging
from pathlib import Path

from bioagentics.voice_pd.data_loaders.base import BaseDatasetLoader
from bioagentics.voice_pd.preprocessing import SUPPORTED_EXTENSIONS

log = logging.getLogger(__name__)


class MPowerLoader(BaseDatasetLoader):
    """Load mPower smartphone PD voice recordings."""

    name = "mpower"

    def load_manifest(self) -> list[dict]:
        """Build manifest from audio files and metadata CSVs."""
        # Load demographics
        demographics = self._load_demographics()
        # Load voice activity metadata
        voice_meta = self._load_voice_metadata()

        # Find audio files
        audio_dir = self._find_audio_dir()
        if audio_dir is None:
            log.warning("mPower: no audio directory found in %s", self.raw_dir)
            return []

        audio_files = sorted(
            f for f in audio_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            log.warning("mPower: no audio files found in %s", audio_dir)
            return []

        manifest: list[dict] = []
        for audio_file in audio_files:
            record_id = audio_file.stem
            meta = voice_meta.get(record_id, {})
            health_code = meta.get("healthCode", "")
            demo = demographics.get(health_code, {})

            # Map professional-diagnosis to pd_status
            diagnosis = demo.get("professional-diagnosis", "").lower()
            if "parkinson" in diagnosis:
                pd_status = "pd"
            elif health_code and not diagnosis:
                pd_status = "unknown"
            else:
                pd_status = "healthy"

            manifest.append({
                "recording_id": record_id,
                "audio_path": str(audio_file),
                "dataset": self.name,
                "pd_status": pd_status,
                "subject_id": health_code or record_id,
                "age": demo.get("age"),
                "sex": demo.get("gender"),
                "med_timepoint": meta.get("medTimepoint", ""),
            })

        log.info("mPower: found %d recordings", len(manifest))
        return manifest

    def _find_audio_dir(self) -> Path | None:
        """Locate the audio directory."""
        for name in ("voice", "audio", "recordings"):
            d = self.raw_dir / name
            if d.is_dir():
                return d
        # Fall back to raw_dir itself if it contains audio
        audio_files = [
            f for f in self.raw_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return self.raw_dir if audio_files else None

    def _load_demographics(self) -> dict[str, dict]:
        """Load demographics CSV keyed by healthCode."""
        info: dict[str, dict] = {}
        for fname in ("demographics.csv", "demographics.tsv"):
            fpath = self.raw_dir / fname
            if fpath.exists():
                with open(fpath, newline="") as f:
                    for row in csv.DictReader(f):
                        hc = row.get("healthCode", "")
                        if hc:
                            info[hc] = row
                log.info("mPower: loaded %d demographic records", len(info))
                break
        return info

    def _load_voice_metadata(self) -> dict[str, dict]:
        """Load voice activity metadata keyed by recordId."""
        meta: dict[str, dict] = {}
        for fname in ("voice_activity.csv", "voice_activity.tsv"):
            fpath = self.raw_dir / fname
            if fpath.exists():
                with open(fpath, newline="") as f:
                    for row in csv.DictReader(f):
                        rid = row.get("recordId", "")
                        if rid:
                            meta[rid] = row
                log.info("mPower: loaded %d voice activity records", len(meta))
                break
        return meta
