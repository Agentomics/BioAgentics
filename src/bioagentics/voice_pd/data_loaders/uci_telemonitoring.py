"""Loader for UCI Parkinson's Telemonitoring Dataset.

Expected structure under raw_dir:
  parkinsons_updrs.data   (CSV with 16 voice measures + UPDRS scores)

5,875 recordings from 42 early-stage PD patients. All subjects have PD.
Contains motor and total UPDRS scores for regression targets.

Reference: Tsanas et al. (2010), UCI ML Repository ID 189.
"""

import csv
import logging
from pathlib import Path

from bioagentics.voice_pd.data_loaders.base import BaseDatasetLoader

log = logging.getLogger(__name__)


class UCITelemonitoringLoader(BaseDatasetLoader):
    """Load UCI Parkinson's Telemonitoring Dataset (pre-extracted features)."""

    name = "uci_telemonitoring"

    # Expected column names in the data file
    EXPECTED_COLS = [
        "subject#", "age", "sex", "test_time",
        "motor_UPDRS", "total_UPDRS",
        "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
        "Shimmer:DDA",
        "NHR", "HNR",
        "RPDE", "DFA", "PPE",
    ]

    def load_manifest(self) -> list[dict]:
        """Load the telemonitoring CSV as manifest rows.

        Each row represents one recording with pre-extracted voice features
        and UPDRS scores. No audio files — features are already extracted.
        """
        data_path = self._find_data_file()
        if data_path is None:
            log.warning("UCI Telemonitoring data file not found in %s", self.raw_dir)
            return []

        manifest: list[dict] = []
        with open(data_path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                subject_id = row.get("subject#", f"subj_{i}")
                motor_updrs = row.get("motor_UPDRS", "")
                total_updrs = row.get("total_UPDRS", "")

                manifest.append({
                    "recording_id": f"tele_{subject_id}_{i}",
                    "audio_path": "",  # no audio, pre-extracted features only
                    "dataset": self.name,
                    "pd_status": "pd",  # all subjects in this dataset have PD
                    "subject_id": str(subject_id),
                    "motor_updrs": _safe_float(motor_updrs),
                    "total_updrs": _safe_float(total_updrs),
                    "age": _safe_float(row.get("age", "")),
                    "sex": row.get("sex", ""),
                    "test_time": _safe_float(row.get("test_time", "")),
                    "preextracted": True,
                    # Voice features
                    "jitter_pct": _safe_float(row.get("Jitter(%)", "")),
                    "jitter_abs": _safe_float(row.get("Jitter(Abs)", "")),
                    "shimmer": _safe_float(row.get("Shimmer", "")),
                    "shimmer_db": _safe_float(row.get("Shimmer(dB)", "")),
                    "nhr": _safe_float(row.get("NHR", "")),
                    "hnr": _safe_float(row.get("HNR", "")),
                    "rpde": _safe_float(row.get("RPDE", "")),
                    "dfa": _safe_float(row.get("DFA", "")),
                    "ppe": _safe_float(row.get("PPE", "")),
                })

        log.info("UCI Telemonitoring: loaded %d recordings", len(manifest))
        return manifest

    def _find_data_file(self) -> Path | None:
        """Find the data file (may have various names)."""
        candidates = [
            "parkinsons_updrs.data",
            "parkinsons_updrs.csv",
            "parkinsons_telemonitoring.csv",
        ]
        for name in candidates:
            path = self.raw_dir / name
            if path.exists():
                return path
        # Try any CSV in the directory
        csvs = list(self.raw_dir.glob("*.csv")) + list(self.raw_dir.glob("*.data"))
        return csvs[0] if csvs else None


def _safe_float(val: str) -> float | None:
    """Convert string to float, returning None on failure."""
    try:
        return float(val) if val else None
    except (ValueError, TypeError):
        return None
