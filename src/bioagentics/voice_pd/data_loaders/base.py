"""Base interface for PD voice dataset loaders.

Each loader discovers audio files and metadata in its dataset directory,
producing a standardized manifest (list of dicts) compatible with the
preprocessing and feature extraction pipelines.
"""

import abc
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class BaseDatasetLoader(abc.ABC):
    """Abstract base class for PD voice dataset loaders."""

    name: str = ""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    def is_available(self) -> bool:
        """Check if dataset directory exists and has expected files."""
        return self.raw_dir.is_dir() and any(self.raw_dir.iterdir())

    @abc.abstractmethod
    def load_manifest(self) -> list[dict]:
        """Discover audio files and metadata, returning a manifest.

        Each row should contain at minimum:
            - recording_id: unique identifier
            - audio_path: absolute path to audio file
            - dataset: dataset name
            - pd_status: 'pd', 'healthy', or 'unknown'
            - subject_id: participant identifier

        Additional columns vary by dataset (e.g., updrs_total, language).
        """
        ...

    def validate(self) -> dict:
        """Return a summary of dataset availability and contents."""
        if not self.is_available():
            return {
                "name": self.name,
                "available": False,
                "n_files": 0,
                "notes": f"Directory not found: {self.raw_dir}",
            }
        manifest = self.load_manifest()
        pd_count = sum(1 for r in manifest if r.get("pd_status") == "pd")
        healthy_count = sum(1 for r in manifest if r.get("pd_status") == "healthy")
        return {
            "name": self.name,
            "available": True,
            "n_files": len(manifest),
            "n_pd": pd_count,
            "n_healthy": healthy_count,
        }
