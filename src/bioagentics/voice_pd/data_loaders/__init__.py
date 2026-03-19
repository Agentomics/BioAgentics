"""Dataset loaders for PD voice datasets.

Provides per-dataset loaders and a convenience function to load all available
datasets into a unified manifest.
"""

import logging

from bioagentics.voice_pd.config import DATASETS
from bioagentics.voice_pd.data_loaders.base import BaseDatasetLoader
from bioagentics.voice_pd.data_loaders.italian import ItalianLoader
from bioagentics.voice_pd.data_loaders.mpower import MPowerLoader
from bioagentics.voice_pd.data_loaders.pcgita import PCGITALoader
from bioagentics.voice_pd.data_loaders.uci_speech import UCISpeechLoader
from bioagentics.voice_pd.data_loaders.uci_telemonitoring import UCITelemonitoringLoader

log = logging.getLogger(__name__)

__all__ = [
    "BaseDatasetLoader",
    "MPowerLoader",
    "UCISpeechLoader",
    "UCITelemonitoringLoader",
    "PCGITALoader",
    "ItalianLoader",
    "get_loader",
    "load_all_available",
]

LOADER_MAP: dict[str, type[BaseDatasetLoader]] = {
    "mpower": MPowerLoader,
    "uci_speech": UCISpeechLoader,
    "uci_telemonitoring": UCITelemonitoringLoader,
    "pcgita": PCGITALoader,
    "italian": ItalianLoader,
}


def get_loader(dataset_name: str) -> BaseDatasetLoader:
    """Get a loader instance for a dataset, using the configured raw_dir."""
    if dataset_name not in LOADER_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}. Known: {list(LOADER_MAP)}")
    raw_dir = DATASETS[dataset_name]["raw_dir"]
    return LOADER_MAP[dataset_name](raw_dir)


def load_all_available() -> list[dict]:
    """Load manifests from all datasets that have data present.

    Returns a combined manifest list.
    """
    combined: list[dict] = []
    for name in LOADER_MAP:
        loader = get_loader(name)
        if loader.is_available():
            rows = loader.load_manifest()
            combined.extend(rows)
            log.info("Loaded %d rows from %s", len(rows), name)
        else:
            log.info("Dataset %s not available (no data at %s)", name, loader.raw_dir)
    log.info("Total rows across all available datasets: %d", len(combined))
    return combined
