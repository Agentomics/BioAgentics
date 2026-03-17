"""Load HMP2/IBDMDB longitudinal multi-omics data for CD flare prediction.

Loads clinical metadata, HBI scores, metagenomics (MetaPhlAn), pathway
abundances (HUMAnN), metabolomics, serology, and host transcriptomics from
the data directory.  Returns standardized pandas DataFrames with consistent
subject/timepoint indexing.

Expected data layout::

    data/crohns/cd-flare-longitudinal-prediction/
    ├── hmp2_metadata.csv          # subject_id, diagnosis, visit_num, date, ...
    ├── hbi_scores.csv             # subject_id, visit_num, date, hbi_score
    ├── metaphlan_species.csv      # rows=samples, cols=species (+ subject_id, visit_num)
    ├── humann_pathways.csv        # rows=samples, cols=pathways (+ subject_id, visit_num)
    ├── metabolomics.csv           # rows=samples, cols=metabolites (+ subject_id, visit_num)
    ├── serology.csv               # subject_id, visit_num, ASCA_IgA, ASCA_IgG, anti_CBir1, anti_OmpC
    └── transcriptomics.csv        # rows=samples, cols=genes (+ subject_id, visit_num)

Usage::

    from bioagentics.crohns.flare_prediction.data_loader import HMP2DataLoader

    loader = HMP2DataLoader()
    meta = loader.load_metadata()
    hbi  = loader.load_hbi()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "crohns" / "cd-flare-longitudinal-prediction"

# Standard index columns shared across all omic tables
_IDX_COLS = ["subject_id", "visit_num"]


class HMP2DataLoader:
    """Load and align HMP2/IBDMDB multi-omic data tables."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_csv(self, name: str) -> pd.DataFrame:
        path = self.data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Expected data file not found: {path}")
        df = pd.read_csv(path)
        logger.info("Loaded %s: %d rows × %d cols", name, *df.shape)
        return df

    @staticmethod
    def _set_index(df: pd.DataFrame) -> pd.DataFrame:
        """Set (subject_id, visit_num) as the index when columns are present."""
        present = [c for c in _IDX_COLS if c in df.columns]
        if present:
            df = df.set_index(present)
        return df

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_metadata(self) -> pd.DataFrame:
        """Load clinical metadata (subject_id, diagnosis, visit dates, etc.)."""
        df = self._read_csv("hmp2_metadata.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return self._set_index(df)

    def load_hbi(self) -> pd.DataFrame:
        """Load Harvey-Bradshaw Index scores time-series.

        Returns a DataFrame indexed by (subject_id, visit_num) with at least
        a ``hbi_score`` column and optionally ``date``.
        """
        df = self._read_csv("hbi_scores.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "hbi_score" not in df.columns:
            raise ValueError("hbi_scores.csv must contain an 'hbi_score' column")
        return self._set_index(df)

    def load_species(self) -> pd.DataFrame:
        """Load MetaPhlAn species-level abundance table.

        Rows = samples (subject_id × visit_num), columns = species relative
        abundances.
        """
        df = self._read_csv("metaphlan_species.csv")
        return self._set_index(df)

    def load_pathways(self) -> pd.DataFrame:
        """Load HUMAnN pathway abundance table."""
        df = self._read_csv("humann_pathways.csv")
        return self._set_index(df)

    def load_metabolomics(self) -> pd.DataFrame:
        """Load untargeted metabolomics feature table (~8 000 features)."""
        df = self._read_csv("metabolomics.csv")
        return self._set_index(df)

    def load_serology(self) -> pd.DataFrame | None:
        """Load serologic markers (ASCA, anti-CBir1, anti-OmpC).

        Returns ``None`` if the file does not exist (serology may not be
        available at every timepoint).
        """
        try:
            df = self._read_csv("serology.csv")
        except FileNotFoundError:
            logger.info("Serology data not available — skipping")
            return None
        return self._set_index(df)

    def load_transcriptomics(self) -> pd.DataFrame | None:
        """Load host biopsy transcriptomics (available for a subset of samples).

        Returns ``None`` if the file does not exist.
        """
        try:
            df = self._read_csv("transcriptomics.csv")
        except FileNotFoundError:
            logger.info("Transcriptomics data not available — skipping")
            return None
        return self._set_index(df)

    # ------------------------------------------------------------------
    # Convenience: load everything at once
    # ------------------------------------------------------------------

    def load_all(
        self, cd_only: bool = True
    ) -> dict[str, pd.DataFrame | None]:
        """Load all available omic layers.

        Parameters
        ----------
        cd_only:
            If True (default), filter to Crohn's disease patients only.
            Non-IBD controls are dropped from all tables.

        Returns a dict keyed by layer name.
        """
        meta = self.load_metadata()
        hbi = self.load_hbi()
        species = self.load_species()
        pathways = self.load_pathways()
        metabolomics = self.load_metabolomics()
        serology = self.load_serology()
        transcriptomics = self.load_transcriptomics()

        result: dict[str, pd.DataFrame | None] = {
            "metadata": meta,
            "hbi": hbi,
            "species": species,
            "pathways": pathways,
            "metabolomics": metabolomics,
            "serology": serology,
            "transcriptomics": transcriptomics,
        }

        if cd_only:
            result = _filter_cd(result)

        return result


def _filter_cd(
    data: dict[str, pd.DataFrame | None],
) -> dict[str, pd.DataFrame | None]:
    """Keep only Crohn's disease subjects across all tables.

    Uses the ``diagnosis`` column in the metadata table to identify CD
    subjects, then filters every table to those subject IDs.
    """
    meta = data["metadata"]
    if meta is None:
        return data

    # diagnosis lives either in the index or as a column
    if "diagnosis" in meta.columns:
        diag = meta["diagnosis"]
    elif meta.index.name == "subject_id" or (
        isinstance(meta.index, pd.MultiIndex)
        and "subject_id" in meta.index.names
    ):
        diag = meta.reset_index()["diagnosis"]
    else:
        logger.warning("No 'diagnosis' column found — cannot filter to CD only")
        return data

    cd_ids = set(
        meta.reset_index()
        .loc[diag.values == "CD", "subject_id"]
        .unique()
    )
    logger.info("Filtering to %d CD subjects", len(cd_ids))

    filtered: dict[str, pd.DataFrame | None] = {}
    for key, df in data.items():
        if df is None:
            filtered[key] = None
            continue
        df_reset = df.reset_index()
        if "subject_id" in df_reset.columns:
            mask = df_reset["subject_id"].isin(list(cd_ids))
            df_filtered = df_reset.loc[mask]
            # Restore original index structure
            idx_cols = [c for c in _IDX_COLS if c in df_filtered.columns]
            if idx_cols:
                df_filtered = df_filtered.set_index(idx_cols)
            filtered[key] = df_filtered
        else:
            filtered[key] = df
    return filtered


def summarize_data(data: dict[str, pd.DataFrame | None]) -> dict[str, dict]:
    """Return summary statistics for each loaded omic layer."""
    summary: dict[str, dict] = {}
    for key, df in data.items():
        if df is None:
            summary[key] = {"available": False}
            continue
        df_reset = df.reset_index()
        n_subjects = df_reset["subject_id"].nunique() if "subject_id" in df_reset.columns else None
        n_samples = len(df)
        feature_cols = [
            c for c in df.columns
            if c not in {"subject_id", "visit_num", "date", "diagnosis"}
        ]
        summary[key] = {
            "available": True,
            "n_subjects": n_subjects,
            "n_samples": n_samples,
            "n_features": len(feature_cols),
            "missing_frac": float(df[feature_cols].isna().mean().mean()) if feature_cols else 0.0,
        }
    return summary
