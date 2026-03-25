"""Download and load HMP2/IBDMDB multi-omics data for CD subtyping.

Downloads metagenomic species abundances, pathway abundances, metabolomic
profiles, and clinical metadata from the IBDMDB portal (ibdmdb.org).
Provides loader functions returning aligned pandas DataFrames with
consistent sample IDs across omic layers.

Expected data layout after download::

    data/hmp2/
    ├── hmp2_metadata.csv
    ├── taxonomic_profiles.tsv.gz
    ├── pathabundances_3.tsv.gz
    └── HMP2_metabolomics.csv.gz

Usage::

    # Download all data files
    uv run python -m bioagentics.data.hmp2_download

    # Load for analysis
    from bioagentics.data.hmp2_download import HMP2Loader
    loader = HMP2Loader()
    data = loader.load_all()
"""

from __future__ import annotations

import gzip
import logging
import shutil
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "hmp2"

TIMEOUT = 120
CHUNK_SIZE = 1024 * 64

# IBDMDB public data product URLs (Globus endpoint, updated 2026-03).
# These are the processed data products from Lloyd-Price et al., Nature 2019.
_GLOBUS_BASE = "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb"

DATA_FILES: dict[str, dict[str, str]] = {
    "metadata": {
        "filename": "hmp2_metadata.csv",
        "url": f"{_GLOBUS_BASE}/metadata/hmp2_metadata_2018-08-20.csv",
        "description": "Clinical metadata (diagnosis, Montreal, CRP, calprotectin, treatment)",
    },
    "taxonomic": {
        "filename": "taxonomic_profiles.tsv.gz",
        "url": f"{_GLOBUS_BASE}/products/HMP2/MGX/2018-05-04/taxonomic_profiles.tsv.gz",
        "description": "MetaPhlAn species-level taxonomic profiles",
    },
    "pathways": {
        "filename": "pathabundances_3.tsv.gz",
        "url": f"{_GLOBUS_BASE}/products/HMP2/MGX/2018-05-04/pathabundances_3.tsv.gz",
        "description": "HUMAnN3 MetaCyc pathway abundances",
    },
    "metabolomics": {
        "filename": "HMP2_metabolomics.biom",
        "url": f"{_GLOBUS_BASE}/products/HMP2/MBX/HMP2_metabolomics.biom",
        "description": "LC-MS untargeted metabolomics (BIOM format)",
    },
    "transcriptomics": {
        "filename": "host_tx_counts.tsv.gz",
        "url": f"{_GLOBUS_BASE}/products/HMP2/HTX/host_tx_counts.tsv.gz",
        "description": "Host biopsy transcriptomics (gene counts)",
    },
    "serology": {
        "filename": "hmp2_serology_Compiled_ELISA_Data.tsv",
        "url": f"{_GLOBUS_BASE}/products/HMP2/Serology/2017-10-05/hmp2_serology_Compiled_ELISA_Data.tsv",
        "description": "Compiled ELISA serology data (ASCA, anti-CBir1, anti-OmpC)",
    },
    "dysbiosis": {
        "filename": "dysbiosis_scores.tsv",
        "url": "https://forum.biobakery.org/uploads/short-url/umwfR0kDJ6s5RXHwtIMgLaOEoOI.tsv",
        "description": "Dysbiosis scores from Lloyd-Price et al. 2019",
    },
}


# ── Download Functions ──


def _download_file(url: str, dest: Path) -> None:
    """Download a file with progress bar, skip if already exists."""
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return

    logger.info("Downloading %s ...", dest.name)
    resp = requests.get(url, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(CHUNK_SIZE):
            f.write(chunk)
            pbar.update(len(chunk))

    shutil.move(str(tmp), str(dest))
    logger.info("Saved: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def download_all(dest_dir: Path | None = None) -> dict[str, Path]:
    """Download all HMP2 data files to dest_dir.

    Returns a dict mapping data type to local file path.
    """
    dest_dir = dest_dir or DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for key, info in DATA_FILES.items():
        dest = dest_dir / info["filename"]
        _download_file(info["url"], dest)
        paths[key] = dest

    return paths


# ── Parsing Functions ──


def _read_tsv_gz(path: Path) -> pd.DataFrame:
    """Read a gzipped TSV file (MetaPhlAn/HUMAnN format).

    These files have features as rows and samples as columns.
    The first column is the feature ID/name.
    """
    with gzip.open(path, "rt") as f:
        # Keep #SampleID header but skip other comment lines
        lines = []
        for line in f:
            if line.startswith("#SampleID") or line.startswith("#clade_name"):
                # MetaPhlAn/HUMAnN header row — strip leading '#'
                lines.append(line.lstrip("#"))
            elif not line.startswith("#"):
                lines.append(line)

    df = pd.read_csv(StringIO("".join(lines)), sep="\t", index_col=0)
    # Transpose: samples as rows, features as columns
    return df.T


def _read_metabolomics(path: Path) -> pd.DataFrame:
    """Read HMP2 metabolomics CSV (gzipped or plain).

    HMP2 metabolomics: samples as rows, metabolites as columns.
    First column is sample ID.
    """
    if path.suffix == ".gz":
        df = pd.read_csv(path, compression="gzip", index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)
    return df


def _extract_sample_id(raw_id: str) -> str | None:
    """Extract a normalized sample identifier from raw HMP2 sample IDs.

    HMP2 uses various ID formats across omic layers. This extracts the
    External ID portion that can be matched across modalities.
    """
    # Common HMP2 format: CSM5FZ4P_P, HSM6XRS9_TR, etc.
    # Strip trailing _P, _TR, _M suffixes (platform identifiers)
    s = str(raw_id).strip()
    for suffix in ("_P", "_TR", "_M", "_MBX"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


# ── Loader Class ──


class HMP2Loader:
    """Load and align HMP2/IBDMDB multi-omic data for CD subtyping."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def _path(self, key: str) -> Path:
        return self.data_dir / DATA_FILES[key]["filename"]

    def load_metadata(self) -> pd.DataFrame:
        """Load clinical metadata.

        Returns DataFrame indexed by External ID with columns:
        Participant ID, diagnosis, site_name, Consent, data_type, etc.
        """
        path = self._path("metadata")
        if not path.exists():
            raise FileNotFoundError(f"Metadata not found: {path}. Run download_all() first.")

        df = pd.read_csv(path)
        logger.info("Loaded metadata: %d rows × %d cols", *df.shape)

        # Normalize column names for common fields
        col_map = {}
        for col in df.columns:
            lower = col.lower().replace(" ", "_")
            if lower in ("external_id", "externalid"):
                col_map[col] = "External ID"
            elif lower in ("participant_id", "participantid"):
                col_map[col] = "Participant ID"
        if col_map:
            df = df.rename(columns=col_map)

        return df

    def load_species(self) -> pd.DataFrame:
        """Load MetaPhlAn species-level abundances.

        Returns DataFrame: rows=samples, columns=species relative abundances.
        """
        path = self._path("taxonomic")
        if not path.exists():
            raise FileNotFoundError(f"Taxonomic profiles not found: {path}. Run download_all() first.")

        df = _read_tsv_gz(path)

        # Filter to species-level taxa only (contain 's__' but not 't__')
        species_cols = [
            c for c in df.columns if "s__" in c and "t__" not in c
        ]
        df = df[species_cols]

        # Simplify column names: extract genus_species
        rename = {}
        for col in df.columns:
            parts = col.split("|")
            species_part = [p for p in parts if p.startswith("s__")]
            if species_part:
                genus_parts = [p for p in parts if p.startswith("g__")]
                genus = genus_parts[0].replace("g__", "") if genus_parts else ""
                species = species_part[0].replace("s__", "")
                rename[col] = f"{genus}_{species}" if genus else species
        df = df.rename(columns=rename)

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        logger.info("Loaded species: %d samples × %d taxa", *df.shape)
        return df

    def load_pathways(self) -> pd.DataFrame:
        """Load HUMAnN pathway abundances.

        Returns DataFrame: rows=samples, columns=MetaCyc pathways.
        """
        path = self._path("pathways")
        if not path.exists():
            raise FileNotFoundError(f"Pathway abundances not found: {path}. Run download_all() first.")

        df = _read_tsv_gz(path)

        # Filter to community-level pathways (exclude per-species stratifications)
        community_cols = [c for c in df.columns if "|" not in c]
        df = df[community_cols]

        # Remove UNMAPPED and UNINTEGRATED
        df = df.drop(
            columns=[c for c in df.columns if c in ("UNMAPPED", "UNINTEGRATED")],
            errors="ignore",
        )

        df = df.apply(pd.to_numeric, errors="coerce")

        logger.info("Loaded pathways: %d samples × %d pathways", *df.shape)
        return df

    def load_metabolomics(self) -> pd.DataFrame:
        """Load LC-MS metabolomics profiles.

        Returns DataFrame: rows=samples, columns=metabolite features.
        """
        path = self._path("metabolomics")
        if not path.exists():
            raise FileNotFoundError(f"Metabolomics not found: {path}. Run download_all() first.")

        df = _read_metabolomics(path)
        df = df.apply(pd.to_numeric, errors="coerce")

        logger.info("Loaded metabolomics: %d samples × %d metabolites", *df.shape)
        return df

    def _build_sample_map(self, metadata: pd.DataFrame) -> dict[str, str]:
        """Build mapping from External ID to Participant ID using metadata."""
        sample_map: dict[str, str] = {}
        if "External ID" in metadata.columns and "Participant ID" in metadata.columns:
            for _, row in metadata.iterrows():
                ext_id = str(row["External ID"]).strip()
                part_id = str(row["Participant ID"]).strip()
                if ext_id and part_id and ext_id != "nan" and part_id != "nan":
                    sample_map[ext_id] = part_id
                    # Also map normalized versions
                    norm = _extract_sample_id(ext_id)
                    if norm and norm != ext_id:
                        sample_map[norm] = part_id
        return sample_map

    def _align_to_participants(
        self, df: pd.DataFrame, sample_map: dict[str, str]
    ) -> pd.DataFrame:
        """Map omic sample IDs to Participant IDs for cross-modal alignment.

        For longitudinal data, aggregates to per-participant means (taking
        the first/average timepoint for cross-sectional subtyping).
        """
        # Try direct mapping first, then normalized mapping
        mapped_ids = []
        for idx in df.index:
            sid = str(idx).strip()
            pid = sample_map.get(sid) or sample_map.get(_extract_sample_id(sid))
            mapped_ids.append(pid)

        df = df.copy()
        df["participant_id"] = mapped_ids
        df = df.dropna(subset=["participant_id"])

        # Aggregate multiple timepoints per participant (mean for subtyping)
        df = df.groupby("participant_id").mean(numeric_only=True)

        return df

    def load_all(
        self,
        diagnosis_filter: list[str] | None = None,
        align: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Load all omic layers, optionally filtered by diagnosis.

        Parameters
        ----------
        diagnosis_filter:
            List of diagnoses to include (e.g., ["CD", "nonIBD"]).
            Default: ["CD", "nonIBD"] (exclude UC for CD subtyping).
        align:
            If True, align all omic layers to shared Participant IDs and
            aggregate longitudinal data to per-participant means.

        Returns
        -------
        dict with keys: "metadata", "species", "pathways", "metabolomics"
        """
        if diagnosis_filter is None:
            diagnosis_filter = ["CD", "nonIBD"]

        metadata = self.load_metadata()
        species = self.load_species()
        pathways = self.load_pathways()
        metabolomics = self.load_metabolomics()

        # Filter metadata by diagnosis
        diag_col = None
        for col in metadata.columns:
            if col.lower() in ("diagnosis", "dx"):
                diag_col = col
                break

        if diag_col and diagnosis_filter:
            metadata = metadata[metadata[diag_col].isin(diagnosis_filter)]
            logger.info(
                "Filtered to %s: %d rows",
                diagnosis_filter,
                len(metadata),
            )

        result: dict[str, pd.DataFrame] = {"metadata": metadata}

        if align:
            sample_map = self._build_sample_map(metadata)
            if sample_map:
                result["species"] = self._align_to_participants(species, sample_map)
                result["pathways"] = self._align_to_participants(pathways, sample_map)
                result["metabolomics"] = self._align_to_participants(
                    metabolomics, sample_map
                )
                # Find participants present in all omic layers
                shared_ids = (
                    set(result["species"].index)
                    & set(result["pathways"].index)
                    & set(result["metabolomics"].index)
                )
                logger.info(
                    "Shared participants across all omics: %d", len(shared_ids)
                )
            else:
                logger.warning(
                    "Could not build sample map — returning unaligned data"
                )
                result["species"] = species
                result["pathways"] = pathways
                result["metabolomics"] = metabolomics
        else:
            result["species"] = species
            result["pathways"] = pathways
            result["metabolomics"] = metabolomics

        return result

    def get_aligned_matrices(
        self,
        diagnosis_filter: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return aligned species, pathways, metabolomics, and metadata matrices.

        All omic DataFrames share the same index (Participant IDs).
        Only participants with data in ALL layers are included.

        Returns
        -------
        species, pathways, metabolomics, metadata
        """
        data = self.load_all(diagnosis_filter=diagnosis_filter, align=True)

        species = data["species"]
        pathways = data["pathways"]
        metabolomics = data["metabolomics"]
        metadata = data["metadata"]

        # Intersect to shared participants
        shared = (
            set(species.index) & set(pathways.index) & set(metabolomics.index)
        )
        if not shared:
            raise ValueError("No shared participants found across all omic layers")

        shared_sorted = sorted(shared)
        species = species.loc[shared_sorted]
        pathways = pathways.loc[shared_sorted]
        metabolomics = metabolomics.loc[shared_sorted]

        # Filter metadata to shared participants
        if "Participant ID" in metadata.columns:
            metadata = metadata[metadata["Participant ID"].isin(shared)]
            # Deduplicate to one row per participant (first visit)
            metadata = metadata.drop_duplicates(subset=["Participant ID"])
            metadata = metadata.set_index("Participant ID")
            metadata = metadata.loc[metadata.index.isin(shared_sorted)]

        logger.info(
            "Aligned matrices: %d participants, %d species, %d pathways, %d metabolites",
            len(shared_sorted),
            species.shape[1],
            pathways.shape[1],
            metabolomics.shape[1],
        )

        return species, pathways, metabolomics, metadata


def summarize(data: dict[str, pd.DataFrame]) -> None:
    """Print summary statistics for loaded HMP2 data."""
    for key, df in data.items():
        print(f"\n{key}:")
        print(f"  Shape: {df.shape}")
        if key == "metadata":
            for col in df.columns:
                if col.lower() in ("diagnosis", "dx"):
                    print(f"  Diagnosis counts:\n{df[col].value_counts().to_string()}")
        else:
            missing = df.isna().mean().mean()
            print(f"  Missing: {missing:.1%}")
            print(f"  Index type: {type(df.index).__name__}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Download HMP2/IBDMDB data")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DATA_DIR,
        help=f"Destination directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, just load and summarize existing data",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.skip_download:
        print("Downloading HMP2/IBDMDB data...")
        paths = download_all(args.dest)
        print(f"\nDownloaded {len(paths)} files to {args.dest}")

    print("\nLoading and summarizing data...")
    try:
        loader = HMP2Loader(args.dest)
        data = loader.load_all()
        summarize(data)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run without --skip-download to fetch data first.", file=sys.stderr)
        sys.exit(1)
