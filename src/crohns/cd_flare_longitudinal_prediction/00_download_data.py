"""Download and prepare HMP2/IBDMDB multi-omics data for CD flare prediction.

Downloads raw data from the IBDMDB Globus public endpoint and transforms it
into the CSV format expected by ``data_loader.HMP2DataLoader``.

Output layout::

    data/crohns/cd-flare-longitudinal-prediction/
    ├── hmp2_metadata.csv
    ├── hbi_scores.csv
    ├── metaphlan_species.csv
    ├── humann_pathways.csv
    ├── metabolomics.csv
    ├── serology.csv
    ├── transcriptomics.csv
    └── raw/                    # original downloaded files
        ├── hmp2_metadata.csv
        ├── taxonomic_profiles.tsv.gz
        ├── pathabundances_3.tsv.gz
        ├── HMP2_metabolomics.biom
        ├── host_tx_counts.tsv.gz
        ├── hmp2_serology.tsv
        └── dysbiosis_scores.tsv

Usage::

    uv run python -m crohns.cd_flare_longitudinal_prediction.00_download_data
    uv run python -m crohns.cd_flare_longitudinal_prediction.00_download_data --skip-download
    uv run python -m crohns.cd_flare_longitudinal_prediction.00_download_data --force

IMPORTANT: 8GB RAM machine.  Files are downloaded with streaming and
processed sequentially to avoid excessive memory use.
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

OUTPUT_DIR = REPO_ROOT / "data" / "crohns" / "cd-flare-longitudinal-prediction"
RAW_DIR = OUTPUT_DIR / "raw"

TIMEOUT = 180
CHUNK_SIZE = 64 * 1024

_GLOBUS = "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb"

# All raw files to download.  Keys match the transform step names.
RAW_FILES: dict[str, dict[str, str]] = {
    "metadata": {
        "filename": "hmp2_metadata.csv",
        "url": f"{_GLOBUS}/metadata/hmp2_metadata_2018-08-20.csv",
    },
    "taxonomic": {
        "filename": "taxonomic_profiles.tsv.gz",
        "url": f"{_GLOBUS}/products/HMP2/MGX/2018-05-04/taxonomic_profiles.tsv.gz",
    },
    "pathways": {
        "filename": "pathabundances_3.tsv.gz",
        "url": f"{_GLOBUS}/products/HMP2/MGX/2018-05-04/pathabundances_3.tsv.gz",
    },
    "metabolomics": {
        "filename": "HMP2_metabolomics.biom",
        "url": f"{_GLOBUS}/products/HMP2/MBX/HMP2_metabolomics.biom",
    },
    "transcriptomics": {
        "filename": "host_tx_counts.tsv.gz",
        "url": f"{_GLOBUS}/products/HMP2/HTX/host_tx_counts.tsv.gz",
    },
    "serology": {
        "filename": "hmp2_serology.tsv",
        "url": f"{_GLOBUS}/products/HMP2/Serology/2017-10-05/"
        "hmp2_serology_Compiled_ELISA_Data.tsv",
    },
    "dysbiosis": {
        "filename": "dysbiosis_scores.tsv",
        "url": "https://forum.biobakery.org/uploads/short-url/"
        "umwfR0kDJ6s5RXHwtIMgLaOEoOI.tsv",
    },
}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, *, force: bool = False) -> Path:
    """Stream-download *url* to *dest*.  Skip if already exists (unless force)."""
    if dest.exists() and not force:
        logger.info("Already downloaded: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url.split("/")[-1], dest.name)

    resp = requests.get(url, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, disable=None
    ) as pbar:
        for chunk in resp.iter_content(CHUNK_SIZE):
            fh.write(chunk)
            pbar.update(len(chunk))

    shutil.move(str(tmp), str(dest))
    logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def download_raw(*, force: bool = False) -> dict[str, Path]:
    """Download all raw HMP2 data files to RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, info in RAW_FILES.items():
        dest = RAW_DIR / info["filename"]
        _download_file(info["url"], dest, force=force)
        paths[key] = dest
    return paths


# ---------------------------------------------------------------------------
# Sample-ID mapping (External ID -> subject_id + visit_num)
# ---------------------------------------------------------------------------


def _build_sample_map(
    meta: pd.DataFrame,
) -> dict[str, tuple[str, int]]:
    """Build {External ID: (subject_id, visit_num)} from metadata.

    The metadata CSV has one row per (sample, data_type).  Multiple rows can
    share the same External ID if the sample was profiled on several platforms.
    We keep the first occurrence.
    """
    mapping: dict[str, tuple[str, int]] = {}
    for _, row in meta.iterrows():
        ext = str(row.get("External ID", "")).strip()
        pid = str(row.get("Participant ID", "")).strip()
        vn = row.get("visit_num")
        if not ext or ext == "nan" or not pid or pid == "nan":
            continue
        try:
            vn_int = int(float(vn))
        except (TypeError, ValueError):
            continue
        if ext not in mapping:
            mapping[ext] = (pid, vn_int)
    return mapping


def _strip_sample_suffix(raw_id: str) -> str:
    """Remove platform suffixes (_P, _TR, _M, _MBX) from HMP2 sample IDs."""
    s = str(raw_id).strip()
    for sfx in ("_P", "_TR", "_M", "_MBX"):
        if s.endswith(sfx):
            return s[: -len(sfx)]
    return s


def _map_samples(
    df: pd.DataFrame,
    sample_map: dict[str, tuple[str, int]],
) -> pd.DataFrame:
    """Add subject_id / visit_num columns by mapping the DataFrame index.

    Drops rows whose index cannot be resolved to a known External ID.
    """
    sids, vnums = [], []
    for idx in df.index:
        raw = str(idx).strip()
        hit = sample_map.get(raw) or sample_map.get(_strip_sample_suffix(raw))
        if hit:
            sids.append(hit[0])
            vnums.append(hit[1])
        else:
            sids.append(None)
            vnums.append(None)

    df = df.copy()
    df.insert(0, "subject_id", sids)
    df.insert(1, "visit_num", vnums)
    df = df.dropna(subset=["subject_id"])
    df["visit_num"] = df["visit_num"].astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-layer transforms  (raw file -> project CSV)
# ---------------------------------------------------------------------------


def _transform_metadata(raw_path: Path, out_dir: Path) -> Path:
    """Produce hmp2_metadata.csv: one row per (subject_id, visit_num)."""
    meta = pd.read_csv(raw_path, low_memory=False)

    keep_cols = [
        "Participant ID",
        "External ID",
        "data_type",
        "visit_num",
        "week_num",
        "date_of_receipt",
        "diagnosis",
        "consent_age",
        "sex",
        "site_name",
        "BMI",
        "race",
        "fecalcal_ng_ml",
        "is_inflamed",
        "hbi",
        "sccai",
        "smoking status",
        "Antibiotics",
        "Immunosuppressants (e.g. oral corticosteroids)",
    ]
    present = [c for c in keep_cols if c in meta.columns]
    df = meta[present].copy()

    # Deduplicate to one row per (Participant ID, visit_num), keeping first
    df = df.drop_duplicates(subset=["Participant ID", "visit_num"], keep="first")
    df = df.rename(columns={
        "Participant ID": "subject_id",
        "date_of_receipt": "date",
    })
    df = df.sort_values(["subject_id", "visit_num"]).reset_index(drop=True)

    out = out_dir / "hmp2_metadata.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out.name, len(df))
    return out


def _transform_hbi(raw_path: Path, out_dir: Path) -> Path:
    """Extract HBI scores into hbi_scores.csv."""
    meta = pd.read_csv(raw_path, low_memory=False)

    cols = {
        "Participant ID": "subject_id",
        "visit_num": "visit_num",
        "date_of_receipt": "date",
        "hbi": "hbi_score",
    }
    present = {k: v for k, v in cols.items() if k in meta.columns}
    df = meta[list(present.keys())].rename(columns=present)

    # Keep only rows with non-null HBI
    df = df.dropna(subset=["hbi_score"])
    df = df.drop_duplicates(subset=["subject_id", "visit_num"], keep="first")
    df["hbi_score"] = pd.to_numeric(df["hbi_score"], errors="coerce")
    df = df.dropna(subset=["hbi_score"])
    df = df.sort_values(["subject_id", "visit_num"]).reset_index(drop=True)

    out = out_dir / "hbi_scores.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out.name, len(df))
    return out


def _read_tsv_gz(path: Path) -> pd.DataFrame:
    """Read gzipped MetaPhlAn/HUMAnN TSV (features-as-rows, samples-as-cols)."""
    with gzip.open(path, "rt") as fh:
        lines = []
        for line in fh:
            if line.startswith("#clade_name") or line.startswith("#SampleID"):
                lines.append(line.lstrip("#"))
            elif line.startswith("# Pathway"):
                lines.append(line.lstrip("# "))
            elif not line.startswith("#"):
                lines.append(line)
    df = pd.read_csv(StringIO("".join(lines)), sep="\t", index_col=0)
    return df.T  # samples as rows


def _transform_species(
    raw_path: Path,
    out_dir: Path,
    sample_map: dict[str, tuple[str, int]],
) -> Path:
    """Produce metaphlan_species.csv (species-level only)."""
    df = _read_tsv_gz(raw_path)

    # Keep species-level taxa (s__ present, t__ absent)
    species_cols = [c for c in df.columns if "s__" in c and "t__" not in c]
    df = df[species_cols]

    # Simplify column names to genus_species
    rename = {}
    for col in species_cols:
        parts = col.split("|")
        sp = [p for p in parts if p.startswith("s__")]
        gn = [p for p in parts if p.startswith("g__")]
        if sp:
            g = gn[0].replace("g__", "") if gn else ""
            s = sp[0].replace("s__", "")
            rename[col] = f"{g}_{s}" if g else s
    df = df.rename(columns=rename)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = _map_samples(df, sample_map)

    out = out_dir / "metaphlan_species.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows × %d species)", out.name, len(df), len(df.columns) - 2)
    return out


def _transform_pathways(
    raw_path: Path,
    out_dir: Path,
    sample_map: dict[str, tuple[str, int]],
) -> Path:
    """Produce humann_pathways.csv (community-level, no per-species strats)."""
    df = _read_tsv_gz(raw_path)

    # Community-level only (no '|' separator indicating species stratification)
    community = [c for c in df.columns if "|" not in c]
    df = df[community]
    df = df.drop(
        columns=[c for c in df.columns if c in ("UNMAPPED", "UNINTEGRATED")],
        errors="ignore",
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df = _map_samples(df, sample_map)

    out = out_dir / "humann_pathways.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows × %d pathways)", out.name, len(df), len(df.columns) - 2)
    return out


def _transform_metabolomics(
    raw_path: Path,
    out_dir: Path,
    sample_map: dict[str, tuple[str, int]],
) -> Path:
    """Convert BIOM metabolomics to metabolomics.csv."""
    try:
        from biom import load_table
    except ImportError:
        raise ImportError(
            "biom-format is required for metabolomics conversion. "
            "Install: uv add --optional research biom-format"
        )

    table = load_table(str(raw_path))
    # to_dataframe: rows=observations (metabolites), cols=samples
    df = table.to_dataframe(dense=True).T  # samples as rows
    df = df.apply(pd.to_numeric, errors="coerce")
    df = _map_samples(df, sample_map)

    out = out_dir / "metabolomics.csv"
    df.to_csv(out, index=False)
    logger.info(
        "Wrote %s (%d rows × %d metabolites)", out.name, len(df), len(df.columns) - 2
    )
    return out


def _transform_transcriptomics(
    raw_path: Path,
    out_dir: Path,
    sample_map: dict[str, tuple[str, int]],
) -> Path:
    """Produce transcriptomics.csv from host gene counts."""
    df = _read_tsv_gz(raw_path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = _map_samples(df, sample_map)

    out = out_dir / "transcriptomics.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows × %d genes)", out.name, len(df), len(df.columns) - 2)
    return out


def _transform_serology(
    raw_path: Path,
    out_dir: Path,
    sample_map: dict[str, tuple[str, int]],
) -> Path:
    """Produce serology.csv (ASCA, anti-CBir1, anti-OmpC)."""
    df = pd.read_csv(raw_path, sep="\t", index_col=0)
    # Serology file: samples as columns, analytes as rows -> transpose
    if df.shape[0] < df.shape[1]:
        df = df.T
    df = df.apply(pd.to_numeric, errors="coerce")
    df = _map_samples(df, sample_map)

    out = out_dir / "serology.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows × %d markers)", out.name, len(df), len(df.columns) - 2)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_data(*, force: bool = False, skip_download: bool = False) -> dict[str, Path]:
    """Download and transform all HMP2 data layers.

    Returns a dict mapping layer name to the output CSV path.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download raw files
    if not skip_download:
        raw_paths = download_raw(force=force)
    else:
        raw_paths = {k: RAW_DIR / v["filename"] for k, v in RAW_FILES.items()}

    # Step 2: Build sample mapping from metadata
    meta_path = raw_paths["metadata"]
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. Run without --skip-download."
        )
    meta_df = pd.read_csv(meta_path, low_memory=False)
    sample_map = _build_sample_map(meta_df)
    logger.info("Built sample map: %d External IDs", len(sample_map))
    del meta_df  # free memory before processing omics

    outputs: dict[str, Path] = {}

    # Step 3: Transform each layer sequentially (memory-safe)
    logger.info("--- Transforming metadata ---")
    outputs["metadata"] = _transform_metadata(meta_path, OUTPUT_DIR)

    logger.info("--- Extracting HBI scores ---")
    outputs["hbi"] = _transform_hbi(meta_path, OUTPUT_DIR)

    if raw_paths["taxonomic"].exists():
        logger.info("--- Transforming taxonomic profiles ---")
        outputs["species"] = _transform_species(
            raw_paths["taxonomic"], OUTPUT_DIR, sample_map
        )

    if raw_paths["pathways"].exists():
        logger.info("--- Transforming pathway abundances ---")
        outputs["pathways"] = _transform_pathways(
            raw_paths["pathways"], OUTPUT_DIR, sample_map
        )

    if raw_paths["metabolomics"].exists():
        logger.info("--- Transforming metabolomics ---")
        outputs["metabolomics"] = _transform_metabolomics(
            raw_paths["metabolomics"], OUTPUT_DIR, sample_map
        )

    if raw_paths["transcriptomics"].exists():
        logger.info("--- Transforming transcriptomics ---")
        outputs["transcriptomics"] = _transform_transcriptomics(
            raw_paths["transcriptomics"], OUTPUT_DIR, sample_map
        )

    if raw_paths["serology"].exists():
        logger.info("--- Transforming serology ---")
        outputs["serology"] = _transform_serology(
            raw_paths["serology"], OUTPUT_DIR, sample_map
        )

    logger.info("=== Done: %d output files in %s ===", len(outputs), OUTPUT_DIR)
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and prepare HMP2/IBDMDB data for CD flare prediction"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, transform existing raw files only",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    outputs = prepare_data(force=args.force, skip_download=args.skip_download)
    for name, path in outputs.items():
        sz = path.stat().st_size / 1e6
        print(f"  {name}: {path.name} ({sz:.1f} MB)")
