"""Download HMBA-BG (Allen Brain Cell Atlas) human subset.

Downloads cell metadata, reference taxonomy, and marker gene tables
from the Allen Institute's ABC Atlas for the Human-Mammalian Brain
Atlas - Basal Ganglia (HMBA-BG) dataset.

Skips macaque/marmoset data. Skips full expression matrices (50GB+).
Uses abc_atlas_access API to stream from AWS S3 (no login required).

Saves to: data/tourettes/cstc-circuit-expression-atlas/hmba_bg_reference/
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/tourettes/cstc-circuit-expression-atlas/hmba_bg_reference")

TAXONOMY_DIR = "HMBA-BG-taxonomy-CCN20250428"
MULTIOME_DIR = "HMBA-10xMultiome-BG"

# Taxonomy files to download (cell type hierarchy, annotations, embeddings)
TAXONOMY_FILES = [
    "cluster",
    "cluster_annotation_term",
    "cluster_annotation_term_set",
    "cluster_to_cluster_annotation_membership",
    "cluster_annotation_to_abbreviation_map",
    "abbreviation_term",
    "cell_to_cluster_membership",
    "cell_2d_embedding_coordinates",
]

# Multiome metadata files to download (human-relevant only)
MULTIOME_FILES = [
    "cell_metadata",
    "donor",
    "library",
    "human_gene",
    "value_sets",
]


def get_cache() -> AbcProjectCache:
    """Create an AbcProjectCache with our data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return AbcProjectCache.from_s3_cache(DATA_DIR)


def download_taxonomy(abc_cache: AbcProjectCache) -> list[Path]:
    """Download taxonomy/clustering reference files."""
    paths = []
    for fname in TAXONOMY_FILES:
        logger.info("Downloading taxonomy: %s", fname)
        path = abc_cache.get_file_path(
            directory=TAXONOMY_DIR, file_name=fname
        )
        logger.info("  Saved: %s", path)
        paths.append(Path(path))
    return paths


def download_multiome_metadata(abc_cache: AbcProjectCache) -> list[Path]:
    """Download multiome metadata files (cell metadata, genes, donors)."""
    paths = []
    for fname in MULTIOME_FILES:
        logger.info("Downloading multiome metadata: %s", fname)
        path = abc_cache.get_file_path(
            directory=MULTIOME_DIR, file_name=fname
        )
        logger.info("  Saved: %s", path)
        paths.append(Path(path))
    return paths


def filter_human_cells(abc_cache: AbcProjectCache) -> Path:
    """Load cell metadata and save human-only subset as parquet.

    The full cell_metadata includes all species. We filter to human
    and save a compact parquet file for downstream use.
    """
    output = DATA_DIR / "human_cell_metadata.parquet"
    if output.exists():
        logger.info("Human cell metadata already exists: %s", output)
        return output

    logger.info("Loading cell metadata (chunked read for memory safety)...")
    cell_meta_path = abc_cache.get_file_path(
        directory=MULTIOME_DIR, file_name="cell_metadata"
    )

    # Load donor table to get human donor labels
    donor_path = abc_cache.get_file_path(
        directory=MULTIOME_DIR, file_name="donor"
    )
    donors = pd.read_csv(donor_path)
    human_donors = set(
        donors.loc[donors["species_genus"] == "Human", "donor_label"]
    )
    logger.info("Human donors: %s", human_donors)

    # Read in chunks, filter to human donors to stay under RAM limits
    chunks = []
    for chunk in pd.read_csv(cell_meta_path, chunksize=100_000):
        human_chunk = chunk[chunk["donor_label"].isin(human_donors)]
        if len(human_chunk) > 0:
            chunks.append(human_chunk)
        logger.info("  Processed chunk: %d human / %d total cells",
                     len(human_chunk), len(chunk))

    human_df = pd.concat(chunks, ignore_index=True)
    logger.info("Total human cells: %d", len(human_df))

    human_df.to_parquet(output, index=False)
    logger.info("Saved human-only metadata: %s (%.1fMB)",
                output, output.stat().st_size / (1024 * 1024))
    return output


def download_all() -> dict[str, list[Path]]:
    """Download all HMBA-BG reference files."""
    abc_cache = get_cache()

    results = {
        "taxonomy": download_taxonomy(abc_cache),
        "multiome_metadata": download_multiome_metadata(abc_cache),
    }

    human_meta = filter_human_cells(abc_cache)
    results["human_cell_metadata"] = [human_meta]

    return results


def verify_downloads() -> bool:
    """Verify key files exist."""
    abc_cache = get_cache()
    ok = True
    for d, files in [(TAXONOMY_DIR, TAXONOMY_FILES), (MULTIOME_DIR, MULTIOME_FILES)]:
        for fname in files:
            try:
                path = abc_cache.get_file_path(directory=d, file_name=fname)
                if Path(path).exists():
                    size_mb = Path(path).stat().st_size / (1024 * 1024)
                    logger.info("OK: %s/%s (%.1fMB)", d, fname, size_mb)
                else:
                    logger.error("Missing: %s/%s", d, fname)
                    ok = False
            except Exception as e:
                logger.error("Error checking %s/%s: %s", d, fname, e)
                ok = False
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_all()
    if verify_downloads():
        logger.info("All HMBA-BG reference files downloaded successfully.")
    else:
        logger.error("Some files are missing!")
        raise SystemExit(1)
