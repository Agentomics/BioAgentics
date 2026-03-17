"""Download and preprocess bulk RNA-seq count data from GEO into AnnData.

Downloads RNA-seq count matrices from GEO (via GEOparse), loads into AnnData,
and applies DESeq2-style median-of-ratios normalization using pydeseq2.

Designed for the transcriptomic-biomarker-panel project — must handle bulk
RNA-seq PBMC datasets. Tested with GSE293230 (PANS bulk RNA-seq, n=158).

Usage:
    uv run python -m bioagentics.data.geo_rnaseq_loader GSE293230 [--dest DIR] [--force]
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "transcriptomic-biomarker-panel"


# ---------------------------------------------------------------------------
# GEO download helpers
# ---------------------------------------------------------------------------


def _download_gse(accession: str, dest_dir: Path) -> "GEOparse.GEOTypes.GSE":  # noqa: F821
    """Download a GEO Series and return the GEOparse GSE object."""
    import GEOparse

    logger.info("Downloading %s from GEO...", accession)
    gse = GEOparse.get_GEO(geo=accession, destdir=str(dest_dir), silent=True)
    return gse


def _extract_sample_metadata(gse) -> pd.DataFrame:
    """Extract sample metadata from a GEOparse GSE object.

    Parses title, source, organism, and all characteristics_ch1 fields
    into a flat DataFrame indexed by GSM accession.
    """
    records: list[dict] = []
    for gsm_name, gsm in gse.gsms.items():
        meta = gsm.metadata
        rec: dict = {
            "gsm": gsm_name,
            "title": meta.get("title", [""])[0],
            "source": meta.get("source_name_ch1", [""])[0],
            "organism": meta.get("organism_ch1", [""])[0],
            "platform": meta.get("platform_id", [""])[0],
        }
        # Flatten characteristics into key-value pairs
        for ch in meta.get("characteristics_ch1", []):
            if ":" in ch:
                key, _, val = ch.partition(":")
                key = key.strip().lower().replace(" ", "_")
                rec[key] = val.strip()
            else:
                rec.setdefault("characteristics", [])
                if isinstance(rec["characteristics"], list):
                    rec["characteristics"].append(ch.strip())
        if isinstance(rec.get("characteristics"), list):
            rec["characteristics"] = "; ".join(rec["characteristics"])
        records.append(rec)
    return pd.DataFrame(records).set_index("gsm")


def _try_load_supplementary_counts(
    gse, dest_dir: Path
) -> pd.DataFrame | None:
    """Try to find and load a count matrix from GSE supplementary files.

    Many RNA-seq datasets provide raw counts as a supplementary CSV/TSV/txt.gz.
    Returns a DataFrame with genes as rows and samples as columns, or None.
    """
    import requests
    from tqdm import tqdm

    supp_urls: list[str] = []
    for key in ("supplementary_file", "supplementary_file_1"):
        urls = gse.metadata.get(key, [])
        if isinstance(urls, str):
            urls = [urls]
        supp_urls.extend(urls)

    count_patterns = re.compile(
        r"(count|raw|expression|fpkm|tpm|matrix).*\.(csv|tsv|txt|tab)(\.gz)?$",
        re.IGNORECASE,
    )

    for url in supp_urls:
        url = url.strip()
        if not url or not count_patterns.search(url):
            continue

        fname = url.rsplit("/", 1)[-1]
        local_path = dest_dir / fname

        if not local_path.exists():
            logger.info("Downloading supplementary file: %s", fname)
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(local_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=fname) as pbar:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info("Reading supplementary count file: %s", local_path.name)
        try:
            df = _read_table_file(local_path)
            if df is not None and df.shape[1] >= 3:
                return df
        except Exception as exc:
            logger.warning("Could not parse %s: %s", fname, exc)

    return None


def _read_table_file(path: Path) -> pd.DataFrame | None:
    """Read a CSV/TSV/txt file (possibly gzipped) into a DataFrame.

    Assumes first column is gene identifiers.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode) as fh:
        first_line = fh.readline()
        sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(path, sep=sep, index_col=0, compression="infer")
    # Drop any non-numeric columns (gene description, etc.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < df.shape[1]:
        logger.info(
            "Dropping %d non-numeric columns", df.shape[1] - len(numeric_cols)
        )
        df = df[numeric_cols]
    return df


def _extract_counts_from_gsm_tables(gse) -> pd.DataFrame | None:
    """Extract expression/count values directly from GSM tables.

    Falls back to per-sample tables when no supplementary count matrix exists.
    """
    samples: dict[str, pd.Series] = {}
    for gsm_name, gsm in gse.gsms.items():
        table = gsm.table
        if table.empty:
            continue

        # Find the gene ID column
        for id_col_candidate in ("GENE_SYMBOL", "Gene Symbol", "gene_symbol", "ID_REF"):
            if id_col_candidate in table.columns:
                id_col = id_col_candidate
                break
        else:
            id_col = table.columns[0]

        val_col = "VALUE" if "VALUE" in table.columns else table.columns[1]
        series = table.set_index(id_col)[val_col]
        series = pd.to_numeric(series, errors="coerce")
        samples[gsm_name] = series

    if not samples:
        return None

    df = pd.DataFrame(samples)
    df.index.name = "gene_symbol"
    return df.dropna(how="all")


# ---------------------------------------------------------------------------
# Gene symbol standardization
# ---------------------------------------------------------------------------


def _standardize_gene_index(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize gene symbols: uppercase, deduplicate by max mean expression."""
    df = df.copy()
    df.index = df.index.astype(str).str.strip().str.upper()
    # Remove empty / NaN gene names
    df = df[df.index != ""]
    df = df[df.index != "NAN"]
    df = df[~df.index.isna()]

    if df.index.duplicated().any():
        n_dups = df.index.duplicated().sum()
        logger.info("Deduplicating %d gene symbol collisions (keeping max mean)", n_dups)
        df["_mean"] = df.mean(axis=1)
        df = df.sort_values("_mean", ascending=False)
        df = df[~df.index.duplicated(keep="first")]
        df = df.drop(columns=["_mean"])

    return df


# ---------------------------------------------------------------------------
# AnnData construction & normalization
# ---------------------------------------------------------------------------


def _build_anndata(
    counts: pd.DataFrame, metadata: pd.DataFrame
) -> ad.AnnData:
    """Build an AnnData from a counts DataFrame and sample metadata.

    counts: genes (rows) x samples (columns) — will be transposed for AnnData
    metadata: samples (rows) x metadata columns
    """
    # Ensure sample IDs align
    common = counts.columns.intersection(metadata.index)
    if len(common) == 0:
        # Try matching by stripping whitespace
        counts.columns = counts.columns.str.strip()
        metadata.index = metadata.index.str.strip()
        common = counts.columns.intersection(metadata.index)
    if len(common) == 0:
        logger.warning(
            "No overlap between count columns and metadata index. "
            "Using counts columns as-is."
        )
        common = counts.columns
        metadata = pd.DataFrame(index=common)

    counts = counts[common]
    metadata = metadata.reindex(common)

    # AnnData: observations=samples, variables=genes
    adata = ad.AnnData(
        X=counts.T.values.astype(np.float32),
        obs=metadata,
        var=pd.DataFrame(index=counts.index),
    )
    adata.var_names_make_unique()
    adata.obs_names = [str(n) for n in common]
    return adata


def normalize_deseq2(adata: ad.AnnData) -> ad.AnnData:
    """Apply DESeq2 median-of-ratios normalization via pydeseq2.

    Stores raw counts in adata.layers["counts"] and replaces adata.X
    with size-factor-normalized values. Size factors are stored in
    adata.obs["size_factors"].

    Requires at least a 'condition' column in adata.obs for pydeseq2.
    If missing, creates a dummy column.
    """
    from pydeseq2.dds import DeseqDataSet

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()

    # pydeseq2 requires integer-like counts — round if needed
    counts_df = pd.DataFrame(
        adata.X, index=adata.obs_names, columns=adata.var_names
    )
    # Remove genes with zero total counts
    gene_sums = counts_df.sum(axis=0)
    nonzero_genes = gene_sums[gene_sums > 0].index
    if len(nonzero_genes) < counts_df.shape[1]:
        n_dropped = counts_df.shape[1] - len(nonzero_genes)
        logger.info("Dropping %d genes with zero total counts", n_dropped)
        counts_df = counts_df[nonzero_genes]

    # Ensure a condition column exists for DeseqDataSet
    meta = adata.obs.copy()
    if "condition" not in meta.columns:
        meta["condition"] = "unknown"

    # Round to integers for DESeq2
    counts_int = counts_df.round().astype(int)

    dds = DeseqDataSet(
        counts=counts_int,
        metadata=meta,
        design="~condition",
        quiet=True,
    )
    dds.fit_size_factors()

    size_factors = dds.obs["size_factors"].values
    adata.obs["size_factors"] = size_factors

    # Normalized counts = raw / size_factor
    normalized = counts_df.values / size_factors[:, np.newaxis]

    # Update adata — subset to nonzero genes
    adata = adata[:, nonzero_genes].copy()
    adata.X = normalized.astype(np.float32)
    adata.layers["counts"] = counts_int.values.astype(np.float32)

    logger.info(
        "DESeq2 normalization complete: %d samples, %d genes. "
        "Size factors range: [%.2f, %.2f]",
        adata.n_obs,
        adata.n_vars,
        size_factors.min(),
        size_factors.max(),
    )
    return adata


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_geo_rnaseq(
    accession: str,
    dest_dir: Path | None = None,
    force: bool = False,
    normalize: bool = True,
) -> ad.AnnData:
    """Download and preprocess a GEO RNA-seq dataset into AnnData.

    Parameters
    ----------
    accession : str
        GEO Series accession (e.g., "GSE293230").
    dest_dir : Path, optional
        Output directory. Defaults to output/pandas_pans/transcriptomic-biomarker-panel/.
    force : bool
        Re-download and reprocess even if cached h5ad exists.
    normalize : bool
        Apply DESeq2 median-of-ratios normalization (default True).

    Returns
    -------
    AnnData with samples as obs, genes as var. If normalized,
    adata.layers["counts"] has raw counts and adata.X has normalized values.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = dest_dir / f"{accession.lower()}_processed.h5ad"
    if not force and h5ad_path.exists():
        logger.info("Loading cached AnnData from %s", h5ad_path)
        return ad.read_h5ad(h5ad_path)

    geo_cache = dest_dir / "geo_cache"
    geo_cache.mkdir(parents=True, exist_ok=True)

    gse = _download_gse(accession, geo_cache)

    # Extract metadata
    metadata = _extract_sample_metadata(gse)
    metadata["dataset"] = accession
    logger.info("Extracted metadata for %d samples", len(metadata))

    # Try supplementary count matrix first, then fall back to GSM tables
    counts = _try_load_supplementary_counts(gse, geo_cache)
    if counts is not None:
        logger.info(
            "Loaded supplementary count matrix: %d genes x %d samples",
            counts.shape[0],
            counts.shape[1],
        )
    else:
        logger.info("No supplementary count matrix found, extracting from GSM tables")
        counts = _extract_counts_from_gsm_tables(gse)
        if counts is None:
            raise ValueError(
                f"Could not extract expression data from {accession}. "
                "No supplementary count files or GSM tables with expression values."
            )
        logger.info(
            "Extracted from GSM tables: %d genes x %d samples",
            counts.shape[0],
            counts.shape[1],
        )

    # Standardize gene symbols
    counts = _standardize_gene_index(counts)

    # Build AnnData
    adata = _build_anndata(counts, metadata)
    logger.info("AnnData: %d samples x %d genes", adata.n_obs, adata.n_vars)

    # Normalize
    if normalize:
        adata = normalize_deseq2(adata)

    # Save
    adata.write_h5ad(h5ad_path)
    logger.info("Saved to %s", h5ad_path)

    # Also save metadata CSV for easy inspection
    adata.obs.to_csv(dest_dir / f"{accession.lower()}_metadata.csv")

    return adata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess GEO RNA-seq data into AnnData"
    )
    parser.add_argument("accession", help="GEO Series accession (e.g., GSE293230)")
    parser.add_argument(
        "--dest", type=Path, default=OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if cached"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip DESeq2 normalization"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    adata = load_geo_rnaseq(
        args.accession,
        dest_dir=args.dest,
        force=args.force,
        normalize=not args.no_normalize,
    )
    print(f"\nResult: {adata.n_obs} samples x {adata.n_vars} genes")
    print(f"Obs columns: {list(adata.obs.columns)}")
    if "counts" in adata.layers:
        print("Layers: counts (raw), X (normalized)")
    print(f"Saved to: {args.dest}")


if __name__ == "__main__":
    main()
