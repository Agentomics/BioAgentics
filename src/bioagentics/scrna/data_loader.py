"""Download and preprocess GEO scRNA-seq datasets for IL-23/Th17 atlas.

Downloads count matrices from NCBI GEO, loads into AnnData format with
metadata, standardizes gene names to HGNC symbols, and saves as h5ad.

Usage:
    uv run python -m bioagentics.scrna.data_loader
    uv run python -m bioagentics.scrna.data_loader --datasets GSE134809
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

# pandas 3+ with pyarrow backend creates ArrowStringArray which anndata cannot
# serialize to h5ad.  Disable inferred-string before importing pandas/anndata.
os.environ.setdefault("PANDAS_FUTURE_INFER_STRING", "0")

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.io as sio  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from tqdm import tqdm  # noqa: E402

from bioagentics.config import REPO_ROOT

DEFAULT_DEST = REPO_ROOT / "data" / "crohns" / "il23-atlas"
TIMEOUT = 120
CHUNK_SIZE = 1024 * 64

# GEO datasets for IL-23/Th17 single-cell atlas
DATASETS: dict[str, dict] = {
    "GSE134809": {
        "description": "CD ileal tissue scRNA-seq, inflamed and non-inflamed (Martin et al. Cell 2019, PMID 31474370)",
        "platform": "10x Genomics",
        "use": "Primary dataset — IL-23 pathway mapping, FCGR1A/IL23A co-expression",
        "format": "supplementary_10x",
    },
    "GSE282122": {
        "description": "Longitudinal scRNA-seq of anti-TNF in IBD, ~1M cells, 216 biopsies, 41 subjects (Thomas et al. Nature Immunology 2024)",
        "platform": "10x Genomics",
        "use": "Treatment response dataset — anti-TNF longitudinal single-cell atlas",
        "format": "supplementary_h5ad",
    },
}

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"


def _geo_ftp_prefix(gse: str) -> str:
    """Get the GEO FTP directory prefix for a GSE accession."""
    numeric = gse.replace("GSE", "")
    prefix = numeric[:-3] if len(numeric) > 3 else ""
    return f"GSE{prefix}nnn/{gse}"


def _download_file(url: str, dest: Path) -> None:
    """Download a file with progress bar."""
    with requests.get(url, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))


def _parse_ftp_listing(html: str) -> list[str]:
    """Extract filenames from an NCBI FTP HTML directory listing."""
    files = []
    for line in html.splitlines():
        if 'href="' in line and not line.strip().startswith(".."):
            start = line.find('href="') + 6
            end = line.find('"', start)
            if end > start:
                name = line[start:end]
                if not name.endswith("/") and name not in (".", ".."):
                    files.append(name)
    return files


def download_supplementary(gse: str, dest_dir: Path) -> list[Path]:
    """Download supplementary files from GEO."""
    prefix = _geo_ftp_prefix(gse)
    suppl_url = f"{GEO_BASE}/{prefix}/suppl/"

    print(f"  Checking supplementary files for {gse}...")
    try:
        resp = requests.get(suppl_url, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.HTTPError:
        print(f"  No supplementary directory found for {gse}")
        return []

    files = _parse_ftp_listing(resp.text)
    if not files:
        print(f"  No supplementary files found")
        return []

    print(f"  Found {len(files)} supplementary files")
    downloaded = []
    suppl_dir = dest_dir / "supplementary"
    suppl_dir.mkdir(exist_ok=True)

    for fname in files:
        dest_file = suppl_dir / fname
        if dest_file.exists():
            print(f"    {fname} already exists, skipping")
            downloaded.append(dest_file)
            continue

        file_url = f"{suppl_url}{fname}"
        print(f"    Downloading {fname}...")
        try:
            _download_file(file_url, dest_file)
            downloaded.append(dest_file)
        except requests.HTTPError as e:
            print(f"    Failed: {e}", file=sys.stderr)

    return downloaded


def load_10x_from_tar(tar_path: Path) -> ad.AnnData | None:
    """Load a 10x Genomics count matrix from a tar.gz archive.

    Expects barcodes.tsv.gz, features.tsv.gz (or genes.tsv.gz), matrix.mtx.gz
    inside the archive.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        members = {m.name.split("/")[-1]: m for m in tar.getmembers() if not m.isdir()}

        # Find the matrix, barcodes, and features/genes files
        matrix_key = next((k for k in members if "matrix" in k.lower() and k.endswith(".mtx.gz")), None)
        barcode_key = next((k for k in members if "barcode" in k.lower()), None)
        feature_key = next(
            (k for k in members if any(x in k.lower() for x in ("features", "genes"))),
            None,
        )

        if matrix_key is None or barcode_key is None or feature_key is None:
            print(f"    Missing 10x files in {tar_path.name}: found {list(members.keys())}")
            return None

        # Read matrix
        mtx_fh = tar.extractfile(members[matrix_key])
        if mtx_fh is None:
            return None
        mtx_data = gzip.decompress(mtx_fh.read())
        matrix = sp.csc_matrix(sio.mmread(io.BytesIO(mtx_data))).T  # cells x genes

        # Read barcodes
        bc_fh = tar.extractfile(members[barcode_key])
        if bc_fh is None:
            return None
        bc_data = bc_fh.read()
        if barcode_key.endswith(".gz"):
            bc_data = gzip.decompress(bc_data)
        barcodes = [line.split("\t")[0] for line in bc_data.decode().strip().split("\n")]

        # Read features/genes
        feat_fh = tar.extractfile(members[feature_key])
        if feat_fh is None:
            return None
        feat_data = feat_fh.read()
        if feature_key.endswith(".gz"):
            feat_data = gzip.decompress(feat_data)
        genes = []
        gene_ids = []
        for line in feat_data.decode().strip().split("\n"):
            parts = line.split("\t")
            gene_ids.append(parts[0])
            genes.append(parts[1] if len(parts) > 1 else parts[0])

    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame({"gene_ids": gene_ids, "gene_symbols": genes}, index=genes),
    )
    adata.var_names_make_unique()
    return adata


def load_10x_from_flat_tar(
    tar_path: Path,
    min_genes: int = 200,
) -> list[ad.AnnData]:
    """Load per-sample 10x matrices from a plain .tar with flat GSM-prefixed files.

    GEO RAW.tar archives often contain files like:
        GSM3972009_barcodes.tsv.gz
        GSM3972009_genes.tsv.gz
        GSM3972009_matrix.mtx.gz

    Groups files by GSM prefix, loads each triplet, and applies
    ``min_genes`` filtering to remove empty droplets before returning.
    """
    _10X_SUFFIXES = {"barcodes.tsv.gz", "genes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"}

    open_mode = "r:gz" if tar_path.name.endswith(".gz") else "r:"
    with tarfile.open(tar_path, open_mode) as tar:
        members = [m for m in tar.getmembers() if not m.isdir()]

        # Group by GSM prefix
        groups: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
        for m in members:
            basename = m.name.split("/")[-1]
            # Match GSM\d+_ prefix
            match = re.match(r"(GSM\d+)_(.+)", basename)
            if not match:
                continue
            gsm, suffix = match.group(1), match.group(2)
            if suffix in _10X_SUFFIXES:
                groups[gsm][suffix] = m

        adatas: list[ad.AnnData] = []
        for gsm in sorted(groups):
            g = groups[gsm]
            mtx_m = g.get("matrix.mtx.gz")
            bc_m = g.get("barcodes.tsv.gz")
            feat_m = g.get("genes.tsv.gz") or g.get("features.tsv.gz")
            if mtx_m is None or bc_m is None or feat_m is None:
                print(f"    {gsm}: incomplete triplet ({list(g.keys())}), skipping")
                continue

            # Read matrix
            fh = tar.extractfile(mtx_m)
            if fh is None:
                continue
            mtx_data = gzip.decompress(fh.read())
            matrix = sp.csc_matrix(sio.mmread(io.BytesIO(mtx_data))).T  # cells x genes

            # Read barcodes
            fh = tar.extractfile(bc_m)
            if fh is None:
                continue
            bc_data = gzip.decompress(fh.read())
            barcodes = [line.split("\t")[0] for line in bc_data.decode().strip().split("\n")]

            # Read features/genes
            fh = tar.extractfile(feat_m)
            if fh is None:
                continue
            feat_data = gzip.decompress(fh.read())
            genes: list[str] = []
            gene_ids: list[str] = []
            for line in feat_data.decode().strip().split("\n"):
                parts = line.split("\t")
                gene_ids.append(parts[0])
                genes.append(parts[1] if len(parts) > 1 else parts[0])

            adata = ad.AnnData(
                X=sp.csr_matrix(matrix),
                obs=pd.DataFrame(index=barcodes),
                var=pd.DataFrame({"gene_ids": gene_ids, "gene_symbols": genes}, index=genes),
            )
            adata.var_names_make_unique()

            # Filter empty droplets — raw 10x may have 737k barcodes
            n_genes_per_cell = np.diff(adata.X.indptr) if sp.issparse(adata.X) else (adata.X > 0).sum(axis=1)
            keep = np.asarray(n_genes_per_cell >= min_genes).flatten()
            before = adata.n_obs
            adata = adata[keep].copy()
            print(f"    {gsm}: {before} -> {adata.n_obs} cells (min_genes={min_genes})")

            adata.obs["sample"] = gsm
            adatas.append(adata)

    return adatas


def load_10x_from_mtx_dir(mtx_dir: Path) -> ad.AnnData | None:
    """Load a 10x Genomics count matrix from a directory with mtx + tsv files."""
    matrix_file = next(mtx_dir.glob("matrix.mtx*"), None)
    barcode_file = next(mtx_dir.glob("barcodes.tsv*"), None)
    feature_file = next(mtx_dir.glob("features.tsv*"), None) or next(
        mtx_dir.glob("genes.tsv*"), None
    )

    if not all([matrix_file, barcode_file, feature_file]):
        return None

    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=True)
    return adata


def load_h5ad_file(path: Path) -> ad.AnnData:
    """Load an h5ad file."""
    return ad.read_h5ad(path)


def load_h5_file(path: Path) -> ad.AnnData:
    """Load a 10x h5 file."""
    return sc.read_10x_h5(path)


def standardize_gene_names(adata: ad.AnnData) -> ad.AnnData:
    """Standardize gene names to uppercase HGNC symbols.

    Removes genes without valid symbols and deduplicates.
    """
    # Store original names
    if "original_gene_name" not in adata.var.columns:
        adata.var["original_gene_name"] = adata.var_names.copy()

    # Uppercase gene symbols
    adata.var_names = [g.upper() for g in adata.var_names]

    # Remove non-gene entries (e.g., ERCC spike-ins, mitochondrial pseudogenes)
    keep = ~adata.var_names.str.startswith("ERCC-")
    adata = adata[:, keep].copy()

    adata.var_names_make_unique()
    return adata


def add_metadata(
    adata: ad.AnnData,
    accession: str,
    batch_label: str | None = None,
    metadata: dict | None = None,
) -> ad.AnnData:
    """Add standard metadata fields to AnnData object."""
    adata.obs["dataset"] = accession
    adata.obs["batch"] = batch_label or accession

    if metadata:
        for key, value in metadata.items():
            if key not in adata.obs.columns:
                adata.obs[key] = value

    return adata


def discover_and_load_supplementary(gse_dir: Path, gse: str) -> list[ad.AnnData]:
    """Auto-discover and load scRNA-seq data from downloaded supplementary files."""
    suppl_dir = gse_dir / "supplementary"
    if not suppl_dir.exists():
        return []

    adatas = []

    # Try h5ad files first
    for h5ad_file in sorted(suppl_dir.glob("*.h5ad")):
        print(f"    Loading h5ad: {h5ad_file.name}")
        adata = load_h5ad_file(h5ad_file)
        adatas.append(adata)

    if adatas:
        return adatas

    # Try h5 files
    for h5_file in sorted(suppl_dir.glob("*.h5")):
        if h5_file.name.endswith(".h5ad"):
            continue
        print(f"    Loading 10x h5: {h5_file.name}")
        try:
            adata = load_h5_file(h5_file)
            adatas.append(adata)
        except Exception as e:
            print(f"    Failed to load {h5_file.name}: {e}")

    if adatas:
        return adatas

    # Try plain .tar archives with flat per-sample 10x files (e.g. GEO RAW.tar)
    for tar_file in sorted(suppl_dir.glob("*.tar")):
        if tar_file.name.endswith(".tar.gz"):
            continue  # handled below
        print(f"    Loading flat 10x tar: {tar_file.name}")
        sample_adatas = load_10x_from_flat_tar(tar_file)
        adatas.extend(sample_adatas)

    if adatas:
        return adatas

    # Try tar.gz archives containing 10x mtx format (single-sample archives)
    for tar_file in sorted(suppl_dir.glob("*.tar.gz")):
        print(f"    Loading 10x tar.gz: {tar_file.name}")
        adata = load_10x_from_tar(tar_file)
        if adata is not None:
            # Use tar filename to label the sample
            sample_name = tar_file.stem.replace(".tar", "").replace(f"{gse}_", "")
            adata.obs["sample"] = sample_name
            adatas.append(adata)

    if adatas:
        return adatas

    # Try extracting and finding mtx directories
    for gz_file in sorted(suppl_dir.glob("*.mtx.gz")):
        parent = gz_file.parent
        print(f"    Loading mtx from: {parent.name}")
        adata = load_10x_from_mtx_dir(parent)
        if adata is not None:
            adatas.append(adata)
            break  # Only load once per directory

    return adatas


def _filter_empty_droplets(adata: ad.AnnData, min_genes: int = 200) -> ad.AnnData:
    """Remove barcodes with fewer than *min_genes* detected genes.

    Raw 10x matrices may contain all 737,280 barcodes including empty droplets.
    Filtering early prevents OOM during concatenation and downstream processing.
    """
    if min_genes <= 0:
        return adata
    n_genes_per_cell = np.diff(adata.X.indptr) if sp.issparse(adata.X) else (adata.X > 0).sum(axis=1)
    keep = np.asarray(n_genes_per_cell >= min_genes).flatten()
    n_removed = int(adata.n_obs - keep.sum())
    if n_removed > 0:
        print(f"  Filtered {n_removed} empty droplets (min_genes={min_genes}), "
              f"{keep.sum()}/{adata.n_obs} cells kept")
        adata = adata[keep].copy()
    return adata


def process_dataset(gse: str, dest_dir: Path, force: bool = False) -> Path | None:
    """Download, load, standardize, and save a single GEO scRNA-seq dataset.

    Returns the path to the saved h5ad file, or None on failure.
    """
    output_path = dest_dir / f"{gse}.h5ad"
    if output_path.exists() and not force:
        print(f"  {output_path.name} already exists, skipping (use --force to reprocess)")
        return output_path

    info = DATASETS.get(gse)
    if not info:
        print(f"  Unknown dataset: {gse}", file=sys.stderr)
        return None

    gse_dir = dest_dir / "raw" / gse
    gse_dir.mkdir(parents=True, exist_ok=True)

    # Download supplementary files
    print(f"\n--- {gse}: {info['description']} ---")
    downloaded = download_supplementary(gse, gse_dir)
    if not downloaded:
        print(f"  No files downloaded for {gse}")
        return None

    # Load data
    print(f"  Loading scRNA-seq data...")
    adatas = discover_and_load_supplementary(gse_dir, gse)

    if not adatas:
        print(f"  Could not load any scRNA-seq data from {gse}")
        return None

    # Concatenate if multiple samples
    if len(adatas) == 1:
        adata = adatas[0]
    else:
        print(f"  Concatenating {len(adatas)} samples...")
        adata = ad.concat(adatas, join="outer", label="sample_file", index_unique="-")

    # Remove empty droplets before heavy downstream work
    adata = _filter_empty_droplets(adata)

    # Standardize
    print(f"  Standardizing gene names...")
    adata = standardize_gene_names(adata)
    adata = add_metadata(adata, gse)

    # Ensure counts are in X
    if sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    print(f"  Final shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"  Saved to {output_path}")

    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess GEO scRNA-seq datasets for IL-23/Th17 atlas"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*DATASETS, "all"],
        default=["all"],
        help="Datasets to download (default: all)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess existing files")
    args = parser.parse_args(argv)

    selected = list(DATASETS) if "all" in args.datasets else args.datasets
    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"IL-23/Th17 Atlas — scRNA-seq Data Download & Preprocessing")
    print(f"Destination: {args.dest}\n")

    results = {}
    for gse in selected:
        try:
            path = process_dataset(gse, args.dest, force=args.force)
            results[gse] = "OK" if path else "FAILED"
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results[gse] = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print("Summary:")
    for gse, status in results.items():
        print(f"  {gse}: {status}")


if __name__ == "__main__":
    main()
