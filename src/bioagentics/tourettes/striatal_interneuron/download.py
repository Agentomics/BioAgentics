"""Download reference atlas datasets from GEO for striatal interneuron analysis.

Downloads supplementary files from:
- GSE151761: Human dorsal striatum interneuron snRNA-seq (Krienen et al.)
  Format: DGE text (genes x cells), filtered to human striatum files only
- GSE152058: Human dorsal striatum snRNA-seq (part 2)
  Format: MTX + metadata TSVs, filtered to human data only

Converts all downloads to h5ad format for downstream analysis.

Usage:
    uv run python -m bioagentics.tourettes.striatal_interneuron.download
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

# Disable pandas 3.0 Arrow-backed strings to avoid h5ad write failures.
# anndata's internal categorical conversion produces ArrowStringArray
# categories that h5py cannot serialize.
pd.options.future.infer_string = False

from bioagentics.tourettes.striatal_interneuron.config import (
    REFERENCE_DIR,
    REFERENCE_DATASETS,
    ensure_dirs,
)

# GEO FTP-over-HTTPS base for supplementary files
GEO_SUPPL_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/"

# Session with retries
_session = requests.Session()
_adapter = HTTPAdapter(max_retries=3)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

# --- File selection patterns ---
# GSE151761: only download human striatum interneuron files
GSE151761_HUMAN_PATTERNS = [
    "hum.str.int",  # human striatum interneurons (10X and Drop-seq)
    "adHumInt",     # adult human interneurons
]

# GSE152058: only download human snRNA files
GSE152058_HUMAN_PATTERNS = [
    "human_snRNA_processed",  # human snRNA processed data (counts + metadata)
]


def _geo_suppl_url(accession: str) -> str:
    """Build the GEO supplementary file directory URL for an accession."""
    prefix = accession[:-3] + "nnn"  # e.g., GSE151nnn
    return f"{GEO_SUPPL_BASE}{prefix}/{accession}/suppl/"


def list_geo_supplementary(accession: str) -> list[str]:
    """List supplementary file names for a GEO accession."""
    url = _geo_suppl_url(accession)
    print(f"  Listing supplementary files at {url}")
    resp = _session.get(url, timeout=30)
    resp.raise_for_status()
    filenames = re.findall(r'href="([^"]+\.(?:h5ad|h5|mtx|tsv|csv|txt|gz)[^"]*)"', resp.text)
    filenames = sorted(set(Path(f).name for f in filenames))
    return filenames


def _filter_files(files: list[str], patterns: list[str]) -> list[str]:
    """Filter file list to those matching any of the given patterns."""
    return [f for f in files if any(p in f for p in patterns)]


def download_file(url: str, dest: Path, chunk_size: int = 65536) -> None:
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  Cached: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    print(f"  Downloading: {dest.name}")
    resp = _session.get(url, timeout=600, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 / total
                mb = downloaded / 1_048_576
                print(f"\r    {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    if total > 0:
        print()  # newline after progress

    tmp.rename(dest)
    size_mb = dest.stat().st_size / 1_048_576
    print(f"  Saved: {dest.name} ({size_mb:.1f} MB)")


def _decompress_gz(gz_path: Path) -> Path:
    """Decompress a .gz file if the uncompressed version doesn't exist."""
    if not gz_path.name.endswith(".gz"):
        return gz_path

    out_path = gz_path.with_suffix("")  # strip .gz
    if out_path.exists():
        print(f"  Already decompressed: {out_path.name}")
        return out_path

    print(f"  Decompressing: {gz_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"  Decompressed: {out_path.name}")
    return out_path


def _is_custom_mm(path: Path) -> bool:
    """Check if file uses the custom MatrixMarket format with %%GENES headers."""
    with open(path, "r") as f:
        return f.readline().startswith("%%MatrixMarket")


def _parse_custom_mm(path: Path):
    """Parse custom MatrixMarket format with %%GENES and %%CELL_BARCODES headers.

    Returns (csr_matrix, gene_names, cell_barcodes).
    Matrix shape is genes x cells (rows=genes, cols=cells).
    """
    import numpy as np
    from scipy.sparse import coo_matrix

    genes: list[str] = []
    barcodes: list[str] = []
    header_lines = 0
    n_rows = 0
    n_cols = 0
    nnz = 0

    with open(path, "r") as f:
        for line in f:
            header_lines += 1
            if line.startswith("%%GENES"):
                genes.extend(line.rstrip("\n").split("\t")[1:])
            elif line.startswith("%%CELL_BARCODES"):
                barcodes.extend(line.rstrip("\n").split("\t")[1:])
            elif line.startswith("%%"):
                continue  # %%MatrixMarket header
            else:
                # Dimensions line: rows cols nnz
                parts = line.strip().split("\t")
                n_rows, n_cols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
                break

    if nnz == 0:
        raise ValueError(f"No dimensions line found in {path}")

    print(f"    Format: custom MatrixMarket ({n_rows} genes x {n_cols} cells, {nnz:,} nonzeros)")
    print(f"    Parsed {len(genes)} gene names, {len(barcodes)} cell barcodes")

    # Read coordinate data in chunks to limit peak memory (8GB constraint)
    import pandas as pd

    chunk_size = 5_000_000
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []

    reader = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_lines,
        header=None,
        names=["row", "col", "val"],
        dtype={"row": np.int32, "col": np.int32, "val": np.int32},
        chunksize=chunk_size,
    )
    for chunk in reader:
        row_chunks.append(chunk["row"].values - 1)  # convert to 0-indexed
        col_chunks.append(chunk["col"].values - 1)
        data_chunks.append(chunk["val"].values)

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    data = np.concatenate(data_chunks)
    del row_chunks, col_chunks, data_chunks

    mat = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    del rows, cols, data

    return mat, genes, barcodes


def dge_to_h5ad(dge_path: Path, output_path: Path) -> Path:
    """Convert a DGE text file (genes x cells) to h5ad format.

    Supports two formats:
    - Custom MatrixMarket with %%GENES/%%CELL_BARCODES headers (GSE151761)
    - Standard tab-delimited DGE (first column = gene name, remaining = cell barcodes)
    """
    if output_path.exists():
        print(f"  Cached h5ad: {output_path.name}")
        return output_path

    import anndata as ad
    import pandas as pd

    print(f"  Converting DGE to h5ad: {dge_path.name}")

    is_mm = _is_custom_mm(dge_path)

    if is_mm:
        mat, genes, barcodes = _parse_custom_mm(dge_path)
        # Matrix is genes x cells — transpose to cells x genes for AnnData
        X = mat.T.tocsr()
        del mat
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=pd.Index(barcodes, dtype="object")),
            var=pd.DataFrame(index=pd.Index(genes, dtype="object")),
        )
    else:
        # Standard tab-delimited DGE format
        df = pd.read_csv(dge_path, sep="\t", index_col=0)
        # DGE is genes x cells — transpose to cells x genes for AnnData
        adata = ad.AnnData(
            X=df.T.values,
            obs=pd.DataFrame(index=pd.Index(df.columns, dtype="object")),
            var=pd.DataFrame(index=pd.Index(df.index, dtype="object")),
        )

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # Store source metadata
    adata.uns["source_file"] = dge_path.name
    adata.uns["source_format"] = "custom_mm" if is_mm else "dge_text"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"  Wrote h5ad: {output_path.name} ({adata.n_obs} cells x {adata.n_vars} genes)")
    return output_path


def mtx_to_h5ad(
    mtx_path: Path,
    coldata_path: Path,
    rowdata_path: Path,
    output_path: Path,
) -> Path:
    """Convert MTX + metadata TSVs to h5ad format.

    Args:
        mtx_path: Sparse matrix file (.mtx)
        coldata_path: Cell metadata TSV (barcodes/observations)
        rowdata_path: Gene metadata TSV (features/variables)
        output_path: Destination h5ad path
    """
    if output_path.exists():
        print(f"  Cached h5ad: {output_path.name}")
        return output_path

    import anndata as ad
    import pandas as pd
    from scipy.io import mmread

    print(f"  Converting MTX to h5ad: {mtx_path.name}")

    # Read sparse matrix (genes x cells in MTX format)
    X = mmread(str(mtx_path)).T.tocsr()  # transpose to cells x genes

    # Read metadata — force object dtype on indices to avoid ArrowStringArray issues
    obs = pd.read_csv(coldata_path, sep="\t", index_col=0)
    var = pd.read_csv(rowdata_path, sep="\t", index_col=0)
    obs.index = obs.index.astype("object")
    var.index = var.index.astype("object")

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    adata.uns["source_file"] = mtx_path.name
    adata.uns["source_format"] = "mtx"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"  Wrote h5ad: {output_path.name} ({adata.n_obs} cells x {adata.n_vars} genes)")
    return output_path


def validate_h5ad(path: Path) -> bool:
    """Validate that an h5ad file can be read and has expected structure."""
    import anndata as ad

    try:
        adata = ad.read_h5ad(path, backed="r")
        n_obs, n_vars = adata.n_obs, adata.n_vars
        adata.file.close()

        if n_obs == 0 or n_vars == 0:
            print(f"  WARN: {path.name} is empty ({n_obs} cells x {n_vars} genes)")
            return False

        print(f"  Valid: {path.name} ({n_obs} cells x {n_vars} genes)")
        return True
    except Exception as e:
        print(f"  ERROR validating {path.name}: {e}")
        return False


# Cell types to extract for the reference interneuron atlas
GSE152058_INTERNEURON_TYPES = [
    "PV_Interneuron",
    "GABAergic_Interneuron",
    "Cholinergic_Interneuron",
    "FOXP2_Neur",
]


def extract_gse152058_interneurons(
    mtx_path: Path,
    coldata_path: Path,
    rowdata_path: Path,
    output_path: Path,
    cell_types: list[str] | None = None,
    condition: str = "Control",
) -> Path:
    """Extract control interneuron cells from GSE152058 without loading full matrix.

    The full MTX (125K cells, 3.6GB) exceeds 8GB RAM when loaded.
    This streams the MTX coordinate data, keeping only entries for
    target cells, to produce a small h5ad suitable for integration.

    Args:
        mtx_path: Sparse matrix file (.mtx), genes x cells
        coldata_path: Cell metadata TSV (barcodes/observations)
        rowdata_path: Gene metadata TSV (features/variables)
        output_path: Destination h5ad path
        cell_types: Cell types to extract (default: interneuron types)
        condition: Condition to filter on (default: "Control")
    """
    if output_path.exists():
        print(f"  Cached h5ad: {output_path.name}")
        return output_path

    import numpy as np
    import pandas as pd

    if cell_types is None:
        cell_types = GSE152058_INTERNEURON_TYPES

    # Step 1: Read coldata to identify target cell indices
    print(f"  Reading coldata to identify {condition} interneurons...")
    coldata = pd.read_csv(coldata_path, sep="\t", index_col=0)
    mask = (coldata["Condition"] == condition) & (coldata["CellType"].isin(cell_types))
    target_positions = set(np.where(mask.values)[0])  # 0-indexed positions

    n_target = len(target_positions)
    print(f"    Found {n_target} target cells out of {len(coldata)} total")
    print(f"    Types: {coldata.loc[mask, 'CellType'].value_counts().to_dict()}")

    # Build index remapping: old_col_idx -> new_col_idx (0-based, contiguous)
    sorted_positions = sorted(target_positions)
    col_remap = {old: new for new, old in enumerate(sorted_positions)}

    # Extract target cell metadata
    obs = coldata.iloc[sorted_positions].copy()
    obs.index = obs.index.astype("object")

    # Read gene metadata
    var = pd.read_csv(rowdata_path, sep="\t", index_col=0)
    var.index = var.index.astype("object")
    n_genes = len(var)

    del coldata, mask

    # Step 2: Stream MTX file, keeping only target cell entries
    print(f"  Streaming MTX to extract {n_target} cells (memory-safe)...")
    from scipy.sparse import coo_matrix

    # Parse MTX header to find data start
    header_lines = 0
    mtx_rows = 0
    mtx_cols = 0
    mtx_nnz = 0
    with open(mtx_path, "r") as f:
        for line in f:
            header_lines += 1
            if line.startswith("%"):
                continue
            # Dimensions line
            parts = line.strip().split()
            mtx_rows, mtx_cols, mtx_nnz = int(parts[0]), int(parts[1]), int(parts[2])
            break

    if mtx_nnz == 0:
        raise ValueError(f"No dimensions line found in {mtx_path}")

    print(f"    MTX dimensions: {mtx_rows} genes x {mtx_cols} cells, {mtx_nnz:,} nonzeros")

    # Stream coordinate data in chunks, filtering to target columns
    chunk_size = 5_000_000
    row_acc: list[np.ndarray] = []
    col_acc: list[np.ndarray] = []
    data_acc: list[np.ndarray] = []
    kept = 0

    reader = pd.read_csv(
        mtx_path,
        sep=r"\s+",
        skiprows=header_lines,
        header=None,
        names=["row", "col", "val"],
        dtype={"row": np.int32, "col": np.int32, "val": np.int32},
        chunksize=chunk_size,
    )
    for chunk in reader:
        # MTX is 1-indexed; convert col to 0-indexed for lookup
        col_0 = chunk["col"].values - 1
        in_target = np.array([c in target_positions for c in col_0])
        if not in_target.any():
            continue

        rows_kept = chunk["row"].values[in_target] - 1  # 0-indexed genes
        cols_kept = np.array([col_remap[c] for c in col_0[in_target]], dtype=np.int32)
        data_kept = chunk["val"].values[in_target]

        row_acc.append(rows_kept)
        col_acc.append(cols_kept)
        data_acc.append(data_kept)
        kept += len(rows_kept)

    print(f"    Kept {kept:,} nonzeros for {n_target} cells")

    # Step 3: Build sparse matrix and AnnData
    import anndata as ad

    rows = np.concatenate(row_acc)
    cols = np.concatenate(col_acc)
    data = np.concatenate(data_acc)
    del row_acc, col_acc, data_acc

    # Matrix: genes x cells, then transpose to cells x genes
    X = coo_matrix((data, (rows, cols)), shape=(n_genes, n_target)).T.tocsr()
    del rows, cols, data

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    adata.uns["source_file"] = mtx_path.name
    adata.uns["source_format"] = "mtx_subset"
    adata.uns["subset_condition"] = condition
    adata.uns["subset_cell_types"] = cell_types

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"  Wrote h5ad: {output_path.name} ({adata.n_obs} cells x {adata.n_vars} genes)")
    return output_path


def _download_and_decompress(suppl_url: str, filenames: list[str], dest_dir: Path) -> list[Path]:
    """Download files and decompress .gz, returning paths to final files."""
    paths: list[Path] = []
    for fn in filenames:
        url = urljoin(suppl_url, fn)
        dest = dest_dir / fn
        download_file(url, dest)

        if fn.endswith(".gz"):
            paths.append(_decompress_gz(dest))
        else:
            paths.append(dest)
    return paths


def download_gse151761(dest_dir: Path) -> list[Path]:
    """Download and convert human striatum interneuron DGE files from GSE151761."""
    accession = "GSE151761"
    suppl_url = _geo_suppl_url(accession)
    all_files = list_geo_supplementary(accession)
    targets = _filter_files(all_files, GSE151761_HUMAN_PATTERNS)

    if not targets:
        print(f"  WARNING: No human striatum files found for {accession}")
        return []

    print(f"  Selected {len(targets)} human striatum files from {len(all_files)} total")
    raw_paths = _download_and_decompress(suppl_url, targets, dest_dir)

    # Convert each DGE file to h5ad
    h5ad_paths: list[Path] = []
    for raw in raw_paths:
        if raw.suffix == ".txt":
            h5ad_path = raw.with_suffix(".h5ad")
            dge_to_h5ad(raw, h5ad_path)
            if validate_h5ad(h5ad_path):
                h5ad_paths.append(h5ad_path)
        else:
            h5ad_paths.append(raw)

    return h5ad_paths


def download_gse152058(dest_dir: Path) -> list[Path]:
    """Download and extract control interneuron subset from GSE152058.

    The full dataset (125K cells, 3.6GB MTX) is too large for 8GB RAM.
    Instead of loading the full matrix, this extracts only control
    interneuron cells (~2.7K cells) using streaming subset extraction.
    """
    accession = "GSE152058"
    suppl_url = _geo_suppl_url(accession)
    all_files = list_geo_supplementary(accession)
    targets = _filter_files(all_files, GSE152058_HUMAN_PATTERNS)

    if not targets:
        print(f"  WARNING: No human snRNA files found for {accession}")
        return []

    print(f"  Selected {len(targets)} human snRNA files from {len(all_files)} total")
    raw_paths = _download_and_decompress(suppl_url, targets, dest_dir)

    # Group by prefix
    mtx_files = [p for p in raw_paths if p.suffix == ".mtx"]
    coldata_files = [p for p in raw_paths if "coldata" in p.name]
    rowdata_files = [p for p in raw_paths if "rowdata" in p.name]

    h5ad_paths: list[Path] = []

    if mtx_files and coldata_files and rowdata_files:
        # Extract control interneurons only (memory-safe for 8GB RAM)
        h5ad_path = dest_dir / f"{accession}_control_interneurons.h5ad"
        extract_gse152058_interneurons(
            mtx_files[0], coldata_files[0], rowdata_files[0], h5ad_path,
        )
        if validate_h5ad(h5ad_path):
            h5ad_paths.append(h5ad_path)
    else:
        print(f"  WARNING: Incomplete MTX file set for {accession}")
        h5ad_paths.extend(raw_paths)

    return h5ad_paths


def download_reference_atlas(dest_dir: Path | None = None) -> dict[str, list[Path]]:
    """Download all reference atlas datasets and convert to h5ad.

    Returns dict mapping accession -> list of h5ad file paths.
    """
    if dest_dir is None:
        dest_dir = REFERENCE_DIR

    ensure_dirs()
    results: dict[str, list[Path]] = {}

    # GSE151761: DGE format
    print(f"\n=== GSE151761: {REFERENCE_DATASETS['GSE151761']['description']} ===")
    acc_dir = dest_dir / "GSE151761"
    results["GSE151761"] = download_gse151761(acc_dir)
    print(f"  Result: {len(results['GSE151761'])} h5ad files for GSE151761")

    # GSE152058: MTX format
    print(f"\n=== GSE152058: {REFERENCE_DATASETS['GSE152058']['description']} ===")
    acc_dir = dest_dir / "GSE152058"
    results["GSE152058"] = download_gse152058(acc_dir)
    print(f"  Result: {len(results['GSE152058'])} h5ad files for GSE152058")

    # Summary
    total = sum(len(v) for v in results.values())
    print(f"\n=== Download complete: {total} h5ad files total ===")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download reference atlas datasets from GEO"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=REFERENCE_DIR,
        help="Destination directory (default: project reference dir)",
    )
    parser.add_argument(
        "--accession",
        type=str,
        default=None,
        choices=["GSE151761", "GSE152058"],
        help="Download a specific accession only",
    )
    args = parser.parse_args()

    if args.accession == "GSE151761":
        download_gse151761(args.dest / "GSE151761")
    elif args.accession == "GSE152058":
        download_gse152058(args.dest / "GSE152058")
    else:
        download_reference_atlas(args.dest)


if __name__ == "__main__":
    main()
