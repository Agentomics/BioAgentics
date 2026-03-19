"""Local L1000/CMAP data processing using public GEO GCTX files.

Replaces clue.io API with local computation using publicly available
Level 5 (replicate-collapsed z-score) data from GEO:
- GSE92742: LINCS Phase II (473,647 profiles, 978 landmark genes)
- GSE70138: LINCS Phase I

GCTX files are HDF5 format. We use h5py directly (lighter than cmapPy)
with slice operations to stay within the 8GB RAM constraint.

Connectivity scoring uses a simplified weighted enrichment approach
(like the original CMAP algorithm) computed locally.

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.l1000_local --download
    uv run python -m bioagentics.data.cd_fibrosis.l1000_local --score path/to/sig.tsv
"""

from __future__ import annotations

import argparse
import gzip
import struct
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "crohns" / "l1000"
OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing"

# GEO supplementary file URLs for Level 5 (replicate-collapsed z-scores)
# Level 5 = most compact; only 978 landmark genes x perturbation signatures
GEO_GCTX_URLS = {
    "phase2": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
        "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
    ),
    "phase1": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/"
        "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz"
    ),
}

# Companion metadata files (signature info: compound, cell line, dose, time)
GEO_META_URLS = {
    "phase2_siginfo": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
        "GSE92742_Broad_LINCS_sig_info.txt.gz"
    ),
    "phase2_geneinfo": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
        "GSE92742_Broad_LINCS_gene_info.txt.gz"
    ),
}

# Fibroblast-relevant cell lines in L1000
FIBROBLAST_CELLS = {"A549", "IMR90", "WI38", "BJ", "HFF", "CCD18CO", "NPC"}

# Batch size for processing GCTX columns (profiles) at a time
# ~978 genes x 5000 profiles x 4 bytes = ~20MB per batch, very safe for 8GB
BATCH_SIZE = 5000


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with progress reporting. Handles .gz decompression."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)

    # If target already exists (uncompressed), skip
    final = dest.with_suffix("") if dest.suffix == ".gz" else dest
    if final.exists():
        print(f"  Already exists: {final}")
        return final

    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct:.1f}%)",
                      end="", flush=True)

    print()

    # Decompress if gzipped
    if dest.suffix == ".gz":
        print(f"  Decompressing {dest.name}...")
        out_path = dest.with_suffix("")
        with gzip.open(dest, "rb") as gz_in, open(out_path, "wb") as f_out:
            while True:
                chunk = gz_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
        dest.unlink()
        return out_path

    return dest


def download_l1000_data(data_dir: Path | None = None, phase: str = "phase2") -> dict[str, Path]:
    """Download L1000 Level 5 GCTX and metadata files from GEO.

    WARNING: Phase II GCTX is ~3GB compressed, ~12GB uncompressed.
    Ensure sufficient disk space.

    Returns dict of downloaded file paths.
    """
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Download GCTX
    if phase in GEO_GCTX_URLS:
        url = GEO_GCTX_URLS[phase]
        fname = url.split("/")[-1]
        paths["gctx"] = download_file(url, data_dir / fname)

    # Download metadata
    for key, url in GEO_META_URLS.items():
        if key.startswith(phase):
            fname = url.split("/")[-1]
            paths[key.split("_", 1)[1]] = download_file(url, data_dir / fname)

    return paths


class GctxReader:
    """Memory-efficient reader for GCTX (HDF5) files.

    GCTX structure:
      /0/DATA/0/matrix  - expression matrix (genes x profiles)
      /0/META/ROW/id    - gene IDs (rows)
      /0/META/COL/id    - signature IDs (columns)
    """

    def __init__(self, path: Path):
        self.path = path
        self._h5: h5py.File | None = None

    def open(self) -> None:
        self._h5 = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._h5:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            raise RuntimeError("GCTX file not open. Use 'with' or call open().")
        return self._h5

    @property
    def shape(self) -> tuple[int, int]:
        """(n_genes, n_profiles)."""
        return self.h5["0/DATA/0/matrix"].shape

    @property
    def row_ids(self) -> list[str]:
        """Gene IDs (row metadata)."""
        raw = self.h5["0/META/ROW/id"][:]
        return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]

    @property
    def col_ids(self) -> list[str]:
        """Signature IDs (column metadata)."""
        raw = self.h5["0/META/COL/id"][:]
        return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]

    def read_slice(
        self,
        row_indices: np.ndarray | list[int] | None = None,
        col_start: int = 0,
        col_end: int | None = None,
    ) -> np.ndarray:
        """Read a slice of the expression matrix.

        Args:
            row_indices: Row (gene) indices to read. None = all rows.
            col_start: Starting column index.
            col_end: Ending column index (exclusive). None = to end.

        Returns:
            2D numpy array (n_selected_genes x n_profiles_in_slice).
        """
        matrix = self.h5["0/DATA/0/matrix"]
        n_rows, n_cols = matrix.shape
        col_end = col_end or n_cols

        if row_indices is not None:
            row_idx = np.array(sorted(row_indices))
            # h5py fancy indexing: read selected rows, column slice
            return matrix[row_idx, col_start:col_end]
        else:
            return matrix[:, col_start:col_end]


def load_siginfo(path: Path) -> pd.DataFrame:
    """Load L1000 signature metadata (sig_id, pert_iname, cell_id, etc.).

    Reads in chunks to limit memory usage.
    """
    return pd.read_csv(path, sep="\t", low_memory=True)


def load_geneinfo(path: Path) -> pd.DataFrame:
    """Load L1000 gene metadata (pr_gene_id, pr_gene_symbol, pr_is_lm)."""
    return pd.read_csv(path, sep="\t", low_memory=True)


def map_genes_to_rows(
    gctx_row_ids: list[str],
    gene_info: pd.DataFrame,
    query_genes: list[str],
) -> dict[str, int]:
    """Map gene symbols to GCTX row indices.

    Gene info maps pr_gene_id (in GCTX row IDs) to pr_gene_symbol.
    Returns {gene_symbol: row_index} for genes found in both query and GCTX.
    """
    # Build gene_id -> symbol mapping
    id_to_symbol = dict(zip(
        gene_info["pr_gene_id"].astype(str),
        gene_info["pr_gene_symbol"].astype(str),
    ))

    # Build row_index -> symbol mapping
    row_to_symbol = {}
    for idx, rid in enumerate(gctx_row_ids):
        symbol = id_to_symbol.get(str(rid), "")
        if symbol:
            row_to_symbol[idx] = symbol.upper()

    # Map query genes to row indices
    query_upper = {g.upper() for g in query_genes}
    symbol_to_row = {}
    for idx, symbol in row_to_symbol.items():
        if symbol in query_upper:
            symbol_to_row[symbol] = idx

    return symbol_to_row


def compute_connectivity_score(
    profile_zscores: np.ndarray,
    up_indices: list[int],
    down_indices: list[int],
) -> float:
    """Compute connectivity score for a single perturbation profile.

    Uses the weighted connectivity score (WTCS) approach:
    - Rank genes by z-score in the perturbation profile
    - Compute enrichment of UP genes in the bottom (downregulated by drug)
    - Compute enrichment of DOWN genes in the top (upregulated by drug)
    - WTCS = mean of absolute enrichment scores
    - Negative WTCS = drug reverses the disease signature

    Simplified from the original Subramanian et al. (2017) algorithm.
    """
    n = len(profile_zscores)
    if n == 0 or (not up_indices and not down_indices):
        return 0.0

    # Rank genes by z-score (ascending: most downregulated first)
    ranks = np.argsort(np.argsort(profile_zscores)) + 1  # 1-based ranks

    scores = []

    # For UP genes: we want them to be DOWNREGULATED by the drug (low rank)
    # So a good reversal has UP genes at low ranks (negative enrichment)
    if up_indices:
        up_ranks = ranks[up_indices]
        # Normalized rank position (0 = most downregulated, 1 = most upregulated)
        up_positions = up_ranks / n
        # Score: negative if UP genes are at low ranks (drug downregulates them)
        up_score = np.mean(up_positions) - 0.5
        scores.append(-up_score)  # Negate so negative = reversal

    # For DOWN genes: we want them to be UPREGULATED by the drug (high rank)
    if down_indices:
        down_ranks = ranks[down_indices]
        down_positions = down_ranks / n
        # Score: positive if DOWN genes are at high ranks (drug upregulates them)
        down_score = np.mean(down_positions) - 0.5
        scores.append(down_score)  # Positive = reversal

    # Combined: negative = signature reversal = anti-fibrotic potential
    return -np.mean(scores)


def score_signatures_batch(
    gctx: GctxReader,
    gene_row_map: dict[str, int],
    up_genes: list[str],
    down_genes: list[str],
    col_start: int,
    col_end: int,
) -> np.ndarray:
    """Score a batch of perturbation profiles against our signature.

    Returns array of connectivity scores for columns [col_start:col_end].
    """
    all_gene_indices = sorted(set(gene_row_map.values()))
    if not all_gene_indices:
        return np.zeros(col_end - col_start)

    # Read only the rows we need
    data = gctx.read_slice(all_gene_indices, col_start, col_end)

    # Map from GCTX row positions to indices in our sliced data
    idx_map = {orig: i for i, orig in enumerate(all_gene_indices)}

    up_local = [idx_map[gene_row_map[g]] for g in up_genes if g in gene_row_map and gene_row_map[g] in idx_map]
    down_local = [idx_map[gene_row_map[g]] for g in down_genes if g in gene_row_map and gene_row_map[g] in idx_map]

    n_profiles = data.shape[1]
    scores = np.zeros(n_profiles)

    for i in range(n_profiles):
        scores[i] = compute_connectivity_score(data[:, i], up_local, down_local)

    return scores


def run_local_scoring(
    gctx_path: Path,
    siginfo_path: Path,
    geneinfo_path: Path,
    up_genes: list[str],
    down_genes: list[str],
    cell_lines: set[str] | None = None,
    pert_types: set[str] | None = None,
    batch_size: int = BATCH_SIZE,
    top_n: int = 200,
) -> pd.DataFrame:
    """Run local connectivity scoring against L1000 GCTX data.

    Processes in batches to stay within 8GB RAM.

    Args:
        gctx_path: Path to Level 5 GCTX file.
        siginfo_path: Path to signature info TSV.
        geneinfo_path: Path to gene info TSV.
        up_genes: Genes upregulated in fibrosis.
        down_genes: Genes downregulated/protective.
        cell_lines: Optional filter for specific cell lines.
        pert_types: Optional filter for perturbation types (default: trt_cp = compounds).
        batch_size: Columns per batch (memory control).
        top_n: Number of top results to return.

    Returns:
        DataFrame with top compound scores.
    """
    cell_lines = cell_lines or FIBROBLAST_CELLS
    pert_types = pert_types or {"trt_cp"}

    print(f"  Loading metadata...")
    siginfo = load_siginfo(siginfo_path)
    geneinfo = load_geneinfo(geneinfo_path)

    # Filter siginfo for relevant perturbation types and cell lines
    mask = siginfo["pert_type"].isin(pert_types)
    if cell_lines:
        mask &= siginfo["cell_id"].isin(cell_lines)
    filtered_siginfo = siginfo[mask].reset_index(drop=True)
    print(f"  Filtered signatures: {len(filtered_siginfo)} "
          f"(from {len(siginfo)} total, "
          f"cell lines: {cell_lines}, pert types: {pert_types})")

    if len(filtered_siginfo) == 0:
        print("  WARNING: No signatures match filters.")
        return pd.DataFrame()

    with GctxReader(gctx_path) as gctx:
        n_genes, n_profiles = gctx.shape
        print(f"  GCTX shape: {n_genes} genes x {n_profiles} profiles")

        # Map genes
        row_ids = gctx.row_ids
        gene_row_map = map_genes_to_rows(row_ids, geneinfo,
                                          up_genes + down_genes)

        up_mapped = [g for g in up_genes if g.upper() in gene_row_map]
        down_mapped = [g for g in down_genes if g.upper() in gene_row_map]
        print(f"  Mapped: {len(up_mapped)}/{len(up_genes)} up, "
              f"{len(down_mapped)}/{len(down_genes)} down genes to L1000")

        if not up_mapped and not down_mapped:
            print("  ERROR: No query genes found in L1000 data.")
            return pd.DataFrame()

        # Build column index for filtered signatures
        col_ids = gctx.col_ids
        col_id_to_idx = {cid: i for i, cid in enumerate(col_ids)}

        filtered_siginfo = filtered_siginfo[
            filtered_siginfo["sig_id"].isin(col_id_to_idx)
        ].copy()
        filtered_siginfo["col_idx"] = filtered_siginfo["sig_id"].map(col_id_to_idx)
        filtered_siginfo = filtered_siginfo.sort_values("col_idx").reset_index(drop=True)

        print(f"  Scoring {len(filtered_siginfo)} profiles in batches of {batch_size}...")

        # Score in batches
        all_scores = []
        col_indices = filtered_siginfo["col_idx"].values

        for batch_start in range(0, len(col_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(col_indices))
            batch_cols = col_indices[batch_start:batch_end]

            # Read each column individually (they may not be contiguous)
            all_gene_idx = sorted(set(gene_row_map.values()))
            if not all_gene_idx:
                all_scores.extend([0.0] * len(batch_cols))
                continue

            idx_map = {orig: i for i, orig in enumerate(all_gene_idx)}
            up_local = [idx_map[gene_row_map[g.upper()]] for g in up_mapped
                        if g.upper() in gene_row_map and gene_row_map[g.upper()] in idx_map]
            down_local = [idx_map[gene_row_map[g.upper()]] for g in down_mapped
                          if g.upper() in gene_row_map and gene_row_map[g.upper()] in idx_map]

            for col_idx in batch_cols:
                # Read single column slice for our gene rows
                profile = gctx.h5["0/DATA/0/matrix"][all_gene_idx, col_idx]
                score = compute_connectivity_score(profile, up_local, down_local)
                all_scores.append(score)

            pct = batch_end / len(col_indices) * 100
            print(f"\r    Progress: {batch_end}/{len(col_indices)} ({pct:.1f}%)",
                  end="", flush=True)

        print()

    # Build results DataFrame
    filtered_siginfo = filtered_siginfo.copy()
    filtered_siginfo["connectivity_score"] = all_scores

    # Aggregate by compound (mean score across cell lines/doses/timepoints)
    compound_scores = (
        filtered_siginfo
        .groupby("pert_iname")
        .agg(
            mean_score=("connectivity_score", "mean"),
            min_score=("connectivity_score", "min"),
            n_profiles=("connectivity_score", "size"),
            cell_lines=("cell_id", lambda x: ";".join(sorted(x.unique()))),
        )
        .reset_index()
        .rename(columns={"pert_iname": "compound"})
        .sort_values("mean_score")
        .head(top_n)
    )

    return compound_scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Local L1000 connectivity scoring for CD fibrosis"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download L1000 Level 5 data from GEO (WARNING: ~3GB compressed)",
    )
    parser.add_argument(
        "--download-meta-only",
        action="store_true",
        help="Download only metadata files (signature info, gene info) — small",
    )
    parser.add_argument(
        "--phase",
        default="phase2",
        choices=["phase1", "phase2"],
        help="L1000 phase (default: phase2 = GSE92742)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show info about downloaded GCTX file",
    )
    args = parser.parse_args(argv)

    if args.download:
        print("Downloading L1000 Level 5 data from GEO...")
        paths = download_l1000_data(args.data_dir, args.phase)
        for key, path in paths.items():
            print(f"  {key}: {path}")
        print("Done.")

    elif args.download_meta_only:
        print("Downloading L1000 metadata from GEO...")
        args.data_dir.mkdir(parents=True, exist_ok=True)
        for key, url in GEO_META_URLS.items():
            fname = url.split("/")[-1]
            download_file(url, args.data_dir / fname)
        print("Done.")

    elif args.info:
        gctx_files = list(args.data_dir.glob("*.gctx"))
        if not gctx_files:
            print(f"No GCTX files found in {args.data_dir}")
            sys.exit(1)
        for gctx_path in gctx_files:
            print(f"\n{gctx_path.name}:")
            with GctxReader(gctx_path) as reader:
                n_genes, n_profiles = reader.shape
                print(f"  Shape: {n_genes} genes x {n_profiles} profiles")
                print(f"  First 5 row IDs: {reader.row_ids[:5]}")
                print(f"  First 5 col IDs: {reader.col_ids[:5]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
