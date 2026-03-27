"""Process GSE282122 (Thomas et al. Nat Immunol 2024) for IL-23 atlas.

Downloads filtered 10x h5 files from the tar.gz, applies QC per sample,
and saves a concatenated h5ad. Uses batch processing to stay within 8GB RAM.

Usage:
    uv run python src/crohns/il23_response_atlas/02_process_gse282122.py
"""

from __future__ import annotations

import gc
import os
import sys
import tarfile
import tempfile
from pathlib import Path

os.environ.setdefault("PANDAS_FUTURE_INFER_STRING", "0")

import anndata as ad  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "crohns" / "il23-atlas"
TAR_PATH = DATA_DIR / "raw" / "GSE282122" / "supplementary" / "GSE282122_filtered_processed_data.tar.gz"
OUTPUT_PATH = DATA_DIR / "GSE282122.h5ad"

# QC thresholds from the research plan
MIN_GENES = 500
MAX_MITO_PCT = 20.0
BATCH_SIZE = 20  # samples per batch to limit memory


def qc_sample(adata: ad.AnnData, sample_id: str) -> ad.AnnData | None:
    """Apply QC to a single sample: min genes, max mito %."""
    adata.var_names_make_unique()

    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False)

    n_before = adata.n_obs

    # Filter cells
    keep = (adata.obs["n_genes_by_counts"] >= MIN_GENES) & (
        adata.obs["pct_counts_mt"] <= MAX_MITO_PCT
    )
    adata = adata[keep].copy()

    if adata.n_obs == 0:
        print(f"    {sample_id}: {n_before} -> 0 cells (all filtered), skipping")
        return None

    # Add sample metadata
    adata.obs["sample"] = sample_id
    adata.obs["dataset"] = "GSE282122"

    # Keep only essential columns to save memory
    keep_cols = ["sample", "dataset", "n_genes_by_counts", "total_counts", "pct_counts_mt"]
    adata.obs = adata.obs[keep_cols]

    # Ensure sparse CSR
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    elif not sp.isspmatrix_csr(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # Keep minimal var columns
    keep_var = [c for c in ["gene_ids", "feature_types"] if c in adata.var.columns]
    adata.var = adata.var[keep_var]

    print(f"    {sample_id}: {n_before} -> {adata.n_obs} cells")
    return adata


def process_batch(tar, members: list[tarfile.TarInfo]) -> ad.AnnData | None:
    """Extract, load, and QC a batch of samples from the tar archive."""
    batch_adatas = []

    for m in members:
        sample_id = m.name.split("/")[1]
        with tempfile.TemporaryDirectory() as tmp:
            tar.extract(m, tmp, filter="data")
            h5_path = os.path.join(tmp, m.name)
            try:
                adata = sc.read_10x_h5(h5_path)
                adata = qc_sample(adata, sample_id)
                if adata is not None:
                    batch_adatas.append(adata)
            except Exception as e:
                print(f"    {sample_id}: ERROR {e}", file=sys.stderr)

    if not batch_adatas:
        return None

    if len(batch_adatas) == 1:
        return batch_adatas[0]

    merged = ad.concat(batch_adatas, join="outer", index_unique="-")
    del batch_adatas
    gc.collect()
    return merged


def main() -> None:
    if OUTPUT_PATH.exists():
        print(f"Output already exists: {OUTPUT_PATH}")
        print("Delete it to reprocess.")
        return

    if not TAR_PATH.exists():
        print(f"TAR file not found: {TAR_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing GSE282122 (Thomas et al. Nat Immunol 2024)")
    print(f"  Source: {TAR_PATH}")
    print(f"  QC: min_genes={MIN_GENES}, max_mito={MAX_MITO_PCT}%")
    print(f"  Batch size: {BATCH_SIZE} samples")
    print()

    # Open tar and get all h5 members
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        h5_members = [m for m in tar.getmembers() if m.name.endswith(".h5")]
        print(f"  Found {len(h5_members)} samples")

        # Process in batches
        batch_results = []
        for i in range(0, len(h5_members), BATCH_SIZE):
            batch = h5_members[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(h5_members) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} samples)")

            result = process_batch(tar, batch)
            if result is not None:
                # Save batch to disk to free memory
                batch_path = DATA_DIR / f"_batch_{batch_num}.h5ad"
                result.write_h5ad(batch_path)
                batch_results.append(batch_path)
                n_cells = result.n_obs
                del result
                gc.collect()
                print(f"  Batch {batch_num}: {n_cells} cells saved to temp file")

    # Concatenate all batches using disk-backed merge (8GB RAM safe)
    if not batch_results:
        print("  ERROR: No data after QC!", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Concatenating {len(batch_results)} batches via concat_on_disk...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ad.experimental.concat_on_disk(
        in_files=[str(p) for p in batch_results],
        out_file=str(OUTPUT_PATH),
        join="inner",
        index_unique="-",
    )

    # Report stats
    final = ad.read_h5ad(OUTPUT_PATH, backed="r")
    print(f"\n  Final shape: {final.n_obs} cells x {final.n_vars} genes")
    print(f"  Samples: {final.obs['sample'].nunique()}")
    print(f"  Median genes/cell: {final.obs['n_genes_by_counts'].median():.0f}")
    print(f"  Median mito%: {final.obs['pct_counts_mt'].median():.1f}%")
    print(f"  Saved to {OUTPUT_PATH}")
    final.file.close()

    # Clean up batch files
    for bp in batch_results:
        bp.unlink(missing_ok=True)
    print("  Cleaned up temporary batch files")


if __name__ == "__main__":
    main()
