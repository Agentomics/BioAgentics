"""Concatenate GSE282122 batch files using disk-backed merge (8GB RAM safe).

The batch extraction + QC was already done by 02_process_gse282122.py.
This script finishes the job by merging batches via concat_on_disk.

Usage:
    uv run python src/crohns/il23_response_atlas/02b_concat_batches.py
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

os.environ.setdefault("PANDAS_FUTURE_INFER_STRING", "0")

import anndata as ad  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "crohns" / "il23-atlas"
OUTPUT_PATH = DATA_DIR / "GSE282122.h5ad"


def main() -> None:
    if OUTPUT_PATH.exists():
        print(f"Output already exists: {OUTPUT_PATH}")
        print("Delete it to reprocess.")
        return

    batches = sorted(DATA_DIR.glob("_batch_*.h5ad"), key=lambda p: int(p.stem.split("_")[-1]))
    if not batches:
        print("No batch files found in", DATA_DIR)
        return

    print(f"Concatenating {len(batches)} batch files via concat_on_disk...")
    for bp in batches:
        a = ad.read_h5ad(bp, backed="r")
        print(f"  {bp.name}: {a.n_obs} cells x {a.n_vars} genes")
        a.file.close()

    # Step 1: Disk-backed concatenation (memory-safe)
    tmp_concat = DATA_DIR / "_concat_tmp.h5ad"
    ad.experimental.concat_on_disk(
        in_files=[str(p) for p in batches],
        out_file=str(tmp_concat),
        join="inner",
        index_unique="-",
    )
    print(f"  Disk concat done -> {tmp_concat}")

    # Step 2: Post-process (gene standardization) one chunk at a time is not
    # needed since all batches share the same gene space from scanpy read_10x_h5.
    # But we still uppercase gene names and remove ERCC spikes.
    # Load in backed mode to check, then do a targeted fix.
    adata = ad.read_h5ad(tmp_concat, backed="r")
    print(f"  Concatenated shape: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Samples: {adata.obs['sample'].nunique()}")

    # Check if gene names need uppercasing
    var_names = list(adata.var_names)
    needs_upper = any(g != g.upper() for g in var_names[:100])
    has_ercc = any(g.upper().startswith("ERCC-") for g in var_names)
    adata.file.close()
    del adata
    gc.collect()

    if needs_upper or has_ercc:
        print("  Pre-processing batch files (uppercase + remove ERCC)...")
        # Fix gene names in each batch individually to avoid loading the full
        # concatenated file into memory (OOM risk on 8GB machine).
        tmp_concat.unlink()
        for bp in batches:
            a = ad.read_h5ad(bp)
            if "original_gene_name" not in a.var.columns:
                a.var["original_gene_name"] = a.var_names.copy()
            a.var.index = pd.Index([g.upper() for g in a.var_names])
            keep = ~a.var.index.str.startswith("ERCC-")
            a = a[:, keep].copy()
            a.var_names_make_unique()
            a.write_h5ad(bp)
            del a
            gc.collect()
        print("  Re-concatenating cleaned batches...")
        ad.experimental.concat_on_disk(
            in_files=[str(p) for p in batches],
            out_file=str(OUTPUT_PATH),
            join="inner",
            index_unique="-",
        )
    else:
        # Just rename the tmp file
        tmp_concat.rename(OUTPUT_PATH)
        print("  Gene names already clean, no post-processing needed.")

    # Report final stats
    final = ad.read_h5ad(OUTPUT_PATH, backed="r")
    print(f"\nFinal GSE282122 dataset:")
    print(f"  Cells: {final.n_obs}")
    print(f"  Genes: {final.n_vars}")
    print(f"  Samples: {final.obs['sample'].nunique()}")
    print(f"  Median genes/cell: {final.obs['n_genes_by_counts'].median():.0f}")
    print(f"  Median mito%: {final.obs['pct_counts_mt'].median():.1f}%")
    print(f"  Saved to: {OUTPUT_PATH}")
    final.file.close()

    # Clean up
    if tmp_concat.exists():
        tmp_concat.unlink()
    for bp in batches:
        bp.unlink(missing_ok=True)
    print("  Cleaned up temporary batch files.")


if __name__ == "__main__":
    main()
