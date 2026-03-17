"""Interneuron subtype classification using reference atlas taxonomy.

Classifies cells into the 14-subclass interneuron taxonomy from the reference
atlas (GSE151761/GSE152058). Uses two approaches:

1. Label transfer via scanpy.tl.ingest — projects query cells onto reference
   PCA/UMAP space and transfers labels via kNN.
2. Marker-based scoring — uses known marker genes to score each subtype
   independently, providing a second line of evidence.

Outputs: cell-level subclass labels, confidence scores, ambiguity flags.

Usage:
    uv run python -m bioagentics.tourettes.striatal_interneuron.classify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ad.settings.allow_write_nullable_strings = True

from bioagentics.tourettes.striatal_interneuron.config import (
    INTEGRATION_DIR,
    INTERNEURON_MARKERS,
    OUTPUT_DIR,
    STRIATAL_CELL_MARKERS,
    ensure_dirs,
)


def score_marker_genes(
    adata: ad.AnnData,
    marker_dict: dict[str, list[str]] | None = None,
    prefix: str = "score_",
) -> ad.AnnData:
    """Score cells for each cell type using marker gene sets.

    Uses scanpy.tl.score_genes to compute per-cell scores for each type.
    Adds columns '{prefix}{type_name}' to adata.obs.
    """
    if marker_dict is None:
        marker_dict = {**INTERNEURON_MARKERS, **STRIATAL_CELL_MARKERS}

    available_genes = set(adata.var_names)

    for cell_type, markers in marker_dict.items():
        valid_markers = [g for g in markers if g in available_genes]
        if len(valid_markers) < 2:
            adata.obs[f"{prefix}{cell_type}"] = 0.0
            continue
        sc.tl.score_genes(adata, valid_markers, score_name=f"{prefix}{cell_type}")

    print(f"  Scored {len(marker_dict)} cell types using marker genes")
    return adata


def classify_by_markers(
    adata: ad.AnnData,
    interneuron_markers: dict[str, list[str]] | None = None,
    broad_markers: dict[str, list[str]] | None = None,
    score_prefix: str = "score_",
) -> ad.AnnData:
    """Classify cells based on marker gene scores.

    Assigns each cell to the type with the highest marker score.
    Adds 'broad_type' and 'interneuron_subclass' columns.
    """
    if interneuron_markers is None:
        interneuron_markers = INTERNEURON_MARKERS
    if broad_markers is None:
        broad_markers = STRIATAL_CELL_MARKERS

    # Broad classification
    broad_cols = [f"{score_prefix}{t}" for t in broad_markers if f"{score_prefix}{t}" in adata.obs]
    if broad_cols:
        broad_scores = adata.obs[broad_cols].values
        best_idx = np.argmax(broad_scores, axis=1)
        type_names = [col.removeprefix(score_prefix) for col in broad_cols]
        adata.obs["broad_type"] = [type_names[i] for i in best_idx]
        adata.obs["broad_type_score"] = np.max(broad_scores, axis=1)
        print(f"  Broad classification: {adata.obs['broad_type'].value_counts().to_dict()}")

    # Interneuron subclass classification
    int_cols = [f"{score_prefix}{t}" for t in interneuron_markers if f"{score_prefix}{t}" in adata.obs]
    if int_cols:
        int_scores = adata.obs[int_cols].values
        best_idx = np.argmax(int_scores, axis=1)
        subclass_names = [col.removeprefix(score_prefix) for col in int_cols]
        adata.obs["interneuron_subclass"] = [subclass_names[i] for i in best_idx]
        adata.obs["interneuron_subclass_score"] = np.max(int_scores, axis=1)

        # Confidence: ratio of best to second-best score
        if int_scores.shape[1] > 1:
            sorted_scores = np.sort(int_scores, axis=1)[:, ::-1]
            second_best = sorted_scores[:, 1]
            # Avoid division by zero
            denom = np.where(np.abs(second_best) > 1e-10, np.abs(second_best), 1e-10)
            confidence = sorted_scores[:, 0] / denom
            adata.obs["classification_confidence"] = confidence
        else:
            adata.obs["classification_confidence"] = 1.0

        print(f"  Interneuron subclass: {adata.obs['interneuron_subclass'].value_counts().to_dict()}")

    return adata


def flag_ambiguous(
    adata: ad.AnnData,
    confidence_threshold: float = 1.5,
    min_score: float = 0.0,
) -> ad.AnnData:
    """Flag cells with ambiguous classification.

    A cell is ambiguous if:
    - Confidence (best/second-best ratio) is below threshold
    - Best score is below min_score
    """
    ambiguous = np.zeros(adata.n_obs, dtype=bool)

    if "classification_confidence" in adata.obs:
        ambiguous |= adata.obs["classification_confidence"].values < confidence_threshold

    if "interneuron_subclass_score" in adata.obs:
        ambiguous |= adata.obs["interneuron_subclass_score"].values < min_score

    adata.obs["classification_ambiguous"] = ambiguous
    n_ambiguous = ambiguous.sum()
    pct = n_ambiguous / adata.n_obs * 100
    print(f"  Ambiguous cells: {n_ambiguous}/{adata.n_obs} ({pct:.1f}%)")
    return adata


def transfer_labels_ingest(
    query: ad.AnnData,
    reference: ad.AnnData,
    label_key: str = "interneuron_subclass",
    embedding_method: str = "umap",
) -> ad.AnnData:
    """Transfer labels from reference to query using scanpy.tl.ingest.

    Projects query cells into the reference PCA/UMAP space and assigns
    labels via kNN voting.
    """
    if label_key not in reference.obs:
        print(f"  WARNING: '{label_key}' not in reference .obs, skipping label transfer")
        return query

    print(f"  Label transfer via ingest (label_key={label_key})")

    # Ensure both have compatible var
    common_genes = reference.var_names.intersection(query.var_names)
    if len(common_genes) < 100:
        print(f"  WARNING: Only {len(common_genes)} common genes, label transfer may be unreliable")

    ref_sub = reference[:, common_genes].copy()
    query_sub = query[:, common_genes].copy()

    # ingest needs PCA and neighbors on reference
    if "X_pca" not in ref_sub.obsm:
        sc.pp.scale(ref_sub, max_value=10)
        sc.tl.pca(ref_sub)
    if "neighbors" not in ref_sub.uns:
        sc.pp.neighbors(ref_sub)

    sc.tl.ingest(query_sub, ref_sub, obs=label_key, embedding_method=embedding_method)

    # Copy transferred labels back to full query
    query.obs[f"transferred_{label_key}"] = query_sub.obs[label_key].values
    if "X_umap" in query_sub.obsm:
        query.obsm["X_umap_transferred"] = query_sub.obsm["X_umap"]

    print(f"  Transferred labels: {query.obs[f'transferred_{label_key}'].value_counts().to_dict()}")
    return query


def consensus_classification(
    adata: ad.AnnData,
    marker_key: str = "interneuron_subclass",
    transfer_key: str = "transferred_interneuron_subclass",
) -> ad.AnnData:
    """Compute consensus classification from marker-based and transfer-based labels.

    If both methods agree, high confidence. If they disagree, flag as ambiguous.
    """
    if marker_key not in adata.obs or transfer_key not in adata.obs:
        # Only one method available — use whatever we have
        if marker_key in adata.obs:
            adata.obs["consensus_subclass"] = adata.obs[marker_key]
        elif transfer_key in adata.obs:
            adata.obs["consensus_subclass"] = adata.obs[transfer_key]
        return adata

    marker_labels = adata.obs[marker_key].values
    transfer_labels = adata.obs[transfer_key].values
    agree = marker_labels == transfer_labels

    # Consensus: use marker label, but flag disagreement
    adata.obs["consensus_subclass"] = marker_labels
    adata.obs["methods_agree"] = agree

    n_agree = agree.sum()
    pct = n_agree / len(agree) * 100
    print(f"  Consensus: {n_agree}/{len(agree)} cells agree ({pct:.1f}%)")

    # Mark disagreements as ambiguous too
    if "classification_ambiguous" in adata.obs:
        adata.obs["classification_ambiguous"] = adata.obs["classification_ambiguous"] | ~agree

    return adata


def run_classification(
    integrated_path: Path | None = None,
    reference_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    """Run full classification pipeline.

    Args:
        integrated_path: Path to integrated h5ad (or first QC'd file)
        reference_path: Path to reference atlas h5ad (for label transfer)
        output_dir: Output directory

    Returns path to classified output h5ad.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "classification"

    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classified.h5ad"

    if output_path.exists():
        print(f"  Classification already done: {output_path.name}")
        return output_path

    # Find input file
    if integrated_path is None:
        candidates = sorted(INTEGRATION_DIR.glob("integrated_*.h5ad"))
        if candidates:
            integrated_path = candidates[0]
        else:
            print("  No integrated file found. Run integration phase first.")
            return None

    print(f"\n=== Classification: {integrated_path.name} ===")
    adata = ad.read_h5ad(integrated_path)
    print(f"  Input: {adata.n_obs} cells x {adata.n_vars} genes")

    # Step 1: Marker-based scoring and classification
    adata = score_marker_genes(adata)
    adata = classify_by_markers(adata)
    adata = flag_ambiguous(adata)

    # Step 2: Label transfer from reference (if available)
    if reference_path is not None and reference_path.exists():
        ref = ad.read_h5ad(reference_path)
        if "interneuron_subclass" in ref.obs:
            adata = transfer_labels_ingest(adata, ref)
            adata = consensus_classification(adata)

    adata.write_h5ad(output_path)
    print(f"  Output: {output_path.name} ({adata.n_obs} cells)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Interneuron subtype classification")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input integrated h5ad file",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference atlas h5ad with 'interneuron_subclass' labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "classification",
        help="Output directory",
    )
    args = parser.parse_args()

    run_classification(args.input, args.reference, args.output_dir)


if __name__ == "__main__":
    main()
