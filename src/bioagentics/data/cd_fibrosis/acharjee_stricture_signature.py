"""Derive Acharjee 8-gene stricture signature for CMAP/L1000 queries.

Validated stricture-associated targets from IBD Journal 2025
(doi:10.1093/ibd/izae295). GREM1 and SERPINE1 overlap with existing
cell-type-resolved signature, validating those targets. HDAC1 is
well-characterized in L1000 (HDAC inhibitors are a strong drug class
for repurposing).

Query separately from fistula signatures — different pathobiology
(stricturing vs fistulizing CD).

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.acharjee_stricture_signature
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.cd_fibrosis.msigdb import (
    DEFAULT_DEST as MSIGDB_DEST,
    filter_fibrosis_sets,
    parse_gmt,
)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing" / "signatures"

# ── Acharjee 8-gene stricture signature ──
#
# Source: IBD Journal 2025, doi:10.1093/ibd/izae295
# Validated stricture-associated gene set derived from multi-cohort
# transcriptomic analysis of stricturing (B2) vs non-stricturing CD.

ACHARJEE_STRICTURE_GENES: dict[str, dict] = {
    "LY96": {
        "direction": "up",
        "description": "Lymphocyte antigen 96 (MD-2), TLR4 co-receptor; innate immune activation in stricture",
    },
    "AKAP11": {
        "direction": "up",
        "description": "A-kinase anchoring protein 11; PKA signaling scaffold in fibroblasts",
    },
    "SRM": {
        "direction": "up",
        "description": "Spermidine synthase; polyamine metabolism, fibroblast proliferation",
    },
    "GREM1": {
        "direction": "up",
        "description": "Gremlin-1, BMP antagonist; promotes fibroblast activation, overlaps cell-type signature",
    },
    "EHD2": {
        "direction": "up",
        "description": "EH domain-containing 2; caveolae-mediated mechanotransduction in fibroblasts",
    },
    "SERPINE1": {
        "direction": "up",
        "description": "PAI-1, serine protease inhibitor; ECM accumulation, druggable, overlaps cell-type signature",
    },
    "HDAC1": {
        "direction": "up",
        "description": "Histone deacetylase 1; epigenetic fibrosis regulator, well-characterized in L1000 (HDAC inhibitors)",
    },
    "FGF2": {
        "direction": "up",
        "description": "Fibroblast growth factor 2; fibroblast proliferation and migration",
    },
}


def build_acharjee_stricture_signature(msigdb_dir: Path | None = None) -> pd.DataFrame:
    """Build the Acharjee 8-gene stricture signature.

    Annotates the validated gene set with MSigDB pathway membership and
    cross-references with other project signatures.
    """
    msigdb_dir = msigdb_dir or MSIGDB_DEST

    records = []
    for gene, info in ACHARJEE_STRICTURE_GENES.items():
        records.append({
            "gene": gene.upper(),
            "direction": info["direction"],
            "description": info["description"],
        })

    df = pd.DataFrame(records)

    # Annotate with MSigDB pathway membership
    msigdb_pathways: dict[str, list[str]] = {}
    for gmt_file in msigdb_dir.glob("*.gmt"):
        if gmt_file.name == "fibrosis_relevant_sets.gmt":
            continue
        try:
            for set_name, genes in filter_fibrosis_sets(parse_gmt(gmt_file)).items():
                for g in genes:
                    msigdb_pathways.setdefault(g.upper(), []).append(set_name)
        except Exception:
            continue

    df["msigdb_pathways"] = df["gene"].map(
        lambda g: "; ".join(sorted(set(msigdb_pathways.get(g, []))))
    )
    df["n_msigdb_pathways"] = df["gene"].map(
        lambda g: len(set(msigdb_pathways.get(g, [])))
    )

    # Flag genes that overlap with other project signatures
    overlap_genes = {"GREM1", "SERPINE1"}
    df["overlaps_celltype_signature"] = df["gene"].isin(overlap_genes)

    # Flag L1000 well-characterized targets
    l1000_targets = {"HDAC1", "FGF2", "SERPINE1"}
    df["l1000_well_characterized"] = df["gene"].isin(l1000_targets)

    df = df.sort_values(
        ["n_msigdb_pathways", "direction"],
        ascending=[False, True],
    )

    return df


def derive_acharjee_stricture_signature(
    msigdb_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Derive and save the Acharjee stricture signature end-to-end."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Acharjee 8-Gene Stricture Signature")
    print("=" * 60)

    df = build_acharjee_stricture_signature(msigdb_dir)

    up_genes = df[df["direction"] == "up"]
    overlaps = df[df["overlaps_celltype_signature"]]
    l1000 = df[df["l1000_well_characterized"]]

    print(f"\n  Total genes: {len(df)}")
    print(f"  Upregulated (stricture-associated): {len(up_genes)}")
    print(f"  Overlap with cell-type signature: {len(overlaps)} ({', '.join(overlaps['gene'])})")
    print(f"  L1000 well-characterized: {len(l1000)} ({', '.join(l1000['gene'])})")

    print(f"\n  Gene details:")
    for _, row in df.iterrows():
        flags = []
        if row["overlaps_celltype_signature"]:
            flags.append("celltype-overlap")
        if row["l1000_well_characterized"]:
            flags.append("L1000")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"    {row['gene']:10s} {row['direction']:4s}{flag_str}")

    out_path = output_dir / "acharjee_stricture_signature.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive Acharjee 8-gene stricture signature"
    )
    parser.add_argument("--msigdb-dir", type=Path, default=MSIGDB_DEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    derive_acharjee_stricture_signature(args.msigdb_dir, args.output_dir)


if __name__ == "__main__":
    main()
