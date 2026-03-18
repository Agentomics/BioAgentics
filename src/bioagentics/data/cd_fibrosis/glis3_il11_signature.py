"""Derive GLIS3/IL-11 axis fibrosis gene signature for CD.

Novel upstream regulatory axis: GLIS3 is master TF controlling the
macrophage-fibroblast-IL-11 fibrotic cascade. GLIS3 KO abolished fibrosis
in organoids (Nature 2026, Xavier/Broad, DOI 10.1038/s41586-025-09907-x).

Orthogonal to existing signatures (bulk, cell-type, transition). IL-11 is
druggable — anti-IL-11 biologics are in development.

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.glis3_il11_signature
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

# ── Literature-curated GLIS3/IL-11 axis gene sets ──
#
# Each dict maps gene_symbol -> direction ("up" = upregulated in fibrosis,
# "down" = protective/anti-fibrotic).

# Component 1: GLIS3 transcriptional targets
# Source: Nature 2026, Xavier/Broad — GLIS3 CRISPR screen + scRNA-seq
GLIS3_TF_TARGETS: dict[str, str] = {
    # Master regulator
    "GLIS3": "up",       # master TF controlling fibroblast fibrotic response
    # Direct transcriptional targets of GLIS3 in fibroblasts
    "IL11": "up",        # key effector cytokine, druggable target
    "COL1A1": "up",      # collagen type I alpha 1
    "COL3A1": "up",      # collagen type III alpha 1
    "COL5A1": "up",      # collagen type V alpha 1
    "FN1": "up",         # fibronectin
    "ACTA2": "up",       # alpha-SMA, myofibroblast marker
    "TIMP1": "up",       # tissue inhibitor of MMPs
    "CTGF": "up",        # CCN2, profibrotic growth factor
    "POSTN": "up",       # periostin, ECM stiffening
    "SPARC": "up",       # secreted protein acidic and cysteine rich
    "LOXL2": "up",       # collagen crosslinker
    "SERPINE1": "up",    # PAI-1, druggable serine protease inhibitor
}

# Component 2: IL-11 signaling pathway
# Source: IL-11 as central fibrosis mediator (Nature 2026 + prior IL-11 literature)
IL11_SIGNALING: dict[str, str] = {
    # IL-11 ligand-receptor
    "IL11": "up",        # interleukin-11
    "IL11RA": "up",      # IL-11 receptor alpha
    "IL6ST": "up",       # gp130, shared signal transducer
    # JAK-STAT3 downstream effectors
    "JAK1": "up",        # Janus kinase 1
    "JAK2": "up",        # Janus kinase 2
    "STAT3": "up",       # signal transducer and activator of transcription 3
    "SOCS3": "up",       # suppressor of cytokine signaling 3 (feedback)
    # ERK/MAPK branch (IL-11 non-canonical signaling)
    "ERK1": "up",        # MAPK3
    "ERK2": "up",        # MAPK1
    "MEK1": "up",        # MAP2K1
    # IL-11-induced profibrotic effectors
    "MMP2": "up",        # matrix metalloproteinase 2
    "SNAI1": "up",       # EMT transcription factor
    "TWIST1": "up",      # fibroblast activation TF
}

# Component 3: TGF-beta/IL-1 upstream activators of GLIS3
# Source: Nature 2026 — macrophage signals that activate GLIS3 in fibroblasts
UPSTREAM_ACTIVATORS: dict[str, str] = {
    # Macrophage-derived signals
    "TGFB1": "up",       # TGF-beta 1, primary GLIS3 inducer
    "TGFB2": "up",       # TGF-beta 2
    "IL1B": "up",        # IL-1 beta, co-activator of GLIS3
    "IL1A": "up",        # IL-1 alpha
    "IL1R1": "up",       # IL-1 receptor type I
    # TGF-beta receptor signaling
    "TGFBR1": "up",      # TGF-beta receptor I (ALK5)
    "TGFBR2": "up",      # TGF-beta receptor II
    "SMAD2": "up",       # SMAD2, TGF-beta intracellular signal
    "SMAD3": "up",       # SMAD3
    "SMAD4": "up",       # SMAD4, common mediator
    "SMAD7": "down",     # inhibitory SMAD (negative feedback, lost in fibrosis)
    # Inflammation-associated fibroblast activation
    "FAP": "up",         # fibroblast activation protein
    "PDGFRA": "up",      # PDGF receptor alpha, fibroblast marker
}


def build_glis3_il11_signature(msigdb_dir: Path | None = None) -> pd.DataFrame:
    """Build the GLIS3/IL-11 axis fibrosis signature.

    Merges literature-curated genes from all three components and annotates
    with MSigDB pathway membership.
    """
    msigdb_dir = msigdb_dir or MSIGDB_DEST

    # Combine all component genes
    all_genes: dict[str, dict] = {}

    component_sets = [
        ("GLIS3_TF_TARGETS", GLIS3_TF_TARGETS),
        ("IL11_SIGNALING", IL11_SIGNALING),
        ("UPSTREAM_ACTIVATORS", UPSTREAM_ACTIVATORS),
    ]

    for component_name, gene_dict in component_sets:
        for gene, direction in gene_dict.items():
            gene_upper = gene.upper()
            if gene_upper in all_genes:
                all_genes[gene_upper]["components"].append(component_name)
            else:
                all_genes[gene_upper] = {
                    "gene": gene_upper,
                    "direction": direction,
                    "components": [component_name],
                }

    # Build DataFrame
    records = []
    for gene_info in all_genes.values():
        records.append({
            "gene": gene_info["gene"],
            "direction": gene_info["direction"],
            "glis3_il11_components": ";".join(gene_info["components"]),
            "n_components": len(gene_info["components"]),
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

    # Sort: multi-component genes first, then by pathway count
    df = df.sort_values(
        ["n_components", "direction", "n_msigdb_pathways"],
        ascending=[False, True, False],
    )

    return df


def derive_glis3_il11_signature(
    msigdb_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Derive and save the GLIS3/IL-11 axis signature end-to-end."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GLIS3/IL-11 Axis Fibrosis Signature")
    print("=" * 60)

    df = build_glis3_il11_signature(msigdb_dir)

    up_genes = df[df["direction"] == "up"]
    down_genes = df[df["direction"] == "down"]
    multi = df[df["n_components"] > 1]

    print(f"\n  Total genes: {len(df)}")
    print(f"  Upregulated (profibrotic): {len(up_genes)}")
    print(f"  Downregulated (protective): {len(down_genes)}")
    print(f"  Multi-component genes: {len(multi)}")

    # Component breakdown
    for component_name, gene_dict in [
        ("GLIS3_TF_TARGETS", GLIS3_TF_TARGETS),
        ("IL11_SIGNALING", IL11_SIGNALING),
        ("UPSTREAM_ACTIVATORS", UPSTREAM_ACTIVATORS),
    ]:
        print(f"  {component_name}: {len(gene_dict)} genes")

    print(f"\n  Multi-component convergence genes:")
    for _, row in multi.iterrows():
        print(f"    {row['gene']:12s} {row['direction']:4s}  [{row['glis3_il11_components']}]")

    # Save
    out_path = output_dir / "glis3_il11_signature.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive GLIS3/IL-11 axis fibrosis signature"
    )
    parser.add_argument("--msigdb-dir", type=Path, default=MSIGDB_DEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    derive_glis3_il11_signature(args.msigdb_dir, args.output_dir)


if __name__ == "__main__":
    main()
