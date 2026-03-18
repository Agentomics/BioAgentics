"""Derive CTHRC1+/YAP-TAZ mechanosensitive fibroblast signature for CD.

Creeping fat-derived CTHRC1+ fibroblasts drive stricture formation via
YAP/TAZ mechanotransduction (Cell 2025, Stanford, PMID 40967215).
YAP/TAZ genetic deletion reduced fibrosis in mice.

Expands beyond partial coverage in existing cell-type (#486) and transition
(#487) signatures with dedicated CTHRC1+ fibroblast-specific genes,
YAP/TAZ transcriptional targets, and creeping fat spatial module genes.

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.cthrc1_yaptaz_signature
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

# ── Literature-curated CTHRC1+/YAP-TAZ gene sets ──

# Component 1: CTHRC1+ fibroblast-specific genes
# Source: Cell 2025 scRNA-seq atlas (101,189 fibroblasts, 7 patients)
CTHRC1_FIBROBLAST_GENES: dict[str, str] = {
    # Core CTHRC1+ fibroblast markers
    "CTHRC1": "up",      # collagen triple helix repeat containing 1
    "POSTN": "up",       # periostin, ECM stiffening (protein-validated JCI Insight 2026)
    "TNC": "up",         # tenascin-C, mechanosensitive (protein-validated)
    "COL1A1": "up",      # collagen type I
    "COL1A2": "up",      # collagen type I alpha 2
    "COL3A1": "up",      # collagen type III
    "COL5A1": "up",      # collagen type V
    "COL5A2": "up",      # collagen type V alpha 2
    "COL6A1": "up",      # collagen type VI
    "COL6A3": "up",      # collagen type VI alpha 3
    "COL11A1": "up",     # collagen type XI alpha 1
    "COL12A1": "up",     # collagen type XII
    "FN1": "up",         # fibronectin
    "SPARC": "up",       # secreted protein acidic and cysteine rich
    "VCAN": "up",        # versican
    "LUM": "up",         # lumican, small leucine-rich proteoglycan
    "DCN": "up",         # decorin
    "BGN": "up",         # biglycan
    "ACTA2": "up",       # alpha-SMA, myofibroblast activation
    "TAGLN": "up",       # transgelin, smooth muscle marker
    "FAP": "up",         # fibroblast activation protein
    "THY1": "up",        # CD90, activated fibroblast marker
    "PDGFRA": "up",      # PDGF receptor alpha
}

# Component 2: YAP/TAZ transcriptional target genes
# Source: Cell 2025 + established YAP/TAZ target literature
YAPTAZ_TARGET_GENES: dict[str, str] = {
    # Core Hippo / YAP-TAZ
    "YAP1": "up",        # Yes-associated protein
    "WWTR1": "up",       # TAZ
    "TEAD1": "up",       # YAP/TAZ transcriptional partner
    "TEAD2": "up",
    "TEAD4": "up",
    # Hippo pathway suppressors (lost in fibrosis)
    "LATS1": "down",     # Hippo kinase, inhibits YAP
    "LATS2": "down",
    "MST1": "down",      # STK4, upstream Hippo kinase
    "SAV1": "down",      # Salvador, Hippo scaffold
    "MOB1A": "down",     # Hippo pathway component
    # YAP/TAZ direct transcriptional targets
    "CTGF": "up",        # CCN2, major profibrotic YAP target
    "CYR61": "up",       # CCN1, YAP target
    "ANKRD1": "up",      # YAP target in fibroblasts
    "AMOTL2": "up",      # angiomotin-like 2
    "AXL": "up",         # receptor tyrosine kinase
    "BIRC5": "up",       # survivin, YAP target
    "FOXF1": "up",       # forkhead box F1, YAP target in mesenchyme
    "IGFBP3": "up",      # insulin-like growth factor binding protein 3
    "SERPINE1": "up",    # PAI-1, YAP target (overlap with Acharjee stricture sig)
    # Shared with CTHRC1+ fibroblast markers (YAP-regulated ECM)
    "POSTN": "up",       # periostin, direct YAP target
    "CTGF": "up",        # CCN2, canonical YAP target (also in fibroblast markers)
    "COL1A1": "up",      # YAP/TAZ promotes collagen deposition
    "FN1": "up",         # fibronectin, YAP-regulated
    "ACTA2": "up",       # alpha-SMA, YAP-driven contractility
}

# Component 3: Mechanosensitive markers / stiffness response
# Source: Cell 2025 — creeping fat mechanotransduction
MECHANOSENSING_GENES: dict[str, str] = {
    # Integrin / mechanosensing
    "ITGB1": "up",       # integrin beta-1, primary mechanosensor
    "ITGAV": "up",       # integrin alpha-V
    "ITGA5": "up",       # integrin alpha-5, fibronectin receptor
    "ITGB5": "up",       # integrin beta-5
    # Rho/ROCK mechanotransduction
    "RHOA": "up",        # Rho GTPase
    "ROCK1": "up",       # Rho kinase 1
    "ROCK2": "up",       # Rho kinase 2
    "MYL9": "up",        # myosin light chain, contractility
    "MYH9": "up",        # non-muscle myosin heavy chain
    "PTK2": "up",        # FAK, focal adhesion kinase
    "MRTFA": "up",       # MKL1, SRF co-activator
    "SRF": "up",         # serum response factor
    # ECM crosslinking / stiffness amplification
    "LOX": "up",         # lysyl oxidase, collagen crosslinker
    "LOXL2": "up",       # lysyl oxidase-like 2
    "LOXL1": "up",       # lysyl oxidase-like 1
    # Verteporfin target — YAP/TAZ inhibitor candidate
    "TEAD3": "up",       # TEAD family member (verteporfin disrupts YAP-TEAD)
    # Shared with fibroblast/YAP targets (mechano-regulated ECM)
    "POSTN": "up",       # mechanosensitive ECM protein
    "TNC": "up",         # tenascin-C, mechanosensitive
    "ACTA2": "up",       # mechano-regulated contractility
    "FN1": "up",         # fibronectin, mechano-regulated assembly
    "COL1A1": "up",      # collagen, mechano-regulated deposition
}

# Component 4: Creeping fat spatial module genes
# Source: Cell 2025 Visium/Xenium spatial transcriptomics
CREEPING_FAT_SPATIAL: dict[str, str] = {
    # Adipose-fibroblast interaction markers
    "PPARG": "down",     # PPARgamma, adipocyte marker (lost in fibrotic fat)
    "ADIPOQ": "down",    # adiponectin (lost in creeping fat)
    "LEP": "up",         # leptin, profibrotic in creeping fat
    "PDGFB": "up",       # PDGF-B, fibroblast recruitment
    "PDGFRB": "up",      # PDGF receptor beta
    # Creeping fat matrix remodeling
    "MMP2": "up",        # matrix metalloproteinase 2
    "MMP14": "up",       # MT1-MMP, membrane-type MMP
    "TIMP1": "up",       # tissue inhibitor of MMP
    "TGFB1": "up",       # TGF-beta 1, creeping fat-derived
    # Shared (ECM deposited in creeping fat)
    "COL1A1": "up",      # collagen in fibrotic creeping fat
    "FN1": "up",         # fibronectin in creeping fat
    "LOX": "up",         # crosslinker in creeping fat stiffening
    # Perivascular / mesenchymal markers in creeping fat
    "MCAM": "up",        # CD146, perivascular marker
    "NOTCH3": "up",      # Notch signaling in pericyte-fibroblast transition
}


def build_cthrc1_yaptaz_signature(msigdb_dir: Path | None = None) -> pd.DataFrame:
    """Build the CTHRC1+/YAP-TAZ mechanosensitive fibroblast signature.

    Merges literature-curated genes from all four components and annotates
    with MSigDB pathway membership.
    """
    msigdb_dir = msigdb_dir or MSIGDB_DEST

    all_genes: dict[str, dict] = {}

    component_sets = [
        ("CTHRC1_FIBROBLAST", CTHRC1_FIBROBLAST_GENES),
        ("YAPTAZ_TARGETS", YAPTAZ_TARGET_GENES),
        ("MECHANOSENSING", MECHANOSENSING_GENES),
        ("CREEPING_FAT_SPATIAL", CREEPING_FAT_SPATIAL),
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

    records = []
    for gene_info in all_genes.values():
        records.append({
            "gene": gene_info["gene"],
            "direction": gene_info["direction"],
            "cthrc1_yaptaz_components": ";".join(gene_info["components"]),
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

    df = df.sort_values(
        ["n_components", "direction", "n_msigdb_pathways"],
        ascending=[False, True, False],
    )

    return df


def derive_cthrc1_yaptaz_signature(
    msigdb_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Derive and save the CTHRC1+/YAP-TAZ signature end-to-end."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CTHRC1+/YAP-TAZ Mechanosensitive Fibroblast Signature")
    print("=" * 60)

    df = build_cthrc1_yaptaz_signature(msigdb_dir)

    up_genes = df[df["direction"] == "up"]
    down_genes = df[df["direction"] == "down"]
    multi = df[df["n_components"] > 1]

    print(f"\n  Total genes: {len(df)}")
    print(f"  Upregulated (profibrotic): {len(up_genes)}")
    print(f"  Downregulated (protective): {len(down_genes)}")
    print(f"  Multi-component genes: {len(multi)}")

    for component_name, gene_dict in [
        ("CTHRC1_FIBROBLAST", CTHRC1_FIBROBLAST_GENES),
        ("YAPTAZ_TARGETS", YAPTAZ_TARGET_GENES),
        ("MECHANOSENSING", MECHANOSENSING_GENES),
        ("CREEPING_FAT_SPATIAL", CREEPING_FAT_SPATIAL),
    ]:
        print(f"  {component_name}: {len(gene_dict)} genes")

    print(f"\n  Multi-component convergence genes:")
    for _, row in multi.iterrows():
        print(f"    {row['gene']:12s} {row['direction']:4s}  [{row['cthrc1_yaptaz_components']}]")

    out_path = output_dir / "cthrc1_yaptaz_signature.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive CTHRC1+/YAP-TAZ mechanosensitive fibroblast signature"
    )
    parser.add_argument("--msigdb-dir", type=Path, default=MSIGDB_DEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    derive_cthrc1_yaptaz_signature(args.msigdb_dir, args.output_dir)


if __name__ == "__main__":
    main()
