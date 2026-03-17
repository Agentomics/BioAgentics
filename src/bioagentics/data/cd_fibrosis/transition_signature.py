"""Derive inflammation-to-fibrosis transition gene signature for CD.

Captures genes marking the therapeutic window where intervention could prevent
irreversible fibrosis. Three complementary transition pathways:

1. CD38/PECAM1 axis (J Crohn's Colitis 2025): CD38 marks the endothelial-
   mesenchymal transition driving fibrotic remodeling; CD38 inhibition
   reduced fibrosis in mouse CD model.

2. YAP/TAZ mechanotransduction (Cell 2025): Creeping fat mechanosensitive
   fibroblasts activate YAP/TAZ-dependent profibrotic transcription in
   response to tissue stiffness.

3. TL1A-DR3/Rho pathway (tulisokibart RNA-seq): TL1A activates Rho signal
   transduction as the major pathway in fibroblasts; duvakitug/tulisokibart
   target this dual inflammation/fibrosis axis.

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.transition_signature
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

# ── Literature-curated transition gene sets ──
#
# Each dict maps gene_symbol -> direction ("up" = upregulated in fibrosis
# transition, "down" = protective/anti-fibrotic, lost during transition).

# Pathway 1: CD38/PECAM1 endothelial-mesenchymal transition axis
# Source: J Crohn's Colitis 2025 — CD38 inhibition reduced fibrosis
CD38_PECAM1_GENES: dict[str, str] = {
    # Core axis
    "CD38": "up",       # NAD+ consuming enzyme, marks EndMT
    "PECAM1": "up",     # CD31, endothelial marker in transition
    "CD31": "up",       # alias for PECAM1
    # EndMT / mesenchymal transition markers
    "SNAI1": "up",      # Snail — EMT transcription factor
    "SNAI2": "up",      # Slug — EMT transcription factor
    "TWIST1": "up",     # master TF for fibroblast activation
    "ZEB1": "up",       # EMT TF
    "ZEB2": "up",       # EMT TF
    "CDH2": "up",       # N-cadherin, mesenchymal marker
    "VIM": "up",        # vimentin, mesenchymal marker
    "CDH5": "down",     # VE-cadherin, endothelial marker (lost in EndMT)
    "TIE1": "down",     # endothelial marker (lost in EndMT)
    # CD38 downstream / NAD+ metabolism
    "NAMPT": "up",      # NAD+ biosynthesis, compensatory
    "SIRT1": "down",    # NAD+-dependent deacetylase (depleted by CD38)
    "PARP1": "up",      # NAD+ consumer, DNA damage response
    # Fibrotic effectors
    "COL1A1": "up",
    "COL1A2": "up",
    "COL3A1": "up",
    "FN1": "up",        # fibronectin
    "ACTA2": "up",      # alpha-SMA
    "TGFB1": "up",
}

# Pathway 2: YAP/TAZ mechanotransduction
# Source: Cell 2025 — creeping fat mechanosensitive fibroblasts
YAP_TAZ_GENES: dict[str, str] = {
    # Core Hippo pathway / YAP-TAZ
    "YAP1": "up",       # Yes-associated protein
    "WWTR1": "up",      # TAZ
    "TEAD1": "up",      # YAP/TAZ transcriptional partner
    "TEAD2": "up",
    "TEAD4": "up",
    "LATS1": "down",    # Hippo kinase (inhibits YAP — lost in fibrosis)
    "LATS2": "down",
    "MST1": "down",     # STK4, upstream Hippo kinase
    "SAV1": "down",     # Salvador, Hippo scaffold
    "MOB1A": "down",    # Hippo pathway component
    # YAP/TAZ target genes (profibrotic)
    "CTGF": "up",       # CCN2, major profibrotic YAP target
    "CYR61": "up",      # CCN1, YAP target
    "ANKRD1": "up",     # YAP target in fibroblasts
    "AMOTL2": "up",     # YAP target
    "AXL": "up",        # YAP target, receptor tyrosine kinase
    # Mechanosensing / stiffness response
    "ITGB1": "up",      # integrin beta-1, mechanosensor
    "ITGAV": "up",      # integrin alpha-V
    "ROCK1": "up",      # Rho kinase, mechanotransduction
    "ROCK2": "up",
    "MYL9": "up",       # myosin light chain, contractility
    "RHOA": "up",       # Rho GTPase
    "FAK": "up",        # PTK2, focal adhesion kinase
    "MRTFA": "up",      # MKL1, SRF co-activator
    "SRF": "up",        # serum response factor
    # ECM stiffness markers
    "LOX": "up",        # lysyl oxidase, collagen crosslinker
    "LOXL2": "up",      # lysyl oxidase-like 2
    "POSTN": "up",      # periostin, ECM stiffening
    "TNC": "up",        # tenascin-C, mechanosensitive
}

# Pathway 3: TL1A-DR3/Rho signaling
# Source: tulisokibart RNA-seq — Rho signal transduction as major
# TL1A-activated pathway in fibroblasts
TL1A_DR3_RHO_GENES: dict[str, str] = {
    # TL1A / DR3 ligand-receptor
    "TNFSF15": "up",    # TL1A
    "TNFRSF25": "up",   # DR3 (death receptor 3)
    # NF-kB downstream of DR3
    "NFKB1": "up",
    "RELA": "up",
    "NFKBIA": "up",     # IkBα, NF-kB inhibitor (transiently up)
    # Rho GTPase pathway (major TL1A-activated pathway in fibroblasts)
    "RHOA": "up",
    "RHOB": "up",
    "RHOC": "up",
    "RAC1": "up",
    "CDC42": "up",
    "ROCK1": "up",
    "ROCK2": "up",
    "LIMK1": "up",      # LIM kinase, Rho/ROCK target
    "CFL1": "down",     # cofilin, inactivated by LIMK (lost function)
    "MYH9": "up",       # non-muscle myosin, Rho effector
    "DIAPH1": "up",     # mDia1, Rho effector formin
    # Fibroblast activation / fibrosis effectors via TL1A
    "COL1A1": "up",
    "COL3A1": "up",
    "ACTA2": "up",
    "SERPINE1": "up",    # PAI-1
    "GREM1": "up",      # BMP antagonist
    "MMP2": "up",
    "MMP9": "up",
    "TIMP1": "up",      # tissue inhibitor of MMPs
    # Inflammatory cytokines in transition
    "IL6": "up",
    "IL1B": "up",
    "TNF": "up",
    "IL33": "up",       # alarmin, profibrotic
    "IL13": "up",       # type 2 cytokine driving fibrosis
    "IL17A": "up",      # Th17, drives fibroblast activation
    "CXCL9": "up",      # drives TWIST1+ fibroblast activation (JCI 2024)
}


def build_transition_signature(msigdb_dir: Path | None = None) -> pd.DataFrame:
    """Build the combined inflammation-to-fibrosis transition signature.

    Merges literature-curated genes from all three pathways and annotates
    with MSigDB pathway membership.
    """
    msigdb_dir = msigdb_dir or MSIGDB_DEST

    # Combine all pathway genes
    all_genes: dict[str, dict] = {}

    pathway_sets = [
        ("CD38_PECAM1", CD38_PECAM1_GENES),
        ("YAP_TAZ", YAP_TAZ_GENES),
        ("TL1A_DR3_RHO", TL1A_DR3_RHO_GENES),
    ]

    for pathway_name, gene_dict in pathway_sets:
        for gene, direction in gene_dict.items():
            gene_upper = gene.upper()
            if gene_upper in all_genes:
                # Gene in multiple pathways — add pathway, keep first direction
                all_genes[gene_upper]["pathways"].append(pathway_name)
            else:
                all_genes[gene_upper] = {
                    "gene": gene_upper,
                    "direction": direction,
                    "pathways": [pathway_name],
                }

    # Build DataFrame
    records = []
    for gene_info in all_genes.values():
        records.append({
            "gene": gene_info["gene"],
            "direction": gene_info["direction"],
            "transition_pathways": ";".join(gene_info["pathways"]),
            "n_transition_pathways": len(gene_info["pathways"]),
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

    # Sort: multi-pathway genes first, then by pathway count
    df = df.sort_values(
        ["n_transition_pathways", "direction", "n_msigdb_pathways"],
        ascending=[False, True, False],
    )

    return df


def derive_transition_signature(
    msigdb_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Derive and save the transition signature end-to-end."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Inflammation-to-Fibrosis Transition Signature")
    print("=" * 60)

    df = build_transition_signature(msigdb_dir)

    up_genes = df[df["direction"] == "up"]
    down_genes = df[df["direction"] == "down"]
    multi = df[df["n_transition_pathways"] > 1]

    print(f"\n  Total genes: {len(df)}")
    print(f"  Upregulated (profibrotic): {len(up_genes)}")
    print(f"  Downregulated (protective): {len(down_genes)}")
    print(f"  Multi-pathway genes: {len(multi)}")

    # Pathway breakdown
    for pathway_name, gene_dict in [
        ("CD38_PECAM1", CD38_PECAM1_GENES),
        ("YAP_TAZ", YAP_TAZ_GENES),
        ("TL1A_DR3_RHO", TL1A_DR3_RHO_GENES),
    ]:
        print(f"  {pathway_name}: {len(gene_dict)} genes")

    print(f"\n  Multi-pathway convergence genes:")
    for _, row in multi.iterrows():
        print(f"    {row['gene']:12s} {row['direction']:4s}  [{row['transition_pathways']}]")

    # Save
    out_path = output_dir / "transition_signature.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive inflammation-to-fibrosis transition signature"
    )
    parser.add_argument("--msigdb-dir", type=Path, default=MSIGDB_DEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    derive_transition_signature(args.msigdb_dir, args.output_dir)


if __name__ == "__main__":
    main()
