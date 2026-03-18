"""Derive dedicated TL1A-DR3/Rho pathway signature for CMAP/L1000 queries.

Expands TL1A-DR3 coverage beyond the partial inclusion in the transition
signature (#487). Sources: tulisokibart fibroblast RNA-seq identified Rho
signal transduction as the major TL1A-activated pathway (journal #823, #950).

Clinical validation: duvakitug (55% endoscopic response at 900mg, 44 weeks),
tulisokibart (ARES-CD Phase 3), ontunisertib (STENOVA Phase 2a met endpoints).

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.tl1a_dr3_rho_signature
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

# ── Literature-curated TL1A-DR3/Rho pathway gene sets ──

# Component 1: TL1A-DR3 ligand-receptor axis and immediate signaling
# Source: duvakitug/tulisokibart mechanism, APOLLO-CD RNA-seq
TL1A_DR3_SIGNALING: dict[str, str] = {
    # TL1A / DR3 ligand-receptor pair
    "TNFSF15": "up",     # TL1A
    "TNFRSF25": "up",    # DR3 (death receptor 3)
    "TNFRSF6B": "up",    # DcR3, decoy receptor (modulates TL1A)
    # NF-kB downstream of DR3
    "NFKB1": "up",       # NF-kB p50
    "NFKB2": "up",       # NF-kB p52
    "RELA": "up",        # NF-kB p65
    "RELB": "up",        # NF-kB RelB
    "NFKBIA": "up",      # IkBα (transiently upregulated)
    "IKBKB": "up",       # IKKβ, NF-kB activating kinase
    # TRAF signaling (DR3 downstream)
    "TRAF2": "up",       # TNF receptor-associated factor 2
    "TRAF5": "up",       # TNF receptor-associated factor 5
    "RIPK1": "up",       # receptor-interacting protein kinase 1
    # MAPK branch
    "MAP3K7": "up",      # TAK1, TL1A-activated kinase
    "MAPK8": "up",       # JNK1
    "MAPK14": "up",      # p38 MAPK
}

# Component 2: Rho GTPase signaling cascade (major TL1A-activated pathway)
# Source: tulisokibart fibroblast RNA-seq (PMID 40456235)
RHO_GTPASE_CASCADE: dict[str, str] = {
    # Rho family GTPases
    "RHOA": "up",        # RhoA, primary Rho GTPase
    "RHOB": "up",        # RhoB
    "RHOC": "up",        # RhoC
    "RAC1": "up",        # Rac1
    "RAC2": "up",        # Rac2
    "CDC42": "up",       # Cdc42
    # Rho GEFs (activators)
    "ARHGEF2": "up",     # GEF-H1, RhoA activator
    "ECT2": "up",        # RhoGEF
    "NET1": "up",        # neuroepithelial cell transforming gene 1
    # Rho effector kinases
    "ROCK1": "up",       # Rho kinase 1
    "ROCK2": "up",       # Rho kinase 2
    "LIMK1": "up",       # LIM kinase 1, ROCK target
    "LIMK2": "up",       # LIM kinase 2
    # Downstream cytoskeletal effectors
    "CFL1": "down",      # cofilin, inactivated by LIMK (function lost)
    "MYH9": "up",        # non-muscle myosin heavy chain IIA
    "MYH10": "up",       # non-muscle myosin heavy chain IIB
    "MYL9": "up",        # myosin light chain
    "DIAPH1": "up",      # mDia1, Rho effector formin
    "DIAPH3": "up",      # mDia2
    # Focal adhesion / stress fibers
    "PTK2": "up",        # FAK, focal adhesion kinase
    "PXN": "up",         # paxillin
    "VCL": "up",         # vinculin
    # Shared with effectors (Rho/ROCK-driven ECM/contractility)
    "ACTA2": "up",       # alpha-SMA, Rho/ROCK-dependent contractility
    "COL1A1": "up",      # Rho/ROCK promotes collagen deposition
    "FN1": "up",         # Rho-regulated fibronectin assembly
    "SERPINE1": "up",    # PAI-1, Rho-regulated
}

# Component 3: TL1A-induced fibroblast activation and fibrosis effectors
# Source: tulisokibart RNA-seq + TL1A fibroblast stimulation studies
TL1A_FIBROBLAST_EFFECTORS: dict[str, str] = {
    # ECM deposition
    "COL1A1": "up",      # collagen type I
    "COL1A2": "up",
    "COL3A1": "up",      # collagen type III
    "COL4A1": "up",      # collagen type IV
    "FN1": "up",         # fibronectin
    "ACTA2": "up",       # alpha-SMA
    # Profibrotic mediators
    "SERPINE1": "up",    # PAI-1
    "GREM1": "up",       # BMP antagonist
    "CTGF": "up",        # CCN2
    "POSTN": "up",       # periostin
    "LOX": "up",         # collagen crosslinker
    "TIMP1": "up",       # tissue inhibitor of MMPs
    "MMP2": "up",        # matrix metalloproteinase 2
    "MMP9": "up",        # matrix metalloproteinase 9
    "MMP14": "up",       # MT1-MMP
    # TL1A-induced cytokines (fibroblast-derived)
    "IL6": "up",         # interleukin-6
    "IL1B": "up",        # interleukin-1 beta
    "IL33": "up",        # alarmin, profibrotic
    "IL13": "up",        # type 2 cytokine
    "IL17A": "up",       # Th17, fibroblast activator
    "CXCL9": "up",       # drives TWIST1+ fibroblast activation (JCI 2024)
    "CCL2": "up",        # monocyte chemoattractant
    "TNF": "up",         # TNF-alpha
    # Fibroblast activation markers
    "FAP": "up",         # fibroblast activation protein
    "TWIST1": "up",      # master TF for fibroblast activation
}

# Component 4: ALK5/TGF-beta receptor axis (ontunisertib target pathway)
# Source: ontunisertib STENOVA Phase 2a transcriptomic data
ALK5_TGFB_AXIS: dict[str, str] = {
    "TGFBR1": "up",      # ALK5, ontunisertib target
    "TGFBR2": "up",      # TGF-beta receptor II
    "TGFB1": "up",       # TGF-beta 1
    "SMAD2": "up",       # SMAD2
    "SMAD3": "up",       # SMAD3
    "SMAD4": "up",       # SMAD4 common mediator
    "SMAD7": "down",     # inhibitory SMAD (negative feedback)
    "SNAI1": "up",       # Snail, EMT TF
    "SNAI2": "up",       # Slug, EMT TF
    "CDH2": "up",        # N-cadherin, mesenchymal marker
    # Shared with effectors (TGF-beta-induced ECM)
    "COL1A1": "up",      # TGF-beta direct target
    "COL3A1": "up",      # TGF-beta direct target
    "ACTA2": "up",       # TGF-beta-induced myofibroblast marker
    "FN1": "up",         # TGF-beta-induced ECM
    "CTGF": "up",        # CCN2, TGF-beta/SMAD target
}


def build_tl1a_dr3_rho_signature(msigdb_dir: Path | None = None) -> pd.DataFrame:
    """Build the dedicated TL1A-DR3/Rho pathway signature.

    Merges literature-curated genes from all four components and annotates
    with MSigDB pathway membership.
    """
    msigdb_dir = msigdb_dir or MSIGDB_DEST

    all_genes: dict[str, dict] = {}

    component_sets = [
        ("TL1A_DR3_SIGNALING", TL1A_DR3_SIGNALING),
        ("RHO_GTPASE_CASCADE", RHO_GTPASE_CASCADE),
        ("TL1A_FIBROBLAST_EFFECTORS", TL1A_FIBROBLAST_EFFECTORS),
        ("ALK5_TGFB_AXIS", ALK5_TGFB_AXIS),
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
            "tl1a_dr3_components": ";".join(gene_info["components"]),
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


def derive_tl1a_dr3_rho_signature(
    msigdb_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Derive and save the TL1A-DR3/Rho pathway signature end-to-end."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TL1A-DR3/Rho Pathway Fibrosis Signature")
    print("=" * 60)

    df = build_tl1a_dr3_rho_signature(msigdb_dir)

    up_genes = df[df["direction"] == "up"]
    down_genes = df[df["direction"] == "down"]
    multi = df[df["n_components"] > 1]

    print(f"\n  Total genes: {len(df)}")
    print(f"  Upregulated (profibrotic): {len(up_genes)}")
    print(f"  Downregulated (protective): {len(down_genes)}")
    print(f"  Multi-component genes: {len(multi)}")

    for component_name, gene_dict in [
        ("TL1A_DR3_SIGNALING", TL1A_DR3_SIGNALING),
        ("RHO_GTPASE_CASCADE", RHO_GTPASE_CASCADE),
        ("TL1A_FIBROBLAST_EFFECTORS", TL1A_FIBROBLAST_EFFECTORS),
        ("ALK5_TGFB_AXIS", ALK5_TGFB_AXIS),
    ]:
        print(f"  {component_name}: {len(gene_dict)} genes")

    print(f"\n  Multi-component convergence genes:")
    for _, row in multi.iterrows():
        print(f"    {row['gene']:12s} {row['direction']:4s}  [{row['tl1a_dr3_components']}]")

    out_path = output_dir / "tl1a_dr3_rho_signature.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive TL1A-DR3/Rho pathway fibrosis signature"
    )
    parser.add_argument("--msigdb-dir", type=Path, default=MSIGDB_DEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    derive_tl1a_dr3_rho_signature(args.msigdb_dir, args.output_dir)


if __name__ == "__main__":
    main()
