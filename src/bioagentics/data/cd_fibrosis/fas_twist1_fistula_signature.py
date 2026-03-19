"""Derive FAS/TWIST1 fistula fibroblast signature for CD.

FAS (fibroblast activation state) fibroblasts in fistula outer zone are
driven by TWIST1, induced by CXCL9+ macrophages via IL-1beta/TGF-beta.
FAP+TWIST1+ fibroblasts are the highest ECM-producing subtype in fibrotic
CD intestine (Nature fistula spatial atlas).

Separate from stricture signatures — different pathobiology (fistulizing
vs stricturing CD).

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.fas_twist1_fistula_signature
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

# ── Literature-curated FAS/TWIST1 fistula fibroblast gene sets ──

# Component 1: FAS fibroblast outer zone markers
# Source: Nature fistula spatial atlas — outer zone fibroblast activation program
FAS_OUTER_ZONE_MARKERS: dict[str, str] = {
    # Core FAS fibroblast markers from fistula outer zone
    "TWIST1": "up",      # master TF, induced by CXCL9+ macrophages via IL-1b/TGF-b
    "RUNX2": "up",       # osteogenic TF co-expressed with TWIST1 in fistula
    "OSR2": "up",        # odd-skipped related 2, mesenchymal progenitor marker
    "PRRX1": "up",       # paired related homeobox 1, EMT/fibroblast TF
    "FAP": "up",         # fibroblast activation protein, FAP+TWIST1+ = highest ECM
    "ACTA2": "up",       # alpha-SMA, myofibroblast activation
    "VIM": "up",         # vimentin, mesenchymal marker
    "THY1": "up",        # CD90, activated fibroblast marker
    "PDGFRA": "up",      # PDGF receptor alpha, fibroblast identity
    "COL1A1": "up",      # collagen type I, major ECM component
    "COL1A2": "up",      # collagen type I alpha 2
    "COL3A1": "up",      # collagen type III
    "FN1": "up",         # fibronectin
    "POSTN": "up",       # periostin, ECM stiffening
}

# Component 2: TWIST1 transcriptional targets in fibroblasts
# Source: TWIST1 target gene literature + fistula RNA-seq
TWIST1_TARGETS: dict[str, str] = {
    # TWIST1-regulated EMT/fibrosis genes
    "TWIST1": "up",      # auto-regulation
    "SNAI1": "up",       # TWIST1 cooperates with SNAIL for EMT
    "SNAI2": "up",       # SLUG, EMT TF
    "ZEB1": "up",        # zinc finger E-box binding homeobox 1
    "ZEB2": "up",        # zinc finger E-box binding homeobox 2
    "CDH2": "up",        # N-cadherin, mesenchymal marker
    "CDH11": "up",       # cadherin-11, fibroblast cadherin
    "MMP2": "up",        # matrix metalloproteinase 2
    "MMP14": "up",       # MT1-MMP, membrane-type MMP
    "SERPINE1": "up",    # PAI-1, TWIST1 target
    "CTGF": "up",        # CCN2, profibrotic growth factor
    "SPARC": "up",       # secreted protein, ECM remodeling
    "LOX": "up",         # lysyl oxidase, collagen crosslinker
    "LOXL2": "up",       # lysyl oxidase-like 2
    # E-cadherin suppression (lost in EMT)
    "CDH1": "down",      # E-cadherin, epithelial marker lost in fistula EMT
}

# Component 3: CXCL9+ macrophage-fibroblast interaction signals
# Source: Nature fistula atlas — macrophage-derived signals inducing TWIST1
MACROPHAGE_SIGNALS: dict[str, str] = {
    # CXCL9+ macrophage-derived inducers of TWIST1 in fibroblasts
    "CXCL9": "up",       # CXCL9+ macrophage marker
    "CXCL10": "up",      # co-expressed with CXCL9 in inflammatory macrophages
    "CXCL11": "up",      # CXCR3 ligand cluster
    "IL1B": "up",        # IL-1 beta, primary TWIST1 inducer
    "IL1A": "up",        # IL-1 alpha
    "TGFB1": "up",       # TGF-beta 1, TWIST1 co-inducer
    "TGFB2": "up",       # TGF-beta 2
    "TNF": "up",         # TNF-alpha, fistula inflammation
    "IL6": "up",         # IL-6, fibroblast activation
    "OSM": "up",         # oncostatin M, fibroblast activation in IBD
    # NF-kB signaling (macrophage-fibroblast crosstalk)
    "NFKB1": "up",       # NF-kB p50
    "RELA": "up",        # NF-kB p65
}

# Component 4: Fistula-specific ECM remodeling program
# Source: Nature fistula atlas — fistula tract ECM composition
FISTULA_ECM_PROGRAM: dict[str, str] = {
    # Fistula-enriched collagens
    "COL5A1": "up",      # collagen type V
    "COL6A1": "up",      # collagen type VI
    "COL6A3": "up",      # collagen type VI alpha 3
    "COL11A1": "up",     # collagen type XI
    "COL12A1": "up",     # collagen type XII
    # ECM glycoproteins in fistula tract
    "TNC": "up",         # tenascin-C, wound healing/fistula ECM
    "VCAN": "up",        # versican, proteoglycan
    "LUM": "up",         # lumican
    "BGN": "up",         # biglycan
    "DCN": "up",         # decorin
    # Matrix remodeling enzymes
    "MMP9": "up",        # gelatinase B, tissue remodeling
    "TIMP1": "up",       # tissue inhibitor of MMPs
    "TIMP3": "up",       # TIMP3, ECM-bound MMP inhibitor
    # Basement membrane disruption (fistula penetration)
    "LAMB1": "up",       # laminin beta 1
    "COL4A1": "up",      # collagen type IV, basement membrane
}


def build_fas_twist1_fistula_signature(msigdb_dir: Path | None = None) -> pd.DataFrame:
    """Build the FAS/TWIST1 fistula fibroblast signature.

    Merges literature-curated genes from all four components and annotates
    with MSigDB pathway membership.
    """
    msigdb_dir = msigdb_dir or MSIGDB_DEST

    all_genes: dict[str, dict] = {}

    component_sets = [
        ("FAS_OUTER_ZONE", FAS_OUTER_ZONE_MARKERS),
        ("TWIST1_TARGETS", TWIST1_TARGETS),
        ("MACROPHAGE_SIGNALS", MACROPHAGE_SIGNALS),
        ("FISTULA_ECM", FISTULA_ECM_PROGRAM),
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
            "fas_twist1_components": ";".join(gene_info["components"]),
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


def derive_fas_twist1_fistula_signature(
    msigdb_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Derive and save the FAS/TWIST1 fistula fibroblast signature end-to-end."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FAS/TWIST1 Fistula Fibroblast Signature")
    print("=" * 60)

    df = build_fas_twist1_fistula_signature(msigdb_dir)

    up_genes = df[df["direction"] == "up"]
    down_genes = df[df["direction"] == "down"]
    multi = df[df["n_components"] > 1]

    print(f"\n  Total genes: {len(df)}")
    print(f"  Upregulated (profibrotic): {len(up_genes)}")
    print(f"  Downregulated (protective): {len(down_genes)}")
    print(f"  Multi-component genes: {len(multi)}")

    for component_name, gene_dict in [
        ("FAS_OUTER_ZONE", FAS_OUTER_ZONE_MARKERS),
        ("TWIST1_TARGETS", TWIST1_TARGETS),
        ("MACROPHAGE_SIGNALS", MACROPHAGE_SIGNALS),
        ("FISTULA_ECM", FISTULA_ECM_PROGRAM),
    ]:
        print(f"  {component_name}: {len(gene_dict)} genes")

    print(f"\n  Multi-component convergence genes:")
    for _, row in multi.iterrows():
        print(f"    {row['gene']:12s} {row['direction']:4s}  [{row['fas_twist1_components']}]")

    out_path = output_dir / "fas_twist1_fistula_signature.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive FAS/TWIST1 fistula fibroblast signature"
    )
    parser.add_argument("--msigdb-dir", type=Path, default=MSIGDB_DEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    derive_fas_twist1_fistula_signature(args.msigdb_dir, args.output_dir)


if __name__ == "__main__":
    main()
