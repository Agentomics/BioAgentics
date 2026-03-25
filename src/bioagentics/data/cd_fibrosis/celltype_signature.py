"""Derive cell-type-resolved fibroblast fibrosis gene signature for CD.

Extracts fibroblast-specific gene programs from published scRNA-seq and
spatial transcriptomics studies, avoiding dilution by non-fibroblast cell
types that limits bulk tissue signatures.

Sources:
- Kong et al. (Nat Genet 2025, SCP2959): inflammatory fibroblast spatial
  module genes from CD intestinal tissue
- GSE275144: CTHRC1+ fibroblast-specific markers from CD stricture tissue
- IBD Journal 2025 (doi:10.1093/ibd/izae295): druggable targets from
  fibrostenotic transcriptomic analysis (HDAC1, GREM1, SERPINE1, etc.)
- JCI 2024 (doi:10.1172/JCI179472): TWIST1+FAP+ fibroblasts as highest
  ECM-producing subtype, driven by CXCL9+ macrophage IL-1β/TGF-β
- JCI Insight 2026: protein-validated fibrosis markers (CTHRC1, POSTN,
  TNC, CPA3)

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.celltype_signature
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.cd_fibrosis.msigdb import (
    DEFAULT_DEST as MSIGDB_DEST,
    filter_fibrosis_sets,
    parse_gmt,
)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing" / "signatures"
BULK_SIG_PATH = OUTPUT_DIR / "bulk_tissue_signature.tsv"

# ── Fibroblast subtype gene programs ──
#
# Each gene entry: (direction, evidence_level)
# direction: "up" in fibrotic fibroblasts vs normal/inflammatory
# evidence: "protein_validated", "scrna_de", "spatial_module",
#           "druggable_target", "pathway_gene"

# CTHRC1+ pathogenic fibroblasts (GSE275144, JCI Insight 2026)
# These are the key profibrotic fibroblast population in CD strictures
CTHRC1_FIBROBLAST_GENES: dict[str, tuple[str, str]] = {
    "CTHRC1": ("up", "protein_validated"),  # Defining marker, collagen triple helix
    "POSTN": ("up", "protein_validated"),    # Periostin, ECM
    "TNC": ("up", "protein_validated"),      # Tenascin-C
    "CPA3": ("up", "protein_validated"),     # Carboxypeptidase A3
    "COL1A1": ("up", "scrna_de"),
    "COL1A2": ("up", "scrna_de"),
    "COL3A1": ("up", "scrna_de"),
    "COL5A1": ("up", "scrna_de"),
    "COL5A2": ("up", "scrna_de"),
    "COL6A1": ("up", "scrna_de"),
    "COL6A2": ("up", "scrna_de"),
    "COL6A3": ("up", "scrna_de"),
    "COL11A1": ("up", "scrna_de"),
    "COL12A1": ("up", "scrna_de"),
    "FN1": ("up", "scrna_de"),              # Fibronectin
    "SPARC": ("up", "scrna_de"),            # Secreted ECM glycoprotein
    "BGN": ("up", "scrna_de"),              # Biglycan, ECM proteoglycan
    "DCN": ("up", "scrna_de"),              # Decorin, ECM proteoglycan
    "LUM": ("up", "scrna_de"),              # Lumican
    "VCAN": ("up", "scrna_de"),             # Versican
    "THBS1": ("up", "scrna_de"),            # Thrombospondin-1, TGFβ activator
    "THBS2": ("up", "scrna_de"),            # Thrombospondin-2
    "COMP": ("up", "scrna_de"),             # Cartilage oligomeric matrix protein
    "AEBP1": ("up", "scrna_de"),            # Adipocyte enhancer-binding protein
    "INHBA": ("up", "scrna_de"),            # Inhibin beta A, TGFβ family
    "TAGLN": ("up", "scrna_de"),            # Transgelin, smooth muscle marker
    "MYH11": ("up", "scrna_de"),            # Myosin heavy chain 11
}

# TWIST1+FAP+ fibroblasts (JCI 2024, doi:10.1172/JCI179472)
# Highest ECM producers in fibrotic CD; driven by CXCL9+ macrophages
TWIST1_FAP_GENES: dict[str, tuple[str, str]] = {
    "TWIST1": ("up", "druggable_target"),   # Master TF, ameliorated fibrosis when inhibited
    "FAP": ("up", "druggable_target"),       # Fibroblast activation protein
    "ACTA2": ("up", "scrna_de"),            # Alpha-SMA
    "PDGFRA": ("up", "scrna_de"),           # PDGF receptor alpha, fibroblast marker
    "PDGFRB": ("up", "scrna_de"),           # PDGF receptor beta
    "VIM": ("up", "scrna_de"),              # Vimentin
    "S100A4": ("up", "scrna_de"),           # FSP1, fibroblast-specific protein
    "THY1": ("up", "scrna_de"),             # CD90, fibroblast marker
    "LRRC15": ("up", "scrna_de"),           # Cancer-associated fibroblast marker
    "MMP2": ("up", "scrna_de"),             # Matrix metalloproteinase 2
    "MMP11": ("up", "scrna_de"),            # Stromelysin-3
    "MMP14": ("up", "scrna_de"),            # MT1-MMP
    "ADAM12": ("up", "scrna_de"),            # Activated fibroblast marker
    "TGFBI": ("up", "scrna_de"),            # TGFβ-induced protein
    "LTBP2": ("up", "scrna_de"),            # Latent TGFβ binding protein
    "CTGF": ("up", "scrna_de"),             # CCN2, profibrotic
    "NOX4": ("up", "scrna_de"),             # NADPH oxidase 4, TGFβ target
    "SERPINE1": ("up", "druggable_target"), # PAI-1
    "LOX": ("up", "scrna_de"),              # Lysyl oxidase
    "LOXL2": ("up", "scrna_de"),            # Lysyl oxidase-like 2
}

# Kong et al. inflammatory fibroblast spatial module (Nat Genet 2025, SCP2959)
# Spatially defined inflammatory fibroblast programs in CD intestine
KONG_SPATIAL_GENES: dict[str, tuple[str, str]] = {
    "IL11": ("up", "spatial_module"),        # IL-11, profibrotic cytokine
    "IL6": ("up", "spatial_module"),         # IL-6, inflammatory/profibrotic
    "IL24": ("up", "spatial_module"),
    "IL33": ("up", "spatial_module"),        # Alarmin, profibrotic
    "CHI3L1": ("up", "spatial_module"),      # YKL-40, chitinase
    "CCL2": ("up", "spatial_module"),        # MCP-1, monocyte recruitment
    "CCL7": ("up", "spatial_module"),
    "CXCL1": ("up", "spatial_module"),
    "CXCL2": ("up", "spatial_module"),
    "CXCL3": ("up", "spatial_module"),
    "CXCL8": ("up", "spatial_module"),       # IL-8
    "CXCL9": ("up", "spatial_module"),       # Drives TWIST1+ fibroblast activation
    "C3": ("up", "spatial_module"),          # Complement C3
    "SFRP2": ("up", "spatial_module"),       # Secreted frizzled-related protein 2
    "SFRP4": ("up", "spatial_module"),
    "WNT2": ("up", "spatial_module"),        # Wnt signaling
    "WNT5A": ("up", "spatial_module"),
    "PTGS2": ("up", "spatial_module"),       # COX-2
    "HAS2": ("up", "spatial_module"),        # Hyaluronan synthase 2
    "TNFAIP6": ("up", "spatial_module"),     # TSG-6
    "PTGIS": ("up", "spatial_module"),       # Prostacyclin synthase
    "SOD2": ("up", "spatial_module"),        # Superoxide dismutase 2
    "ICAM1": ("up", "spatial_module"),
}

# PI16+ universal fibroblasts (Buechler et al., Nature 2021)
# Quiescent/progenitor fibroblast subtype that can transition to activated
# states. AREG (amphiregulin) drives EGFR-mediated fibroblast activation.
PI16_FIBROBLAST_GENES: dict[str, tuple[str, str]] = {
    "PI16": ("up", "scrna_de"),       # Peptidase inhibitor 16, defining marker
    "AREG": ("up", "scrna_de"),       # Amphiregulin, EGFR ligand, profibrotic
    "DPP4": ("up", "scrna_de"),       # CD26, associated with PI16+ fibroblasts
    "SFRP1": ("up", "scrna_de"),      # Secreted frizzled-related protein 1
    "CLU": ("up", "scrna_de"),        # Clusterin, stress-response marker
    "MFAP5": ("up", "scrna_de"),      # Microfibrillar-associated protein 5
    "GSN": ("up", "scrna_de"),        # Gelsolin, cytoskeletal regulator
    "CXCL12": ("up", "scrna_de"),     # SDF-1, immune cell recruitment
    "CFD": ("up", "scrna_de"),        # Complement factor D (adipsin)
    "PTGDS": ("up", "scrna_de"),      # Prostaglandin D2 synthase
}

# IBD Journal 2025 druggable targets (doi:10.1093/ibd/izae295)
IBD_DRUGGABLE_TARGETS: dict[str, tuple[str, str]] = {
    "HDAC1": ("up", "druggable_target"),    # HDAC inhibitors well-characterized in L1000
    "GREM1": ("up", "druggable_target"),    # BMP antagonist
    "SERPINE1": ("up", "druggable_target"), # PAI-1, druggable serine protease inhibitor
    "LY96": ("up", "druggable_target"),     # MD-2, TLR4 co-receptor
    "AKAP11": ("up", "druggable_target"),   # A-kinase anchoring protein 11
    "SRM": ("up", "druggable_target"),      # Spermidine synthase
    "EHD2": ("up", "druggable_target"),     # EH domain-containing protein 2
    "FGF2": ("up", "druggable_target"),     # bFGF, fibroblast growth factor
    "TGFB1": ("up", "pathway_gene"),
    "TGFB2": ("up", "pathway_gene"),
    "TGFBR1": ("up", "pathway_gene"),
    "TGFBR2": ("up", "pathway_gene"),
    "SMAD2": ("up", "pathway_gene"),
    "SMAD3": ("up", "pathway_gene"),
    "SMAD7": ("down", "pathway_gene"),      # Inhibitory SMAD (anti-fibrotic)
}


def build_celltype_signature(
    msigdb_dir: Path | None = None,
    bulk_sig_path: Path | None = None,
) -> pd.DataFrame:
    """Build the cell-type-resolved fibroblast fibrosis signature.

    Merges genes from all fibroblast subtype programs, deduplicates,
    cross-references with bulk DE results, and annotates with MSigDB.
    """
    msigdb_dir = msigdb_dir or MSIGDB_DEST
    bulk_sig_path = bulk_sig_path or BULK_SIG_PATH

    # Combine all fibroblast gene programs
    all_genes: dict[str, dict] = {}

    source_sets = [
        ("CTHRC1_fibroblast", CTHRC1_FIBROBLAST_GENES),
        ("TWIST1_FAP_fibroblast", TWIST1_FAP_GENES),
        ("Kong_spatial_module", KONG_SPATIAL_GENES),
        ("PI16_fibroblast", PI16_FIBROBLAST_GENES),
        ("IBD_druggable_targets", IBD_DRUGGABLE_TARGETS),
    ]

    for source_name, gene_dict in source_sets:
        for gene, (direction, evidence) in gene_dict.items():
            gene_upper = gene.upper()
            if gene_upper in all_genes:
                all_genes[gene_upper]["sources"].append(source_name)
                all_genes[gene_upper]["evidence_types"].add(evidence)
            else:
                all_genes[gene_upper] = {
                    "gene": gene_upper,
                    "direction": direction,
                    "sources": [source_name],
                    "evidence_types": {evidence},
                }

    # Build DataFrame
    records = []
    for gene_info in all_genes.values():
        best_evidence = _best_evidence(gene_info["evidence_types"])
        records.append({
            "gene": gene_info["gene"],
            "direction": gene_info["direction"],
            "sources": ";".join(gene_info["sources"]),
            "n_sources": len(gene_info["sources"]),
            "evidence_level": best_evidence,
            "evidence_types": ";".join(sorted(gene_info["evidence_types"])),
        })

    df = pd.DataFrame(records)

    # Cross-reference with bulk DE results
    if bulk_sig_path.exists():
        bulk = pd.read_csv(bulk_sig_path, sep="\t")
        bulk_lookup = {}
        for _, row in bulk.iterrows():
            gene = str(row.get("gene", "")).upper()
            if gene:
                bulk_lookup[gene] = {
                    "bulk_log2fc": row.get("mean_log2fc", np.nan),
                    "bulk_pvalue": row.get("meta_pvalue", np.nan),
                    "bulk_fdr": row.get("meta_fdr", np.nan),
                }

        df["bulk_log2fc"] = df["gene"].map(lambda g: bulk_lookup.get(g, {}).get("bulk_log2fc", np.nan))
        df["bulk_pvalue"] = df["gene"].map(lambda g: bulk_lookup.get(g, {}).get("bulk_pvalue", np.nan))
        df["bulk_fdr"] = df["gene"].map(lambda g: bulk_lookup.get(g, {}).get("bulk_fdr", np.nan))
        df["bulk_validated"] = df["bulk_fdr"] < 0.05
    else:
        df["bulk_log2fc"] = np.nan
        df["bulk_pvalue"] = np.nan
        df["bulk_fdr"] = np.nan
        df["bulk_validated"] = False

    # MSigDB pathway annotation
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

    # Sort: multi-source > protein_validated > druggable > scrna_de > spatial
    evidence_rank = {
        "protein_validated": 4,
        "druggable_target": 3,
        "scrna_de": 2,
        "spatial_module": 1,
        "pathway_gene": 0,
    }
    df["_evidence_rank"] = df["evidence_level"].map(evidence_rank).fillna(0)
    df = df.sort_values(
        ["n_sources", "_evidence_rank", "bulk_pvalue"],
        ascending=[False, False, True],
    )
    df = df.drop(columns=["_evidence_rank"])

    return df


def _best_evidence(evidence_types: set[str]) -> str:
    """Return the highest-quality evidence type from a set."""
    priority = ["protein_validated", "druggable_target", "scrna_de",
                "spatial_module", "pathway_gene"]
    for ev in priority:
        if ev in evidence_types:
            return ev
    return "unknown"


def derive_celltype_signature(
    msigdb_dir: Path | None = None,
    bulk_sig_path: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Derive and save the cell-type-resolved fibroblast signature."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cell-Type-Resolved Fibroblast Fibrosis Signature")
    print("=" * 60)

    df = build_celltype_signature(msigdb_dir, bulk_sig_path)

    up = df[df["direction"] == "up"]
    down = df[df["direction"] == "down"]
    multi = df[df["n_sources"] > 1]
    validated = df[df["bulk_validated"] == True]

    print(f"\n  Total genes: {len(df)}")
    print(f"  Upregulated: {len(up)}")
    print(f"  Downregulated: {len(down)}")
    print(f"  Multi-source genes: {len(multi)}")
    print(f"  Bulk-validated (FDR<0.05): {len(validated)}")

    # Source breakdown
    for src_name, gene_dict in [
        ("CTHRC1_fibroblast", CTHRC1_FIBROBLAST_GENES),
        ("TWIST1_FAP_fibroblast", TWIST1_FAP_GENES),
        ("Kong_spatial_module", KONG_SPATIAL_GENES),
        ("PI16_fibroblast", PI16_FIBROBLAST_GENES),
        ("IBD_druggable_targets", IBD_DRUGGABLE_TARGETS),
    ]:
        print(f"  {src_name}: {len(gene_dict)} genes")

    # Evidence breakdown
    print(f"\n  Evidence levels:")
    for ev in ["protein_validated", "druggable_target", "scrna_de",
               "spatial_module", "pathway_gene"]:
        n = (df["evidence_level"] == ev).sum()
        if n > 0:
            print(f"    {ev}: {n}")

    # Top multi-source genes
    print(f"\n  Multi-source convergence genes:")
    for _, row in multi.head(20).iterrows():
        val = "*" if row.get("bulk_validated") else " "
        fc = f"fc={row['bulk_log2fc']:+.2f}" if pd.notna(row["bulk_log2fc"]) else "fc=N/A"
        print(f"   {val} {row['gene']:12s} {row['direction']:4s}  {fc}  [{row['sources']}]")

    # Required druggable targets check
    required = {"HDAC1", "GREM1", "SERPINE1", "TWIST1", "FAP", "FGF2",
                "LY96", "AKAP11", "SRM", "EHD2"}
    found = required & set(df["gene"])
    missing = required - found
    print(f"\n  Required druggable targets: {len(found)}/{len(required)}")
    if missing:
        print(f"  MISSING: {missing}")

    # Save
    out_path = output_dir / "celltype_fibroblast_signature.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive cell-type-resolved fibroblast fibrosis signature"
    )
    parser.add_argument("--msigdb-dir", type=Path, default=MSIGDB_DEST)
    parser.add_argument("--bulk-sig", type=Path, default=BULK_SIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    derive_celltype_signature(args.msigdb_dir, args.bulk_sig, args.output_dir)


if __name__ == "__main__":
    main()
