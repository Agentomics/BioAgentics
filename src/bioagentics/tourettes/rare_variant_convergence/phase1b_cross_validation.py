"""Phase 1b: Cross-validation with Costa Rica pedigree and clinical exome data.

Extends Phase 1 gene set with:
1. Costa Rica pedigree IBD pathway enrichment (Mol Psychiatry 2022) — neuronal
   development and signal transduction pathways as a third convergence stream.
2. Clinical exome study cross-validation (Frontiers Psychiatry Feb 2026, 80
   pediatric TS patients) — ACMG-classified potentially causative variants.
3. Functional category stratification: synaptic, chromatin remodeling, signaling.

Flags PPP5C/EXOC1/GXYLT1 as lacking functional characterization for Phase 3 PPI
prediction.

Output: data/results/ts-rare-variant-convergence/phase1b/

Usage:
    uv run python -m bioagentics.tourettes.rare_variant_convergence.phase1b_cross_validation
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from bioagentics.config import REPO_ROOT
from bioagentics.tourettes.rare_variant_convergence.rare_variant_genes import (
    RareVariantGene,
    get_all_rare_variant_genes,
    summary_stats,
)

OUTPUT_DIR = REPO_ROOT / "data" / "results" / "ts-rare-variant-convergence" / "phase1b"

# ── Functional categories for gene stratification ──

FUNCTIONAL_CATEGORIES: dict[str, dict[str, str | list[str]]] = {
    "synaptic": {
        "description": "Synaptic transmission, adhesion, and vesicle trafficking",
        "genes": [
            "SLITRK1",
            "NRXN1",
            "CNTN6",
            "SLC6A1",
            "CNTNAP2",
            "EXOC1",
            "HDC",
        ],
    },
    "chromatin_remodeling": {
        "description": "Chromatin modification and transcriptional regulation",
        "genes": ["ASH1L", "KMT2C", "SMARCA2"],
    },
    "signaling": {
        "description": "Intracellular signaling cascades",
        "genes": ["WWC1", "PPP5C", "OPRK1", "COMT", "GXYLT1"],
    },
    "neurodevelopmental": {
        "description": "Neuronal migration, axon guidance, and circuit formation",
        "genes": ["CELSR3", "NDE1", "FN1", "TBX1"],
    },
    "protein_homeostasis": {
        "description": "Protein degradation and processing",
        "genes": ["NTAN1"],
    },
}

# ── Costa Rica pedigree IBD pathway enrichment (Mol Psychiatry 2022) ──
# Abdulkadir et al. performed IBD sharing analysis in a large Costa Rican TS
# pedigree and found significant enrichment in neuronal development and signal
# transduction pathways. These represent a third convergence evidence stream
# alongside rare variants and common GWAS variants.

COSTA_RICA_IBD_PATHWAYS: dict[str, dict] = {
    "neuronal_development": {
        "description": (
            "Neuronal development pathway enrichment from Costa Rica pedigree "
            "IBD analysis — includes axon guidance, neurite outgrowth, and "
            "neural circuit formation genes"
        ),
        "reference": "Abdulkadir_2022_Mol_Psychiatry",
        "overlapping_phase1_genes": [
            "SLITRK1",  # axon guidance / neurite outgrowth
            "NRXN1",  # synapse formation
            "CNTN6",  # neural circuit formation
            "CELSR3",  # axon guidance / planar cell polarity
            "NDE1",  # cortical development / neuronal migration
        ],
        "pathway_terms": [
            "GO:0007411",  # axon guidance
            "GO:0048666",  # neuron development
            "GO:0007409",  # axonogenesis
            "GO:0030182",  # neuron differentiation
        ],
    },
    "signal_transduction": {
        "description": (
            "Signal transduction pathway enrichment from Costa Rica pedigree "
            "IBD analysis — includes dopaminergic, opioid, and Hippo signaling"
        ),
        "reference": "Abdulkadir_2022_Mol_Psychiatry",
        "overlapping_phase1_genes": [
            "HDC",  # histamine signaling / basal ganglia
            "COMT",  # dopamine degradation
            "OPRK1",  # opioid signaling / dopamine modulation
            "WWC1",  # Hippo signaling
            "PPP5C",  # phosphatase / stress signaling
        ],
        "pathway_terms": [
            "GO:0007165",  # signal transduction
            "GO:0007186",  # G protein-coupled receptor signaling
            "GO:0007212",  # dopamine receptor signaling
        ],
    },
}

# ── Clinical exome ACMG-classified potentially causative variants ──
# Saia et al. (Frontiers Psychiatry Feb 2026): 80 pediatric TS patients
# 13.7% (11/80) had ACMG potentially causative variants (PC-Vs)
# Key finding: dual genetic model — rare variant subgroup shows higher tic
# severity, supporting rare+common variant convergence hypothesis.

CLINICAL_EXOME_ACMG_VARIANTS: dict[str, dict] = {
    "SLC6A1": {
        "acmg_class": "likely_pathogenic",
        "variant_effect": "missense",
        "clinical_notes": (
            "GAT-1 GABA transporter. Rare variant subgroup with higher tic "
            "severity in dual genetic model."
        ),
        "in_phase1": True,
    },
    "KMT2C": {
        "acmg_class": "likely_pathogenic",
        "variant_effect": "lof",
        "clinical_notes": (
            "MLL3 histone methyltransferase. Chromatin remodeling gene — "
            "Kleefstra syndrome spectrum."
        ),
        "in_phase1": True,
    },
    "SMARCA2": {
        "acmg_class": "likely_pathogenic",
        "variant_effect": "missense",
        "clinical_notes": (
            "SWI/SNF ATPase. Nicolaides-Baraitser syndrome — intellectual "
            "disability + behavioral features."
        ),
        "in_phase1": True,
    },
}

# Genes flagged for Phase 3 PPI prediction (no functional characterization)
UNCHARACTERIZED_GENES = ["PPP5C", "EXOC1", "GXYLT1"]


@dataclass
class GeneAnnotation:
    """Extended gene annotation with cross-validation and functional category."""

    gene_symbol: str
    evidence_strength: str
    functional_category: str
    costa_rica_ibd_pathway: str | None  # neuronal_development, signal_transduction
    clinical_exome_validated: bool
    clinical_exome_acmg_class: str | None
    needs_ppi_prediction: bool
    evidence_sources: list[str]  # which data sources support this gene


def classify_gene(gene: RareVariantGene) -> str:
    """Assign functional category to a gene."""
    symbol = gene.gene_symbol
    for category, info in FUNCTIONAL_CATEGORIES.items():
        if symbol in info["genes"]:
            return category
    return "unclassified"


def get_ibd_pathway_overlap(gene_symbol: str) -> str | None:
    """Check if gene overlaps with Costa Rica IBD enriched pathways."""
    for pathway_name, pathway_info in COSTA_RICA_IBD_PATHWAYS.items():
        if gene_symbol in pathway_info["overlapping_phase1_genes"]:
            return pathway_name
    return None


def build_evidence_sources(gene: RareVariantGene) -> list[str]:
    """List all data sources supporting this gene."""
    sources = []
    # Original Phase 1 sources from references
    for ref in gene.references:
        if "Zhan" in ref:
            sources.append("trio_exome_zhan_2025")
        elif "Clinical_exome" in ref:
            sources.append("clinical_exome_2026")
        elif "Chen_2025" in ref:
            sources.append("functional_validation_chen_2025")
        elif "Willsey" in ref:
            sources.append("trio_study_willsey_2024")
        elif "Sundaram" in ref or "Rooney" in ref or "Mao" in ref:
            sources.append("cnv_study")
        else:
            sources.append("literature_curation")

    # Costa Rica IBD pathway overlap
    if get_ibd_pathway_overlap(gene.gene_symbol):
        sources.append("costa_rica_ibd_pedigree_2022")

    # Clinical exome cross-validation
    if gene.gene_symbol in CLINICAL_EXOME_ACMG_VARIANTS:
        sources.append("clinical_exome_acmg_2026")

    return sorted(set(sources))


def annotate_genes(
    genes: list[RareVariantGene],
) -> list[GeneAnnotation]:
    """Build extended annotations for all Phase 1 genes."""
    annotations = []
    for gene in genes:
        symbol = gene.gene_symbol
        ibd_pathway = get_ibd_pathway_overlap(symbol)
        exome_info = CLINICAL_EXOME_ACMG_VARIANTS.get(symbol)

        annotations.append(
            GeneAnnotation(
                gene_symbol=symbol,
                evidence_strength=gene.evidence_strength,
                functional_category=classify_gene(gene),
                costa_rica_ibd_pathway=ibd_pathway,
                clinical_exome_validated=exome_info is not None,
                clinical_exome_acmg_class=(
                    exome_info["acmg_class"] if exome_info else None
                ),
                needs_ppi_prediction=symbol in UNCHARACTERIZED_GENES,
                evidence_sources=build_evidence_sources(gene),
            )
        )
    return annotations


def convergence_evidence_count(ann: GeneAnnotation) -> int:
    """Count distinct convergence evidence streams supporting a gene.

    Streams: (1) rare variant literature, (2) Costa Rica IBD pedigree,
    (3) clinical exome ACMG, (4) functional validation.
    """
    count = 1  # baseline: at least in Phase 1 rare variant set
    if ann.costa_rica_ibd_pathway is not None:
        count += 1
    if ann.clinical_exome_validated:
        count += 1
    if "functional_validation_chen_2025" in ann.evidence_sources:
        count += 1
    return count


def category_summary(
    annotations: list[GeneAnnotation],
) -> dict[str, list[str]]:
    """Group genes by functional category."""
    groups: dict[str, list[str]] = defaultdict(list)
    for ann in annotations:
        groups[ann.functional_category].append(ann.gene_symbol)
    return dict(groups)


def save_outputs(
    genes: list[RareVariantGene],
    annotations: list[GeneAnnotation],
    output_dir: Path,
) -> None:
    """Save Phase 1b cross-validation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON output ──
    json_path = output_dir / "phase1b_cross_validation.json"
    output = {
        "metadata": {
            "description": (
                "Phase 1b: Cross-validation of TS rare variant genes with "
                "Costa Rica pedigree IBD pathways and clinical exome data"
            ),
            "project": "ts-rare-variant-convergence",
            "phase": "1b",
            "sources": [
                "Phase 1 rare variant curation",
                "Abdulkadir et al. 2022 Mol Psychiatry (Costa Rica pedigree IBD)",
                "Saia et al. 2026 Frontiers Psychiatry (clinical exome, 80 patients)",
            ],
        },
        "costa_rica_ibd_pathways": COSTA_RICA_IBD_PATHWAYS,
        "clinical_exome_acmg_variants": CLINICAL_EXOME_ACMG_VARIANTS,
        "functional_categories": FUNCTIONAL_CATEGORIES,
        "gene_annotations": [asdict(a) for a in annotations],
        "summary": {
            "total_genes": len(annotations),
            "genes_with_ibd_overlap": sum(
                1 for a in annotations if a.costa_rica_ibd_pathway
            ),
            "genes_with_exome_validation": sum(
                1 for a in annotations if a.clinical_exome_validated
            ),
            "genes_needing_ppi_prediction": [
                a.gene_symbol for a in annotations if a.needs_ppi_prediction
            ],
            "multi_stream_genes": [
                a.gene_symbol
                for a in annotations
                if convergence_evidence_count(a) >= 2
            ],
            "category_breakdown": category_summary(annotations),
        },
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    # ── CSV output ──
    csv_path = output_dir / "phase1b_gene_annotations.csv"
    fieldnames = [
        "gene_symbol",
        "evidence_strength",
        "functional_category",
        "costa_rica_ibd_pathway",
        "clinical_exome_validated",
        "clinical_exome_acmg_class",
        "needs_ppi_prediction",
        "convergence_streams",
        "evidence_sources",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ann in annotations:
            writer.writerow(
                {
                    "gene_symbol": ann.gene_symbol,
                    "evidence_strength": ann.evidence_strength,
                    "functional_category": ann.functional_category,
                    "costa_rica_ibd_pathway": ann.costa_rica_ibd_pathway or "",
                    "clinical_exome_validated": ann.clinical_exome_validated,
                    "clinical_exome_acmg_class": ann.clinical_exome_acmg_class or "",
                    "needs_ppi_prediction": ann.needs_ppi_prediction,
                    "convergence_streams": convergence_evidence_count(ann),
                    "evidence_sources": ";".join(ann.evidence_sources),
                }
            )
    print(f"  Saved CSV:  {csv_path}")

    # ── Text report ──
    report_path = output_dir / "phase1b_report.txt"
    stats = summary_stats(genes)
    with open(report_path, "w") as f:
        f.write("TS Rare Variant Gene Cross-Validation Report — Phase 1b\n")
        f.write("=" * 58 + "\n\n")

        f.write("## Gene Set Summary\n")
        f.write(f"Total genes (from Phase 1): {stats['total_genes']}\n")
        f.write(
            f"Strong/moderate evidence: {stats['strong_moderate_count']}\n\n"
        )

        f.write("## Costa Rica Pedigree IBD Pathway Overlap\n")
        f.write(
            "Source: Abdulkadir et al. 2022 Mol Psychiatry\n"
        )
        for pw_name, pw_info in COSTA_RICA_IBD_PATHWAYS.items():
            overlap = pw_info["overlapping_phase1_genes"]
            f.write(
                f"  {pw_name}: {len(overlap)} genes overlap "
                f"({', '.join(overlap)})\n"
            )
        ibd_genes = {
            g
            for pw in COSTA_RICA_IBD_PATHWAYS.values()
            for g in pw["overlapping_phase1_genes"]
        }
        f.write(
            f"  Total unique genes with IBD pathway support: {len(ibd_genes)}\n\n"
        )

        f.write("## Clinical Exome Cross-Validation\n")
        f.write(
            "Source: Saia et al. 2026 Frontiers Psychiatry (80 pediatric TS)\n"
        )
        f.write(
            f"  ACMG potentially causative variants matching Phase 1: "
            f"{len(CLINICAL_EXOME_ACMG_VARIANTS)}\n"
        )
        for gene, info in CLINICAL_EXOME_ACMG_VARIANTS.items():
            f.write(
                f"    {gene}: {info['acmg_class']} ({info['variant_effect']})\n"
            )
        f.write("\n")

        f.write("## Functional Category Stratification\n")
        cats = category_summary(annotations)
        for cat_name, cat_genes in sorted(cats.items()):
            desc = FUNCTIONAL_CATEGORIES.get(cat_name, {}).get(
                "description", ""
            )
            f.write(f"  {cat_name} ({len(cat_genes)}): {', '.join(cat_genes)}\n")
            if desc:
                f.write(f"    — {desc}\n")
        f.write("\n")

        f.write("## Multi-Stream Convergence Genes\n")
        f.write("Genes supported by >=2 independent evidence streams:\n")
        for ann in sorted(annotations, key=convergence_evidence_count, reverse=True):
            count = convergence_evidence_count(ann)
            if count >= 2:
                streams = []
                streams.append("rare_variant")
                if ann.costa_rica_ibd_pathway:
                    streams.append(f"ibd_pedigree({ann.costa_rica_ibd_pathway})")
                if ann.clinical_exome_validated:
                    streams.append("clinical_exome_acmg")
                if "functional_validation_chen_2025" in ann.evidence_sources:
                    streams.append("functional_validation")
                f.write(
                    f"  {ann.gene_symbol} ({count} streams): "
                    f"{', '.join(streams)}\n"
                )
        f.write("\n")

        f.write("## Genes Needing Phase 3 PPI Prediction\n")
        f.write("No functional characterization — mechanism to be predicted:\n")
        for sym in UNCHARACTERIZED_GENES:
            ann_match = next(
                (a for a in annotations if a.gene_symbol == sym), None
            )
            if ann_match:
                f.write(
                    f"  {sym} (category: {ann_match.functional_category})\n"
                )
        f.write("\n")
    print(f"  Saved report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1b: Cross-validate TS gene set with pedigree and clinical data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for results",
    )
    args = parser.parse_args()

    print("Phase 1b: Cross-validating TS rare variant genes...")

    # Load Phase 1 genes
    genes = get_all_rare_variant_genes()
    print(f"  Loaded {len(genes)} genes from Phase 1")

    # Annotate with cross-validation data
    annotations = annotate_genes(genes)

    # Summary
    ibd_count = sum(1 for a in annotations if a.costa_rica_ibd_pathway)
    exome_count = sum(1 for a in annotations if a.clinical_exome_validated)
    multi_stream = [
        a.gene_symbol
        for a in annotations
        if convergence_evidence_count(a) >= 2
    ]
    print(f"  Genes with Costa Rica IBD pathway overlap: {ibd_count}")
    print(f"  Genes with clinical exome ACMG validation: {exome_count}")
    print(f"  Multi-stream convergence genes (>=2): {len(multi_stream)}")
    print(f"    {', '.join(multi_stream)}")

    # Category breakdown
    cats = category_summary(annotations)
    print("  Functional categories:")
    for cat, cat_genes in sorted(cats.items()):
        print(f"    {cat}: {', '.join(cat_genes)}")

    # Uncharacterized genes
    print(f"  Flagged for Phase 3 PPI prediction: {', '.join(UNCHARACTERIZED_GENES)}")

    # Save outputs
    save_outputs(genes, annotations, args.output)
    print("\n  Phase 1b complete.")


if __name__ == "__main__":
    main()
