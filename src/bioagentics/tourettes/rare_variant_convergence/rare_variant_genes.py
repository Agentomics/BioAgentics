"""Phase 1: Curate TS-implicated rare variant genes with evidence scoring.

Compiles a structured dataset of all Tourette syndrome genes identified through
rare variant studies (de novo, segregation, functional validation, CNV, case reports).
Each gene is annotated with evidence type, evidence strength, variant type, and
pathway membership.

Evidence strength criteria:
- strong: Replicated across independent studies AND/OR functional validation
- moderate: Multiple independent reports OR strong statistical support
- suggestive: Single report, no functional data, or indirect evidence

Data sources:
- Abelson et al. (Science 2005): SLITRK1
- Ercan-Sencicek et al. (NEJM 2010): HDC
- Fernandez et al. (PNAS 2012): NRXN1 deletions
- Paschou et al. (Mol Psychiatry 2014): CNTN6
- Chen et al. (Science Advances 2025): WWC1 W88C functional validation
- Zhan et al. (bioRxiv 2025): 1,466 trio exome study (PPP5C, EXOC1, GXYLT1)
- Clinical exome study (Frontiers Psychiatry 2026): 80 pediatric TS patients
- TSAICG GWAS (2019/2024): genes with both rare and common variant support

Output: data/results/ts-rare-variant-convergence/phase1/rare_variant_genes.{json,csv}

Usage:
    uv run python -m bioagentics.tourettes.rare_variant_convergence.rare_variant_genes
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "data" / "results" / "ts-rare-variant-convergence" / "phase1"


@dataclass
class RareVariantGene:
    """A TS-implicated gene with evidence annotation."""

    gene_symbol: str
    evidence_types: list[str]  # de_novo, segregation, functional, cnv, case_report
    evidence_strength: str  # strong, moderate, suggestive
    variant_types: list[str]  # lof, missense, structural, regulatory
    pathways: list[str]
    chromosome: str
    references: list[str]
    notes: str = ""


# ── Established TS rare variant genes (strong evidence) ──

ESTABLISHED_GENES: list[RareVariantGene] = [
    RareVariantGene(
        gene_symbol="SLITRK1",
        evidence_types=["segregation", "case_report", "functional"],
        evidence_strength="strong",
        variant_types=["missense", "regulatory"],
        pathways=["axon_guidance", "neurite_outgrowth", "synapse_formation"],
        chromosome="13q31.1",
        references=[
            "Abelson_2005_Science",
            "O'Roak_2010_PNAS",
            "Karagiannidis_2012_Mol_Psychiatry",
        ],
        notes="First TS risk gene identified. var321 3'UTR variant disrupts miR-189 "
        "binding. Functional studies show role in neurite outgrowth. Some "
        "replication failures noted but gene remains implicated.",
    ),
    RareVariantGene(
        gene_symbol="HDC",
        evidence_types=["segregation", "functional"],
        evidence_strength="strong",
        variant_types=["lof"],
        pathways=["histamine_synthesis", "basal_ganglia_modulation"],
        chromosome="15q21.2",
        references=[
            "Ercan-Sencicek_2010_NEJM",
            "Baldan_2014_Neuron",
            "Rapanelli_2014_J_Neurosci",
        ],
        notes="W317X nonsense mutation in TS pedigree. Hdc-knockout mice show TS-like "
        "stereotypies and striatal dopamine dysregulation. Led to pitolisant "
        "(H3 agonist) clinical trials.",
    ),
    RareVariantGene(
        gene_symbol="NRXN1",
        evidence_types=["cnv", "case_report"],
        evidence_strength="strong",
        variant_types=["structural"],
        pathways=["synaptic_adhesion", "synapse_formation", "neurotransmitter_release"],
        chromosome="2p16.3",
        references=[
            "Fernandez_2012_PNAS",
            "Huang_2017_J_Med_Genet",
            "Sundaram_2010_Hum_Mol_Genet",
        ],
        notes="Recurrent exonic deletions in TS. Also implicated in autism and "
        "schizophrenia — shared neurodevelopmental risk gene. Presynaptic "
        "adhesion molecule essential for synapse specification.",
    ),
    RareVariantGene(
        gene_symbol="CNTN6",
        evidence_types=["case_report", "cnv"],
        evidence_strength="strong",
        variant_types=["structural", "missense"],
        pathways=["axon_guidance", "cell_adhesion", "neural_circuit_formation"],
        chromosome="3p26.3",
        references=[
            "Paschou_2014_Mol_Psychiatry",
            "Fernandez_2019_Brain_Sci",
        ],
        notes="Contactin-6 involved in neural circuit formation. CNVs and point "
        "mutations reported. Contactin family broadly implicated in "
        "neurodevelopmental disorders.",
    ),
    RareVariantGene(
        gene_symbol="WWC1",
        evidence_types=["de_novo", "functional"],
        evidence_strength="strong",
        variant_types=["missense"],
        pathways=[
            "hippo_signaling",
            "dopamine_regulation",
            "synaptic_function",
            "neurodevelopment",
        ],
        chromosome="5q34",
        references=[
            "Willsey_2024_Cell",
            "Chen_2025_Science_Advances",
        ],
        notes="W88C knock-in mouse shows TS-like phenotype: repetitive motor "
        "behaviors, sensorimotor gating deficits, excess striatal dopamine. "
        "Mechanism: KIBRA protein degradation via Hippo pathway. Rescue is "
        "developmental-stage-specific, aligning with childhood TS onset. "
        "First functional validation connecting TS to Hippo signaling.",
    ),
]

# ── De novo variant genes from trio studies (moderate evidence) ──

DE_NOVO_GENES: list[RareVariantGene] = [
    RareVariantGene(
        gene_symbol="PPP5C",
        evidence_types=["de_novo"],
        evidence_strength="moderate",
        variant_types=["lof", "missense"],
        pathways=["stress_signaling", "phosphatase_activity"],
        chromosome="19q13.33",
        references=["Zhan_2025_bioRxiv"],
        notes="Serine/threonine phosphatase involved in stress response signaling. "
        "No TS-specific functional characterization yet. PPI network may "
        "predict mechanism.",
    ),
    RareVariantGene(
        gene_symbol="EXOC1",
        evidence_types=["de_novo"],
        evidence_strength="moderate",
        variant_types=["lof", "missense"],
        pathways=["vesicle_trafficking", "neurite_outgrowth", "exocytosis"],
        chromosome="4q12",
        references=["Zhan_2025_bioRxiv"],
        notes="Exocyst complex component 1. Involved in vesicle trafficking and "
        "neurite outgrowth. No TS-specific functional characterization yet.",
    ),
    RareVariantGene(
        gene_symbol="GXYLT1",
        evidence_types=["de_novo"],
        evidence_strength="moderate",
        variant_types=["missense"],
        pathways=["notch_signaling", "glycosylation"],
        chromosome="12q12",
        references=["Zhan_2025_bioRxiv"],
        notes="Glucoside xylosyltransferase 1. Modifies Notch receptor — a pathway "
        "involved in neuronal differentiation. No TS-specific functional data.",
    ),
    RareVariantGene(
        gene_symbol="CELSR3",
        evidence_types=["de_novo", "case_report"],
        evidence_strength="moderate",
        variant_types=["missense"],
        pathways=["planar_cell_polarity", "axon_guidance", "neural_development"],
        chromosome="3p21.31",
        references=[
            "Willsey_2024_Cell",
            "Zhan_2025_bioRxiv",
        ],
        notes="Planar cell polarity protein. Also appears in TSAICG GWAS — rare and "
        "common variant support. Involved in axon guidance in striatal circuits.",
    ),
    RareVariantGene(
        gene_symbol="ASH1L",
        evidence_types=["de_novo"],
        evidence_strength="moderate",
        variant_types=["lof", "missense"],
        pathways=["chromatin_remodeling", "histone_methylation", "transcription_regulation"],
        chromosome="1q22",
        references=[
            "Willsey_2024_Cell",
            "Zhan_2025_bioRxiv",
        ],
        notes="Histone methyltransferase (H3K36). Also implicated in ASD with "
        "intellectual disability. Also appears in TSAICG GWAS as suggestive "
        "locus — dual rare/common support.",
    ),
]

# ── Clinical exome study genes (Frontiers Psychiatry 2026) ──

CLINICAL_EXOME_GENES: list[RareVariantGene] = [
    RareVariantGene(
        gene_symbol="SLC6A1",
        evidence_types=["case_report"],
        evidence_strength="moderate",
        variant_types=["missense"],
        pathways=["gaba_transport", "synaptic_transmission", "inhibitory_signaling"],
        chromosome="3p25.3",
        references=["Clinical_exome_2026_Frontiers_Psychiatry"],
        notes="GABA transporter 1 (GAT-1). ACMG-classified potentially causative "
        "variant in pediatric TS cohort. Synaptic gene — fits dual genetic "
        "model (rare variant subgroup with higher tic severity).",
    ),
    RareVariantGene(
        gene_symbol="KMT2C",
        evidence_types=["case_report"],
        evidence_strength="moderate",
        variant_types=["lof", "missense"],
        pathways=["chromatin_remodeling", "histone_methylation", "transcription_regulation"],
        chromosome="7q36.1",
        references=["Clinical_exome_2026_Frontiers_Psychiatry"],
        notes="Histone-lysine methyltransferase 2C (MLL3). ACMG potentially causative "
        "in TS. Chromatin remodeling gene — known NDD gene in Kleefstra "
        "syndrome spectrum.",
    ),
    RareVariantGene(
        gene_symbol="SMARCA2",
        evidence_types=["case_report"],
        evidence_strength="moderate",
        variant_types=["missense"],
        pathways=["chromatin_remodeling", "swi_snf_complex", "transcription_regulation"],
        chromosome="9p24.3",
        references=["Clinical_exome_2026_Frontiers_Psychiatry"],
        notes="SWI/SNF chromatin remodeling complex ATPase. ACMG potentially causative "
        "in TS. Mutations cause Nicolaides-Baraitser syndrome (intellectual "
        "disability + behavioral features).",
    ),
]

# ── CNV region genes ──

CNV_REGION_GENES: list[RareVariantGene] = [
    RareVariantGene(
        gene_symbol="NDE1",
        evidence_types=["cnv"],
        evidence_strength="moderate",
        variant_types=["structural"],
        pathways=["neuronal_migration", "cortical_development", "centrosome_function"],
        chromosome="16p13.11",
        references=[
            "Sundaram_2010_Hum_Mol_Genet",
            "Rooney_2021_Am_J_Med_Genet",
        ],
        notes="Nuclear distribution E homologue 1. 16p13.11 microduplication region "
        "associated with TS and other NDDs. Essential for cortical neuronal "
        "migration and brain size regulation.",
    ),
    RareVariantGene(
        gene_symbol="NTAN1",
        evidence_types=["cnv"],
        evidence_strength="suggestive",
        variant_types=["structural"],
        pathways=["protein_degradation", "n_end_rule_pathway"],
        chromosome="16p13.11",
        references=["Sundaram_2010_Hum_Mol_Genet"],
        notes="16p13.11 region gene. N-terminal asparagine amidase — protein "
        "degradation pathway. Less direct neurological function than NDE1.",
    ),
    RareVariantGene(
        gene_symbol="COMT",
        evidence_types=["cnv", "case_report"],
        evidence_strength="moderate",
        variant_types=["structural", "missense"],
        pathways=["dopamine_degradation", "catecholamine_metabolism"],
        chromosome="22q11.21",
        references=[
            "Gothelf_2007_Am_J_Psychiatry",
            "Mao_2015_Genet_Med",
        ],
        notes="Catechol-O-methyltransferase. 22q11.2 deletion syndrome (DiGeorge) "
        "includes elevated TS risk. Val158Met polymorphism modulates dopamine "
        "in prefrontal cortex. Direct relevance to dopaminergic TS mechanism.",
    ),
    RareVariantGene(
        gene_symbol="TBX1",
        evidence_types=["cnv"],
        evidence_strength="suggestive",
        variant_types=["structural"],
        pathways=["neural_crest_development", "pharyngeal_arch_patterning"],
        chromosome="22q11.21",
        references=["Mao_2015_Genet_Med"],
        notes="T-box transcription factor 1. Key driver of 22q11.2 deletion "
        "phenotype. Neural crest and brain development roles. Indirect "
        "TS evidence via 22q11.2DS comorbidity.",
    ),
]

# ── Additional candidate genes (suggestive evidence) ──

CANDIDATE_GENES: list[RareVariantGene] = [
    RareVariantGene(
        gene_symbol="OPRK1",
        evidence_types=["case_report"],
        evidence_strength="suggestive",
        variant_types=["missense"],
        pathways=["opioid_signaling", "dopamine_modulation", "reward_circuitry"],
        chromosome="8q11.23",
        references=["Willsey_2024_Cell"],
        notes="Kappa opioid receptor. Modulates dopamine release in striatum. "
        "Single candidate gene identification — awaiting replication.",
    ),
    RareVariantGene(
        gene_symbol="FN1",
        evidence_types=["case_report"],
        evidence_strength="suggestive",
        variant_types=["missense"],
        pathways=["extracellular_matrix", "cell_adhesion", "neural_migration"],
        chromosome="2q35",
        references=["Willsey_2024_Cell"],
        notes="Fibronectin 1. ECM glycoprotein involved in cell adhesion and "
        "migration including neural crest. Limited TS-specific evidence.",
    ),
    RareVariantGene(
        gene_symbol="CNTNAP2",
        evidence_types=["case_report", "cnv"],
        evidence_strength="moderate",
        variant_types=["structural", "missense"],
        pathways=["synaptic_adhesion", "potassium_channel_clustering", "language_development"],
        chromosome="7q35-q36.1",
        references=[
            "Verkerk_2003_Am_J_Hum_Genet",
            "Poot_2010_Eur_J_Hum_Genet",
        ],
        notes="Contactin-associated protein-like 2. Disrupted in TS families and "
        "also implicated in autism, language disorders. Neurexin superfamily "
        "member involved in axon-glial interactions.",
    ),
]


def get_all_rare_variant_genes() -> list[RareVariantGene]:
    """Return all curated TS rare variant genes."""
    return (
        ESTABLISHED_GENES
        + DE_NOVO_GENES
        + CLINICAL_EXOME_GENES
        + CNV_REGION_GENES
        + CANDIDATE_GENES
    )


def genes_to_records(genes: list[RareVariantGene]) -> list[dict]:
    """Convert gene list to flat dictionaries for serialization."""
    records = []
    for g in genes:
        rec = asdict(g)
        # Flatten lists to semicolon-separated strings for CSV
        rec["evidence_types"] = ";".join(rec["evidence_types"])
        rec["variant_types"] = ";".join(rec["variant_types"])
        rec["pathways"] = ";".join(rec["pathways"])
        rec["references"] = ";".join(rec["references"])
        records.append(rec)
    return records


def genes_to_json(genes: list[RareVariantGene]) -> list[dict]:
    """Convert gene list to JSON-friendly dictionaries (lists preserved)."""
    return [asdict(g) for g in genes]


def summary_stats(genes: list[RareVariantGene]) -> dict:
    """Compute summary statistics for the gene set."""
    from collections import Counter

    strength_counts = Counter(g.evidence_strength for g in genes)
    evidence_type_counts: dict[str, int] = {}
    pathway_counts: dict[str, int] = {}

    for g in genes:
        for et in g.evidence_types:
            evidence_type_counts[et] = evidence_type_counts.get(et, 0) + 1
        for pw in g.pathways:
            pathway_counts[pw] = pathway_counts.get(pw, 0) + 1

    strong_moderate = [g for g in genes if g.evidence_strength in ("strong", "moderate")]

    return {
        "total_genes": len(genes),
        "strong_moderate_count": len(strong_moderate),
        "by_evidence_strength": dict(strength_counts.most_common()),
        "by_evidence_type": dict(
            sorted(evidence_type_counts.items(), key=lambda x: -x[1])
        ),
        "top_pathways": dict(
            sorted(pathway_counts.items(), key=lambda x: -x[1])[:15]
        ),
        "gene_symbols": [g.gene_symbol for g in genes],
        "strong_moderate_symbols": [g.gene_symbol for g in strong_moderate],
    }


def save_outputs(genes: list[RareVariantGene], output_dir: Path) -> None:
    """Save gene curation dataset as JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output (preserves list structure)
    json_path = output_dir / "rare_variant_genes.json"
    output_data = {
        "metadata": {
            "description": "TS-implicated rare variant genes with evidence scoring",
            "project": "ts-rare-variant-convergence",
            "phase": "1",
            "target": ">=15 genes with strong/moderate evidence",
        },
        "summary": summary_stats(genes),
        "genes": genes_to_json(genes),
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved JSON: {json_path} ({len(genes)} genes)")

    # CSV output (flat format)
    csv_path = output_dir / "rare_variant_genes.csv"
    records = genes_to_records(genes)
    fieldnames = [
        "gene_symbol",
        "evidence_types",
        "evidence_strength",
        "variant_types",
        "pathways",
        "chromosome",
        "references",
        "notes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved CSV:  {csv_path}")

    # Summary report
    stats = summary_stats(genes)
    report_path = output_dir / "curation_report.txt"
    with open(report_path, "w") as f:
        f.write("TS Rare Variant Gene Curation Report — Phase 1\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total genes curated: {stats['total_genes']}\n")
        f.write(f"Strong/moderate evidence: {stats['strong_moderate_count']}\n")
        meets_target = stats["strong_moderate_count"] >= 15
        f.write(f"Meets target (>=15): {'YES' if meets_target else 'NO'}\n\n")
        f.write("By evidence strength:\n")
        for strength, count in stats["by_evidence_strength"].items():
            f.write(f"  {strength}: {count}\n")
        f.write("\nBy evidence type:\n")
        for etype, count in stats["by_evidence_type"].items():
            f.write(f"  {etype}: {count}\n")
        f.write("\nTop pathways (by gene count):\n")
        for pw, count in stats["top_pathways"].items():
            f.write(f"  {pw}: {count}\n")
        f.write(f"\nAll gene symbols: {', '.join(stats['gene_symbols'])}\n")
        f.write(
            f"Strong/moderate: {', '.join(stats['strong_moderate_symbols'])}\n"
        )
    print(f"  Saved report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate TS rare variant gene set (Phase 1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for results",
    )
    args = parser.parse_args()

    print("Phase 1: Curating TS rare variant genes...")
    genes = get_all_rare_variant_genes()

    stats = summary_stats(genes)
    print(f"  Total genes: {stats['total_genes']}")
    print(f"  Strong/moderate: {stats['strong_moderate_count']}")
    for strength, count in stats["by_evidence_strength"].items():
        print(f"    {strength}: {count}")

    save_outputs(genes, args.output)

    if stats["strong_moderate_count"] >= 15:
        print("\n  SUCCESS: Target met (>=15 genes with strong/moderate evidence)")
    else:
        print(
            f"\n  WARNING: Below target ({stats['strong_moderate_count']}/15 "
            "genes with strong/moderate evidence)"
        )


if __name__ == "__main__":
    main()
