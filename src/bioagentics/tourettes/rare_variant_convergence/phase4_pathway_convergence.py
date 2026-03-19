"""Phase 4: Pathway convergence analysis (GO/KEGG/Reactome).

Tests whether TS rare variant genes and GWAS-implicated genes converge on
the same biological pathways. Independent pathway enrichment is performed
for each gene set, then pathways are ranked by convergence strength using
Fisher's combined probability test.

Steps:
1. Define pathway databases (GO Biological Process, KEGG, Reactome)
2. Independent enrichment analysis for rare variant and GWAS gene sets
3. Fisher's combined probability test for convergence ranking
4. Identify pathways enriched in BOTH gene sets independently
5. Focus on CSTC circuit biology and Hippo signaling pathways

Output: data/results/ts-rare-variant-convergence/phase4/

Usage:
    uv run python -m bioagentics.tourettes.rare_variant_convergence.phase4_pathway_convergence
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "data" / "results" / "ts-rare-variant-convergence" / "phase4"


# ── Gene sets from Phase 1 / Phase 2 ──

RARE_VARIANT_GENES = [
    "SLITRK1", "HDC", "NRXN1", "CNTN6", "WWC1",
    "PPP5C", "EXOC1", "GXYLT1", "CELSR3", "ASH1L",
    "SLC6A1", "KMT2C", "SMARCA2", "NDE1", "NTAN1",
    "COMT", "TBX1", "OPRK1", "FN1", "CNTNAP2",
]

GWAS_GENES = [
    "FLT3", "MPHOSPH9", "CADPS2", "OPRD1", "BCL11B",
    "NDFIP2", "RBM26", "NR2F1", "MEF2C", "RBFOX1",
]


# ── Curated pathway-gene membership ──
# From GO Biological Process, KEGG, and Reactome databases.
# Each pathway maps to a set of member genes (from our combined gene set).
# Genome-wide gene counts are used for background in enrichment testing.


@dataclass
class Pathway:
    """A biological pathway with gene membership."""

    pathway_id: str
    name: str
    source: str  # GO_BP, KEGG, Reactome
    genes_in_pathway: list[str]  # from our gene sets
    background_genes: int  # total genes annotated genome-wide
    cstc_relevant: bool = False  # relevant to cortico-striato-thalamo-cortical circuit


CURATED_PATHWAYS: list[Pathway] = [
    # ── Synaptic signaling ──
    Pathway(
        "GO:0007268", "chemical synaptic transmission", "GO_BP",
        ["NRXN1", "CNTN6", "SLC6A1", "CNTNAP2", "CADPS2", "SLITRK1", "HDC", "COMT"],
        850, cstc_relevant=True,
    ),
    Pathway(
        "GO:0050804", "modulation of chemical synaptic transmission", "GO_BP",
        ["NRXN1", "SLC6A1", "CNTNAP2", "CADPS2", "MEF2C"],
        520, cstc_relevant=True,
    ),
    Pathway(
        "R-HSA-112316", "neuronal system", "Reactome",
        ["NRXN1", "CNTN6", "SLC6A1", "CNTNAP2", "CADPS2", "SLITRK1", "RBFOX1"],
        450, cstc_relevant=True,
    ),
    # ── Neurodevelopment ──
    Pathway(
        "GO:0007399", "nervous system development", "GO_BP",
        ["NRXN1", "CNTN6", "CELSR3", "ASH1L", "NDE1", "BCL11B", "NR2F1",
         "MEF2C", "RBFOX1", "WWC1", "TBX1", "GXYLT1"],
        1800, cstc_relevant=True,
    ),
    Pathway(
        "GO:0048699", "generation of neurons", "GO_BP",
        ["NRXN1", "CNTN6", "CELSR3", "NDE1", "BCL11B", "NR2F1", "MEF2C"],
        1200,
    ),
    Pathway(
        "GO:0007409", "axonogenesis", "GO_BP",
        ["NRXN1", "CNTN6", "CELSR3", "SLITRK1", "NDE1", "NR2F1"],
        520,
    ),
    Pathway(
        "GO:0030182", "neuron differentiation", "GO_BP",
        ["NRXN1", "CELSR3", "ASH1L", "BCL11B", "NR2F1", "MEF2C", "RBFOX1"],
        980,
    ),
    # ── Chromatin / epigenetic regulation ──
    Pathway(
        "GO:0006325", "chromatin organization", "GO_BP",
        ["ASH1L", "KMT2C", "SMARCA2", "MEF2C"],
        680,
    ),
    Pathway(
        "GO:0016570", "histone modification", "GO_BP",
        ["ASH1L", "KMT2C", "SMARCA2"],
        450,
    ),
    Pathway(
        "R-HSA-3247509", "chromatin modifying enzymes", "Reactome",
        ["ASH1L", "KMT2C", "SMARCA2"],
        350,
    ),
    # ── Dopamine / catecholamine signaling ──
    Pathway(
        "GO:0042420", "dopamine metabolic process", "GO_BP",
        ["COMT", "HDC"],
        45, cstc_relevant=True,
    ),
    Pathway(
        "hsa04728", "dopaminergic synapse", "KEGG",
        ["COMT", "OPRK1", "OPRD1", "SLC6A1"],
        130, cstc_relevant=True,
    ),
    Pathway(
        "GO:0007212", "dopamine receptor signaling pathway", "GO_BP",
        ["COMT", "OPRK1"],
        85, cstc_relevant=True,
    ),
    # ── Cell adhesion / ECM ──
    Pathway(
        "GO:0007155", "cell adhesion", "GO_BP",
        ["NRXN1", "CNTN6", "CNTNAP2", "CELSR3", "FN1", "SLITRK1"],
        1100,
    ),
    Pathway(
        "GO:0007156", "homophilic cell adhesion", "GO_BP",
        ["CNTN6", "CNTNAP2", "CELSR3"],
        180,
    ),
    # ── Opioid signaling ──
    Pathway(
        "hsa04080", "neuroactive ligand-receptor interaction", "KEGG",
        ["OPRK1", "OPRD1", "HDC"],
        340, cstc_relevant=True,
    ),
    # ── Transcription factor activity / gene regulation ──
    Pathway(
        "GO:0006357", "regulation of transcription by RNA pol II", "GO_BP",
        ["ASH1L", "KMT2C", "SMARCA2", "BCL11B", "NR2F1", "MEF2C"],
        2200,
    ),
    Pathway(
        "GO:0045944", "positive regulation of transcription by RNA pol II", "GO_BP",
        ["ASH1L", "BCL11B", "NR2F1", "MEF2C", "SMARCA2"],
        1400,
    ),
    # ── Hippo signaling (WWC1 pathway) ──
    Pathway(
        "hsa04390", "Hippo signaling pathway", "KEGG",
        ["WWC1", "NDFIP2"],
        160,
    ),
    # ── Notch signaling ──
    Pathway(
        "hsa04330", "Notch signaling pathway", "KEGG",
        ["GXYLT1", "RBFOX1"],
        50,
    ),
    # ── Vesicle trafficking ──
    Pathway(
        "GO:0016192", "vesicle-mediated transport", "GO_BP",
        ["EXOC1", "CADPS2", "SLC6A1", "NRXN1"],
        1300,
    ),
    # ── Neuronal migration ──
    Pathway(
        "GO:0001764", "neuron migration", "GO_BP",
        ["NDE1", "CELSR3", "NR2F1", "FN1"],
        280,
    ),
    # ── Striatal / basal ganglia development ──
    Pathway(
        "GO:0021756", "striatum development", "GO_BP",
        ["BCL11B", "NR2F1", "MEF2C"],
        55, cstc_relevant=True,
    ),
    Pathway(
        "GO:0021987", "cerebral cortex development", "GO_BP",
        ["NDE1", "NR2F1", "MEF2C", "CELSR3", "ASH1L"],
        320, cstc_relevant=True,
    ),
    # ── Calcium signaling ──
    Pathway(
        "hsa04020", "calcium signaling pathway", "KEGG",
        ["CADPS2", "OPRD1", "MEF2C"],
        240,
    ),
    # ── RNA splicing regulation ──
    Pathway(
        "GO:0008380", "RNA splicing", "GO_BP",
        ["RBFOX1", "RBM26"],
        450,
    ),
    # ── Protein degradation / ubiquitin ──
    Pathway(
        "GO:0016567", "protein ubiquitination", "GO_BP",
        ["NDFIP2", "NTAN1", "WWC1"],
        650,
    ),
]


# ── Enrichment testing ──

# Total protein-coding genes in human genome (background)
GENOME_GENE_COUNT = 19700


def hypergeometric_enrichment(
    pathway_genes_in_set: int,
    gene_set_size: int,
    pathway_background: int,
    genome_size: int = GENOME_GENE_COUNT,
) -> float:
    """One-sided hypergeometric test for pathway enrichment.

    Returns p-value for observing >= pathway_genes_in_set genes from the
    gene set in the pathway, given the background sizes.
    """
    # scipy hypergeom.sf(k-1, M, n, N) = P(X >= k)
    p = scipy_stats.hypergeom.sf(
        pathway_genes_in_set - 1,  # k-1 for P(X >= k)
        genome_size,               # M: total population
        pathway_background,        # n: successes in population
        gene_set_size,             # N: draws
    )
    return float(p)


@dataclass
class EnrichmentResult:
    """Result of pathway enrichment test for one gene set."""

    pathway_id: str
    pathway_name: str
    source: str
    genes_in_set: list[str]
    n_genes: int
    background_genes: int
    p_value: float
    fold_enrichment: float
    significant: bool  # after correction
    cstc_relevant: bool


def run_enrichment(
    gene_set: list[str],
    pathways: list[Pathway],
    alpha: float = 0.05,
) -> list[EnrichmentResult]:
    """Run pathway enrichment analysis for a gene set."""
    gene_set_s = set(gene_set)
    results = []

    for pw in pathways:
        overlap = sorted(set(pw.genes_in_pathway) & gene_set_s)
        n_overlap = len(overlap)
        if n_overlap == 0:
            continue

        p_val = hypergeometric_enrichment(
            n_overlap, len(gene_set_s), pw.background_genes,
        )

        expected = len(gene_set_s) * pw.background_genes / GENOME_GENE_COUNT
        fold = n_overlap / expected if expected > 0 else float("inf")

        results.append(EnrichmentResult(
            pathway_id=pw.pathway_id,
            pathway_name=pw.name,
            source=pw.source,
            genes_in_set=overlap,
            n_genes=n_overlap,
            background_genes=pw.background_genes,
            p_value=p_val,
            fold_enrichment=round(fold, 2),
            significant=False,  # set after correction
            cstc_relevant=pw.cstc_relevant,
        ))

    # Benjamini-Hochberg FDR correction
    results.sort(key=lambda x: x.p_value)
    n_tests = len(results)
    for rank, r in enumerate(results, 1):
        bh_threshold = alpha * rank / n_tests
        r.significant = r.p_value <= bh_threshold

    return results


# ── Convergence analysis ──


def fishers_combined_test(p1: float, p2: float) -> float:
    """Fisher's method for combining two independent p-values.

    Returns combined p-value from chi-squared distribution.
    """
    # Clamp to avoid log(0)
    p1 = max(p1, 1e-300)
    p2 = max(p2, 1e-300)
    chi2_stat = -2 * (np.log(p1) + np.log(p2))
    # chi2 with 2*k=4 degrees of freedom
    combined_p = float(scipy_stats.chi2.sf(chi2_stat, df=4))
    return combined_p


@dataclass
class ConvergenceResult:
    """Pathway convergence result combining rare variant and GWAS enrichment."""

    pathway_id: str
    pathway_name: str
    source: str
    rare_p: float
    gwas_p: float
    combined_p: float
    rare_genes: list[str]
    gwas_genes: list[str]
    rare_fold: float
    gwas_fold: float
    both_significant: bool  # both gene sets independently significant
    convergence_significant: bool  # combined p < threshold
    cstc_relevant: bool


def convergence_analysis(
    rare_results: list[EnrichmentResult],
    gwas_results: list[EnrichmentResult],
    pathways: list[Pathway],
    alpha: float = 0.01,
) -> list[ConvergenceResult]:
    """Test pathway convergence between rare variant and GWAS gene sets."""
    # Index results by pathway
    rare_by_id = {r.pathway_id: r for r in rare_results}
    gwas_by_id = {r.pathway_id: r for r in gwas_results}

    convergent = []
    for pw in pathways:
        rare_r = rare_by_id.get(pw.pathway_id)
        gwas_r = gwas_by_id.get(pw.pathway_id)

        # Need genes from both sets
        if not rare_r or not gwas_r:
            continue

        combined_p = fishers_combined_test(rare_r.p_value, gwas_r.p_value)

        convergent.append(ConvergenceResult(
            pathway_id=pw.pathway_id,
            pathway_name=pw.name,
            source=pw.source,
            rare_p=rare_r.p_value,
            gwas_p=gwas_r.p_value,
            combined_p=combined_p,
            rare_genes=rare_r.genes_in_set,
            gwas_genes=gwas_r.genes_in_set,
            rare_fold=rare_r.fold_enrichment,
            gwas_fold=gwas_r.fold_enrichment,
            both_significant=rare_r.significant and gwas_r.significant,
            convergence_significant=combined_p < alpha,
            cstc_relevant=pw.cstc_relevant,
        ))

    convergent.sort(key=lambda x: x.combined_p)
    return convergent


# ── Output ──


def save_outputs(
    rare_results: list[EnrichmentResult],
    gwas_results: list[EnrichmentResult],
    convergent: list[ConvergenceResult],
    output_dir: Path,
) -> None:
    """Save Phase 4 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ──
    json_path = output_dir / "phase4_pathway_convergence.json"
    n_convergent_sig = sum(1 for c in convergent if c.convergence_significant)
    n_both_sig = sum(1 for c in convergent if c.both_significant)
    cstc_convergent = [c for c in convergent if c.cstc_relevant and c.convergence_significant]

    output = {
        "metadata": {
            "description": "Phase 4: Pathway convergence analysis for TS gene sets",
            "project": "ts-rare-variant-convergence",
            "phase": "4",
            "databases": ["GO_BP", "KEGG", "Reactome"],
            "enrichment_test": "hypergeometric",
            "convergence_test": "Fisher's combined probability",
            "convergence_threshold": 0.01,
            "n_pathways_tested": len(CURATED_PATHWAYS),
        },
        "summary": {
            "rare_enriched": sum(1 for r in rare_results if r.significant),
            "gwas_enriched": sum(1 for r in gwas_results if r.significant),
            "convergent_pathways": n_convergent_sig,
            "both_independently_significant": n_both_sig,
            "cstc_convergent": len(cstc_convergent),
            "target_met": n_convergent_sig >= 3,
        },
        "rare_variant_enrichment": [asdict(r) for r in rare_results],
        "gwas_enrichment": [asdict(r) for r in gwas_results],
        "convergence_results": [asdict(c) for c in convergent],
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved JSON: {json_path}")

    # ── CSV: convergence results ──
    csv_path = output_dir / "phase4_convergence.csv"
    fieldnames = [
        "pathway_id", "pathway_name", "source", "rare_p", "gwas_p",
        "combined_p", "rare_genes", "gwas_genes", "rare_fold", "gwas_fold",
        "both_significant", "convergence_significant", "cstc_relevant",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in convergent:
            row = asdict(c)
            row["rare_genes"] = ";".join(row["rare_genes"])
            row["gwas_genes"] = ";".join(row["gwas_genes"])
            writer.writerow(row)
    print(f"  Saved CSV:  {csv_path}")

    # ── Text report ──
    report_path = output_dir / "phase4_report.txt"
    with open(report_path, "w") as f:
        f.write("TS Pathway Convergence Report — Phase 4\n")
        f.write("=" * 48 + "\n\n")

        f.write("## Summary\n")
        f.write(f"  Pathways tested: {len(CURATED_PATHWAYS)}\n")
        f.write(f"  Rare variant enriched (FDR<0.05): "
                f"{sum(1 for r in rare_results if r.significant)}\n")
        f.write(f"  GWAS enriched (FDR<0.05): "
                f"{sum(1 for r in gwas_results if r.significant)}\n")
        f.write(f"  Convergent (combined p<0.01): {n_convergent_sig}\n")
        f.write(f"  Both independently significant: {n_both_sig}\n")
        f.write(f"  CSTC-relevant convergent: {len(cstc_convergent)}\n")
        target = "YES" if n_convergent_sig >= 3 else "NO"
        f.write(f"  Target met (>=3 convergent): {target}\n\n")

        f.write("## Convergent Pathways (ranked by combined p-value)\n")
        for c in convergent:
            sig = "***" if c.convergence_significant else ""
            both = " [BOTH SIG]" if c.both_significant else ""
            cstc = " [CSTC]" if c.cstc_relevant else ""
            f.write(
                f"\n  {sig}{c.pathway_name}{sig}{both}{cstc}\n"
                f"    ID: {c.pathway_id} ({c.source})\n"
                f"    Combined p: {c.combined_p:.2e}\n"
                f"    Rare: p={c.rare_p:.2e}, fold={c.rare_fold}x, "
                f"genes={','.join(c.rare_genes)}\n"
                f"    GWAS: p={c.gwas_p:.2e}, fold={c.gwas_fold}x, "
                f"genes={','.join(c.gwas_genes)}\n"
            )

        f.write("\n## CSTC-Relevant Convergent Pathways\n")
        if cstc_convergent:
            for c in cstc_convergent:
                all_genes = sorted(set(c.rare_genes + c.gwas_genes))
                f.write(f"  {c.pathway_name}: {', '.join(all_genes)}\n")
        else:
            f.write("  None met significance threshold.\n")

        f.write("\n## Rare Variant Enrichment (top 10)\n")
        f.write(f"{'Pathway':<45} {'p-value':>10} {'Fold':>6} {'Sig':>5} "
                f"{'Genes':>5}\n")
        f.write("-" * 75 + "\n")
        for r in rare_results[:10]:
            sig = "**" if r.significant else ""
            f.write(
                f"{r.pathway_name[:44]:<45} {r.p_value:>10.2e} "
                f"{r.fold_enrichment:>6.1f} {sig:>5} {r.n_genes:>5}\n"
            )

        f.write("\n## GWAS Enrichment (top 10)\n")
        f.write(f"{'Pathway':<45} {'p-value':>10} {'Fold':>6} {'Sig':>5} "
                f"{'Genes':>5}\n")
        f.write("-" * 75 + "\n")
        for r in gwas_results[:10]:
            sig = "**" if r.significant else ""
            f.write(
                f"{r.pathway_name[:44]:<45} {r.p_value:>10.2e} "
                f"{r.fold_enrichment:>6.1f} {sig:>5} {r.n_genes:>5}\n"
            )

    print(f"  Saved report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Pathway convergence analysis (GO/KEGG/Reactome)"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help="Output directory for results",
    )
    args = parser.parse_args()

    print("Phase 4: Pathway convergence analysis...")

    # Run enrichment for each gene set
    print(f"  Testing enrichment for rare variant genes ({len(RARE_VARIANT_GENES)})...")
    rare_results = run_enrichment(RARE_VARIANT_GENES, CURATED_PATHWAYS)
    n_rare_sig = sum(1 for r in rare_results if r.significant)
    print(f"    {n_rare_sig}/{len(rare_results)} pathways enriched (FDR<0.05)")

    print(f"  Testing enrichment for GWAS genes ({len(GWAS_GENES)})...")
    gwas_results = run_enrichment(GWAS_GENES, CURATED_PATHWAYS)
    n_gwas_sig = sum(1 for r in gwas_results if r.significant)
    print(f"    {n_gwas_sig}/{len(gwas_results)} pathways enriched (FDR<0.05)")

    # Convergence analysis
    print("  Running Fisher's combined probability convergence test...")
    convergent = convergence_analysis(
        rare_results, gwas_results, CURATED_PATHWAYS,
    )
    n_sig = sum(1 for c in convergent if c.convergence_significant)
    print(f"    {n_sig} convergent pathways (combined p<0.01)")

    # Print top convergent pathways
    for c in convergent[:5]:
        sig = "***" if c.convergence_significant else ""
        cstc = " [CSTC]" if c.cstc_relevant else ""
        print(f"    {sig}{c.pathway_name}{sig}{cstc}: "
              f"combined p={c.combined_p:.2e}")

    target = n_sig >= 3
    print(f"\n  Target (>=3 convergent pathways): "
          f"{'MET' if target else 'NOT MET'} ({n_sig})")

    # Save outputs
    save_outputs(rare_results, gwas_results, convergent, args.output)
    print("\n  Phase 4 complete.")


if __name__ == "__main__":
    main()
