"""Phase 5: Cross-disorder specificity analysis.

Compares TS convergent pathways (from Phase 4) with convergent pathways from
autism (ASD), OCD, and ADHD to identify TS-specific pathways vs shared
neurodevelopmental pathways.

Approach:
1. Define rare variant + GWAS gene sets for ASD, OCD, ADHD
2. Run pathway enrichment and convergence for each disorder using the same
   curated pathway database as Phase 4 (consistent method)
3. Compute specificity scores: for each TS convergent pathway, measure
   how much stronger the TS convergence is relative to other disorders
4. Classify pathways as TS-specific, shared, or NDD-general

Success criterion: >=1 TS-specific convergent pathway distinguishable from
autism/OCD/ADHD (plan success criteria #5).

Output: data/results/ts-rare-variant-convergence/phase5/

Usage:
    uv run python -m bioagentics.tourettes.rare_variant_convergence.phase5_cross_disorder
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bioagentics.config import REPO_ROOT
from bioagentics.tourettes.rare_variant_convergence.phase4_pathway_convergence import (
    CURATED_PATHWAYS,
    GWAS_GENES,
    RARE_VARIANT_GENES,
    ConvergenceResult,
    convergence_analysis,
    run_enrichment,
)

OUTPUT_DIR = REPO_ROOT / "data" / "results" / "ts-rare-variant-convergence" / "phase5"


# ── Cross-disorder gene sets ──
# Curated from SFARI Gene, published GWAS, and rare variant studies.
# Focused on genes that overlap the same pathway space to enable
# fair comparison with TS convergence results.

# ASD: SFARI Gene (high-confidence, Category 1-2) + ASD GWAS (PGC)
ASD_RARE_GENES = [
    "CHD8", "SCN2A", "SYNGAP1", "ADNP", "DYRK1A",
    "PTEN", "SHANK3", "ANK2", "ASH1L", "CHD2",
    "KMT2C", "NRXN1", "FOXP1", "GRIN2B", "KMT5B",
    "TBR1", "TCF4", "ARID1B", "SETD5", "MED13L",
]

ASD_GWAS_GENES = [
    "KMT2E", "MACROD2", "XRN2", "NEGR1", "PTBP2",
    "CADPS", "KMT2A", "NR2F1", "RBFOX1", "MEF2C",
]

# OCD: rare variant candidates + OCD GWAS
OCD_RARE_GENES = [
    "SLITRK5", "DLGAP1", "SAPAP3", "HTR2A", "SLC1A1",
    "NRXN1", "CNTNAP2", "CHD8", "SETD5", "ARID1B",
]

OCD_GWAS_GENES = [
    "PTPRD", "GRID2", "DLGAP1", "RSRC1", "WDR7",
    "MEF2C", "RBFOX1", "NR2F1", "KIT", "NRXN3",
]

# ADHD: rare variant candidates + ADHD GWAS (Demontis et al. 2023)
ADHD_RARE_GENES = [
    "NRXN1", "CNTNAP2", "GRM5", "NOS1", "SLC6A3",
    "DRD4", "DRD5", "SNAP25", "CDH13", "LPHN3",
]

ADHD_GWAS_GENES = [
    "FOXP2", "DUSP6", "SEMA6D", "ST3GAL3", "SORCS3",
    "MEF2C", "RBFOX1", "PTPRF", "PCDH7", "LINC00461",
]


@dataclass
class DisorderConvergence:
    """Convergence results for a single disorder."""

    disorder: str
    rare_genes: list[str]
    gwas_genes: list[str]
    convergence_results: list[ConvergenceResult]
    n_convergent: int  # pathways with combined_p < 0.01


@dataclass
class SpecificityResult:
    """Specificity score for a TS convergent pathway."""

    pathway_id: str
    pathway_name: str
    source: str
    ts_combined_p: float
    asd_combined_p: float | None
    ocd_combined_p: float | None
    adhd_combined_p: float | None
    specificity_score: float
    classification: str  # ts_specific, shared_ndd, disorder_pair
    shared_with: list[str]  # which disorders also show convergence
    ts_rare_genes: list[str]
    ts_gwas_genes: list[str]
    cstc_relevant: bool


def run_disorder_convergence(
    rare_genes: list[str],
    gwas_genes: list[str],
    disorder: str,
) -> DisorderConvergence:
    """Run pathway enrichment + convergence for a disorder's gene sets."""
    rare_results = run_enrichment(rare_genes, CURATED_PATHWAYS)
    gwas_results = run_enrichment(gwas_genes, CURATED_PATHWAYS)
    convergent = convergence_analysis(
        rare_results, gwas_results, CURATED_PATHWAYS, alpha=0.01,
    )
    n_sig = sum(1 for c in convergent if c.convergence_significant)
    return DisorderConvergence(
        disorder=disorder,
        rare_genes=rare_genes,
        gwas_genes=gwas_genes,
        convergence_results=convergent,
        n_convergent=n_sig,
    )


def compute_specificity(
    ts_convergent: list[ConvergenceResult],
    disorder_results: dict[str, DisorderConvergence],
    alpha: float = 0.01,
) -> list[SpecificityResult]:
    """Compute specificity scores for each TS convergent pathway.

    Specificity score = -log10(TS combined p) - mean(-log10(other disorder p))
    Higher score = more TS-specific.

    Classification:
    - ts_specific: pathway significant in TS but NOT in any other disorder
    - shared_ndd: pathway significant in TS and >=2 other disorders
    - disorder_pair: pathway significant in TS and exactly 1 other disorder
    """
    results = []

    for ts_c in ts_convergent:
        if not ts_c.convergence_significant:
            continue

        # Look up this pathway in other disorders
        other_ps: dict[str, float | None] = {}
        shared_with = []

        for disorder, dc in disorder_results.items():
            match = None
            for c in dc.convergence_results:
                if c.pathway_id == ts_c.pathway_id:
                    match = c
                    break
            if match is not None:
                other_ps[disorder] = match.combined_p
                if match.combined_p < alpha:
                    shared_with.append(disorder)
            else:
                other_ps[disorder] = None

        # Specificity score
        ts_score = -np.log10(max(ts_c.combined_p, 1e-300))
        other_scores = []
        for p in other_ps.values():
            if p is not None:
                other_scores.append(-np.log10(max(p, 1e-300)))
            else:
                other_scores.append(0.0)  # no enrichment = no signal

        mean_other = np.mean(other_scores) if other_scores else 0.0
        specificity = float(ts_score - mean_other)

        # Classification
        if len(shared_with) == 0:
            classification = "ts_specific"
        elif len(shared_with) >= 2:
            classification = "shared_ndd"
        else:
            classification = "disorder_pair"

        results.append(SpecificityResult(
            pathway_id=ts_c.pathway_id,
            pathway_name=ts_c.pathway_name,
            source=ts_c.source,
            ts_combined_p=ts_c.combined_p,
            asd_combined_p=other_ps.get("ASD"),
            ocd_combined_p=other_ps.get("OCD"),
            adhd_combined_p=other_ps.get("ADHD"),
            specificity_score=round(specificity, 3),
            classification=classification,
            shared_with=shared_with,
            ts_rare_genes=ts_c.rare_genes,
            ts_gwas_genes=ts_c.gwas_genes,
            cstc_relevant=ts_c.cstc_relevant,
        ))

    results.sort(key=lambda x: -x.specificity_score)
    return results


def save_outputs(
    ts_convergent: list[ConvergenceResult],
    disorder_results: dict[str, DisorderConvergence],
    specificity: list[SpecificityResult],
    output_dir: Path,
) -> None:
    """Save Phase 5 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_ts_specific = sum(1 for s in specificity if s.classification == "ts_specific")
    n_shared = sum(1 for s in specificity if s.classification == "shared_ndd")
    n_pair = sum(1 for s in specificity if s.classification == "disorder_pair")

    # ── JSON ──
    json_path = output_dir / "phase5_cross_disorder.json"
    output = {
        "metadata": {
            "description": "Phase 5: Cross-disorder specificity analysis",
            "project": "ts-rare-variant-convergence",
            "phase": "5",
            "disorders_compared": ["ASD", "OCD", "ADHD"],
            "convergence_threshold": 0.01,
            "specificity_method": "-log10(TS_p) - mean(-log10(other_p))",
        },
        "summary": {
            "ts_convergent_pathways": sum(
                1 for c in ts_convergent if c.convergence_significant
            ),
            "ts_specific": n_ts_specific,
            "shared_ndd": n_shared,
            "disorder_pair": n_pair,
            "target_met": n_ts_specific >= 1,
            "disorder_convergent_counts": {
                d: dc.n_convergent for d, dc in disorder_results.items()
            },
        },
        "specificity_results": [asdict(s) for s in specificity],
        "disorder_summaries": {
            d: {
                "rare_genes": dc.rare_genes,
                "gwas_genes": dc.gwas_genes,
                "n_convergent": dc.n_convergent,
                "convergent_pathways": [
                    {"pathway_id": c.pathway_id, "pathway_name": c.pathway_name,
                     "combined_p": c.combined_p}
                    for c in dc.convergence_results if c.convergence_significant
                ],
            }
            for d, dc in disorder_results.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved JSON: {json_path}")

    # ── CSV ──
    csv_path = output_dir / "phase5_specificity.csv"
    fieldnames = [
        "pathway_id", "pathway_name", "source",
        "ts_combined_p", "asd_combined_p", "ocd_combined_p", "adhd_combined_p",
        "specificity_score", "classification", "shared_with",
        "ts_rare_genes", "ts_gwas_genes", "cstc_relevant",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in specificity:
            row = asdict(s)
            row["shared_with"] = ";".join(row["shared_with"])
            row["ts_rare_genes"] = ";".join(row["ts_rare_genes"])
            row["ts_gwas_genes"] = ";".join(row["ts_gwas_genes"])
            writer.writerow(row)
    print(f"  Saved CSV:  {csv_path}")

    # ── Text report ──
    report_path = output_dir / "phase5_report.txt"
    with open(report_path, "w") as f:
        f.write("TS Cross-Disorder Specificity Report — Phase 5\n")
        f.write("=" * 52 + "\n\n")

        f.write("## Summary\n")
        f.write(f"  TS convergent pathways (from Phase 4): "
                f"{sum(1 for c in ts_convergent if c.convergence_significant)}\n")
        f.write(f"  TS-specific: {n_ts_specific}\n")
        f.write(f"  Shared NDD (>=2 disorders): {n_shared}\n")
        f.write(f"  Disorder pair: {n_pair}\n")
        target = "YES" if n_ts_specific >= 1 else "NO"
        f.write(f"  Target met (>=1 TS-specific): {target}\n\n")

        f.write("## Cross-Disorder Convergence Counts\n")
        f.write(f"  TS:   {sum(1 for c in ts_convergent if c.convergence_significant)} "
                f"convergent pathways\n")
        for d, dc in disorder_results.items():
            f.write(f"  {d}: {dc.n_convergent} convergent pathways\n")

        f.write("\n## TS-Specific Pathways\n")
        ts_specific = [s for s in specificity if s.classification == "ts_specific"]
        if ts_specific:
            for s in ts_specific:
                f.write(f"\n  *** {s.pathway_name} ***\n")
                f.write(f"    ID: {s.pathway_id} ({s.source})\n")
                f.write(f"    TS combined p: {s.ts_combined_p:.2e}\n")
                f.write(f"    Specificity score: {s.specificity_score:.3f}\n")
                f.write(f"    CSTC-relevant: {s.cstc_relevant}\n")
                f.write(f"    TS rare genes: {', '.join(s.ts_rare_genes)}\n")
                f.write(f"    TS GWAS genes: {', '.join(s.ts_gwas_genes)}\n")
                asd_str = f"{s.asd_combined_p:.2e}" if s.asd_combined_p is not None else "n/a"
                ocd_str = f"{s.ocd_combined_p:.2e}" if s.ocd_combined_p is not None else "n/a"
                adhd_str = f"{s.adhd_combined_p:.2e}" if s.adhd_combined_p is not None else "n/a"
                f.write(f"    ASD p: {asd_str} | OCD p: {ocd_str} | ADHD p: {adhd_str}\n")
        else:
            f.write("  None identified.\n")

        f.write("\n## Shared Neurodevelopmental Pathways\n")
        shared = [s for s in specificity if s.classification in ("shared_ndd", "disorder_pair")]
        if shared:
            for s in shared:
                f.write(f"\n  {s.pathway_name} [{s.classification}]\n")
                f.write(f"    Shared with: {', '.join(s.shared_with)}\n")
                f.write(f"    TS p: {s.ts_combined_p:.2e} | "
                        f"Specificity: {s.specificity_score:.3f}\n")
        else:
            f.write("  None identified.\n")

        f.write("\n## All Pathways Ranked by Specificity\n")
        f.write(f"{'Pathway':<40} {'TS p':>10} {'Spec':>7} {'Class':>15} "
                f"{'CSTC':>5}\n")
        f.write("-" * 80 + "\n")
        for s in specificity:
            cstc = "yes" if s.cstc_relevant else ""
            f.write(
                f"{s.pathway_name[:39]:<40} {s.ts_combined_p:>10.2e} "
                f"{s.specificity_score:>7.3f} {s.classification:>15} "
                f"{cstc:>5}\n"
            )

    print(f"  Saved report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Cross-disorder specificity analysis"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help="Output directory for results",
    )
    args = parser.parse_args()

    print("Phase 5: Cross-disorder specificity analysis...")

    # Step 1: Run TS convergence (re-run Phase 4 logic)
    print("  Running TS pathway convergence...")
    ts_rare_results = run_enrichment(RARE_VARIANT_GENES, CURATED_PATHWAYS)
    ts_gwas_results = run_enrichment(GWAS_GENES, CURATED_PATHWAYS)
    ts_convergent = convergence_analysis(
        ts_rare_results, ts_gwas_results, CURATED_PATHWAYS,
    )
    n_ts = sum(1 for c in ts_convergent if c.convergence_significant)
    print(f"    TS: {n_ts} convergent pathways")

    # Step 2: Run convergence for each comparison disorder
    print("  Running cross-disorder convergence...")
    disorder_results = {}

    print("    ASD...")
    disorder_results["ASD"] = run_disorder_convergence(
        ASD_RARE_GENES, ASD_GWAS_GENES, "ASD",
    )
    print(f"      {disorder_results['ASD'].n_convergent} convergent pathways")

    print("    OCD...")
    disorder_results["OCD"] = run_disorder_convergence(
        OCD_RARE_GENES, OCD_GWAS_GENES, "OCD",
    )
    print(f"      {disorder_results['OCD'].n_convergent} convergent pathways")

    print("    ADHD...")
    disorder_results["ADHD"] = run_disorder_convergence(
        ADHD_RARE_GENES, ADHD_GWAS_GENES, "ADHD",
    )
    print(f"      {disorder_results['ADHD'].n_convergent} convergent pathways")

    # Step 3: Compute specificity
    print("  Computing specificity scores...")
    specificity = compute_specificity(ts_convergent, disorder_results)

    n_ts_specific = sum(1 for s in specificity if s.classification == "ts_specific")
    n_shared = sum(1 for s in specificity if s.classification == "shared_ndd")
    n_pair = sum(1 for s in specificity if s.classification == "disorder_pair")

    print(f"    TS-specific: {n_ts_specific}")
    print(f"    Shared NDD: {n_shared}")
    print(f"    Disorder pair: {n_pair}")

    # Top TS-specific pathways
    ts_spec = [s for s in specificity if s.classification == "ts_specific"]
    if ts_spec:
        print("\n  Top TS-specific pathways:")
        for s in ts_spec[:5]:
            cstc = " [CSTC]" if s.cstc_relevant else ""
            print(f"    *** {s.pathway_name}{cstc}: "
                  f"specificity={s.specificity_score:.3f}, "
                  f"TS p={s.ts_combined_p:.2e}")

    target = n_ts_specific >= 1
    print(f"\n  Target (>=1 TS-specific pathway): "
          f"{'MET' if target else 'NOT MET'} ({n_ts_specific})")

    # Save outputs
    save_outputs(ts_convergent, disorder_results, specificity, args.output)
    print("\n  Phase 5 complete.")


if __name__ == "__main__":
    main()
