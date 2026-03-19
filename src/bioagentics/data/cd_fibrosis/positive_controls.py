"""Positive control validation for CD fibrosis drug repurposing pipeline.

Verifies that compounds with known anti-fibrotic activity in CD preclinical
models rank highly in CMAP/iLINCS connectivity results. If positive controls
do not rank in the top 20%, the fibrosis signatures should be re-evaluated.

Positive controls:
- Obefazimod (miR-124 enhancer) — ECCO 2026 preclinical anti-fibrotic
- Anti-TL1A pathway (duvakitug, tulisokibart) — dual anti-inflammatory/anti-fibrotic
- Pirfenidone/nintedanib — approved for IPF, tested in IBD preclinical models
- CD38 inhibitors (daratumumab, isatuximab) — J Crohn's Colitis 2025
- JAK inhibitors (upadacitinib, tofacitinib) — JCC 2025 jjaf087

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.positive_controls --results path/to/cmap_hits.tsv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing"


@dataclass
class PositiveControl:
    """A compound with known anti-fibrotic activity in CD models."""

    name: str
    aliases: list[str]
    mechanism: str
    pathway: str
    evidence: str
    expected_rank_percentile: float = 0.20  # should be in top 20%


# ── Positive control compound definitions ──

POSITIVE_CONTROLS: list[PositiveControl] = [
    PositiveControl(
        name="pirfenidone",
        aliases=["pirfenidone", "esbriet", "pirespa"],
        mechanism="TGF-beta/TNF-alpha inhibition, collagen synthesis reduction",
        pathway="TGF-beta",
        evidence="Approved for IPF. Reduced intestinal fibrosis in TNBS mouse model. "
        "Inhibits collagen I/III, TGF-beta1, CTGF in intestinal fibroblasts.",
    ),
    PositiveControl(
        name="nintedanib",
        aliases=["nintedanib", "ofev", "bibf-1120", "bibf1120"],
        mechanism="Triple angiokinase inhibitor (VEGFR/FGFR/PDGFR)",
        pathway="RTK/PDGFR",
        evidence="Approved for IPF. Reduced intestinal fibrosis in chronic DSS model. "
        "Inhibits fibroblast proliferation and collagen deposition.",
    ),
    PositiveControl(
        name="obefazimod",
        aliases=["obefazimod", "abx464"],
        mechanism="miR-124 enhancer, anti-inflammatory and anti-fibrotic",
        pathway="miR-124/epigenetic",
        evidence="ECCO 2026: reduced COL1A1, ACTA2, FN1 expression in human intestinal "
        "fibroblasts stimulated with TGF-beta. Phase 3 in UC.",
    ),
    # Anti-TL1A pathway compounds
    PositiveControl(
        name="duvakitug",
        aliases=["duvakitug", "pf-07321332-tl1a", "pra023"],
        mechanism="Anti-TL1A monoclonal antibody",
        pathway="TL1A-DR3",
        evidence="APOLLO-CD Phase 2b: 55% endoscopic response at 900mg (44 weeks). "
        "Inhibits TL1A/DR3 signaling, reduces fibroblast activation and ECM deposition.",
    ),
    PositiveControl(
        name="tulisokibart",
        aliases=["tulisokibart", "pf-06480605"],
        mechanism="Anti-TL1A monoclonal antibody",
        pathway="TL1A-DR3",
        evidence="ARES-CD Phase 3 ongoing. RNA-seq showed Rho GTPase signaling as major "
        "TL1A-activated fibroblast pathway (PMID 40456235).",
    ),
    # CD38 inhibitors
    PositiveControl(
        name="daratumumab",
        aliases=["daratumumab", "darzalex"],
        mechanism="Anti-CD38 monoclonal antibody",
        pathway="CD38/NAD+",
        evidence="J Crohn's Colitis 2025: CD38 inhibition reduced fibrosis in chronic "
        "DSS mouse model. CD38/PECAM1 axis marks inflammation-to-fibrosis transition.",
    ),
    PositiveControl(
        name="isatuximab",
        aliases=["isatuximab", "sarclisa"],
        mechanism="Anti-CD38 monoclonal antibody",
        pathway="CD38/NAD+",
        evidence="Approved for myeloma. Targets same CD38 pathway as daratumumab. "
        "Potential CD fibrosis repurposing candidate.",
    ),
    # JAK inhibitors
    PositiveControl(
        name="upadacitinib",
        aliases=["upadacitinib", "rinvoq"],
        mechanism="Selective JAK1 inhibitor",
        pathway="JAK-STAT",
        evidence="JCC 2025 jjaf087: superior to tofacitinib for fibrosis modulation. "
        "Inhibits fibroblast JAK-STAT3 pathway. Approved for CD.",
    ),
    PositiveControl(
        name="tofacitinib",
        aliases=["tofacitinib", "xeljanz"],
        mechanism="Pan-JAK inhibitor (JAK1/JAK3)",
        pathway="JAK-STAT",
        evidence="JCC 2025 jjaf087: anti-fibrotic in intestinal fibroblasts via "
        "JAK-STAT3 inhibition. Less selective than upadacitinib.",
    ),
    # HDAC inhibitors (well-characterized in L1000, target HDAC1 in fibrosis)
    PositiveControl(
        name="vorinostat",
        aliases=["vorinostat", "saha", "zolinza", "suberoylanilide"],
        mechanism="Pan-HDAC inhibitor",
        pathway="epigenetic/HDAC",
        evidence="HDAC1 identified as druggable fibrostenotic target (IBD 2025, "
        "doi:10.1093/ibd/izae295). HDAC inhibitors reduce collagen synthesis. "
        "Well-represented in L1000 with strong negative connectivity to fibrosis.",
    ),
    PositiveControl(
        name="trichostatin-a",
        aliases=["trichostatin-a", "trichostatin a", "tsa"],
        mechanism="HDAC inhibitor (class I/II)",
        pathway="epigenetic/HDAC",
        evidence="Classic HDAC inhibitor. Anti-fibrotic in hepatic and pulmonary "
        "fibrosis models. Expected strong negative CMAP connectivity.",
    ),
    # ALK5/TGF-beta receptor inhibitor (ontunisertib — STENOVA Phase 2a)
    PositiveControl(
        name="ontunisertib",
        aliases=["ontunisertib", "ly-3200882", "ly3200882"],
        mechanism="Selective ALK5 (TGFBR1) kinase inhibitor",
        pathway="ALK5/TGF-beta",
        evidence="STENOVA Phase 2a met primary and key secondary endpoints in "
        "fibrostenotic CD. Directly targets ALK5/SMAD2/3 axis in fibroblasts.",
    ),
]

# Group positive controls by pathway class for validation reporting
PATHWAY_CLASSES = {
    "TGF-beta/anti-fibrotic": ["pirfenidone", "nintedanib"],
    "TL1A-DR3": ["duvakitug", "tulisokibart"],
    "CD38/NAD+": ["daratumumab", "isatuximab"],
    "JAK-STAT": ["upadacitinib", "tofacitinib"],
    "epigenetic/HDAC": ["vorinostat", "trichostatin-a"],
    "miR-124": ["obefazimod"],
    "ALK5/TGF-beta": ["ontunisertib"],
}

# TL1A benchmark: compounds that should rank in top 15% against the
# TL1A-DR3/Rho signature specifically (task #848)
TL1A_BENCHMARK_COMPOUNDS = ["duvakitug", "tulisokibart", "ontunisertib"]
TL1A_BENCHMARK_PERCENTILE = 0.15


def match_compound_name(
    compound: str,
    controls: list[PositiveControl] | None = None,
) -> PositiveControl | None:
    """Match a compound name from CMAP/iLINCS results to a positive control.

    Uses case-insensitive substring matching against aliases.
    """
    controls = controls or POSITIVE_CONTROLS
    compound_lower = compound.lower().strip()

    for ctrl in controls:
        for alias in ctrl.aliases:
            if alias in compound_lower or compound_lower in alias:
                return ctrl

    return None


def validate_positive_controls(
    ranked_results: pd.DataFrame,
    controls: list[PositiveControl] | None = None,
    compound_column: str = "compound",
    score_column: str = "mean_concordance",
) -> pd.DataFrame:
    """Check how positive controls rank in CMAP/iLINCS results.

    Args:
        ranked_results: DataFrame from rank_compounds_across_signatures,
            sorted by score (best first).
        controls: Positive control definitions (default: POSITIVE_CONTROLS).
        compound_column: Column name containing compound names.
        score_column: Column name containing concordance/connectivity scores.

    Returns:
        DataFrame with one row per positive control showing its rank,
        percentile, score, and whether it passes the threshold.
    """
    controls = controls or POSITIVE_CONTROLS
    total_compounds = len(ranked_results)

    if total_compounds == 0:
        return pd.DataFrame([
            {
                "control_name": ctrl.name,
                "mechanism": ctrl.mechanism,
                "pathway": ctrl.pathway,
                "matched_compound": None,
                "rank": None,
                "percentile": None,
                "score": None,
                "passes_threshold": False,
                "n_signatures_hit": None,
                "has_fibroblast_hit": None,
            }
            for ctrl in controls
        ])

    records = []
    for ctrl in controls:
        # Find the best-matching row for this control
        best_rank = None
        best_row = None

        for idx, (_, row) in enumerate(ranked_results.iterrows()):
            compound_name = str(row.get(compound_column, ""))
            if match_compound_name(compound_name, [ctrl]) is not None:
                if best_rank is None or idx < best_rank:
                    best_rank = idx
                    best_row = row

        if best_row is not None:
            rank = best_rank + 1  # 1-indexed
            percentile = rank / total_compounds
            score = best_row.get(score_column, None)
            passes = percentile <= ctrl.expected_rank_percentile

            records.append({
                "control_name": ctrl.name,
                "mechanism": ctrl.mechanism,
                "pathway": ctrl.pathway,
                "matched_compound": str(best_row.get(compound_column, "")),
                "rank": rank,
                "percentile": round(percentile, 4),
                "score": round(float(score), 4) if score is not None else None,
                "passes_threshold": passes,
                "n_signatures_hit": best_row.get("n_signatures_queried", None),
                "has_fibroblast_hit": best_row.get("has_fibroblast_hit", None),
            })
        else:
            records.append({
                "control_name": ctrl.name,
                "mechanism": ctrl.mechanism,
                "pathway": ctrl.pathway,
                "matched_compound": None,
                "rank": None,
                "percentile": None,
                "score": None,
                "passes_threshold": False,
                "n_signatures_hit": None,
                "has_fibroblast_hit": None,
            })

    return pd.DataFrame(records)


def validate_by_pathway_class(
    validation_df: pd.DataFrame,
) -> dict[str, dict]:
    """Assess validation results grouped by pathway class.

    Returns dict mapping pathway class to summary stats.
    """
    results = {}
    for pathway, compounds in PATHWAY_CLASSES.items():
        pathway_rows = validation_df[
            validation_df["control_name"].isin(compounds)
        ]
        found = pathway_rows[pathway_rows["matched_compound"].notna()]
        passing = pathway_rows[pathway_rows["passes_threshold"] == True]  # noqa: E712

        results[pathway] = {
            "compounds": compounds,
            "n_total": len(compounds),
            "n_found": len(found),
            "n_passing": len(passing),
            "best_percentile": (
                found["percentile"].min() if len(found) > 0 else None
            ),
            "class_passes": len(passing) > 0,
        }

    return results


def check_success_criteria(
    validation_df: pd.DataFrame,
    min_classes_passing: int = 3,
    min_total_classes: int = 4,
) -> dict:
    """Check whether positive control validation passes success criteria.

    From the research plan: "At least 3 of 4 positive controls (obefazimod,
    anti-TL1A, pirfenidone, CD38 inhibitors) rank in top 20% of CMAP hits"

    We interpret this as: at least 3 out of the specified pathway classes
    must have at least one compound passing the threshold.

    Args:
        validation_df: Output of validate_positive_controls.
        min_classes_passing: Minimum number of pathway classes that must pass.
        min_total_classes: Denominator for the ratio (default 4, matching plan).

    Returns:
        Dict with overall pass/fail and detailed pathway results.
    """
    # The 4 pathway classes specified in the plan
    plan_classes = [
        "miR-124",           # obefazimod
        "TL1A-DR3",          # anti-TL1A
        "TGF-beta/anti-fibrotic",  # pirfenidone/nintedanib
        "CD38/NAD+",         # CD38 inhibitors
    ]

    pathway_results = validate_by_pathway_class(validation_df)

    classes_passing = sum(
        1 for cls in plan_classes
        if cls in pathway_results and pathway_results[cls]["class_passes"]
    )

    overall_pass = classes_passing >= min_classes_passing

    return {
        "overall_pass": overall_pass,
        "classes_passing": classes_passing,
        "min_required": min_classes_passing,
        "total_classes": min_total_classes,
        "pathway_results": pathway_results,
    }


def generate_validation_report(
    ranked_results: pd.DataFrame,
    output_dir: Path | None = None,
    compound_column: str = "compound",
    score_column: str = "mean_concordance",
) -> tuple[pd.DataFrame, dict]:
    """Run full positive control validation and generate report.

    Args:
        ranked_results: DataFrame from the CMAP/iLINCS pipeline.
        output_dir: Directory to save report files.
        compound_column: Column with compound names.
        score_column: Column with concordance scores.

    Returns:
        Tuple of (validation_df, success_criteria_dict).
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Positive Control Validation Report")
    print("=" * 60)

    total = len(ranked_results)
    print(f"\n  Total compounds in results: {total}")

    # Run validation
    validation_df = validate_positive_controls(
        ranked_results,
        compound_column=compound_column,
        score_column=score_column,
    )

    # Print per-compound results
    print(f"\n  Individual Control Results:")
    print(f"  {'Control':20s} {'Pathway':22s} {'Rank':>6s} {'%ile':>8s} {'Score':>10s} {'Pass':>5s}")
    print(f"  {'-'*20} {'-'*22} {'-'*6} {'-'*8} {'-'*10} {'-'*5}")

    for _, row in validation_df.iterrows():
        rank_str = str(row["rank"]) if row["rank"] is not None else "N/F"
        pct_str = f"{row['percentile']:.1%}" if row["percentile"] is not None else "N/F"
        score_str = f"{row['score']:+.4f}" if row["score"] is not None else "N/F"
        pass_str = "YES" if row["passes_threshold"] else "NO"

        print(f"  {row['control_name']:20s} {row['pathway']:22s} "
              f"{rank_str:>6s} {pct_str:>8s} {score_str:>10s} {pass_str:>5s}")

    # Pathway class summary
    criteria = check_success_criteria(validation_df)

    print(f"\n  Pathway Class Summary:")
    for pathway, result in criteria["pathway_results"].items():
        status = "PASS" if result["class_passes"] else "FAIL"
        best = (
            f"best={result['best_percentile']:.1%}"
            if result["best_percentile"] is not None
            else "not found"
        )
        print(f"    {pathway:25s} {result['n_passing']}/{result['n_total']} passing  "
              f"({best})  [{status}]")

    # Overall verdict
    print(f"\n  SUCCESS CRITERIA: {criteria['classes_passing']}/{criteria['total_classes']} "
          f"pathway classes pass (need >= {criteria['min_required']})")
    print(f"  VERDICT: {'PASS' if criteria['overall_pass'] else 'FAIL'}")

    # Save results
    val_path = output_dir / "positive_control_validation.tsv"
    validation_df.to_csv(val_path, sep="\t", index=False)
    print(f"\n  Saved: {val_path}")

    return validation_df, criteria


def validate_tl1a_benchmark(
    tl1a_signature_results: pd.DataFrame,
    compound_column: str = "compound",
    score_column: str = "concordance",
    benchmark_percentile: float | None = None,
) -> dict:
    """Validate TL1A pathway compounds against TL1A-DR3/Rho signature results.

    TL1A inhibitors (duvakitug, tulisokibart) and ALK5 inhibitors (ontunisertib)
    have strong clinical validation. If our scoring correctly identifies these
    known modulators against the TL1A-DR3/Rho signature, it validates the
    computational approach.

    Args:
        tl1a_signature_results: CMAP/iLINCS results for the TL1A-DR3/Rho
            signature query specifically (single signature results, not
            the aggregate). Should be sorted by score (most negative first).
        compound_column: Column name for compound names.
        score_column: Column name for concordance/connectivity scores.
        benchmark_percentile: Required ranking percentile (default: 0.15 = top 15%).

    Returns:
        Dict with benchmark pass/fail and per-compound results.
    """
    threshold = benchmark_percentile or TL1A_BENCHMARK_PERCENTILE
    total = len(tl1a_signature_results)

    benchmark_controls = [
        ctrl for ctrl in POSITIVE_CONTROLS
        if ctrl.name in TL1A_BENCHMARK_COMPOUNDS
    ]

    compound_results = []
    for ctrl in benchmark_controls:
        best_rank = None
        best_score = None
        matched_name = None

        for idx, (_, row) in enumerate(tl1a_signature_results.iterrows()):
            name = str(row.get(compound_column, ""))
            if match_compound_name(name, [ctrl]) is not None:
                if best_rank is None or idx < best_rank:
                    best_rank = idx
                    best_score = float(row.get(score_column, 0))
                    matched_name = name

        if best_rank is not None:
            rank = best_rank + 1
            percentile = rank / total if total > 0 else 1.0
            passes = percentile <= threshold
        else:
            rank = None
            percentile = None
            passes = False

        compound_results.append({
            "control_name": ctrl.name,
            "mechanism": ctrl.mechanism,
            "pathway": ctrl.pathway,
            "matched_compound": matched_name,
            "rank": rank,
            "percentile": round(percentile, 4) if percentile is not None else None,
            "score": round(best_score, 4) if best_score is not None else None,
            "passes_threshold": passes,
        })

    n_passing = sum(1 for r in compound_results if r["passes_threshold"])
    n_found = sum(1 for r in compound_results if r["matched_compound"] is not None)

    return {
        "benchmark_pass": n_passing > 0,
        "n_compounds": len(compound_results),
        "n_found": n_found,
        "n_passing": n_passing,
        "threshold_percentile": threshold,
        "total_compounds_in_results": total,
        "compound_results": compound_results,
    }


def generate_tl1a_benchmark_report(
    tl1a_signature_results: pd.DataFrame,
    output_dir: Path | None = None,
    compound_column: str = "compound",
    score_column: str = "concordance",
) -> dict:
    """Run TL1A benchmark validation and print report.

    Args:
        tl1a_signature_results: Results from querying the TL1A-DR3/Rho
            signature against CMAP/iLINCS.
        output_dir: Directory to save results.
        compound_column: Column with compound names.
        score_column: Column with concordance scores.

    Returns:
        Benchmark results dict.
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("TL1A Pathway Benchmark Validation")
    print("=" * 60)

    total = len(tl1a_signature_results)
    print(f"\n  TL1A-DR3/Rho signature results: {total} compounds")

    results = validate_tl1a_benchmark(
        tl1a_signature_results,
        compound_column=compound_column,
        score_column=score_column,
    )

    print(f"  Benchmark threshold: top {results['threshold_percentile']:.0%}")
    print(f"\n  {'Compound':20s} {'Mechanism':35s} {'Rank':>6s} {'%ile':>8s} {'Score':>10s} {'Pass':>5s}")
    print(f"  {'-'*20} {'-'*35} {'-'*6} {'-'*8} {'-'*10} {'-'*5}")

    for r in results["compound_results"]:
        rank_str = str(r["rank"]) if r["rank"] is not None else "N/F"
        pct_str = f"{r['percentile']:.1%}" if r["percentile"] is not None else "N/F"
        score_str = f"{r['score']:+.4f}" if r["score"] is not None else "N/F"
        pass_str = "YES" if r["passes_threshold"] else "NO"

        print(f"  {r['control_name']:20s} {r['mechanism']:35s} "
              f"{rank_str:>6s} {pct_str:>8s} {score_str:>10s} {pass_str:>5s}")

    print(f"\n  BENCHMARK: {results['n_passing']}/{results['n_compounds']} "
          f"TL1A-class compounds in top {results['threshold_percentile']:.0%}")
    print(f"  VERDICT: {'PASS' if results['benchmark_pass'] else 'FAIL'}")

    if not results["benchmark_pass"] and results["n_found"] == 0:
        print("\n  NOTE: No TL1A-class compounds found in results.")
        print("  These biologics may not have L1000 signatures.")
        print("  Consider using gene-level surrogate matching instead.")

    # Save benchmark results
    if results["compound_results"]:
        bench_df = pd.DataFrame(results["compound_results"])
        bench_path = output_dir / "tl1a_benchmark_validation.tsv"
        bench_df.to_csv(bench_path, sep="\t", index=False)
        print(f"\n  Saved: {bench_path}")

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate positive controls in CMAP/iLINCS results"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to ranked CMAP hits TSV (output of cmap_pipeline)",
    )
    parser.add_argument(
        "--compound-column",
        default="compound",
        help="Column name for compound names (default: compound)",
    )
    parser.add_argument(
        "--score-column",
        default="mean_concordance",
        help="Column name for scores (default: mean_concordance)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        return

    ranked = pd.read_csv(args.results, sep="\t")
    print(f"Loaded {len(ranked)} compounds from {args.results}")

    generate_validation_report(
        ranked,
        output_dir=args.output_dir,
        compound_column=args.compound_column,
        score_column=args.score_column,
    )


if __name__ == "__main__":
    main()
