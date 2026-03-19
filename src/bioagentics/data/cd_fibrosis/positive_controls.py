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
]

# Group positive controls by pathway class for validation reporting
PATHWAY_CLASSES = {
    "TGF-beta/anti-fibrotic": ["pirfenidone", "nintedanib"],
    "TL1A-DR3": ["duvakitug", "tulisokibart"],
    "CD38/NAD+": ["daratumumab", "isatuximab"],
    "JAK-STAT": ["upadacitinib", "tofacitinib"],
    "epigenetic/HDAC": ["vorinostat", "trichostatin-a"],
    "miR-124": ["obefazimod"],
}


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
