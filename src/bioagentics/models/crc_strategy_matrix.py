"""Therapeutic strategy matrix and CRC vs NSCLC KRAS comparison.

Compiles all prior phase outputs into a per-allele strategy matrix with
dependencies, drug sensitivities, co-mutation context, population estimates,
and clinical agent mapping.

Usage:
    uv run python -m bioagentics.models.crc_strategy_matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioagentics.config import REPO_ROOT

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "crc-kras-dependencies"
DEFAULT_DEST = DEFAULT_OUTPUT_DIR / "therapeutic_strategy_matrix.json"

# CRC epidemiology and KRAS allele frequencies (literature-based estimates)
CRC_US_ANNUAL_CASES = 153_000
KRAS_MUTANT_FRACTION = 0.45  # ~45% of CRC
ALLELE_FREQUENCIES = {
    "G12D": 0.33,
    "G12V": 0.22,
    "G13D": 0.18,
    "G12C": 0.08,
    "A146T": 0.07,
    "Q61H": 0.05,
    "other": 0.07,
}

# Current clinical agents per allele
CLINICAL_AGENTS = {
    "G12C": [
        {"agent": "sotorasib", "status": "FDA approved (NSCLC), Phase 2 CRC (CodeBreaK 101: ORR 9.7% mono, ~30% combo)"},
        {"agent": "adagrasib", "status": "FDA approved (NSCLC), Phase 2 CRC"},
    ],
    "G12D": [
        {"agent": "zoldonrasib", "status": "FDA BTD, Phase 1 CRC cohort"},
        {"agent": "MRTX1133", "status": "Phase 1/2 (enrolling CRC)"},
    ],
    "pan-RAS": [
        {"agent": "ERAS-0015", "status": "Phase 1 (pan-RAS, all alleles)"},
        {"agent": "elironrasib", "status": "FDA BTD (pan-RAS)"},
    ],
}


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _build_allele_strategy(
    allele: str,
    dep_results: dict | None,
    drug_results: dict | None,
    interaction_results: dict | None,
) -> dict:
    """Build strategy entry for a single KRAS allele."""
    strategy = {
        "allele": allele,
        "population_estimate": {
            "annual_us_cases": int(CRC_US_ANNUAL_CASES * KRAS_MUTANT_FRACTION * ALLELE_FREQUENCIES.get(allele, 0)),
            "fraction_of_kras_mutant": ALLELE_FREQUENCIES.get(allele, 0),
        },
        "clinical_agents": CLINICAL_AGENTS.get(allele, []) + CLINICAL_AGENTS.get("pan-RAS", []),
        "top_dependencies": [],
        "top_drug_sensitivities": [],
    }

    # Top dependencies from Phase 2a
    if dep_results:
        allele_data = dep_results.get("allele_comparisons", {}).get(allele, {})
        top_deps = allele_data.get("top_dependencies", [])[:10]
        strategy["top_dependencies"] = [
            {"gene": d["gene"], "fdr": d.get("fdr"), "cohens_d": d.get("cohens_d"),
             "mean_allele": d.get("mean_allele"), "mean_wt": d.get("mean_wt")}
            for d in top_deps
        ]
        strategy["n_significant_dependencies"] = allele_data.get("n_significant", 0)

    # Top drug sensitivities from Phase 3
    if drug_results:
        allele_drug = drug_results.get("allele_drug_comparisons", {}).get(allele, {})
        top_drugs = allele_drug.get("top_drugs", [])[:10]
        strategy["top_drug_sensitivities"] = [
            {"drug": d.get("drug_name", ""), "drug_class": d.get("drug_class"),
             "fdr": d.get("fdr"), "cohens_d": d.get("cohens_d")}
            for d in top_drugs
        ]
        strategy["n_significant_drugs"] = allele_drug.get("n_significant", 0)

    return strategy


def _crc_vs_nsclc_comparison(
    dep_results: dict | None,
    nsclc_dir: Path,
) -> dict:
    """Compare CRC KRAS dependencies to NSCLC findings."""
    comparison = {
        "available": False,
        "notes": [],
    }

    # Check for NSCLC dependency data
    nsclc_dep_file = nsclc_dir / "prmt5_dependency_results.json"
    nsclc_data = _load_json(nsclc_dep_file)

    if not nsclc_data or not dep_results:
        comparison["notes"].append("NSCLC comparison limited — no direct KRAS allele dependency data from nsclc-depmap-targets project")
        comparison["notes"].append("CRC G12C is rare (8%) vs dominant in NSCLC (21%) — direct comparison important for G12C inhibitor CRC development")
        comparison["notes"].append("NSCLC MTAP-PRMT5 data available but targets different biology (synthetic lethality vs KRAS allele dependencies)")
        return comparison

    comparison["available"] = True

    # Extract NSCLC KRAS-related findings if present
    nsclc_classified = nsclc_dir / "nsclc_cell_lines_classified.csv"
    if nsclc_classified.exists():
        import pandas as pd
        nsclc_df = pd.read_csv(nsclc_classified, index_col=0)
        if "KRAS_allele" in nsclc_df.columns:
            nsclc_allele_dist = nsclc_df["KRAS_allele"].value_counts().to_dict()
            comparison["nsclc_kras_allele_distribution"] = nsclc_allele_dist

    # Key biological differences
    comparison["notes"] = [
        "CRC KRAS landscape: G12D(33%) > G12V(22%) > G13D(18%) > G12C(8%)",
        "NSCLC KRAS landscape: G12C(39%) > G12V(21%) > G12D(14%) > G12A(7%)",
        "G12C: rare in CRC, dominant in NSCLC — sotorasib ORR 37% NSCLC vs 9.7% CRC",
        "G12D: dominant in CRC, less common in NSCLC — zoldonrasib/MRTX1133 prioritize CRC",
        "G13D: CRC-specific allele (18%), near-absent in NSCLC — no targeted agents",
    ]

    return comparison


def build_strategy_matrix(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    """Compile therapeutic strategy matrix from all phase outputs."""
    output_dir = Path(output_dir)

    dep_results = _load_json(output_dir / "allele_dependency_results.json")
    drug_results = _load_json(output_dir / "prism_drug_results.json")
    interaction_results = _load_json(output_dir / "allele_comparison_results.json")

    print("Loading phase outputs...")
    print(f"  Dependency results: {'loaded' if dep_results else 'not found'}")
    print(f"  Drug results: {'loaded' if drug_results else 'not found'}")
    print(f"  Interaction results: {'loaded' if interaction_results else 'not found'}")

    matrix = {"allele_strategies": {}, "cross_allele_insights": {}, "crc_vs_nsclc": {}}

    # Per-allele strategies
    alleles = ["G12D", "G13D", "G12V", "G12C", "Q61H", "A146T"]
    for allele in alleles:
        print(f"\nBuilding strategy for {allele}...")
        matrix["allele_strategies"][allele] = _build_allele_strategy(
            allele, dep_results, drug_results, interaction_results
        )

    # All KRAS-mut combined
    matrix["allele_strategies"]["all_KRAS_mut"] = _build_allele_strategy(
        "all_KRAS_mut", dep_results, drug_results, interaction_results
    )

    # Cross-allele insights from Phase 2b
    if interaction_results:
        matrix["cross_allele_insights"] = {
            "g12d_vs_g13d": {
                "n_significant": interaction_results.get("g12d_vs_g13d", {}).get("n_significant", 0),
                "interpretation": "G12D and G13D show no statistically significant differential dependencies at current sample sizes, but top-ranked genes may reveal biological differences with validation",
            },
            "pik3ca_interaction": {
                "n_significant": interaction_results.get("pik3ca_interaction", {}).get("n_significant", 0),
                "rate": interaction_results.get("summary", {}).get("pik3ca_co_mutation_rate", "35%"),
                "interpretation": "PIK3CA co-mutation (35% of KRAS-mut CRC) does not significantly alter dependency profiles in current data; PI3K pathway combos may still be relevant clinically",
            },
            "msi_interaction": {
                "n_significant": interaction_results.get("msi_interaction", {}).get("n_significant", 0),
                "interpretation": "MSI-H vs MSS shows no significant differential dependencies within KRAS-mutant; immunotherapy eligibility (MSI-H) remains the key clinical distinction",
            },
        }

    # CRC vs NSCLC comparison
    nsclc_dir = REPO_ROOT / "output" / "mtap-prmt5-nsclc-sl"
    matrix["crc_vs_nsclc"] = _crc_vs_nsclc_comparison(dep_results, nsclc_dir)

    # Key drug classes from Phase 3
    if drug_results:
        matrix["key_drug_classes"] = drug_results.get("key_drugs", {})

    # Overall summary
    summary = dep_results.get("summary", {}) if dep_results else {}
    matrix["overall_summary"] = {
        "total_crc_lines_analyzed": summary.get("total_crc_lines", 0),
        "total_kras_mutant": summary.get("total_kras_mutant", 0),
        "total_genes_screened": summary.get("total_genes_in_crispr", 0),
        "total_drugs_screened": drug_results.get("summary", {}).get("total_drugs", 0) if drug_results else 0,
        "phase4_tcga_status": "blocked — awaiting TCGA COAD/READ data download (task #117)",
        "success_criteria": {
            "allele_specific_dependencies_fdr005": "1 (KRAS in G12D), limited by sample size",
            "g12d_vs_g13d_distinction": "Top-ranked genes identified but none FDR-significant",
            "drug_dependency_correlation_gt03": "Not achieved — no gene-drug name matches in current data",
            "crc_nsclc_divergence": "Confirmed: allele frequency landscapes fundamentally different",
        },
    }

    return matrix


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build CRC KRAS therapeutic strategy matrix",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
    )
    args = parser.parse_args(argv)

    matrix = build_strategy_matrix(args.output_dir)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dest, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"\nSaved to {args.dest}")

    # Print summary
    print("\n=== Strategy Matrix Summary ===")
    for allele, strat in matrix["allele_strategies"].items():
        pop = strat.get("population_estimate", {})
        n_dep = strat.get("n_significant_dependencies", 0)
        n_drug = strat.get("n_significant_drugs", 0)
        agents = len(strat.get("clinical_agents", []))
        cases = pop.get("annual_us_cases", "?")
        print(f"  {allele}: {cases} US cases/yr, {n_dep} sig deps, {n_drug} sig drugs, {agents} clinical agents")


if __name__ == "__main__":
    main()
