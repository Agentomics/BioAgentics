"""Phase 6: Population-level risk modeling.

Models PANDAS susceptibility risk based on HLA allele frequencies across
10 global populations using AFND data, OR-weighted risk scoring, and
haplotype combination analysis under Hardy-Weinberg equilibrium.
"""

import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "pandas_pans" / "hla-peptide-presentation"

# Odds ratios for risk weighting (from literature)
ALLELE_WEIGHTS: dict[str, float] = {
    "DRB1*07:01": 1.68,   # RF meta-analysis
    "DRB1*04:01": 1.40,   # RF association (estimated)
    "DRB1*01:01": 1.25,   # RF association (estimated)
    "DRB1*15:01": 0.60,   # Protective
}


@dataclass
class PopulationFrequencies:
    """HLA allele frequencies for a population."""

    name: str
    drb1_07_01: float
    drb1_04_01: float
    drb1_01_01: float
    drb1_15_01: float


# AFND population allele frequencies (curated from published data)
POPULATIONS = [
    PopulationFrequencies("Northern European (UK)", 0.146, 0.089, 0.095, 0.137),
    PopulationFrequencies("Northern European (Scandinavia)", 0.120, 0.102, 0.078, 0.142),
    PopulationFrequencies("Western European (Germany)", 0.131, 0.095, 0.088, 0.140),
    PopulationFrequencies("Southern European (Italy)", 0.155, 0.092, 0.090, 0.085),
    PopulationFrequencies("Southern European (Spain)", 0.142, 0.088, 0.082, 0.095),
    PopulationFrequencies("Eastern European (Poland)", 0.128, 0.084, 0.075, 0.130),
    PopulationFrequencies("North American (USA Caucasian)", 0.134, 0.088, 0.086, 0.135),
    PopulationFrequencies("South Asian (India)", 0.098, 0.065, 0.048, 0.110),
    PopulationFrequencies("East Asian (Japan)", 0.015, 0.078, 0.010, 0.095),
    PopulationFrequencies("Sub-Saharan African (Kenya)", 0.082, 0.035, 0.040, 0.068),
]


def carrier_rate(freq: float) -> float:
    """Carrier rate under Hardy-Weinberg: 1 - (1-f)^2."""
    return 1.0 - (1.0 - freq) ** 2


def any_susceptibility_carrier_rate(pop: PopulationFrequencies) -> float:
    """Probability of carrying at least one susceptibility allele (union)."""
    freqs = [pop.drb1_07_01, pop.drb1_04_01, pop.drb1_01_01]
    # P(at least one) = 1 - P(none) = 1 - product(1 - carrier_rate_i)
    # Approximation assuming independence (different loci irrelevant — these
    # are all at the same DRB1 locus, so we use allele frequencies directly)
    total_sus_freq = sum(freqs)
    return carrier_rate(min(total_sus_freq, 1.0))


def weighted_risk_score(pop: PopulationFrequencies) -> float:
    """Compute OR-weighted risk score for a population.

    Score = sum(freq_i * ln(OR_i)) for all alleles with known ORs.
    Positive score = net susceptibility, negative = net protection.
    """
    score = 0.0
    allele_freqs = {
        "DRB1*07:01": pop.drb1_07_01,
        "DRB1*04:01": pop.drb1_04_01,
        "DRB1*01:01": pop.drb1_01_01,
        "DRB1*15:01": pop.drb1_15_01,
    }

    for allele, freq in allele_freqs.items():
        or_val = ALLELE_WEIGHTS.get(allele)
        if or_val and or_val > 0:
            score += freq * math.log(or_val)

    return round(score, 4)


def compute_haplotype_combinations(populations: list[PopulationFrequencies]) -> list[dict]:
    """Compute two-allele susceptibility haplotype combinations per population.

    For each population, considers all pairs of susceptibility alleles,
    computes compound carrier frequency and combined OR.
    """
    susceptibility_alleles = ["DRB1*07:01", "DRB1*04:01", "DRB1*01:01"]
    results = []

    for pop in populations:
        allele_freqs = {
            "DRB1*07:01": pop.drb1_07_01,
            "DRB1*04:01": pop.drb1_04_01,
            "DRB1*01:01": pop.drb1_01_01,
        }

        for a1, a2 in combinations(susceptibility_alleles, 2):
            f1 = allele_freqs[a1]
            f2 = allele_freqs[a2]

            # Compound carrier freq (HWE, same locus — heterozygote for both)
            compound_freq = 2 * f1 * f2

            # Combined OR (multiplicative model)
            or1 = ALLELE_WEIGHTS.get(a1, 1.0)
            or2 = ALLELE_WEIGHTS.get(a2, 1.0)
            combined_or = round(or1 * or2, 2)

            # Population attributable risk fraction approximation
            par = round(compound_freq * (combined_or - 1) / (1 + compound_freq * (combined_or - 1)), 4)

            results.append({
                "population": pop.name,
                "allele_1": a1,
                "allele_2": a2,
                "freq_1": round(f1, 4),
                "freq_2": round(f2, 4),
                "compound_carrier_freq": round(compound_freq, 6),
                "combined_or": combined_or,
                "population_attributable_risk": par,
            })

    results.sort(key=lambda x: x["population_attributable_risk"], reverse=True)
    return results


def run_population_risk(output_base: Path | None = None) -> dict:
    """Run full population risk analysis.

    Computes per-population risk scores, carrier rates, and haplotype combinations.
    Writes results to population_risk/ directory.
    """
    if output_base is None:
        output_base = DATA_DIR

    out_dir = output_base / "population_risk"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-population scores
    pop_scores = []
    for pop in POPULATIONS:
        score = weighted_risk_score(pop)
        car = any_susceptibility_carrier_rate(pop)
        sus_total = pop.drb1_07_01 + pop.drb1_04_01 + pop.drb1_01_01
        prot_total = pop.drb1_15_01

        pop_scores.append({
            "population": pop.name,
            "susceptibility_freq_total": round(sus_total, 4),
            "protective_freq_total": round(prot_total, 4),
            "carrier_rate_any_susceptibility": round(car, 4),
            "unweighted_risk_ratio": round(sus_total / prot_total, 4) if prot_total > 0 else None,
            "weighted_risk_score": score,
            "DRB1_07_01_freq": pop.drb1_07_01,
            "DRB1_04_01_freq": pop.drb1_04_01,
            "DRB1_01_01_freq": pop.drb1_01_01,
            "DRB1_15_01_freq": pop.drb1_15_01,
        })

    pop_scores.sort(key=lambda x: x["weighted_risk_score"], reverse=True)

    # Write population risk scores TSV
    scores_path = out_dir / "population_risk_scores.tsv"
    fieldnames = list(pop_scores[0].keys())
    with open(scores_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(pop_scores)

    # Compute haplotype combinations
    haplotypes = compute_haplotype_combinations(POPULATIONS)

    # Write haplotype table TSV
    hap_path = out_dir / "haplotype_risk_table.tsv"
    hap_fieldnames = list(haplotypes[0].keys())
    with open(hap_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hap_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(haplotypes)

    # Summary
    summary = {
        "populations_analyzed": len(POPULATIONS),
        "susceptibility_alleles": list(ALLELE_WEIGHTS.keys())[:3],
        "protective_alleles": ["DRB1*15:01"],
        "allele_weights_or": ALLELE_WEIGHTS,
        "population_rankings": pop_scores,
        "top_haplotype_combinations": haplotypes[:20],
        "notes": "Risk scores are relative, not absolute prevalence estimates. "
        "Hardy-Weinberg equilibrium assumed. DQA1*03:01 excluded from population "
        "risk analysis due to sparse multi-population AFND data.",
    }

    summary_path = out_dir / "risk_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Population risk analysis complete: {len(POPULATIONS)} populations")
    print(f"  Highest risk: {pop_scores[0]['population']} (score={pop_scores[0]['weighted_risk_score']})")
    print(f"  Lowest risk: {pop_scores[-1]['population']} (score={pop_scores[-1]['weighted_risk_score']})")

    return summary


if __name__ == "__main__":
    run_population_risk()
