"""Population-level HLA risk modeling for PANDAS susceptibility.

Uses HLA allele frequencies from AFND (Allele Frequency Net Database) across
multiple populations to model PANDAS susceptibility risk based on the frequency
of susceptibility vs protective alleles.

Outputs:
- Per-population risk scores
- High-risk haplotype combinations
- Haplotype risk table

Usage:
    uv run python -m bioagentics.pipelines.population_risk [--hla-panel FILE] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bioagentics.config import DATA_DIR

logger = logging.getLogger(__name__)

HLA_PANEL_PATH = DATA_DIR / "pandas_pans" / "hla-peptide-presentation" / "hla_allele_panel.json"
OUTPUT_DIR = DATA_DIR / "pandas_pans" / "hla-peptide-presentation" / "population_risk"

# AFND population allele frequencies (published data, 4-digit resolution).
# Sources: Allele Frequency Net Database (www.allelefrequencies.net),
# Gonzalez-Galarza et al. 2020, NMDP registry data.
# Frequencies are 2N (allele frequency, not carrier frequency).
AFND_FREQUENCIES: dict[str, dict[str, float]] = {
    "European Caucasian (USA NMDP)": {
        "DRB1*07:01": 0.1342,
        "DRB1*04:01": 0.0878,
        "DRB1*01:01": 0.0860,
        "DRB1*15:01": 0.1346,
        "DRB1*03:01": 0.1216,
        "DRB1*13:01": 0.0563,
    },
    "African American (USA NMDP)": {
        "DRB1*07:01": 0.0812,
        "DRB1*04:01": 0.0199,
        "DRB1*01:01": 0.0192,
        "DRB1*15:01": 0.0854,
        "DRB1*03:01": 0.0803,
        "DRB1*13:01": 0.0626,
    },
    "Hispanic (USA NMDP)": {
        "DRB1*07:01": 0.0943,
        "DRB1*04:01": 0.0792,
        "DRB1*01:01": 0.0466,
        "DRB1*15:01": 0.0925,
        "DRB1*03:01": 0.0600,
        "DRB1*13:01": 0.0548,
    },
    "Asian (USA NMDP)": {
        "DRB1*07:01": 0.0621,
        "DRB1*04:01": 0.0623,
        "DRB1*01:01": 0.0311,
        "DRB1*15:01": 0.1050,
        "DRB1*03:01": 0.0354,
        "DRB1*13:01": 0.0301,
    },
    "South Asian (India)": {
        "DRB1*07:01": 0.0832,
        "DRB1*04:01": 0.0351,
        "DRB1*01:01": 0.0300,
        "DRB1*15:01": 0.1549,
        "DRB1*03:01": 0.0506,
        "DRB1*13:01": 0.0344,
    },
    "East Asian (Japan)": {
        "DRB1*07:01": 0.0081,
        "DRB1*04:01": 0.0350,
        "DRB1*01:01": 0.0525,
        "DRB1*15:01": 0.0900,
        "DRB1*03:01": 0.0090,
        "DRB1*13:01": 0.0180,
    },
    "Middle Eastern (Iran)": {
        "DRB1*07:01": 0.0641,
        "DRB1*04:01": 0.0572,
        "DRB1*01:01": 0.0389,
        "DRB1*15:01": 0.0893,
        "DRB1*03:01": 0.0917,
        "DRB1*13:01": 0.0610,
    },
    "Northern European (UK)": {
        "DRB1*07:01": 0.1310,
        "DRB1*04:01": 0.1068,
        "DRB1*01:01": 0.0873,
        "DRB1*15:01": 0.1333,
        "DRB1*03:01": 0.1186,
        "DRB1*13:01": 0.0573,
    },
    "Southern European (Italy)": {
        "DRB1*07:01": 0.1190,
        "DRB1*04:01": 0.0502,
        "DRB1*01:01": 0.0672,
        "DRB1*15:01": 0.0702,
        "DRB1*03:01": 0.0969,
        "DRB1*13:01": 0.0793,
    },
    "Sub-Saharan African (Kenya)": {
        "DRB1*07:01": 0.0520,
        "DRB1*04:01": 0.0125,
        "DRB1*01:01": 0.0237,
        "DRB1*15:01": 0.0614,
        "DRB1*03:01": 0.0842,
        "DRB1*13:01": 0.1003,
    },
}

SUSCEPTIBILITY_ALLELES = ["DRB1*07:01", "DRB1*04:01", "DRB1*01:01"]
PROTECTIVE_ALLELES = ["DRB1*15:01"]
CONTROL_ALLELES = ["DRB1*03:01", "DRB1*13:01"]

# Known odds ratios from literature (used for weighted risk scoring)
ALLELE_WEIGHTS: dict[str, float] = {
    "DRB1*07:01": 1.68,   # RF meta-analysis OR
    "DRB1*04:01": 1.40,   # Estimated from RF literature
    "DRB1*01:01": 1.25,   # Estimated from RF literature
    "DRB1*15:01": 0.60,   # Protective OR from RF
}


@dataclass
class PopulationRisk:
    """Risk profile for a single population."""
    population: str
    susceptibility_freq: dict[str, float] = field(default_factory=dict)
    protective_freq: dict[str, float] = field(default_factory=dict)
    control_freq: dict[str, float] = field(default_factory=dict)

    @property
    def combined_susceptibility_freq(self) -> float:
        """Sum of susceptibility allele frequencies."""
        return sum(self.susceptibility_freq.values())

    @property
    def combined_protective_freq(self) -> float:
        """Sum of protective allele frequencies."""
        return sum(self.protective_freq.values())

    @property
    def susceptibility_carrier_rate(self) -> float:
        """Estimated probability of carrying at least one susceptibility allele.

        Uses 1 - product(1-2p+p^2) for diploid under HWE, simplified as
        1 - product((1-p)^2) for independent alleles.
        """
        prob_none = 1.0
        for freq in self.susceptibility_freq.values():
            prob_none *= (1 - freq) ** 2
        return 1 - prob_none

    @property
    def unweighted_risk_score(self) -> float:
        """Simple ratio: combined susceptibility freq / (combined protective freq + epsilon)."""
        prot = self.combined_protective_freq
        if prot <= 0:
            prot = 0.001
        return self.combined_susceptibility_freq / prot

    @property
    def weighted_risk_score(self) -> float:
        """OR-weighted susceptibility score.

        Sum of (allele_freq * ln(OR)) for susceptibility alleles,
        minus (protective_freq * |ln(OR)|) for protective alleles.
        """
        score = 0.0
        for allele, freq in self.susceptibility_freq.items():
            weight = ALLELE_WEIGHTS.get(allele, 1.0)
            score += freq * np.log(weight) if weight > 0 else 0.0
        for allele, freq in self.protective_freq.items():
            weight = ALLELE_WEIGHTS.get(allele, 1.0)
            if weight > 0:
                score -= freq * abs(np.log(weight))
        return float(score)


def load_hla_panel(panel_path: Path) -> dict:
    """Load HLA allele panel JSON."""
    with open(panel_path) as f:
        return json.load(f)


def compute_population_risks(
    afnd_data: dict[str, dict[str, float]] | None = None,
) -> list[PopulationRisk]:
    """Compute risk profiles for each population in AFND data."""
    if afnd_data is None:
        afnd_data = AFND_FREQUENCIES

    risks = []
    for pop_name, freqs in afnd_data.items():
        risk = PopulationRisk(population=pop_name)
        for allele in SUSCEPTIBILITY_ALLELES:
            if allele in freqs:
                risk.susceptibility_freq[allele] = freqs[allele]
        for allele in PROTECTIVE_ALLELES:
            if allele in freqs:
                risk.protective_freq[allele] = freqs[allele]
        for allele in CONTROL_ALLELES:
            if allele in freqs:
                risk.control_freq[allele] = freqs[allele]
        risks.append(risk)

    # Sort by weighted risk score descending
    risks.sort(key=lambda r: r.weighted_risk_score, reverse=True)
    return risks


def compute_haplotype_risks(
    afnd_data: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    """Model risk for common 2-allele haplotype combinations.

    Under HWE, frequency of homozygous = p^2, heterozygous = 2pq.
    For two susceptibility alleles A and B, compound carrier rate
    approximates 2*p_A*p_B (heterozygous for both).
    """
    if afnd_data is None:
        afnd_data = AFND_FREQUENCIES

    haplotype_risks = []
    n_susc = len(SUSCEPTIBILITY_ALLELES)

    for pop_name, freqs in afnd_data.items():
        # Pairwise susceptibility combinations
        for i in range(n_susc):
            for j in range(i + 1, n_susc):
                a1 = SUSCEPTIBILITY_ALLELES[i]
                a2 = SUSCEPTIBILITY_ALLELES[j]
                f1 = freqs.get(a1, 0)
                f2 = freqs.get(a2, 0)
                # Compound heterozygote frequency (carrying one of each)
                compound_freq = 2 * f1 * f2
                # Combined OR (multiplicative model)
                or1 = ALLELE_WEIGHTS.get(a1, 1.0)
                or2 = ALLELE_WEIGHTS.get(a2, 1.0)
                combined_or = or1 * or2

                haplotype_risks.append({
                    "population": pop_name,
                    "allele_1": a1,
                    "allele_2": a2,
                    "freq_1": f1,
                    "freq_2": f2,
                    "compound_carrier_freq": round(compound_freq, 6),
                    "combined_or": round(combined_or, 2),
                    "population_attributable_risk": round(compound_freq * (combined_or - 1), 6),
                })

        # Homozygous susceptibility (highest risk)
        for allele in SUSCEPTIBILITY_ALLELES:
            freq = freqs.get(allele, 0)
            or_val = ALLELE_WEIGHTS.get(allele, 1.0)
            homozygous_freq = freq ** 2
            # Homozygous OR often > heterozygous (additive/recessive models)
            homozygous_or = or_val ** 1.5  # approximate dosage effect

            haplotype_risks.append({
                "population": pop_name,
                "allele_1": allele,
                "allele_2": allele,
                "freq_1": freq,
                "freq_2": freq,
                "compound_carrier_freq": round(homozygous_freq, 6),
                "combined_or": round(homozygous_or, 2),
                "population_attributable_risk": round(
                    homozygous_freq * (homozygous_or - 1), 6
                ),
            })

    # Sort by population attributable risk
    haplotype_risks.sort(key=lambda h: h["population_attributable_risk"], reverse=True)
    return haplotype_risks


def write_population_risk_tsv(risks: list[PopulationRisk], output_path: Path) -> None:
    """Write population risk scores to TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "population",
        "susceptibility_freq_total",
        "protective_freq_total",
        "carrier_rate_any_susceptibility",
        "unweighted_risk_ratio",
        "weighted_risk_score",
        "DRB1_07_01_freq",
        "DRB1_04_01_freq",
        "DRB1_01_01_freq",
        "DRB1_15_01_freq",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in risks:
            writer.writerow({
                "population": r.population,
                "susceptibility_freq_total": f"{r.combined_susceptibility_freq:.4f}",
                "protective_freq_total": f"{r.combined_protective_freq:.4f}",
                "carrier_rate_any_susceptibility": f"{r.susceptibility_carrier_rate:.4f}",
                "unweighted_risk_ratio": f"{r.unweighted_risk_score:.3f}",
                "weighted_risk_score": f"{r.weighted_risk_score:.6f}",
                "DRB1_07_01_freq": f"{r.susceptibility_freq.get('DRB1*07:01', 0):.4f}",
                "DRB1_04_01_freq": f"{r.susceptibility_freq.get('DRB1*04:01', 0):.4f}",
                "DRB1_01_01_freq": f"{r.susceptibility_freq.get('DRB1*01:01', 0):.4f}",
                "DRB1_15_01_freq": f"{r.protective_freq.get('DRB1*15:01', 0):.4f}",
            })


def write_haplotype_risk_tsv(haplotypes: list[dict], output_path: Path) -> None:
    """Write haplotype risk table to TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "population", "allele_1", "allele_2", "freq_1", "freq_2",
        "compound_carrier_freq", "combined_or", "population_attributable_risk",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for h in haplotypes:
            writer.writerow(h)


def run_analysis(hla_panel_path: Path, output_dir: Path) -> dict:
    """Run full population risk analysis. Returns summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load HLA panel for metadata
    panel = load_hla_panel(hla_panel_path)
    logger.info("Loaded HLA panel with %d alleles", len(panel.get("alleles", [])))

    # Population risk scores
    risks = compute_population_risks()
    logger.info("Computed risk profiles for %d populations", len(risks))

    pop_risk_path = output_dir / "population_risk_scores.tsv"
    write_population_risk_tsv(risks, pop_risk_path)
    logger.info("Wrote population risk scores to %s", pop_risk_path)

    # Haplotype risk combinations
    haplotypes = compute_haplotype_risks()
    logger.info("Computed %d haplotype risk combinations", len(haplotypes))

    hap_risk_path = output_dir / "haplotype_risk_table.tsv"
    write_haplotype_risk_tsv(haplotypes, hap_risk_path)
    logger.info("Wrote haplotype risk table to %s", hap_risk_path)

    # Build summary
    highest_risk = risks[0] if risks else None
    lowest_risk = risks[-1] if risks else None
    top_haplotypes = haplotypes[:5]

    summary = {
        "populations_analyzed": len(risks),
        "susceptibility_alleles": SUSCEPTIBILITY_ALLELES,
        "protective_alleles": PROTECTIVE_ALLELES,
        "allele_weights_or": ALLELE_WEIGHTS,
        "population_rankings": [
            {
                "population": r.population,
                "weighted_risk_score": round(r.weighted_risk_score, 6),
                "carrier_rate": round(r.susceptibility_carrier_rate, 4),
                "susceptibility_freq": round(r.combined_susceptibility_freq, 4),
                "protective_freq": round(r.combined_protective_freq, 4),
            }
            for r in risks
        ],
        "highest_risk_population": {
            "name": highest_risk.population if highest_risk else None,
            "weighted_score": round(highest_risk.weighted_risk_score, 6) if highest_risk else None,
            "carrier_rate": round(highest_risk.susceptibility_carrier_rate, 4) if highest_risk else None,
        },
        "lowest_risk_population": {
            "name": lowest_risk.population if lowest_risk else None,
            "weighted_score": round(lowest_risk.weighted_risk_score, 6) if lowest_risk else None,
            "carrier_rate": round(lowest_risk.susceptibility_carrier_rate, 4) if lowest_risk else None,
        },
        "top_haplotype_combinations": top_haplotypes,
        "notes": [
            "Risk scores are relative, not absolute PANDAS prevalence estimates.",
            "Allele frequencies from AFND/NMDP published data.",
            "Weighted scores use ln(OR) from rheumatic fever meta-analyses as proxy.",
            "Carrier rates assume Hardy-Weinberg equilibrium and independent allele segregation.",
            "DQA1*03:01 excluded from population analysis due to sparse multi-population AFND data.",
        ],
    }

    summary_path = output_dir / "risk_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote analysis summary to %s", summary_path)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Population-level HLA risk modeling for PANDAS susceptibility",
    )
    parser.add_argument("--hla-panel", type=Path, default=HLA_PANEL_PATH,
                        help="HLA allele panel JSON (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_analysis(args.hla_panel, args.output_dir)

    print(f"\nPopulation risk analysis complete.")
    print(f"Populations analyzed: {summary['populations_analyzed']}")
    print(f"Highest risk: {summary['highest_risk_population']['name']} "
          f"(score={summary['highest_risk_population']['weighted_score']:.4f}, "
          f"carrier rate={summary['highest_risk_population']['carrier_rate']:.1%})")
    print(f"Lowest risk: {summary['lowest_risk_population']['name']} "
          f"(score={summary['lowest_risk_population']['weighted_score']:.4f}, "
          f"carrier rate={summary['lowest_risk_population']['carrier_rate']:.1%})")


if __name__ == "__main__":
    main()
