"""Differential presentation analysis for HLA-GAS peptide binding.

Compares binding affinity of GAS peptides to PANDAS-susceptibility HLA alleles
vs control alleles, identifying peptides preferentially presented by
susceptibility alleles (>10-fold affinity difference).

Includes:
- Fold-change analysis (susceptibility vs control mean rank)
- Fisher's exact test for virulence factor enrichment
- Serotype comparison of differential presenters
- Ranked output of differentially presented peptides

Usage:
    uv run python -m bioagentics.pipelines.differential_presentation [--predictions FILE] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from bioagentics.config import DATA_DIR

logger = logging.getLogger(__name__)

PREDICTIONS_DIR = DATA_DIR / "pandas_pans" / "hla-peptide-presentation" / "binding_predictions"
HLA_PANEL_PATH = DATA_DIR / "pandas_pans" / "hla-peptide-presentation" / "hla_allele_panel.json"
OUTPUT_DIR = DATA_DIR / "pandas_pans" / "hla-peptide-presentation" / "differential_analysis"

# Threshold for differential presentation (fold-change in percent rank)
FOLD_CHANGE_THRESHOLD = 10.0

# Allele categories from HLA panel
SUSCEPTIBILITY_ALLELES = {"DRB1*07:01", "DRB1*04:01", "DRB1*01:01"}
CONTROL_ALLELES = {"DRB1*03:01", "DRB1*13:01"}
PROTECTIVE_ALLELES = {"DRB1*15:01", "DQA1*03:01"}


@dataclass
class PeptideBindingProfile:
    """Binding profile for a single peptide across all alleles."""
    peptide: str
    source_protein: str
    source_accession: str
    serotype: str
    is_virulence_factor: bool
    allele_ranks: dict[str, float] = field(default_factory=dict)

    @property
    def mean_susceptibility_rank(self) -> float:
        """Mean percentile rank across susceptibility alleles."""
        ranks = [self.allele_ranks[a] for a in SUSCEPTIBILITY_ALLELES
                 if a in self.allele_ranks]
        return float(np.mean(ranks)) if ranks else 100.0

    @property
    def min_susceptibility_rank(self) -> float:
        """Best (minimum) rank across susceptibility alleles."""
        ranks = [self.allele_ranks[a] for a in SUSCEPTIBILITY_ALLELES
                 if a in self.allele_ranks]
        return min(ranks) if ranks else 100.0

    @property
    def mean_control_rank(self) -> float:
        """Mean percentile rank across control alleles."""
        ranks = [self.allele_ranks[a] for a in CONTROL_ALLELES
                 if a in self.allele_ranks]
        return float(np.mean(ranks)) if ranks else 100.0

    @property
    def mean_protective_rank(self) -> float:
        """Mean percentile rank across protective alleles."""
        ranks = [self.allele_ranks[a] for a in PROTECTIVE_ALLELES
                 if a in self.allele_ranks]
        return float(np.mean(ranks)) if ranks else 100.0

    @property
    def fold_change(self) -> float:
        """Fold-change: control rank / susceptibility rank.

        Higher = more preferentially presented by susceptibility alleles.
        A peptide with susceptibility rank 1.0 and control rank 50.0 has FC=50.
        """
        susc = self.mean_susceptibility_rank
        ctrl = self.mean_control_rank
        if susc <= 0:
            susc = 0.01  # Avoid division by zero
        return ctrl / susc

    @property
    def is_differential(self) -> bool:
        """Whether peptide meets differential presentation threshold."""
        return self.fold_change >= FOLD_CHANGE_THRESHOLD

    @property
    def best_susceptibility_allele(self) -> str:
        """Allele with strongest binding among susceptibility alleles."""
        susc = {a: r for a, r in self.allele_ranks.items()
                if a in SUSCEPTIBILITY_ALLELES}
        if not susc:
            return ""
        return min(susc, key=lambda a: susc[a])


def load_predictions(predictions_path: Path) -> list[dict]:
    """Load binding prediction results from TSV."""
    rows = []
    with open(predictions_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["percentile_rank"] = float(row["percentile_rank"])
            rows.append(row)
    return rows


def _normalize_allele(allele_str: str) -> str:
    """Normalize allele names: 'HLA-DRB1*07:01' -> 'DRB1*07:01'."""
    return allele_str.replace("HLA-", "")


def build_binding_profiles(predictions: list[dict]) -> dict[str, PeptideBindingProfile]:
    """Group predictions by peptide to build per-peptide binding profiles."""
    profiles: dict[str, PeptideBindingProfile] = {}

    for row in predictions:
        pep = row["peptide"]
        allele = _normalize_allele(row["allele"])
        rank = row["percentile_rank"]

        if pep not in profiles:
            profiles[pep] = PeptideBindingProfile(
                peptide=pep,
                source_protein=row.get("source_protein", ""),
                source_accession=row.get("source_accession", ""),
                serotype=row.get("serotype", ""),
                is_virulence_factor=str(row.get("is_virulence_factor", "")).lower() == "true",
            )

        profiles[pep].allele_ranks[allele] = rank

    return profiles


def find_differential_peptides(profiles: dict[str, PeptideBindingProfile],
                               fc_threshold: float = FOLD_CHANGE_THRESHOLD,
                               ) -> list[PeptideBindingProfile]:
    """Identify peptides preferentially presented by susceptibility alleles.

    Returns profiles sorted by fold-change (highest first).
    """
    differential = []
    for profile in profiles.values():
        # Require at least one susceptibility and one control allele prediction
        has_susc = any(a in profile.allele_ranks for a in SUSCEPTIBILITY_ALLELES)
        has_ctrl = any(a in profile.allele_ranks for a in CONTROL_ALLELES)
        if not (has_susc and has_ctrl):
            continue
        if profile.fold_change >= fc_threshold:
            differential.append(profile)

    differential.sort(key=lambda p: p.fold_change, reverse=True)
    return differential


def virulence_enrichment_test(profiles: dict[str, PeptideBindingProfile],
                              differential: list[PeptideBindingProfile],
                              ) -> dict:
    """Fisher's exact test: are virulence factor peptides enriched among differential presenters?

    Contingency table:
                    Differential    Not differential
    VF peptide      a               b
    Non-VF peptide  c               d
    """
    all_with_both = [p for p in profiles.values()
                     if any(a in p.allele_ranks for a in SUSCEPTIBILITY_ALLELES)
                     and any(a in p.allele_ranks for a in CONTROL_ALLELES)]

    diff_set = {p.peptide for p in differential}

    a = sum(1 for p in all_with_both if p.peptide in diff_set and p.is_virulence_factor)
    b = sum(1 for p in all_with_both if p.peptide not in diff_set and p.is_virulence_factor)
    c = sum(1 for p in all_with_both if p.peptide in diff_set and not p.is_virulence_factor)
    d = sum(1 for p in all_with_both if p.peptide not in diff_set and not p.is_virulence_factor)

    table = np.array([[a, b], [c, d]])

    if table.sum() == 0:
        return {"odds_ratio": None, "p_value": None, "table": table.tolist(),
                "significant": False, "note": "No data for enrichment test"}

    fe_result = scipy_stats.fisher_exact(table, alternative="greater")
    odds_ratio = float(fe_result[0])
    p_value = float(fe_result[1])

    return {
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "table": table.tolist(),
        "table_labels": ["[VF+diff, VF+non-diff]", "[nonVF+diff, nonVF+non-diff]"],
        "significant": p_value < 0.05,
        "vf_differential": a,
        "vf_non_differential": b,
        "non_vf_differential": c,
        "non_vf_non_differential": d,
    }


def serotype_comparison(differential: list[PeptideBindingProfile]) -> dict:
    """Compare which serotypes produce the most differentially presented peptides."""
    counts: dict[str, int] = defaultdict(int)
    by_serotype: dict[str, list[str]] = defaultdict(list)

    for p in differential:
        sero = p.serotype or "unknown"
        counts[sero] += 1
        by_serotype[sero].append(p.peptide)

    return {
        "counts": dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)),
        "top_peptides_per_serotype": {
            sero: peptides[:5] for sero, peptides in by_serotype.items()
        },
    }


def write_differential_tsv(differential: list[PeptideBindingProfile],
                           output_path: Path) -> None:
    """Write ranked list of differentially presented peptides to TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "peptide", "fold_change", "mean_susceptibility_rank", "mean_control_rank",
        "mean_protective_rank", "min_susceptibility_rank", "best_susceptibility_allele",
        "source_protein", "source_accession", "serotype",
        "is_virulence_factor",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for p in differential:
            writer.writerow({
                "peptide": p.peptide,
                "fold_change": f"{p.fold_change:.2f}",
                "mean_susceptibility_rank": f"{p.mean_susceptibility_rank:.2f}",
                "mean_control_rank": f"{p.mean_control_rank:.2f}",
                "mean_protective_rank": f"{p.mean_protective_rank:.2f}",
                "min_susceptibility_rank": f"{p.min_susceptibility_rank:.2f}",
                "best_susceptibility_allele": p.best_susceptibility_allele,
                "source_protein": p.source_protein,
                "source_accession": p.source_accession,
                "serotype": p.serotype,
                "is_virulence_factor": p.is_virulence_factor,
            })


def run_analysis(predictions_path: Path, output_dir: Path) -> dict:
    """Run full differential presentation analysis.

    Returns summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading predictions from %s", predictions_path)
    predictions = load_predictions(predictions_path)
    logger.info("Loaded %d prediction rows", len(predictions))

    profiles = build_binding_profiles(predictions)
    logger.info("Built %d peptide binding profiles", len(profiles))

    differential = find_differential_peptides(profiles)
    logger.info("Found %d differentially presented peptides (FC >= %.1f)",
                len(differential), FOLD_CHANGE_THRESHOLD)

    # Write ranked output
    diff_tsv = output_dir / "differential_peptides.tsv"
    write_differential_tsv(differential, diff_tsv)
    logger.info("Wrote ranked differential peptides to %s", diff_tsv)

    # Virulence factor enrichment
    enrichment = virulence_enrichment_test(profiles, differential)
    logger.info("VF enrichment: OR=%.2f, p=%.4f, significant=%s",
                enrichment.get("odds_ratio", 0) or 0,
                enrichment.get("p_value", 1) or 1,
                enrichment.get("significant", False))

    # Serotype comparison
    sero_comparison = serotype_comparison(differential)
    logger.info("Serotype distribution: %s", sero_comparison["counts"])

    # Summary
    summary = {
        "total_predictions": len(predictions),
        "unique_peptides": len(profiles),
        "differential_peptides": len(differential),
        "fold_change_threshold": FOLD_CHANGE_THRESHOLD,
        "top_10_peptides": [
            {
                "peptide": p.peptide,
                "fold_change": round(p.fold_change, 2),
                "susceptibility_rank": round(p.mean_susceptibility_rank, 2),
                "control_rank": round(p.mean_control_rank, 2),
                "best_allele": p.best_susceptibility_allele,
                "source_protein": p.source_protein,
                "serotype": p.serotype,
                "is_vf": p.is_virulence_factor,
            }
            for p in differential[:10]
        ],
        "virulence_enrichment": enrichment,
        "serotype_comparison": sero_comparison,
    }

    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Differential presentation analysis of GAS peptides",
    )
    parser.add_argument("--predictions", type=Path,
                        default=PREDICTIONS_DIR / "binding_predictions_mhc_ii.tsv",
                        help="Binding prediction TSV (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_analysis(args.predictions, args.output_dir)


if __name__ == "__main__":
    main()
