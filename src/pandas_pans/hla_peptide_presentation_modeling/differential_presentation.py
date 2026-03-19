"""Phase 4: Differential presentation analysis.

Identifies GAS peptides with significantly stronger binding to susceptibility
HLA alleles vs. control alleles, performs virulence factor enrichment, and
compares serotype distributions among differential binders.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy import stats as scipy_stats

from .hla_allele_panel import ALLELE_GROUPS

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "pandas_pans" / "hla-peptide-presentation"

FOLD_CHANGE_THRESHOLD = 10.0


def load_binding_predictions(pred_file: Path) -> list[dict]:
    """Load binding predictions TSV into memory.

    These files are typically small (the full IEDB run output, not the raw
    peptide libraries), so loading entirely is safe.
    """
    rows = []
    with open(pred_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["percentile_rank"] = float(row["percentile_rank"])
            row["ic50"] = float(row["ic50"])
            row["is_virulence_factor"] = str(row.get("is_virulence_factor", "False")).lower() == "true"
            rows.append(row)
    return rows


def compute_differential_peptides(predictions: list[dict]) -> list[dict]:
    """Compute fold-change between susceptibility and control allele ranks.

    For each peptide, calculates mean percentile rank across susceptibility
    alleles and control alleles, then computes fold change = control / susceptibility.
    Higher fold change means the peptide binds better to susceptibility alleles.
    """
    susceptibility_alleles = set()
    for name in ALLELE_GROUPS["susceptibility"]:
        # Match by four-digit resolution (strip HLA- prefix)
        susceptibility_alleles.add(name.replace("HLA-", ""))

    control_alleles = set()
    for name in ALLELE_GROUPS["control"]:
        control_alleles.add(name.replace("HLA-", ""))

    protective_alleles = set()
    for name in ALLELE_GROUPS["protective"]:
        protective_alleles.add(name.replace("HLA-", ""))

    # Group predictions by peptide
    peptide_data: dict[str, dict] = defaultdict(lambda: {
        "susceptibility_ranks": [],
        "control_ranks": [],
        "protective_ranks": [],
        "meta": {},
    })

    for pred in predictions:
        pep = pred["peptide"]
        allele = pred["allele"]

        if allele in susceptibility_alleles:
            peptide_data[pep]["susceptibility_ranks"].append(pred["percentile_rank"])
        elif allele in control_alleles:
            peptide_data[pep]["control_ranks"].append(pred["percentile_rank"])
        elif allele in protective_alleles:
            peptide_data[pep]["protective_ranks"].append(pred["percentile_rank"])

        # Store metadata from first occurrence
        if not peptide_data[pep]["meta"]:
            peptide_data[pep]["meta"] = {
                "source_protein": pred.get("source_protein", ""),
                "source_accession": pred.get("source_accession", ""),
                "serotype": pred.get("serotype", ""),
                "is_virulence_factor": pred.get("is_virulence_factor", False),
            }

    # Compute fold changes
    results = []
    for pep, data in peptide_data.items():
        sus_ranks = data["susceptibility_ranks"]
        ctrl_ranks = data["control_ranks"]
        prot_ranks = data["protective_ranks"]

        if not sus_ranks or not ctrl_ranks:
            continue

        mean_sus = sum(sus_ranks) / len(sus_ranks)
        mean_ctrl = sum(ctrl_ranks) / len(ctrl_ranks)
        mean_prot = sum(prot_ranks) / len(prot_ranks) if prot_ranks else None

        # Fold change: higher means stronger binding to susceptibility alleles
        # (lower rank = stronger binding, so fold = ctrl_rank / sus_rank)
        if mean_sus > 0:
            fold_change = mean_ctrl / mean_sus
        else:
            fold_change = float("inf")

        results.append({
            "peptide": pep,
            "fold_change": round(fold_change, 2),
            "mean_susceptibility_rank": round(mean_sus, 4),
            "mean_control_rank": round(mean_ctrl, 4),
            "mean_protective_rank": round(mean_prot, 4) if mean_prot is not None else None,
            "min_susceptibility_rank": round(min(sus_ranks), 4),
            "best_susceptibility_allele": _best_allele(pep, predictions, susceptibility_alleles),
            **data["meta"],
        })

    results.sort(key=lambda x: x["fold_change"], reverse=True)
    return results


def _best_allele(peptide: str, predictions: list[dict], allele_set: set) -> str:
    """Find the allele with best (lowest) percentile rank for a peptide."""
    best_rank = float("inf")
    best_allele = ""
    for pred in predictions:
        if pred["peptide"] == peptide and pred["allele"] in allele_set:
            if pred["percentile_rank"] < best_rank:
                best_rank = pred["percentile_rank"]
                best_allele = pred["allele"]
    return best_allele


def virulence_enrichment_test(differential: list[dict], all_peptides: list[dict]) -> dict:
    """Fisher exact test for virulence factor enrichment among differential peptides."""
    diff_peps = {d["peptide"] for d in differential}
    all_peps = {p["peptide"] for p in all_peptides}

    # 2x2 table: [differential & VF, differential & non-VF]
    #             [non-diff & VF, non-diff & non-VF]
    vf_lookup = {}
    for p in all_peptides:
        if p["peptide"] not in vf_lookup:
            vf_lookup[p["peptide"]] = p.get("is_virulence_factor", False)

    a = sum(1 for p in diff_peps if vf_lookup.get(p, False))  # diff & VF
    b = len(diff_peps) - a  # diff & non-VF
    c = sum(1 for p in (all_peps - diff_peps) if vf_lookup.get(p, False))  # non-diff & VF
    d = len(all_peps - diff_peps) - c  # non-diff & non-VF

    table = [[a, b], [c, d]]
    try:
        fe_result = scipy_stats.fisher_exact(table)
        odds_ratio = float(fe_result[0])  # type: ignore[arg-type]
        p_value = float(fe_result[1])  # type: ignore[arg-type]
    except ValueError:
        odds_ratio, p_value = float("nan"), 1.0

    return {
        "contingency_table": table,
        "odds_ratio": odds_ratio if not math.isnan(odds_ratio) else None,
        "p_value": round(p_value, 6),
        "differential_vf": a,
        "differential_non_vf": b,
        "total_unique_peptides": len(all_peps),
    }


def run_differential_analysis(
    mhc_class: str = "II",
    output_base: Path | None = None,
) -> dict:
    """Run full differential presentation analysis.

    Reads binding predictions, computes fold changes, tests enrichment,
    and writes results.
    """
    if output_base is None:
        output_base = DATA_DIR

    suffix = "mhc_ii" if mhc_class == "II" else "mhc_i"
    pred_file = output_base / "binding_predictions" / f"binding_predictions_{suffix}.tsv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")

    out_dir = output_base / "differential_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and analyze
    predictions = load_binding_predictions(pred_file)
    all_ranked = compute_differential_peptides(predictions)

    # Filter differential peptides
    differential = [p for p in all_ranked if p["fold_change"] >= FOLD_CHANGE_THRESHOLD]

    # Virulence enrichment
    enrichment = virulence_enrichment_test(differential, predictions)

    # Serotype comparison
    serotype_counts: dict[str, int] = defaultdict(int)
    for d in differential:
        serotype_counts[d.get("serotype", "unknown")] += 1

    # Write differential peptides TSV
    diff_path = out_dir / "differential_peptides.tsv"
    fieldnames = [
        "peptide", "fold_change", "mean_susceptibility_rank", "mean_control_rank",
        "mean_protective_rank", "min_susceptibility_rank", "best_susceptibility_allele",
        "source_protein", "source_accession", "serotype", "is_virulence_factor",
    ]

    with open(diff_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(differential)

    # Write summary JSON
    summary = {
        "total_predictions": len(predictions),
        "unique_peptides": len(all_ranked),
        "differential_peptides": len(differential),
        "fold_change_threshold": FOLD_CHANGE_THRESHOLD,
        "top_10_peptides": all_ranked[:10],
        "virulence_enrichment": enrichment,
        "serotype_comparison": dict(serotype_counts),
    }

    summary_path = out_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Found {len(differential)} differential peptides (>= {FOLD_CHANGE_THRESHOLD}x fold change)")
    return summary


if __name__ == "__main__":
    run_differential_analysis()
