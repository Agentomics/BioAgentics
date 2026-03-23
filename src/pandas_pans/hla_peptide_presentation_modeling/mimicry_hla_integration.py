"""Phase 5: Mimicry-HLA integration.

Cross-references differentially presented GAS peptides (Phase 4) with
molecular mimicry candidates from the gas-molecular-mimicry-mapping project.
Tests whether mimicry hits are enriched among differential presenters using
Fisher's exact test, and builds the HLA -> peptide -> human mimic network.
"""

import csv
import json
import math
from pathlib import Path

from scipy import stats as scipy_stats

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "pandas_pans" / "hla-peptide-presentation"
MIMICRY_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "pandas_pans" / "gas-molecular-mimicry-mapping"

# Known PANDAS autoantibody target proteins
PANDAS_TARGETS = [
    "DRD1", "DRD2", "tubulin", "CaMKII", "GAPDH", "enolase", "GM1 synthase",
]


def load_mimicry_hits(mimicry_dir: Path | None = None) -> list[dict]:
    """Load molecular mimicry hits from the gas-molecular-mimicry-mapping project.

    Looks for mimicry results CSV/TSV files in the mimicry data directory.
    Returns list of dicts with at minimum: gas_protein, human_protein, score.
    """
    if mimicry_dir is None:
        mimicry_dir = MIMICRY_DATA_DIR

    hits = []

    # Try common output file patterns, deduplicating across patterns
    seen_paths: set[Path] = set()
    for pattern in ["*mimicry*.tsv", "*mimicry*.csv", "*hits*.tsv", "*results*.tsv"]:
        for path in mimicry_dir.rglob(pattern):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            delimiter = "\t" if path.suffix == ".tsv" else ","
            try:
                with open(path) as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for row in reader:
                        hit = {}
                        # Flexible column name matching
                        for key in row:
                            kl = key.lower()
                            if "gas" in kl and "protein" in kl:
                                hit["gas_protein"] = row[key]
                            elif "human" in kl and "protein" in kl:
                                hit["human_protein"] = row[key]
                            elif "score" in kl or "similarity" in kl:
                                try:
                                    hit["score"] = float(row[key])
                                except (ValueError, TypeError):
                                    pass
                        if hit.get("gas_protein"):
                            hits.append(hit)
            except (OSError, csv.Error):
                continue

    return hits


def load_differential_peptides(diff_file: Path | None = None) -> list[dict]:
    """Load differential peptides from Phase 4 output."""
    if diff_file is None:
        diff_file = DATA_DIR / "differential_analysis" / "differential_peptides.tsv"

    if not diff_file.exists():
        return []

    rows = []
    with open(diff_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def load_binding_predictions(pred_file: Path | None = None) -> list[dict]:
    """Load binding predictions from Phase 3 output."""
    if pred_file is None:
        pred_file = DATA_DIR / "binding_predictions" / "binding_predictions_mhc_ii.tsv"

    if not pred_file.exists():
        return []

    rows = []
    with open(pred_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def compute_overlap(
    mimicry_hits: list[dict],
    differential_peptides: list[dict],
    binding_predictions: list[dict],
) -> dict:
    """Compute overlap between mimicry hits and binding/differential predictions.

    Tests whether GAS proteins with mimicry hits are enriched among those
    producing differentially presented peptides.
    """
    # Extract unique protein sets
    mimicry_proteins = {h.get("gas_protein", "") for h in mimicry_hits if h.get("gas_protein")}
    diff_proteins = set()
    for d in differential_peptides:
        acc = d.get("source_accession", "")
        name = d.get("source_protein", "")
        if acc:
            diff_proteins.add(acc)
        if name:
            diff_proteins.add(name)

    binding_proteins = set()
    for b in binding_predictions:
        acc = b.get("source_accession", "")
        name = b.get("source_protein", "")
        if acc:
            binding_proteins.add(acc)
        if name:
            binding_proteins.add(name)

    # Overlap
    proteins_in_both = mimicry_proteins & binding_proteins
    diff_and_mimicry = mimicry_proteins & diff_proteins

    # Fisher exact test: are mimicry proteins enriched among differential?
    all_predicted = binding_proteins | mimicry_proteins
    a = len(diff_and_mimicry)              # differential & mimicry
    b = len(diff_proteins - mimicry_proteins)  # differential & no mimicry
    c = len(mimicry_proteins - diff_proteins)  # mimicry & not differential
    d = len(all_predicted - diff_proteins - mimicry_proteins)  # neither

    table = [[a, b], [c, d]]
    try:
        fe_result = scipy_stats.fisher_exact(table)
        odds_ratio = float(fe_result[0])  # type: ignore[arg-type]
        p_value = float(fe_result[1])  # type: ignore[arg-type]
    except ValueError:
        odds_ratio, p_value = float("nan"), 1.0

    # Check PANDAS targets
    pandas_in_mimicry = [
        t for t in PANDAS_TARGETS
        if any(t.lower() in h.get("human_protein", "").lower() for h in mimicry_hits)
    ]

    # Build network edges (HLA -> peptide -> human mimic)
    network_edges = []
    for dp in differential_peptides:
        pep_protein = dp.get("source_accession", "") or dp.get("source_protein", "")
        matching_mimicry = [
            h for h in mimicry_hits
            if h.get("gas_protein", "") == pep_protein
        ]
        for match in matching_mimicry:
            network_edges.append({
                "peptide": dp.get("peptide", ""),
                "gas_protein": pep_protein,
                "human_mimic": match.get("human_protein", ""),
                "best_hla_allele": dp.get("best_susceptibility_allele", ""),
                "fold_change": dp.get("fold_change", ""),
            })

    return {
        "mimicry_hits_total": len(mimicry_hits),
        "unique_mimicry_proteins": len(mimicry_proteins),
        "differential_peptides_count": len(differential_peptides),
        "binding_predictions_count": len(binding_predictions),
        "overlap": {
            "proteins_in_both": len(proteins_in_both),
            "differential_and_mimicry": len(diff_and_mimicry),
        },
        "significance_test": {
            "contingency_table": table,
            "odds_ratio": odds_ratio if not math.isnan(odds_ratio) else None,
            "p_value": round(p_value, 6),
        },
        "network_edges": len(network_edges),
        "pandas_target_analysis": {
            "pandas_targets_in_mimicry": len(pandas_in_mimicry),
            "pandas_targets": PANDAS_TARGETS,
            "targets_found": pandas_in_mimicry,
        },
        "notes": "Overlap depends on breadth of binding predictions. Full IEDB run "
        "(~500K peptides x 7 alleles) needed for meaningful overlap analysis.",
    }


def run_mimicry_integration(output_base: Path | None = None) -> dict:
    """Run full mimicry-HLA integration analysis."""
    if output_base is None:
        output_base = DATA_DIR

    out_dir = output_base / "mimicry_integration"
    out_dir.mkdir(parents=True, exist_ok=True)

    mimicry_hits = load_mimicry_hits()
    differential = load_differential_peptides()
    predictions = load_binding_predictions()

    result = compute_overlap(mimicry_hits, differential, predictions)

    summary_path = out_dir / "integration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Mimicry integration: {result['overlap']['differential_and_mimicry']} "
          f"overlapping proteins, {result['network_edges']} network edges")
    return result


if __name__ == "__main__":
    run_mimicry_integration()
