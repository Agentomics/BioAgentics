"""Phase 3: IEDB MHC-II binding prediction wrapper.

Submits GAS peptides to the IEDB T-cell epitope prediction API (netmhciipan)
for binding affinity prediction against PANDAS-associated HLA alleles.

Processes peptides in batches to respect API rate limits and memory constraints.
Reads peptide libraries from Phase 2 output via streaming to handle large files.
"""

import csv
import json
import time
from pathlib import Path

import requests

from .hla_allele_panel import get_mhc_ii_alleles

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "pandas_pans" / "hla-peptide-presentation"

IEDB_API_URL = "http://tools-cluster-interface.iedb.org/tools_api/mhcii/"

# Binding classification thresholds (percentile rank)
STRONG_BINDER_THRESHOLD = 0.5   # < 0.5% rank
WEAK_BINDER_THRESHOLD = 2.0     # < 2.0% rank

BATCH_SIZE = 100  # Peptides per API call
API_DELAY_SECONDS = 1.0  # Rate limiting delay between calls


def classify_binding(percentile_rank: float) -> str:
    """Classify binding strength by percentile rank."""
    if percentile_rank < STRONG_BINDER_THRESHOLD:
        return "strong_binder"
    elif percentile_rank < WEAK_BINDER_THRESHOLD:
        return "weak_binder"
    return "non_binder"


def call_iedb_api(peptides: list[str], allele: str, method: str = "netmhciipan") -> list[dict]:
    """Submit peptides to IEDB MHC-II binding prediction API.

    Returns list of prediction dicts with keys:
        peptide, allele, ic50, percentile_rank, method
    """
    payload = {
        "method": method,
        "sequence_text": "\n".join(peptides),
        "allele": allele,
    }

    resp = requests.post(IEDB_API_URL, data=payload, timeout=300)
    resp.raise_for_status()

    results = []
    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        return results

    header = lines[0].split("\t")
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < len(header):
            continue

        row = dict(zip(header, fields))
        try:
            ic50 = float(row.get("ic50", row.get("ic50_nM", "0")))
            rank = float(row.get("percentile_rank", row.get("rank", "100")))
        except (ValueError, TypeError):
            continue

        results.append({
            "peptide": row.get("peptide", row.get("core_peptide", "")),
            "allele": allele,
            "ic50": ic50,
            "percentile_rank": rank,
            "method": method,
        })

    return results


def load_peptides_sampled(
    peptide_tsv: Path,
    max_peptides: int | None = None,
) -> list[dict]:
    """Load peptides from Phase 2 TSV, streaming to limit memory.

    Returns list of dicts with keys: peptide, protein_accession, protein_name,
    is_virulence_factor.
    """
    peptides = []
    seen = set()

    with open(peptide_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pep = row["peptide"]
            if pep in seen:
                continue
            seen.add(pep)
            peptides.append({
                "peptide": pep,
                "protein_accession": row.get("protein_accession", ""),
                "protein_name": row.get("protein_name", ""),
                "is_virulence_factor": row.get("is_virulence_factor", "False") == "True",
            })
            if max_peptides and len(peptides) >= max_peptides:
                break

    return peptides


def run_binding_predictions(
    serotype: str = "M12",
    mhc_class: str = "II",
    max_peptides: int | None = None,
    output_base: Path | None = None,
) -> dict:
    """Run IEDB binding predictions for a serotype's peptides against the HLA panel.

    For MHC-II, submits 15mer peptides against all MHC-II alleles.
    Writes results to binding_predictions/ directory.
    Returns prediction summary dict.
    """
    if output_base is None:
        output_base = DATA_DIR

    peptide_dir = output_base / "peptide_libraries"
    pred_dir = output_base / "binding_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Load peptides
    suffix = "mhc_ii" if mhc_class == "II" else "mhc_i"
    peptide_file = peptide_dir / f"{serotype.lower()}_{suffix}_peptides.tsv"
    if not peptide_file.exists():
        raise FileNotFoundError(f"Peptide file not found: {peptide_file}")

    peptides = load_peptides_sampled(peptide_file, max_peptides=max_peptides)
    print(f"Loaded {len(peptides)} unique peptides from {serotype}")

    # Get alleles for this MHC class
    alleles = get_mhc_ii_alleles() if mhc_class == "II" else []
    allele_names = [a.four_digit_resolution for a in alleles]

    # Run predictions
    all_results = []
    pep_sequences = [p["peptide"] for p in peptides]
    pep_meta = {p["peptide"]: p for p in peptides}

    for allele_name in allele_names:
        print(f"  Predicting {allele_name} ({len(pep_sequences)} peptides)...")

        for batch_start in range(0, len(pep_sequences), BATCH_SIZE):
            batch = pep_sequences[batch_start : batch_start + BATCH_SIZE]

            try:
                results = call_iedb_api(batch, allele_name)
                for r in results:
                    meta = pep_meta.get(r["peptide"], {})
                    r["binding_class"] = classify_binding(r["percentile_rank"])
                    r["source_protein"] = meta.get("protein_name", "")
                    r["source_accession"] = meta.get("protein_accession", "")
                    r["serotype"] = serotype
                    r["is_virulence_factor"] = meta.get("is_virulence_factor", False)
                    all_results.append(r)
            except requests.RequestException as e:
                print(f"    API error for {allele_name} batch {batch_start}: {e}")

            if batch_start + BATCH_SIZE < len(pep_sequences):
                time.sleep(API_DELAY_SECONDS)

        time.sleep(API_DELAY_SECONDS)

    # Write predictions TSV
    output_file = pred_dir / f"binding_predictions_{suffix}.tsv"
    fieldnames = [
        "peptide", "allele", "ic50", "percentile_rank", "binding_class",
        "method", "source_protein", "source_accession", "serotype",
        "is_virulence_factor",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_results)

    # Summary
    binding_counts = {"strong_binder": 0, "weak_binder": 0, "non_binder": 0}
    for r in all_results:
        binding_counts[r["binding_class"]] += 1

    summary = {
        "mhc_class": mhc_class,
        "alleles": allele_names,
        "unique_peptides": len(set(r["peptide"] for r in all_results)),
        "total_predictions": len(all_results),
        **binding_counts,
        "output_file": output_file.name,
    }

    summary_path = pred_dir / f"prediction_summary_{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(all_results)} predictions to {output_file}")
    return summary


if __name__ == "__main__":
    run_binding_predictions(serotype="M12", mhc_class="II", max_peptides=20)
