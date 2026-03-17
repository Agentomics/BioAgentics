"""Download and parse DrugBank FDA-approved drug-target interactions for TS.

Reuses the core DrugBank XML parser from cd_fibrosis, then produces a
flattened drug-target table with TS-specific annotations (positive/negative
control drugs for scoring calibration).

DrugBank XML must be downloaded manually (requires academic license):
  https://go.drugbank.com/releases/latest

Place at:
  data/crohns/cd-fibrosis-drug-repurposing/drugbank/drugbank_all_full_database.xml

Output: data/tourettes/ts-drug-repurposing-network/drugbank_targets.tsv
  Columns: [drugbank_id, drug_name, target_gene, target_uniprot, action_type,
            approved_indication, ts_control_label]

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.download_drugbank
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from bioagentics.config import REPO_ROOT
from bioagentics.data.cd_fibrosis.drugbank import (
    DEFAULT_XML,
    filter_approved_drugs,
    parse_drugbank_xml,
)

DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network"
OUTPUT_PATH = DATA_DIR / "drugbank_targets.tsv"

# Known TS drugs for positive/negative control calibration
TS_CONTROL_DRUGS: dict[str, str] = {
    # Positive controls (clinical efficacy in TS)
    "haloperidol": "positive_control",
    "pimozide": "positive_control",
    "aripiprazole": "positive_control",
    "fluphenazine": "positive_control",
    "risperidone": "positive_control",
    "clonidine": "positive_control",
    "guanfacine": "positive_control",
    "ecopipam": "positive_control",        # D1 antagonist, Phase 3 success
    "tetrabenazine": "positive_control",    # VMAT2 — limited evidence
    # Negative controls (mechanistic rationale but disappointing trials)
    "valbenazine": "negative_control",      # VMAT2 inhibitor, failed TS trials
    "deutetrabenazine": "negative_control", # VMAT2 inhibitor, limited TS efficacy
    # Emerging candidates (for validation)
    "xanomeline": "emerging_candidate",     # M1/M4 agonist (KarXT component)
    "pitolisant": "emerging_candidate",     # H3R inverse agonist
}


def flatten_drug_targets(drugs: list[dict]) -> list[dict]:
    """Flatten drug records into one row per drug-target pair.

    Extracts target gene symbols and produces individual rows,
    making it easier to query which drugs target which genes.
    """
    rows = []
    for drug in drugs:
        targets = drug["target_genes"].split(";") if drug["target_genes"] else []
        name_lower = drug["name"].lower()
        ts_label = TS_CONTROL_DRUGS.get(name_lower, "")

        if not targets:
            # Keep drugs with no known targets for completeness
            rows.append({
                "drugbank_id": drug["drugbank_id"],
                "drug_name": drug["name"],
                "target_gene": "",
                "target_uniprot": "",
                "action_type": drug.get("mechanism_of_action", "")[:200],
                "approved_indication": drug.get("indication", "")[:200],
                "ts_control_label": ts_label,
            })
        else:
            for gene in targets:
                rows.append({
                    "drugbank_id": drug["drugbank_id"],
                    "drug_name": drug["name"],
                    "target_gene": gene.strip(),
                    "target_uniprot": "",  # Would need separate lookup
                    "action_type": drug.get("mechanism_of_action", "")[:200],
                    "approved_indication": drug.get("indication", "")[:200],
                    "ts_control_label": ts_label,
                })
    return rows


def save_targets_tsv(rows: list[dict], output: Path) -> None:
    """Save flattened drug-target rows to TSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["drugbank_id", "drug_name", "target_gene", "target_uniprot",
                  "action_type", "approved_indication", "ts_control_label"]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} drug-target pairs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse DrugBank for TS drug repurposing")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--approved-only", action="store_true", default=True)
    args = parser.parse_args()

    if not args.xml.exists():
        print(
            f"DrugBank XML not found at {args.xml}\n\n"
            "DrugBank requires an academic license. To obtain the data:\n"
            "  1. Register at https://go.drugbank.com/\n"
            "  2. Download the full database XML\n"
            "  3. Extract and place at the path above\n",
            file=sys.stderr,
        )
        sys.exit(1)

    drugs = parse_drugbank_xml(args.xml)
    if args.approved_only:
        drugs = filter_approved_drugs(drugs)
        print(f"  Filtered to {len(drugs)} approved drugs")

    rows = flatten_drug_targets(drugs)

    # Check TS control drugs are present
    found_controls = {r["drug_name"].lower() for r in rows if r["ts_control_label"]}
    expected = set(TS_CONTROL_DRUGS.keys())
    missing = expected - found_controls
    if missing:
        print(f"  WARNING: TS control drugs not found in DrugBank: {missing}")

    save_targets_tsv(rows, args.output)

    # Summary
    unique_drugs = {r["drugbank_id"] for r in rows}
    unique_targets = {r["target_gene"] for r in rows if r["target_gene"]}
    print(f"  Unique drugs: {len(unique_drugs)}")
    print(f"  Unique target genes: {len(unique_targets)}")

    # Show TS control drugs found
    for label in ["positive_control", "negative_control", "emerging_candidate"]:
        names = sorted({r["drug_name"] for r in rows if r["ts_control_label"] == label})
        if names:
            print(f"  {label}: {', '.join(names)}")


if __name__ == "__main__":
    main()
