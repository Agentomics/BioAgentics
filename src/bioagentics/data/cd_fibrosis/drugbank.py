"""DrugBank target annotation parser for drug repurposing candidate filtering.

Parses DrugBank XML data to extract drug target annotations, mechanism of action,
safety/approval status, and pharmacokinetic properties for filtering CMAP hits.

DrugBank XML must be downloaded manually (requires academic license):
  https://go.drugbank.com/releases/latest

Place the extracted XML at:
  data/crohns/cd-fibrosis-drug-repurposing/drugbank/drugbank_all_full_database.xml

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.drugbank
    uv run python -m bioagentics.data.cd_fibrosis.drugbank --xml path/to/drugbank.xml
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from bioagentics.config import REPO_ROOT

DEFAULT_XML = (
    REPO_ROOT / "data" / "crohns" / "cd-fibrosis-drug-repurposing"
    / "drugbank" / "drugbank_all_full_database.xml"
)
DEFAULT_DEST = (
    REPO_ROOT / "data" / "crohns" / "cd-fibrosis-drug-repurposing" / "drugbank"
)

NS = "{http://www.drugbank.ca}"


def parse_drugbank_xml(xml_path: Path) -> list[dict]:
    """Parse DrugBank XML and extract drug records with targets.

    Returns a list of dicts with fields:
        drugbank_id, name, type, groups (approval status), categories,
        mechanism_of_action, targets (list of gene symbols),
        indication, pharmacodynamics
    """
    print(f"Parsing DrugBank XML: {xml_path}")
    print("  (this may take a few minutes for the full database)")

    drugs = []
    # Track nesting depth to only process top-level <drug> elements
    depth = 0
    context = ET.iterparse(xml_path, events=("start", "end"))

    for event, elem in context:
        if event == "start" and elem.tag == f"{NS}drug":
            depth += 1
        elif event == "end" and elem.tag == f"{NS}drug":
            if depth == 1:
                drug = _parse_drug_element(elem)
                if drug:
                    drugs.append(drug)
                elem.clear()
            depth -= 1

    print(f"  Parsed {len(drugs)} drug records")
    return drugs


def _parse_drug_element(elem: ET.Element) -> dict | None:
    """Extract relevant fields from a <drug> element."""
    drugbank_id_elem = elem.find(f"{NS}drugbank-id[@primary='true']")
    if drugbank_id_elem is None:
        return None

    name_elem = elem.find(f"{NS}name")
    moa_elem = elem.find(f"{NS}mechanism-of-action")
    indication_elem = elem.find(f"{NS}indication")
    pd_elem = elem.find(f"{NS}pharmacodynamics")

    # Approval groups (approved, experimental, investigational, etc.)
    groups = []
    groups_elem = elem.find(f"{NS}groups")
    if groups_elem is not None:
        for g in groups_elem.findall(f"{NS}group"):
            if g.text:
                groups.append(g.text.strip())

    # Drug categories
    categories = []
    cats_elem = elem.find(f"{NS}categories")
    if cats_elem is not None:
        for cat in cats_elem.findall(f"{NS}category/{NS}category"):
            if cat.text:
                categories.append(cat.text.strip())

    # Targets — extract gene names
    targets = _parse_targets(elem)

    return {
        "drugbank_id": drugbank_id_elem.text.strip() if drugbank_id_elem.text else "",
        "name": name_elem.text.strip() if name_elem is not None and name_elem.text else "",
        "type": elem.get("type", ""),
        "groups": ";".join(groups),
        "categories": ";".join(categories[:10]),  # Limit to avoid huge fields
        "mechanism_of_action": (
            moa_elem.text.strip()[:500] if moa_elem is not None and moa_elem.text else ""
        ),
        "indication": (
            indication_elem.text.strip()[:500]
            if indication_elem is not None and indication_elem.text
            else ""
        ),
        "pharmacodynamics": (
            pd_elem.text.strip()[:500] if pd_elem is not None and pd_elem.text else ""
        ),
        "target_genes": ";".join(targets),
        "n_targets": len(targets),
    }


def _parse_targets(drug_elem: ET.Element) -> list[str]:
    """Extract target gene symbols from a drug element."""
    genes = []
    targets_elem = drug_elem.find(f"{NS}targets")
    if targets_elem is None:
        return genes

    for target in targets_elem.findall(f"{NS}target"):
        polypeptide = target.find(f"{NS}polypeptide")
        if polypeptide is not None:
            gene_elem = polypeptide.find(f"{NS}gene-name")
            if gene_elem is not None and gene_elem.text:
                genes.append(gene_elem.text.strip())

    return genes


def save_drug_targets_tsv(drugs: list[dict], dest: Path) -> None:
    """Save parsed drug records to TSV."""
    if not drugs:
        print("  No drugs to save")
        return

    fieldnames = list(drugs[0].keys())
    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(drugs)

    print(f"  Saved {len(drugs)} drugs to {dest}")


def load_drug_targets(tsv_path: Path) -> dict[str, dict]:
    """Load drug target TSV into a dict keyed by lowercase drug name.

    Useful for looking up drug annotations for CMAP compound hits.
    """
    drugs = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            drugs[row["name"].lower()] = row
    return drugs


def filter_approved_drugs(drugs: list[dict]) -> list[dict]:
    """Filter to approved drugs only (for prioritizing repurposing candidates)."""
    return [d for d in drugs if "approved" in d["groups"].lower()]


def find_drugs_targeting_gene(drugs: list[dict], gene: str) -> list[dict]:
    """Find all drugs that target a specific gene."""
    gene_upper = gene.upper()
    return [
        d for d in drugs
        if gene_upper in d["target_genes"].upper().split(";")
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Parse DrugBank XML for drug target annotations"
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help=f"Path to DrugBank XML (default: {DEFAULT_XML})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Output directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    if not args.xml.exists():
        print(
            f"DrugBank XML not found at {args.xml}\n\n"
            "DrugBank requires an academic license. To obtain the data:\n"
            "  1. Register at https://go.drugbank.com/\n"
            "  2. Download the full database XML\n"
            "  3. Extract and place at: {args.xml}\n\n"
            "Alternatively, run with --xml /path/to/your/drugbank.xml",
            file=sys.stderr,
        )
        sys.exit(1)

    args.dest.mkdir(parents=True, exist_ok=True)

    drugs = parse_drugbank_xml(args.xml)

    # Save full dataset
    full_path = args.dest / "drugbank_targets.tsv"
    save_drug_targets_tsv(drugs, full_path)

    # Save approved-only subset
    approved = filter_approved_drugs(drugs)
    approved_path = args.dest / "drugbank_approved_targets.tsv"
    save_drug_targets_tsv(approved, approved_path)
    print(f"  Approved drugs: {len(approved)}/{len(drugs)}")

    # Check for drugs targeting known fibrosis genes
    fibrosis_targets = ["HDAC1", "GREM1", "SERPINE1", "TWIST1", "FAP", "FGF2",
                        "TGFBR1", "CD38", "JAK1", "JAK2", "YAP1"]
    print(f"\nDrugs targeting key fibrosis genes:")
    for gene in fibrosis_targets:
        hits = find_drugs_targeting_gene(drugs, gene)
        approved_hits = [d for d in hits if "approved" in d["groups"].lower()]
        if hits:
            print(f"  {gene}: {len(hits)} drugs ({len(approved_hits)} approved)")

    print("\nDrugBank parsing complete.")


if __name__ == "__main__":
    main()
