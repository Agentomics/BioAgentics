"""Download ChEMBL bioactivity data for TS-relevant targets.

Queries ChEMBL REST API for bioactivity data on key Tourette syndrome targets:
DRD1, DRD2, SLC18A2 (VMAT2), SLC6A4 (SERT), HRH3, GABA receptor subtypes,
PDE10A, CHRM1, CHRM4, CNR1, CNR2.

Output: data/tourettes/ts-drug-repurposing-network/chembl_bioactivity.tsv

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.download_chembl
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network"
OUTPUT_PATH = DATA_DIR / "chembl_bioactivity.tsv"

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

# TS-relevant targets: gene_symbol -> ChEMBL target preferred name for lookup
TS_TARGETS: dict[str, str] = {
    "DRD1": "Dopamine D1 receptor",
    "DRD2": "Dopamine D2 receptor",
    "DRD3": "Dopamine D3 receptor",
    "DRD4": "Dopamine D4 receptor",
    "SLC18A2": "Vesicular monoamine transporter 2",
    "SLC6A4": "Serotonin transporter",
    "HRH3": "Histamine H3 receptor",
    "GABRA1": "GABA-A receptor; alpha-1 subunit",
    "PDE10A": "Phosphodiesterase 10A",
    "CHRM1": "Muscarinic acetylcholine receptor M1",
    "CHRM4": "Muscarinic acetylcholine receptor M4",
    "CNR1": "Cannabinoid CB1 receptor",
    "CNR2": "Cannabinoid CB2 receptor",
    "HTR2A": "Serotonin 2a (5-HT2a) receptor",
    "HTR2C": "Serotonin 2c (5-HT2c) receptor",
    "ADRA2A": "Alpha-2a adrenergic receptor",
}


def search_target_chembl_id(gene_symbol: str) -> str | None:
    """Look up ChEMBL target ID for a human gene symbol."""
    try:
        resp = requests.get(
            f"{CHEMBL_API}/target/search.json",
            params={"q": gene_symbol, "limit": 10},
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        for target in data.get("targets", []):
            # Match human targets
            if target.get("organism") == "Homo sapiens":
                components = target.get("target_components", [])
                for comp in components:
                    acc = comp.get("accession", "")
                    # Check if gene symbol matches via component description
                    if gene_symbol.upper() in str(comp).upper():
                        return target.get("target_chembl_id")
                # Fallback: return first human SINGLE PROTEIN target
                if target.get("target_type") == "SINGLE PROTEIN":
                    return target.get("target_chembl_id")
    except requests.RequestException:
        pass
    return None


def fetch_bioactivities(
    target_chembl_id: str,
    gene_symbol: str,
    limit: int = 500,
) -> list[dict]:
    """Fetch bioactivity data for a target from ChEMBL."""
    rows = []
    offset = 0

    while True:
        try:
            resp = requests.get(
                f"{CHEMBL_API}/activity.json",
                params={
                    "target_chembl_id": target_chembl_id,
                    "type__in": "IC50,Ki,EC50,Kd",
                    "limit": min(limit - len(rows), 500),
                    "offset": offset,
                },
                timeout=60,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            activities = data.get("activities", [])
            if not activities:
                break

            for act in activities:
                # Filter for quantitative data with values
                if not act.get("standard_value"):
                    continue
                rows.append({
                    "chembl_id": act.get("molecule_chembl_id", ""),
                    "compound_name": act.get("molecule_pref_name", "") or "",
                    "target_gene": gene_symbol,
                    "target_chembl_id": target_chembl_id,
                    "activity_type": act.get("standard_type", ""),
                    "activity_value": act.get("standard_value", ""),
                    "activity_units": act.get("standard_units", ""),
                    "assay_type": act.get("assay_type", ""),
                    "pchembl_value": act.get("pchembl_value", "") or "",
                    "max_phase": act.get("molecule_max_phase", "") or "",
                })

            offset += len(activities)
            if len(rows) >= limit or len(activities) < 500:
                break
            time.sleep(0.3)  # Rate limiting
        except requests.RequestException as e:
            print(f"    Error fetching activities: {e}")
            break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ChEMBL bioactivity data")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit-per-target", type=int, default=500)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for gene_symbol in TS_TARGETS:
        print(f"  Querying ChEMBL for {gene_symbol}...")
        target_id = search_target_chembl_id(gene_symbol)
        if not target_id:
            print(f"    WARNING: No ChEMBL target found for {gene_symbol}")
            continue
        print(f"    Target ID: {target_id}")

        activities = fetch_bioactivities(target_id, gene_symbol, limit=args.limit_per_target)
        print(f"    {len(activities)} bioactivity records")
        all_rows.extend(activities)

        time.sleep(0.5)  # Rate limiting between targets

    # Save
    if all_rows:
        fieldnames = [
            "chembl_id", "compound_name", "target_gene", "target_chembl_id",
            "activity_type", "activity_value", "activity_units", "assay_type",
            "pchembl_value", "max_phase",
        ]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved {len(all_rows)} bioactivity records to {args.output}")

    # Summary
    unique_compounds = {r["chembl_id"] for r in all_rows}
    clinical_compounds = {r["chembl_id"] for r in all_rows if r.get("max_phase") and str(r["max_phase"]) >= "3"}
    print(f"Unique compounds: {len(unique_compounds)}")
    print(f"Clinical-stage compounds (Phase 3+): {len(clinical_compounds)}")

    # Per-target summary
    from collections import Counter
    target_counts = Counter(r["target_gene"] for r in all_rows)
    for gene, count in target_counts.most_common():
        print(f"  {gene}: {count} records")


if __name__ == "__main__":
    main()
