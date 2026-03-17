"""Fetch active TS clinical trials from ClinicalTrials.gov.

Queries the ClinicalTrials.gov v2 API for Tourette syndrome trials
(active, recruiting, or completed in last 3 years).

Output: data/tourettes/ts-drug-repurposing-network/ts_clinical_trials.csv

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.fetch_trials
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import requests

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network"
OUTPUT_PATH = DATA_DIR / "ts_clinical_trials.csv"

CT_API = "https://clinicaltrials.gov/api/v2/studies"


def fetch_ts_trials(max_results: int = 500) -> list[dict]:
    """Fetch Tourette syndrome clinical trials from ClinicalTrials.gov v2 API."""
    rows = []
    page_token = None

    while len(rows) < max_results:
        params = {
            "query.cond": "Tourette Syndrome",
            "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED,ENROLLING_BY_INVITATION",
            "pageSize": min(100, max_results - len(rows)),
            "fields": (
                "NCTId,BriefTitle,InterventionName,InterventionType,"
                "InterventionDescription,Phase,OverallStatus,"
                "EnrollmentCount,StartDate,CompletionDate,"
                "PrimaryCompletionDate,Condition"
            ),
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = requests.get(CT_API, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  API error: {resp.status_code}")
                break
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            break

        studies = data.get("studies", [])
        if not studies:
            break

        for study in studies:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            arms = proto.get("armsInterventionsModule", {})

            nct_id = ident.get("nctId", "")
            title = ident.get("briefTitle", "")

            # Extract drug interventions
            interventions = arms.get("interventions", [])
            drug_interventions = []
            for intv in interventions:
                if intv.get("type", "").upper() in ("DRUG", "BIOLOGICAL", "DIETARY_SUPPLEMENT"):
                    drug_interventions.append({
                        "name": intv.get("name", ""),
                        "description": intv.get("description", "")[:200],
                    })

            if not drug_interventions:
                # Skip non-drug trials (behavioral, device, etc.)
                drug_interventions = [{"name": "non-drug", "description": ""}]

            phases = design.get("phases", [])
            phase_str = ";".join(phases) if phases else ""

            enrollment = design.get("enrollmentInfo", {}).get("count", "")
            start_date = status_mod.get("startDateStruct", {}).get("date", "")
            completion_date = status_mod.get("completionDateStruct", {}).get("date", "")
            overall_status = status_mod.get("overallStatus", "")

            for drug in drug_interventions:
                rows.append({
                    "nct_id": nct_id,
                    "title": title,
                    "intervention_drug": drug["name"],
                    "mechanism": drug["description"],
                    "phase": phase_str,
                    "status": overall_status,
                    "enrollment": enrollment,
                    "start_date": start_date,
                })

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TS clinical trials")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max-results", type=int, default=500)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching TS clinical trials from ClinicalTrials.gov...")
    trials = fetch_ts_trials(max_results=args.max_results)

    if trials:
        fieldnames = ["nct_id", "title", "intervention_drug", "mechanism",
                      "phase", "status", "enrollment", "start_date"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trials)
        print(f"  Saved {len(trials)} trial records to {args.output}")
    else:
        print("  No trials found")

    # Summary
    unique_trials = {r["nct_id"] for r in trials}
    unique_drugs = {r["intervention_drug"] for r in trials if r["intervention_drug"] != "non-drug"}
    print(f"  Unique trials: {len(unique_trials)}")
    print(f"  Drug interventions: {len(unique_drugs)}")

    # Phase breakdown
    from collections import Counter
    phases = Counter(r["phase"] for r in trials if r["phase"])
    for phase, count in phases.most_common():
        print(f"    {phase}: {count}")

    # Notable drugs
    drug_counts = Counter(r["intervention_drug"] for r in trials if r["intervention_drug"] != "non-drug")
    print(f"\n  Most studied drugs:")
    for drug, count in drug_counts.most_common(15):
        print(f"    {drug}: {count} trials")


if __name__ == "__main__":
    main()
