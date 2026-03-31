"""Phase 1a: Build TCGA pan-cancer per-patient binary alteration matrix for 11 TSG/driver genes.

Queries the GDC REST API for somatic mutations (ssm_occurrences) and copy number
variations (cnv_occurrences) across all TCGA projects. Classifies each patient
as altered or WT for each gene based on gene-appropriate criteria.

Gene classification rules:
  - TSGs (PTEN, RB1, ARID1A, SMARCA4, BRCA1/2, KEAP1, NF1): truncating + homdel
  - TP53: truncating + missense (most TP53 missense are GOF/dominant-negative)
  - PIK3CA: hotspot missense (oncogene, activating mutations)
  - MTAP, CDKN2A: homdel + truncating (9p21 co-deletion)

Output: per-patient binary alteration matrix (patients x genes)
  -> output/cancer/tsg-co-alteration-dependency-interactions/phase1_alteration_matrix.csv

Usage:
    uv run python -m cancer.tsg_co_alteration_dependency_interactions.01_tcga_coalteration_matrix
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "tsg-co-alteration-dependency-interactions"

GDC_API = "https://api.gdc.cancer.gov"
TIMEOUT = 60
PAGE_SIZE = 5000

# The 12 target gene symbols (BRCA1 and BRCA2 counted separately)
TARGET_GENES = [
    "MTAP", "ARID1A", "SMARCA4", "PIK3CA", "PTEN", "RB1",
    "BRCA1", "BRCA2", "TP53", "KEAP1", "NF1", "CDKN2A",
]

# VEP consequence types considered truncating / loss-of-function
LOF_CONSEQUENCES = {
    "stop_gained",
    "frameshift_variant",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "start_lost",
}

# PIK3CA activating hotspot residue prefixes
PIK3CA_HOTSPOT_PREFIXES = ["E542", "E545", "H1047", "C420", "N345", "R88", "Q546"]

# Per-gene classification rules
GENE_RULES: dict[str, str] = {
    "MTAP": "deletion",     # 9p21 homdel + truncating
    "ARID1A": "tsg",        # truncating + homdel
    "SMARCA4": "tsg",
    "PIK3CA": "oncogene",   # hotspot missense
    "PTEN": "tsg",
    "RB1": "tsg",
    "BRCA1": "tsg",
    "BRCA2": "tsg",
    "TP53": "tp53",         # truncating + missense
    "KEAP1": "tsg",
    "NF1": "tsg",
    "CDKN2A": "deletion",   # 9p21 homdel + truncating
}


def _gdc_get_paginated(
    endpoint: str,
    filters: dict,
    fields: str,
    max_results: int = 50000,
) -> list[dict]:
    """Paginated GET from GDC API. Returns all hits."""
    all_hits: list[dict] = []
    offset = 0

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "size": min(PAGE_SIZE, max_results - len(all_hits)),
            "from": offset,
            "format": "json",
        }
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"{GDC_API}/{endpoint}", params=params, timeout=TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"GDC API failed: {endpoint} — {e}") from e

        hits = data["data"]["hits"]
        all_hits.extend(hits)

        total = data["data"]["pagination"]["total"]
        if len(all_hits) >= total or len(all_hits) >= max_results:
            break
        offset += PAGE_SIZE
        time.sleep(0.2)

    return all_hits


def fetch_all_tcga_patients() -> pd.DataFrame:
    """Fetch all TCGA cases with project IDs (cancer types)."""
    filters = {
        "op": "=",
        "content": {"field": "project.program.name", "value": "TCGA"},
    }
    hits = _gdc_get_paginated(
        "cases",
        filters,
        "submitter_id,project.project_id",
        max_results=15000,
    )
    rows = []
    for h in hits:
        pid = h.get("submitter_id", "")
        project = h.get("project", {}).get("project_id", "")
        cancer_type = project.replace("TCGA-", "") if project.startswith("TCGA-") else project
        rows.append({"patient_id": pid, "cancer_type": cancer_type})
    return pd.DataFrame(rows).drop_duplicates(subset="patient_id")


def fetch_mutations_for_gene(gene: str) -> list[dict]:
    """Fetch all TCGA somatic mutation occurrences for a single gene."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "case.project.program.name", "value": "TCGA"}},
            {"op": "=", "content": {
                "field": "ssm.consequence.transcript.gene.symbol", "value": gene,
            }},
        ],
    }
    fields = (
        "case.submitter_id,case.project.project_id,"
        "ssm.consequence.transcript.consequence_type,"
        "ssm.consequence.transcript.aa_change,"
        "ssm.consequence.transcript.gene.symbol"
    )
    return _gdc_get_paginated("ssm_occurrences", filters, fields)


def fetch_homdel_for_gene(gene: str) -> list[dict]:
    """Fetch TCGA CNV occurrences with homozygous deletion for a gene."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "case.project.program.name", "value": "TCGA"}},
            {"op": "=", "content": {
                "field": "cnv.consequence.gene.symbol", "value": gene,
            }},
            {"op": "=", "content": {
                "field": "cnv.cnv_change_5_category", "value": "Homozygous Deletion",
            }},
        ],
    }
    fields = "case.submitter_id,case.project.project_id,cnv.cnv_change_5_category"
    return _gdc_get_paginated("cnv_occurrences", filters, fields)


def _extract_consequence_types(hit: dict) -> set[str]:
    """Extract unique consequence types from an SSM occurrence hit."""
    types = set()
    for csq in hit.get("ssm", {}).get("consequence", []):
        ct = csq.get("transcript", {}).get("consequence_type", "")
        if ct:
            types.add(ct)
    return types


def _extract_aa_changes(hit: dict) -> list[str]:
    """Extract amino acid changes from an SSM occurrence hit."""
    changes = []
    for csq in hit.get("ssm", {}).get("consequence", []):
        aa = csq.get("transcript", {}).get("aa_change", "")
        if aa:
            changes.append(aa)
    return changes


def _is_pik3ca_hotspot(aa_changes: list[str]) -> bool:
    """Check if any amino acid change is at a PIK3CA hotspot residue."""
    for aa in aa_changes:
        clean = aa.lstrip("p.")
        if any(clean.startswith(hp) for hp in PIK3CA_HOTSPOT_PREFIXES):
            return True
    return False


def classify_mutations(gene: str, hits: list[dict]) -> set[str]:
    """Classify which patients have damaging mutations in a gene.

    Returns set of patient IDs with qualifying mutations.
    """
    rule = GENE_RULES[gene]
    altered_patients: set[str] = set()

    for hit in hits:
        patient_id = hit.get("case", {}).get("submitter_id", "")
        if not patient_id:
            continue

        csq_types = _extract_consequence_types(hit)

        if rule == "tsg" or rule == "deletion":
            # Truncating mutations
            if csq_types & LOF_CONSEQUENCES:
                altered_patients.add(patient_id)

        elif rule == "tp53":
            # Truncating OR missense (TP53 missense are generally pathogenic)
            if csq_types & LOF_CONSEQUENCES:
                altered_patients.add(patient_id)
            elif "missense_variant" in csq_types:
                altered_patients.add(patient_id)

        elif rule == "oncogene":
            # PIK3CA: hotspot missense only
            if "missense_variant" in csq_types:
                aa_changes = _extract_aa_changes(hit)
                if _is_pik3ca_hotspot(aa_changes):
                    altered_patients.add(patient_id)

    return altered_patients


def extract_homdel_patients(hits: list[dict]) -> set[str]:
    """Extract patient IDs with homozygous deletions from CNV hits."""
    patients: set[str] = set()
    for hit in hits:
        patient_id = hit.get("case", {}).get("submitter_id", "")
        if patient_id:
            patients.add(patient_id)
    return patients


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1a: TCGA Pan-Cancer Co-Alteration Matrix ===\n")

    # Step 1: Get all TCGA patients
    print("Fetching all TCGA cases...")
    patients_df = fetch_all_tcga_patients()
    all_patients = set(patients_df["patient_id"])
    patient_cancer_type = dict(zip(patients_df["patient_id"], patients_df["cancer_type"]))
    n_types = patients_df["cancer_type"].nunique()
    print(f"  {len(all_patients)} patients across {n_types} cancer types\n")

    # Step 2: For each gene, fetch mutations and classify
    gene_altered: dict[str, set[str]] = {}

    print("Fetching mutation data per gene...")
    for gene in TARGET_GENES:
        print(f"  {gene}...", end=" ", flush=True)
        mut_hits = fetch_mutations_for_gene(gene)
        mut_patients = classify_mutations(gene, mut_hits)
        print(f"{len(mut_hits)} SSM occurrences, {len(mut_patients)} patients with damaging mutations")
        gene_altered[gene] = mut_patients
        time.sleep(0.3)

    # Step 3: For each gene, fetch homozygous deletions
    print("\nFetching homozygous deletion data per gene...")
    for gene in TARGET_GENES:
        print(f"  {gene}...", end=" ", flush=True)
        homdel_hits = fetch_homdel_for_gene(gene)
        homdel_patients = extract_homdel_patients(homdel_hits)
        print(f"{len(homdel_patients)} patients with homdel")
        # Merge mutation and homdel patients
        gene_altered[gene] = gene_altered[gene] | homdel_patients
        time.sleep(0.3)

    # Step 4: Build binary alteration matrix
    print("\nBuilding per-patient binary alteration matrix...")
    rows = []
    for patient_id in sorted(all_patients):
        row = {
            "patient_id": patient_id,
            "cancer_type": patient_cancer_type.get(patient_id, ""),
        }
        for gene in TARGET_GENES:
            row[gene] = 1 if patient_id in gene_altered[gene] else 0
        rows.append(row)

    matrix = pd.DataFrame(rows)

    # Step 5: Save outputs
    out_path = OUTPUT_DIR / "phase1_alteration_matrix.csv"
    matrix.to_csv(out_path, index=False)
    print(f"Saved alteration matrix: {out_path}")

    # Step 6: Summary statistics
    n_patients = len(matrix)
    n_types_actual = matrix["cancer_type"].nunique()

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Patients:     {n_patients}")
    print(f"  Cancer types: {n_types_actual}")
    print(f"\nPer-gene alteration frequencies:")
    for gene in TARGET_GENES:
        n_alt = int(matrix[gene].sum())
        pct = n_alt / n_patients * 100
        print(f"  {gene:10s}  {n_alt:5d} ({pct:5.1f}%)")

    # Combined BRCA1/2
    brca_any = int(((matrix["BRCA1"] == 1) | (matrix["BRCA2"] == 1)).sum())
    print(f"  {'BRCA1/2':10s}  {brca_any:5d} ({brca_any / n_patients * 100:5.1f}%)")

    # Validation checks
    print(f"\nValidation:")
    print(f"  Patients >= 10,000: {'PASS' if n_patients >= 10000 else 'FAIL'} ({n_patients})")
    print(f"  Cancer types >= 20: {'PASS' if n_types_actual >= 20 else 'FAIL'} ({n_types_actual})")

    # Save summary JSON
    summary = {
        "n_patients": n_patients,
        "n_cancer_types": n_types_actual,
        "genes": TARGET_GENES,
        "gene_alteration_counts": {g: int(matrix[g].sum()) for g in TARGET_GENES},
        "gene_alteration_pcts": {
            g: round(matrix[g].sum() / n_patients * 100, 2) for g in TARGET_GENES
        },
        "cancer_type_counts": matrix["cancer_type"].value_counts().to_dict(),
        "validation": {
            "patients_gte_10000": n_patients >= 10000,
            "types_gte_20": n_types_actual >= 20,
        },
    }
    summary_path = OUTPUT_DIR / "phase1_alteration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
