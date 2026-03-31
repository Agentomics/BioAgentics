"""Download TCGA expression + mutation data for expanded cohort analysis.

Downloads expression data per-file in memory, extracts only target gene TPMs,
and saves a slim matrix (~100KB per cohort instead of ~2GB). Downloads mutation
MAF files to disk (small: 15-186 MB per cohort).

Usage:
    PYTHONPATH=src/cancer:src uv run python src/cancer/swisnf_metabolic_convergence/download_expanded_tcga.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from bioagentics.config import REPO_ROOT
from bioagentics.data.download_tcga import (
    GDC_API,
    TIMEOUT,
    DATA_TYPES,
    download_gdc_file,
    query_files,
    save_manifest,
)

PHASE2_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase2"
TCGA_DIR = REPO_ROOT / "data" / "tcga"

# Cohorts to download — high ARID1A/SMARCA4 mutation frequency
COHORTS = ["UCEC", "OV", "STAD", "COAD"]


def load_target_genes() -> list[str]:
    """Load the convergent metabolic genes from Phase 2 + SWI/SNF genes."""
    convergent_df = pd.read_csv(PHASE2_DIR / "convergent_metabolic_genes.csv")
    genes = convergent_df["gene"].tolist()
    # Also include SWI/SNF genes for QC
    genes.extend(["ARID1A", "SMARCA4", "ARID1B", "SMARCA2"])
    return list(set(genes))


def parse_expression_bytes(data: bytes, target_genes: set[str]) -> dict[str, object]:
    """Parse a STAR-Counts TSV from bytes, extract TPM for target genes."""
    text = data.decode("utf-8")
    result = {}
    for line in text.split("\n"):
        if line.startswith("#") or line.startswith("N_"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        gene_name = parts[1]  # gene_name column
        gene_type = parts[2]  # gene_type column
        if gene_type != "protein_coding":
            continue
        if gene_name not in target_genes:
            continue
        try:
            tpm = float(parts[6])  # tpm_unstranded column
        except (ValueError, IndexError):
            continue
        if gene_name not in result:
            result[gene_name] = tpm
    return result


def resolve_uuids_to_patients(file_ids: list[str]) -> dict[str, str]:
    """Query GDC API to map file UUIDs to patient barcodes (TCGA-XX-XXXX)."""
    file_to_patient: dict[str, str] = {}
    for i in range(0, len(file_ids), 500):
        batch = file_ids[i : i + 500]
        filt = {
            "op": "in",
            "content": {"field": "file_id", "value": batch},
        }
        payload = {
            "filters": json.dumps(filt),
            "fields": "file_id,cases.submitter_id",
            "size": len(batch),
            "format": "json",
        }
        resp = requests.post(f"{GDC_API}/files", json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        for hit in resp.json()["data"]["hits"]:
            cases = hit.get("cases", [])
            if cases:
                file_to_patient[hit["file_id"]] = cases[0]["submitter_id"]
    return file_to_patient


def download_slim_expression(
    cohort: str,
    target_genes: list[str],
    output_dir: Path,
) -> Path | None:
    """Download expression data for a cohort, extracting only target genes.

    Returns path to the saved slim matrix CSV, or None on failure.
    """
    project_id = f"TCGA-{cohort}"
    cfg = DATA_TYPES["expression"]
    slim_path = output_dir / f"{cohort.lower()}_expression_slim.csv"

    if slim_path.exists():
        print(f"  Slim matrix already exists: {slim_path}")
        return slim_path

    print(f"  Querying GDC for {project_id} expression files...")
    files = query_files(project_id, cfg)
    if not files:
        print(f"  No expression files found for {project_id}")
        return None

    print(f"  Found {len(files)} expression files")

    # Resolve file UUIDs to patient barcodes
    print("  Resolving UUIDs to patient barcodes...")
    file_to_patient = resolve_uuids_to_patients([f["file_id"] for f in files])
    print(f"  Mapped {len(file_to_patient)}/{len(files)} files to patients")

    # Download each file in memory, extract target genes
    target_set = set(target_genes)
    rows: list[dict[str, object]] = []
    failed = 0

    for entry in tqdm(files, desc=f"  {cohort} expression", unit="file"):
        fid = entry["file_id"]
        patient = file_to_patient.get(fid)
        if not patient:
            continue

        url = f"{GDC_API}/data/{fid}"
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            failed += 1
            if failed > 10:
                print(f"\n  Too many failures ({failed}), stopping")
                break
            continue

        gene_tpms: dict[str, object] = parse_expression_bytes(resp.content, target_set)
        gene_tpms["patient"] = patient
        rows.append(gene_tpms)

        # Rate limit to avoid GDC throttling
        time.sleep(0.05)

    if not rows:
        print(f"  No expression data extracted for {cohort}")
        return None

    # Build matrix: patients x genes
    df = pd.DataFrame(rows).set_index("patient")
    # Remove duplicate patients (keep first)
    df = df[~df.index.duplicated(keep="first")]

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(slim_path)
    print(f"  Saved slim matrix: {df.shape[0]} patients x {df.shape[1]} genes -> {slim_path}")

    if failed:
        print(f"  ({failed} files failed to download)")

    return slim_path


def download_mutations(cohort: str, output_dir: Path) -> Path | None:
    """Download mutation MAF files for a cohort."""
    project_id = f"TCGA-{cohort}"
    cfg = DATA_TYPES["mutations"]
    mut_dir = output_dir / "mutations"

    # Check if already downloaded
    if mut_dir.exists():
        existing = list(mut_dir.glob("*.maf.gz"))
        if existing:
            print(f"  Mutations already downloaded: {len(existing)} files in {mut_dir}")
            return mut_dir

    print(f"  Querying GDC for {project_id} mutation files...")
    files = query_files(project_id, cfg)
    if not files:
        print(f"  No mutation files found for {project_id}")
        return None

    total_mb = sum(f["file_size"] for f in files) / 1e6
    print(f"  Found {len(files)} MAF files ({total_mb:.0f} MB)")

    mut_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(files, mut_dir / "manifest.tsv")

    failed = 0
    for entry in tqdm(files, desc=f"  {cohort} mutations", unit="file"):
        try:
            download_gdc_file(
                entry["file_id"],
                entry["file_name"],
                mut_dir,
                expected_md5=entry.get("md5sum"),
            )
        except (requests.RequestException, ValueError, OSError) as e:
            failed += 1
            if failed > 5:
                print(f"\n  Too many failures ({failed}), stopping")
                break

    if failed:
        print(f"  {failed}/{len(files)} mutation files failed")
        return None

    print(f"  Downloaded {len(files) - failed} mutation files to {mut_dir}")
    return mut_dir


def main() -> None:
    print("=== Download Expanded TCGA Cohort Data ===\n")

    target_genes = load_target_genes()
    print(f"Target genes: {len(target_genes)}")

    for cohort in COHORTS:
        print(f"\n{'='*60}")
        print(f"COHORT: {cohort}")
        print(f"{'='*60}")

        cohort_dir = TCGA_DIR / cohort.lower()
        cohort_dir.mkdir(parents=True, exist_ok=True)

        # Download slim expression matrix
        slim_path = download_slim_expression(cohort, target_genes, cohort_dir)
        if slim_path:
            print(f"  Expression: OK")
        else:
            print(f"  Expression: FAILED")

        # Download mutations
        mut_dir = download_mutations(cohort, cohort_dir)
        if mut_dir:
            print(f"  Mutations: OK")
        else:
            print(f"  Mutations: FAILED")

    print("\nDone.")


if __name__ == "__main__":
    main()
