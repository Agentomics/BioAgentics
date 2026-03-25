"""Parse GSE93624 series matrix and download expression data.

GSE93624 (Marigorta et al. 2017, PMID:28805827) is a subset of the RISK
cohort with 210 CD + 35 non-IBD controls. Critically, it includes a binary
'progression to complication' outcome (27 progressors to B2/B3 at 3-year
follow-up), replacing the deep_ulcer surrogate from GSE57945.

Per-sample supplementary files contain TMM-normalized log-expression values
for 13,769 genes. This script downloads them and builds a combined matrix.

Outputs:
  - data/.../processed/gse93624_phenotype.tsv
  - data/.../processed/gse93624_expression.tsv.gz

Usage:
    uv run python -m crohns.cd_stricture_risk_prediction.04_gse93624_phenotype
"""

from __future__ import annotations

import gzip
import io
import sys
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "crohns" / "cd-stricture-risk-prediction"
SERIES_MATRIX = DATA_DIR / "risk-cohort" / "GSE93624_series_matrix.txt"
SAMPLE_DIR = DATA_DIR / "risk-cohort" / "GSE93624"
OUTPUT_DIR = DATA_DIR / "processed"


def parse_series_matrix(path: Path) -> pd.DataFrame:
    """Parse GSE93624 series matrix to extract sample metadata.

    Returns DataFrame indexed by GSM ID with clinical fields.
    """
    fields: dict[str, list[str]] = {}
    gsm_ids: list[str] = []

    with open(path) as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                gsm_ids = [p.strip('"') for p in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = line.strip().split("\t")
                vals = [p.strip('"') for p in parts[1:]]
                if ":" in vals[0]:
                    key = vals[0].split(":")[0].strip()
                    values = [v.split(":", 1)[1].strip() for v in vals]
                    fields[key] = values
            elif line.startswith("!Sample_supplementary_file_1"):
                parts = line.strip().split("\t")
                fields["suppl_url"] = [p.strip('"') for p in parts[1:]]

    df = pd.DataFrame(fields, index=gsm_ids)
    df.index.name = "gsm_id"
    return df


def build_phenotype_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Build clean phenotype table from parsed metadata."""
    pheno = pd.DataFrame(index=raw.index)
    pheno.index.name = "gsm_id"

    pheno["diagnosis"] = raw["diagnosis"]
    pheno["is_cd"] = raw["diagnosis"].str.contains("Crohn", case=False)
    pheno["gender"] = raw["gender"]
    pheno["age_at_diagnosis"] = pd.to_numeric(raw["age at diagnosis"], errors="coerce")
    pheno["paris_age"] = raw["paris age"]
    pheno["ancestry"] = raw.get("ancestry", pd.Series(dtype=str))
    pheno["tissue"] = raw.get("tissue", pd.Series(dtype=str))

    # Complication outcome — the key field
    prog = raw["progression to complication"].str.strip()
    pheno["complication_progressed"] = prog.map({"yes": True, "-": False})

    return pheno


def download_sample_expression(
    gsm_id: str,
    url: str,
    cache_dir: Path,
    max_retries: int = 3,
) -> pd.Series | None:
    """Download and parse a single per-sample expression file.

    Each file has two columns: gene, value (TMM-normalized log-expression).
    Returns a Series indexed by gene name, or None on failure.
    """
    # Convert ftp:// to https:// for reliability
    url = url.replace(
        "ftp://ftp.ncbi.nlm.nih.gov",
        "https://ftp.ncbi.nlm.nih.gov",
    )

    # Check cache first
    filename = url.split("/")[-1]
    cache_path = cache_dir / filename
    if cache_path.exists():
        with gzip.open(cache_path, "rt") as f:
            df = pd.read_csv(f, sep="\t", index_col=0)
        return df.iloc[:, 0]

    for attempt in range(max_retries):
        try:
            with urlopen(url, timeout=30) as resp:
                data = resp.read()
            # Cache to disk
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(data)
            # Parse
            with gzip.open(io.BytesIO(data), "rt") as f:
                df = pd.read_csv(f, sep="\t", index_col=0)
            return df.iloc[:, 0]
        except (URLError, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED {gsm_id}: {e}", file=sys.stderr)
                return None


def build_expression_matrix(
    pheno_raw: pd.DataFrame,
    cache_dir: Path,
) -> pd.DataFrame:
    """Download per-sample files and combine into genes x samples matrix.

    Downloads sequentially to respect memory and network constraints.
    """
    all_series: dict[str, pd.Series] = {}
    n_total = len(pheno_raw)

    for i, (gsm_id, row) in enumerate(pheno_raw.iterrows()):
        url = row["suppl_url"]
        if i % 25 == 0:
            print(f"  Downloading {i+1}/{n_total}...")
        series = download_sample_expression(gsm_id, url, cache_dir)
        if series is not None:
            all_series[gsm_id] = series

    print(f"  Downloaded: {len(all_series)}/{n_total} samples")

    # Combine into genes x samples DataFrame
    expr = pd.DataFrame(all_series)
    expr.index.name = "gene"
    return expr


def report_summary(pheno: pd.DataFrame) -> None:
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("GSE93624 Phenotype Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(pheno)}")
    print(f"\nDiagnosis:")
    print(pheno["diagnosis"].value_counts().to_string())
    cd = pheno[pheno["is_cd"]]
    print(f"\nCD patients: {len(cd)}")
    print(f"\nGender (CD):")
    print(cd["gender"].value_counts().to_string())
    print(f"\nAge at diagnosis (CD): mean={cd['age_at_diagnosis'].mean():.1f}, "
          f"median={cd['age_at_diagnosis'].median():.1f}")
    print(f"\nParis age (CD):")
    print(cd["paris_age"].value_counts().to_string())
    print(f"\nComplication progression (CD):")
    print(cd["complication_progressed"].value_counts().to_string())
    n_prog = cd["complication_progressed"].sum()
    n_nonprog = (~cd["complication_progressed"]).sum()
    print(f"  -> {n_prog} progressors, {n_nonprog} non-progressors")
    print(f"  -> Prevalence: {n_prog/len(cd):.1%}")


def main() -> None:
    if not SERIES_MATRIX.exists():
        print(f"ERROR: Series matrix not found: {SERIES_MATRIX}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse metadata
    print("Parsing GSE93624 series matrix...")
    raw = parse_series_matrix(SERIES_MATRIX)
    print(f"  {len(raw)} samples parsed")

    # Step 2: Build phenotype table
    pheno = build_phenotype_table(raw)
    report_summary(pheno)

    pheno_path = OUTPUT_DIR / "gse93624_phenotype.tsv"
    pheno.to_csv(pheno_path, sep="\t")
    print(f"\nPhenotype table saved: {pheno_path}")

    # Step 3: Download and combine expression data
    print(f"\nDownloading per-sample expression files...")
    expr = build_expression_matrix(raw, SAMPLE_DIR)
    print(f"Expression matrix: {expr.shape[0]} genes x {expr.shape[1]} samples")

    expr_path = OUTPUT_DIR / "gse93624_expression.tsv.gz"
    expr.to_csv(expr_path, sep="\t", compression="gzip")
    print(f"Expression matrix saved: {expr_path}")

    # Step 4: Report sample counts with expression data
    cd_gsm = set(pheno[pheno["is_cd"]].index)
    expr_gsm = set(expr.columns)
    cd_with_expr = cd_gsm & expr_gsm
    print(f"\nCD patients with expression data: {len(cd_with_expr)}")
    # Progressors with expression data
    cd_pheno = pheno[pheno["is_cd"] & pheno.index.isin(expr_gsm)]
    n_prog_expr = cd_pheno["complication_progressed"].sum()
    n_nonprog_expr = (~cd_pheno["complication_progressed"]).sum()
    print(f"  Progressors with expression: {n_prog_expr}")
    print(f"  Non-progressors with expression: {n_nonprog_expr}")


if __name__ == "__main__":
    main()
