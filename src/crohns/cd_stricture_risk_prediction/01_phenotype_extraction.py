"""Extract phenotype data from the RISK cohort (GSE57945).

Parses clinical metadata from the GEO series matrix and SOFT files to build
a phenotype table for the stricture risk prediction model.

Available clinical fields from GEO:
  - diagnosis (CD, UC, Not IBD)
  - sex
  - age_at_diagnosis
  - paris_age (A1a: <10y, A1b: 10-17y)
  - disease_location (l2_type: iCD=ileal, cCD=colonic)
  - histopathology (macroscopic/microscopic inflammation, normal)
  - deep_ulcer (yes/no) — proxy for complicated disease course

NOTE: Montreal behavior classification (B1/B2/B3) and disease progression
outcomes are NOT available in the GEO deposit. These were published in
Kugathasan et al. 2017 (Lancet) and need to be obtained separately.
Until then, deep_ulcer is used as a surrogate outcome.

Usage:
    uv run python -m crohns.cd_stricture_risk_prediction.01_phenotype_extraction
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "crohns" / "cd-stricture-risk-prediction"
GSE57945_DIR = DATA_DIR / "risk-cohort" / "GSE57945"
OUTPUT_DIR = DATA_DIR / "processed"


def parse_soft_metadata(soft_path: Path) -> pd.DataFrame:
    """Parse full sample metadata from the SOFT file.

    Returns a DataFrame with one row per sample, columns for each
    clinical characteristic.
    """
    samples: dict[str, dict] = {}
    current_sample: str | None = None

    opener = gzip.open if str(soft_path).endswith(".gz") else open
    with opener(soft_path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("^SAMPLE"):
                current_sample = line.split("=")[-1].strip()
                samples[current_sample] = {}
            elif current_sample is None:
                continue
            elif line.startswith("!Sample_title"):
                samples[current_sample]["title"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_source_name"):
                samples[current_sample]["source"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                val = line.split("=", 1)[-1].strip()
                if ":" in val:
                    key, v = val.split(":", 1)
                    samples[current_sample][key.strip()] = v.strip()
            elif line.startswith("^PLATFORM") or line.startswith("^SERIES"):
                current_sample = None

    df = pd.DataFrame.from_dict(samples, orient="index")
    df.index.name = "gsm_id"
    return df


def extract_risk_id(title: str) -> str | None:
    """Extract CCFA_Risk_XXX ID from sample title.

    Title format: 'CD Female with Macroscopic inflammation and No Deep Ulcer (CCFA_Risk_001)'
    """
    if "CCFA_Risk_" in title:
        start = title.index("CCFA_Risk_")
        end = title.index(")", start) if ")" in title[start:] else len(title)
        return title[start:end]
    return None


def build_phenotype_table(soft_path: Path) -> pd.DataFrame:
    """Build a clean phenotype table from RISK cohort metadata.

    Filters to CD patients only and standardizes field names.
    """
    raw = parse_soft_metadata(soft_path)
    print(f"Total samples in SOFT: {len(raw)}")

    # Extract CCFA Risk IDs from titles
    raw["risk_id"] = raw["title"].apply(extract_risk_id)

    # Filter to CD patients only
    cd_mask = raw["diagnosis"].str.upper() == "CD"
    cd = raw[cd_mask].copy()
    print(f"CD patients: {len(cd)}")

    # Standardize fields
    pheno = pd.DataFrame(index=cd.index)
    pheno.index.name = "gsm_id"
    pheno["risk_id"] = cd["risk_id"]
    pheno["sex"] = cd["Sex"].str.capitalize()
    pheno["age_at_diagnosis"] = pd.to_numeric(cd["age at diagnosis"], errors="coerce")
    pheno["paris_age"] = cd["paris age"]
    pheno["disease_location"] = cd["l2 type"].map({"iCD": "ileal", "cCD": "colonic"})
    pheno["histopathology"] = cd["histopathology"]

    # Deep ulcer — standardize to boolean
    deep_ulcer_raw = cd["deep ulcer"].str.strip().str.lower()
    pheno["deep_ulcer"] = deep_ulcer_raw.map({"yes": True, "no": False})
    # Leave NA as NaN

    # Placeholder columns for future progression data
    # These will be populated when B2/B3 outcome data is obtained
    pheno["behavior_at_diagnosis"] = "B1"  # All RISK patients are newly diagnosed
    pheno["behavior_at_followup"] = None  # NOT AVAILABLE in GEO
    pheno["progressed"] = None  # NOT AVAILABLE — requires Kugathasan 2017 data

    return pheno


def report_summary(pheno: pd.DataFrame) -> None:
    """Print summary statistics of the phenotype table."""
    print(f"\n{'='*60}")
    print("RISK Cohort Phenotype Summary (CD patients)")
    print(f"{'='*60}")
    print(f"Total CD patients: {len(pheno)}")
    print(f"\nSex distribution:")
    print(pheno["sex"].value_counts().to_string())
    print(f"\nAge at diagnosis: mean={pheno['age_at_diagnosis'].mean():.1f}, "
          f"median={pheno['age_at_diagnosis'].median():.1f}")
    print(f"\nParis age classification:")
    print(pheno["paris_age"].value_counts().to_string())
    print(f"\nDisease location:")
    print(pheno["disease_location"].value_counts(dropna=False).to_string())
    print(f"\nHistopathology:")
    print(pheno["histopathology"].value_counts().to_string())
    print(f"\nDeep ulcer at diagnosis:")
    print(pheno["deep_ulcer"].value_counts(dropna=False).to_string())
    print(f"\n⚠ NOTE: B2/B3 behavior progression data NOT available in GEO.")
    print("  Deep ulcer serves as surrogate outcome until Kugathasan 2017 data obtained.")


def main() -> None:
    soft_path = GSE57945_DIR / "GSE57945_family.soft.gz"
    if not soft_path.exists():
        print(f"ERROR: SOFT file not found: {soft_path}", file=sys.stderr)
        print("Run data download first or check symlink.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pheno = build_phenotype_table(soft_path)
    report_summary(pheno)

    out_path = OUTPUT_DIR / "phenotype_table.tsv"
    pheno.to_csv(out_path, sep="\t")
    print(f"\nPhenotype table saved: {out_path}")
    print(f"Shape: {pheno.shape}")


if __name__ == "__main__":
    main()
