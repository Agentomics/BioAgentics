"""Download and munge 2019 PGC TS GWAS (Yu et al.) to LDSC format.

Converts the PGC TS GWAS summary statistics (TS_Oct2018.gz) from
OR/SE format to LDSC-compatible Z-score format (SNP, A1, A2, Z, N).

Source: figshare.com/articles/dataset/ts2019/14672232
GWAS Catalog: GCST007277
PubMed: 30818990
Sample: 4,819 cases + 9,488 controls = 14,307 total (European)

Usage:
    uv run python -m bioagentics.data.tourettes.neuroimmune.munge_ts_gwas
    uv run python -m bioagentics.data.tourettes.neuroimmune.munge_ts_gwas --download
"""

from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-neuroimmune-subtyping" / "ts_gwas_2019"
RAW_FILE = DATA_DIR / "TS_Oct2018.gz"
LDSC_FILE = DATA_DIR / "ts_yu_2019.ldsc.tsv.gz"

FIGSHARE_URL = "https://ndownloader.figshare.com/files/28169940"
FIGSHARE_README_URL = "https://ndownloader.figshare.com/files/28169937"

N_CASES = 4819
N_CONTROLS = 9488
N_TOTAL = N_CASES + N_CONTROLS

# Effective sample size for case-control: 4 / (1/Ncases + 1/Ncontrols)
N_EFFECTIVE = 4 / (1 / N_CASES + 1 / N_CONTROLS)

INFO_THRESHOLD = 0.9
CHUNK_SIZE = 500_000


def download_gwas(force: bool = False) -> Path:
    """Download TS GWAS from figshare if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and not force:
        logger.info("Raw GWAS file already exists: %s", RAW_FILE)
        return RAW_FILE

    logger.info("Downloading TS GWAS from figshare (178 MB)...")
    resp = requests.get(FIGSHARE_URL, stream=True, timeout=600)
    resp.raise_for_status()
    with open(RAW_FILE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    logger.info("Downloaded: %s (%.1f MB)", RAW_FILE.name, RAW_FILE.stat().st_size / 1e6)

    # Also download readme
    readme_path = DATA_DIR / "ts2019-readme.docx"
    if not readme_path.exists():
        resp2 = requests.get(FIGSHARE_README_URL, timeout=60)
        resp2.raise_for_status()
        readme_path.write_bytes(resp2.content)

    return RAW_FILE


def munge_to_ldsc(force: bool = False) -> Path:
    """Convert TS GWAS to LDSC format using chunked processing.

    Input columns: SNP CHR BP A1 A2 INFO OR SE P (space-separated)
    Output columns: SNP A1 A2 Z N (tab-separated, gzipped)

    Z = log(OR) / SE
    N = effective sample size (4 / (1/Ncases + 1/Ncontrols))
    Filters: INFO >= 0.9, valid OR/SE, no duplicates
    """
    if LDSC_FILE.exists() and not force:
        logger.info("LDSC file already exists: %s", LDSC_FILE)
        return LDSC_FILE

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw GWAS file not found: {RAW_FILE}")

    logger.info("Munging TS GWAS to LDSC format (chunked, N_eff=%.0f)...", N_EFFECTIVE)

    total_read = 0
    total_written = 0
    tmp_path = LDSC_FILE.with_suffix(".tmp.gz")

    with gzip.open(tmp_path, "wt") as out_f:
        out_f.write("SNP\tA1\tA2\tZ\tN\n")

        for chunk in pd.read_csv(
            RAW_FILE,
            sep=r"\s+",
            dtype={"SNP": str, "CHR": str, "A1": str, "A2": str},
            chunksize=CHUNK_SIZE,
        ):
            total_read += len(chunk)

            # Filter by INFO score
            chunk = chunk[chunk["INFO"].astype(float) >= INFO_THRESHOLD]

            # Drop rows with invalid OR or SE
            chunk = chunk.dropna(subset=["OR", "SE"])
            chunk["OR"] = pd.to_numeric(chunk["OR"], errors="coerce")
            chunk["SE"] = pd.to_numeric(chunk["SE"], errors="coerce")
            chunk = chunk.dropna(subset=["OR", "SE"])
            chunk = chunk[(chunk["OR"] > 0) & (chunk["SE"] > 0)]

            # Compute Z score: log(OR) / SE
            chunk["Z"] = np.log(chunk["OR"]) / chunk["SE"]

            # Filter infinite/NaN Z scores
            chunk = chunk[np.isfinite(chunk["Z"])]

            # Drop duplicate SNPs within chunk
            chunk = chunk.drop_duplicates(subset=["SNP"], keep="first")

            # Write LDSC format
            chunk["N"] = int(round(N_EFFECTIVE))
            out = chunk[["SNP", "A1", "A2", "Z", "N"]]
            out.to_csv(out_f, sep="\t", index=False, header=False)
            total_written += len(out)

            if total_read % 2_000_000 < CHUNK_SIZE:
                logger.info("  Processed %d SNPs, written %d...", total_read, total_written)

    tmp_path.rename(LDSC_FILE)
    logger.info(
        "Done: %d/%d SNPs written to %s",
        total_written,
        total_read,
        LDSC_FILE.name,
    )
    return LDSC_FILE


def validate_ldsc(path: Path | None = None) -> dict:
    """Validate the munged LDSC file and report QC metrics."""
    path = path or LDSC_FILE
    if not path.exists():
        raise FileNotFoundError(f"LDSC file not found: {path}")

    df = pd.read_csv(path, sep="\t", nrows=100_000)

    # Count total SNPs (streaming)
    total_snps = 0
    sum_z2 = 0.0
    for chunk in pd.read_csv(path, sep="\t", chunksize=CHUNK_SIZE):
        total_snps += len(chunk)
        sum_z2 += (chunk["Z"] ** 2).sum()

    mean_chi2 = sum_z2 / total_snps
    # Median from sample
    median_chi2 = float(np.median(df["Z"] ** 2))
    lambda_gc = median_chi2 / 0.4549364  # chi2(1) median

    stats = {
        "file": str(path.name),
        "total_snps": total_snps,
        "mean_chi_sq": round(mean_chi2, 4),
        "median_chi_sq": round(median_chi2, 4),
        "lambda_gc": round(lambda_gc, 4),
        "n_effective": int(round(N_EFFECTIVE)),
        "n_cases": N_CASES,
        "n_controls": N_CONTROLS,
        "columns": list(df.columns),
    }

    logger.info("Validation results:")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Munge 2019 PGC TS GWAS to LDSC format")
    parser.add_argument("--download", action="store_true", help="Download raw GWAS from figshare")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    args = parser.parse_args()

    if args.validate_only:
        validate_ldsc()
        return

    if args.download:
        download_gwas(force=args.force)

    munge_to_ldsc(force=args.force)
    validate_ldsc()


if __name__ == "__main__":
    main()
