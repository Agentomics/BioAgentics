"""Download autoimmune/inflammatory disease GWAS summary statistics for LDSC analysis.

Downloads harmonized GWAS summary statistics for 15+ autoimmune/inflammatory
diseases from GWAS Catalog FTP and OpenGWAS (IEU). Converts all files to a
standard LDSC-compatible format (SNP, A1, A2, Z, N) and generates a manifest.

Target diseases: rheumatoid arthritis, SLE, multiple sclerosis, type 1 diabetes,
IBD (Crohn + UC separately), celiac disease, psoriasis, ankylosing spondylitis,
Graves disease, Hashimoto thyroiditis, primary biliary cholangitis, myasthenia
gravis, autoimmune hepatitis, vitiligo, alopecia areata.

Output: data/tourettes/ts-neuroimmune-subtyping/autoimmune_gwas/

Usage:
    uv run python -m bioagentics.data.tourettes.neuroimmune.download_autoimmune_gwas
    uv run python -m bioagentics.data.tourettes.neuroimmune.download_autoimmune_gwas --disease ra
    uv run python -m bioagentics.data.tourettes.neuroimmune.download_autoimmune_gwas --list
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-neuroimmune-subtyping" / "autoimmune_gwas"
MANIFEST_PATH = DATA_DIR / "manifest.json"

TIMEOUT = 300  # seconds for download requests
CHUNK_SIZE = 65536


@dataclass
class GWASStudy:
    """Metadata for an autoimmune GWAS study."""

    disease: str
    abbreviation: str
    gwas_catalog_id: str
    pmid: str
    first_author: str
    year: int
    sample_size: int
    n_cases: int
    n_controls: int
    ancestry: str
    download_url: str
    filename: str
    source: str  # "gwas_catalog" or "opengwas"
    notes: str = ""


# Curated list of autoimmune/inflammatory disease GWAS with full summary statistics.
# Prioritizes large, recent, multi-ancestry or European meta-analyses with publicly
# available harmonized summary statistics on the GWAS Catalog FTP.
AUTOIMMUNE_STUDIES: list[GWASStudy] = [
    GWASStudy(
        disease="Rheumatoid arthritis",
        abbreviation="ra",
        gwas_catalog_id="GCST90132223",
        pmid="34741163",
        first_author="Ishigaki",
        year=2022,
        sample_size=276020,
        n_cases=37578,
        n_controls=238442,
        ancestry="Multi-ancestry",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000/GCST90132223/harmonised/GCST90132223.h.tsv.gz",
        filename="ra_ishigaki_2022.tsv.gz",
        source="gwas_catalog",
    ),
    GWASStudy(
        disease="Systemic lupus erythematosus",
        abbreviation="sle",
        gwas_catalog_id="GCST003156",
        pmid="26502338",
        first_author="Bentham",
        year=2015,
        sample_size=23210,
        n_cases=7219,
        n_controls=15991,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003156/harmonised/26502338-GCST003156-EFO_0002690.h.tsv.gz",
        filename="sle_bentham_2015.tsv.gz",
        source="gwas_catalog",
    ),
    GWASStudy(
        disease="Multiple sclerosis",
        abbreviation="ms",
        gwas_catalog_id="GCST90277800",
        pmid="37012484",
        first_author="IMSGC",
        year=2024,
        sample_size=174006,
        n_cases=20530,
        n_controls=153476,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90277001-GCST90278000/GCST90277800/harmonised/GCST90277800.h.tsv.gz",
        filename="ms_imsgc_2024.tsv.gz",
        source="gwas_catalog",
        notes="IMSGC discovery meta-analysis",
    ),
    GWASStudy(
        disease="Type 1 diabetes",
        abbreviation="t1d",
        gwas_catalog_id="GCST90014023",
        pmid="34012112",
        first_author="Chiou",
        year=2021,
        sample_size=520580,
        n_cases=18942,
        n_controls=501638,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90014001-GCST90015000/GCST90014023/harmonised/34012112-GCST90014023-EFO_0001359.h.tsv.gz",
        filename="t1d_chiou_2021.tsv.gz",
        source="gwas_catalog",
    ),
    GWASStudy(
        disease="Inflammatory bowel disease",
        abbreviation="ibd",
        gwas_catalog_id="GCST004131",
        pmid="28067908",
        first_author="de Lange",
        year=2017,
        sample_size=59957,
        n_cases=25042,
        n_controls=34915,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004131/harmonised/28067908-GCST004131-EFO_0003767.h.tsv.gz",
        filename="ibd_delange_2017.tsv.gz",
        source="gwas_catalog",
        notes="IBD combined (Crohn's + UC)",
    ),
    GWASStudy(
        disease="Crohn's disease",
        abbreviation="cd",
        gwas_catalog_id="GCST004132",
        pmid="28067908",
        first_author="de Lange",
        year=2017,
        sample_size=40266,
        n_cases=12194,
        n_controls=28072,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004132/harmonised/28067908-GCST004132-EFO_0000384.h.tsv.gz",
        filename="cd_delange_2017.tsv.gz",
        source="gwas_catalog",
    ),
    GWASStudy(
        disease="Ulcerative colitis",
        abbreviation="uc",
        gwas_catalog_id="GCST004133",
        pmid="28067908",
        first_author="de Lange",
        year=2017,
        sample_size=45975,
        n_cases=12366,
        n_controls=33609,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004133/harmonised/28067908-GCST004133-EFO_0000729.h.tsv.gz",
        filename="uc_delange_2017.tsv.gz",
        source="gwas_catalog",
    ),
    GWASStudy(
        disease="Celiac disease",
        abbreviation="cel",
        gwas_catalog_id="GCST005523",
        pmid="20190752",
        first_author="Dubois",
        year=2010,
        sample_size=24269,
        n_cases=4533,
        n_controls=10750,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005523/harmonised/22057235-GCST005523-EFO_0001060.h.tsv.gz",
        filename="cel_dubois_2010.tsv.gz",
        source="gwas_catalog",
        notes="Immunochip + GWAS meta-analysis",
    ),
    GWASStudy(
        disease="Psoriasis",
        abbreviation="pso",
        gwas_catalog_id="GCST005527",
        pmid="29083406",
        first_author="Tsoi",
        year=2017,
        sample_size=39051,
        n_cases=13229,
        n_controls=25822,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005527/harmonised/23143594-GCST005527-EFO_0000676.h.tsv.gz",
        filename="pso_tsoi_2017.tsv.gz",
        source="gwas_catalog",
    ),
    GWASStudy(
        disease="Ankylosing spondylitis",
        abbreviation="as",
        gwas_catalog_id="GCST005529",
        pmid="26974007",
        first_author="IGAS",
        year=2016,
        sample_size=33720,
        n_cases=9069,
        n_controls=24651,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005529/harmonised/23749187-GCST005529-EFO_0003898.h.tsv.gz",
        filename="as_igas_2016.tsv.gz",
        source="gwas_catalog",
    ),
    GWASStudy(
        disease="Graves' disease",
        abbreviation="gd",
        gwas_catalog_id="GCST90018885",
        pmid="33440007",
        first_author="Sakaue",
        year=2021,
        sample_size=458620,
        n_cases=3524,
        n_controls=455096,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90018001-GCST90019000/GCST90018885/harmonised/GCST90018885.h.tsv.gz",
        filename="gd_sakaue_2021.tsv.gz",
        source="gwas_catalog",
        notes="FinnGen+UKB+BBJ meta-analysis (EUR subset)",
    ),
    GWASStudy(
        disease="Primary biliary cholangitis",
        abbreviation="pbc",
        gwas_catalog_id="GCST90061440",
        pmid="34033851",
        first_author="Cordell",
        year=2022,
        sample_size=24510,
        n_cases=8021,
        n_controls=16489,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90061001-GCST90062000/GCST90061440/harmonised/34033851-GCST90061440-EFO_1001486.h.tsv.gz",
        filename="pbc_cordell_2022.tsv.gz",
        source="gwas_catalog",
    ),
    GWASStudy(
        disease="Vitiligo",
        abbreviation="vit",
        gwas_catalog_id="GCST90018935",
        pmid="33440007",
        first_author="Sakaue",
        year=2021,
        sample_size=458620,
        n_cases=3524,
        n_controls=455096,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90018001-GCST90019000/GCST90018935/harmonised/GCST90018935.h.tsv.gz",
        filename="vit_sakaue_2021.tsv.gz",
        source="gwas_catalog",
        notes="FinnGen+UKB+BBJ meta-analysis (EUR subset)",
    ),
    GWASStudy(
        disease="Alopecia areata",
        abbreviation="aa",
        gwas_catalog_id="GCST90018875",
        pmid="33440007",
        first_author="Sakaue",
        year=2021,
        sample_size=459208,
        n_cases=4112,
        n_controls=455096,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90018001-GCST90019000/GCST90018875/harmonised/GCST90018875.h.tsv.gz",
        filename="aa_sakaue_2021.tsv.gz",
        source="gwas_catalog",
        notes="FinnGen+UKB+BBJ meta-analysis (EUR subset)",
    ),
    GWASStudy(
        disease="Myasthenia gravis",
        abbreviation="mg",
        gwas_catalog_id="GCST90018897",
        pmid="33440007",
        first_author="Sakaue",
        year=2021,
        sample_size=456027,
        n_cases=931,
        n_controls=455096,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90018001-GCST90019000/GCST90018897/harmonised/GCST90018897.h.tsv.gz",
        filename="mg_sakaue_2021.tsv.gz",
        source="gwas_catalog",
        notes="FinnGen+UKB+BBJ meta-analysis (EUR subset)",
    ),
    GWASStudy(
        disease="Hashimoto thyroiditis",
        abbreviation="ht",
        gwas_catalog_id="GCST90018890",
        pmid="33440007",
        first_author="Sakaue",
        year=2021,
        sample_size=462245,
        n_cases=7149,
        n_controls=455096,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90018001-GCST90019000/GCST90018890/harmonised/GCST90018890.h.tsv.gz",
        filename="ht_sakaue_2021.tsv.gz",
        source="gwas_catalog",
        notes="FinnGen+UKB+BBJ meta-analysis (EUR subset)",
    ),
]

# Column name mappings for GWAS Catalog harmonized files
GWAS_CATALOG_COL_MAP = {
    "hm_rsid": "SNP",
    "hm_other_allele": "A2",
    "hm_effect_allele": "A1",
    "hm_beta": "BETA",
    "hm_odds_ratio": "OR",
    "standard_error": "SE",
    "p_value": "P",
    "hm_effect_allele_frequency": "FRQ",
    "hm_variant_id": "VARIANT_ID",
    "hm_chrom": "CHR",
    "hm_pos": "BP",
}


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file with streaming, skip if cached."""
    if dest.exists() and not force:
        logger.info("  Cached: %s", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    logger.info("  Downloading: %s", url.split("/")[-1])
    try:
        resp = requests.get(url, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("  Saved: %s (%.1f MB)", dest.name, size_mb)
        return True
    except requests.RequestException as e:
        logger.error("  Download failed: %s — %s", url, e)
        if tmp.exists():
            tmp.unlink()
        return False


def _resolve_columns(columns: list[str]) -> dict:
    """Resolve column names from a header, returning a mapping of roles to actual names."""
    col_lower = {c.lower(): c for c in columns}

    snp_col = None
    for candidate in ["hm_rsid", "rsid", "snp", "variant_id", "hm_variant_id"]:
        if candidate in col_lower:
            snp_col = col_lower[candidate]
            break

    a1_col = col_lower.get("hm_effect_allele") or col_lower.get("effect_allele")
    a2_col = col_lower.get("hm_other_allele") or col_lower.get("other_allele")
    beta_col = col_lower.get("hm_beta") or col_lower.get("beta")
    or_col = col_lower.get("hm_odds_ratio") or col_lower.get("odds_ratio") or col_lower.get("or")
    se_col = col_lower.get("standard_error") or col_lower.get("se")
    p_col = col_lower.get("p_value") or col_lower.get("p")

    return {
        "snp": snp_col, "a1": a1_col, "a2": a2_col,
        "beta": beta_col, "or": or_col, "se": se_col, "p": p_col,
    }


def _process_chunk(chunk: pd.DataFrame, cols: dict, sample_size: int) -> pd.DataFrame:
    """Convert a chunk of raw GWAS data to LDSC format."""
    df_out = pd.DataFrame()
    df_out["SNP"] = chunk[cols["snp"]]
    df_out["A1"] = chunk[cols["a1"]].str.upper()
    df_out["A2"] = chunk[cols["a2"]].str.upper()

    if cols["beta"] and cols["se"]:
        beta = pd.to_numeric(chunk[cols["beta"]], errors="coerce")
        se = pd.to_numeric(chunk[cols["se"]], errors="coerce")
        df_out["Z"] = beta / se
    elif cols["or"] and cols["se"]:
        odds = pd.to_numeric(chunk[cols["or"]], errors="coerce")
        se = pd.to_numeric(chunk[cols["se"]], errors="coerce")
        df_out["Z"] = np.log(odds) / se
    elif cols["p"] and cols["beta"]:
        from scipy.stats import norm
        p = pd.to_numeric(chunk[cols["p"]], errors="coerce")
        beta = pd.to_numeric(chunk[cols["beta"]], errors="coerce")
        z_abs = norm.ppf(1 - p / 2)
        df_out["Z"] = z_abs * np.sign(beta)
    else:
        return pd.DataFrame()

    df_out["N"] = sample_size

    # Clean
    df_out = df_out.dropna(subset=["SNP", "Z"])
    df_out = df_out[df_out["SNP"].str.startswith("rs")]
    df_out = df_out[np.isfinite(df_out["Z"])]
    return df_out


# Threshold for chunked reading: files larger than this (bytes) use chunked processing
_LARGE_FILE_THRESHOLD = 500 * 1024 * 1024  # 500 MB
_CHUNK_ROWS = 500_000


def convert_to_ldsc_format(
    raw_path: Path, study: GWASStudy, output_dir: Path
) -> Path | None:
    """Convert harmonized GWAS summary stats to LDSC-compatible format.

    LDSC format columns: SNP, A1, A2, Z, N
    Where Z = BETA / SE (or log(OR) / SE for odds-ratio studies).
    Uses chunked reading for files larger than 500 MB to avoid OOM.
    """
    ldsc_path = output_dir / raw_path.name.replace(".tsv.gz", ".ldsc.tsv.gz")
    if ldsc_path.exists():
        logger.info("  LDSC file exists: %s", ldsc_path.name)
        return ldsc_path

    logger.info("  Converting to LDSC format: %s", raw_path.name)
    file_size = raw_path.stat().st_size
    use_chunks = file_size > _LARGE_FILE_THRESHOLD

    if use_chunks:
        logger.info("  Large file (%.0f MB), using chunked reading", file_size / 1e6)

    # Read header to resolve columns
    try:
        header_df = pd.read_csv(raw_path, sep="\t", nrows=0)
    except Exception as e:
        logger.error("  Failed to read header of %s: %s", raw_path.name, e)
        return None

    cols = _resolve_columns(list(header_df.columns))
    if cols["snp"] is None:
        logger.error("  No SNP column found in %s. Columns: %s", raw_path.name, list(header_df.columns))
        return None
    if not cols["a1"] or not cols["a2"]:
        logger.error("  Missing allele columns in %s", raw_path.name)
        return None
    if not ((cols["beta"] and cols["se"]) or (cols["or"] and cols["se"]) or (cols["p"] and cols["beta"])):
        logger.error("  Cannot compute Z score for %s (need beta+SE, OR+SE, or p+beta)", raw_path.name)
        return None

    if use_chunks:
        # Chunked processing: write directly to gzip output
        # Only deduplicate within each chunk (not across chunks) to avoid
        # O(n*m) memory/time scaling from a cross-chunk seen_snps set.
        # GWAS Catalog harmonized files rarely have cross-chunk duplicates,
        # and LDSC handles them gracefully.
        total_written = 0
        total_read = 0
        tmp_path = ldsc_path.with_suffix(ldsc_path.suffix + ".tmp")
        with gzip.open(tmp_path, "wt") as out_f:
            out_f.write("SNP\tA1\tA2\tZ\tN\n")
            for chunk in pd.read_csv(raw_path, sep="\t", dtype=str, chunksize=_CHUNK_ROWS):
                total_read += len(chunk)
                processed = _process_chunk(chunk, cols, study.sample_size)
                if processed.empty:
                    continue
                processed = processed.drop_duplicates(subset=["SNP"], keep="first")
                processed.to_csv(out_f, sep="\t", index=False, header=False)
                total_written += len(processed)
                if total_read % 2_000_000 == 0:
                    logger.info("  ... processed %d rows, %d SNPs written", total_read, total_written)
        tmp_path.rename(ldsc_path)
        logger.info("  LDSC output: %d SNPs (from %d rows)", total_written, total_read)
    else:
        # Standard in-memory processing for smaller files
        try:
            df = pd.read_csv(raw_path, sep="\t", dtype=str, low_memory=False)
        except Exception as e:
            logger.error("  Failed to read %s: %s", raw_path.name, e)
            return None

        df_out = _process_chunk(df, cols, study.sample_size)
        if df_out.empty:
            logger.error("  No valid rows after conversion for %s", raw_path.name)
            return None

        df_out = df_out.drop_duplicates(subset=["SNP"], keep="first")
        logger.info("  LDSC output: %d SNPs (from %d rows)", len(df_out), len(df))
        df_out.to_csv(ldsc_path, sep="\t", index=False, compression="gzip")

    return ldsc_path


def download_and_convert(
    study: GWASStudy, data_dir: Path, force: bool = False
) -> dict:
    """Download a single study and convert to LDSC format."""
    raw_dir = data_dir / "raw"
    ldsc_dir = data_dir / "ldsc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ldsc_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / study.filename
    result = {
        "disease": study.disease,
        "abbreviation": study.abbreviation,
        "gwas_catalog_id": study.gwas_catalog_id,
        "pmid": study.pmid,
        "first_author": study.first_author,
        "year": study.year,
        "sample_size": study.sample_size,
        "n_cases": study.n_cases,
        "n_controls": study.n_controls,
        "ancestry": study.ancestry,
        "source": study.source,
        "raw_file": str(raw_path.relative_to(data_dir)),
        "ldsc_file": None,
        "n_snps": None,
        "status": "pending",
        "notes": study.notes,
    }

    # Download
    ok = download_file(study.download_url, raw_path, force=force)
    if not ok:
        result["status"] = "download_failed"
        return result

    # Convert
    ldsc_path = convert_to_ldsc_format(raw_path, study, ldsc_dir)
    if ldsc_path is None:
        result["status"] = "conversion_failed"
        return result

    # Count SNPs
    try:
        n_snps = sum(1 for _ in gzip.open(ldsc_path, "rt")) - 1  # minus header
        result["n_snps"] = n_snps
    except Exception:
        pass

    result["ldsc_file"] = str(ldsc_path.relative_to(data_dir))
    result["status"] = "complete"
    return result


def write_manifest(results: list[dict], manifest_path: Path) -> None:
    """Write manifest file documenting all downloaded studies."""
    manifest = {
        "description": "Autoimmune/inflammatory disease GWAS summary statistics for LDSC cross-trait genetic correlation with Tourette syndrome",
        "project": "ts-neuroimmune-subtyping",
        "phase": "1 — Genetic Correlation Screening",
        "format": "LDSC-compatible (SNP, A1, A2, Z, N)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_studies": len(results),
        "n_complete": sum(1 for r in results if r["status"] == "complete"),
        "studies": results,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written: %s", manifest_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download autoimmune GWAS summary statistics for LDSC analysis"
    )
    parser.add_argument(
        "--disease",
        type=str,
        default="",
        help="Download only this disease (by abbreviation, e.g. 'ra', 'ms')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_studies",
        help="List available studies and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    studies = AUTOIMMUNE_STUDIES
    if args.list_studies:
        print(f"{'Abbr':<6} {'Disease':<35} {'GWAS ID':<16} {'N':>8} {'Cases':>8} {'Source'}")
        print("-" * 90)
        for s in studies:
            print(
                f"{s.abbreviation:<6} {s.disease:<35} {s.gwas_catalog_id:<16} "
                f"{s.sample_size:>8} {s.n_cases:>8} {s.source}"
            )
        return

    if args.disease:
        studies = [s for s in studies if s.abbreviation == args.disease.lower()]
        if not studies:
            valid = ", ".join(s.abbreviation for s in AUTOIMMUNE_STUDIES)
            parser.error(f"Unknown disease '{args.disease}'. Valid: {valid}")

    data_dir = args.dest
    logger.info("Downloading %d autoimmune GWAS studies to %s", len(studies), data_dir)

    results = []
    for i, study in enumerate(studies, 1):
        logger.info("[%d/%d] %s (%s)", i, len(studies), study.disease, study.gwas_catalog_id)
        result = download_and_convert(study, data_dir, force=args.force)
        results.append(result)

        status_icon = "OK" if result["status"] == "complete" else "FAIL"
        snps = result.get("n_snps") or "?"
        logger.info("  %s — %s SNPs", status_icon, snps)

    # Write manifest
    write_manifest(results, data_dir / "manifest.json")

    # Summary
    complete = sum(1 for r in results if r["status"] == "complete")
    failed = len(results) - complete
    logger.info("Done: %d/%d complete, %d failed", complete, len(results), failed)
    if failed:
        for r in results:
            if r["status"] != "complete":
                logger.warning("  FAILED: %s — %s", r["disease"], r["status"])


if __name__ == "__main__":
    main()
