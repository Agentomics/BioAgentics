"""Download immune biomarker GWAS instruments for Mendelian randomization.

Downloads GWAS summary statistics for immune biomarker exposures needed for
Phase 3 two-sample MR: CRP, IL-6, TNF-alpha, lymphocyte/monocyte/neutrophil
counts, IgG levels, NK cell count, IL-17, CD4+ T cell count, CCL2/MCP-1.

Sources include UK Biobank blood cell traits, cytokine/protein pQTL studies,
and large immune biomarker GWAS.

Output: data/tourettes/ts-neuroimmune-subtyping/mr_instruments/

Usage:
    uv run python -m bioagentics.data.tourettes.neuroimmune.download_mr_instruments
    uv run python -m bioagentics.data.tourettes.neuroimmune.download_mr_instruments --list
    uv run python -m bioagentics.data.tourettes.neuroimmune.download_mr_instruments --biomarker crp
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = (
    REPO_ROOT / "data" / "tourettes" / "ts-neuroimmune-subtyping" / "mr_instruments"
)

TIMEOUT = 300
CHUNK_SIZE = 65536


@dataclass
class MRExposure:
    """Metadata for an MR exposure GWAS."""

    biomarker: str
    abbreviation: str
    gwas_id: str
    pmid: str
    first_author: str
    year: int
    sample_size: int
    ancestry: str
    download_url: str
    filename: str
    source: str
    unit: str
    notes: str = ""


# Curated immune biomarker GWAS for MR instruments.
# Prioritizes large European-ancestry studies with publicly available summary stats.
MR_EXPOSURES: list[MRExposure] = [
    MRExposure(
        biomarker="C-reactive protein",
        abbreviation="crp",
        gwas_id="GCST90029070",
        pmid="34226706",
        first_author="Said",
        year=2022,
        sample_size=575531,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90029001-GCST90030000/GCST90029070/harmonised/35459240-GCST90029070-EFO_0004458.h.tsv.gz",
        filename="crp_said_2022.tsv.gz",
        source="gwas_catalog",
        unit="mg/L",
        notes="Largest CRP GWAS meta-analysis",
    ),
    MRExposure(
        biomarker="IL-6",
        abbreviation="il6",
        gwas_id="GCST90274758",
        pmid="36635385",
        first_author="Zhao",
        year=2023,
        sample_size=67428,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90274001-GCST90275000/GCST90274758/harmonised/GCST90274758.h.tsv.gz",
        filename="il6_zhao_2023.tsv.gz",
        source="gwas_catalog",
        unit="pg/mL",
        notes="Circulating IL-6 protein levels",
    ),
    MRExposure(
        biomarker="TNF-alpha",
        abbreviation="tnfa",
        gwas_id="GCST90274806",
        pmid="36635385",
        first_author="Zhao",
        year=2023,
        sample_size=67428,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90274001-GCST90275000/GCST90274806/harmonised/GCST90274806.h.tsv.gz",
        filename="tnfa_zhao_2023.tsv.gz",
        source="gwas_catalog",
        unit="pg/mL",
        notes="Circulating TNF-alpha protein levels",
    ),
    MRExposure(
        biomarker="Lymphocyte count",
        abbreviation="lymph",
        gwas_id="GCST90002311",
        pmid="32888494",
        first_author="Chen",
        year=2020,
        sample_size=408112,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90002001-GCST90003000/GCST90002311/harmonised/32888493-GCST90002311-EFO_0004509.h.tsv.gz",
        filename="lymph_chen_2020.tsv.gz",
        source="gwas_catalog",
        unit="10^9/L",
        notes="UKB+INTERVAL blood cell traits",
    ),
    MRExposure(
        biomarker="Monocyte count",
        abbreviation="mono",
        gwas_id="GCST90002314",
        pmid="32888494",
        first_author="Chen",
        year=2020,
        sample_size=408112,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90002001-GCST90003000/GCST90002314/harmonised/32888493-GCST90002314-EFO_0004509.h.tsv.gz",
        filename="mono_chen_2020.tsv.gz",
        source="gwas_catalog",
        unit="10^9/L",
        notes="UKB+INTERVAL blood cell traits",
    ),
    MRExposure(
        biomarker="Neutrophil count",
        abbreviation="neut",
        gwas_id="GCST90002315",
        pmid="32888494",
        first_author="Chen",
        year=2020,
        sample_size=408112,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90002001-GCST90003000/GCST90002315/harmonised/32888493-GCST90002315-EFO_0004587.h.tsv.gz",
        filename="neut_chen_2020.tsv.gz",
        source="gwas_catalog",
        unit="10^9/L",
        notes="UKB+INTERVAL blood cell traits",
    ),
    MRExposure(
        biomarker="NK cell count",
        abbreviation="nk",
        gwas_id="GCST90001475",
        pmid="32929287",
        first_author="Orru",
        year=2020,
        sample_size=3757,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90001001-GCST90002000/GCST90001475/harmonised/32929287-GCST90001475-EFO_0007937.h.tsv.gz",
        filename="nk_orru_2020.tsv.gz",
        source="gwas_catalog",
        unit="cells/uL",
        notes="Sardinian immune cell GWAS (flow cytometry)",
    ),
    MRExposure(
        biomarker="CD4+ T cell count",
        abbreviation="cd4",
        gwas_id="GCST90001399",
        pmid="32929287",
        first_author="Orru",
        year=2020,
        sample_size=3757,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90001001-GCST90002000/GCST90001399/harmonised/32929287-GCST90001399-EFO_0007937.h.tsv.gz",
        filename="cd4_orru_2020.tsv.gz",
        source="gwas_catalog",
        unit="cells/uL",
        notes="Sardinian immune cell GWAS (flow cytometry)",
    ),
    MRExposure(
        biomarker="IgG levels",
        abbreviation="igg",
        gwas_id="GCST90018994",
        pmid="33440007",
        first_author="Sakaue",
        year=2021,
        sample_size=458620,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90018001-GCST90019000/GCST90018994/harmonised/GCST90018994.h.tsv.gz",
        filename="igg_sakaue_2021.tsv.gz",
        source="gwas_catalog",
        unit="g/L",
        notes="FinnGen+UKB meta-analysis",
    ),
    MRExposure(
        biomarker="CCL2/MCP-1",
        abbreviation="ccl2",
        gwas_id="GCST90274650",
        pmid="36635385",
        first_author="Zhao",
        year=2023,
        sample_size=67428,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90274001-GCST90275000/GCST90274650/harmonised/GCST90274650.h.tsv.gz",
        filename="ccl2_zhao_2023.tsv.gz",
        source="gwas_catalog",
        unit="pg/mL",
        notes="CSF MCP-1/CCL2 elevated in TS — links peripheral to microglial activation",
    ),
    MRExposure(
        biomarker="IL-17",
        abbreviation="il17",
        gwas_id="GCST90274760",
        pmid="36635385",
        first_author="Zhao",
        year=2023,
        sample_size=67428,
        ancestry="European",
        download_url="https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90274001-GCST90275000/GCST90274760/harmonised/GCST90274760.h.tsv.gz",
        filename="il17_zhao_2023.tsv.gz",
        source="gwas_catalog",
        unit="pg/mL",
        notes="IL-17 elevated in TS patients — Th17 pathway biomarker",
    ),
]


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


def extract_instruments(
    raw_path: Path, exposure: MRExposure, output_dir: Path,
    p_threshold: float = 5e-8, r2_threshold: float = 0.001,
) -> dict:
    """Extract MR instruments from GWAS summary statistics.

    Selects genome-wide significant SNPs, computes F-statistics,
    and prepares instrument data for TwoSampleMR harmonization.
    """
    inst_path = output_dir / raw_path.name.replace(".tsv.gz", ".instruments.tsv")
    result = {
        "biomarker": exposure.biomarker,
        "abbreviation": exposure.abbreviation,
        "gwas_id": exposure.gwas_id,
        "raw_file": raw_path.name,
        "instrument_file": None,
        "n_instruments": 0,
        "median_f_stat": None,
        "status": "pending",
    }

    logger.info("  Extracting instruments for %s...", exposure.biomarker)

    # Use chunked reading for large files (>500MB compressed) to stay within RAM
    file_size_mb = raw_path.stat().st_size / (1024 * 1024)
    use_chunked = file_size_mb > 500

    if use_chunked:
        logger.info("  Large file (%.0f MB) — using chunked reading", file_size_mb)

    try:
        if use_chunked:
            # Read in chunks, filter to significant SNPs in each chunk
            sig_chunks = []
            col_lower = None
            snp_col = a1_col = a2_col = beta_col = or_col = se_col = p_col = eaf_col = None
            total_rows = 0

            for chunk in pd.read_csv(
                raw_path, sep="\t", dtype=str, chunksize=200_000
            ):
                total_rows += len(chunk)

                # Determine column names from first chunk
                if col_lower is None:
                    col_lower = {c.lower(): c for c in chunk.columns}
                    snp_col = col_lower.get("hm_rsid") or col_lower.get("rsid") or col_lower.get("snp")
                    a1_col = col_lower.get("hm_effect_allele") or col_lower.get("effect_allele")
                    a2_col = col_lower.get("hm_other_allele") or col_lower.get("other_allele")
                    beta_col = col_lower.get("hm_beta") or col_lower.get("beta")
                    or_col = col_lower.get("hm_odds_ratio") or col_lower.get("odds_ratio")
                    se_col = col_lower.get("standard_error") or col_lower.get("se")
                    p_col = col_lower.get("p_value") or col_lower.get("p")
                    eaf_col = col_lower.get("hm_effect_allele_frequency") or col_lower.get("effect_allele_frequency")

                    if not all([snp_col, a1_col, a2_col, p_col]):
                        logger.error("  Missing required columns in %s", raw_path.name)
                        result["status"] = "missing_columns"
                        return result

                # Filter this chunk to suggestive threshold (keep superset, refine later)
                p_vals = pd.to_numeric(chunk[p_col], errors="coerce")
                mask = p_vals < 1e-5
                if mask.any():
                    sig_chunks.append(chunk[mask].copy())

            logger.info("  Scanned %d rows in chunks", total_rows)
            if not sig_chunks:
                logger.warning("  No significant SNPs found")
                result["status"] = "no_instruments"
                return result
            df = pd.concat(sig_chunks, ignore_index=True)
            del sig_chunks
        else:
            df = pd.read_csv(raw_path, sep="\t", dtype=str, low_memory=False)
    except Exception as e:
        logger.error("  Failed to read %s: %s", raw_path.name, e)
        result["status"] = "read_failed"
        return result

    if not use_chunked:
        # Find columns (already done for chunked path)
        col_lower = {c.lower(): c for c in df.columns}

        snp_col = col_lower.get("hm_rsid") or col_lower.get("rsid") or col_lower.get("snp")
        a1_col = col_lower.get("hm_effect_allele") or col_lower.get("effect_allele")
        a2_col = col_lower.get("hm_other_allele") or col_lower.get("other_allele")
        beta_col = col_lower.get("hm_beta") or col_lower.get("beta")
        or_col = col_lower.get("hm_odds_ratio") or col_lower.get("odds_ratio")
        se_col = col_lower.get("standard_error") or col_lower.get("se")
        p_col = col_lower.get("p_value") or col_lower.get("p")
        eaf_col = col_lower.get("hm_effect_allele_frequency") or col_lower.get("effect_allele_frequency")

        if not all([snp_col, a1_col, a2_col, p_col]):
            logger.error("  Missing required columns in %s", raw_path.name)
            result["status"] = "missing_columns"
            return result

    # Convert numeric columns
    df["_p"] = pd.to_numeric(df[p_col], errors="coerce")
    if beta_col:
        df["_beta"] = pd.to_numeric(df[beta_col], errors="coerce")
    elif or_col:
        df["_beta"] = np.log(pd.to_numeric(df[or_col], errors="coerce"))
    else:
        result["status"] = "no_effect_size"
        return result

    if se_col:
        df["_se"] = pd.to_numeric(df[se_col], errors="coerce")

    # Filter to genome-wide significant
    sig = df[df["_p"] < p_threshold].copy()
    logger.info("  GW-significant SNPs (p < %.0e): %d", p_threshold, len(sig))

    if len(sig) == 0:
        # Try suggestive threshold
        sig = df[df["_p"] < 1e-5].copy()
        logger.warning(
            "  No GW-significant SNPs. Using suggestive (p < 1e-5): %d", len(sig)
        )
        if len(sig) == 0:
            result["status"] = "no_instruments"
            return result

    # Build instrument table
    instruments = pd.DataFrame()
    instruments["SNP"] = sig[snp_col]
    instruments["effect_allele"] = sig[a1_col].str.upper()
    instruments["other_allele"] = sig[a2_col].str.upper()
    instruments["beta"] = sig["_beta"]
    instruments["pval"] = sig["_p"]
    if se_col:
        instruments["se"] = sig["_se"]
    if eaf_col:
        instruments["eaf"] = pd.to_numeric(sig[eaf_col], errors="coerce")

    instruments["samplesize"] = exposure.sample_size

    # Compute F-statistic: F = beta^2 / se^2
    if "se" in instruments.columns:
        instruments["f_stat"] = instruments["beta"] ** 2 / instruments["se"] ** 2
        # Filter weak instruments
        strong = instruments[instruments["f_stat"] > 10]
        logger.info(
            "  Instruments with F > 10: %d / %d", len(strong), len(instruments)
        )
        instruments = strong if len(strong) > 0 else instruments

    # Drop duplicates, keep strongest signal per SNP
    instruments = instruments.dropna(subset=["SNP"])
    instruments = instruments[instruments["SNP"].str.startswith("rs")]
    instruments = instruments.sort_values("pval").drop_duplicates(subset=["SNP"])

    instruments.to_csv(inst_path, sep="\t", index=False)
    logger.info("  Final instruments: %d SNPs", len(instruments))

    result["instrument_file"] = inst_path.name
    result["n_instruments"] = len(instruments)
    if "f_stat" in instruments.columns:
        result["median_f_stat"] = round(float(instruments["f_stat"].median()), 1)
    result["status"] = "complete"
    return result


def download_and_extract(
    exposure: MRExposure, data_dir: Path, force: bool = False
) -> dict:
    """Download exposure GWAS and extract MR instruments."""
    raw_dir = data_dir / "raw"
    inst_dir = data_dir / "instruments"
    raw_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / exposure.filename
    ok = download_file(exposure.download_url, raw_path, force=force)
    if not ok:
        return {
            "biomarker": exposure.biomarker,
            "abbreviation": exposure.abbreviation,
            "status": "download_failed",
        }

    return extract_instruments(raw_path, exposure, inst_dir)


def write_manifest(results: list[dict], data_dir: Path) -> None:
    """Write manifest documenting MR instrument sources."""
    manifest = {
        "description": "Immune biomarker GWAS instruments for two-sample Mendelian randomization",
        "project": "ts-neuroimmune-subtyping",
        "phase": "3 — Mendelian Randomization",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_exposures": len(results),
        "n_complete": sum(1 for r in results if r["status"] == "complete"),
        "total_instruments": sum(r.get("n_instruments", 0) for r in results),
        "exposures": results,
    }
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written: %s", manifest_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download immune biomarker GWAS for MR instruments"
    )
    parser.add_argument(
        "--biomarker",
        type=str,
        default="",
        help="Download only this biomarker (by abbreviation, e.g. 'crp', 'il6')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_exposures",
        help="List available exposures and exit",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dest", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    exposures = MR_EXPOSURES
    if args.list_exposures:
        print(f"{'Abbr':<8} {'Biomarker':<25} {'GWAS ID':<16} {'N':>8} {'Source'}")
        print("-" * 75)
        for e in exposures:
            print(
                f"{e.abbreviation:<8} {e.biomarker:<25} {e.gwas_id:<16} "
                f"{e.sample_size:>8} {e.source}"
            )
        return

    if args.biomarker:
        exposures = [e for e in exposures if e.abbreviation == args.biomarker.lower()]
        if not exposures:
            valid = ", ".join(e.abbreviation for e in MR_EXPOSURES)
            parser.error(f"Unknown biomarker '{args.biomarker}'. Valid: {valid}")

    data_dir = args.dest
    logger.info("Downloading %d MR exposure GWAS to %s", len(exposures), data_dir)

    results = []
    for i, exposure in enumerate(exposures, 1):
        logger.info("[%d/%d] %s (%s)", i, len(exposures), exposure.biomarker, exposure.gwas_id)
        result = download_and_extract(exposure, data_dir, force=args.force)
        results.append(result)

        status_icon = "OK" if result["status"] == "complete" else "FAIL"
        n_inst = result.get("n_instruments", "?")
        f_stat = result.get("median_f_stat", "?")
        logger.info("  %s — %s instruments (median F = %s)", status_icon, n_inst, f_stat)

    write_manifest(results, data_dir)

    complete = sum(1 for r in results if r["status"] == "complete")
    total_inst = sum(r.get("n_instruments", 0) for r in results)
    logger.info(
        "Done: %d/%d complete, %d total instruments", complete, len(results), total_inst
    )


if __name__ == "__main__":
    main()
