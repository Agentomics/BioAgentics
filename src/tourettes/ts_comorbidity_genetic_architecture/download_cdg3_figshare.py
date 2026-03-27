"""Download Grotzinger CDG3 five-factor GWAS summary statistics from figshare.

Grotzinger et al. 2026 Nature (PMID:41372416) — GenomicSEM across 14 psychiatric
disorders identified 5 factors: Compulsive, Schizophrenia-Bipolar, Neurodevelopmental,
Internalizing, Substance Use.  Factor GWAS summary stats deposited on figshare:
DOI:10.6084/m9.figshare.30359017 (listed as CDG2025 on PGC download page).

Saves to: data/tourettes/ts-comorbidity-genetic-architecture/grotzinger_cdg3_factor_gwas/
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

FIGSHARE_ARTICLE_ID = "30359017"
DATA_DIR = Path("data/tourettes/ts-comorbidity-genetic-architecture/grotzinger_cdg3_factor_gwas")

# Primary factor GWAS files from figshare
DOWNLOADS = [
    {
        "name": "F1_CompulsiveDisorders",
        "filename": "F1_CompulsiveDisorders_2025.tsv.gz",
        "url": "https://ndownloader.figshare.com/files/58731913",
        "factor": "compulsive",
        "description": "Factor 1: Compulsive Disorders GWAS summary statistics",
    },
    {
        "name": "F2_SchizophreniaBipolar",
        "filename": "F2_SchizophreniaBipolar_2025.tsv.gz",
        "url": "https://ndownloader.figshare.com/files/58731907",
        "factor": "sb",
        "description": "Factor 2: Schizophrenia-Bipolar GWAS summary statistics",
    },
    {
        "name": "F3_Neurodevelopmental",
        "filename": "F3_Neurodevelopmental_2025.tsv.gz",
        "url": "https://ndownloader.figshare.com/files/58731910",
        "factor": "neurodevelopmental",
        "description": "Factor 3: Neurodevelopmental GWAS summary statistics",
    },
    {
        "name": "F4_Internalizing",
        "filename": "F4_Internalizing_2025.tsv.gz",
        "url": "https://ndownloader.figshare.com/files/58731904",
        "factor": "internalizing",
        "description": "Factor 4: Internalizing GWAS summary statistics",
    },
    {
        "name": "F5_SubstanceUse",
        "filename": "F5_SubstanceUse_2025.tsv.gz",
        "url": "https://ndownloader.figshare.com/files/58731901",
        "factor": "substance_use",
        "description": "Factor 5: Substance Use GWAS summary statistics",
    },
    {
        "name": "PFactor",
        "filename": "PFactor_2025.tsv.gz",
        "url": "https://ndownloader.figshare.com/files/58731898",
        "factor": "pfactor",
        "description": "General psychopathology p-factor GWAS summary statistics",
    },
    {
        "name": "CDG3_README",
        "filename": "CDG3_README_2025.pdf",
        "url": "https://ndownloader.figshare.com/files/58731889",
        "factor": None,
        "description": "README documentation for CDG3 dataset",
    },
    {
        "name": "CDG3_Hits",
        "filename": "CDG3_Hits_2025.txt",
        "url": "https://ndownloader.figshare.com/files/58731892",
        "factor": None,
        "description": "Genome-wide significant hits across factors",
    },
]

EXPECTED_FACTORS = ["compulsive", "sb", "neurodevelopmental", "internalizing", "substance_use"]


def _download_file(url: str, dest: Path) -> None:
    """Download a file using curl with resume support."""
    logger.info("Downloading %s -> %s", url, dest)
    subprocess.run(
        [
            "curl", "-fSL",
            "--retry", "3",
            "--retry-delay", "5",
            "-C", "-",
            "-o", str(dest),
            url,
        ],
        check=True,
    )


def _md5sum(path: Path) -> str:
    """Compute MD5 checksum of a file (streaming, memory-safe)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_gwas_header(path: Path) -> dict:
    """Peek at the first lines of a gzipped TSV to verify GWAS format.

    Returns a dict with column names and row count estimate.
    """
    if not path.name.endswith(".gz"):
        return {"columns": [], "n_preview_rows": 0, "valid": True}

    columns = []
    n_rows = 0
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i == 0:
                columns = line.strip().split("\t")
            elif i <= 5:
                n_rows += 1
            else:
                break

    return {
        "columns": columns,
        "n_preview_rows": n_rows,
        "valid": len(columns) > 0 and n_rows > 0,
    }


def download_all(
    data_dir: Path | None = None,
    skip_existing: bool = True,
) -> dict:
    """Download CDG3 factor GWAS summary statistics from figshare.

    Args:
        data_dir: Target directory. Defaults to DATA_DIR.
        skip_existing: Skip files that already exist on disk.

    Returns:
        Dict with download results and verification info.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    results = {"files": {}, "verification": {}, "factors_found": []}
    downloads = list(DOWNLOADS)

    for item in downloads:
        dest = data_dir / item["filename"]

        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info("Skipping %s (already exists: %s)", item["name"], dest)
        else:
            _download_file(item["url"], dest)

        md5 = _md5sum(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("%s: %.1f MB, MD5=%s", item["filename"], size_mb, md5)

        results["files"][item["name"]] = {
            "path": str(dest),
            "filename": item["filename"],
            "size_mb": round(size_mb, 1),
            "md5": md5,
            "factor": item["factor"],
        }

        # Verify GWAS files have proper headers
        if dest.name.endswith(".tsv.gz"):
            verify = _verify_gwas_header(dest)
            results["verification"][item["name"]] = verify
            if verify["valid"]:
                logger.info(
                    "  Verified %s: %d columns, %d preview rows",
                    item["name"],
                    len(verify["columns"]),
                    verify["n_preview_rows"],
                )
                if item["factor"]:
                    results["factors_found"].append(item["factor"])
            else:
                logger.warning("  FAILED verification for %s", item["name"])

    # Check all expected factors are present
    missing_factors = set(EXPECTED_FACTORS) - set(results["factors_found"])
    if missing_factors:
        logger.warning("Missing expected factors: %s", missing_factors)
    else:
        logger.info("All %d expected factors present", len(EXPECTED_FACTORS))

    results["all_factors_present"] = len(missing_factors) == 0
    results["missing_factors"] = list(missing_factors)

    # Write manifest
    _write_manifest(data_dir, results)

    return results


def _write_manifest(data_dir: Path, results: dict) -> None:
    """Write a JSON manifest documenting the downloaded data."""
    manifest_path = data_dir / "manifest.json"
    manifest = {
        "source": "Grotzinger et al. 2026, Nature 649:406-415",
        "doi_paper": "10.1038/s41586-025-09820-3",
        "doi_figshare": "10.6084/m9.figshare.30359017",
        "pmid": "41372416",
        "figshare_article_id": FIGSHARE_ARTICLE_ID,
        "description": (
            "CDG3: GenomicSEM 5-factor model across 14 psychiatric disorders. "
            "Factor GWAS summary statistics from the PGC cross-disorder working group."
        ),
        "factors": {
            "F1": "Compulsive Disorders",
            "F2": "Schizophrenia-Bipolar",
            "F3": "Neurodevelopmental",
            "F4": "Internalizing",
            "F5": "Substance Use",
            "PFactor": "General psychopathology p-factor",
        },
        "files": results["files"],
        "verification": results["verification"],
        "all_factors_present": results["all_factors_present"],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Starting CDG3 factor GWAS download from figshare")
    results = download_all()
    logger.info("Download complete.")
    logger.info("Factors found: %s", results["factors_found"])
    if results["missing_factors"]:
        logger.warning("Missing factors: %s", results["missing_factors"])
    else:
        logger.info("All expected factors present.")

    for name, info in results["files"].items():
        logger.info("  %s: %.1f MB", name, info["size_mb"])

    if results["verification"]:
        logger.info("Verification results:")
        for name, v in results["verification"].items():
            cols = ", ".join(v["columns"][:5])
            logger.info("  %s: columns=[%s...] valid=%s", name, cols, v["valid"])


if __name__ == "__main__":
    main()
