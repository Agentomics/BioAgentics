"""Download LD reference panel and baseline-LD annotations for LDSC pipelines.

Downloads from the Broad Institute (Alkes Group):
1. EUR LD scores (eur_w_ld_chr)
2. 1000G Phase 3 plink files
3. Baseline-LD annotations (Phase 1)
4. HM3 regression weights (no HLA)
5. HapMap3 SNP list

All data is hg19/GRCh37 build. Saved to:
  data/tourettes/ts-comorbidity-genetic-architecture/reference/
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://data.broadinstitute.org/alkesgroup/LDSCORE"
DATA_DIR = Path("data/tourettes/ts-comorbidity-genetic-architecture/reference")

DOWNLOADS = [
    {
        "name": "EUR LD scores",
        "url": f"{BASE_URL}/eur_w_ld_chr.tar.bz2",
        "filename": "eur_w_ld_chr.tar.bz2",
        "extract_dir": "eur_w_ld_chr",
        "description": "Pre-computed European LD scores for h2/rg estimation",
    },
    {
        "name": "1000G Phase 3 plink files",
        "url": f"{BASE_URL}/1000G_Phase3_plinkfiles.tgz",
        "filename": "1000G_Phase3_plinkfiles.tgz",
        "extract_dir": "1000G_Phase3_plinkfiles",
        "description": "Reference genotypes for LD score computation",
    },
    {
        "name": "Baseline-LD annotations",
        "url": f"{BASE_URL}/1000G_Phase1_baseline_ldscores.tgz",
        "filename": "1000G_Phase1_baseline_ldscores.tgz",
        "extract_dir": "baseline",
        "description": "Functional annotation categories for S-LDSC partitioned heritability",
    },
    {
        "name": "HM3 regression weights (no HLA)",
        "url": f"{BASE_URL}/weights_hm3_no_hla.tgz",
        "filename": "weights_hm3_no_hla.tgz",
        "extract_dir": "weights_hm3_no_hla",
        "description": "LD-aware regression weights excluding HLA region",
    },
    {
        "name": "HapMap3 SNPs",
        "url": f"{BASE_URL}/hapmap3_snps.tgz",
        "filename": "hapmap3_snps.tgz",
        "extract_dir": "hapmap3_snps",
        "description": "HapMap3 SNP list used for filtering GWAS summary stats",
    },
]


def _download_file(url: str, dest: Path) -> None:
    """Download a file using curl with progress."""
    logger.info("Downloading %s -> %s", url, dest)
    subprocess.run(
        ["curl", "-fSL", "--retry", "3", "--retry-delay", "5", "-o", str(dest), url],
        check=True,
    )


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract tar.bz2 or tgz archive."""
    logger.info("Extracting %s -> %s", archive.name, dest_dir)
    if archive.name.endswith(".tar.bz2"):
        mode = "r:bz2"
    elif archive.name.endswith(".tgz") or archive.name.endswith(".tar.gz"):
        mode = "r:gz"
    else:
        raise ValueError(f"Unknown archive format: {archive.name}")
    with tarfile.open(archive, mode) as tf:
        tf.extractall(path=dest_dir, filter="data")


def _md5sum(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(data_dir: Path | None = None, skip_existing: bool = True) -> dict[str, Path]:
    """Download all LDSC reference datasets.

    Args:
        data_dir: Target directory. Defaults to DATA_DIR.
        skip_existing: Skip download if extracted directory already exists.

    Returns:
        Dict mapping dataset name to extracted directory path.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for item in DOWNLOADS:
        extract_path = data_dir / item["extract_dir"]
        archive_path = data_dir / item["filename"]

        if skip_existing and extract_path.exists():
            logger.info("Skipping %s (already exists at %s)", item["name"], extract_path)
            results[item["name"]] = extract_path
            continue

        _download_file(item["url"], archive_path)

        md5 = _md5sum(archive_path)
        logger.info("%s MD5: %s", item["filename"], md5)

        _extract_archive(archive_path, data_dir)

        # Remove archive after successful extraction to save disk space
        archive_path.unlink()
        logger.info("Removed archive %s after extraction", archive_path.name)

        results[item["name"]] = extract_path

    # Write manifest
    _write_manifest(data_dir, results)
    return results


def _write_manifest(data_dir: Path, results: dict[str, Path]) -> None:
    """Write a README manifest documenting the downloaded data."""
    manifest_path = data_dir / "README.txt"
    lines = [
        "LDSC Reference Data Manifest",
        "=" * 40,
        "",
        "Genome Build: hg19 / GRCh37",
        "Source: Broad Institute (Alkes Group LDSCORE)",
        f"Base URL: {BASE_URL}",
        "",
        "Datasets:",
        "",
    ]
    for item in DOWNLOADS:
        extract_path = results.get(item["name"], data_dir / item["extract_dir"])
        lines.append(f"  {item['name']}")
        lines.append(f"    File: {item['filename']}")
        lines.append(f"    Dir:  {extract_path.name}/")
        lines.append(f"    Desc: {item['description']}")
        lines.append("")

    lines.extend([
        "Build Compatibility Notes:",
        "  - All data uses hg19/GRCh37 coordinates",
        "  - GWAS summary statistics must be in hg19 or lifted over before use",
        "  - EUR LD scores are European-ancestry specific",
        "  - Baseline-LD annotations are from 1000G Phase 1",
        "  - HapMap3 SNPs are the standard filter set for LDSC rg/h2",
        "",
    ])
    manifest_path.write_text("\n".join(lines))
    logger.info("Wrote manifest to %s", manifest_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Starting LDSC reference data download")
    results = download_all()
    logger.info("Download complete. %d datasets ready.", len(results))
    for name, path in results.items():
        logger.info("  %s: %s", name, path)


if __name__ == "__main__":
    main()
