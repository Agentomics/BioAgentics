"""Download LD reference panel and baseline-LD annotations for LDSC pipelines.

Downloads from Zenodo (mirrors of Broad/Alkes Group data):
1. EUR LD scores (1000G Phase 3)
2. 1000G Phase 3 plink files (for LD clumping in PRS)
3. Baseline-LD v2.2 annotations (for S-LDSC partitioned heritability)
4. HM3 regression weights (no MHC)
5. HapMap3 SNP list
6. 1000G Phase 3 allele frequencies

NOTE: Original Broad Institute URLs (data.broadinstitute.org/alkesgroup/LDSCORE/)
are now dead (404). Data moved to requester-pays GCS bucket. Zenodo mirrors
provide free access to the core reference files.

Primary mirror: https://zenodo.org/records/7768714
Extended mirror: https://zenodo.org/records/10515792 (adds v2.3, hm3 list)

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

ZENODO_BASE = "https://zenodo.org/records/7768714/files"
ZENODO_EXT = "https://zenodo.org/records/10515792/files"
DATA_DIR = Path("data/tourettes/ts-comorbidity-genetic-architecture/reference")

DOWNLOADS = [
    {
        "name": "EUR LD scores",
        "url": f"{ZENODO_BASE}/1000G_Phase3_ldscores.tgz",
        "filename": "1000G_Phase3_ldscores.tgz",
        "extract_dir": "LDscore",
        "description": "Pre-computed European LD scores for h2/rg estimation",
    },
    {
        "name": "1000G Phase 3 plink files",
        "url": f"{ZENODO_BASE}/1000G_Phase3_plinkfiles.tgz",
        "filename": "1000G_Phase3_plinkfiles.tgz",
        "extract_dir": "1000G_Phase3_plinkfiles",
        "description": "Reference genotypes for LD clumping in PRS",
    },
    {
        "name": "Baseline-LD v2.2 annotations",
        "url": f"{ZENODO_BASE}/1000G_Phase3_baselineLD_v2.2_ldscores.tgz",
        "filename": "1000G_Phase3_baselineLD_v2.2_ldscores.tgz",
        "extract_dir": "baselineLD_v2.2",
        "description": "Functional annotation categories for S-LDSC partitioned heritability (v2.2)",
    },
    {
        "name": "HM3 regression weights (no MHC)",
        "url": f"{ZENODO_BASE}/1000G_Phase3_weights_hm3_no_MHC.tgz",
        "filename": "1000G_Phase3_weights_hm3_no_MHC.tgz",
        "extract_dir": "1000G_Phase3_weights_hm3_no_MHC",
        "description": "LD-aware regression weights excluding MHC region",
    },
    {
        "name": "HapMap3 SNP list",
        "url": f"{ZENODO_EXT}/hm3_no_MHC.list.txt",
        "filename": "hm3_no_MHC.list.txt",
        "extract_dir": None,
        "description": "HapMap3 SNP list for filtering GWAS summary stats",
    },
    {
        "name": "1000G Phase 3 allele frequencies",
        "url": f"{ZENODO_BASE}/1000G_Phase3_frq.tgz",
        "filename": "1000G_Phase3_frq.tgz",
        "extract_dir": "1000G_Phase3_frq",
        "description": "Reference allele frequencies for QC and strand alignment",
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
        file_path = data_dir / item["filename"]
        extract_dir = item["extract_dir"]

        if extract_dir is not None:
            extract_path = data_dir / extract_dir
            if skip_existing and extract_path.exists():
                logger.info("Skipping %s (already exists at %s)", item["name"], extract_path)
                results[item["name"]] = extract_path
                continue
        else:
            # Plain file (no extraction needed)
            if skip_existing and file_path.exists():
                logger.info("Skipping %s (already exists at %s)", item["name"], file_path)
                results[item["name"]] = file_path
                continue

        _download_file(item["url"], file_path)

        md5 = _md5sum(file_path)
        logger.info("%s MD5: %s", item["filename"], md5)

        if extract_dir is not None:
            _extract_archive(file_path, data_dir)
            # Remove archive after successful extraction to save disk space
            file_path.unlink()
            logger.info("Removed archive %s after extraction", file_path.name)
            results[item["name"]] = data_dir / extract_dir
        else:
            results[item["name"]] = file_path

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
        "Source: Broad Institute (Alkes Group) via Zenodo mirrors",
        f"Zenodo: {ZENODO_BASE.rsplit('/files', 1)[0]}",
        "",
        "Datasets:",
        "",
    ]
    for item in DOWNLOADS:
        fallback = data_dir / item["extract_dir"] if item["extract_dir"] else data_dir / item["filename"]
        extract_path = results.get(item["name"], fallback)
        lines.append(f"  {item['name']}")
        lines.append(f"    File: {item['filename']}")
        if item["extract_dir"]:
            lines.append(f"    Dir:  {extract_path.name}/")
        lines.append(f"    Desc: {item['description']}")
        lines.append("")

    lines.extend([
        "Build Compatibility Notes:",
        "  - All data uses hg19/GRCh37 coordinates",
        "  - GWAS summary statistics must be in hg19 or lifted over before use",
        "  - EUR LD scores are European-ancestry specific",
        "  - Baseline-LD v2.2 annotations for S-LDSC partitioned heritability",
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
