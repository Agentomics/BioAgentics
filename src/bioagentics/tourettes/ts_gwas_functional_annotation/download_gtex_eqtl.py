"""Download GTEx v8 brain eQTL data for TS GWAS functional annotation.

Downloads significant variant-gene pair files for 5 TS-relevant brain regions
from the GTEx Portal (v8 release, Google Cloud Storage). Extracts from the
bulk tar archive and verifies file integrity via MD5 checksums.

Usage:
    uv run python -m bioagentics.tourettes.ts_gwas_functional_annotation.download_gtex_eqtl
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from bioagentics.tourettes.ts_gwas_functional_annotation.config import DATA_DIR

logger = logging.getLogger(__name__)

# Target directory
GTEX_EQTL_DIR = DATA_DIR / "gtex_v8_eqtl"

# GTEx v8 eQTL tar on Google Cloud Storage (open access)
GTEX_V8_EQTL_TAR_URL = (
    "https://storage.googleapis.com/adult-gtex/"
    "bulk-qtl/v8/single-tissue-cis-qtl/GTEx_Analysis_v8_eQTL.tar"
)

# 5 TS-relevant brain regions (CSTC circuit + cerebellum)
BRAIN_REGIONS = [
    "Brain_Caudate_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Cerebellum",
]

FILE_SUFFIX = ".v8.signif_variant_gene_pairs.txt.gz"

# Session with retries
_session = requests.Session()
_adapter = HTTPAdapter(max_retries=3)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def compute_md5(path: Path) -> str:
    """Compute MD5 checksum of a file (streaming, memory-safe)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_tar(url: str, dest: Path, chunk_size: int = 65536) -> None:
    """Stream-download a file to disk (low memory usage)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    logger.info("Downloading %s", url)
    resp = _session.get(url, timeout=1800, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 / total
                mb = downloaded / 1_048_576
                print(
                    f"\r  {mb:.0f}/{total / 1_048_576:.0f} MB ({pct:.0f}%)",
                    end="",
                    flush=True,
                )

    if total > 0:
        print()

    tmp.rename(dest)
    logger.info("Saved: %s (%.0f MB)", dest, dest.stat().st_size / 1_048_576)


def _extract_brain_eqtls(tar_path: Path, dest_dir: Path) -> list[Path]:
    """Extract brain region signif_variant_gene_pairs files from the tar."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_filenames = {f"{region}{FILE_SUFFIX}" for region in BRAIN_REGIONS}
    extracted: list[Path] = []

    logger.info("Extracting %d brain region files from tar...", len(target_filenames))
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            basename = Path(member.name).name
            if basename not in target_filenames:
                continue

            member_file = tar.extractfile(member)
            if member_file is None:
                logger.warning("Could not extract %s", member.name)
                continue

            out_path = dest_dir / basename
            with open(out_path, "wb") as f:
                while True:
                    chunk = member_file.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)

            size_mb = out_path.stat().st_size / 1_048_576
            logger.info("  Extracted: %s (%.1f MB)", basename, size_mb)
            extracted.append(out_path)
            target_filenames.discard(basename)

            if not target_filenames:
                break

    if target_filenames:
        logger.warning("Not found in tar: %s", target_filenames)

    return extracted


def _write_checksums(files: list[Path], dest_dir: Path) -> dict[str, str]:
    """Compute MD5 checksums and write to md5sums.txt."""
    checksums: dict[str, str] = {}
    logger.info("Computing MD5 checksums...")
    for f in sorted(files):
        md5 = compute_md5(f)
        checksums[f.name] = md5
        logger.info("  %s  %s", md5, f.name)

    checksum_path = dest_dir / "md5sums.txt"
    with open(checksum_path, "w") as fp:
        for name, md5 in sorted(checksums.items()):
            fp.write(f"{md5}  {name}\n")
    logger.info("Checksums written to %s", checksum_path)
    return checksums


def verify_checksums(dest_dir: Path) -> bool:
    """Verify files against stored MD5 checksums."""
    checksum_path = dest_dir / "md5sums.txt"
    if not checksum_path.exists():
        logger.warning("No checksum file found at %s", checksum_path)
        return False

    all_ok = True
    with open(checksum_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            expected_md5, filename = line.split("  ", 1)
            filepath = dest_dir / filename
            if not filepath.exists():
                logger.error("MISSING: %s", filename)
                all_ok = False
                continue
            actual_md5 = compute_md5(filepath)
            if actual_md5 == expected_md5:
                logger.info("  OK: %s", filename)
            else:
                logger.error(
                    "MISMATCH: %s (expected %s, got %s)",
                    filename, expected_md5, actual_md5,
                )
                all_ok = False

    return all_ok


def _write_readme(dest_dir: Path, checksums: dict[str, str]) -> None:
    """Generate README with data provenance."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# GTEx v8 Brain eQTL Data",
        "",
        "## Source",
        "- **Dataset:** GTEx Analysis v8 single-tissue cis-eQTL significant variant-gene pairs",
        f"- **URL:** {GTEX_V8_EQTL_TAR_URL}",
        "- **Portal:** https://gtexportal.org/home/datasets",
        f"- **Downloaded:** {now}",
        "- **License:** Open access (dbGaP accession phs000424)",
        "",
        "## Brain Regions",
        "These 5 brain regions are relevant to the cortico-striato-thalamo-cortical (CSTC)",
        "circuit implicated in Tourette syndrome:",
        "",
        "| Region | File |",
        "|--------|------|",
    ]
    for region in BRAIN_REGIONS:
        filename = f"{region}{FILE_SUFFIX}"
        lines.append(f"| {region.replace('_', ' ')} | `{filename}` |")

    lines += [
        "",
        "## File Format",
        "Tab-separated, gzipped. Columns:",
        "- `variant_id`: variant ID (chr_pos_ref_alt_b38)",
        "- `gene_id`: GENCODE/Ensembl gene ID",
        "- `tss_distance`: distance to transcription start site",
        "- `ma_samples`: samples carrying minor allele",
        "- `ma_count`: total minor allele count",
        "- `maf`: minor allele frequency",
        "- `pval_nominal`: nominal p-value",
        "- `slope`: regression slope (effect size)",
        "- `slope_se`: standard error of slope",
        "- `pval_nominal_threshold`: significance threshold for the gene",
        "- `min_pval_nominal`: smallest nominal p-value for the gene",
        "- `pval_beta`: beta-approximated permutation p-value",
        "",
        "## MD5 Checksums",
    ]
    for name, md5 in sorted(checksums.items()):
        lines.append(f"- `{name}`: `{md5}`")

    lines += [
        "",
        "## Usage",
        "These files are consumed by the SNP-to-gene eQTL mapping module:",
        "```",
        "uv run python -m bioagentics.tourettes.ts_gwas_functional_annotation.snp_to_gene \\",
        f"    --eqtl-dir {dest_dir}",
        "```",
        "",
        "## Citation",
        "GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects",
        "across human tissues. Science 369, 1318-1330 (2020).",
        "DOI: 10.1126/science.aaz1776",
        "",
    ]

    readme_path = dest_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("README written to %s", readme_path)


def download_gtex_v8_eqtl(
    dest_dir: Path | None = None,
    tar_url: str = GTEX_V8_EQTL_TAR_URL,
    keep_tar: bool = False,
    verify_only: bool = False,
) -> list[Path]:
    """Download and extract GTEx v8 brain eQTL data.

    Parameters
    ----------
    dest_dir : Path
        Destination directory for extracted files.
    tar_url : str
        URL of the GTEx v8 eQTL tar archive.
    keep_tar : bool
        If True, keep the tar file after extraction.
    verify_only : bool
        If True, only verify existing checksums without downloading.

    Returns
    -------
    list[Path]
        Paths to extracted eQTL files.
    """
    if dest_dir is None:
        dest_dir = GTEX_EQTL_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)

    if verify_only:
        logger.info("Verifying existing files...")
        ok = verify_checksums(dest_dir)
        logger.info("Verification: %s", "PASSED" if ok else "FAILED")
        return sorted(dest_dir.glob(f"*{FILE_SUFFIX}"))

    # Check if files already exist
    existing = sorted(dest_dir.glob(f"*{FILE_SUFFIX}"))
    existing_names = {f.name for f in existing}
    expected_names = {f"{r}{FILE_SUFFIX}" for r in BRAIN_REGIONS}

    if expected_names.issubset(existing_names):
        logger.info(
            "All %d brain region files already present in %s",
            len(BRAIN_REGIONS), dest_dir,
        )
        logger.info("Run with --verify to check checksums, or --force to re-download")
        return existing

    # Download tar to a temp directory (cleaned up automatically)
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "GTEx_Analysis_v8_eQTL.tar"
        _download_tar(tar_url, tar_path)

        extracted = _extract_brain_eqtls(tar_path, dest_dir)

        if keep_tar:
            final_tar = dest_dir / "GTEx_Analysis_v8_eQTL.tar"
            tar_path.rename(final_tar)
            logger.info("Tar kept at %s", final_tar)

    if not extracted:
        logger.error("No files were extracted")
        return []

    checksums = _write_checksums(extracted, dest_dir)
    _write_readme(dest_dir, checksums)

    logger.info("Done. %d files in %s", len(extracted), dest_dir)
    return extracted


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download GTEx v8 brain eQTL data for TS GWAS annotation",
    )
    parser.add_argument(
        "--dest", type=Path, default=GTEX_EQTL_DIR,
        help=f"Destination directory (default: {GTEX_EQTL_DIR})",
    )
    parser.add_argument(
        "--url", type=str, default=GTEX_V8_EQTL_TAR_URL,
        help="URL of the GTEx v8 eQTL tar archive",
    )
    parser.add_argument(
        "--keep-tar", action="store_true",
        help="Keep the tar file after extraction",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify existing file checksums",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if files exist",
    )
    args = parser.parse_args()

    if args.force:
        for f in args.dest.glob(f"*{FILE_SUFFIX}"):
            f.unlink()

    download_gtex_v8_eqtl(
        dest_dir=args.dest,
        tar_url=args.url,
        keep_tar=args.keep_tar,
        verify_only=args.verify,
    )


if __name__ == "__main__":
    main()
