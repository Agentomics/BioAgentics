"""Download PsychENCODE brain Hi-C chromatin loop data.

Downloads Hi-C-derived enhancer-promoter linkages from the PsychENCODE
resource portal (open access). Converts to the BEDPE-like TSV format
expected by the snp_to_gene Hi-C mapping module.

Data source: Wang et al. (2018) Science — PsychENCODE Phase I.
Hi-C was performed on NeuN+ and NeuN- sorted DLPFC nuclei.

Usage:
    uv run python -m bioagentics.tourettes.ts_gwas_functional_annotation.download_psychencode_hic
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from bioagentics.tourettes.ts_gwas_functional_annotation.config import DATA_DIR

logger = logging.getLogger(__name__)

# Target directory
HIC_DATA_DIR = DATA_DIR / "hic_data"

# PsychENCODE resource portal URLs (open access, no auth required)
# INT-16: Hi-C-derived enhancer-promoter linkages from DLPFC
RESOURCE_URLS = {
    "INT-16_HiC_EP_linkages_cross_assembly.csv": [
        "http://resource.psychencode.org/Datasets/Integrative/INT-16_HiC_EP_linkages_cross_assembly.csv",
        "http://adult.psychencode.org/Datasets/Integrative/INT-16_HiC_EP_linkages_cross_assembly.csv",
    ],
    "INT-16_HiC_EP_linkages.csv": [
        "http://resource.psychencode.org/Datasets/Integrative/INT-16_HiC_EP_linkages.csv",
        "http://adult.psychencode.org/Datasets/Integrative/INT-16_HiC_EP_linkages.csv",
    ],
}

# Promoter window around TSS (bp) for creating BEDPE-format anchors
PROMOTER_WINDOW_BP = 2000

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


def _download_file(urls: list[str], dest: Path, chunk_size: int = 65536) -> bool:
    """Try downloading from multiple mirror URLs. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    for url in urls:
        logger.info("Trying: %s", url)
        try:
            resp = _session.get(url, timeout=120, stream=True)
            resp.raise_for_status()
        except (requests.RequestException, ConnectionError) as e:
            logger.warning("  Failed: %s", e)
            continue

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    print(
                        f"\r  {downloaded / 1_048_576:.1f}/{total / 1_048_576:.1f} MB ({pct:.0f}%)",
                        end="",
                        flush=True,
                    )

        if total > 0:
            print()

        tmp.rename(dest)
        size_kb = dest.stat().st_size / 1024
        logger.info("Saved: %s (%.0f KB)", dest.name, size_kb)
        return True

    return False


def _convert_cross_assembly_to_bedpe(csv_path: Path, dest_dir: Path) -> Path:
    """Convert INT-16 cross-assembly CSV to BEDPE-like TSV for Hi-C mapping.

    The cross-assembly file has columns:
        Enhancer_Chromosome_hg19, Enhancer_Start_hg19, Enhancer_End_hg19,
        PEC_Enhancer_ID, Enhancer_Chromosome_hg38, Enhancer_Start_hg38,
        Enhancer_End_hg38, Transcription_Start_Site_hg19,
        Target_Gene_Name, Target_Ensembl_Name

    Output format (TSV): CHR1, START1, END1, CHR2, START2, END2, TISSUE, GENE
    where anchor1=enhancer, anchor2=promoter (TSS ± window).
    Uses hg19 coordinates (matching typical GWAS summary stats).
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d enhancer-promoter linkages from %s", len(df), csv_path.name)

    # Use hg19 coordinates (standard for most GWAS)
    enh_chr_col = "Enhancer_Chromosome_hg19"
    enh_start_col = "Enhancer_Start_hg19"
    enh_end_col = "Enhancer_End_hg19"
    tss_col = "Transcription_Start_Site_hg19"
    gene_col = "Target_Gene_Name"

    required = {enh_chr_col, enh_start_col, enh_end_col, tss_col, gene_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Cross-assembly CSV missing columns: {missing}. Got: {list(df.columns)}")

    # Drop rows with missing coordinates
    df = df.dropna(subset=[enh_chr_col, enh_start_col, enh_end_col, tss_col])

    # Clean chromosome (strip 'chr' prefix, keep autosomal)
    df["_chr"] = df[enh_chr_col].astype(str).str.replace("chr", "", case=False)
    df["_chr"] = pd.to_numeric(df["_chr"], errors="coerce")
    df = df.dropna(subset=["_chr"])
    df["_chr"] = df["_chr"].astype(int)
    df = df[df["_chr"].between(1, 22)]

    # Build BEDPE-like output
    tss_vals = df[tss_col].astype(int)
    tss_start = np.maximum(tss_vals - PROMOTER_WINDOW_BP, 0)
    gene_series = df[gene_col].astype(str)
    gene_series = gene_series.mask(gene_series == "nan", "")
    result = pd.DataFrame({
        "CHR1": df["_chr"],
        "START1": df[enh_start_col].astype(int),
        "END1": df[enh_end_col].astype(int),
        "CHR2": df["_chr"],
        "START2": tss_start,
        "END2": tss_vals + PROMOTER_WINDOW_BP,
        "TISSUE": "DLPFC",
        "GENE": gene_series,
    })

    out_path = dest_dir / "psychencode_hic_ep_linkages_hg19.tsv"
    result.to_csv(out_path, sep="\t", index=False)
    logger.info("Wrote %d interactions to %s", len(result), out_path.name)
    return out_path


def _convert_simple_to_bedpe(csv_path: Path, dest_dir: Path) -> Path:
    """Convert INT-16 simple CSV to BEDPE-like TSV.

    The simple file has columns:
        Chromosome, Transcription_Start_Site, Target_Gene_Name,
        Target_Ensembl_Name, Enhancer_Start, Enhancer_End
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d enhancer-promoter linkages from %s", len(df), csv_path.name)

    required = {"Chromosome", "Enhancer_Start", "Enhancer_End", "Transcription_Start_Site"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Simple CSV missing columns: {missing}. Got: {list(df.columns)}")

    df = df.dropna(subset=list(required))

    # Clean chromosome
    df["_chr"] = df["Chromosome"].astype(str).str.replace("chr", "", case=False)
    df["_chr"] = pd.to_numeric(df["_chr"], errors="coerce")
    df = df.dropna(subset=["_chr"])
    df["_chr"] = df["_chr"].astype(int)
    df = df[df["_chr"].between(1, 22)]

    gene_col = "Target_Gene_Name" if "Target_Gene_Name" in df.columns else None

    tss_vals = df["Transcription_Start_Site"].astype(int)
    tss_start = np.maximum(tss_vals - PROMOTER_WINDOW_BP, 0)
    if gene_col is not None:
        gene_series = df[gene_col].astype(str)
        gene_series = gene_series.mask(gene_series == "nan", "")
    else:
        gene_series = pd.Series("", index=df.index)
    result = pd.DataFrame({
        "CHR1": df["_chr"],
        "START1": df["Enhancer_Start"].astype(int),
        "END1": df["Enhancer_End"].astype(int),
        "CHR2": df["_chr"],
        "START2": tss_start,
        "END2": tss_vals + PROMOTER_WINDOW_BP,
        "TISSUE": "DLPFC",
        "GENE": gene_series,
    })

    out_path = dest_dir / "psychencode_hic_ep_linkages.tsv"
    result.to_csv(out_path, sep="\t", index=False)
    logger.info("Wrote %d interactions to %s", len(result), out_path.name)
    return out_path


def _write_checksums(files: list[Path], dest_dir: Path) -> dict[str, str]:
    """Compute MD5 checksums and write md5sums.txt."""
    checksums: dict[str, str] = {}
    for f in sorted(files):
        md5 = compute_md5(f)
        checksums[f.name] = md5
        logger.info("  %s  %s", md5, f.name)

    checksum_path = dest_dir / "md5sums.txt"
    with open(checksum_path, "w") as fp:
        for name, md5 in sorted(checksums.items()):
            fp.write(f"{md5}  {name}\n")
    return checksums


def verify_checksums(dest_dir: Path) -> bool:
    """Verify files against stored MD5 checksums."""
    checksum_path = dest_dir / "md5sums.txt"
    if not checksum_path.exists():
        logger.warning("No checksum file at %s", checksum_path)
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
            if actual_md5 != expected_md5:
                logger.error("MISMATCH: %s", filename)
                all_ok = False
            else:
                logger.info("  OK: %s", filename)

    return all_ok


def _write_readme(dest_dir: Path, checksums: dict[str, str], files: list[Path]) -> None:
    """Generate README with data provenance."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# PsychENCODE Brain Hi-C Chromatin Interaction Data",
        "",
        "## Source",
        "- **Dataset:** PsychENCODE INT-16 Hi-C-derived enhancer-promoter linkages",
        "- **Portal:** http://resource.psychencode.org/",
        "- **Tissue:** DLPFC (dorsolateral prefrontal cortex), NeuN+ and NeuN- sorted nuclei",
        f"- **Downloaded:** {now}",
        "- **License:** Open access (PsychENCODE Consortium)",
        "",
        "## Description",
        "Hi-C chromatin interaction data identifying enhancer-promoter linkages in the",
        "human dorsolateral prefrontal cortex (DLPFC). The original Hi-C data was generated",
        "from NeuN+ (neuronal) and NeuN- (non-neuronal) sorted nuclei as part of",
        "PsychENCODE Phase I (Wang et al., Science, 2018).",
        "",
        "The INT-16 dataset provides enhancer-promoter linkages derived from Hi-C contact",
        "maps, identifying distal regulatory elements connected to gene promoters via",
        "chromatin loops.",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]

    for f in sorted(files):
        if f.suffix == ".csv":
            lines.append(f"| `{f.name}` | Raw PsychENCODE INT-16 CSV |")
        elif "hg19" in f.name:
            lines.append(f"| `{f.name}` | Processed BEDPE-like TSV (hg19 coords) |")
        elif f.name.endswith(".tsv"):
            lines.append(f"| `{f.name}` | Processed BEDPE-like TSV |")

    lines += [
        "",
        "## Processed TSV Format",
        "Tab-separated with columns:",
        "- `CHR1`, `START1`, `END1`: Enhancer anchor coordinates",
        "- `CHR2`, `START2`, `END2`: Promoter anchor coordinates (TSS +/- 2kb)",
        "- `TISSUE`: Brain tissue of origin (DLPFC)",
        "- `GENE`: Target gene name",
        "",
        "## Usage",
        "These files are consumed by the SNP-to-gene Hi-C mapping module:",
        "```",
        "uv run python -m bioagentics.tourettes.ts_gwas_functional_annotation.snp_to_gene \\",
        f"    --hic-dir {dest_dir}",
        "```",
        "",
        "## Synapse Resources",
        "Additional raw Hi-C data (contact matrices, loop calls, TAD boundaries)",
        "is available on Synapse with a registered account:",
        "- PsychENCODE Knowledge Portal: https://psychencode.synapse.org/",
        "- Hi-C data: syn21760712",
        "- NeuN+/NeuN- data: syn4566010",
        "",
        "## Citation",
        "Wang D, Liu S, Warrell J, et al. Comprehensive functional genomic resource",
        "and integrative model for the human brain. Science 362, eaat8464 (2018).",
        "DOI: 10.1126/science.aat8464",
        "",
        "## MD5 Checksums",
    ]
    for name, md5 in sorted(checksums.items()):
        lines.append(f"- `{name}`: `{md5}`")

    lines.append("")

    readme_path = dest_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("README written to %s", readme_path)


def download_psychencode_hic(
    dest_dir: Path | None = None,
    verify_only: bool = False,
    prefer_cross_assembly: bool = True,
) -> list[Path]:
    """Download and process PsychENCODE Hi-C enhancer-promoter linkages.

    Parameters
    ----------
    dest_dir : Path
        Destination directory.
    verify_only : bool
        Only verify existing checksums.
    prefer_cross_assembly : bool
        Prefer the cross-assembly file (has hg19 + hg38 coords).

    Returns
    -------
    list[Path]
        Paths to processed TSV files.
    """
    if dest_dir is None:
        dest_dir = HIC_DATA_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)

    if verify_only:
        ok = verify_checksums(dest_dir)
        logger.info("Verification: %s", "PASSED" if ok else "FAILED")
        return sorted(dest_dir.glob("*.tsv"))

    # Check for existing processed files
    existing_tsv = sorted(dest_dir.glob("psychencode_hic_ep_linkages*.tsv"))
    if existing_tsv:
        logger.info(
            "Processed Hi-C files already present in %s: %s",
            dest_dir, [f.name for f in existing_tsv],
        )
        logger.info("Run with --verify to check checksums, or --force to re-download")
        return existing_tsv

    downloaded_files: list[Path] = []
    processed_files: list[Path] = []

    # Download cross-assembly file first (preferred — has hg19 coords)
    if prefer_cross_assembly:
        cross_name = "INT-16_HiC_EP_linkages_cross_assembly.csv"
        cross_dest = dest_dir / cross_name
        if _download_file(RESOURCE_URLS[cross_name], cross_dest):
            downloaded_files.append(cross_dest)
            tsv = _convert_cross_assembly_to_bedpe(cross_dest, dest_dir)
            processed_files.append(tsv)

    # Also download simple file as fallback
    simple_name = "INT-16_HiC_EP_linkages.csv"
    simple_dest = dest_dir / simple_name
    if not processed_files:
        if _download_file(RESOURCE_URLS[simple_name], simple_dest):
            downloaded_files.append(simple_dest)
            tsv = _convert_simple_to_bedpe(simple_dest, dest_dir)
            processed_files.append(tsv)

    if not processed_files:
        logger.error(
            "Could not download PsychENCODE Hi-C data. "
            "The resource portal may be temporarily unavailable. "
            "Try again later or download manually from: "
            "http://resource.psychencode.org/ or https://psychencode.synapse.org/"
        )
        return []

    all_files = downloaded_files + processed_files
    checksums = _write_checksums(all_files, dest_dir)
    _write_readme(dest_dir, checksums, all_files)

    logger.info("Done. %d files in %s", len(all_files), dest_dir)
    return processed_files


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download PsychENCODE brain Hi-C chromatin interaction data",
    )
    parser.add_argument(
        "--dest", type=Path, default=HIC_DATA_DIR,
        help=f"Destination directory (default: {HIC_DATA_DIR})",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify existing file checksums",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--simple-only", action="store_true",
        help="Only download simple CSV (no cross-assembly)",
    )
    args = parser.parse_args()

    if args.force:
        for f in args.dest.glob("psychencode_hic_ep_linkages*"):
            f.unlink()
        for f in args.dest.glob("INT-16_HiC_EP_linkages*"):
            f.unlink()

    download_psychencode_hic(
        dest_dir=args.dest,
        verify_only=args.verify,
        prefer_cross_assembly=not args.simple_only,
    )


if __name__ == "__main__":
    main()
