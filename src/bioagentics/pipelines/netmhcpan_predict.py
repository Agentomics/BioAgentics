"""HLA-peptide binding prediction via IEDB API (NetMHCpan/NetMHCIIpan backend).

Predicts binding affinity of GAS peptides to HLA alleles using the IEDB
Analysis Resource API, which wraps NetMHCpan 4.1 (MHC-I) and NetMHCIIpan 4.3
(MHC-II) prediction tools.

Classifies each peptide-allele pair as:
- strong_binder: %rank < 0.5
- weak_binder: %rank < 2.0
- non_binder: %rank >= 2.0

Usage:
    uv run python -m bioagentics.pipelines.netmhcpan_predict [--peptide-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import requests

from bioagentics.config import DATA_DIR

logger = logging.getLogger(__name__)

PEPTIDE_DIR = DATA_DIR / "pandas_pans" / "hla-peptide-presentation" / "peptide_libraries"
HLA_PANEL_PATH = DATA_DIR / "pandas_pans" / "hla-peptide-presentation" / "hla_allele_panel.json"
OUTPUT_DIR = DATA_DIR / "pandas_pans" / "hla-peptide-presentation" / "binding_predictions"

# IEDB API endpoints
IEDB_MHC_I_URL = "http://tools-cluster-interface.iedb.org/tools_api/mhci/"
IEDB_MHC_II_URL = "http://tools-cluster-interface.iedb.org/tools_api/mhcii/"

# Rate limiting
IEDB_BATCH_SIZE = 500  # Max peptides per request
IEDB_DELAY_SECONDS = 1.0  # Delay between requests
MAX_RETRIES = 3

# Binding thresholds (percent rank)
STRONG_BINDER_THRESHOLD = 0.5
WEAK_BINDER_THRESHOLD = 2.0

# IEDB allele name format mapping
# IEDB uses specific naming conventions for HLA alleles
IEDB_ALLELE_MAP_I = {
    "A*02:01": "HLA-A*02:01",
    "B*07:02": "HLA-B*07:02",
}

IEDB_ALLELE_MAP_II = {
    "DRB1*07:01": "HLA-DRB1*07:01",
    "DRB1*04:01": "HLA-DRB1*04:01",
    "DRB1*01:01": "HLA-DRB1*01:01",
    "DRB1*15:01": "HLA-DRB1*15:01",
    "DRB1*03:01": "HLA-DRB1*03:01",
    "DRB1*13:01": "HLA-DRB1*13:01",
}


@dataclass
class BindingPrediction:
    """A single peptide-allele binding prediction result."""
    peptide: str
    allele: str
    ic50: float | None
    percentile_rank: float
    binding_class: str  # strong_binder, weak_binder, non_binder
    method: str
    source_protein: str
    source_accession: str
    serotype: str
    is_virulence_factor: bool


def classify_binding(percentile_rank: float) -> str:
    """Classify binding based on percentile rank thresholds."""
    if percentile_rank < STRONG_BINDER_THRESHOLD:
        return "strong_binder"
    elif percentile_rank < WEAK_BINDER_THRESHOLD:
        return "weak_binder"
    else:
        return "non_binder"


def load_hla_panel(panel_path: Path) -> dict:
    """Load HLA allele panel and return grouped alleles by class."""
    with open(panel_path) as f:
        panel = json.load(f)

    groups = {"I": [], "II": []}
    for allele in panel["alleles"]:
        groups[allele["class"]].append(allele["four_digit_resolution"])

    return groups


def load_peptides(peptide_tsv: Path) -> list[dict]:
    """Load peptide library from TSV."""
    peptides = []
    with open(peptide_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            peptides.append(row)
    return peptides


def _iedb_request(url: str, data: dict) -> str:
    """Make IEDB API request with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, data=data, timeout=300)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning("IEDB rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            logger.warning("IEDB returned %d: %s", resp.status_code, resp.text[:200])
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("IEDB request failed (attempt %d): %s", attempt + 1, e)
                time.sleep(5 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"IEDB API request failed after {MAX_RETRIES} attempts")


def predict_mhc_i_batch(peptide_sequences: list[str], allele: str,
                        method: str = "netmhcpan_ba") -> list[dict]:
    """Predict MHC-I binding for a batch of peptides against one allele.

    Args:
        peptide_sequences: List of peptide sequences (8-11mer).
        allele: HLA allele in IEDB format (e.g., "HLA-A*02:01").
        method: Prediction method (default: netmhcpan_ba for NetMHCpan 4.1 BA).
    """
    iedb_allele = IEDB_ALLELE_MAP_I.get(allele, f"HLA-{allele}")
    sequence_text = "\n".join(peptide_sequences)

    data = {
        "method": method,
        "sequence_text": sequence_text,
        "allele": iedb_allele,
    }

    response_text = _iedb_request(IEDB_MHC_I_URL, data)
    return _parse_mhc_i_response(response_text)


def predict_mhc_ii_batch(peptide_sequences: list[str], allele: str,
                         method: str = "netmhciipan") -> list[dict]:
    """Predict MHC-II binding for a batch of peptides against one allele.

    Args:
        peptide_sequences: List of peptide sequences (15mer).
        allele: HLA allele in IEDB format (e.g., "HLA-DRB1*07:01").
        method: Prediction method (default: netmhciipan for NetMHCIIpan 4.3).
    """
    iedb_allele = IEDB_ALLELE_MAP_II.get(allele, f"HLA-{allele}")
    sequence_text = "\n".join(peptide_sequences)

    data = {
        "method": method,
        "sequence_text": sequence_text,
        "allele": iedb_allele,
    }

    response_text = _iedb_request(IEDB_MHC_II_URL, data)
    return _parse_mhc_ii_response(response_text)


def _parse_mhc_i_response(response_text: str) -> list[dict]:
    """Parse IEDB MHC-I prediction response (tab-separated)."""
    results = []
    reader = csv.DictReader(StringIO(response_text), delimiter="\t")
    for row in reader:
        try:
            rank = float(row.get("percentile_rank", row.get("rank", "999")))
            ic50 = float(row.get("ic50", "0")) if row.get("ic50") else None
            results.append({
                "peptide": row.get("peptide", ""),
                "allele": row.get("allele", ""),
                "ic50": ic50,
                "percentile_rank": rank,
                "binding_class": classify_binding(rank),
                "method": row.get("method", "netmhcpan_ba"),
            })
        except (ValueError, KeyError) as e:
            logger.debug("Skipping unparseable row: %s", e)
    return results


def _parse_mhc_ii_response(response_text: str) -> list[dict]:
    """Parse IEDB MHC-II prediction response (tab-separated)."""
    results = []
    reader = csv.DictReader(StringIO(response_text), delimiter="\t")
    for row in reader:
        try:
            rank = float(row.get("percentile_rank", row.get("rank", "999")))
            ic50 = float(row.get("ic50", "0")) if row.get("ic50") else None
            results.append({
                "peptide": row.get("peptide", row.get("core_peptide", "")),
                "allele": row.get("allele", ""),
                "ic50": ic50,
                "percentile_rank": rank,
                "binding_class": classify_binding(rank),
                "method": row.get("method", "netmhciipan"),
            })
        except (ValueError, KeyError) as e:
            logger.debug("Skipping unparseable row: %s", e)
    return results


def deduplicate_peptides(peptides: list[dict]) -> list[str]:
    """Extract unique peptide sequences from peptide library."""
    seen = set()
    unique = []
    for p in peptides:
        seq = p["sequence"]
        if seq not in seen:
            seen.add(seq)
            unique.append(seq)
    return unique


def build_peptide_metadata(peptides: list[dict]) -> dict[str, dict]:
    """Build a lookup from peptide sequence to metadata (first occurrence)."""
    meta = {}
    for p in peptides:
        seq = p["sequence"]
        if seq not in meta:
            meta[seq] = {
                "source_protein": p["source_protein"],
                "source_accession": p["source_accession"],
                "serotype": p["serotype"],
                "is_virulence_factor": p["is_virulence_factor"] == "True",
            }
    return meta


def run_predictions(peptide_dir: Path, hla_panel_path: Path,
                    output_dir: Path, mhc_class: str = "II",
                    max_peptides: int | None = None) -> Path:
    """Run binding predictions for all peptides against all alleles of a given MHC class.

    Args:
        peptide_dir: Directory containing peptide TSV files.
        hla_panel_path: Path to HLA allele panel JSON.
        output_dir: Output directory for prediction results.
        mhc_class: "I" or "II".
        max_peptides: Optional cap on unique peptides (for testing).

    Returns:
        Path to output TSV file with all predictions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    allele_groups = load_hla_panel(hla_panel_path)
    alleles = allele_groups[mhc_class]

    suffix = "mhc_i" if mhc_class == "I" else "mhc_ii"
    pattern = f"*_{suffix}_peptides.tsv"
    peptide_files = sorted(peptide_dir.glob(pattern))

    if not peptide_files:
        raise FileNotFoundError(f"No {suffix} peptide files in {peptide_dir}")

    # Collect all peptides across serotypes
    all_peptides: list[dict] = []
    for pf in peptide_files:
        all_peptides.extend(load_peptides(pf))

    unique_seqs = deduplicate_peptides(all_peptides)
    peptide_meta = build_peptide_metadata(all_peptides)

    if max_peptides:
        unique_seqs = unique_seqs[:max_peptides]

    logger.info("MHC-%s predictions: %d unique peptides x %d alleles",
                mhc_class, len(unique_seqs), len(alleles))

    predict_fn = predict_mhc_i_batch if mhc_class == "I" else predict_mhc_ii_batch

    output_path = output_dir / f"binding_predictions_{suffix}.tsv"
    fieldnames = [
        "peptide", "allele", "ic50", "percentile_rank", "binding_class",
        "method", "source_protein", "source_accession", "serotype",
        "is_virulence_factor",
    ]

    total_predictions = 0
    total_strong = 0
    total_weak = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for allele in alleles:
            logger.info("  Predicting %s vs %d peptides...", allele, len(unique_seqs))

            for batch_start in range(0, len(unique_seqs), IEDB_BATCH_SIZE):
                batch = unique_seqs[batch_start:batch_start + IEDB_BATCH_SIZE]
                batch_num = batch_start // IEDB_BATCH_SIZE + 1
                total_batches = (len(unique_seqs) + IEDB_BATCH_SIZE - 1) // IEDB_BATCH_SIZE

                logger.info("    Batch %d/%d (%d peptides)", batch_num, total_batches, len(batch))

                try:
                    results = predict_fn(batch, allele)
                except Exception as e:
                    logger.error("    Batch failed: %s", e)
                    continue

                for r in results:
                    pep_seq = r["peptide"]
                    meta = peptide_meta.get(pep_seq, {})
                    row = {
                        "peptide": pep_seq,
                        "allele": r["allele"],
                        "ic50": r["ic50"],
                        "percentile_rank": r["percentile_rank"],
                        "binding_class": r["binding_class"],
                        "method": r["method"],
                        "source_protein": meta.get("source_protein", ""),
                        "source_accession": meta.get("source_accession", ""),
                        "serotype": meta.get("serotype", ""),
                        "is_virulence_factor": meta.get("is_virulence_factor", False),
                    }
                    writer.writerow(row)
                    total_predictions += 1
                    if r["binding_class"] == "strong_binder":
                        total_strong += 1
                    elif r["binding_class"] == "weak_binder":
                        total_weak += 1

                time.sleep(IEDB_DELAY_SECONDS)

    logger.info("MHC-%s done: %d predictions, %d strong binders, %d weak binders",
                mhc_class, total_predictions, total_strong, total_weak)

    # Write summary
    summary = {
        "mhc_class": mhc_class,
        "alleles": alleles,
        "unique_peptides": len(unique_seqs),
        "total_predictions": total_predictions,
        "strong_binders": total_strong,
        "weak_binders": total_weak,
        "non_binders": total_predictions - total_strong - total_weak,
        "output_file": output_path.name,
    }
    summary_path = output_dir / f"prediction_summary_{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="HLA-peptide binding prediction via IEDB API",
    )
    parser.add_argument("--peptide-dir", type=Path, default=PEPTIDE_DIR,
                        help="Directory with peptide TSV files (default: %(default)s)")
    parser.add_argument("--hla-panel", type=Path, default=HLA_PANEL_PATH,
                        help="HLA allele panel JSON (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--mhc-class", choices=["I", "II", "both"], default="II",
                        help="MHC class to predict (default: II, per research director)")
    parser.add_argument("--max-peptides", type=int, default=None,
                        help="Cap unique peptides for testing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    classes = ["I", "II"] if args.mhc_class == "both" else [args.mhc_class]
    for mhc_class in classes:
        run_predictions(args.peptide_dir, args.hla_panel, args.output_dir,
                        mhc_class=mhc_class, max_peptides=args.max_peptides)


if __name__ == "__main__":
    main()
