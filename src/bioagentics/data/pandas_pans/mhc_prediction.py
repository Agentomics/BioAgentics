"""MHC class II binding prediction for GAS-human mimicry hit pairs.

Predicts peptide binding to HLA-DR alleles relevant to PANDAS-susceptible
populations using position-specific scoring matrices (PSSMs) derived from
published HLA-DR binding motifs. Identifies GAS peptides that mimic human
neuronal peptides in MHC-II presentation — these drive T-cell cross-reactivity.

HLA-DR alleles of interest:
- DRB1*04:01 — associated with autoimmune susceptibility
- DRB1*07:01 — common in Caucasian populations
- DRB1*01:01 — well-characterized binding motif

Usage:
    uv run python -m bioagentics.data.pandas_pans.mhc_prediction [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping")
SCREEN_DIR = PROJECT_DIR / "mimicry_screen"
PROTEOME_DIR = PROJECT_DIR / "proteomes"
HUMAN_DIR = PROJECT_DIR / "human_targets"
OUTPUT_DIR = Path("output/pandas_pans/gas-molecular-mimicry-mapping/mhc_predictions")

# MHC-II binding core is typically 9 amino acids
CORE_LENGTH = 9
# 15-mer peptides are standard for MHC-II (9aa core + 3aa flanking each side)
PEPTIDE_LENGTH = 15
# Binding affinity threshold (percentile rank, lower = better binder)
BINDING_THRESHOLD = 2.0  # top 2% = strong binder

# ── HLA-DR binding motifs ──
# Position-specific scoring matrices for HLA-DR alleles.
# Based on published binding motifs and anchor residue preferences.
# Positions are P1-P9 of the 9-mer binding core.
# Values represent relative preference (log-odds style, higher = preferred).

# Standard amino acid alphabet
_AA = "ACDEFGHIKLMNPQRSTVWY"

# DRB1*04:01 — autoimmune-associated allele
# P1: aromatic/hydrophobic anchor (F, Y, W, I, L)
# P4: small/polar (S, T, D)
# P6: small (A, G, S)
# P9: hydrophobic (L, I, V, F)
_DRB1_0401_MOTIF: dict[int, dict[str, float]] = {
    1: {"F": 2.0, "Y": 1.8, "W": 1.5, "I": 1.2, "L": 1.0, "V": 0.5, "M": 0.5},
    4: {"S": 1.5, "T": 1.3, "D": 1.0, "N": 0.8, "A": 0.5, "Q": 0.5},
    6: {"A": 1.5, "G": 1.3, "S": 1.0, "T": 0.5, "V": 0.3},
    9: {"L": 2.0, "I": 1.5, "V": 1.3, "F": 1.0, "M": 0.8, "A": 0.3},
}

# DRB1*07:01 — common Caucasian allele
# P1: hydrophobic (L, I, V, F, Y)
# P4: positive charge tolerated (K, R) + small
# P6: small residues
# P9: hydrophobic
_DRB1_0701_MOTIF: dict[int, dict[str, float]] = {
    1: {"L": 2.0, "I": 1.5, "V": 1.3, "F": 1.8, "Y": 1.5, "W": 1.2, "M": 0.5},
    4: {"K": 1.0, "R": 0.8, "S": 1.2, "T": 1.0, "Q": 0.8, "N": 0.5},
    6: {"S": 1.5, "A": 1.3, "G": 1.0, "T": 0.5},
    9: {"L": 2.0, "V": 1.5, "I": 1.3, "F": 1.0, "Y": 0.8, "M": 0.5},
}

# DRB1*01:01 — well-characterized reference allele
# P1: large hydrophobic anchor (Y, F, W > I, L, V)
# P4: acidic/small (D, E, Q, S)
# P6: small preferred (A, G, S)
# P9: hydrophobic (L, I, V, F, M)
_DRB1_0101_MOTIF: dict[int, dict[str, float]] = {
    1: {"Y": 2.0, "F": 1.8, "W": 1.5, "I": 1.0, "L": 0.8, "V": 0.5},
    4: {"D": 1.5, "E": 1.3, "Q": 1.0, "S": 0.8, "N": 0.5, "T": 0.5},
    6: {"A": 1.5, "G": 1.3, "S": 1.0, "T": 0.5, "V": 0.3},
    9: {"L": 2.0, "I": 1.5, "V": 1.3, "F": 1.0, "M": 0.8, "Y": 0.5},
}

HLA_ALLELES: dict[str, dict[int, dict[str, float]]] = {
    "DRB1*04:01": _DRB1_0401_MOTIF,
    "DRB1*07:01": _DRB1_0701_MOTIF,
    "DRB1*01:01": _DRB1_0101_MOTIF,
}


def score_peptide_core(core: str, motif: dict[int, dict[str, float]]) -> float:
    """Score a 9-mer peptide core against an HLA-DR binding motif.

    Returns a binding score (higher = better predicted binder).
    """
    if len(core) != CORE_LENGTH:
        return 0.0

    score = 0.0
    for pos, preferences in motif.items():
        aa = core[pos - 1]  # motif positions are 1-based
        score += preferences.get(aa, -0.5)  # penalty for non-preferred

    return score


def predict_binding(sequence: str, allele: str = "DRB1*04:01") -> list[dict]:
    """Predict MHC-II binding peptides from a protein sequence.

    Slides a 15-mer window across the sequence, scores each possible
    9-mer core alignment within the 15-mer, and returns the best core
    for each position.
    """
    motif = HLA_ALLELES.get(allele)
    if motif is None:
        logger.warning("Unknown allele: %s", allele)
        return []

    if len(sequence) < PEPTIDE_LENGTH:
        return []

    results: list[dict] = []

    for i in range(len(sequence) - PEPTIDE_LENGTH + 1):
        peptide = sequence[i:i + PEPTIDE_LENGTH]

        # Try all possible 9-mer core alignments within the 15-mer
        best_score = -float("inf")
        best_core_offset = 0

        for offset in range(PEPTIDE_LENGTH - CORE_LENGTH + 1):
            core = peptide[offset:offset + CORE_LENGTH]
            score = score_peptide_core(core, motif)
            if score > best_score:
                best_score = score
                best_core_offset = offset

        best_core = peptide[best_core_offset:best_core_offset + CORE_LENGTH]

        results.append({
            "position": i + 1,  # 1-based
            "peptide": peptide,
            "core": best_core,
            "core_offset": best_core_offset,
            "score": round(best_score, 3),
        })

    return results


def get_top_binders(predictions: list[dict], percentile: float = BINDING_THRESHOLD) -> list[dict]:
    """Filter predictions to keep only strong binders (top percentile)."""
    if not predictions:
        return []

    scores = [p["score"] for p in predictions]
    threshold = np.percentile(scores, 100 - percentile)

    return [p for p in predictions if p["score"] >= threshold]


def find_cross_reactive_peptides(
    gas_binders: list[dict],
    human_binders: list[dict],
    gas_start: int,
    gas_end: int,
    human_start: int,
    human_end: int,
) -> list[dict]:
    """Find peptides that bind the same HLA allele in both GAS and human proteins
    within the aligned mimicry region."""
    cross_reactive: list[dict] = []

    # Filter to peptides in the aligned region
    gas_in_region = [
        b for b in gas_binders
        if b["position"] >= gas_start and b["position"] + PEPTIDE_LENGTH - 1 <= gas_end
    ]
    human_in_region = [
        b for b in human_binders
        if b["position"] >= human_start and b["position"] + PEPTIDE_LENGTH - 1 <= human_end
    ]

    if not gas_in_region or not human_in_region:
        return cross_reactive

    for gb in gas_in_region:
        for hb in human_in_region:
            cross_reactive.append({
                "gas_position": gb["position"],
                "gas_peptide": gb["peptide"],
                "gas_core": gb["core"],
                "gas_score": gb["score"],
                "human_position": hb["position"],
                "human_peptide": hb["peptide"],
                "human_core": hb["core"],
                "human_score": hb["score"],
                "cross_reactivity_score": (gb["score"] + hb["score"]) / 2,
            })

    # Sort by cross-reactivity score
    cross_reactive.sort(key=lambda x: x["cross_reactivity_score"], reverse=True)
    return cross_reactive


def load_fasta_sequences(fasta_path: Path) -> dict[str, str]:
    """Load protein sequences from FASTA file, keyed by ID."""
    sequences: dict[str, str] = {}
    current_id = ""
    current_seq: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_id and current_seq:
        sequences[current_id] = "".join(current_seq)
    return sequences


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="MHC-II binding prediction for GAS-human mimicry pairs",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.dest.mkdir(parents=True, exist_ok=True)

    # Load mimicry hits
    filtered_path = SCREEN_DIR / "hits_filtered.tsv"
    if not filtered_path.exists():
        raise FileNotFoundError(f"Run mimicry_screen.py first: {filtered_path}")
    hits_df = pd.read_csv(filtered_path, sep="\t")

    # Deduplicate to unique pairs
    pairs = (
        hits_df.sort_values("bitscore", ascending=False)
        .drop_duplicates(subset=["qseqid", "human_accession"], keep="first")
        .reset_index(drop=True)
    )
    logger.info("Analyzing %d unique mimicry pairs across %d HLA alleles",
                len(pairs), len(HLA_ALLELES))

    # Load sequences
    gas_seqs = load_fasta_sequences(PROTEOME_DIR / "gas_combined.fasta")
    human_seqs_raw = load_fasta_sequences(HUMAN_DIR / "human_bg_targets.fasta")
    human_seqs = {k.split("|")[0]: v for k, v in human_seqs_raw.items()}

    # Predict for each allele
    all_results: list[dict] = []
    binder_cache: dict[str, dict[str, list[dict]]] = {}  # protein -> allele -> binders

    for idx, row in pairs.iterrows():
        gas_id = row["qseqid"]
        human_acc = row["human_accession"]
        human_gene = str(row.get("human_gene", "")) if pd.notna(row.get("human_gene")) else ""

        gas_seq = gas_seqs.get(gas_id, "")
        human_seq = human_seqs.get(human_acc, "")

        if not gas_seq or not human_seq:
            continue

        logger.info("  [%d/%d] %s vs %s (%s)",
                     idx + 1, len(pairs), gas_id[:30], human_acc, human_gene)  # type: ignore[operator]

        for allele in HLA_ALLELES:
            # Cache predictions
            if gas_id not in binder_cache:
                binder_cache[gas_id] = {}
            if allele not in binder_cache[gas_id]:
                preds = predict_binding(gas_seq, allele)
                binder_cache[gas_id][allele] = get_top_binders(preds)

            if human_acc not in binder_cache:
                binder_cache[human_acc] = {}
            if allele not in binder_cache[human_acc]:
                preds = predict_binding(human_seq, allele)
                binder_cache[human_acc][allele] = get_top_binders(preds)

            gas_binders = binder_cache[gas_id][allele]
            human_binders = binder_cache[human_acc][allele]

            # Find cross-reactive peptides in aligned region
            cross = find_cross_reactive_peptides(
                gas_binders, human_binders,
                int(row["qstart"]), int(row["qend"]),
                int(row["sstart"]), int(row["send"]),
            )

            best_score = max((c["cross_reactivity_score"] for c in cross), default=0.0)

            all_results.append({
                "gas_protein": gas_id,
                "human_accession": human_acc,
                "human_gene": human_gene,
                "hla_allele": allele,
                "pident": row["pident"],
                "alignment_length": row["length"],
                "gas_binder_count": len(gas_binders),
                "human_binder_count": len(human_binders),
                "cross_reactive_pairs": len(cross),
                "best_cross_reactivity_score": round(best_score, 3),
                "gas_aligned_region": f"{int(row['qstart'])}-{int(row['qend'])}",
                "human_aligned_region": f"{int(row['sstart'])}-{int(row['send'])}",
            })

    results_df = pd.DataFrame(all_results)
    results_path = args.dest / "mhc_predictions.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    logger.info("Results saved: %s (%d entries)", results_path.name, len(results_df))

    # Summary per allele
    if not results_df.empty:
        for allele in HLA_ALLELES:
            allele_df = results_df[results_df["hla_allele"] == allele]
            with_cross = allele_df[allele_df["cross_reactive_pairs"] > 0]
            logger.info(
                "  %s: %d pairs, %d with cross-reactive peptides",
                allele, len(allele_df), len(with_cross),
            )

    logger.info("Done. MHC-II predictions in %s", args.dest)


if __name__ == "__main__":
    main()
