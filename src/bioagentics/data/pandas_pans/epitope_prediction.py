"""B-cell epitope prediction for GAS-human mimicry hit pairs.

Predicts linear B-cell epitopes on both GAS and human protein sequences
from mimicry screening hits. Uses a local composite antigenicity score
combining Parker hydrophilicity, Kolaskar-Tongaonkar antigenicity, and
Emini surface accessibility scales — the same features underlying classical
B-cell epitope prediction tools.

Identifies shared/overlapping epitope regions between GAS and human proteins
that represent potential cross-reactive antibody binding sites.

Usage:
    uv run python -m bioagentics.data.pandas_pans.epitope_prediction [--dest DIR]
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
OUTPUT_DIR = Path("output/pandas_pans/gas-molecular-mimicry-mapping/epitope_predictions")

# Minimum epitope length for relevance (B-cell epitopes are typically 8-20 aa)
MIN_EPITOPE_LENGTH = 8

# Antigenicity threshold (combined score, 0-1 scale)
ANTIGENICITY_THRESHOLD = 0.55

# ── Amino acid propensity scales ──
# Parker hydrophilicity (Parker et al. 1986) — higher = more hydrophilic/surface
_PARKER_HYDROPHILICITY: dict[str, float] = {
    "A": 2.1, "R": 4.2, "N": 7.0, "D": 10.0, "C": 1.4,
    "Q": 6.0, "E": 7.8, "G": 5.7, "H": 2.1, "I": -8.0,
    "L": -9.2, "K": 5.7, "M": -4.2, "F": -9.2, "P": 2.1,
    "S": 6.5, "T": 5.2, "W": -10.0, "Y": -1.9, "V": -3.7,
}

# Kolaskar-Tongaonkar antigenicity (1990) — propensity for antigenic determinants
_KOLASKAR_ANTIGENICITY: dict[str, float] = {
    "A": 1.064, "R": 0.873, "N": 0.776, "D": 0.866, "C": 1.412,
    "Q": 0.761, "E": 0.851, "G": 0.874, "H": 1.105, "I": 1.152,
    "L": 1.25, "K": 0.930, "M": 0.826, "F": 1.091, "P": 1.064,
    "S": 1.012, "T": 0.909, "W": 0.893, "Y": 1.161, "V": 1.383,
}

# Emini surface accessibility (1985) — relative surface probability
_EMINI_SURFACE: dict[str, float] = {
    "A": 0.815, "R": 1.475, "N": 1.296, "D": 1.283, "C": 0.394,
    "Q": 1.348, "E": 1.445, "G": 0.714, "H": 1.180, "I": 0.603,
    "L": 0.603, "K": 1.596, "M": 0.714, "F": 0.603, "P": 1.236,
    "S": 1.115, "T": 1.184, "W": 0.603, "Y": 0.714, "V": 0.606,
}

# Chou-Fasman turn propensity — turns are exposed and antigenic
_TURN_PROPENSITY: dict[str, float] = {
    "A": 0.66, "R": 0.95, "N": 1.56, "D": 1.46, "C": 1.19,
    "Q": 0.98, "E": 0.74, "G": 1.56, "H": 0.95, "I": 0.47,
    "L": 0.59, "K": 1.01, "M": 0.60, "F": 0.60, "P": 1.52,
    "S": 1.43, "T": 0.96, "W": 0.96, "Y": 1.14, "V": 0.50,
}


def _normalize_scale(scale: dict[str, float]) -> dict[str, float]:
    """Normalize a propensity scale to 0-1 range."""
    values = list(scale.values())
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    return {k: (v - mn) / rng for k, v in scale.items()}


# Pre-normalize all scales
_NORM_PARKER = _normalize_scale(_PARKER_HYDROPHILICITY)
_NORM_KOLASKAR = _normalize_scale(_KOLASKAR_ANTIGENICITY)
_NORM_EMINI = _normalize_scale(_EMINI_SURFACE)
_NORM_TURN = _normalize_scale(_TURN_PROPENSITY)


def compute_antigenicity_profile(sequence: str, window: int = 7) -> list[float]:
    """Compute per-residue antigenicity score using composite propensity scales.

    Combines Parker hydrophilicity (surface exposure), Kolaskar-Tongaonkar
    antigenicity, Emini surface accessibility, and Chou-Fasman turn propensity
    with a sliding window average. Returns scores for each residue (0-1 scale).
    """
    if len(sequence) < window:
        return [0.5] * len(sequence)

    # Compute raw per-residue composite score (equal weight to each scale)
    raw_scores = []
    for aa in sequence.upper():
        parker = _NORM_PARKER.get(aa, 0.5)
        kolaskar = _NORM_KOLASKAR.get(aa, 0.5)
        emini = _NORM_EMINI.get(aa, 0.5)
        turn = _NORM_TURN.get(aa, 0.5)
        # Weight: hydrophilicity + surface > antigenicity + turn
        raw_scores.append(0.3 * parker + 0.2 * kolaskar + 0.3 * emini + 0.2 * turn)

    # Sliding window smoothing
    scores = np.convolve(raw_scores, np.ones(window) / window, mode="same").tolist()

    return scores


def predict_epitopes(sequence: str, threshold: float = ANTIGENICITY_THRESHOLD,
                     min_length: int = MIN_EPITOPE_LENGTH) -> list[dict]:
    """Predict linear B-cell epitopes from protein sequence.

    Uses composite antigenicity scoring with sliding window smoothing.
    Returns list of predicted epitope regions with positions and scores.
    """
    if not sequence or len(sequence) < min_length:
        return []

    scores = compute_antigenicity_profile(sequence)

    # Extract contiguous regions above threshold
    epitopes: list[dict] = []
    current_start = -1
    current_residues: list[str] = []
    current_scores: list[float] = []

    for i, (aa, score) in enumerate(zip(sequence, scores)):
        if score >= threshold:
            if current_start < 0:
                current_start = i + 1  # 1-based
            current_residues.append(aa)
            current_scores.append(score)
        else:
            if current_start >= 0 and len(current_residues) >= min_length:
                epitopes.append({
                    "start": current_start,
                    "end": current_start + len(current_residues) - 1,
                    "length": len(current_residues),
                    "sequence": "".join(current_residues),
                    "mean_score": sum(current_scores) / len(current_scores),
                    "max_score": max(current_scores),
                })
            current_start = -1
            current_residues = []
            current_scores = []

    # Handle last region
    if current_start >= 0 and len(current_residues) >= min_length:
        epitopes.append({
            "start": current_start,
            "end": current_start + len(current_residues) - 1,
            "length": len(current_residues),
            "sequence": "".join(current_residues),
            "mean_score": sum(current_scores) / len(current_scores),
            "max_score": max(current_scores),
        })

    return epitopes


def load_fasta_sequences(fasta_path: Path) -> dict[str, str]:
    """Load protein sequences from FASTA file, keyed by ID (first field)."""
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


def find_overlapping_epitopes(
    gas_epitopes: list[dict],
    human_epitopes: list[dict],
    gas_start: int,
    gas_end: int,
    human_start: int,
    human_end: int,
) -> list[dict]:
    """Find epitope regions that overlap with the aligned mimicry region.

    Checks if predicted epitopes on GAS or human proteins fall within
    the aligned (mimicry) region, suggesting cross-reactive antibody binding.
    """
    overlaps: list[dict] = []

    gas_in_region = [
        e for e in gas_epitopes
        if e["start"] <= gas_end and e["end"] >= gas_start
    ]
    human_in_region = [
        e for e in human_epitopes
        if e["start"] <= human_end and e["end"] >= human_start
    ]

    for ge in gas_in_region:
        for he in human_in_region:
            overlaps.append({
                "gas_epitope_start": ge["start"],
                "gas_epitope_end": ge["end"],
                "gas_epitope_seq": ge["sequence"],
                "gas_epitope_score": ge["mean_score"],
                "human_epitope_start": he["start"],
                "human_epitope_end": he["end"],
                "human_epitope_seq": he["sequence"],
                "human_epitope_score": he["mean_score"],
                "cross_reactivity_score": (ge["mean_score"] + he["mean_score"]) / 2,
            })

    return overlaps


def get_unique_hit_pairs(hits_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate hits to unique GAS-human protein pairs (best hit per pair)."""
    return (
        hits_df.sort_values("bitscore", ascending=False)
        .drop_duplicates(subset=["qseqid", "human_accession"], keep="first")
        .reset_index(drop=True)
    )


def run_epitope_analysis(hits_df: pd.DataFrame,
                          gas_seqs: dict[str, str],
                          human_seqs: dict[str, str],
                          dest: Path) -> pd.DataFrame:
    """Run B-cell epitope prediction for all mimicry hit pairs."""
    pairs = get_unique_hit_pairs(hits_df)
    logger.info("Analyzing %d unique GAS-human mimicry pairs", len(pairs))

    results: list[dict] = []
    epitope_cache: dict[str, list[dict]] = {}

    for idx, row in pairs.iterrows():
        gas_id = row["qseqid"]
        human_acc = row["human_accession"]
        human_gene = row.get("human_gene", "")
        if pd.isna(human_gene):
            human_gene = ""

        logger.info(
            "  [%d/%d] %s vs %s (%s)",
            idx + 1, len(pairs), gas_id[:30], human_acc, human_gene,  # type: ignore[operator]
        )

        # Get sequences
        gas_seq = gas_seqs.get(gas_id, "")
        human_seq = human_seqs.get(human_acc, "")

        if not gas_seq or not human_seq:
            logger.warning("    Missing sequence: gas=%s human=%s", bool(gas_seq), bool(human_seq))
            continue

        # Predict epitopes (with caching)
        if gas_id not in epitope_cache:
            epitope_cache[gas_id] = predict_epitopes(gas_seq)
        gas_epitopes = epitope_cache[gas_id]

        if human_acc not in epitope_cache:
            epitope_cache[human_acc] = predict_epitopes(human_seq)
        human_epitopes = epitope_cache[human_acc]

        # Find overlapping epitopes in aligned region
        overlaps = find_overlapping_epitopes(
            gas_epitopes, human_epitopes,
            int(row["qstart"]), int(row["qend"]),
            int(row["sstart"]), int(row["send"]),
        )

        best_score = max((o["cross_reactivity_score"] for o in overlaps), default=0.0)

        results.append({
            "gas_protein": gas_id,
            "human_accession": human_acc,
            "human_gene": human_gene,
            "pident": row["pident"],
            "alignment_length": row["length"],
            "gas_epitope_count": len(gas_epitopes),
            "human_epitope_count": len(human_epitopes),
            "overlapping_epitope_pairs": len(overlaps),
            "best_cross_reactivity_score": round(best_score, 4),
            "gas_aligned_region": f"{int(row['qstart'])}-{int(row['qend'])}",
            "human_aligned_region": f"{int(row['sstart'])}-{int(row['send'])}",
        })

        # Save per-pair overlap details
        if overlaps:
            overlap_df = pd.DataFrame(overlaps)
            pair_name = f"{gas_id.replace('|', '_')}__vs__{human_acc}"
            overlap_df.to_csv(dest / f"overlap_{pair_name}.tsv", sep="\t", index=False)

    return pd.DataFrame(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="B-cell epitope prediction for GAS-human mimicry pairs",
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
    logger.info("Loaded %d filtered mimicry hits", len(hits_df))

    # Load sequences
    gas_seqs = load_fasta_sequences(PROTEOME_DIR / "gas_combined.fasta")
    human_seqs_raw = load_fasta_sequences(HUMAN_DIR / "human_bg_targets.fasta")
    # Human FASTA IDs are ACCESSION|GENE|REGIONS — index by accession only
    human_seqs = {k.split("|")[0]: v for k, v in human_seqs_raw.items()}
    logger.info("Loaded %d GAS and %d human protein sequences", len(gas_seqs), len(human_seqs))

    # Run epitope analysis
    results = run_epitope_analysis(hits_df, gas_seqs, human_seqs, args.dest)

    # Save results
    results_path = args.dest / "epitope_predictions.tsv"
    results.to_csv(results_path, sep="\t", index=False)
    logger.info("Results saved: %s (%d pairs)", results_path.name, len(results))

    # Summary
    if not results.empty and "overlapping_epitope_pairs" in results.columns:
        with_overlaps = results[results["overlapping_epitope_pairs"] > 0]
        logger.info(
            "Summary: %d pairs analyzed, %d with overlapping epitopes in aligned region",
            len(results), len(with_overlaps),
        )
    else:
        logger.warning("No pairs could be analyzed — check sequence availability")

    logger.info("Done. Epitope predictions in %s", args.dest)


if __name__ == "__main__":
    main()
