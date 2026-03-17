"""VMAT2 negative control calibration for TS drug scoring pipeline.

VMAT2 inhibitors (valbenazine, deutetrabenazine) showed disappointing
TS trial results despite strong mechanistic rationale. This module:
1. Computes network proximity and signature scores for VMAT2 inhibitors
2. Establishes a "false positive" threshold
3. Penalizes drugs scoring similarly to VMAT2 inhibitors

Also uses ecopipam as a positive control (Phase 3 success).

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.vmat2_calibration
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network"

PROXIMITY_PATH = OUTPUT_DIR / "network_proximity_scores.csv"
SIGNATURE_PATH = OUTPUT_DIR / "lincs_signature_scores.csv"
OUTPUT_PATH = OUTPUT_DIR / "vmat2_calibration.csv"

# Control drugs for calibration
POSITIVE_CONTROLS = {
    "aripiprazole", "ecopipam", "haloperidol", "pimozide",
    "risperidone", "clonidine", "guanfacine",
}
NEGATIVE_CONTROLS = {
    "valbenazine", "deutetrabenazine",
}

# VMAT2-related gene targets (drugs targeting these get penalized if
# they don't also hit other TS-relevant pathways)
VMAT2_GENES = {"SLC18A2", "SLC18A1"}


def load_signature_scores(path: Path) -> dict[str, float]:
    """Load signature concordance scores, keyed by lowercase compound name."""
    scores: dict[str, float] = {}
    if not path.exists():
        return scores
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["compound_name"].lower().strip()
            try:
                scores[name] = float(row["concordance_score"])
            except (ValueError, KeyError):
                pass
    return scores


def compute_vmat2_penalty(
    drug_name: str,
    drug_targets: set[str] | None = None,
    sig_score: float | None = None,
    vmat2_sig_range: tuple[float, float] = (-0.20, 0.0),
) -> float:
    """Compute penalty for drugs that resemble VMAT2 inhibitor profile.

    Returns a penalty value between 0.0 (no penalty) and 1.0 (full penalty).
    """
    penalty = 0.0

    # 1. Direct VMAT2 targeting penalty
    if drug_targets and drug_targets & VMAT2_GENES:
        # Pure VMAT2 inhibitors get high penalty
        non_vmat2_targets = drug_targets - VMAT2_GENES
        if not non_vmat2_targets:
            penalty = max(penalty, 0.8)
        else:
            # Polypharmacology — lower penalty if also hitting other targets
            penalty = max(penalty, 0.3)

    # 2. Signature similarity to VMAT2 inhibitors
    if sig_score is not None:
        low, high = vmat2_sig_range
        if low <= sig_score <= high:
            # Score falls in the VMAT2 "false positive" range
            penalty = max(penalty, 0.5)

    # 3. Known negative control
    if drug_name.lower() in {nc.lower() for nc in NEGATIVE_CONTROLS}:
        penalty = max(penalty, 0.9)

    return penalty


def calibrate_scoring(
    sig_scores: dict[str, float],
) -> dict[str, dict]:
    """Run calibration analysis and return drug-level calibration data."""
    # Compute calibration metrics
    pos_scores = [sig_scores.get(d.lower(), None) for d in POSITIVE_CONTROLS]
    neg_scores = [sig_scores.get(d.lower(), None) for d in NEGATIVE_CONTROLS]

    pos_valid = [s for s in pos_scores if s is not None]
    neg_valid = [s for s in neg_scores if s is not None]

    print(f"  Positive control signature scores: {pos_valid}")
    print(f"  Negative control signature scores: {neg_valid}")

    if pos_valid and neg_valid:
        pos_mean = sum(pos_valid) / len(pos_valid)
        neg_mean = sum(neg_valid) / len(neg_valid)
        print(f"  Positive mean: {pos_mean:.3f}, Negative mean: {neg_mean:.3f}")
        print(f"  Separation: {abs(pos_mean - neg_mean):.3f}")

        # VMAT2 false positive range
        vmat2_range = (min(neg_valid) - 0.05, max(neg_valid) + 0.05)
        print(f"  VMAT2 false positive range: {vmat2_range}")
    else:
        vmat2_range = (-0.20, 0.0)

    # Compute penalties for all drugs in signature scores
    calibration: dict[str, dict] = {}
    for drug_name, score in sig_scores.items():
        penalty = compute_vmat2_penalty(drug_name, sig_score=score, vmat2_sig_range=vmat2_range)
        calibration[drug_name] = {
            "drug_name": drug_name,
            "signature_score": score,
            "vmat2_penalty": penalty,
            "is_positive_control": drug_name in {d.lower() for d in POSITIVE_CONTROLS},
            "is_negative_control": drug_name in {d.lower() for d in NEGATIVE_CONTROLS},
        }

    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="VMAT2 negative control calibration")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running VMAT2 calibration...")
    sig_scores = load_signature_scores(SIGNATURE_PATH)
    print(f"  Loaded {len(sig_scores)} signature scores")

    calibration = calibrate_scoring(sig_scores)

    # Save
    if calibration:
        rows = sorted(calibration.values(), key=lambda x: x["vmat2_penalty"], reverse=True)
        fieldnames = ["drug_name", "signature_score", "vmat2_penalty",
                      "is_positive_control", "is_negative_control"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved calibration data for {len(rows)} drugs to {args.output}")

    # Summary
    penalized = [d for d in calibration.values() if d["vmat2_penalty"] > 0]
    print(f"\nDrugs with VMAT2 penalty: {len(penalized)}")
    for d in sorted(penalized, key=lambda x: -x["vmat2_penalty"]):
        ctrl = ""
        if d["is_positive_control"]:
            ctrl = " [POSITIVE CONTROL]"
        elif d["is_negative_control"]:
            ctrl = " [NEGATIVE CONTROL]"
        print(f"  {d['drug_name']}: penalty={d['vmat2_penalty']:.2f}{ctrl}")


if __name__ == "__main__":
    main()
