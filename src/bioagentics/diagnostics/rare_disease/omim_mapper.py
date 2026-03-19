"""Parse HPO annotation file (phenotype.hpoa) to map OMIM diseases to HPO terms.

The phenotype.hpoa file contains curated disease-phenotype associations with
frequency qualifiers and evidence codes. This module parses it and produces
a mapping: {disease_id: [(hpo_id, frequency, evidence_code, onset), ...]}.

Supports both OMIM and ORPHA disease prefixes.

Output:
    data/diagnostics/rare-disease-phenotype-matcher/disease_hpo_map.json

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.omim_mapper [--hpoa PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "rare-disease-phenotype-matcher"

# Standard HPO frequency terms and their approximate numeric values
FREQUENCY_MAP: dict[str, float] = {
    "HP:0040280": 1.00,  # Obligate (100%)
    "HP:0040281": 0.90,  # Very frequent (80-99%)
    "HP:0040282": 0.50,  # Frequent (30-79%)
    "HP:0040283": 0.12,  # Occasional (5-29%)
    "HP:0040284": 0.02,  # Very rare (<4-1%)
    "HP:0040285": 0.00,  # Excluded (0%)
}

FREQUENCY_LABELS: dict[str, str] = {
    "HP:0040280": "obligate",
    "HP:0040281": "very_frequent",
    "HP:0040282": "frequent",
    "HP:0040283": "occasional",
    "HP:0040284": "very_rare",
    "HP:0040285": "excluded",
}


@dataclass
class DiseaseAnnotation:
    """A single disease-phenotype annotation from phenotype.hpoa."""

    database_id: str  # e.g. "OMIM:100300"
    disease_name: str
    hpo_id: str
    evidence_code: str  # e.g. "PCS", "IEA", "TAS"
    frequency_raw: str  # raw frequency string from file
    frequency_value: float  # numeric frequency estimate
    frequency_label: str  # human-readable label
    onset: str  # HPO onset term ID, or empty
    qualifier: str  # "NOT" if negated, else empty
    aspect: str  # P=Phenotype, I=Inheritance, C=Clinical course, M=Clinical modifier


def parse_frequency(raw: str) -> tuple[float, str]:
    """Parse a frequency string into a numeric value and label.

    Handles HPO frequency terms, fractions (e.g. "3/7"), and percentages.
    Returns (value, label) tuple.
    """
    if not raw:
        return 0.50, "unknown"

    raw = raw.strip()

    # HPO frequency term
    if raw in FREQUENCY_MAP:
        return FREQUENCY_MAP[raw], FREQUENCY_LABELS[raw]

    # Fraction like "3/7" or "12/44"
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 2:
            try:
                num, denom = float(parts[0]), float(parts[1])
                if denom > 0:
                    return num / denom, f"{raw}"
            except ValueError:
                pass

    # Percentage like "45%"
    if raw.endswith("%"):
        try:
            return float(raw[:-1]) / 100.0, f"{raw}"
        except ValueError:
            pass

    return 0.50, "unknown"


def parse_hpoa(path: Path, disease_prefix: str = "") -> list[DiseaseAnnotation]:
    """Parse a phenotype.hpoa file into a list of annotations.

    Args:
        path: Path to phenotype.hpoa file.
        disease_prefix: If set, only return annotations for diseases with this
            prefix (e.g. "OMIM" or "ORPHA"). Empty string = all.

    Returns:
        List of DiseaseAnnotation objects.
    """
    annotations: list[DiseaseAnnotation] = []

    with open(path, encoding="utf-8") as f:
        # Skip comment lines
        for line in f:
            if not line.startswith("#"):
                # This is the header line
                header = line.strip().split("\t")
                break
        else:
            return annotations

        reader = csv.DictReader(f, fieldnames=header, delimiter="\t")
        for row in reader:
            db_id = row.get("database_id", row.get("DatabaseID", "")).strip()

            if disease_prefix and not db_id.startswith(disease_prefix + ":"):
                continue

            hpo_id = row.get("hpo_id", row.get("HPO_ID", "")).strip()
            if not hpo_id:
                continue

            freq_raw = row.get("frequency", row.get("Frequency", "")).strip()
            freq_val, freq_label = parse_frequency(freq_raw)

            qualifier = row.get("qualifier", row.get("Qualifier", "")).strip()
            aspect = row.get("aspect", row.get("Aspect", "")).strip()

            annotations.append(
                DiseaseAnnotation(
                    database_id=db_id,
                    disease_name=row.get("disease_name", row.get("DiseaseName", "")).strip(),
                    hpo_id=hpo_id,
                    evidence_code=row.get("evidence", row.get("Evidence", "")).strip(),
                    frequency_raw=freq_raw,
                    frequency_value=freq_val,
                    frequency_label=freq_label,
                    onset=row.get("onset", row.get("Onset", "")).strip(),
                    qualifier=qualifier,
                    aspect=aspect,
                )
            )

    return annotations


def build_disease_hpo_map(
    annotations: list[DiseaseAnnotation],
    aspect_filter: str = "P",
    exclude_negated: bool = True,
) -> dict[str, list[dict]]:
    """Build a mapping from disease IDs to their HPO phenotype annotations.

    Args:
        annotations: Parsed annotations from parse_hpoa().
        aspect_filter: Only include annotations with this aspect.
            "P" = phenotype (default), "" = all aspects.
        exclude_negated: If True, exclude annotations with "NOT" qualifier.

    Returns:
        Dict mapping disease_id to list of annotation dicts with keys:
        hpo_id, frequency, frequency_label, evidence, onset.
    """
    disease_map: dict[str, list[dict]] = defaultdict(list)

    for ann in annotations:
        if aspect_filter and ann.aspect != aspect_filter:
            continue
        if exclude_negated and ann.qualifier.upper() == "NOT":
            continue

        disease_map[ann.database_id].append(
            {
                "hpo_id": ann.hpo_id,
                "frequency": ann.frequency_value,
                "frequency_label": ann.frequency_label,
                "evidence": ann.evidence_code,
                "onset": ann.onset,
            }
        )

    return dict(disease_map)


def get_disease_hpo_terms(
    disease_map: dict[str, list[dict]],
    disease_id: str,
) -> list[str]:
    """Get the set of HPO term IDs annotated to a disease.

    Args:
        disease_map: Output of build_disease_hpo_map().
        disease_id: OMIM or ORPHA disease ID (e.g. "OMIM:100300").

    Returns:
        List of HPO term IDs.
    """
    if disease_id not in disease_map:
        return []
    return [a["hpo_id"] for a in disease_map[disease_id]]


def load_disease_hpo_map(path: Path | None = None) -> dict[str, list[dict]]:
    """Load a previously saved disease-HPO map from JSON."""
    if path is None:
        path = DATA_DIR / "disease_hpo_map.json"
    if not path.exists():
        raise FileNotFoundError(f"Disease-HPO map not found at {path}. Run the mapper first.")
    with open(path) as f:
        return json.load(f)


def parse_and_build(
    hpoa_path: Path,
    output_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """End-to-end: parse HPOA file, build mapping, and save.

    Returns:
        The disease-HPO mapping dict.
    """
    if output_dir is None:
        output_dir = DATA_DIR

    logger.info("Parsing HPOA file: %s", hpoa_path)
    annotations = parse_hpoa(hpoa_path)
    logger.info("Parsed %d annotations", len(annotations))

    disease_map = build_disease_hpo_map(annotations)
    logger.info("Mapped %d diseases to HPO terms", len(disease_map))

    # Stats
    omim_count = sum(1 for d in disease_map if d.startswith("OMIM:"))
    orpha_count = sum(1 for d in disease_map if d.startswith("ORPHA:"))
    logger.info("  OMIM diseases: %d, ORPHA diseases: %d", omim_count, orpha_count)

    avg_terms = sum(len(v) for v in disease_map.values()) / max(len(disease_map), 1)
    logger.info("  Average HPO terms per disease: %.1f", avg_terms)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "disease_hpo_map.json"
    with open(out_path, "w") as f:
        json.dump(disease_map, f, indent=2)
    logger.info("Saved disease-HPO map to %s", out_path)

    return disease_map


def main():
    parser = argparse.ArgumentParser(description="Parse phenotype.hpoa into disease-HPO mapping")
    parser.add_argument(
        "--hpoa",
        type=Path,
        default=DATA_DIR / "phenotype.hpoa",
        help="Path to phenotype.hpoa file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    disease_map = parse_and_build(args.hpoa, args.output_dir)

    print(f"\nDisease-HPO Mapping Summary:")
    print(f"  Total diseases: {len(disease_map)}")
    print(f"  OMIM: {sum(1 for d in disease_map if d.startswith('OMIM:'))}")
    print(f"  ORPHA: {sum(1 for d in disease_map if d.startswith('ORPHA:'))}")
    hpo_counts = [len(v) for v in disease_map.values()]
    if hpo_counts:
        print(f"  HPO terms per disease: min={min(hpo_counts)}, "
              f"max={max(hpo_counts)}, avg={sum(hpo_counts)/len(hpo_counts):.1f}")


if __name__ == "__main__":
    main()
