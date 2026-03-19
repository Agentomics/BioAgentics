"""Parse Orphanet XML export for disease-phenotype associations with frequency detail.

Orphanet provides disease-phenotype associations with frequency qualifiers:
  Obligate (100%), Very frequent (80-99%), Frequent (30-79%),
  Occasional (5-29%), Very rare (<5%), Excluded (0%).

This module parses the Orphanet product XML (en_product4.xml) and produces
a mapping: {orphanet_id: [(hpo_id, frequency_category, frequency_value), ...]}.

It also provides cross-referencing to OMIM IDs where available (from en_product1.xml
or an inline ExternalReference list in product4).

Output:
    data/diagnostics/rare-disease-phenotype-matcher/orphanet_disease_hpo.json

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.orphanet_parser [--xml PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree.ElementTree import iterparse

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "rare-disease-phenotype-matcher"

# Orphanet frequency categories mapped to numeric midpoint estimates
ORPHANET_FREQUENCY: dict[str, float] = {
    "Obligate (100%)": 1.00,
    "Very frequent (99-80%)": 0.90,
    "Frequent (79-30%)": 0.50,
    "Occasional (29-5%)": 0.12,
    "Very rare (<4-1%)": 0.02,
    "Excluded (0%)": 0.00,
}

# Normalized category labels
FREQUENCY_CATEGORY: dict[str, str] = {
    "Obligate (100%)": "obligate",
    "Very frequent (99-80%)": "very_frequent",
    "Frequent (79-30%)": "frequent",
    "Occasional (29-5%)": "occasional",
    "Very rare (<4-1%)": "very_rare",
    "Excluded (0%)": "excluded",
}


@dataclass
class OrphanetAnnotation:
    """A single Orphanet disease-phenotype association."""

    orphanet_id: str  # e.g. "ORPHA:558"
    disease_name: str
    hpo_id: str  # e.g. "HP:0000256"
    hpo_term_name: str
    frequency_category: str  # normalized label
    frequency_value: float  # numeric midpoint


@dataclass
class OrphanetDisease:
    """An Orphanet disease with its phenotype annotations and cross-references."""

    orphanet_id: str
    disease_name: str
    annotations: list[OrphanetAnnotation] = field(default_factory=list)
    omim_ids: list[str] = field(default_factory=list)  # Cross-referenced OMIM IDs


def _normalize_frequency(raw: str) -> tuple[str, float]:
    """Normalize an Orphanet frequency string to (category_label, numeric_value).

    Falls back to 'unknown' / 0.50 if the string is not recognized.
    """
    raw = raw.strip()
    if not raw:
        return "unknown", 0.50
    if raw in FREQUENCY_CATEGORY:
        return FREQUENCY_CATEGORY[raw], ORPHANET_FREQUENCY[raw]
    # Try partial match (some versions omit parenthetical)
    raw_lower = raw.lower()
    for key, label in FREQUENCY_CATEGORY.items():
        if label in raw_lower or key.lower().startswith(raw_lower):
            return label, ORPHANET_FREQUENCY[key]
    return "unknown", 0.50


def parse_orphanet_xml(path: Path) -> list[OrphanetDisease]:
    """Parse Orphanet product4 XML (disease-HPO associations) using streaming.

    Uses iterparse to avoid loading the entire XML tree into memory.

    Args:
        path: Path to en_product4.xml file.

    Returns:
        List of OrphanetDisease objects with annotations.
    """
    diseases: list[OrphanetDisease] = []

    # Track context for streaming parse
    current_disease: OrphanetDisease | None = None
    in_disorder = False
    in_hpo_assoc = False
    in_hpo_term = False
    in_frequency = False
    in_external_ref = False

    # Temp state for current annotation
    current_hpo_id = ""
    current_hpo_name = ""
    current_freq = ""

    for event, elem in iterparse(path, events=("start", "end")):
        tag = elem.tag

        if event == "start":
            if tag == "Disorder":
                in_disorder = True
                current_disease = OrphanetDisease(orphanet_id="", disease_name="")
            elif tag == "HPODisorderAssociation" and in_disorder:
                in_hpo_assoc = True
                current_hpo_id = ""
                current_hpo_name = ""
                current_freq = ""
            elif tag == "HPO" and in_hpo_assoc:
                in_hpo_term = True
            elif tag == "HPOFrequency" and in_hpo_assoc:
                in_frequency = True
            elif tag == "ExternalReference" and in_disorder and not in_hpo_assoc:
                in_external_ref = True

        elif event == "end":
            if tag == "Disorder" and in_disorder:
                if current_disease and current_disease.orphanet_id:
                    diseases.append(current_disease)
                current_disease = None
                in_disorder = False
                # Free memory from parsed elements
                elem.clear()

            elif tag == "OrphaCode" and in_disorder and current_disease and not in_hpo_assoc:
                if elem.text:
                    current_disease.orphanet_id = f"ORPHA:{elem.text.strip()}"

            elif tag == "Name":
                if in_hpo_term and in_hpo_assoc:
                    current_hpo_name = (elem.text or "").strip()
                elif in_frequency and in_hpo_assoc:
                    current_freq = (elem.text or "").strip()
                elif in_disorder and current_disease and not in_hpo_assoc and not in_external_ref:
                    if not current_disease.disease_name:
                        current_disease.disease_name = (elem.text or "").strip()

            elif tag == "HPOId" and in_hpo_term:
                current_hpo_id = (elem.text or "").strip()

            elif tag == "HPO" and in_hpo_assoc:
                in_hpo_term = False

            elif tag == "HPOFrequency" and in_hpo_assoc:
                in_frequency = False

            elif tag == "HPODisorderAssociation" and in_hpo_assoc:
                in_hpo_assoc = False
                if current_disease and current_hpo_id:
                    cat, val = _normalize_frequency(current_freq)
                    current_disease.annotations.append(
                        OrphanetAnnotation(
                            orphanet_id=current_disease.orphanet_id,
                            disease_name=current_disease.disease_name,
                            hpo_id=current_hpo_id,
                            hpo_term_name=current_hpo_name,
                            frequency_category=cat,
                            frequency_value=val,
                        )
                    )

            elif tag == "Source" and in_external_ref and current_disease:
                source = (elem.text or "").strip()
                if source == "OMIM":
                    pass  # Will capture reference in next tag
            elif tag == "Reference" and in_external_ref and current_disease:
                ref = (elem.text or "").strip()
                if ref:
                    # Store reference; we determine it's OMIM by context
                    current_disease.omim_ids.append(f"OMIM:{ref}")
            elif tag == "ExternalReference" and in_external_ref:
                in_external_ref = False

    return diseases


def parse_orphanet_crossref_xml(path: Path) -> dict[str, list[str]]:
    """Parse Orphanet product1 XML for ORPHA→OMIM cross-references.

    This is an alternative to inline ExternalReference parsing if a separate
    cross-reference file is available.

    Args:
        path: Path to en_product1.xml file.

    Returns:
        Dict mapping ORPHA IDs to lists of OMIM IDs.
    """
    crossref: dict[str, list[str]] = defaultdict(list)
    current_orpha = ""
    in_disorder = False
    in_external_ref = False
    current_source = ""

    for event, elem in iterparse(path, events=("start", "end")):
        tag = elem.tag

        if event == "start":
            if tag == "Disorder":
                in_disorder = True
                current_orpha = ""
            elif tag == "ExternalReference" and in_disorder:
                in_external_ref = True
                current_source = ""

        elif event == "end":
            if tag == "Disorder" and in_disorder:
                in_disorder = False
                elem.clear()
            elif tag == "OrphaCode" and in_disorder and not in_external_ref:
                if elem.text:
                    current_orpha = f"ORPHA:{elem.text.strip()}"
            elif tag == "Source" and in_external_ref:
                current_source = (elem.text or "").strip()
            elif tag == "Reference" and in_external_ref:
                ref = (elem.text or "").strip()
                if current_source == "OMIM" and ref and current_orpha:
                    crossref[current_orpha].append(f"OMIM:{ref}")
            elif tag == "ExternalReference":
                in_external_ref = False

    return dict(crossref)


def build_orphanet_hpo_map(
    diseases: list[OrphanetDisease],
    exclude_excluded: bool = True,
) -> dict[str, list[dict]]:
    """Build a mapping from Orphanet disease IDs to HPO annotations.

    Args:
        diseases: Parsed Orphanet diseases from parse_orphanet_xml().
        exclude_excluded: If True, omit annotations with frequency "excluded".

    Returns:
        Dict mapping orphanet_id to list of annotation dicts with keys:
        hpo_id, hpo_term_name, frequency_category, frequency_value.
    """
    disease_map: dict[str, list[dict]] = {}

    for disease in diseases:
        annotations = []
        for ann in disease.annotations:
            if exclude_excluded and ann.frequency_category == "excluded":
                continue
            annotations.append(
                {
                    "hpo_id": ann.hpo_id,
                    "hpo_term_name": ann.hpo_term_name,
                    "frequency_category": ann.frequency_category,
                    "frequency_value": ann.frequency_value,
                }
            )
        if annotations:
            disease_map[disease.orphanet_id] = annotations

    return disease_map


def build_omim_crossref(diseases: list[OrphanetDisease]) -> dict[str, list[str]]:
    """Extract ORPHA→OMIM cross-references from parsed diseases.

    Returns:
        Dict mapping ORPHA IDs to lists of OMIM IDs.
    """
    crossref: dict[str, list[str]] = {}
    for disease in diseases:
        if disease.omim_ids:
            crossref[disease.orphanet_id] = disease.omim_ids
    return crossref


def get_frequency_weights(
    disease_map: dict[str, list[dict]],
    orphanet_id: str,
) -> dict[str, float]:
    """Get a mapping from HPO term ID to frequency weight for a disease.

    Args:
        disease_map: Output of build_orphanet_hpo_map().
        orphanet_id: Orphanet disease ID (e.g. "ORPHA:558").

    Returns:
        Dict mapping HPO term IDs to their frequency values.
    """
    if orphanet_id not in disease_map:
        return {}
    return {a["hpo_id"]: a["frequency_value"] for a in disease_map[orphanet_id]}


def parse_and_build(
    xml_path: Path,
    output_dir: Path | None = None,
    crossref_path: Path | None = None,
) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """End-to-end: parse Orphanet XML, build mappings, and save.

    Args:
        xml_path: Path to en_product4.xml.
        output_dir: Directory to save outputs. Defaults to DATA_DIR.
        crossref_path: Optional path to en_product1.xml for extra cross-refs.

    Returns:
        Tuple of (disease_hpo_map, omim_crossref).
    """
    if output_dir is None:
        output_dir = DATA_DIR

    logger.info("Parsing Orphanet XML: %s", xml_path)
    diseases = parse_orphanet_xml(xml_path)
    logger.info("Parsed %d diseases", len(diseases))

    disease_map = build_orphanet_hpo_map(diseases)
    logger.info("Mapped %d diseases to HPO terms", len(disease_map))

    crossref = build_omim_crossref(diseases)
    logger.info("Found %d diseases with OMIM cross-references", len(crossref))

    # Merge extra cross-references if available
    if crossref_path and crossref_path.exists():
        logger.info("Parsing cross-reference file: %s", crossref_path)
        extra = parse_orphanet_crossref_xml(crossref_path)
        for orpha_id, omim_ids in extra.items():
            if orpha_id not in crossref:
                crossref[orpha_id] = omim_ids
            else:
                existing = set(crossref[orpha_id])
                crossref[orpha_id].extend(i for i in omim_ids if i not in existing)
        logger.info("After merging: %d diseases with cross-references", len(crossref))

    # Stats
    total_ann = sum(len(v) for v in disease_map.values())
    avg_ann = total_ann / max(len(disease_map), 1)
    logger.info("  Total annotations: %d, avg per disease: %.1f", total_ann, avg_ann)

    freq_counts: dict[str, int] = defaultdict(int)
    for anns in disease_map.values():
        for a in anns:
            freq_counts[a["frequency_category"]] += 1
    for cat, count in sorted(freq_counts.items()):
        logger.info("  Frequency '%s': %d annotations", cat, count)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    hpo_path = output_dir / "orphanet_disease_hpo.json"
    with open(hpo_path, "w") as f:
        json.dump(disease_map, f, indent=2)
    logger.info("Saved disease-HPO map to %s", hpo_path)

    xref_path = output_dir / "orphanet_omim_crossref.json"
    with open(xref_path, "w") as f:
        json.dump(crossref, f, indent=2)
    logger.info("Saved OMIM cross-references to %s", xref_path)

    return disease_map, crossref


def main():
    parser = argparse.ArgumentParser(
        description="Parse Orphanet XML into disease-HPO mapping with frequency detail"
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=DATA_DIR / "en_product4.xml",
        help="Path to Orphanet en_product4.xml file",
    )
    parser.add_argument(
        "--crossref-xml",
        type=Path,
        default=None,
        help="Optional path to en_product1.xml for OMIM cross-references",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    disease_map, crossref = parse_and_build(args.xml, args.output_dir, args.crossref_xml)

    print(f"\nOrphanet Disease-HPO Mapping Summary:")
    print(f"  Total diseases with phenotypes: {len(disease_map)}")
    total_ann = sum(len(v) for v in disease_map.values())
    ann_counts = [len(v) for v in disease_map.values()]
    print(f"  Total annotations: {total_ann}")
    if ann_counts:
        print(
            f"  Annotations per disease: min={min(ann_counts)}, "
            f"max={max(ann_counts)}, avg={sum(ann_counts)/len(ann_counts):.1f}"
        )
    print(f"  Diseases with OMIM cross-refs: {len(crossref)}")


if __name__ == "__main__":
    main()
