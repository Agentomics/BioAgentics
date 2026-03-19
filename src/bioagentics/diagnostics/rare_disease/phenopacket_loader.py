"""Loader for GA4GH Phenopacket Store files.

Parses phenopacket JSON files into BenchmarkCase objects for evaluation.
Supports streaming through large directories to respect memory constraints.

Phenopacket Store format (v0.1.26):
- phenotypicFeatures[].type.id → HPO term IDs
- phenotypicFeatures[].excluded → skip if True
- interpretations[0].diagnosis.disease.id → true diagnosis (preferred)
- diseases[0].term.id → fallback diagnosis source

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.phenopacket_loader
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from bioagentics.config import DATA_DIR
from bioagentics.diagnostics.rare_disease.evaluation import BenchmarkCase

logger = logging.getLogger(__name__)

PHENOPACKET_DIR = (
    DATA_DIR
    / "diagnostics"
    / "rare-disease-phenotype-matcher"
    / "phenopacket_store"
    / "0.1.26"
)


@dataclass
class PhenopacketSummary:
    """Summary statistics from loading phenopackets."""

    total_files: int = 0
    loaded: int = 0
    skipped_no_diagnosis: int = 0
    skipped_no_phenotypes: int = 0
    skipped_parse_error: int = 0
    unique_diseases: int = 0


def extract_hpo_terms(phenopacket: dict) -> list[str]:
    """Extract observed (non-excluded) HPO term IDs from a phenopacket.

    Args:
        phenopacket: Parsed phenopacket JSON dict.

    Returns:
        List of HPO term IDs (e.g. ["HP:0001250", "HP:0000252"]).
    """
    terms = []
    for feature in phenopacket.get("phenotypicFeatures", []):
        if feature.get("excluded", False):
            continue
        term_id = feature.get("type", {}).get("id", "")
        if term_id.startswith("HP:"):
            terms.append(term_id)
    return terms


def extract_disease_id(phenopacket: dict) -> str | None:
    """Extract the true disease ID from a phenopacket.

    Prefers interpretations[].diagnosis.disease.id (solved cases),
    falls back to diseases[].term.id.

    Args:
        phenopacket: Parsed phenopacket JSON dict.

    Returns:
        Disease ID string (e.g. "OMIM:620371") or None if not found.
    """
    # Try interpretations first (solved cases with diagnosis)
    for interp in phenopacket.get("interpretations", []):
        diagnosis = interp.get("diagnosis", {})
        disease = diagnosis.get("disease", {})
        disease_id = disease.get("id", "")
        if disease_id:
            return disease_id

    # Fallback to diseases list
    for disease in phenopacket.get("diseases", []):
        term = disease.get("term", {})
        disease_id = term.get("id", "")
        if disease_id:
            return disease_id

    return None


def load_phenopacket(path: Path) -> BenchmarkCase | None:
    """Load a single phenopacket file into a BenchmarkCase.

    Args:
        path: Path to a phenopacket JSON file.

    Returns:
        BenchmarkCase or None if the file can't be parsed or lacks
        required fields (diagnosis + phenotype terms).
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to parse %s: %s", path.name, e)
        return None

    disease_id = extract_disease_id(data)
    if not disease_id:
        return None

    hpo_terms = extract_hpo_terms(data)
    if not hpo_terms:
        return None

    case_id = data.get("id", path.stem)

    return BenchmarkCase(
        case_id=case_id,
        query_hpo_terms=hpo_terms,
        true_disease_id=disease_id,
        metadata={
            "source": "phenopacket_store",
            "file": path.name,
            "gene_dir": path.parent.name,
        },
    )


def iter_phenopackets(
    base_dir: Path | None = None,
    min_hpo_terms: int = 1,
) -> Iterator[BenchmarkCase]:
    """Iterate over phenopacket files, yielding BenchmarkCase objects.

    Streams through files one at a time to minimize memory usage.

    Args:
        base_dir: Root directory of phenopacket store. Defaults to
            PHENOPACKET_DIR.
        min_hpo_terms: Minimum observed HPO terms required.

    Yields:
        BenchmarkCase for each valid phenopacket.
    """
    if base_dir is None:
        base_dir = PHENOPACKET_DIR

    if not base_dir.exists():
        logger.warning("Phenopacket directory not found: %s", base_dir)
        return

    for json_path in sorted(base_dir.rglob("*.json")):
        case = load_phenopacket(json_path)
        if case is not None and len(case.query_hpo_terms) >= min_hpo_terms:
            yield case


def load_all_phenopackets(
    base_dir: Path | None = None,
    min_hpo_terms: int = 1,
) -> tuple[list[BenchmarkCase], PhenopacketSummary]:
    """Load all phenopackets from the store directory.

    Args:
        base_dir: Root directory of phenopacket store.
        min_hpo_terms: Minimum observed HPO terms required.

    Returns:
        Tuple of (list of BenchmarkCase, PhenopacketSummary).
    """
    if base_dir is None:
        base_dir = PHENOPACKET_DIR

    summary = PhenopacketSummary()
    cases: list[BenchmarkCase] = []
    diseases_seen: set[str] = set()

    if not base_dir.exists():
        logger.warning("Phenopacket directory not found: %s", base_dir)
        return cases, summary

    json_files = sorted(base_dir.rglob("*.json"))
    summary.total_files = len(json_files)

    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            summary.skipped_parse_error += 1
            continue

        disease_id = extract_disease_id(data)
        if not disease_id:
            summary.skipped_no_diagnosis += 1
            continue

        hpo_terms = extract_hpo_terms(data)
        if len(hpo_terms) < min_hpo_terms:
            summary.skipped_no_phenotypes += 1
            continue

        case_id = data.get("id", json_path.stem)
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                query_hpo_terms=hpo_terms,
                true_disease_id=disease_id,
                metadata={
                    "source": "phenopacket_store",
                    "file": json_path.name,
                    "gene_dir": json_path.parent.name,
                },
            )
        )
        diseases_seen.add(disease_id)
        summary.loaded += 1

    summary.unique_diseases = len(diseases_seen)

    logger.info(
        "Loaded %d phenopackets (%d diseases) from %d files. "
        "Skipped: %d no diagnosis, %d no phenotypes, %d parse errors",
        summary.loaded,
        summary.unique_diseases,
        summary.total_files,
        summary.skipped_no_diagnosis,
        summary.skipped_no_phenotypes,
        summary.skipped_parse_error,
    )

    return cases, summary
