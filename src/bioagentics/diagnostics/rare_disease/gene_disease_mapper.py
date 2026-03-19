"""Parse OMIM genemap2 file to extract gene-disease associations.

Genemap2 is a tab-delimited file from OMIM that maps genes to their
associated phenotypes (diseases). Each row contains gene information and
a semicolon-separated list of phenotype entries with MIM numbers and
inheritance patterns.

Phenotype entry format (in column 13):
  "Disease Name, MIM_NUMBER (inheritance_key), inheritance_code"
  e.g. "{Cystic fibrosis}, 219700 (3), Autosomal recessive"

Inheritance key meanings:
  (1) = gene with known sequence and phenotype association
  (2) = gene with known sequence, phenotype mapped by linkage
  (3) = molecular basis of phenotype known
  (4) = contiguous gene duplication/deletion syndrome

Output:
    data/diagnostics/rare-disease-phenotype-matcher/gene_disease_map.json

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.gene_disease_mapper [--genemap2 PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "rare-disease-phenotype-matcher"

# Mapping key codes to descriptions
MAPPING_KEY: dict[int, str] = {
    1: "association",
    2: "linkage",
    3: "molecular_basis_known",
    4: "contiguous_gene_syndrome",
}

# Recognized inheritance patterns
INHERITANCE_PATTERNS: set[str] = {
    "Autosomal dominant",
    "Autosomal recessive",
    "X-linked",
    "X-linked dominant",
    "X-linked recessive",
    "Y-linked",
    "Mitochondrial",
    "Digenic dominant",
    "Digenic recessive",
    "Somatic mutation",
    "Somatic mosaicism",
    "Isolated cases",
    "Multifactorial",
}


@dataclass
class GeneDiseaseAssociation:
    """A single gene-disease association from genemap2."""

    gene_symbol: str
    gene_mim: str  # Gene MIM number
    disease_name: str
    disease_mim: str  # Phenotype MIM number (may be empty)
    mapping_key: int  # 1-4
    mapping_description: str
    inheritance_patterns: list[str] = field(default_factory=list)
    is_provisional: bool = False  # True if disease name was in {}


def parse_phenotype_entry(entry: str) -> dict:
    """Parse a single phenotype entry string from genemap2.

    Args:
        entry: A phenotype string like "Cystic fibrosis, 219700 (3), Autosomal recessive"

    Returns:
        Dict with keys: disease_name, disease_mim, mapping_key, inheritance_patterns,
        is_provisional.
    """
    entry = entry.strip()
    if not entry:
        return {}

    # Check for provisional (in braces)
    is_provisional = entry.startswith("{") or entry.startswith("[")

    # Remove leading brackets/braces; closing ones stripped from name below
    clean = entry.lstrip("{[?")

    # Try to find MIM number and mapping key
    # Pattern: "Name, 123456 (3), Inheritance1, Inheritance2"
    mim_match = re.search(r"(\d{6})\s*\((\d)\)", clean)

    disease_mim = ""
    mapping_key = 0
    inheritance_list: list[str] = []

    if mim_match:
        disease_mim = mim_match.group(1)
        mapping_key = int(mim_match.group(2))

        # Disease name is everything before the MIM number
        name_part = clean[: mim_match.start()].rstrip(", ").strip("}]")

        # Inheritance is everything after the (key)
        after_key = clean[mim_match.end() :].strip().lstrip(",").strip()
        if after_key:
            # Split on comma, check each against known patterns
            for part in after_key.split(","):
                part = part.strip()
                if part:
                    inheritance_list.append(part)
    else:
        # No MIM number found — just use the whole thing as the name
        # Try to find standalone mapping key like "(3)" at end
        key_match = re.search(r"\((\d)\)\s*$", clean)
        if key_match:
            mapping_key = int(key_match.group(1))
            name_part = clean[: key_match.start()].rstrip(", ")
        else:
            name_part = clean

    disease_name = name_part.strip("}] ,").strip()

    return {
        "disease_name": disease_name,
        "disease_mim": disease_mim,
        "mapping_key": mapping_key,
        "inheritance_patterns": inheritance_list,
        "is_provisional": is_provisional,
    }


def parse_genemap2(path: Path) -> list[GeneDiseaseAssociation]:
    """Parse OMIM genemap2 tab-delimited file.

    Args:
        path: Path to genemap2.txt file.

    Returns:
        List of GeneDiseaseAssociation objects.
    """
    associations: list[GeneDiseaseAssociation] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            # Skip comment lines
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")

            # genemap2 has at least 13 columns; phenotype info is in column 13 (0-indexed: 12)
            if len(fields) < 13:
                continue

            gene_symbol = fields[8].strip() if len(fields) > 8 else ""
            gene_mim = fields[5].strip() if len(fields) > 5 else ""
            phenotypes_str = fields[12].strip() if len(fields) > 12 else ""

            if not gene_symbol or not phenotypes_str:
                continue

            # Split phenotypes by semicolons
            for pheno_entry in phenotypes_str.split(";"):
                parsed = parse_phenotype_entry(pheno_entry)
                if not parsed or not parsed["disease_name"]:
                    continue

                assoc = GeneDiseaseAssociation(
                    gene_symbol=gene_symbol,
                    gene_mim=gene_mim,
                    disease_name=parsed["disease_name"],
                    disease_mim=parsed["disease_mim"],
                    mapping_key=parsed["mapping_key"],
                    mapping_description=MAPPING_KEY.get(parsed["mapping_key"], "unknown"),
                    inheritance_patterns=parsed["inheritance_patterns"],
                    is_provisional=parsed["is_provisional"],
                )
                associations.append(assoc)

    return associations


def build_gene_disease_map(
    associations: list[GeneDiseaseAssociation],
    min_mapping_key: int = 0,
) -> dict[str, list[dict]]:
    """Build a mapping from gene symbols to their disease associations.

    Args:
        associations: Parsed associations from parse_genemap2().
        min_mapping_key: Minimum mapping key to include. Use 3 to filter
            to molecular-basis-known associations only.

    Returns:
        Dict mapping gene_symbol to list of association dicts.
    """
    gene_map: dict[str, list[dict]] = defaultdict(list)

    for assoc in associations:
        if assoc.mapping_key < min_mapping_key:
            continue

        gene_map[assoc.gene_symbol].append(
            {
                "disease_name": assoc.disease_name,
                "disease_mim": assoc.disease_mim,
                "omim_id": f"OMIM:{assoc.disease_mim}" if assoc.disease_mim else "",
                "mapping_key": assoc.mapping_key,
                "mapping_description": assoc.mapping_description,
                "inheritance_patterns": assoc.inheritance_patterns,
                "is_provisional": assoc.is_provisional,
            }
        )

    return dict(gene_map)


def build_disease_gene_map(
    associations: list[GeneDiseaseAssociation],
    min_mapping_key: int = 0,
) -> dict[str, list[dict]]:
    """Build a mapping from disease OMIM IDs to associated genes.

    Args:
        associations: Parsed associations from parse_genemap2().
        min_mapping_key: Minimum mapping key to include.

    Returns:
        Dict mapping OMIM disease IDs to list of gene info dicts.
    """
    disease_map: dict[str, list[dict]] = defaultdict(list)

    for assoc in associations:
        if assoc.mapping_key < min_mapping_key:
            continue
        if not assoc.disease_mim:
            continue

        omim_id = f"OMIM:{assoc.disease_mim}"
        disease_map[omim_id].append(
            {
                "gene_symbol": assoc.gene_symbol,
                "gene_mim": assoc.gene_mim,
                "mapping_key": assoc.mapping_key,
                "inheritance_patterns": assoc.inheritance_patterns,
            }
        )

    return dict(disease_map)


def parse_and_build(
    genemap2_path: Path,
    output_dir: Path | None = None,
    min_mapping_key: int = 0,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """End-to-end: parse genemap2, build mappings, and save.

    Returns:
        Tuple of (gene_disease_map, disease_gene_map).
    """
    if output_dir is None:
        output_dir = DATA_DIR

    logger.info("Parsing genemap2: %s", genemap2_path)
    associations = parse_genemap2(genemap2_path)
    logger.info("Parsed %d gene-disease associations", len(associations))

    gene_map = build_gene_disease_map(associations, min_mapping_key)
    disease_map = build_disease_gene_map(associations, min_mapping_key)
    logger.info("Gene→disease map: %d genes", len(gene_map))
    logger.info("Disease→gene map: %d diseases", len(disease_map))

    # Stats
    key_counts: dict[int, int] = defaultdict(int)
    for assoc in associations:
        key_counts[assoc.mapping_key] += 1
    for key in sorted(key_counts):
        desc = MAPPING_KEY.get(key, "unknown")
        logger.info("  Mapping key %d (%s): %d", key, desc, key_counts[key])

    inh_counts: dict[str, int] = defaultdict(int)
    for assoc in associations:
        for inh in assoc.inheritance_patterns:
            inh_counts[inh] += 1
    for inh, count in sorted(inh_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info("  Inheritance '%s': %d", inh, count)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    gene_path = output_dir / "gene_disease_map.json"
    with open(gene_path, "w") as f:
        json.dump(gene_map, f, indent=2)
    logger.info("Saved gene→disease map to %s", gene_path)

    disease_path = output_dir / "disease_gene_map.json"
    with open(disease_path, "w") as f:
        json.dump(disease_map, f, indent=2)
    logger.info("Saved disease→gene map to %s", disease_path)

    return gene_map, disease_map


def main():
    parser = argparse.ArgumentParser(description="Parse OMIM genemap2 into gene-disease mapping")
    parser.add_argument(
        "--genemap2",
        type=Path,
        default=DATA_DIR / "genemap2.txt",
        help="Path to genemap2.txt file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
    )
    parser.add_argument(
        "--min-mapping-key",
        type=int,
        default=0,
        help="Minimum mapping key to include (3 = molecular basis known)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gene_map, disease_map = parse_and_build(
        args.genemap2, args.output_dir, args.min_mapping_key,
    )

    print(f"\nGene-Disease Mapping Summary:")
    print(f"  Genes with disease associations: {len(gene_map)}")
    print(f"  Diseases with gene associations: {len(disease_map)}")
    disease_counts = [len(v) for v in gene_map.values()]
    if disease_counts:
        print(
            f"  Diseases per gene: min={min(disease_counts)}, "
            f"max={max(disease_counts)}, avg={sum(disease_counts)/len(disease_counts):.1f}"
        )


if __name__ == "__main__":
    main()
