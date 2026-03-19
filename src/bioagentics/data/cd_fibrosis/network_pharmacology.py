"""Network pharmacology validation for CD fibrosis drug repurposing candidates.

For top CMAP/iLINCS hits, validates compound targets against fibrosis-relevant
protein interaction networks and known fibrosis pathways.

Validation layers:
1. Map compound targets to fibrosis PPI network (STRING database)
2. Assess target overlap with known fibrosis pathways
3. Cross-reference with anti-fibrotic activity in other organs
4. Score for druggability (safety, approval status)

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.network_pharmacology \
        --hits path/to/cmap_hits.tsv --drugbank path/to/drugbank_targets.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing"

STRING_API_BASE = "https://string-db.org/api"
STRING_SPECIES = 9606  # Homo sapiens
STRING_TIMEOUT = 30

# ── Known fibrosis pathway gene sets ──

FIBROSIS_PATHWAYS: dict[str, set[str]] = {
    "TGF-beta": {
        "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2",
        "SMAD2", "SMAD3", "SMAD4", "SMAD7",
        "SNAI1", "SNAI2", "CTGF", "SERPINE1",
        "COL1A1", "COL1A2", "COL3A1", "ACTA2", "FN1",
    },
    "Wnt": {
        "WNT1", "WNT2", "WNT3A", "WNT5A", "WNT7A",
        "FZD1", "FZD2", "FZD4", "FZD7",
        "CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3B",
        "LEF1", "TCF7L2", "MYC", "CCND1",
    },
    "YAP/TAZ": {
        "YAP1", "WWTR1", "LATS1", "LATS2", "MST1", "MST2",
        "TEAD1", "TEAD2", "TEAD3", "TEAD4",
        "CYR61", "CTGF", "ANKRD1",
        "RHOA", "ROCK1", "ROCK2",
    },
    "TL1A-DR3": {
        "TNFSF15", "TNFRSF25", "TNFRSF6B",
        "NFKB1", "RELA", "RELB",
        "TRAF2", "TRAF5", "RIPK1",
        "MAP3K7", "MAPK8", "MAPK14",
        "RHOA", "RHOB", "RAC1", "CDC42",
        "ROCK1", "ROCK2",
    },
    "JAK-STAT": {
        "JAK1", "JAK2", "JAK3", "TYK2",
        "STAT1", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6",
        "SOCS1", "SOCS3",
        "IL6", "IL6R", "IL6ST",
    },
    "ECM_remodeling": {
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1",
        "FN1", "TNC", "POSTN", "CTHRC1",
        "MMP2", "MMP9", "MMP14",
        "TIMP1", "TIMP2", "TIMP3",
        "LOX", "LOXL2",
        "SERPINE1", "GREM1",
    },
    "epigenetic": {
        "HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6",
        "KDM6A", "KDM6B", "EZH2",
        "DNMT1", "DNMT3A", "DNMT3B",
        "BRD4", "BRD2",
    },
}

# Genes associated with anti-fibrotic activity across organs
CROSS_ORGAN_FIBROSIS_GENES: dict[str, set[str]] = {
    "IPF_lung": {
        "TGFB1", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
        "COL1A1", "COL3A1", "ACTA2", "FN1",
        "CTGF", "SERPINE1", "LOX", "LOXL2",
        "PDGFRA", "PDGFRB", "FGFR1", "FGFR2",
        "MMP7", "MUC5B", "SFTPC",
    },
    "liver_fibrosis": {
        "TGFB1", "TGFBR1", "SMAD2", "SMAD3",
        "COL1A1", "COL3A1", "ACTA2", "TIMP1",
        "PDGFRA", "PDGFRB", "VEGFA",
        "CTGF", "LOX", "LOXL2",
        "SERPINE1", "NOTCH1",
    },
    "kidney_fibrosis": {
        "TGFB1", "TGFBR1", "SMAD2", "SMAD3",
        "COL1A1", "COL4A1", "FN1", "ACTA2",
        "CTGF", "SERPINE1", "BMP7",
        "WNT1", "CTNNB1",
        "SNAI1", "SNAI2", "TWIST1",
    },
}

# All fibrosis-relevant genes (union of all pathways)
ALL_FIBROSIS_GENES: set[str] = set()
for _genes in FIBROSIS_PATHWAYS.values():
    ALL_FIBROSIS_GENES |= _genes
for _genes in CROSS_ORGAN_FIBROSIS_GENES.values():
    ALL_FIBROSIS_GENES |= _genes


class StringClient:
    """Client for the STRING database REST API."""

    def __init__(self, species: int = STRING_SPECIES):
        self.species = species
        self.session = requests.Session()

    def get_interactions(
        self,
        proteins: list[str],
        required_score: int = 400,
    ) -> list[dict]:
        """Get protein-protein interactions from STRING.

        Args:
            proteins: Gene symbols to query.
            required_score: Minimum combined score (0-1000, default 400 = medium).

        Returns:
            List of interaction dicts with preferredName_A/B and score.
        """
        resp = self.session.post(
            f"{STRING_API_BASE}/json/network",
            data={
                "identifiers": "\r".join(proteins),
                "species": self.species,
                "required_score": required_score,
                "caller_identity": "bioagentics_cd_fibrosis",
            },
            timeout=STRING_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def get_enrichment(
        self,
        proteins: list[str],
    ) -> list[dict]:
        """Get functional enrichment for a set of proteins.

        Returns enriched GO terms, KEGG pathways, etc.
        """
        resp = self.session.post(
            f"{STRING_API_BASE}/json/enrichment",
            data={
                "identifiers": "\r".join(proteins),
                "species": self.species,
                "caller_identity": "bioagentics_cd_fibrosis",
            },
            timeout=STRING_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def get_interaction_partners(
        self,
        proteins: list[str],
        limit: int = 10,
    ) -> list[dict]:
        """Get interaction partners for proteins (expand the network).

        Returns partners not in the input set.
        """
        resp = self.session.post(
            f"{STRING_API_BASE}/json/interaction_partners",
            data={
                "identifiers": "\r".join(proteins),
                "species": self.species,
                "limit": limit,
                "caller_identity": "bioagentics_cd_fibrosis",
            },
            timeout=STRING_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()


def compute_pathway_overlap(
    target_genes: set[str],
    pathways: dict[str, set[str]] | None = None,
) -> dict[str, dict]:
    """Compute overlap between compound targets and fibrosis pathways.

    Args:
        target_genes: Set of gene symbols targeted by the compound.
        pathways: Pathway gene sets (default: FIBROSIS_PATHWAYS).

    Returns:
        Dict mapping pathway name to overlap stats.
    """
    pathways = pathways or FIBROSIS_PATHWAYS
    results = {}

    for pathway_name, pathway_genes in pathways.items():
        overlap = target_genes & pathway_genes
        results[pathway_name] = {
            "n_targets_in_pathway": len(overlap),
            "pathway_size": len(pathway_genes),
            "overlap_fraction": (
                len(overlap) / len(pathway_genes) if pathway_genes else 0
            ),
            "overlapping_genes": sorted(overlap),
        }

    return results


def compute_cross_organ_overlap(
    target_genes: set[str],
) -> dict[str, dict]:
    """Check if compound targets overlap with fibrosis genes in other organs.

    Shared targets across organs suggest conserved anti-fibrotic mechanisms.
    """
    return compute_pathway_overlap(target_genes, CROSS_ORGAN_FIBROSIS_GENES)


def score_druggability(drug_info: dict | None) -> dict:
    """Score a compound's druggability from DrugBank data.

    Assigns a numeric score (0-5) based on approval status.

    Args:
        drug_info: Drug record from DrugBank (output of drugbank.load_drug_targets).
            If None, compound is not in DrugBank.

    Returns:
        Dict with druggability score and details.
    """
    if drug_info is None:
        return {
            "druggability_score": 0,
            "approval_status": "unknown",
            "in_drugbank": False,
            "n_known_targets": 0,
        }

    groups = drug_info.get("groups", "").lower()

    if "approved" in groups:
        score = 5
        status = "approved"
    elif "investigational" in groups and "phase 3" in drug_info.get("indication", "").lower():
        score = 4
        status = "phase_3"
    elif "investigational" in groups:
        score = 3
        status = "investigational"
    elif "experimental" in groups:
        score = 2
        status = "experimental"
    else:
        score = 1
        status = "other"

    n_targets = int(drug_info.get("n_targets", 0))

    return {
        "druggability_score": score,
        "approval_status": status,
        "in_drugbank": True,
        "n_known_targets": n_targets,
    }


def compute_network_score(
    pathway_overlap: dict[str, dict],
    cross_organ: dict[str, dict],
    druggability: dict,
) -> float:
    """Compute composite network pharmacology score.

    Components (weighted):
    - Pathway coverage: fraction of fibrosis pathways hit (0-1) * 0.35
    - Best pathway overlap: max overlap fraction (0-1) * 0.25
    - Cross-organ signal: fraction of organs with overlapping targets (0-1) * 0.20
    - Druggability: normalized score (0-1) * 0.20

    Returns:
        Composite score (0-1), higher = better validated candidate.
    """
    # Pathway coverage: how many pathways have at least 1 target
    n_pathways_hit = sum(
        1 for v in pathway_overlap.values()
        if v["n_targets_in_pathway"] > 0
    )
    pathway_coverage = n_pathways_hit / len(pathway_overlap) if pathway_overlap else 0

    # Best pathway overlap
    best_overlap = max(
        (v["overlap_fraction"] for v in pathway_overlap.values()),
        default=0,
    )

    # Cross-organ signal
    n_organs_hit = sum(
        1 for v in cross_organ.values()
        if v["n_targets_in_pathway"] > 0
    )
    organ_coverage = n_organs_hit / len(cross_organ) if cross_organ else 0

    # Druggability (0-5 -> 0-1)
    drug_score = druggability.get("druggability_score", 0) / 5.0

    composite = (
        0.35 * pathway_coverage
        + 0.25 * best_overlap
        + 0.20 * organ_coverage
        + 0.20 * drug_score
    )

    return round(composite, 4)


def validate_compound(
    compound_name: str,
    target_genes: set[str],
    drug_info: dict | None = None,
) -> dict:
    """Run full network pharmacology validation for a single compound.

    Args:
        compound_name: Name of the compound.
        target_genes: Gene symbols targeted by the compound.
        drug_info: Optional DrugBank record.

    Returns:
        Validation result dict with all scores and details.
    """
    pathway_overlap = compute_pathway_overlap(target_genes)
    cross_organ = compute_cross_organ_overlap(target_genes)
    druggability = score_druggability(drug_info)
    network_score = compute_network_score(pathway_overlap, cross_organ, druggability)

    # Summarize pathway hits
    pathways_hit = [
        name for name, v in pathway_overlap.items()
        if v["n_targets_in_pathway"] > 0
    ]
    all_overlapping = set()
    for v in pathway_overlap.values():
        all_overlapping.update(v["overlapping_genes"])

    organs_hit = [
        name for name, v in cross_organ.items()
        if v["n_targets_in_pathway"] > 0
    ]

    return {
        "compound": compound_name,
        "n_targets": len(target_genes),
        "n_fibrosis_targets": len(target_genes & ALL_FIBROSIS_GENES),
        "fibrosis_target_fraction": (
            round(len(target_genes & ALL_FIBROSIS_GENES) / len(target_genes), 4)
            if target_genes else 0
        ),
        "n_pathways_hit": len(pathways_hit),
        "pathways_hit": ";".join(pathways_hit),
        "fibrosis_genes_targeted": ";".join(sorted(all_overlapping)),
        "n_organs_with_overlap": len(organs_hit),
        "organs_hit": ";".join(organs_hit),
        "druggability_score": druggability["druggability_score"],
        "approval_status": druggability["approval_status"],
        "network_score": network_score,
        "pathway_details": pathway_overlap,
        "cross_organ_details": cross_organ,
    }


def validate_candidates(
    ranked_hits: pd.DataFrame,
    drug_targets: dict[str, dict] | None = None,
    top_n: int = 50,
    compound_column: str = "compound",
) -> pd.DataFrame:
    """Validate top CMAP/iLINCS candidates with network pharmacology.

    Args:
        ranked_hits: Ranked compounds from CMAP/iLINCS pipeline.
        drug_targets: DrugBank data keyed by lowercase compound name.
        top_n: Number of top compounds to validate.
        compound_column: Column name for compound names.

    Returns:
        DataFrame with network validation scores for each compound.
    """
    drug_targets = drug_targets or {}
    candidates = ranked_hits.head(top_n)

    records = []
    for _, row in candidates.iterrows():
        compound = str(row.get(compound_column, ""))
        compound_lower = compound.lower().strip()

        # Look up targets from DrugBank
        drug_info = drug_targets.get(compound_lower)
        if drug_info and drug_info.get("target_genes"):
            targets = set(drug_info["target_genes"].upper().split(";"))
        else:
            targets = set()

        result = validate_compound(compound, targets, drug_info)

        # Carry over CMAP/iLINCS scores
        result["mean_concordance"] = row.get("mean_concordance")
        result["n_signatures_queried"] = row.get("n_signatures_queried")
        result["convergent_anti_fibrotic"] = row.get("convergent_anti_fibrotic")

        # Drop detailed nested dicts for the flat DataFrame
        result.pop("pathway_details", None)
        result.pop("cross_organ_details", None)

        records.append(result)

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("network_score", ascending=False)

    return df


def generate_network_report(
    ranked_hits: pd.DataFrame,
    drug_targets: dict[str, dict] | None = None,
    output_dir: Path | None = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """Run network pharmacology validation and generate report.

    Args:
        ranked_hits: Ranked compounds from CMAP/iLINCS pipeline.
        drug_targets: DrugBank data keyed by lowercase compound name.
        output_dir: Output directory.
        top_n: Number of top compounds to validate.

    Returns:
        Validated DataFrame sorted by network score.
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Network Pharmacology Validation")
    print("=" * 60)

    validated = validate_candidates(ranked_hits, drug_targets, top_n)

    if len(validated) == 0:
        print("  No compounds to validate.")
        return validated

    # Summary stats
    with_targets = validated[validated["n_targets"] > 0]
    with_fibrosis = validated[validated["n_fibrosis_targets"] > 0]
    approved = validated[validated["approval_status"] == "approved"]

    print(f"\n  Compounds analyzed: {len(validated)}")
    print(f"  With known targets (DrugBank): {len(with_targets)}")
    print(f"  Targeting fibrosis genes: {len(with_fibrosis)}")
    print(f"  Approved drugs: {len(approved)}")

    # Pathway coverage summary
    print(f"\n  Pathway Coverage:")
    for pathway in FIBROSIS_PATHWAYS:
        n_hitting = sum(
            1 for _, row in validated.iterrows()
            if pathway in str(row.get("pathways_hit", ""))
        )
        print(f"    {pathway:20s} {n_hitting:>3d}/{len(validated)} compounds")

    # Top candidates by network score
    print(f"\n  Top 20 by Network Score:")
    print(f"  {'Compound':30s} {'NetScore':>9s} {'Targets':>8s} {'Fib':>4s} "
          f"{'Pathways':>8s} {'Status':>12s}")
    print(f"  {'-'*30} {'-'*9} {'-'*8} {'-'*4} {'-'*8} {'-'*12}")

    for _, row in validated.head(20).iterrows():
        print(f"  {str(row['compound']):30s} "
              f"{row['network_score']:>9.4f} "
              f"{row['n_targets']:>8d} "
              f"{row['n_fibrosis_targets']:>4d} "
              f"{row['n_pathways_hit']:>8d} "
              f"{str(row['approval_status']):>12s}")

    # Save
    out_path = output_dir / "network_validation.tsv"
    validated.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    return validated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Network pharmacology validation for CD fibrosis candidates"
    )
    parser.add_argument(
        "--hits",
        type=Path,
        required=True,
        help="Path to ranked CMAP hits TSV",
    )
    parser.add_argument(
        "--drugbank",
        type=Path,
        help="Path to DrugBank targets TSV (optional)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top compounds to validate (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args(argv)

    if not args.hits.exists():
        print(f"Hits file not found: {args.hits}")
        return

    ranked = pd.read_csv(args.hits, sep="\t")
    print(f"Loaded {len(ranked)} compounds from {args.hits}")

    drug_targets = None
    if args.drugbank and args.drugbank.exists():
        from bioagentics.data.cd_fibrosis.drugbank import load_drug_targets
        drug_targets = load_drug_targets(args.drugbank)
        print(f"Loaded {len(drug_targets)} DrugBank records")

    generate_network_report(
        ranked,
        drug_targets=drug_targets,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
