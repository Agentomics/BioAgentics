"""Candidate prioritization and composite scoring for CD fibrosis drug repurposing.

Ranks drug candidates by a weighted composite score integrating:
- CMAP/iLINCS reversal strength across fibrosis signatures
- Network pharmacology validation (pathway coverage, cross-organ, druggability)
- Safety profile (approval status, clinical stage)
- Mechanistic novelty (novel targets vs known pathway agents)
- Pharmacokinetics (oral bioavailability, route of administration)

Output: final ranked list of 10-20 candidates with composite scores and
mechanistic classification by pathway (TGFbeta, Wnt, YAP/TAZ, epigenetic, novel).

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.candidate_prioritization \
        --hits path/to/cmap_hits.tsv --drugbank path/to/drugbank_targets.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.cd_fibrosis.network_pharmacology import (
    compute_cross_organ_overlap,
    compute_network_score,
    compute_pathway_overlap,
    score_druggability,
)
from bioagentics.data.cd_fibrosis.positive_controls import (
    POSITIVE_CONTROLS,
    match_compound_name,
)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing"

# ── Composite score weights ──

SCORE_WEIGHTS = {
    "cmap_reversal": 0.25,
    "network_pharmacology": 0.25,
    "safety_profile": 0.20,
    "mechanistic_novelty": 0.15,
    "pharmacokinetics": 0.15,
}

# When CMAP data is not available, redistribute weight
SCORE_WEIGHTS_NO_CMAP = {
    "network_pharmacology": 0.35,
    "safety_profile": 0.25,
    "mechanistic_novelty": 0.20,
    "pharmacokinetics": 0.20,
}

# ── Signature weights for CMAP reversal scoring ──
# Cell-type-resolved signatures weighted higher (less dilution, more specific)

SIGNATURE_WEIGHTS = {
    "celltype": 1.5,
    "cthrc1_yaptaz": 1.4,
    "glis3_il11": 1.3,
    "tl1a_dr3_rho": 1.3,
    "transition": 1.1,
    "bulk": 0.8,
}

# ── Pathway classification and novelty ──

PATHWAY_NOVELTY: dict[str, float] = {
    "TGF-beta": 0.3,       # Most studied fibrosis pathway
    "JAK-STAT": 0.4,       # Approved for CD, growing anti-fibrotic evidence
    "TL1A-DR3": 0.5,       # CD-specific dual pathway, multiple Phase 3 agents
    "ECM_remodeling": 0.6,  # Direct matrix targets (LOX/LOXL2)
    "epigenetic": 0.7,      # HDAC/BRD4 — cross-indication anti-fibrotic data
    "Wnt": 0.7,             # Emerging fibrosis pathway, limited GI clinical data
    "miR-124": 0.7,         # RNA modulation — obefazimod as prototype
    "YAP/TAZ": 0.8,         # Recent mechanosensitive pathway discovery
    "CD38/NAD+": 0.8,       # Novel fibrosis transition marker
    "novel": 1.0,           # No established fibrosis pathway association
}

# ── Pharmacokinetics scoring ──
# Curated PK properties for known anti-fibrotic compound classes

COMPOUND_PK: dict[str, dict] = {
    # Small molecules (oral, gut-absorbed)
    "pirfenidone": {"route": "oral", "mol_type": "small_molecule", "gut_relevant": True},
    "nintedanib": {"route": "oral", "mol_type": "small_molecule", "gut_relevant": True},
    "obefazimod": {"route": "oral", "mol_type": "small_molecule", "gut_relevant": True},
    "upadacitinib": {"route": "oral", "mol_type": "small_molecule", "gut_relevant": True},
    "tofacitinib": {"route": "oral", "mol_type": "small_molecule", "gut_relevant": True},
    "vorinostat": {"route": "oral", "mol_type": "small_molecule", "gut_relevant": True},
    "trichostatin-a": {"route": "oral", "mol_type": "small_molecule", "gut_relevant": True},
    "ontunisertib": {"route": "oral", "mol_type": "small_molecule", "gut_relevant": True},
    # Biologics (IV/SC)
    "duvakitug": {"route": "iv", "mol_type": "biologic", "gut_relevant": True},
    "tulisokibart": {"route": "iv", "mol_type": "biologic", "gut_relevant": True},
    "daratumumab": {"route": "iv", "mol_type": "biologic", "gut_relevant": False},
    "isatuximab": {"route": "iv", "mol_type": "biologic", "gut_relevant": False},
}

# Safety score by clinical stage
SAFETY_SCORES: dict[str, float] = {
    "approved": 1.0,
    "phase_3": 0.8,
    "investigational": 0.6,
    "experimental": 0.3,
    "other": 0.2,
    "unknown": 0.1,
}

# ── Curated compound data (used when DrugBank XML is not available) ──
# Targets and approval data for positive controls and key anti-fibrotic compounds

CURATED_COMPOUNDS: dict[str, dict] = {
    "pirfenidone": {
        "target_genes": "TGFB1;TNF;PDGFRA;COL1A1;COL3A1",
        "groups": "approved",
        "indication": "Approved for idiopathic pulmonary fibrosis (IPF)",
    },
    "nintedanib": {
        "target_genes": "VEGFR1;VEGFR2;VEGFR3;FGFR1;FGFR2;FGFR3;PDGFRA;PDGFRB",
        "groups": "approved",
        "indication": "Approved for IPF and systemic sclerosis-ILD",
    },
    "obefazimod": {
        "target_genes": "MIR124;HDAC1;HDAC2;HDAC3",
        "groups": "investigational",
        "indication": "Phase 3 for ulcerative colitis; preclinical anti-fibrotic",
    },
    "duvakitug": {
        "target_genes": "TNFSF15",
        "groups": "investigational",
        "indication": "Phase 2b for CD (APOLLO-CD: 55% endoscopic response)",
    },
    "tulisokibart": {
        "target_genes": "TNFSF15",
        "groups": "investigational",
        "indication": "Phase 3 for CD (ARES-CD); anti-TL1A mAb",
    },
    "daratumumab": {
        "target_genes": "CD38",
        "groups": "approved",
        "indication": "Approved for multiple myeloma; preclinical CD fibrosis",
    },
    "isatuximab": {
        "target_genes": "CD38",
        "groups": "approved",
        "indication": "Approved for multiple myeloma",
    },
    "upadacitinib": {
        "target_genes": "JAK1",
        "groups": "approved",
        "indication": "Approved for CD, UC, RA; JAK1-selective",
    },
    "tofacitinib": {
        "target_genes": "JAK1;JAK3",
        "groups": "approved",
        "indication": "Approved for UC, RA; pan-JAK inhibitor",
    },
    "vorinostat": {
        "target_genes": "HDAC1;HDAC2;HDAC3;HDAC6",
        "groups": "approved",
        "indication": "Approved for CTCL; pan-HDAC inhibitor",
    },
    "trichostatin-a": {
        "target_genes": "HDAC1;HDAC2;HDAC3;HDAC4;HDAC6",
        "groups": "experimental",
        "indication": "Research tool; HDAC class I/II inhibitor",
    },
    "ontunisertib": {
        "target_genes": "TGFBR1",
        "groups": "investigational",
        "indication": "Phase 2a for stricturing CD (STENOVA met endpoints)",
    },
}

# ── Clinical benchmarks ──
# Reference clinical response rates for anti-fibrotic agents in CD

CLINICAL_BENCHMARKS: dict[str, dict] = {
    "duvakitug": {
        "trial": "APOLLO-CD Phase 2b",
        "endoscopic_response": 0.55,  # 55%
        "metric": "endoscopic_response_rate",
    },
    "upadacitinib": {
        "trial": "U-ACHIEVE/U-ACCOMPLISH",
        "endoscopic_response": 0.45,
        "metric": "endoscopic_response_rate",
    },
}

BENCHMARK_REFERENCE = "duvakitug"  # 55% endoscopic response as reference


def classify_pathway(
    compound_name: str,
    pathway_overlap: dict[str, dict] | None = None,
) -> str:
    """Classify a compound by its primary fibrosis pathway.

    Strategy:
    1. Check if compound is a known positive control (use curated pathway)
    2. Otherwise use network pharmacology pathway with best overlap
    3. Default to "novel" if no pathway hit

    Args:
        compound_name: Compound name.
        pathway_overlap: Output of compute_pathway_overlap.

    Returns:
        Pathway classification string (e.g., "TGF-beta", "YAP/TAZ", "novel").
    """
    # Check positive controls first
    ctrl = match_compound_name(compound_name)
    if ctrl is not None:
        # Map positive control pathway to FIBROSIS_PATHWAYS key
        pathway_map = {
            "TGF-beta": "TGF-beta",
            "RTK/PDGFR": "TGF-beta",  # nintedanib grouped with TGF-beta class
            "TL1A-DR3": "TL1A-DR3",
            "CD38/NAD+": "CD38/NAD+",
            "JAK-STAT": "JAK-STAT",
            "epigenetic/HDAC": "epigenetic",
            "miR-124/epigenetic": "miR-124",
            "ALK5/TGF-beta": "TGF-beta",
        }
        return pathway_map.get(ctrl.pathway, ctrl.pathway)

    # Use pathway overlap from network pharmacology
    if pathway_overlap:
        best_pathway = None
        best_overlap = 0.0
        for name, stats in pathway_overlap.items():
            if stats["n_targets_in_pathway"] > 0:
                frac = stats["overlap_fraction"]
                if frac > best_overlap:
                    best_overlap = frac
                    best_pathway = name

        if best_pathway is not None:
            return best_pathway

    return "novel"


def score_cmap_reversal(
    compound_row: dict | pd.Series,
    signature_weights: dict[str, float] | None = None,
) -> float:
    """Score CMAP/iLINCS reversal strength for a compound.

    Uses mean concordance weighted by signature specificity.
    More negative concordance = stronger anti-fibrotic reversal.

    Args:
        compound_row: Row from CMAP pipeline results.
        signature_weights: Weights per signature type (default: SIGNATURE_WEIGHTS).

    Returns:
        Score in [0, 1] where 1 = strongest reversal.
    """
    sig_weights = signature_weights or SIGNATURE_WEIGHTS

    mean_conc = compound_row.get("mean_concordance")
    if mean_conc is None or pd.isna(mean_conc):
        return 0.0

    mean_conc = float(mean_conc)

    # Apply signature-type weighting if per-signature scores available
    signatures_hit = str(compound_row.get("signatures_hit", ""))
    if signatures_hit:
        sig_names = [s.strip() for s in signatures_hit.split(";") if s.strip()]
        total_weight = sum(sig_weights.get(s, 1.0) for s in sig_names)
        n_sigs = len(sig_names)
        weight_factor = (total_weight / n_sigs) if n_sigs > 0 else 1.0
    else:
        weight_factor = 1.0

    # Normalize concordance: range [-1, 1] -> [0, 1] (more negative = higher score)
    # Clamp to [-1, 1]
    clamped = max(-1.0, min(1.0, mean_conc))
    base_score = (-clamped + 1.0) / 2.0  # -1 -> 1.0, 0 -> 0.5, +1 -> 0.0

    # Apply signature weight factor (caps at 1.0)
    weighted = min(1.0, base_score * weight_factor)

    # Bonus for convergent anti-fibrotic signal
    convergent = compound_row.get("convergent_anti_fibrotic", False)
    if convergent:
        weighted = min(1.0, weighted * 1.15)

    return round(weighted, 4)


def _resolve_drug_info(compound_name: str, drug_info: dict | None = None) -> dict | None:
    """Resolve drug info from DrugBank or curated fallback.

    Args:
        compound_name: Compound name.
        drug_info: DrugBank record (preferred) or None.

    Returns:
        Drug info dict or None.
    """
    if drug_info is not None:
        return drug_info
    return CURATED_COMPOUNDS.get(compound_name.lower().strip())


def score_safety(drug_info: dict | None) -> float:
    """Score compound safety based on clinical development stage.

    Args:
        drug_info: DrugBank record or curated compound dict.

    Returns:
        Score in [0, 1] where 1 = approved drug with known safety profile.
    """
    if drug_info is None:
        return SAFETY_SCORES["unknown"]

    groups = drug_info.get("groups", "").lower()

    if "approved" in groups:
        return SAFETY_SCORES["approved"]
    elif "phase 3" in drug_info.get("indication", "").lower():
        return SAFETY_SCORES["phase_3"]
    elif "investigational" in groups:
        return SAFETY_SCORES["investigational"]
    elif "experimental" in groups:
        return SAFETY_SCORES["experimental"]
    else:
        return SAFETY_SCORES["other"]


def score_clinical_benchmark(
    compound_name: str,
    reference: str | None = None,
) -> float:
    """Score compound against the clinical benchmark reference.

    Compounds with clinical trial data are scored relative to the benchmark
    (duvakitug 55% endoscopic response). Compounds without data get a
    neutral score.

    Args:
        compound_name: Compound name.
        reference: Reference compound name (default: BENCHMARK_REFERENCE).

    Returns:
        Score in [0, 1] where 1 = meets or exceeds benchmark.
    """
    ref = reference or BENCHMARK_REFERENCE
    ref_data = CLINICAL_BENCHMARKS.get(ref)
    if ref_data is None:
        return 0.5  # No reference available

    ref_rate = ref_data["endoscopic_response"]
    compound_data = CLINICAL_BENCHMARKS.get(compound_name.lower().strip())

    if compound_data is None:
        return 0.5  # No clinical data, neutral

    rate = compound_data["endoscopic_response"]
    # Score as ratio to reference, capped at 1.0
    return round(min(1.0, rate / ref_rate), 4)


def score_novelty(pathway_class: str) -> float:
    """Score mechanistic novelty of a compound's pathway.

    Novel targets for CD fibrosis (less-exploited pathways) score higher.

    Args:
        pathway_class: Pathway classification from classify_pathway().

    Returns:
        Score in [0, 1] where 1 = most novel mechanism.
    """
    return PATHWAY_NOVELTY.get(pathway_class, PATHWAY_NOVELTY["novel"])


def score_pharmacokinetics(compound_name: str) -> float:
    """Score compound pharmacokinetics for CD fibrosis treatment.

    Considers route of administration, molecular type, and gut relevance.

    Args:
        compound_name: Compound name.

    Returns:
        Score in [0, 1] where 1 = ideal PK profile (oral, gut-absorbed).
    """
    pk = COMPOUND_PK.get(compound_name.lower().strip())

    if pk is None:
        return 0.5  # Unknown PK, neutral score

    score = 0.0

    # Route of administration
    route_scores = {"oral": 0.45, "subcutaneous": 0.30, "iv": 0.20}
    score += route_scores.get(pk.get("route", ""), 0.20)

    # Molecular type
    mol_scores = {"small_molecule": 0.35, "biologic": 0.20}
    score += mol_scores.get(pk.get("mol_type", ""), 0.20)

    # Gut relevance bonus
    if pk.get("gut_relevant", False):
        score += 0.20

    return round(min(1.0, score), 4)


def compute_composite_score(
    cmap_score: float | None,
    network_score: float,
    safety_score: float,
    novelty_score: float,
    pk_score: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted composite prioritization score.

    Args:
        cmap_score: CMAP reversal score [0-1] or None if unavailable.
        network_score: Network pharmacology score [0-1].
        safety_score: Safety profile score [0-1].
        novelty_score: Mechanistic novelty score [0-1].
        pk_score: Pharmacokinetics score [0-1].
        weights: Score component weights (default: auto-select based on CMAP availability).

    Returns:
        Composite score in [0, 1].
    """
    has_cmap = cmap_score is not None and cmap_score > 0

    if weights is None:
        weights = SCORE_WEIGHTS if has_cmap else SCORE_WEIGHTS_NO_CMAP

    composite = 0.0

    if has_cmap and "cmap_reversal" in weights:
        assert cmap_score is not None
        composite += weights["cmap_reversal"] * cmap_score

    if "network_pharmacology" in weights:
        composite += weights["network_pharmacology"] * network_score

    if "safety_profile" in weights:
        composite += weights["safety_profile"] * safety_score

    if "mechanistic_novelty" in weights:
        composite += weights["mechanistic_novelty"] * novelty_score

    if "pharmacokinetics" in weights:
        composite += weights["pharmacokinetics"] * pk_score

    return round(composite, 4)


def prioritize_candidates(
    ranked_hits: pd.DataFrame | None = None,
    drug_targets: dict[str, dict] | None = None,
    top_n: int = 20,
    compound_column: str = "compound",
) -> pd.DataFrame:
    """Score and rank drug candidates with composite prioritization.

    Combines CMAP reversal, network pharmacology, safety, novelty, and PK
    into a single composite score. When CMAP data is not available, weights
    are redistributed to other components.

    Args:
        ranked_hits: CMAP/iLINCS pipeline results (optional).
        drug_targets: DrugBank data keyed by lowercase compound name.
        top_n: Maximum number of candidates to return.
        compound_column: Column name for compound names.

    Returns:
        DataFrame with composite scores, sorted by composite_score descending.
    """
    drug_targets = drug_targets or {}

    # Build candidate list from CMAP results if available
    if ranked_hits is not None and len(ranked_hits) > 0:
        candidates = ranked_hits.copy()
    else:
        # No CMAP data — use positive controls as seed candidates
        candidates = pd.DataFrame([
            {
                compound_column: ctrl.name,
                "mean_concordance": None,
                "n_signatures_queried": 0,
                "convergent_anti_fibrotic": False,
                "signatures_hit": "",
            }
            for ctrl in POSITIVE_CONTROLS
        ])

    records = []
    for _, row in candidates.iterrows():
        compound = str(row.get(compound_column, "")).strip()
        compound_lower = compound.lower()

        # Resolve targets: DrugBank > curated fallback
        drug_info = _resolve_drug_info(compound, drug_targets.get(compound_lower))
        if drug_info and drug_info.get("target_genes"):
            targets = set(drug_info["target_genes"].upper().split(";"))
        else:
            targets = set()

        # Network pharmacology
        pathway_overlap = compute_pathway_overlap(targets)
        cross_organ = compute_cross_organ_overlap(targets)
        druggability = score_druggability(drug_info)
        net_score = compute_network_score(pathway_overlap, cross_organ, druggability)

        # Pathway classification
        pathway_class = classify_pathway(compound, pathway_overlap)

        # Component scores
        cmap_score = score_cmap_reversal(row)
        safety = score_safety(drug_info)
        novelty = score_novelty(pathway_class)
        pk = score_pharmacokinetics(compound)
        clinical = score_clinical_benchmark(compound)

        # Composite
        composite = compute_composite_score(
            cmap_score if cmap_score > 0 else None,
            net_score,
            safety,
            novelty,
            pk,
        )

        records.append({
            "compound": compound,
            "composite_score": composite,
            "pathway_class": pathway_class,
            "cmap_reversal_score": cmap_score if cmap_score > 0 else None,
            "network_score": net_score,
            "safety_score": safety,
            "novelty_score": novelty,
            "pk_score": pk,
            "clinical_benchmark": clinical,
            "n_targets": len(targets),
            "n_pathways_hit": sum(
                1 for v in pathway_overlap.values()
                if v["n_targets_in_pathway"] > 0
            ),
            "pathways_hit": ";".join(
                name for name, v in pathway_overlap.items()
                if v["n_targets_in_pathway"] > 0
            ),
            "approval_status": druggability["approval_status"],
            "mean_concordance": row.get("mean_concordance"),
            "n_signatures_queried": row.get("n_signatures_queried"),
            "convergent_anti_fibrotic": row.get("convergent_anti_fibrotic"),
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    return df.head(top_n)


def generate_candidate_report(
    ranked_hits: pd.DataFrame | None = None,
    drug_targets: dict[str, dict] | None = None,
    output_dir: Path | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Run candidate prioritization and generate report.

    Args:
        ranked_hits: CMAP/iLINCS pipeline results (optional).
        drug_targets: DrugBank data keyed by lowercase compound name.
        output_dir: Output directory.
        top_n: Maximum candidates.

    Returns:
        Prioritized DataFrame.
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    n_hits = len(ranked_hits) if ranked_hits is not None else 0
    has_cmap = n_hits > 0

    print("=" * 60)
    print("Candidate Prioritization Report")
    print("=" * 60)

    if has_cmap:
        print(f"\n  Input: {n_hits} CMAP/iLINCS hits")
        print("  Scoring mode: full (CMAP + network + safety + novelty + PK)")
    else:
        print("\n  Input: positive control seed candidates (no CMAP data)")
        print("  Scoring mode: network + safety + novelty + PK (CMAP weight redistributed)")

    result = prioritize_candidates(
        ranked_hits,
        drug_targets=drug_targets,
        top_n=top_n,
    )

    if len(result) == 0:
        print("  No candidates to prioritize.")
        return result

    # Summary by pathway
    print(f"\n  Candidates by Pathway:")
    for pathway in result["pathway_class"].unique():
        n = len(result[result["pathway_class"] == pathway])
        print(f"    {pathway:20s} {n:>3d} candidates")

    # Summary by approval status
    print(f"\n  Candidates by Clinical Stage:")
    for status in result["approval_status"].unique():
        n = len(result[result["approval_status"] == status])
        print(f"    {status:20s} {n:>3d} candidates")

    # Top candidates table
    print(f"\n  Top {min(top_n, len(result))} Candidates:")
    print(f"  {'Rank':>4s} {'Compound':25s} {'Score':>7s} {'Pathway':20s} "
          f"{'Net':>5s} {'Safety':>7s} {'Novel':>6s} {'PK':>5s} {'Status':>12s}")
    print(f"  {'-'*4} {'-'*25} {'-'*7} {'-'*20} "
          f"{'-'*5} {'-'*7} {'-'*6} {'-'*5} {'-'*12}")

    for i, (_, row) in enumerate(result.iterrows()):
        print(f"  {i+1:>4d} {str(row['compound']):25s} "
              f"{row['composite_score']:>7.4f} "
              f"{str(row['pathway_class']):20s} "
              f"{row['network_score']:>5.3f} "
              f"{row['safety_score']:>7.3f} "
              f"{row['novelty_score']:>6.3f} "
              f"{row['pk_score']:>5.3f} "
              f"{str(row['approval_status']):>12s}")

    # Success criteria check
    approved_with_novel = result[
        (result["approval_status"].isin(["approved", "phase_3"]))
        & (result["novelty_score"] >= 0.5)
    ]
    print(f"\n  Success Criteria:")
    print(f"    Candidates with approved/Phase3 safety + novel mechanism: "
          f"{len(approved_with_novel)} (target: >= 5)")
    n_pathways = result["pathway_class"].nunique()
    print(f"    Distinct pathway classes represented: {n_pathways}")

    # Save
    out_path = output_dir / "final_candidates.tsv"
    result.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Candidate prioritization for CD fibrosis drug repurposing"
    )
    parser.add_argument(
        "--hits",
        type=Path,
        help="Path to ranked CMAP hits TSV (optional — uses positive controls if absent)",
    )
    parser.add_argument(
        "--drugbank",
        type=Path,
        help="Path to DrugBank targets TSV (optional)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of candidates (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args(argv)

    ranked = None
    if args.hits and args.hits.exists():
        ranked = pd.read_csv(args.hits, sep="\t")
        print(f"Loaded {len(ranked)} compounds from {args.hits}")

    drug_targets = None
    if args.drugbank and args.drugbank.exists():
        from bioagentics.data.cd_fibrosis.drugbank import load_drug_targets
        drug_targets = load_drug_targets(args.drugbank)
        print(f"Loaded {len(drug_targets)} DrugBank records")

    generate_candidate_report(
        ranked,
        drug_targets=drug_targets,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
