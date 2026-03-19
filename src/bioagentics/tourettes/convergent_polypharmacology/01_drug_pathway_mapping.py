"""Phase 1a: Drug-pathway mapping for convergent TS pathways.

For each of the 11 convergent pathways identified in ts-rare-variant-convergence
Phase 4, enumerate FDA-approved drugs with known targets in that pathway using
ChEMBL bioactivity data and curated pathway-target memberships.

Computes a binary pathway coverage vector for each drug: which of the 11
convergent pathways does the drug's target profile cover?

Inputs:
  - data/results/ts-rare-variant-convergence/phase4/phase4_pathway_convergence.json
  - data/tourettes/ts-drug-repurposing-network/chembl_bioactivity.tsv
  - output/tourettes/ts-drug-repurposing-network/ranked_candidates.csv

Outputs:
  - output/tourettes/ts-convergent-polypharmacology/drug_pathway_coverage.csv
  - output/tourettes/ts-convergent-polypharmacology/pathway_druggability.csv

Usage:
    uv run python -m bioagentics.tourettes.convergent_polypharmacology.01_drug_pathway_mapping
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT

# --- Paths ---
CONVERGENCE_PATH = (
    REPO_ROOT
    / "data"
    / "results"
    / "ts-rare-variant-convergence"
    / "phase4"
    / "phase4_pathway_convergence.json"
)
CHEMBL_PATH = (
    REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network" / "chembl_bioactivity.tsv"
)
RANKED_PATH = (
    REPO_ROOT
    / "output"
    / "tourettes"
    / "ts-drug-repurposing-network"
    / "ranked_candidates.csv"
)
OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-convergent-polypharmacology"

# --- Curated target-to-pathway mapping ---
# Maps each convergent pathway ID to the ChEMBL drug targets that are
# established members of that pathway (KEGG, GO, Reactome databases).
#
# The 16 ChEMBL targets: DRD1-4, HTR2A, HTR2C, ADRA2A, GABRA1, HRH3,
# CHRM1, CHRM4, CNR1, CNR2, PDE10A, SLC18A2, SLC6A4
TARGET_PATHWAY_MEMBERSHIP: dict[str, list[str]] = {
    # neuronal system (Reactome): neurotransmitter receptors, channels
    "R-HSA-112316": [
        "DRD1", "DRD2", "DRD3", "DRD4",
        "HTR2A", "HTR2C",
        "GABRA1",
        "CHRM1", "CHRM4",
        "ADRA2A",
        "SLC18A2", "SLC6A4",
    ],
    # nervous system development (GO:0007399): developmental, not directly druggable
    "GO:0007399": [],
    # chemical synaptic transmission (GO:0007268): all synaptic signaling
    "GO:0007268": [
        "DRD1", "DRD2", "DRD3", "DRD4",
        "HTR2A", "HTR2C",
        "GABRA1",
        "CHRM1", "CHRM4",
        "ADRA2A",
        "CNR1", "CNR2",
        "HRH3",
        "SLC18A2", "SLC6A4",
    ],
    # dopaminergic synapse (KEGG hsa04728)
    "hsa04728": ["DRD1", "DRD2", "DRD3", "DRD4", "SLC18A2"],
    # axonogenesis (GO:0007409): guidance/growth, not targeted by these drugs
    "GO:0007409": [],
    # cerebral cortex development (GO:0021987): developmental
    "GO:0021987": [],
    # neuron differentiation (GO:0030182): developmental
    "GO:0030182": [],
    # neuron migration (GO:0001764): developmental
    "GO:0001764": [],
    # modulation of chemical synaptic transmission (GO:0050804)
    "GO:0050804": [
        "DRD1", "DRD2", "DRD3", "DRD4",
        "HTR2A", "HTR2C",
        "GABRA1",
        "CHRM1", "CHRM4",
        "ADRA2A",
        "CNR1",
        "HRH3",
    ],
    # generation of neurons (GO:0048699): developmental
    "GO:0048699": [],
    # Notch signaling pathway (KEGG hsa04330): developmental signaling
    "hsa04330": [],
}


def load_convergent_pathways(path: Path) -> list[dict]:
    """Load the 11 convergent pathways from Phase 4 results."""
    with open(path) as f:
        data = json.load(f)

    pathways = []
    for entry in data["convergence_results"]:
        if entry.get("convergence_significant"):
            pathways.append({
                "pathway_id": entry["pathway_id"],
                "pathway_name": entry["pathway_name"],
                "source": entry["source"],
                "combined_p": entry["combined_p"],
                "rare_genes": entry["rare_genes"],
                "gwas_genes": entry["gwas_genes"],
                "all_genes": sorted(set(entry["rare_genes"] + entry["gwas_genes"])),
                "cstc_relevant": entry.get("cstc_relevant", False),
            })

    pathways.sort(key=lambda x: x["combined_p"])
    return pathways


def load_chembl_drugs(path: Path, pchembl_min: float = 6.0) -> dict[str, dict]:
    """Load drug-target mappings from ChEMBL bioactivity data.

    Filters for named compounds with pChEMBL >= 6.0 (sub-micromolar activity).
    Returns: {compound_name: {"targets": set[str], "chembl_id": str, "max_phase": str}}
    """
    df = pd.read_csv(path, sep="\t")
    drugs: dict[str, dict] = {}

    for _, row in df.iterrows():
        name = str(row.get("compound_name", "")).strip()
        if not name or name == "nan":
            continue  # Skip unnamed compounds

        pchembl = row.get("pchembl_value", "")
        try:
            if pchembl and float(pchembl) < pchembl_min:
                continue
        except (ValueError, TypeError):
            continue

        if name not in drugs:
            drugs[name] = {
                "targets": set(),
                "chembl_id": str(row.get("chembl_id", "")),
                "max_phase": "",
            }
        drugs[name]["targets"].add(row["target_gene"])

        # Track highest phase seen
        phase = str(row.get("max_phase", ""))
        if phase and phase > drugs[name]["max_phase"]:
            drugs[name]["max_phase"] = phase

    return drugs


def load_ranked_candidates(path: Path) -> dict[str, dict]:
    """Load previously ranked drug candidates for enrichment."""
    if not path.exists():
        return {}
    candidates = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates[row["drug_name"]] = {
                "rank": int(row["rank"]),
                "final_score": float(row["final_score"]),
                "safety_tier": row["safety_tier"],
                "bbb_penetrant": row["bbb_penetrant"] == "True",
                "in_ts_trials": row["in_ts_trials"] == "True",
                "pathway_annotations": row.get("pathway_annotations", ""),
            }
    return candidates


def compute_pathway_coverage(
    drugs: dict[str, dict],
    pathways: list[dict],
    ranked: dict[str, dict],
) -> list[dict]:
    """Compute binary pathway coverage vector for each drug.

    A drug 'covers' a pathway if at least one of its targets is a member
    of that pathway (using curated TARGET_PATHWAY_MEMBERSHIP).
    """
    # Build reverse map: target -> set of pathway IDs
    target_to_pathways: dict[str, set[str]] = {}
    for pw_id, targets in TARGET_PATHWAY_MEMBERSHIP.items():
        for t in targets:
            target_to_pathways.setdefault(t, set()).add(pw_id)

    pathway_ids = [p["pathway_id"] for p in pathways]
    results = []

    for drug_name, drug_info in sorted(drugs.items()):
        covered = set()
        for target in drug_info["targets"]:
            covered.update(target_to_pathways.get(target, set()))

        # Only keep pathways that are in our convergent set
        covered_convergent = covered & set(pathway_ids)

        if not covered_convergent:
            continue  # Drug doesn't cover any convergent pathway

        row = {
            "drug_name": drug_name,
            "chembl_id": drug_info["chembl_id"],
            "targets": ";".join(sorted(drug_info["targets"])),
            "n_targets": len(drug_info["targets"]),
            "n_pathways_covered": len(covered_convergent),
        }

        # Binary coverage for each pathway
        for pw in pathways:
            row[pw["pathway_id"]] = 1 if pw["pathway_id"] in covered_convergent else 0

        # Add ranked candidate info if available
        rinfo = ranked.get(drug_name, {})
        row["repurposing_rank"] = rinfo.get("rank", "")
        row["repurposing_score"] = rinfo.get("final_score", "")
        row["safety_tier"] = rinfo.get("safety_tier", "")
        row["bbb_penetrant"] = rinfo.get("bbb_penetrant", "")
        row["in_ts_trials"] = rinfo.get("in_ts_trials", "")

        results.append(row)

    results.sort(key=lambda x: (-x["n_pathways_covered"], x["drug_name"]))
    return results


def compute_pathway_druggability(
    pathways: list[dict],
    drugs: dict[str, dict],
) -> list[dict]:
    """For each convergent pathway, summarize druggability.

    Reports number of drug targets in the pathway, number of drugs
    covering it, and whether it's a 'druggability gap'.
    """
    results = []
    for pw in pathways:
        pw_id = pw["pathway_id"]
        pw_targets = TARGET_PATHWAY_MEMBERSHIP.get(pw_id, [])

        # Count drugs covering this pathway
        n_drugs = 0
        drug_names = []
        for drug_name, drug_info in drugs.items():
            if drug_info["targets"] & set(pw_targets):
                n_drugs += 1
                drug_names.append(drug_name)

        results.append({
            "pathway_id": pw_id,
            "pathway_name": pw["pathway_name"],
            "source": pw["source"],
            "combined_p": pw["combined_p"],
            "n_convergent_genes": len(pw["all_genes"]),
            "convergent_genes": ";".join(pw["all_genes"]),
            "n_druggable_targets": len(pw_targets),
            "druggable_targets": ";".join(pw_targets) if pw_targets else "",
            "n_drugs_covering": n_drugs,
            "is_druggability_gap": n_drugs == 0,
            "cstc_relevant": pw["cstc_relevant"],
        })

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 1a: Drug-Pathway Mapping")
    print("=" * 60)

    # Load convergent pathways
    print("\n1. Loading convergent pathways...")
    pathways = load_convergent_pathways(CONVERGENCE_PATH)
    print(f"   {len(pathways)} convergent pathways loaded")
    for pw in pathways:
        print(f"   - {pw['pathway_name']} ({pw['pathway_id']}): p={pw['combined_p']:.2e}")

    # Load ChEMBL drug-target data
    print("\n2. Loading ChEMBL drug-target data...")
    drugs = load_chembl_drugs(CHEMBL_PATH)
    print(f"   {len(drugs)} named compounds with pChEMBL >= 6.0")

    all_targets = set()
    for d in drugs.values():
        all_targets.update(d["targets"])
    print(f"   Targeting {len(all_targets)} unique genes: {', '.join(sorted(all_targets))}")

    # Load ranked candidates
    print("\n3. Loading ranked candidates from drug repurposing...")
    ranked = load_ranked_candidates(RANKED_PATH)
    print(f"   {len(ranked)} previously ranked candidates")

    # Compute pathway coverage
    print("\n4. Computing pathway coverage vectors...")
    coverage = compute_pathway_coverage(drugs, pathways, ranked)
    print(f"   {len(coverage)} drugs cover at least one convergent pathway")

    # Save drug-pathway coverage
    coverage_path = OUTPUT_DIR / "drug_pathway_coverage.csv"
    pathway_ids = [p["pathway_id"] for p in pathways]
    fieldnames = [
        "drug_name", "chembl_id", "targets", "n_targets", "n_pathways_covered",
    ] + pathway_ids + [
        "repurposing_rank", "repurposing_score", "safety_tier", "bbb_penetrant", "in_ts_trials",
    ]
    with open(coverage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage)
    print(f"   Saved to {coverage_path}")

    # Compute pathway druggability
    print("\n5. Computing pathway druggability summary...")
    druggability = compute_pathway_druggability(pathways, drugs)
    druggability_path = OUTPUT_DIR / "pathway_druggability.csv"
    drug_fields = [
        "pathway_id", "pathway_name", "source", "combined_p",
        "n_convergent_genes", "convergent_genes",
        "n_druggable_targets", "druggable_targets",
        "n_drugs_covering", "is_druggability_gap", "cstc_relevant",
    ]
    with open(druggability_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=drug_fields)
        writer.writeheader()
        writer.writerows(druggability)
    print(f"   Saved to {druggability_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    druggable = [d for d in druggability if not d["is_druggability_gap"]]
    gaps = [d for d in druggability if d["is_druggability_gap"]]
    print(f"\nDruggable pathways: {len(druggable)}/{len(pathways)}")
    for d in druggable:
        print(f"  + {d['pathway_name']}: {d['n_drugs_covering']} drugs, "
              f"{d['n_druggable_targets']} targets")

    print(f"\nDruggability gaps: {len(gaps)}/{len(pathways)}")
    for d in gaps:
        print(f"  - {d['pathway_name']} ({d['n_convergent_genes']} genes: "
              f"{d['convergent_genes']})")

    # Top drugs by pathway coverage
    print(f"\nTop drugs by pathway coverage:")
    for row in coverage[:20]:
        covered_names = [
            p["pathway_name"]
            for p in pathways
            if row.get(p["pathway_id"]) == 1
        ]
        rank_info = f" [rank #{row['repurposing_rank']}]" if row["repurposing_rank"] else ""
        print(f"  {row['drug_name']}: {row['n_pathways_covered']} pathways "
              f"(targets: {row['targets']}){rank_info}")
        if covered_names:
            print(f"    -> {', '.join(covered_names)}")

    # Highlight multi-pathway drugs (polypharmacological candidates)
    multi_pw = [r for r in coverage if r["n_pathways_covered"] >= 2]
    print(f"\nPolypharmacological candidates (>=2 pathways): {len(multi_pw)}")
    for row in multi_pw[:10]:
        print(f"  {row['drug_name']}: {row['n_pathways_covered']} pathways via {row['targets']}")


if __name__ == "__main__":
    main()
