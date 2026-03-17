"""Phase 1: Classify PIK3CA allele status across DepMap cell lines.

Annotates all DepMap cell lines with PIK3CA mutation status and allele
classification (H1047R, E545K, E542K, etc.). Identifies cancer types with
sufficient sample sizes for mutant-vs-WT and allele-vs-allele analyses.

Usage:
    uv run python -m pik3ca_allele_dependencies.01_pik3ca_classifier
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import (
    load_depmap_model_metadata,
    load_depmap_mutations,
)
from bioagentics.data.pik3ca_common import (
    classify_pik3ca_allele,
    get_domain,
)

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "pik3ca_allele_dependencies"

MIN_PER_GROUP = 5  # minimum lines per group for powered analysis


def load_cell_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap metadata, keeping lines with cancer type annotation."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreePrimaryDisease"].notna()].copy()
    return meta


def get_lines_with_dependency_data(depmap_dir: Path) -> set[str]:
    """Get ModelIDs that have CRISPR dependency data."""
    crispr_ids = pd.read_csv(
        depmap_dir / "CRISPRGeneEffect.csv", usecols=[0]
    ).iloc[:, 0]
    return set(crispr_ids)


def add_pik3ca_mutations(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add PIK3CA mutation status and allele classification."""
    muts = load_depmap_mutations(depmap_dir / "OmicsSomaticMutations.csv")

    # Filter to PIK3CA, HIGH/MODERATE impact
    pik3ca_muts = muts[
        (muts["HugoSymbol"] == "PIK3CA")
        & (muts["VepImpact"].isin(["HIGH", "MODERATE"]))
    ].copy()

    # Collect protein changes per model
    pc_per_model = pik3ca_muts.groupby("ModelID")["ProteinChange"].apply(list)

    # Binary mutation status
    mutated_ids = set(pik3ca_muts["ModelID"])
    df["PIK3CA_mutated"] = df.index.isin(mutated_ids)

    # Allele classification
    def get_allele(model_id: str) -> str:
        if model_id in pc_per_model.index:
            return classify_pik3ca_allele(pc_per_model[model_id])
        return "WT"

    df["PIK3CA_allele"] = [get_allele(mid) for mid in df.index]

    # Domain classification
    df["PIK3CA_domain"] = df["PIK3CA_allele"].apply(get_domain)

    # Store raw protein changes for reference
    pc_str = pik3ca_muts.groupby("ModelID")["ProteinChange"].apply(
        lambda x: ";".join(x.dropna().unique())
    )
    df["PIK3CA_protein_change"] = df.index.map(pc_str).fillna("")

    print(f"  PIK3CA mutations found in {len(mutated_ids)} cell lines")
    allele_counts = df[df["PIK3CA_mutated"]]["PIK3CA_allele"].value_counts()
    for allele, count in allele_counts.items():
        print(f"    {allele}: {count}")

    return df


def build_cancer_type_summary(df: pd.DataFrame) -> list[dict]:
    """Summarize PIK3CA mutation status per cancer type."""
    summary_rows = []

    for cancer_type, group in df.groupby("OncotreePrimaryDisease"):
        n_total = len(group)
        n_mutant = int(group["PIK3CA_mutated"].sum())
        n_wt = n_total - n_mutant
        freq = n_mutant / n_total if n_total > 0 else 0.0

        # Allele breakdown within mutants
        allele_counts = {}
        if n_mutant > 0:
            ac = group[group["PIK3CA_mutated"]]["PIK3CA_allele"].value_counts()
            allele_counts = ac.to_dict()

        # Power flags
        powered_mutant_vs_wt = n_mutant >= MIN_PER_GROUP and n_wt >= MIN_PER_GROUP
        n_h1047r = allele_counts.get("H1047R", 0) + allele_counts.get("H1047L", 0)
        n_helical = allele_counts.get("E545K", 0) + allele_counts.get("E542K", 0)
        powered_allele_vs_allele = (
            n_h1047r >= MIN_PER_GROUP and n_helical >= MIN_PER_GROUP
        )

        summary_rows.append({
            "cancer_type": cancer_type,
            "N_total": n_total,
            "N_mutant": n_mutant,
            "N_wt": n_wt,
            "mutation_freq": round(freq, 4),
            "allele_counts": allele_counts,
            "N_h1047r_group": n_h1047r,
            "N_helical_group": n_helical,
            "powered_mutant_vs_wt": powered_mutant_vs_wt,
            "powered_allele_vs_allele": powered_allele_vs_allele,
        })

    summary_rows.sort(key=lambda x: x["mutation_freq"], reverse=True)
    return summary_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading DepMap cell lines...")
    df = load_cell_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with cancer type annotation")

    print("Filtering to lines with CRISPR dependency data...")
    crispr_ids = get_lines_with_dependency_data(DEPMAP_DIR)
    df["has_crispr_data"] = df.index.isin(crispr_ids)
    n_crispr = df["has_crispr_data"].sum()
    print(f"  {n_crispr} lines have CRISPR dependency data")

    print("Adding PIK3CA mutation status...")
    df = add_pik3ca_mutations(df, DEPMAP_DIR)

    # Build classified output table
    output_cols = [
        "CellLineName", "StrippedCellLineName",
        "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
        "PIK3CA_mutated", "PIK3CA_allele", "PIK3CA_domain",
        "PIK3CA_protein_change", "has_crispr_data",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()

    out_classified = OUTPUT_DIR / "pik3ca_classified_lines.csv"
    result.to_csv(out_classified)
    print(f"\nSaved {len(result)} classified cell lines to {out_classified.name}")

    # Cancer type summary (restrict to lines with CRISPR data for power analysis)
    crispr_df = df[df["has_crispr_data"]].copy()
    print(f"\nCancer type summary (among {len(crispr_df)} lines with CRISPR data):")
    summary = build_cancer_type_summary(crispr_df)

    powered_mvw = [s for s in summary if s["powered_mutant_vs_wt"]]
    powered_ava = [s for s in summary if s["powered_allele_vs_allele"]]

    print(f"\n  {len(powered_mvw)} cancer types powered for mutant-vs-WT analysis:")
    for s in powered_mvw:
        print(f"    {s['cancer_type']}: {s['N_mutant']}/{s['N_total']} mutant "
              f"({s['mutation_freq']:.1%}), N_wt={s['N_wt']}")

    print(f"\n  {len(powered_ava)} cancer types powered for allele-vs-allele "
          f"(H1047R vs helical):")
    for s in powered_ava:
        print(f"    {s['cancer_type']}: H1047R_group={s['N_h1047r_group']}, "
              f"helical_group={s['N_helical_group']}")

    # Pan-cancer allele distribution
    pan_mutant = crispr_df[crispr_df["PIK3CA_mutated"]]
    print(f"\n  Pan-cancer allele distribution ({len(pan_mutant)} mutant lines "
          f"with CRISPR data):")
    for allele, count in pan_mutant["PIK3CA_allele"].value_counts().items():
        print(f"    {allele}: {count}")

    # Domain-level distribution
    print("\n  Domain distribution:")
    for domain, count in pan_mutant["PIK3CA_domain"].value_counts().items():
        print(f"    {domain}: {count}")

    # Save summary JSON
    out_summary = OUTPUT_DIR / "cancer_type_summary.json"
    with open(out_summary, "w") as f:
        json.dump({
            "total_lines_with_crispr": len(crispr_df),
            "total_pik3ca_mutant_with_crispr": len(pan_mutant),
            "powered_mutant_vs_wt_count": len(powered_mvw),
            "powered_allele_vs_allele_count": len(powered_ava),
            "cancer_types": summary,
        }, f, indent=2)
    print(f"\nSaved summary to {out_summary.name}")


if __name__ == "__main__":
    main()
