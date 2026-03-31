"""Phase 4b addendum: Annotate BRD compound IDs from genome-wide drug hits.

Joins Phase 4b genome-wide MTAP-stratified PRISM results with the Extended
Primary Compound List to add drug names, MOA, and targets to BRD compound IDs.

Usage:
    uv run python -m cancer.cdkn2a_pancancer_dependency_atlas.04b_annotate_drug_hits
"""

from __future__ import annotations

import pandas as pd

from bioagentics.config import REPO_ROOT

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE4B_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase4b_mtap_stratified"

COMPOUND_LIST = DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv"
GENOMEWIDE_RESULTS = PHASE4B_DIR / "genomewide_mtap_stratified.csv"

FDR_THRESHOLD = 0.05


def main() -> None:
    # Load compound annotations
    compounds = pd.read_csv(COMPOUND_LIST)
    # Build lookup: full BRD ID (with BRD: prefix) -> annotation
    compounds["treatment_id"] = compounds["IDs"].str.strip()
    annot = compounds.drop_duplicates("treatment_id").set_index("treatment_id")[
        ["Drug.Name", "MOA", "repurposing_target", "Synonyms"]
    ]

    # Load genome-wide results
    gw = pd.read_csv(GENOMEWIDE_RESULTS)

    # Join annotations
    gw = gw.merge(
        annot, left_on="treatment_id", right_index=True, how="left",
    )
    # Fill in drug_name column from Drug.Name where available
    gw["drug_name"] = gw["Drug.Name"].fillna(gw["drug_name"])

    # Save full annotated results
    gw.to_csv(GENOMEWIDE_RESULTS, index=False)
    print(f"Updated {GENOMEWIDE_RESULTS} with annotations")
    annotated = gw["Drug.Name"].notna().sum()
    total = len(gw)
    print(f"  Annotated: {annotated}/{total} rows ({annotated/total*100:.1f}%)")

    # Extract significant CDKN2A-specific hits with annotations
    cdkn2a_hits = gw[
        (gw["comparison"] == "CDKN2A-del_MTAP-intact_vs_intact")
        & (gw["fdr"] < FDR_THRESHOLD)
        & (gw["cohens_d"] < 0)
    ].sort_values("cohens_d")

    # Create summary table of annotated drug hits
    summary_cols = [
        "treatment_id", "Drug.Name", "MOA", "repurposing_target",
        "cohens_d", "fdr", "n_group_a", "n_group_b",
    ]
    hits_summary = cdkn2a_hits[summary_cols].copy()
    hits_summary.columns = [
        "BRD_ID", "drug_name", "mechanism_of_action", "target",
        "cohens_d", "FDR", "n_CDKN2A_del", "n_control",
    ]

    out_path = PHASE4B_DIR / "annotated_drug_hits.csv"
    hits_summary.to_csv(out_path, index=False)
    print(f"\nSaved {len(hits_summary)} annotated CDKN2A-specific drug hits to {out_path}")

    # Print summary for journal
    print("\n" + "=" * 80)
    print("ANNOTATED CDKN2A-SPECIFIC DRUG HITS (MTAP-corrected, FDR < 0.05)")
    print("=" * 80)

    # Group by MOA
    moa_groups: dict[str, list[dict]] = {}
    for _, row in hits_summary.iterrows():
        moa = str(row["mechanism_of_action"]) if pd.notna(row["mechanism_of_action"]) else "Unknown"
        if moa not in moa_groups:
            moa_groups[moa] = []
        moa_groups[moa].append(row.to_dict())

    for moa, drugs in sorted(moa_groups.items(), key=lambda x: min(d["cohens_d"] for d in x[1])):
        print(f"\n  {moa}:")
        for d in sorted(drugs, key=lambda x: x["cohens_d"]):
            name = d["drug_name"] if pd.notna(d["drug_name"]) else d["BRD_ID"]
            target = f" [{d['target']}]" if pd.notna(d["target"]) else ""
            print(f"    {name}{target}: d={d['cohens_d']:.3f}, FDR={d['FDR']:.4e}")


if __name__ == "__main__":
    main()
