"""Phase 4: Cross-cancer comparison with NSCLC reference and ferroptosis-analog identification.

Uses the NSCLC KEAP1-mutant ferroptosis profile as reference to identify cancer
types with analogous or novel ferroptosis vulnerability profiles.

Usage:
    uv run python -m pipelines.pancancer-ferroptosis-atlas.phase4_nsclc_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from bioagentics.config import REPO_ROOT

PHASE1_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase1"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase2"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase4"

# All ferroptosis genes from Phase 1
FERROPTOSIS_GENES = [
    "AIFM2", "GPX4", "SLC7A11", "GLS", "GCLC", "GCLM",
    "TXNRD1", "NQO1", "FTH1", "HMOX1",
    "ACSL4", "LPCAT3", "SAT1", "NCOA4", "TFRC", "ALOX15",
    "SHMT1", "SHMT2", "MTHFD2", "CBS",
]

# Cancer types to flag with special notes
SPECIAL_CASES = {
    "Kidney": "ChRCC: Salem et al. showed GPX4+FSP1 dual vulnerability — compare with NSCLC FSP1-alone",
    "Skin": "Melanoma: lymph node context-specific FSP1 dependence reported",
    "Breast": "FSP1 > NRF2 reliance — FSP1 targeting may work independent of NRF2 status",
}


def extract_nsclc_keap1_profile(
    deps: pd.DataFrame, classification: pd.DataFrame
) -> pd.Series:
    """Extract mean ferroptosis dependency profile for NSCLC KEAP1-mutant lines."""
    # NSCLC = Lung lineage
    lung = deps[deps["OncotreeLineage"] == "Lung"]

    # Filter to KEAP1-mutant
    keap1_class = classification[classification["KEAP1_mutant"]]
    keap1_lung = lung[lung.index.isin(keap1_class.index)]

    gene_cols = [g for g in FERROPTOSIS_GENES if g in deps.columns]

    if len(keap1_lung) == 0:
        print("  WARNING: No KEAP1-mutant NSCLC lines found — using all Lung lines as reference")
        keap1_lung = lung

    print(f"  NSCLC KEAP1-mutant reference: {len(keap1_lung)} cell lines")
    profile = keap1_lung[gene_cols].mean()
    return profile


def compute_similarities(
    deps: pd.DataFrame, ref_profile: pd.Series
) -> pd.DataFrame:
    """Compute cosine similarity and correlation of each cancer type to NSCLC KEAP1-mut profile."""
    gene_cols = [g for g in FERROPTOSIS_GENES if g in deps.columns and g in ref_profile.index]
    ref = ref_profile[gene_cols].values

    mean_deps = deps.groupby("OncotreeLineage")[gene_cols].mean()

    rows = []
    for ct, ct_profile in mean_deps.iterrows():
        ct_vals = ct_profile.values

        # Skip if all NaN
        if np.all(np.isnan(ct_vals)):
            continue

        # Replace NaN with 0 for distance computation
        ct_clean = np.nan_to_num(ct_vals, nan=0.0)
        ref_clean = np.nan_to_num(ref, nan=0.0)

        # Cosine similarity (1 - distance)
        cos_sim = 1 - cosine(ct_clean, ref_clean) if np.any(ct_clean) else 0.0

        # Pearson correlation
        valid = ~(np.isnan(ct_vals) | np.isnan(ref))
        if valid.sum() >= 3:
            corr = np.corrcoef(ct_clean[valid], ref_clean[valid])[0, 1]
        else:
            corr = np.nan

        # FSP1-specific similarity (just AIFM2 dependency difference)
        aifm2_ct = ct_profile.get("AIFM2", np.nan)
        aifm2_ref = ref_profile.get("AIFM2", np.nan)
        fsp1_diff = abs(aifm2_ct - aifm2_ref) if pd.notna(aifm2_ct) and pd.notna(aifm2_ref) else np.nan

        n_lines = len(deps[deps["OncotreeLineage"] == ct])
        note = SPECIAL_CASES.get(ct, "")

        rows.append({
            "cancer_type": ct,
            "n_lines": n_lines,
            "cosine_similarity": cos_sim,
            "pearson_correlation": corr,
            "AIFM2_dependency": aifm2_ct,
            "AIFM2_ref_nsclc_keap1": aifm2_ref,
            "AIFM2_abs_diff": fsp1_diff,
            "special_note": note,
        })

    results = pd.DataFrame(rows)
    return results


def identify_novel_profiles(
    similarities: pd.DataFrame,
    deps: pd.DataFrame,
    ref_profile: pd.Series,
) -> pd.DataFrame:
    """Identify cancer types with divergent ferroptosis profiles."""
    gene_cols = [g for g in FERROPTOSIS_GENES if g in deps.columns and g in ref_profile.index]
    mean_deps = deps.groupby("OncotreeLineage")[gene_cols].mean()

    # Divergent = low cosine similarity (bottom quartile) and not Lung
    non_lung = similarities[similarities["cancer_type"] != "Lung"]
    threshold = non_lung["cosine_similarity"].quantile(0.25)
    divergent = non_lung[non_lung["cosine_similarity"] <= threshold].copy()

    # For each divergent type, identify which genes differ most
    rows = []
    for _, row in divergent.iterrows():
        ct = row["cancer_type"]
        if ct not in mean_deps.index:
            continue

        ct_profile = mean_deps.loc[ct]
        diffs = {}
        for gene in gene_cols:
            diff = ct_profile[gene] - ref_profile[gene]
            diffs[gene] = diff

        # Top 3 most divergent genes
        sorted_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)
        top_divergent = sorted_diffs[:3]

        rows.append({
            "cancer_type": ct,
            "cosine_similarity": row["cosine_similarity"],
            "n_lines": row["n_lines"],
            "top_divergent_gene_1": top_divergent[0][0] if len(top_divergent) > 0 else "",
            "gene_1_diff": top_divergent[0][1] if len(top_divergent) > 0 else np.nan,
            "top_divergent_gene_2": top_divergent[1][0] if len(top_divergent) > 1 else "",
            "gene_2_diff": top_divergent[1][1] if len(top_divergent) > 1 else np.nan,
            "top_divergent_gene_3": top_divergent[2][0] if len(top_divergent) > 2 else "",
            "gene_3_diff": top_divergent[2][1] if len(top_divergent) > 2 else np.nan,
            "special_note": row.get("special_note", ""),
        })

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Cross-cancer NSCLC comparison")
    parser.add_argument("--phase1-dir", type=Path, default=PHASE1_DIR)
    parser.add_argument("--phase2-dir", type=Path, default=PHASE2_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading Phase 1 dependency matrix...")
    deps = pd.read_csv(args.phase1_dir / "ferroptosis_dependency_matrix.csv", index_col="ModelID")

    print("Loading Phase 2 NRF2/KEAP1 classification...")
    classification = pd.read_csv(args.phase2_dir / "nrf2_keap1_classification.csv", index_col="ModelID")

    # Step 1: Extract NSCLC KEAP1-mutant reference profile
    print("\nExtracting NSCLC KEAP1-mutant reference profile...")
    ref_profile = extract_nsclc_keap1_profile(deps, classification)

    ref_path = args.results_dir / "nsclc_reference_profile.csv"
    ref_df = ref_profile.reset_index()
    ref_df.columns = ["gene", "mean_dependency"]
    ref_df.to_csv(ref_path, index=False)
    print(f"Saved: {ref_path}")
    print(f"  Reference profile:\n{ref_df.to_string(index=False)}")

    # Step 2: Compute similarities for all cancer types
    print("\nComputing similarities to NSCLC KEAP1-mutant profile...")
    similarities = compute_similarities(deps, ref_profile)

    # Rank by cosine similarity (excluding Lung itself)
    similarities = similarities.sort_values("cosine_similarity", ascending=False)

    # Full ranking
    analog_path = args.results_dir / "ferroptosis_analog_ranking.csv"
    similarities.to_csv(analog_path, index=False)
    print(f"Saved: {analog_path}")

    # FSP1-specific ranking (by smallest AIFM2 difference)
    fsp1_ranking = similarities[similarities["cancer_type"] != "Lung"].sort_values("AIFM2_abs_diff")
    fsp1_path = args.results_dir / "fsp1_analog_ranking.csv"
    fsp1_ranking.to_csv(fsp1_path, index=False)
    print(f"Saved: {fsp1_path}")

    # Print top analogs
    non_lung = similarities[similarities["cancer_type"] != "Lung"]
    print(f"\nTop 10 ferroptosis-analogous cancer types (most similar to NSCLC KEAP1-mut):")
    print(non_lung.head(10)[
        ["cancer_type", "cosine_similarity", "pearson_correlation", "AIFM2_dependency", "n_lines", "special_note"]
    ].to_string(index=False))

    # Step 3: Identify novel/divergent profiles
    print("\nIdentifying cancer types with novel ferroptosis profiles...")
    novel = identify_novel_profiles(similarities, deps, ref_profile)
    if not novel.empty:
        novel_path = args.results_dir / "novel_profiles.csv"
        novel.to_csv(novel_path, index=False)
        print(f"Saved: {novel_path}")
        print(f"\nDivergent profiles ({len(novel)} cancer types):")
        print(novel[["cancer_type", "cosine_similarity", "top_divergent_gene_1", "gene_1_diff"]].to_string(index=False))
    else:
        print("  No strongly divergent profiles identified")

    # Step 4: Highlight special cases
    print("\nSpecial case flags:")
    for ct, note in SPECIAL_CASES.items():
        match = similarities[similarities["cancer_type"] == ct]
        if not match.empty:
            row = match.iloc[0]
            print(f"  {ct}: cosine_sim={row['cosine_similarity']:.3f}, AIFM2={row['AIFM2_dependency']:.3f} — {note}")
        else:
            print(f"  {ct}: not in dataset — {note}")

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
