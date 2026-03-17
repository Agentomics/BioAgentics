"""Phase 2: NRF2/KEAP1 classification and stratified ferroptosis dependency analysis.

Classifies DepMap cell lines by NRF2/KEAP1 mutation status, performs stratified
comparisons of ferroptosis gene dependencies, and tests FSP1/NRF2 independence.

Usage:
    uv run python -m pipelines.pancancer-ferroptosis-atlas.phase2_nrf2_stratification
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from bioagentics.config import REPO_ROOT

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase1"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase2"

# Genes to test in stratified analysis (ferroptosis defense + metabolic)
FERROPTOSIS_TEST_GENES = [
    "AIFM2", "GPX4", "SLC7A11", "GLS", "GCLC", "GCLM",
    "TXNRD1", "NQO1", "FTH1", "HMOX1",
    "SHMT1", "SHMT2", "MTHFD2", "CBS",
]

MIN_GROUP_SIZE = 5


def classify_nrf2_keap1(depmap_dir: Path, model_ids: set[str]) -> pd.DataFrame:
    """Classify cell lines by NRF2/KEAP1 mutation status from OmicsSomaticMutations.csv.

    KEAP1-mutant: loss-of-function (LikelyLoF, TumorSuppressorHighImpact, or
                  VepImpact HIGH/MODERATE damaging mutations)
    NFE2L2-mutant: gain-of-function (Hotspot, OncogeneHighImpact)
    """
    print("Loading OmicsSomaticMutations.csv (KEAP1/NFE2L2 only)...")
    cols = [
        "ModelID", "HugoSymbol", "ProteinChange", "VepImpact",
        "MolecularConsequence", "LikelyLoF", "Hotspot",
        "OncogeneHighImpact", "TumorSuppressorHighImpact",
    ]

    # Read in chunks to filter efficiently — file is ~596MB
    chunks = []
    for chunk in pd.read_csv(depmap_dir / "OmicsSomaticMutations.csv", usecols=cols, chunksize=500000):
        filtered = chunk[
            (chunk["HugoSymbol"].isin(["KEAP1", "NFE2L2"]))
            & (chunk["ModelID"].isin(model_ids))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        print("  WARNING: No KEAP1/NFE2L2 mutations found")
        # Return all WT classification
        return pd.DataFrame({
            "ModelID": list(model_ids),
            "KEAP1_mutant": False,
            "NFE2L2_mutant": False,
            "NRF2_active": False,
            "nrf2_status": "WT",
        }).set_index("ModelID")

    muts = pd.concat(chunks, ignore_index=True)
    print(f"  Found {len(muts)} KEAP1/NFE2L2 mutations across {muts['ModelID'].nunique()} lines")

    # Convert boolean-like columns
    for col in ["LikelyLoF", "Hotspot", "OncogeneHighImpact", "TumorSuppressorHighImpact"]:
        if col in muts.columns:
            muts[col] = muts[col].astype(str).str.lower().isin(["true", "1", "yes"])

    # KEAP1 loss-of-function
    keap1_muts = muts[muts["HugoSymbol"] == "KEAP1"]
    keap1_lof = keap1_muts[
        keap1_muts["LikelyLoF"]
        | keap1_muts["TumorSuppressorHighImpact"]
        | (keap1_muts["VepImpact"].isin(["HIGH"]))
    ]
    keap1_mutant_ids = set(keap1_lof["ModelID"].unique())

    # NFE2L2 gain-of-function
    nfe2l2_muts = muts[muts["HugoSymbol"] == "NFE2L2"]
    nfe2l2_gof = nfe2l2_muts[
        nfe2l2_muts["Hotspot"]
        | nfe2l2_muts["OncogeneHighImpact"]
        | ((nfe2l2_muts["VepImpact"].isin(["HIGH", "MODERATE"]))
           & nfe2l2_muts["MolecularConsequence"].str.contains("missense", case=False, na=False))
    ]
    nfe2l2_mutant_ids = set(nfe2l2_gof["ModelID"].unique())

    print(f"  KEAP1-mutant (LOF): {len(keap1_mutant_ids)} lines")
    print(f"  NFE2L2-mutant (GOF): {len(nfe2l2_mutant_ids)} lines")
    print(f"  NRF2-active (either): {len(keap1_mutant_ids | nfe2l2_mutant_ids)} lines")

    # Build classification table
    rows = []
    for mid in model_ids:
        keap1 = mid in keap1_mutant_ids
        nfe2l2 = mid in nfe2l2_mutant_ids
        nrf2_active = keap1 or nfe2l2
        rows.append({
            "ModelID": mid,
            "KEAP1_mutant": keap1,
            "NFE2L2_mutant": nfe2l2,
            "NRF2_active": nrf2_active,
            "nrf2_status": "NRF2-active" if nrf2_active else "WT",
        })

    return pd.DataFrame(rows).set_index("ModelID")


def stratified_comparison(
    deps: pd.DataFrame,
    classification: pd.DataFrame,
) -> pd.DataFrame:
    """Compare ferroptosis dependencies between NRF2-active vs WT per cancer type."""
    merged = deps.join(classification[["nrf2_status"]], how="inner")

    rows = []
    for lineage, grp in merged.groupby("OncotreeLineage"):
        active = grp[grp["nrf2_status"] == "NRF2-active"]
        wt = grp[grp["nrf2_status"] == "WT"]

        if len(active) < MIN_GROUP_SIZE or len(wt) < MIN_GROUP_SIZE:
            continue

        for gene in FERROPTOSIS_TEST_GENES:
            if gene not in grp.columns:
                continue
            a_vals = active[gene].dropna()
            w_vals = wt[gene].dropna()
            if len(a_vals) < MIN_GROUP_SIZE or len(w_vals) < MIN_GROUP_SIZE:
                continue

            # Mann-Whitney U test
            stat, pval = mannwhitneyu(a_vals, w_vals, alternative="two-sided")

            # Rank-biserial correlation as effect size
            n1, n2 = len(a_vals), len(w_vals)
            rbc = 1 - (2 * stat) / (n1 * n2)

            # Cohen's d
            pooled_std = np.sqrt(
                ((n1 - 1) * a_vals.std() ** 2 + (n2 - 1) * w_vals.std() ** 2) / (n1 + n2 - 2)
            )
            cohens_d = (a_vals.mean() - w_vals.mean()) / pooled_std if pooled_std > 0 else 0.0

            rows.append({
                "cancer_type": lineage,
                "gene": gene,
                "n_nrf2_active": len(a_vals),
                "n_wt": len(w_vals),
                "mean_nrf2_active": a_vals.mean(),
                "mean_wt": w_vals.mean(),
                "diff_active_minus_wt": a_vals.mean() - w_vals.mean(),
                "cohens_d": cohens_d,
                "rank_biserial": rbc,
                "mannwhitney_U": stat,
                "p_value": pval,
            })

    if not rows:
        print("  WARNING: No cancer types with sufficient N in both groups")
        return pd.DataFrame()

    results = pd.DataFrame(rows)

    # FDR correction across all tests
    _, fdr_pvals, _, _ = multipletests(results["p_value"], method="fdr_bh")
    results["fdr_q_value"] = fdr_pvals

    return results


def fsp1_independence_analysis(comparison: pd.DataFrame) -> pd.DataFrame:
    """Extract AIFM2 (FSP1) results to test NRF2 independence.

    FSP1 is NOT regulated by NRF2 — expect no significant difference.
    """
    if comparison.empty:
        return pd.DataFrame()

    fsp1 = comparison[comparison["gene"] == "AIFM2"].copy()
    fsp1["nrf2_independent"] = fsp1["fdr_q_value"] > 0.05
    fsp1["note"] = fsp1.apply(
        lambda r: "EXPECTED: FSP1 independent of NRF2"
        if r["nrf2_independent"]
        else "UNEXPECTED: FSP1 correlates with NRF2 — flag for investigation",
        axis=1,
    )
    return fsp1


def plot_effect_size_heatmap(comparison: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of Cohen's d by cancer type x ferroptosis gene."""
    if comparison.empty:
        print("  No data for effect size heatmap — skipping")
        return

    pivot = comparison.pivot_table(
        index="cancer_type", columns="gene", values="cohens_d", aggfunc="first"
    )

    if pivot.empty or len(pivot) < 2:
        print("  Too few cancer types for heatmap — skipping")
        return

    # Mark significant results
    sig_pivot = comparison.pivot_table(
        index="cancer_type", columns="gene", values="fdr_q_value", aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.5)))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.5)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    # Mark significant cells with asterisk
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            gene = pivot.columns[j]
            ct = pivot.index[i]
            if gene in sig_pivot.columns and ct in sig_pivot.index:
                q = sig_pivot.loc[ct, gene]
                if pd.notna(q) and q < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Cohen's d (NRF2-active − WT)")
    ax.set_title("NRF2/KEAP1 effect on ferroptosis dependencies (* = FDR < 0.05)", fontsize=11)
    ax.set_xlabel("Ferroptosis gene")
    ax.set_ylabel("Cancer type")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 2: NRF2/KEAP1 stratified analysis")
    parser.add_argument("--depmap-dir", type=Path, default=DEPMAP_DIR)
    parser.add_argument("--phase1-dir", type=Path, default=PHASE1_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 1 dependency matrix
    print("Loading Phase 1 dependency matrix...")
    deps = pd.read_csv(args.phase1_dir / "ferroptosis_dependency_matrix.csv", index_col="ModelID")

    model_ids = set(deps.index)

    # Step 1: Classify NRF2/KEAP1 status
    classification = classify_nrf2_keap1(args.depmap_dir, model_ids)

    class_path = args.results_dir / "nrf2_keap1_classification.csv"
    classification.to_csv(class_path)
    print(f"\nClassification summary:")
    print(classification["nrf2_status"].value_counts().to_string())

    # Flag ovarian cancer NRF2 status
    if "OncotreeLineage" in deps.columns:
        ovarian = deps[deps["OncotreeLineage"] == "Ovary/Fallopian Tube"]
        ov_class = classification.loc[classification.index.isin(ovarian.index)]
        ov_nrf2 = ov_class["NRF2_active"].sum()
        print(f"\n  NOTE: Ovarian cancer — {ov_nrf2}/{len(ov_class)} lines NRF2-active by mutation.")
        if ov_nrf2 < 3:
            print("  Ovarian NRF2 may be activated epigenetically (TFAP2C/HDAC) — flag for analyst.")

    # Step 2: Stratified comparison
    print("\nRunning stratified ferroptosis dependency comparison...")
    comparison = stratified_comparison(deps, classification)

    if not comparison.empty:
        comp_path = args.results_dir / "stratified_dependency_comparison.csv"
        comparison.to_csv(comp_path, index=False)
        print(f"Saved comparison: {comp_path} ({len(comparison)} tests)")

        sig = comparison[comparison["fdr_q_value"] < 0.05]
        print(f"  Significant (FDR < 0.05): {len(sig)} tests")
        if not sig.empty:
            print("\n  Top significant results:")
            print(sig.sort_values("fdr_q_value")[
                ["cancer_type", "gene", "cohens_d", "fdr_q_value"]
            ].head(10).to_string(index=False))

        # Step 3: FSP1/NRF2 independence analysis
        print("\nFSP1/NRF2 independence analysis...")
        fsp1_results = fsp1_independence_analysis(comparison)
        if not fsp1_results.empty:
            fsp1_path = args.results_dir / "fsp1_nrf2_independence_test.csv"
            fsp1_results.to_csv(fsp1_path, index=False)
            print(f"  FSP1 independence results:")
            for _, row in fsp1_results.iterrows():
                print(f"    {row['cancer_type']}: p={row['p_value']:.3f}, q={row['fdr_q_value']:.3f} — {row['note']}")

        # Step 4: Effect size heatmap
        heatmap_path = args.results_dir / "nrf2_effect_size_heatmap.png"
        plot_effect_size_heatmap(comparison, heatmap_path)
    else:
        print("  No testable cancer types (need N >= 5 in both NRF2-active and WT groups)")

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
