"""Analyst validation: PIK3CA co-mutation impact on EZH2 dependency.

Tests whether ARID1A-mutant lines with co-occurring PIK3CA mutations show
reduced EZH2 dependency vs ARID1A-mut/PIK3CA-WT, as predicted by the
PIK3IP1 silencing mechanism (ARID1A loss -> EZH2 silences PIK3IP1 ->
PI3K-AKT activation; PIK3CA co-mutation may render EZH2i redundant).

Usage:
    uv run python -m arid1a_pancancer_sl_atlas.analyst_validation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "analyst"


def load_pik3ca_mutations(depmap_dir: Path) -> set[str]:
    """Identify cell lines with PIK3CA activating mutations."""
    cols = ["ModelID", "HugoSymbol", "VariantInfo", "VepImpact", "LikelyLoF"]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    # PIK3CA activating mutations = missense (hotspot) or any non-synonymous
    pik3ca = mutations[
        (mutations["HugoSymbol"] == "PIK3CA")
        & (mutations["VariantInfo"].notna())
        & (mutations["VariantInfo"] != "SILENT")
    ]
    return set(pik3ca["ModelID"].unique())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Analyst Validation: PIK3CA Co-Mutation & EZH2 Dependency ===\n")

    # Load classified cell lines
    classified = pd.read_csv(PHASE1_DIR / "all_cell_lines_classified.csv", index_col=0)
    arid1a_mut = classified[classified["ARID1A_status"] == "mutant"]
    print(f"ARID1A-mutant lines: {len(arid1a_mut)}")

    # Load PIK3CA mutations
    pik3ca_lines = load_pik3ca_mutations(DEPMAP_DIR)
    print(f"PIK3CA-mutant lines (non-silent): {len(pik3ca_lines)}")

    # Split ARID1A-mutant by PIK3CA status
    arid1a_mut = arid1a_mut.copy()
    arid1a_mut["PIK3CA_status"] = arid1a_mut.index.isin(pik3ca_lines)
    n_comut = arid1a_mut["PIK3CA_status"].sum()
    n_pik3ca_wt = len(arid1a_mut) - n_comut
    print(f"ARID1A-mut/PIK3CA-mut: {n_comut}")
    print(f"ARID1A-mut/PIK3CA-WT: {n_pik3ca_wt}")

    # Load EZH2 dependency
    dep = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    if "EZH2" not in dep.columns:
        print("ERROR: EZH2 not in dependency data")
        return

    arid1a_mut = arid1a_mut.join(dep[["EZH2", "ARID1B"]], how="left")
    arid1a_mut = arid1a_mut.dropna(subset=["EZH2"])

    print(f"\nWith dependency data: {len(arid1a_mut)} lines")

    # Pan-cancer comparison
    comut = arid1a_mut[arid1a_mut["PIK3CA_status"]]["EZH2"]
    wt = arid1a_mut[~arid1a_mut["PIK3CA_status"]]["EZH2"]

    results = []
    if len(comut) >= 3 and len(wt) >= 3:
        stat, pval = stats.mannwhitneyu(comut, wt, alternative="greater")
        d = (comut.mean() - wt.mean()) / arid1a_mut["EZH2"].std()
        print(f"\n--- Pan-cancer EZH2 dependency (within ARID1A-mut) ---")
        print(f"  PIK3CA-mut: median={comut.median():.4f}, n={len(comut)}")
        print(f"  PIK3CA-WT:  median={wt.median():.4f}, n={len(wt)}")
        print(f"  Cohen's d={d:.3f}, Mann-Whitney p={pval:.4f} (H1: PIK3CA-mut > PIK3CA-WT)")
        results.append({
            "cancer_type": "Pan-cancer",
            "gene": "EZH2",
            "n_comut": len(comut),
            "n_pik3ca_wt": len(wt),
            "median_dep_comut": comut.median(),
            "median_dep_pik3ca_wt": wt.median(),
            "cohens_d": d,
            "p_value": pval,
        })

    # Also check ARID1B
    comut_b = arid1a_mut[arid1a_mut["PIK3CA_status"]]["ARID1B"]
    wt_b = arid1a_mut[~arid1a_mut["PIK3CA_status"]]["ARID1B"]
    if len(comut_b.dropna()) >= 3 and len(wt_b.dropna()) >= 3:
        comut_b = comut_b.dropna()
        wt_b = wt_b.dropna()
        stat, pval = stats.mannwhitneyu(comut_b, wt_b, alternative="greater")
        d = (comut_b.mean() - wt_b.mean()) / arid1a_mut["ARID1B"].dropna().std()
        print(f"\n--- Pan-cancer ARID1B dependency (within ARID1A-mut) ---")
        print(f"  PIK3CA-mut: median={comut_b.median():.4f}, n={len(comut_b)}")
        print(f"  PIK3CA-WT:  median={wt_b.median():.4f}, n={len(wt_b)}")
        print(f"  Cohen's d={d:.3f}, Mann-Whitney p={pval:.4f}")
        results.append({
            "cancer_type": "Pan-cancer",
            "gene": "ARID1B",
            "n_comut": len(comut_b),
            "n_pik3ca_wt": len(wt_b),
            "median_dep_comut": comut_b.median(),
            "median_dep_pik3ca_wt": wt_b.median(),
            "cohens_d": d,
            "p_value": pval,
        })

    # Per cancer type analysis (where enough co-mutations exist)
    print("\n--- Per-cancer-type EZH2 dependency (ARID1A-mut, by PIK3CA) ---")
    for lineage, group in arid1a_mut.groupby("OncotreeLineage"):
        g_comut = group[group["PIK3CA_status"]]["EZH2"].dropna()
        g_wt = group[~group["PIK3CA_status"]]["EZH2"].dropna()
        if len(g_comut) >= 2 and len(g_wt) >= 2:
            stat, pval = stats.mannwhitneyu(g_comut, g_wt, alternative="greater")
            d = (g_comut.mean() - g_wt.mean()) / group["EZH2"].std()
            print(f"  {lineage}: co-mut n={len(g_comut)} (med={g_comut.median():.3f}), "
                  f"WT n={len(g_wt)} (med={g_wt.median():.3f}), d={d:.3f}, p={pval:.3f}")
            results.append({
                "cancer_type": lineage,
                "gene": "EZH2",
                "n_comut": len(g_comut),
                "n_pik3ca_wt": len(g_wt),
                "median_dep_comut": g_comut.median(),
                "median_dep_pik3ca_wt": g_wt.median(),
                "cohens_d": d,
                "p_value": pval,
            })

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_DIR / "pik3ca_comutation_ezh2_dependency.csv", index=False)
        print(f"\nSaved to {OUTPUT_DIR / 'pik3ca_comutation_ezh2_dependency.csv'}")


if __name__ == "__main__":
    main()
