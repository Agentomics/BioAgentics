"""Phase 3: PRISM drug sensitivity stratified by TP53 allele.

Genome-wide screen of all PRISM 24Q2 drugs for differential sensitivity
between TP53 mutant vs WT, structural vs contact allele classes, and
individual hotspot alleles.

Usage:
    uv run python -m tp53_hotspot_allele_dependencies.04_prism_drug_sensitivity
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bioagentics.config import REPO_ROOT
from bioagentics.data.tp53_common import HOTSPOT_ALLELES

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "tp53_hotspot_allele_dependencies"
FIG_DIR = OUTPUT_DIR / "figures"

PRISM_MATRIX = DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv"
PRISM_META = DEPMAP_DIR / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv"

MIN_PER_GROUP = 5  # minimum cell lines per group for a valid comparison


def load_drug_names(meta_path: Path) -> dict[str, str]:
    """Map BRD:broad_id -> drug name from PRISM treatment metadata."""
    meta = pd.read_csv(meta_path)
    dedup = meta.drop_duplicates("broad_id")[["broad_id", "name"]]
    return {f"BRD:{bid}": name for bid, name in zip(dedup["broad_id"], dedup["name"])}


def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled)


def genome_wide_screen(
    prism: pd.DataFrame,
    group1_ids: list[str],
    group2_ids: list[str],
    drug_names: dict[str, str],
) -> pd.DataFrame:
    """Test all drugs for differential sensitivity between two cell line groups.

    Returns DataFrame sorted by p-value with BH-FDR correction.
    More negative PRISM values = more sensitive.
    Positive Cohen's d = group1 has HIGHER values (LESS sensitive) than group2.
    """
    g1_cols = sorted(set(group1_ids) & set(prism.columns))
    g2_cols = sorted(set(group2_ids) & set(prism.columns))

    g1_data = prism[g1_cols].values  # (n_drugs, n_g1)
    g2_data = prism[g2_cols].values  # (n_drugs, n_g2)

    results = []
    for i in range(len(prism)):
        g1_vals = g1_data[i]
        g1_vals = g1_vals[~np.isnan(g1_vals)]
        g2_vals = g2_data[i]
        g2_vals = g2_vals[~np.isnan(g2_vals)]

        if len(g1_vals) < MIN_PER_GROUP or len(g2_vals) < MIN_PER_GROUP:
            continue

        stat, pval = stats.mannwhitneyu(g1_vals, g2_vals, alternative="two-sided")
        d = cohens_d(g1_vals, g2_vals)

        drug_id = prism.index[i]
        results.append({
            "drug_id": drug_id,
            "drug_name": drug_names.get(drug_id, ""),
            "n_group1": len(g1_vals),
            "n_group2": len(g2_vals),
            "median_group1": round(float(np.median(g1_vals)), 4),
            "median_group2": round(float(np.median(g2_vals)), 4),
            "cohens_d": round(d, 4),
            "mannwhitney_U": float(stat),
            "pvalue": float(pval),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    _, fdr, _, _ = multipletests(df["pvalue"], method="fdr_bh")
    df["fdr"] = fdr
    df = df.sort_values("pvalue").reset_index(drop=True)
    return df


def plot_top_drugs(
    prism: pd.DataFrame,
    group1_ids: list[str],
    group2_ids: list[str],
    results: pd.DataFrame,
    label1: str,
    label2: str,
    title: str,
    out_path: Path,
    n_top: int = 12,
) -> None:
    """Box/strip plots for top N differentially sensitive drugs."""
    top = results.head(n_top)
    if len(top) == 0:
        return

    g1_cols = sorted(set(group1_ids) & set(prism.columns))
    g2_cols = sorted(set(group2_ids) & set(prism.columns))

    n_drugs = len(top)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top.iterrows()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        drug_id = row["drug_id"]
        drug_row = prism.loc[drug_id]

        g1_vals = drug_row[g1_cols].dropna().values.astype(float)
        g2_vals = drug_row[g2_cols].dropna().values.astype(float)

        bp = ax.boxplot([g1_vals, g2_vals], tick_labels=[
            f"{label1}\n(n={len(g1_vals)})", f"{label2}\n(n={len(g2_vals)})"
        ], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#D95319")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("#0072BD")
        bp["boxes"][1].set_alpha(0.5)

        name = row["drug_name"] or row["drug_id"][:25]
        ax.set_title(f"{name}\nd={row['cohens_d']:.2f} FDR={row['fdr']:.2e}", fontsize=8)
        ax.set_ylabel("PRISM sensitivity", fontsize=7)
        ax.tick_params(labelsize=7)

    for idx in range(n_drugs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_allele_heatmap(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    mutant_vs_wt_results: pd.DataFrame,
    out_path: Path,
    n_top: int = 30,
) -> None:
    """Heatmap of median drug sensitivity by TP53 allele for top hits."""
    # Use top drugs from mutant-vs-WT screen
    top_drugs = mutant_vs_wt_results.head(n_top)
    if len(top_drugs) == 0:
        return

    # Define allele groups to show
    allele_groups = ["TP53_WT", "R175H", "R248W", "R273H", "G245S", "Y220C", "R282W",
                     "other_missense", "truncating"]
    present_alleles = [a for a in allele_groups
                       if a in classified["TP53_allele"].values]

    # Build median sensitivity matrix
    cell_lines_in_prism = set(prism.columns) & set(classified.index)
    clf = classified.loc[classified.index.isin(cell_lines_in_prism)]

    heatmap_data = []
    drug_labels = []
    for _, row in top_drugs.iterrows():
        drug_id = row["drug_id"]
        drug_row = prism.loc[drug_id]
        medians = []
        for allele in present_alleles:
            ids = clf[clf["TP53_allele"] == allele].index
            vals = drug_row.reindex(ids).dropna()
            medians.append(float(vals.median()) if len(vals) >= 3 else np.nan)
        heatmap_data.append(medians)
        name = row["drug_name"] or drug_id[:25]
        drug_labels.append(name)

    hm = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(10, max(8, n_top * 0.35)))
    im = ax.imshow(hm, aspect="auto", cmap="RdBu", interpolation="nearest")

    ax.set_xticks(range(len(present_alleles)))
    ax.set_xticklabels(present_alleles, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(drug_labels)))
    ax.set_yticklabels(drug_labels, fontsize=8)

    # Center colorbar at 0
    vmax = max(abs(np.nanmin(hm)), abs(np.nanmax(hm)))
    im.set_clim(-vmax, vmax)

    plt.colorbar(im, ax=ax, label="Median PRISM sensitivity\n(more negative = more sensitive)")
    ax.set_title("Top Drug Sensitivities by TP53 Allele", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_priority_compounds(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    drug_ids: dict[str, str],
    out_path: Path,
) -> None:
    """Box plots for priority compound classes by TP53 status and allele class."""
    cell_lines_in_prism = set(prism.columns) & set(classified.index)
    clf = classified.loc[classified.index.isin(cell_lines_in_prism)]

    n_compounds = len(drug_ids)
    if n_compounds == 0:
        return

    fig, axes = plt.subplots(1, n_compounds, figsize=(5 * n_compounds, 5))
    if n_compounds == 1:
        axes = [axes]

    groups = {
        "WT": clf[clf["TP53_allele"] == "TP53_WT"].index.tolist(),
        "Structural": clf[clf["is_structural"]].index.tolist(),
        "Contact": clf[clf["is_contact"]].index.tolist(),
        "Other mut": clf[
            clf["TP53_mutated"] & ~clf["is_structural"] & ~clf["is_contact"]
        ].index.tolist(),
    }
    colors = {"WT": "#808080", "Structural": "#D95319", "Contact": "#0072BD",
              "Other mut": "#77AC30"}

    for ax, (drug_id, drug_name) in zip(axes, drug_ids.items()):
        if drug_id not in prism.index:
            ax.set_title(f"{drug_name}\n(not in PRISM)")
            continue

        drug_row = prism.loc[drug_id]
        data_groups = []
        labels = []
        box_colors = []
        for grp_name, grp_ids in groups.items():
            vals = drug_row.reindex(grp_ids).dropna().values.astype(float)
            if len(vals) >= 3:
                data_groups.append(vals)
                labels.append(f"{grp_name}\n(n={len(vals)})")
                box_colors.append(colors[grp_name])

        if not data_groups:
            ax.set_title(f"{drug_name}\n(insufficient data)")
            continue

        bp = ax.boxplot(data_groups, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Strip plot
        rng = np.random.default_rng(42)
        for i, (vals, color) in enumerate(zip(data_groups, box_colors)):
            jitter = rng.uniform(-0.1, 0.1, len(vals))
            ax.scatter(i + 1 + jitter, vals, c=color, alpha=0.5, s=15,
                       edgecolors="k", linewidths=0.3, zorder=3)

        ax.set_title(drug_name, fontweight="bold", fontsize=10)
        ax.set_ylabel("PRISM sensitivity")
        ax.tick_params(labelsize=8)

    fig.suptitle("Priority Compounds: TP53 Allele Class Sensitivity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading classified TP53 cell lines...")
    classified = pd.read_csv(OUTPUT_DIR / "tp53_classified_lines.csv", index_col=0)
    classified["TP53_mutated"] = classified["TP53_mutated"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    classified["is_structural"] = classified["is_structural"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    classified["is_contact"] = classified["is_contact"].map(
        {"True": True, "False": False, True: True, False: False}
    )

    print("Loading PRISM data matrix (this may take a moment)...")
    prism = pd.read_csv(PRISM_MATRIX, index_col=0)
    print(f"  {prism.shape[0]} drugs x {prism.shape[1]} cell lines")

    print("Loading drug metadata...")
    drug_names = load_drug_names(PRISM_META)
    n_named = sum(1 for d in prism.index if d in drug_names)
    print(f"  {n_named}/{len(prism)} drugs have name annotations")

    # Overlap
    common = sorted(set(classified.index) & set(prism.columns))
    print(f"  {len(common)} cell lines with both TP53 classification and PRISM data")

    clf = classified.loc[classified.index.isin(common)]
    mut_ids = clf[clf["TP53_mutated"]].index.tolist()
    wt_ids = clf[~clf["TP53_mutated"]].index.tolist()
    structural_ids = clf[clf["is_structural"]].index.tolist()
    contact_ids = clf[clf["is_contact"]].index.tolist()
    print(f"  TP53-mutant: {len(mut_ids)}, WT: {len(wt_ids)}")
    print(f"  Structural: {len(structural_ids)}, Contact: {len(contact_ids)}")

    summary = {"n_cell_lines": len(common), "n_drugs": len(prism),
               "n_mutant": len(mut_ids), "n_wt": len(wt_ids),
               "n_structural": len(structural_ids), "n_contact": len(contact_ids),
               "screens": {}}

    # --- Screen 1: TP53 mutant vs WT ---
    print("\n=== Genome-wide screen: TP53 mutant vs WT ===")
    mvw = genome_wide_screen(prism, mut_ids, wt_ids, drug_names)
    mvw.to_csv(OUTPUT_DIR / "prism_mutant_vs_wt.csv", index=False)
    n_sig = (mvw["fdr"] < 0.05).sum()
    print(f"  {len(mvw)} drugs tested, {n_sig} FDR < 0.05")
    if n_sig > 0:
        print("  Top 10 hits:")
        for _, row in mvw.head(10).iterrows():
            name = row["drug_name"] or row["drug_id"][:30]
            direction = "WT more sensitive" if row["cohens_d"] > 0 else "mutant more sensitive"
            print(f"    {name}: d={row['cohens_d']:.3f} FDR={row['fdr']:.2e} ({direction})")
    summary["screens"]["mutant_vs_wt"] = {
        "n_tested": len(mvw), "n_fdr05": int(n_sig),
        "top_hit": mvw.iloc[0].to_dict() if len(mvw) > 0 else None,
    }

    # --- Screen 2: Structural vs Contact ---
    print("\n=== Genome-wide screen: Structural vs Contact ===")
    svc = genome_wide_screen(prism, structural_ids, contact_ids, drug_names)
    svc.to_csv(OUTPUT_DIR / "prism_structural_vs_contact.csv", index=False)
    n_sig_svc = (svc["fdr"] < 0.05).sum()
    print(f"  {len(svc)} drugs tested, {n_sig_svc} FDR < 0.05")
    if n_sig_svc > 0:
        print("  Top 10 hits:")
        for _, row in svc.head(10).iterrows():
            name = row["drug_name"] or row["drug_id"][:30]
            direction = "contact more sensitive" if row["cohens_d"] > 0 else "structural more sensitive"
            print(f"    {name}: d={row['cohens_d']:.3f} FDR={row['fdr']:.2e} ({direction})")
    else:
        print("  Top 5 by nominal p:")
        for _, row in svc.head(5).iterrows():
            name = row["drug_name"] or row["drug_id"][:30]
            print(f"    {name}: d={row['cohens_d']:.3f} p={row['pvalue']:.4f}")
    summary["screens"]["structural_vs_contact"] = {
        "n_tested": len(svc), "n_fdr05": int(n_sig_svc),
        "top_hit": svc.iloc[0].to_dict() if len(svc) > 0 else None,
    }

    # --- Screen 3: Per-allele vs all other TP53-mutant ---
    print("\n=== Per-allele drug sensitivity ===")
    allele_results = []
    for allele in HOTSPOT_ALLELES:
        allele_ids = clf[clf["TP53_allele"] == allele].index.tolist()
        other_mut_ids = [m for m in mut_ids if m not in set(allele_ids)]
        n_allele = len(set(allele_ids) & set(prism.columns))
        if n_allele < MIN_PER_GROUP:
            print(f"  {allele}: skipped (n={n_allele} < {MIN_PER_GROUP})")
            continue

        print(f"  {allele} (n={n_allele}) vs other mutant (n={len(other_mut_ids)})...")
        res = genome_wide_screen(prism, allele_ids, other_mut_ids, drug_names)
        if len(res) > 0:
            res.insert(0, "allele", allele)
            allele_results.append(res)
            n_sig_a = (res["fdr"] < 0.05).sum()
            print(f"    {len(res)} drugs tested, {n_sig_a} FDR < 0.05")
            if n_sig_a > 0:
                for _, row in res[res["fdr"] < 0.05].head(5).iterrows():
                    name = row["drug_name"] or row["drug_id"][:30]
                    print(f"      {name}: d={row['cohens_d']:.3f} FDR={row['fdr']:.2e}")
            else:
                top = res.iloc[0]
                name = top["drug_name"] or top["drug_id"][:30]
                print(f"    Top nominal: {name} d={top['cohens_d']:.3f} p={top['pvalue']:.4f}")

    if allele_results:
        allele_df = pd.concat(allele_results, ignore_index=True)
        allele_df.to_csv(OUTPUT_DIR / "prism_allele_specific.csv", index=False)
        summary["screens"]["allele_specific"] = {
            "alleles_tested": [r["allele"].iloc[0] for r in allele_results],
            "total_tests": len(allele_df),
            "n_fdr05": int((allele_df["fdr"] < 0.05).sum()),
        }

    # --- Priority compounds ---
    print("\n=== Priority compound analysis ===")
    priority_drugs: dict[str, str] = {}
    # Search for available priority compounds
    priority_search = {
        "bortezomib": "Bortezomib (proteasome inhibitor)",
        "tucidinostat": "Tucidinostat (HDACi)",
        "geldanamycin": "Geldanamycin (HSP90i)",
    }
    for keyword, label in priority_search.items():
        for drug_id, name in drug_names.items():
            if isinstance(name, str) and keyword.lower() in name.lower() and drug_id in prism.index:
                priority_drugs[drug_id] = label
                break

    if priority_drugs:
        print(f"  Found {len(priority_drugs)} priority compounds in PRISM:")
        for did, name in priority_drugs.items():
            print(f"    {name} ({did})")
        plot_priority_compounds(prism, classified, priority_drugs,
                                FIG_DIR / "prism_priority_compounds.png")
        print("  Saved priority compound plots")
    else:
        print("  No priority compounds found in PRISM data")

    # --- Plots ---
    print("\n=== Generating plots ===")
    if len(mvw) > 0:
        plot_top_drugs(prism, mut_ids, wt_ids, mvw,
                       "TP53-mut", "TP53-WT",
                       "Top Drugs: TP53 Mutant vs WT Sensitivity",
                       FIG_DIR / "prism_top_mutant_vs_wt.png")
        print("  Saved mutant vs WT top drugs plot")

    if len(svc) > 0:
        plot_top_drugs(prism, structural_ids, contact_ids, svc,
                       "Structural", "Contact",
                       "Top Drugs: Structural vs Contact Allele Sensitivity",
                       FIG_DIR / "prism_top_structural_vs_contact.png")
        print("  Saved structural vs contact top drugs plot")

    if len(mvw) > 0:
        plot_allele_heatmap(prism, classified, mvw,
                            FIG_DIR / "prism_allele_heatmap.png")
        print("  Saved allele heatmap")

    # Save summary
    with open(OUTPUT_DIR / "prism_drug_sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDone. Results in {OUTPUT_DIR.name}/")


if __name__ == "__main__":
    main()
