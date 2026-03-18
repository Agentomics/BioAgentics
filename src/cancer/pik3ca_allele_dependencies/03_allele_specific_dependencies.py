"""Phase 2b: H1047R vs helical domain allele-specific dependency analysis.

Core hypothesis: different PIK3CA alleles create distinct gene dependencies.
Compares kinase-domain (H1047R/L) vs helical-domain (E545K/E542K) mutations
pan-cancer and within powered cancer types.

Usage:
    uv run python -m pik3ca_allele_dependencies.03_allele_specific_dependencies
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "pik3ca_allele_dependencies"
FIG_DIR = OUTPUT_DIR / "figures"

MIN_N = 5
FDR_THRESHOLD = 0.05
EFFECT_THRESHOLD = 0.3

# Allele groupings
KINASE_ALLELES = {"H1047R", "H1047L"}
HELICAL_ALLELES = {"E545K", "E542K"}

# Pathway genes for annotation
PI3K_PATHWAY = {"PIK3CA", "PIK3R1", "PIK3CB", "PIK3CD", "PIK3CG",
                "AKT1", "AKT2", "AKT3", "MTOR", "PTEN", "TSC1", "TSC2",
                "RPTOR", "RICTOR", "SGK1", "SGK3"}
MAPK_PATHWAY = {"KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "MAP2K1", "MAP2K2",
                "MAPK1", "MAPK3", "NF1"}
CELL_CYCLE = {"CDK4", "CDK6", "CDK2", "CCND1", "CCNE1", "RB1", "CDKN2A", "CDKN2B"}
EPIGENETIC = {"EZH2", "KMT2A", "KMT2D", "ARID1A", "SMARCB1", "BRD4",
              "HDAC1", "HDAC2", "HDAC3", "KDM5A", "KDM6A"}

ALL_PATHWAY_GENES = PI3K_PATHWAY | MAPK_PATHWAY | CELL_CYCLE | EPIGENETIC


def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled)


def cohens_d_ci(d: float, n1: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    se = np.sqrt(1 / n1 + 1 / n2 + d**2 / (2 * (n1 + n2)))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(d - z * se), float(d + z * se)


def annotate_pathway(gene: str) -> str:
    """Annotate a gene with its pathway membership."""
    if gene in PI3K_PATHWAY:
        return "PI3K/AKT/mTOR"
    if gene in MAPK_PATHWAY:
        return "MAPK"
    if gene in CELL_CYCLE:
        return "Cell_cycle"
    if gene in EPIGENETIC:
        return "Epigenetic"
    return ""


def allele_screen(
    crispr: pd.DataFrame, kinase_ids: list[str], helical_ids: list[str], label: str,
) -> pd.DataFrame:
    """Mann-Whitney U for every gene: kinase-domain vs helical-domain."""
    print(f"  Screening {label}: {len(kinase_ids)} kinase vs {len(helical_ids)} helical "
          f"({len(crispr.columns)} genes)...")

    results = []
    n_genes = len(crispr.columns)

    for i, gene in enumerate(crispr.columns):
        if (i + 1) % 3000 == 0:
            print(f"    {i + 1}/{n_genes} genes...", file=sys.stderr)

        k_vals = crispr.loc[kinase_ids, gene].dropna()
        h_vals = crispr.loc[helical_ids, gene].dropna()

        if len(k_vals) < MIN_N or len(h_vals) < MIN_N:
            continue

        k_arr = k_vals.values.astype(float)
        h_arr = h_vals.values.astype(float)
        try:
            stat, pval = stats.mannwhitneyu(k_arr, h_arr, alternative="two-sided")
        except ValueError:
            continue

        d = cohens_d(k_arr, h_arr)  # negative = kinase more dependent
        ci_lo, ci_hi = cohens_d_ci(d, len(k_arr), len(h_arr))

        results.append({
            "gene": gene,
            "n_kinase": len(k_arr),
            "n_helical": len(h_arr),
            "median_kinase": float(np.median(k_arr)),
            "median_helical": float(np.median(h_arr)),
            "cohens_d": d,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "mannwhitney_U": float(stat),
            "mannwhitney_p": pval,
            "pathway": annotate_pathway(gene),
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    reject, fdr, _, _ = multipletests(df["mannwhitney_p"], method="fdr_bh")
    df["fdr"] = fdr
    df["significant"] = reject & (df["cohens_d"].abs() > EFFECT_THRESHOLD)

    return df.sort_values("cohens_d")


def plot_volcano(screen: pd.DataFrame, label: str, out_path: Path) -> None:
    if len(screen) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    neglog_fdr = -np.log10(screen["fdr"].clip(lower=1e-50))
    d = screen["cohens_d"]

    sig = screen["fdr"] < FDR_THRESHOLD
    strong = sig & (d.abs() > EFFECT_THRESHOLD)

    ax.scatter(d[~sig], neglog_fdr[~sig], alpha=0.15, s=8, color="gray", label="NS")
    ax.scatter(d[sig & ~strong], neglog_fdr[sig & ~strong], alpha=0.4, s=12,
               color="#4DBEEE", label=f"FDR<{FDR_THRESHOLD}")
    ax.scatter(d[strong], neglog_fdr[strong], alpha=0.6, s=20,
               color="#D95319", label=f"FDR<{FDR_THRESHOLD} & |d|>{EFFECT_THRESHOLD}")

    # Highlight pathway genes
    for _, row in screen.iterrows():
        if row["gene"] in ALL_PATHWAY_GENES and row["fdr"] < 0.2:
            x = row["cohens_d"]
            y = -np.log10(max(row["fdr"], 1e-50))
            ax.annotate(row["gene"], (x, y), fontsize=7, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
            ax.scatter([x], [y], s=50, edgecolors="black", facecolors="none",
                       linewidths=1.5, zorder=5)

    ax.axhline(-np.log10(FDR_THRESHOLD), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_THRESHOLD, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(EFFECT_THRESHOLD, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cohen's d (kinase - helical)\n← helical more dep.  |  kinase more dep. →")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Allele-Specific Dependencies: {label}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forest(screen: pd.DataFrame, label: str, out_path: Path, n: int = 20) -> None:
    """Forest plot of top allele-specific dependencies with CIs."""
    if len(screen) == 0:
        return

    # Take top N by absolute effect size among nominally significant
    nom_sig = screen[screen["mannwhitney_p"] < 0.05].copy()
    if len(nom_sig) == 0:
        nom_sig = screen.copy()
    nom_sig["abs_d"] = nom_sig["cohens_d"].abs()
    top = nom_sig.nlargest(n, "abs_d").sort_values("cohens_d")

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)))
    y_pos = range(len(top))

    colors = []
    for _, row in top.iterrows():
        if row["cohens_d"] < 0:
            colors.append("#D95319")  # kinase more dependent
        else:
            colors.append("#0072BD")  # helical more dependent

    ax.barh(list(y_pos), top["cohens_d"], xerr=[
        top["cohens_d"] - top["ci_lower"],
        top["ci_upper"] - top["cohens_d"],
    ], color=colors, alpha=0.7, capsize=3)

    gene_labels = []
    for _, row in top.iterrows():
        suffix = ""
        if row["pathway"]:
            suffix = f" [{row['pathway']}]"
        if row["fdr"] < FDR_THRESHOLD:
            suffix += " *"
        gene_labels.append(f"{row['gene']}{suffix}")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(gene_labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Cohen's d (← kinase more dep. | helical more dep. →)")
    ax.set_title(f"Top Allele-Specific Dependencies: {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_boxplots(
    crispr: pd.DataFrame, kinase_ids: list[str], helical_ids: list[str],
    screen: pd.DataFrame, label: str, out_dir: Path, n: int = 5,
) -> None:
    """Box plots for top N allele-specific dependencies."""
    if len(screen) == 0:
        return

    nom_sig = screen[screen["mannwhitney_p"] < 0.05].copy()
    if len(nom_sig) == 0:
        nom_sig = screen.copy()
    nom_sig["abs_d"] = nom_sig["cohens_d"].abs()
    top_genes = nom_sig.nlargest(n, "abs_d")["gene"].tolist()

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, gene in zip(axes, top_genes):
        k_vals = crispr.loc[kinase_ids, gene].dropna().values
        h_vals = crispr.loc[helical_ids, gene].dropna().values

        bp = ax.boxplot([k_vals, h_vals], tick_labels=["Kinase", "Helical"],
                        patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor("#D95319")
        bp["boxes"][1].set_facecolor("#0072BD")
        for b in bp["boxes"]:
            b.set_alpha(0.5)

        ax.set_title(gene, fontsize=9, fontweight="bold")
        ax.set_ylabel("CRISPR Effect" if ax == axes[0] else "")

    fig.suptitle(f"Top Allele-Specific Hits: {label}", fontsize=11)
    fig.tight_layout()
    safe = label.lower().replace(" ", "_").replace("/", "_")
    fig.savefig(out_dir / f"boxplots_allele_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_allele_analysis(
    crispr: pd.DataFrame, classified: pd.DataFrame,
    label: str, mask: pd.Series | None = None,
) -> dict | None:
    """Run allele-specific analysis for a given subset."""
    subset = classified[mask] if mask is not None else classified

    kinase_ids = list(
        set(subset[subset["PIK3CA_allele"].isin(KINASE_ALLELES)].index) & set(crispr.index)
    )
    helical_ids = list(
        set(subset[subset["PIK3CA_allele"].isin(HELICAL_ALLELES)].index) & set(crispr.index)
    )

    if len(kinase_ids) < MIN_N or len(helical_ids) < MIN_N:
        print(f"  {label}: insufficient samples (kinase={len(kinase_ids)}, "
              f"helical={len(helical_ids)})")
        return None

    screen = allele_screen(crispr, kinase_ids, helical_ids, label)
    if len(screen) == 0:
        return None

    safe_name = label.lower().replace(" ", "_").replace("/", "_")
    out_csv = OUTPUT_DIR / f"allele_specific_{safe_name}.csv"
    screen.to_csv(out_csv, index=False)

    plot_volcano(screen, label, FIG_DIR / f"volcano_allele_{safe_name}.png")
    plot_forest(screen, label, FIG_DIR / f"forest_allele_{safe_name}.png")
    plot_top_boxplots(crispr, kinase_ids, helical_ids, screen, label, FIG_DIR)

    n_sig = int(screen["significant"].sum())
    n_nominal = int((screen["mannwhitney_p"] < 0.05).sum())

    # Pathway enrichment among top hits
    pathway_hits = []
    top_nominal = screen[screen["mannwhitney_p"] < 0.05].copy()
    top_nominal["abs_d"] = top_nominal["cohens_d"].abs()
    for _, row in top_nominal.nlargest(20, "abs_d").iterrows():
        entry = {
            "gene": row["gene"],
            "cohens_d": round(float(row["cohens_d"]), 4),
            "fdr": float(row["fdr"]),
            "p_value": float(row["mannwhitney_p"]),
            "pathway": row["pathway"],
            "direction": "kinase_more_dependent" if row["cohens_d"] < 0 else "helical_more_dependent",
        }
        pathway_hits.append(entry)

    return {
        "label": label,
        "n_kinase": len(kinase_ids),
        "n_helical": len(helical_ids),
        "n_genes_tested": len(screen),
        "n_significant_fdr": n_sig,
        "n_nominal_p05": n_nominal,
        "top_hits": pathway_hits,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classified cell lines...")
    classified = pd.read_csv(OUTPUT_DIR / "pik3ca_classified_lines.csv", index_col=0)

    print("Loading CRISPR dependency data...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    common = set(classified.index) & set(crispr.index)
    classified = classified.loc[classified.index.isin(common)]
    print(f"  {len(classified)} lines with both data")

    all_results = []

    # 1. Pan-cancer: kinase vs helical domain
    print("\n=== Pan-cancer: Kinase vs Helical ===")
    result = run_allele_analysis(crispr, classified, "Pan-cancer kinase vs helical")
    if result:
        all_results.append(result)
        print(f"  {result['n_significant_fdr']} FDR-significant, "
              f"{result['n_nominal_p05']} nominal p<0.05")

    # 2. Within powered cancer types
    with open(OUTPUT_DIR / "cancer_type_summary.json") as f:
        summary = json.load(f)

    for ct_info in summary["cancer_types"]:
        ct = ct_info["cancer_type"]
        n_kinase = ct_info.get("N_h1047r_group", 0)
        n_helical = ct_info.get("N_helical_group", 0)
        if n_kinase >= MIN_N and n_helical >= MIN_N:
            print(f"\n=== {ct}: Kinase vs Helical ===")
            mask = classified["OncotreePrimaryDisease"] == ct
            result = run_allele_analysis(crispr, classified, ct, mask)
            if result:
                all_results.append(result)
                print(f"  {result['n_significant_fdr']} FDR-significant, "
                      f"{result['n_nominal_p05']} nominal p<0.05")

    # Save summary
    out_summary = OUTPUT_DIR / "allele_specific_summary.json"
    with open(out_summary, "w") as f:
        json.dump({"analyses": all_results}, f, indent=2)
    print(f"\nSaved allele-specific summary to {out_summary.name}")

    # Print key findings
    if all_results:
        print("\n=== KEY FINDINGS ===")
        for r in all_results:
            print(f"\n{r['label']} ({r['n_kinase']} kinase, {r['n_helical']} helical):")
            print(f"  FDR<0.05 & |d|>0.3: {r['n_significant_fdr']} genes")
            print(f"  Nominal p<0.05: {r['n_nominal_p05']} genes")
            if r["top_hits"]:
                print("  Top hits:")
                for h in r["top_hits"][:10]:
                    direction = "<-kinase" if h["direction"] == "kinase_more_dependent" else "helical->"
                    pw = f" [{h['pathway']}]" if h["pathway"] else ""
                    print(f"    {h['gene']}: d={h['cohens_d']:.3f}, "
                          f"p={h['p_value']:.4f}, FDR={h['fdr']:.3f} "
                          f"{direction}{pw}")


if __name__ == "__main__":
    main()
