"""Phase 2b: TP53 allele-specific dependency analysis.

Three analysis layers:
1. PRIMARY: Structural (R175H, G245S, R282W, Y220C) vs Contact (R248W, R273H, R249S)
2. INDIVIDUAL ALLELE: Each hotspot with N>=10 vs all-other-TP53-mutant
3. KEY BIOLOGY: HSP90, CREBBP/EP300, USP7, DDR genes

Usage:
    uv run python -m tp53_hotspot_allele_dependencies.03_allele_specific_dependencies
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
from bioagentics.data.tp53_common import STRUCTURAL_ALLELES, CONTACT_ALLELES, HOTSPOT_ALLELES

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "tp53_hotspot_allele_dependencies"
FIG_DIR = OUTPUT_DIR / "figures"

MIN_N = 10
FDR_THRESHOLD = 0.05
EFFECT_THRESHOLD = 0.3

# Key biology genes to check
KEY_BIOLOGY = {
    "HSP90AA1", "HSP90AB1",  # chaperone — structural mutants need HSP90
    "CREBBP", "EP300",  # transcriptional GOF — contact mutants
    "USP7",  # literature SL candidate
    "CHEK1", "WEE1", "ATR",  # DDR — R175H replication stress
    "MDM2", "MDM4",  # p53 regulators
    "TP53",  # self-dependency
}

# Broader pathway annotation
DDR_GENES = {"CHEK1", "CHEK2", "WEE1", "ATR", "ATM", "BRCA1", "BRCA2",
             "RAD51", "PARP1", "PARP2"}
CHAPERONE_GENES = {"HSP90AA1", "HSP90AB1", "HSPA1A", "HSPA1B", "HSPA8",
                   "DNAJB1", "STIP1", "CDC37"}
TRANSCRIPTION_GENES = {"CREBBP", "EP300", "BRD4", "MYC", "CDK9", "CCNT1"}

ALL_HIGHLIGHT_GENES = KEY_BIOLOGY | DDR_GENES | CHAPERONE_GENES | TRANSCRIPTION_GENES


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


def annotate_biology(gene: str) -> str:
    if gene in DDR_GENES:
        return "DDR"
    if gene in CHAPERONE_GENES:
        return "Chaperone"
    if gene in TRANSCRIPTION_GENES:
        return "Transcription"
    if gene in {"MDM2", "MDM4", "TP53", "CDKN1A"}:
        return "p53_pathway"
    return ""


def genome_wide_screen(
    crispr: pd.DataFrame, group1_ids: list[str], group2_ids: list[str],
    label: str, g1_name: str = "group1", g2_name: str = "group2",
) -> pd.DataFrame:
    """Run Mann-Whitney U for every gene between two groups."""
    print(f"  Screening {label}: {len(group1_ids)} {g1_name} vs "
          f"{len(group2_ids)} {g2_name} ({len(crispr.columns)} genes)...")

    results = []
    n_genes = len(crispr.columns)

    for i, gene in enumerate(crispr.columns):
        if (i + 1) % 3000 == 0:
            print(f"    {i + 1}/{n_genes} genes...", file=sys.stderr)

        g1_vals = crispr.loc[group1_ids, gene].dropna()
        g2_vals = crispr.loc[group2_ids, gene].dropna()

        if len(g1_vals) < MIN_N or len(g2_vals) < MIN_N:
            continue

        g1_arr = g1_vals.values.astype(float)
        g2_arr = g2_vals.values.astype(float)
        try:
            stat, pval = stats.mannwhitneyu(g1_arr, g2_arr, alternative="two-sided")
        except ValueError:
            continue

        d = cohens_d(g1_arr, g2_arr)
        ci_lo, ci_hi = cohens_d_ci(d, len(g1_arr), len(g2_arr))

        results.append({
            "gene": gene,
            f"n_{g1_name}": len(g1_arr),
            f"n_{g2_name}": len(g2_arr),
            f"median_{g1_name}": float(np.median(g1_arr)),
            f"median_{g2_name}": float(np.median(g2_arr)),
            "cohens_d": d,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "mann_whitney_U": float(stat),
            "p_value": pval,
            "biology": annotate_biology(gene),
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    reject, fdr, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["fdr"] = fdr
    df["significant"] = reject & (df["cohens_d"].abs() > EFFECT_THRESHOLD)
    return df.sort_values("cohens_d")


def plot_volcano(screen: pd.DataFrame, label: str, out_path: Path,
                 g1_name: str = "group1", g2_name: str = "group2") -> None:
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

    # Highlight key biology genes
    for _, row in screen.iterrows():
        if row["gene"] in ALL_HIGHLIGHT_GENES and row["fdr"] < 0.2:
            x = row["cohens_d"]
            y = -np.log10(max(row["fdr"], 1e-50))
            ax.annotate(row["gene"], (x, y), fontsize=7, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
            ax.scatter([x], [y], s=50, edgecolors="black", facecolors="none",
                       linewidths=1.5, zorder=5)

    ax.axhline(-np.log10(FDR_THRESHOLD), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_THRESHOLD, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(EFFECT_THRESHOLD, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel(f"Cohen's d ({g1_name} − {g2_name})\n"
                  f"← {g1_name} more essential | {g2_name} more essential →")
    ax.set_ylabel("−log10(FDR)")
    ax.set_title(f"TP53 Allele-Specific: {label}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(allele_results: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Heatmap of Cohen's d for key biology genes across all comparisons."""
    genes_to_show = sorted(KEY_BIOLOGY)
    comparisons = list(allele_results.keys())

    if not comparisons:
        return

    matrix = pd.DataFrame(index=genes_to_show, columns=comparisons, dtype=float)
    fdr_matrix = pd.DataFrame(index=genes_to_show, columns=comparisons, dtype=float)

    for comp, screen in allele_results.items():
        for gene in genes_to_show:
            row = screen[screen["gene"] == gene]
            if not row.empty:
                matrix.loc[gene, comp] = row.iloc[0]["cohens_d"]
                fdr_matrix.loc[gene, comp] = row.iloc[0]["fdr"]

    matrix = matrix.fillna(0)
    fdr_matrix = fdr_matrix.fillna(1)

    fig, ax = plt.subplots(figsize=(max(8, len(comparisons) * 1.5), max(6, len(genes_to_show) * 0.4)))
    im = ax.imshow(matrix.values.astype(float), aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)

    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(comparisons, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(genes_to_show)))
    ax.set_yticklabels(genes_to_show, fontsize=9)

    # Add significance stars
    for i, gene in enumerate(genes_to_show):
        for j, comp in enumerate(comparisons):
            fdr_val = fdr_matrix.loc[gene, comp]
            d_val = matrix.loc[gene, comp]
            text = f"{d_val:.2f}"
            if fdr_val < 0.05:
                text += "*"
            ax.text(j, i, text, ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label="Cohen's d", shrink=0.7)
    ax.set_title("TP53 Allele-Specific Dependencies: Key Biology Genes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_key_biology_summary(screen: pd.DataFrame) -> dict:
    """Extract results for key biology genes."""
    summary = {}
    for gene in KEY_BIOLOGY:
        row = screen[screen["gene"] == gene]
        if not row.empty:
            r = row.iloc[0]
            summary[gene] = {
                "cohens_d": round(r["cohens_d"], 4),
                "fdr": float(r["fdr"]),
                "p_value": float(r["p_value"]),
            }
    return summary


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classified cell lines...")
    classified = pd.read_csv(OUTPUT_DIR / "tp53_classified_lines.csv", index_col=0)

    print("Loading CRISPR dependency data...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    common = set(classified.index) & set(crispr.index)
    classified = classified.loc[classified.index.isin(common)]
    print(f"  {len(classified)} lines with both data")

    # All TP53-mutant lines (for individual allele comparisons)
    all_mutant = classified[classified["TP53_mutated"].astype(bool)]

    all_results = []
    allele_screens = {}  # for heatmap

    # ================================================================
    # LAYER 1: Structural vs Contact (PRIMARY CONTRAST)
    # ================================================================
    print("\n=== PRIMARY: Structural vs Contact ===")
    struct_ids = list(
        set(all_mutant[all_mutant["is_structural"].astype(bool)].index) & set(crispr.index)
    )
    contact_ids = list(
        set(all_mutant[all_mutant["is_contact"].astype(bool)].index) & set(crispr.index)
    )

    print(f"  Structural: {len(struct_ids)}, Contact: {len(contact_ids)}")

    if len(struct_ids) >= MIN_N and len(contact_ids) >= MIN_N:
        screen = genome_wide_screen(
            crispr, struct_ids, contact_ids,
            "Structural vs Contact", "structural", "contact"
        )
        if len(screen) > 0:
            screen.to_csv(OUTPUT_DIR / "structural_vs_contact.csv", index=False)
            plot_volcano(screen, "Structural vs Contact",
                        FIG_DIR / "volcano_structural_vs_contact.png",
                        "structural", "contact")
            allele_screens["Structural_vs_Contact"] = screen

            n_sig = int(screen["significant"].sum())
            bio = get_key_biology_summary(screen)
            all_results.append({
                "comparison": "Structural vs Contact",
                "n_group1": len(struct_ids),
                "n_group2": len(contact_ids),
                "n_significant": n_sig,
                "key_biology": bio,
            })
            print(f"  {n_sig} significant genes")
            for gene, info in bio.items():
                sig_flag = "*" if info["fdr"] < FDR_THRESHOLD else ""
                print(f"    {gene}: d={info['cohens_d']:.3f}, FDR={info['fdr']:.2e}{sig_flag}")

    # ================================================================
    # LAYER 2: Individual allele vs all-other-TP53-mutant
    # ================================================================
    print("\n=== INDIVIDUAL ALLELE ANALYSES ===")

    for allele in HOTSPOT_ALLELES:
        allele_mask = all_mutant["TP53_allele"] == allele
        allele_ids = list(set(all_mutant[allele_mask].index) & set(crispr.index))
        other_ids = list(set(all_mutant[~allele_mask].index) & set(crispr.index))

        if len(allele_ids) < MIN_N:
            print(f"\n  {allele}: N={len(allele_ids)} — SKIPPING (underpowered)")
            continue

        print(f"\n  --- {allele} vs other TP53-mutant ---")
        screen = genome_wide_screen(
            crispr, allele_ids, other_ids,
            f"{allele} vs other", allele, "other_mut"
        )
        if len(screen) > 0:
            safe = allele.lower()
            screen.to_csv(OUTPUT_DIR / f"allele_{safe}_vs_other.csv", index=False)
            plot_volcano(screen, f"{allele} vs Other TP53-mut",
                        FIG_DIR / f"volcano_{safe}_vs_other.png",
                        allele, "other_mut")
            allele_screens[allele] = screen

            n_sig = int(screen["significant"].sum())
            bio = get_key_biology_summary(screen)
            all_results.append({
                "comparison": f"{allele} vs other TP53-mutant",
                "n_allele": len(allele_ids),
                "n_other": len(other_ids),
                "n_significant": n_sig,
                "key_biology": bio,
            })
            print(f"  {n_sig} significant genes")
            # Print key biology
            for gene in ["HSP90AA1", "HSP90AB1", "CREBBP", "EP300", "USP7",
                         "CHEK1", "WEE1", "ATR"]:
                if gene in bio:
                    sig_flag = "*" if bio[gene]["fdr"] < FDR_THRESHOLD else ""
                    print(f"    {gene}: d={bio[gene]['cohens_d']:.3f}, "
                          f"FDR={bio[gene]['fdr']:.2e}{sig_flag}")

    # ================================================================
    # HEATMAP across all comparisons
    # ================================================================
    if allele_screens:
        print("\nGenerating key biology heatmap...")
        plot_heatmap(allele_screens, FIG_DIR / "heatmap_key_biology.png")

    # Save summary
    out_summary = OUTPUT_DIR / "allele_specific_summary.json"
    with open(out_summary, "w") as f:
        json.dump({"analyses": all_results}, f, indent=2)
    print(f"\nSaved allele-specific summary to {out_summary.name}")


if __name__ == "__main__":
    main()
