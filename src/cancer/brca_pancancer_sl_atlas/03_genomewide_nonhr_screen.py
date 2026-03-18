"""Phase 3: Genome-wide non-HR dependency screen.

For each qualifying cancer type, screens all ~18,000 genes for differential
dependency between BRCA-deficient and proficient lines. Identifies non-HR
synthetic lethal targets, categorizes by pathway, and stratifies by BRCA1 vs
BRCA2. Includes priority analysis for POLB and cGAS-STING pathway, plus
53BP1/SHLD stratification of top hits.

Usage:
    uv run python -m brca_pancancer_sl_atlas.03_genomewide_nonhr_screen
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

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase3"

# Significance thresholds
FDR_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.3  # |Cohen's d| per plan

# BRCA group labels
BRCA_GROUPS = ["any_brca", "brca1_only", "brca2_only"]

# Minimum samples per group
MIN_DEF = 3
MIN_PROF = 3

# --- Known HR pathway genes (exclude from "novel non-HR" list) ---
HR_PATHWAY_GENES = {
    "BRCA1", "BRCA2", "PALB2", "RAD51", "RAD51B", "RAD51C", "RAD51D",
    "XRCC2", "XRCC3", "DMC1", "RAD54L", "RAD54B", "BRIP1", "BARD1",
    "NBN", "MRE11", "RAD50", "ATM", "ATR", "CHEK1", "CHEK2",
    "RPA1", "RPA2", "RPA3", "PARP1", "PARP2", "POLQ",
    "TP53BP1", "RIF1", "SHLD1", "SHLD2", "SHLD3", "MAD2L2",
    "TOPBP1", "ATRIP", "RBBP8",  # CtIP
    "EXO1", "DNA2", "BLM",
}

# --- Priority targets from addendum ---
POLB_GENE = "POLB"
CGAS_STING_GENES = ["CGAS", "STING1", "TBK1", "IRF3"]

# --- Pathway categorization ---
PATHWAY_GENES = {
    "DNA repair (BER)": {"POLB", "APEX1", "APEX2", "XRCC1", "PNKP", "LIG3",
                         "OGG1", "MUTYH", "UNG", "TDG", "NEIL1", "NEIL2", "NEIL3",
                         "SMUG1", "MPG", "NTHL1", "MBD4", "FEN1"},
    "DNA repair (NER)": {"XPA", "XPB", "XPC", "XPD", "XPF", "XPG", "ERCC1",
                         "ERCC2", "ERCC3", "ERCC4", "ERCC5", "ERCC6", "ERCC8",
                         "DDB1", "DDB2", "RAD23A", "RAD23B"},
    "DNA repair (MMR)": {"MLH1", "MSH2", "MSH3", "MSH6", "PMS1", "PMS2",
                         "PCNA", "EXO1"},
    "DNA repair (NHEJ)": {"XRCC4", "XRCC5", "XRCC6", "LIG4", "DCLRE1C",
                          "PRKDC", "NHEJ1"},
    "DNA repair (MMEJ)": {"POLQ", "LIG3", "PARP1", "XRCC1"},
    "DNA repair (Fanconi anemia)": {"FANCA", "FANCB", "FANCC", "FANCD2", "FANCE",
                                    "FANCF", "FANCG", "FANCI", "FANCL", "FANCM",
                                    "UBE2T", "SLX4"},
    "Metabolic": {"MTHFD2", "SHMT2", "MTHFD1", "DHFR", "TYMS", "RRM1", "RRM2",
                  "DHODH", "UMPS", "CAD", "IMPDH1", "IMPDH2",
                  "GPX4", "SLC7A11", "NFE2L2", "KEAP1", "FTH1", "FTL",
                  "SLC3A2", "HMOX1"},
    "Epigenetic": {"EZH2", "EED", "SUZ12", "KDM5A", "KDM5B", "KDM6A",
                   "SMARCA2", "SMARCA4", "SMARCB1", "ARID1A", "ARID1B",
                   "PBRM1", "BRD4", "BRD2", "BRD9",
                   "HDAC1", "HDAC2", "HDAC3", "HDAC6",
                   "DNMT1", "DNMT3A", "DNMT3B", "TET2",
                   "KAT6A", "KAT6B", "CREBBP", "EP300"},
    "Cell cycle": {"CDK1", "CDK2", "CDK4", "CDK6", "CDK7", "CDK9", "CDK12",
                   "CDC7", "DBF4", "WEE1", "PLK1", "PLK4", "AURKB", "AURKA",
                   "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
                   "ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
                   "CDC6", "CDT1", "CCNE1", "CCND1"},
    "Immune/signaling": {"CGAS", "STING1", "TBK1", "IRF3", "IFNAR1", "IFNAR2",
                         "NFKB1", "NFKB2", "RELA", "RELB", "IKBKB", "IKBKG",
                         "JAK1", "JAK2", "STAT1", "STAT2"},
}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD): group1 - group2."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_sd)


def fdr_correction(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    fdr = pvals * n / ranks
    sorted_idx = np.argsort(pvals)[::-1]
    sorted_fdr = fdr[sorted_idx]
    for i in range(1, len(sorted_fdr)):
        sorted_fdr[i] = min(sorted_fdr[i], sorted_fdr[i - 1])
    fdr[sorted_idx] = sorted_fdr
    return np.minimum(fdr, 1.0)


def get_pathway(gene: str) -> str:
    """Return pathway category for a gene, or 'Other'."""
    for pathway, genes in PATHWAY_GENES.items():
        if gene in genes:
            return pathway
    if gene in HR_PATHWAY_GENES:
        return "DNA repair (HR)"
    return "Other"


def screen_cancer_type(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    cancer_type: str,
    brca_group: str,
) -> list[dict]:
    """Run genome-wide differential dependency for one cancer type + BRCA group."""
    common = classified.index.intersection(crispr.index)
    merged = classified.loc[common]

    ct_mask = merged["OncotreeLineage"] == cancer_type

    if brca_group == "any_brca":
        def_mask = merged["brca_combined_status"] == "deficient"
    elif brca_group == "brca1_only":
        def_mask = merged["brca1_status"] == "deficient"
    elif brca_group == "brca2_only":
        def_mask = merged["brca2_status"] == "deficient"
    else:
        raise ValueError(f"Unknown brca_group: {brca_group}")

    prof_mask = merged["brca_combined_status"] == "proficient"

    def_ids = merged[ct_mask & def_mask].index.intersection(crispr.index)
    prof_ids = merged[ct_mask & prof_mask].index.intersection(crispr.index)

    if len(def_ids) < MIN_DEF or len(prof_ids) < MIN_PROF:
        return []

    rows = []
    pvals = []

    for gene in crispr.columns:
        def_vals = crispr.loc[def_ids, gene].dropna().values
        prof_vals = crispr.loc[prof_ids, gene].dropna().values

        if len(def_vals) < MIN_DEF or len(prof_vals) < MIN_PROF:
            continue

        _, pval = stats.mannwhitneyu(def_vals, prof_vals, alternative="two-sided")
        d = cohens_d(def_vals, prof_vals)

        rows.append({
            "cancer_type": cancer_type,
            "gene": gene,
            "brca_group": brca_group,
            "cohens_d": round(d, 4),
            "pvalue": float(pval),
            "n_mut": len(def_vals),
            "n_wt": len(prof_vals),
            "mean_def": round(float(def_vals.mean()), 4),
            "mean_prof": round(float(prof_vals.mean()), 4),
        })
        pvals.append(pval)

    # FDR per cancer type
    if pvals:
        fdrs = fdr_correction(np.array(pvals))
        for i, row in enumerate(rows):
            row["fdr"] = float(fdrs[i])
    return rows


def classify_brca_specificity(hits: pd.DataFrame) -> pd.DataFrame:
    """Classify hits as BRCA1-specific, BRCA2-specific, or shared."""
    if len(hits) == 0:
        return hits

    b1_genes = set(hits[hits["brca_group"] == "brca1_only"]["gene"])
    b2_genes = set(hits[hits["brca_group"] == "brca2_only"]["gene"])
    shared = b1_genes & b2_genes

    def _classify(row):
        g = row["gene"]
        if g in shared:
            return "shared"
        elif g in b1_genes and row["brca_group"] == "brca1_only":
            return "BRCA1-specific"
        elif g in b2_genes and row["brca_group"] == "brca2_only":
            return "BRCA2-specific"
        else:
            return "any_brca_only"

    hits = hits.copy()
    hits["specificity"] = hits.apply(_classify, axis=1)
    return hits


def shld_stratify_hits(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    top_genes: list[str],
) -> pd.DataFrame:
    """For top SL hits, test if dependency persists in 53BP1/SHLD-lost context."""
    common = classified.index.intersection(crispr.index)
    merged = classified.loc[common]
    brca_def = merged[merged["brca_combined_status"] == "deficient"]

    intact_ids = brca_def[brca_def["shld_complex_status"] == "SHLD-intact"].index
    lost_ids = brca_def[brca_def["shld_complex_status"] == "SHLD-lost"].index

    intact_ids = intact_ids.intersection(crispr.index)
    lost_ids = lost_ids.intersection(crispr.index)

    if len(intact_ids) < 2 or len(lost_ids) < 2:
        return pd.DataFrame()

    rows = []
    for gene in top_genes:
        if gene not in crispr.columns:
            continue
        intact_vals = crispr.loc[intact_ids, gene].dropna().values
        lost_vals = crispr.loc[lost_ids, gene].dropna().values

        if len(intact_vals) < 2 or len(lost_vals) < 2:
            continue

        _, pval = stats.mannwhitneyu(intact_vals, lost_vals, alternative="two-sided")
        d = cohens_d(lost_vals, intact_vals)  # lost - intact

        rows.append({
            "gene": gene,
            "comparison": "SHLD-lost vs SHLD-intact (BRCA-deficient only)",
            "cohens_d_lost_vs_intact": round(d, 4),
            "pvalue": pval,
            "n_shld_intact": len(intact_vals),
            "n_shld_lost": len(lost_vals),
            "mean_intact": round(float(intact_vals.mean()), 4),
            "mean_lost": round(float(lost_vals.mean()), 4),
            "retains_efficacy_in_shld_lost": bool(lost_vals.mean() < -0.3),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
    return result


def plot_volcano(
    results_ct: pd.DataFrame,
    cancer_type: str,
    brca_group: str,
    out_dir: Path,
) -> None:
    """Volcano plot: effect size vs -log10(FDR)."""
    if len(results_ct) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x = results_ct["cohens_d"].values
    y = -np.log10(results_ct["fdr"].values.clip(min=1e-50))

    sig = (results_ct["fdr"] < FDR_THRESHOLD) & (results_ct["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)
    ax.scatter(x[sig], y[sig], c="#D95319", s=15, alpha=0.8)

    # Label top SL hits (most negative d)
    top = results_ct[sig].nsmallest(10, "cohens_d")
    for _, row in top.iterrows():
        ax.annotate(
            row["gene"], (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
            fontsize=7, ha="right",
        )

    # Highlight priority targets
    for gene in [POLB_GENE] + CGAS_STING_GENES:
        match = results_ct[results_ct["gene"] == gene]
        if len(match) > 0:
            row = match.iloc[0]
            ax.scatter(
                [row["cohens_d"]],
                [-np.log10(max(row["fdr"], 1e-50))],
                c="#1E88E5", s=40, marker="D", zorder=5, edgecolors="black",
            )
            ax.annotate(
                gene, (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                fontsize=7, ha="left", color="#1E88E5", fontweight="bold",
            )

    ax.axhline(-np.log10(FDR_THRESHOLD), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (negative = more essential in BRCA-deficient)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Genome-wide SL Screen: {cancer_type} ({brca_group})")

    fig.tight_layout()
    safe = f"{cancer_type}_{brca_group}".replace("/", "_").replace(" ", "_")
    fig.savefig(out_dir / f"volcano_{safe}.png", dpi=150)
    plt.close(fig)


def plot_pan_cancer_heatmap(hits: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of top SL genes across cancer types."""
    if len(hits) == 0:
        return

    # Use any_brca group for heatmap
    any_brca = hits[hits["brca_group"] == "any_brca"]
    if len(any_brca) == 0:
        any_brca = hits

    # Get top genes by frequency across cancer types
    gene_counts = any_brca.groupby("gene")["cancer_type"].nunique().sort_values(ascending=False)
    top_genes = gene_counts.head(30).index.tolist()

    if not top_genes:
        return

    # Pivot for heatmap
    pivot = any_brca[any_brca["gene"].isin(top_genes)].pivot_table(
        index="gene", columns="cancer_type", values="cohens_d", aggfunc="first"
    )
    pivot = pivot.reindex(top_genes)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.6)
    ax.set_title("Pan-Cancer SL Heatmap: Top Hits (any BRCA-deficient)")
    fig.tight_layout()
    fig.savefig(output_dir / "pan_cancer_sl_heatmap.png", dpi=150)
    plt.close(fig)


def plot_pathway_enrichment(hits: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of pathway distribution among SL hits."""
    if len(hits) == 0:
        return

    pathway_counts = hits.groupby("pathway_category")["gene"].nunique().sort_values(ascending=True)
    # Remove 'Other' if present for cleaner plot, but keep if it's the only category
    if len(pathway_counts) > 1 and "Other" in pathway_counts.index:
        other_count = pathway_counts["Other"]
        pathway_counts = pathway_counts.drop("Other")
    else:
        other_count = 0

    fig, ax = plt.subplots(figsize=(8, max(4, len(pathway_counts) * 0.4)))
    colors = plt.cm.Set2(np.linspace(0, 1, len(pathway_counts)))
    bars = ax.barh(range(len(pathway_counts)), pathway_counts.values, color=colors, alpha=0.85)

    ax.set_yticks(range(len(pathway_counts)))
    ax.set_yticklabels(pathway_counts.index, fontsize=8)
    ax.set_xlabel("Number of unique SL genes")
    ax.set_title(f"Pathway Distribution of SL Hits\n(+{other_count} genes in 'Other')")

    for bar, val in zip(bars, pathway_counts.values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "pathway_enrichment.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    volcano_dir = OUTPUT_DIR / "volcano_plots"
    volcano_dir.mkdir(exist_ok=True)

    print("=== Phase 3: Genome-Wide Non-HR Dependency Screen ===\n")

    # --- Load data ---
    print("Loading Phase 1 classifier output...")
    classified = pd.read_csv(PHASE1_DIR / "brca_classified_lines.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "cancer_type_summary.csv")
    print(f"  {len(classified)} classified lines")

    qualifying = summary[
        summary["qualifies_primary"] | summary["qualifies_exploratory"]
    ]["cancer_type"].tolist()
    print(f"  {len(qualifying)} qualifying cancer types: {qualifying}")

    print("\nLoading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {crispr.shape[0]} cell lines, {crispr.shape[1]} genes")

    # --- Genome-wide screen per cancer type per BRCA group ---
    print("\n--- Genome-wide screen ---")
    all_rows: list[dict] = []

    for brca_group in BRCA_GROUPS:
        print(f"\n  BRCA group: {brca_group}")
        for cancer_type in qualifying:
            rows = screen_cancer_type(classified, crispr, cancer_type, brca_group)
            if rows:
                n_sig = sum(1 for r in rows if r.get("fdr", 1) < FDR_THRESHOLD
                            and abs(r["cohens_d"]) > EFFECT_SIZE_THRESHOLD)
                print(f"    {cancer_type}: {len(rows)} genes tested, {n_sig} significant")
                all_rows.extend(rows)
            else:
                print(f"    {cancer_type}: insufficient samples for {brca_group}")

    all_results = pd.DataFrame(all_rows)
    print(f"\n  Total tests: {len(all_results)}")

    # --- Filter significant SL hits ---
    sig_hits = all_results[
        (all_results["fdr"] < FDR_THRESHOLD)
        & (all_results["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ].copy()

    sl_hits = sig_hits[sig_hits["cohens_d"] < 0].copy()
    print(f"  Significant SL hits (FDR<0.05, d<-0.3): {len(sl_hits)}")
    if len(sl_hits) > 0:
        print(f"  Unique SL genes: {sl_hits['gene'].nunique()}")

    # Nominal fallback if FDR too stringent for small samples
    nominal_hits = all_results[
        (all_results["pvalue"] < 0.05)
        & (all_results["cohens_d"] < -EFFECT_SIZE_THRESHOLD)
    ].copy()
    print(f"  Nominal SL hits (p<0.05, d<-0.3): {len(nominal_hits)}")

    use_nominal = False
    if len(sl_hits) == 0 and len(nominal_hits) > 0:
        print("  NOTE: Using nominal hits (FDR too conservative for sample sizes)")
        sl_hits = nominal_hits
        use_nominal = True

    # Add pathway categories
    if len(sl_hits) > 0:
        sl_hits["pathway_category"] = sl_hits["gene"].apply(get_pathway)
        sl_hits = classify_brca_specificity(sl_hits)
        sl_hits = sl_hits.sort_values("cohens_d").reset_index(drop=True)

    sl_hits.to_csv(OUTPUT_DIR / "genomewide_sl_hits.csv", index=False)
    print(f"\n  Saved genomewide_sl_hits.csv ({len(sl_hits)} rows)")

    # --- Novel non-HR hits ---
    print("\n--- Novel Non-HR Hits ---")
    if len(sl_hits) > 0:
        novel = sl_hits[~sl_hits["gene"].isin(HR_PATHWAY_GENES)].copy()
        if len(novel) > 0:
            novel_summary = (
                novel.groupby("gene")
                .agg(
                    n_cancer_types=("cancer_type", "nunique"),
                    n_brca_groups=("brca_group", "nunique"),
                    mean_cohens_d=("cohens_d", "mean"),
                    min_fdr=("fdr", "min") if "fdr" in novel.columns else ("pvalue", "min"),
                    cancer_types=("cancer_type", lambda x: ";".join(sorted(set(x)))),
                    pathway=("pathway_category", "first"),
                )
                .sort_values(["n_cancer_types", "mean_cohens_d"], ascending=[False, True])
                .reset_index()
            )
        else:
            novel_summary = pd.DataFrame()
    else:
        novel_summary = pd.DataFrame()

    novel_summary.to_csv(OUTPUT_DIR / "novel_nonhr_hits_summary.csv", index=False)
    print(f"  {len(novel_summary)} unique novel non-HR genes")
    for _, row in novel_summary.head(15).iterrows():
        print(f"    {row['gene']}: d={row['mean_cohens_d']:.3f}, "
              f"{row['n_cancer_types']} types, [{row['pathway']}]")

    # --- BRCA1 vs BRCA2 specific hits ---
    print("\n--- BRCA1 vs BRCA2 Specificity ---")
    if len(sl_hits) > 0 and "specificity" in sl_hits.columns:
        spec_summary = (
            sl_hits[sl_hits["specificity"].isin(["BRCA1-specific", "BRCA2-specific", "shared"])]
            .groupby(["gene", "specificity"])
            .agg(
                mean_d=("cohens_d", "mean"),
                cancer_types=("cancer_type", lambda x: ";".join(sorted(set(x)))),
            )
            .reset_index()
            .sort_values("mean_d")
        )
        spec_summary.to_csv(OUTPUT_DIR / "brca1_vs_brca2_specific_hits.csv", index=False)
        print(f"  {len(spec_summary)} gene-specificity pairs")

        for spec in ["BRCA1-specific", "BRCA2-specific", "shared"]:
            subset = spec_summary[spec_summary["specificity"] == spec]
            print(f"  {spec}: {len(subset)} genes")
            for _, row in subset.head(5).iterrows():
                print(f"    {row['gene']}: d={row['mean_d']:.3f}")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "brca1_vs_brca2_specific_hits.csv", index=False)

    # --- Priority targets: POLB and cGAS-STING ---
    print("\n--- Priority Targets: POLB & cGAS-STING ---")
    priority_genes = [POLB_GENE] + CGAS_STING_GENES
    priority_rows = all_results[all_results["gene"].isin(priority_genes)].copy()
    if len(priority_rows) > 0:
        priority_rows["pathway_category"] = priority_rows["gene"].apply(get_pathway)
        priority_rows = priority_rows.sort_values(["gene", "cancer_type", "brca_group"])

    priority_rows.to_csv(OUTPUT_DIR / "priority_targets_polb_cgasSting.csv", index=False)
    print(f"  {len(priority_rows)} results for priority targets")

    for gene in priority_genes:
        g_data = priority_rows[priority_rows["gene"] == gene]
        if len(g_data) == 0:
            print(f"  {gene}: not found in CRISPR data")
            continue
        sig_g = g_data[(g_data["fdr"] < FDR_THRESHOLD) & (g_data["cohens_d"] < -EFFECT_SIZE_THRESHOLD)]
        if len(sig_g) > 0:
            best = sig_g.loc[sig_g["cohens_d"].idxmin()]
            print(f"  {gene}: SIGNIFICANT SL — best d={best['cohens_d']:.3f} "
                  f"FDR={best['fdr']:.4f} ({best['cancer_type']}, {best['brca_group']})")
        else:
            best = g_data.loc[g_data["cohens_d"].idxmin()]
            print(f"  {gene}: not significant — best d={best['cohens_d']:.3f} "
                  f"p={best['pvalue']:.4f} ({best['cancer_type']}, {best['brca_group']})")

    # --- 53BP1/SHLD stratification ---
    print("\n--- 53BP1/SHLD Stratification of Top Hits ---")
    if len(sl_hits) > 0:
        top_sl_genes = sl_hits.groupby("gene")["cohens_d"].mean().nsmallest(30).index.tolist()
        shld_results = shld_stratify_hits(classified, crispr, top_sl_genes)
    else:
        shld_results = pd.DataFrame()

    shld_results.to_csv(OUTPUT_DIR / "shld_stratified_hits.csv", index=False)
    if len(shld_results) > 0:
        retains = shld_results[shld_results["retains_efficacy_in_shld_lost"]]
        print(f"  {len(shld_results)} genes tested, {len(retains)} retain efficacy in SHLD-lost")
        for _, row in retains.head(10).iterrows():
            print(f"    {row['gene']}: mean_lost={row['mean_lost']:.3f}, "
                  f"d(lost-intact)={row['cohens_d_lost_vs_intact']:.3f}")
    else:
        print("  Insufficient SHLD-lost lines for stratification")

    # --- Volcano plots ---
    print("\nGenerating volcano plots...")
    n_plots = 0
    for brca_group in BRCA_GROUPS:
        for cancer_type in qualifying:
            ct_results = all_results[
                (all_results["cancer_type"] == cancer_type)
                & (all_results["brca_group"] == brca_group)
            ]
            if len(ct_results) > 0:
                plot_volcano(ct_results, cancer_type, brca_group, volcano_dir)
                n_plots += 1
    print(f"  Saved {n_plots} volcano plots")

    # --- Pan-cancer heatmap ---
    print("Generating pan-cancer SL heatmap...")
    plot_pan_cancer_heatmap(sl_hits, OUTPUT_DIR)

    # --- Pathway enrichment ---
    print("Generating pathway enrichment plot...")
    plot_pathway_enrichment(sl_hits, OUTPUT_DIR)

    # --- Summary ---
    n_novel = len(novel_summary) if len(novel_summary) > 0 else 0
    summary_data = {
        "total_tests": len(all_results),
        "significant_sl_hits": len(sl_hits),
        "unique_sl_genes": int(sl_hits["gene"].nunique()) if len(sl_hits) > 0 else 0,
        "novel_nonhr_genes": n_novel,
        "used_nominal_threshold": use_nominal,
        "thresholds": {
            "fdr": FDR_THRESHOLD,
            "effect_size": EFFECT_SIZE_THRESHOLD,
        },
        "qualifying_cancer_types": qualifying,
        "brca_groups_tested": BRCA_GROUPS,
        "success_criterion_met": n_novel >= 5,
        "priority_targets": {},
        "shld_stratification": {
            "genes_tested": len(shld_results),
            "genes_retaining_efficacy": int(shld_results["retains_efficacy_in_shld_lost"].sum())
            if len(shld_results) > 0 else 0,
        },
    }

    for gene in priority_genes:
        g_data = all_results[all_results["gene"] == gene]
        if len(g_data) > 0:
            best = g_data.loc[g_data["cohens_d"].idxmin()]
            sig = bool(best["fdr"] < FDR_THRESHOLD and best["cohens_d"] < -EFFECT_SIZE_THRESHOLD)
            summary_data["priority_targets"][gene] = {
                "best_d": float(best["cohens_d"]),
                "best_fdr": float(best["fdr"]),
                "significant": sig,
                "cancer_type": best["cancer_type"],
                "brca_group": best["brca_group"],
            }

    with open(OUTPUT_DIR / "phase3_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n=== Phase 3 Complete ===")
    print(f"  {n_novel} novel non-HR dependencies (criterion: >=5)")
    print(f"  Success criterion met: {summary_data['success_criterion_met']}")


if __name__ == "__main__":
    main()
