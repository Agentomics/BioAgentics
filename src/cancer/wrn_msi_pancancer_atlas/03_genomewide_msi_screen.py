"""Phase 3: Genome-wide MSI-H dependency screen per cancer type.

For each qualifying cancer type and pan-cancer pooled, runs differential
dependency analysis (MSI-H vs MSS) across all ~18,000 genes in
CRISPRGeneEffect. Identifies MSI-H-specific gained and lost dependencies.

Includes Fisher enrichment against Vermeulen SL benchmark and
WRN anti-correlated dependency analysis.

Usage:
    uv run python -m wrn_msi_pancancer_atlas.03_genomewide_msi_screen
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix
from cancer.wrn_msi_pancancer_atlas.stats_utils import cohens_d, fdr_correction

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "output" / "wrn-msi-pancancer-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "wrn-msi-pancancer-atlas" / "phase3"
BENCHMARK_DIR = REPO_ROOT / "data" / "benchmarks"

# Significance thresholds
FDR_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.5

# Known MSI-related genes for annotation
KNOWN_MSI_TARGETS = {
    "WRN", "WRNIP1", "MLH1", "MSH2", "MSH6", "PMS2",
    "RFC2", "RFC3", "RFC4", "RFC5",  # Replication fork
    "RPA1", "RPA2", "RPA3",  # ssDNA binding
}

# DNA repair pathway genes for pathway context
DDR_GENES = {
    "ATR", "ATRIP", "CHEK1", "WEE1", "PARP1", "PARP2",
    "BRCA1", "BRCA2", "RAD51", "PALB2", "FANCD2",
    "BLM", "RECQL", "WRN", "TOP1", "TOP2A",
}

MIN_SAMPLES = 3


def screen_one_context(
    msi_h_data: pd.DataFrame,
    mss_data: pd.DataFrame,
    crispr_cols: list[str],
    context_name: str,
) -> list[dict]:
    """Run genome-wide differential dependency for one context."""
    rows = []
    pvals = []

    for gene in crispr_cols:
        msi_vals = msi_h_data[gene].dropna().values
        mss_vals = mss_data[gene].dropna().values

        if len(msi_vals) < MIN_SAMPLES or len(mss_vals) < MIN_SAMPLES:
            continue

        _, pval = stats.mannwhitneyu(msi_vals, mss_vals, alternative="two-sided")
        d = cohens_d(msi_vals, mss_vals)

        rows.append({
            "cancer_type": context_name,
            "gene": gene,
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_msi": len(msi_vals),
            "n_mss": len(mss_vals),
            "median_dep_msi": round(float(np.median(msi_vals)), 4),
            "median_dep_mss": round(float(np.median(mss_vals)), 4),
        })
        pvals.append(pval)

    if pvals:
        fdrs = fdr_correction(np.array(pvals))
        for i, row in enumerate(rows):
            row["fdr"] = float(fdrs[i])

    return rows


def load_vermeulen_msi_sl(benchmark_dir: Path) -> set[str]:
    """Load genes with SL interactions with MMR genes from Vermeulen benchmark."""
    verm_path = benchmark_dir / "vermeulen_sl" / "vermeulen_sl_interactions.tsv"
    if not verm_path.exists():
        return set()
    verm = pd.read_csv(verm_path, sep="\t")
    mmr_genes = {"MLH1", "MSH2", "MSH6", "PMS2", "WRN"}
    # Get targets where source is an MMR gene
    mmr_sl = verm[verm["source"].isin(mmr_genes)]
    return set(mmr_sl["target"].unique())


def fisher_enrichment(
    hit_genes: set[str],
    benchmark_genes: set[str],
    universe_genes: set[str],
) -> dict:
    """Fisher exact test for enrichment of benchmark genes in hit set."""
    a = len(hit_genes & benchmark_genes)
    b = len(hit_genes - benchmark_genes)
    c = len(benchmark_genes - hit_genes)
    d = len(universe_genes - hit_genes - benchmark_genes)

    _, pval = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    total_hits = len(hit_genes)
    expected = len(benchmark_genes) * total_hits / len(universe_genes) if len(universe_genes) > 0 else 0

    return {
        "observed_overlap": a,
        "expected_overlap": round(expected, 2),
        "fold_enrichment": round(a / expected, 2) if expected > 0 else float("inf"),
        "fisher_p": float(pval),
        "n_hits": total_hits,
        "n_benchmark": len(benchmark_genes),
        "n_universe": len(universe_genes),
    }


def wrn_anticorrelation(
    msi_h_data: pd.DataFrame,
    crispr_cols: list[str],
) -> pd.DataFrame:
    """Find genes whose dependency anti-correlates with WRN in MSI-H lines."""
    if "WRN" not in msi_h_data.columns:
        return pd.DataFrame()

    wrn_dep = msi_h_data["WRN"].dropna()
    valid_lines = wrn_dep.index

    rows = []
    for gene in crispr_cols:
        if gene == "WRN":
            continue
        gene_dep = msi_h_data.loc[valid_lines, gene].dropna()
        common = wrn_dep.index.intersection(gene_dep.index)
        if len(common) < 5:
            continue
        r, p = stats.spearmanr(wrn_dep.loc[common], gene_dep.loc[common])
        rows.append({"gene": gene, "spearman_r": round(float(r), 4), "pvalue": float(p)})

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
        result = result.sort_values("spearman_r").reset_index(drop=True)
    return result


def plot_volcano(results_ct: pd.DataFrame, context_name: str, out_dir: Path) -> None:
    """Volcano plot for one context."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x = results_ct["cohens_d"].values
    y = -np.log10(results_ct["fdr"].values.clip(min=1e-50))

    sig = (results_ct["fdr"] < FDR_THRESHOLD) & (results_ct["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)

    gained = sig & (results_ct["cohens_d"] < 0)
    lost = sig & (results_ct["cohens_d"] > 0)
    ax.scatter(x[gained], y[gained], c="#D95319", s=15, alpha=0.8, label="Gained dep. in MSI-H")
    ax.scatter(x[lost], y[lost], c="#4DBEEE", s=15, alpha=0.8, label="Lost dep. in MSI-H")

    # Label top gained hits
    top = results_ct[gained].nsmallest(10, "cohens_d")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                     fontsize=7, ha="right")

    ax.axhline(-np.log10(FDR_THRESHOLD), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (MSI-H vs MSS)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Genome-wide MSI-H Dependency Screen: {context_name}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    safe = context_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"volcano_{safe}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3: Genome-Wide MSI-H Dependency Screen ===\n")

    # Load Phase 1
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "msi_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(qualifying_types)} qualifying cancer types: {qualifying_types}")

    # Load CRISPR
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    crispr_cols = list(crispr.columns)
    print(f"  {len(crispr_cols)} genes")

    merged = classified.join(crispr, how="inner")

    # Screen each cancer type + pan-cancer
    all_rows = []
    contexts = qualifying_types + ["Pan-cancer (pooled)"]

    for context in contexts:
        if context == "Pan-cancer (pooled)":
            ct_data = merged
        else:
            ct_data = merged[merged["OncotreeLineage"] == context]

        msi_h = ct_data[ct_data["msi_status"] == "MSI-H"]
        mss = ct_data[ct_data["msi_status"] == "MSS"]
        print(f"  Screening {context} ({len(msi_h)} MSI-H, {len(mss)} MSS)...")

        rows = screen_one_context(msi_h, mss, crispr_cols, context)
        all_rows.extend(rows)

    all_results = pd.DataFrame(all_rows)
    print(f"\n  Total tests: {len(all_results)}")

    # Save per-context full results
    for context in contexts:
        ct_res = all_results[all_results["cancer_type"] == context]
        safe = context.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        ct_res.to_csv(OUTPUT_DIR / f"genomewide_msi_dependencies_{safe}.csv", index=False)

    # Filter significant hits
    sig_hits = all_results[
        (all_results["fdr"] < FDR_THRESHOLD) & (all_results["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ].copy()

    # Gained dependency in MSI-H (d < 0 = more essential)
    gained = sig_hits[sig_hits["cohens_d"] < 0].sort_values("cohens_d").reset_index(drop=True)
    # Lost dependency (d > 0 = less essential in MSI-H)
    lost = sig_hits[sig_hits["cohens_d"] > 0].sort_values("cohens_d", ascending=False).reset_index(drop=True)

    # Also collect nominal hits (p < 0.05) for contexts with small N
    nominal = all_results[
        (all_results["p_value"] < 0.05) & (all_results["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ].copy()

    # Use nominal if FDR yields too few
    use_nominal = len(gained) < 5
    if use_nominal:
        print("  NOTE: Using nominal hits (p<0.05) — FDR too conservative for small N")
        gained = nominal[nominal["cohens_d"] < 0].sort_values("cohens_d").reset_index(drop=True)

    print(f"  Gained dependencies (FDR<0.05, d<-0.5): {len(sig_hits[sig_hits['cohens_d'] < 0])}")
    print(f"  Lost dependencies (FDR<0.05, d>0.5): {len(sig_hits[sig_hits['cohens_d'] > 0])}")
    print(f"  Nominal gained (p<0.05, d<-0.5): {len(nominal[nominal['cohens_d'] < 0])}")

    # Summary of top hits
    gained.to_csv(OUTPUT_DIR / "msi_specific_hits_summary.csv", index=False)

    print(f"\nTop MSI-H-specific gained dependencies:")
    for _, row in gained.head(20).iterrows():
        label = ""
        if row["gene"] in KNOWN_MSI_TARGETS:
            label = " [known MSI target]"
        elif row["gene"] in DDR_GENES:
            label = " [DDR]"
        print(f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
              f"FDR={row.get('fdr', float('nan')):.3e}{label}")

    # Fisher enrichment against Vermeulen benchmark
    print("\nFisher enrichment vs Vermeulen SL benchmark...")
    verm_genes = load_vermeulen_msi_sl(BENCHMARK_DIR)
    if verm_genes:
        hit_genes = set(gained["gene"].unique())
        universe = set(all_results["gene"].unique())
        enrichment = fisher_enrichment(hit_genes, verm_genes, universe)
        print(f"  Overlap: {enrichment['observed_overlap']} / {enrichment['n_hits']} hits")
        print(f"  Expected: {enrichment['expected_overlap']}")
        print(f"  Fold enrichment: {enrichment['fold_enrichment']}x, p={enrichment['fisher_p']:.3e}")

        enrich_df = pd.DataFrame([{"benchmark": "Vermeulen_MMR_SL", **enrichment}])
        enrich_df.to_csv(OUTPUT_DIR / "sl_benchmark_enrichment.csv", index=False)
    else:
        print("  Vermeulen benchmark not available for MMR genes")

    # WRN anti-correlated dependencies in MSI-H
    print("\nWRN anti-correlated dependencies in MSI-H lines...")
    msi_h_all = merged[merged["msi_status"] == "MSI-H"]
    anticorr = wrn_anticorrelation(msi_h_all, crispr_cols)
    if len(anticorr) > 0:
        anticorr.to_csv(OUTPUT_DIR / "wrn_anticorrelated_dependencies.csv", index=False)
        sig_anti = anticorr[anticorr["fdr"] < 0.05]
        print(f"  {len(sig_anti)} genes with significant anti-correlation (FDR<0.05)")
        for _, row in sig_anti.head(10).iterrows():
            print(f"    {row['gene']}: r={row['spearman_r']:.3f}, FDR={row['fdr']:.3e}")

    # Volcano plots
    print("\nGenerating volcano plots...")
    for context in contexts:
        ct_res = all_results[all_results["cancer_type"] == context]
        if "fdr" in ct_res.columns:
            plot_volcano(ct_res, context, OUTPUT_DIR)
    print(f"  Saved {len(contexts)} volcano plots")

    # Summary text
    summary_lines = [
        "=" * 60,
        "WRN-MSI Pan-Cancer Atlas — Phase 3: Genome-Wide MSI-H Screen",
        "=" * 60,
        "",
        f"Total tests: {len(all_results)}",
        f"Gained dependencies (FDR<0.05, d<-0.5): {len(sig_hits[sig_hits['cohens_d'] < 0])}",
        f"Lost dependencies (FDR<0.05, d>0.5): {len(sig_hits[sig_hits['cohens_d'] > 0])}",
        "",
        "TOP GAINED DEPENDENCIES IN MSI-H",
        "-" * 50,
    ]
    for _, row in gained.head(30).iterrows():
        summary_lines.append(
            f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
            f"FDR={row.get('fdr', float('nan')):.3e}"
        )
    summary_lines.append("")

    with open(OUTPUT_DIR / "genomewide_msi_screen_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  genomewide_msi_screen_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
