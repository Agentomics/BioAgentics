"""Phase 1: LDSC cross-trait genetic correlations — TS vs 16 autoimmune diseases.

Computes bivariate LDSC genetic correlations between Yu et al. 2019 PGC TS GWAS
and 16 autoimmune/inflammatory disease GWAS. Applies FDR correction and generates
a forest plot and heatmap.

Uses the built-in LDSC pipeline (src/bioagentics/pipelines/ldsc_correlation/).

Input:
    - data/tourettes/ts-neuroimmune-subtyping/ts_gwas_2019/ts_yu_2019.ldsc.tsv.gz
    - data/tourettes/ts-neuroimmune-subtyping/autoimmune_gwas/ldsc/*.ldsc.tsv.gz
    - data/tourettes/ts-comorbidity-genetic-architecture/reference/LDscore/

Output:
    - output/tourettes/ts-neuroimmune-subtyping/phase1_ldsc/

Usage:
    uv run python src/tourettes/ts_neuroimmune_subtyping/01_ldsc_genetic_correlations.py
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.pipelines.ldsc_correlation.pipeline import (
    ldsc_regression,
    load_ld_scores,
    load_sumstats,
    munge_sumstats,
)

logger = logging.getLogger(__name__)

# Paths
TS_LDSC = REPO_ROOT / "data" / "tourettes" / "ts-neuroimmune-subtyping" / "ts_gwas_2019" / "ts_yu_2019.ldsc.tsv.gz"
AUTOIMMUNE_DIR = REPO_ROOT / "data" / "tourettes" / "ts-neuroimmune-subtyping" / "autoimmune_gwas"
LDSC_DIR = AUTOIMMUNE_DIR / "ldsc"
LD_SCORES_DIR = REPO_ROOT / "data" / "tourettes" / "ts-comorbidity-genetic-architecture" / "reference" / "LDscore"
MANIFEST_PATH = AUTOIMMUNE_DIR / "manifest.json"
OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-neuroimmune-subtyping" / "phase1_ldsc"

# TS study metadata
TS_N_CASES = 4819
TS_N_CONTROLS = 9488
TS_N_TOTAL = TS_N_CASES + TS_N_CONTROLS


def load_manifest() -> dict:
    """Load the autoimmune GWAS manifest for disease metadata."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def fdr_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            sorted_p[i] * n / (i + 1),
            adjusted[sorted_idx[i + 1]],
        )
    return np.clip(adjusted, 0, 1)


def run_phase1() -> pd.DataFrame:
    """Run LDSC genetic correlations between TS and all autoimmune diseases."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load LD scores
    logger.info("Loading LD scores from %s", LD_SCORES_DIR)
    ld_scores = load_ld_scores(LD_SCORES_DIR)

    # Load TS summary stats
    logger.info("Loading TS GWAS: %s", TS_LDSC.name)
    ts_df = munge_sumstats(load_sumstats(TS_LDSC), "TS")
    logger.info("TS GWAS: %d SNPs after munging", len(ts_df))

    # Load manifest for disease metadata
    manifest = load_manifest()
    disease_meta = {s["abbreviation"]: s for s in manifest["studies"]}

    # Discover autoimmune LDSC files
    ldsc_files = sorted(LDSC_DIR.glob("*.ldsc.tsv.gz"))
    logger.info("Found %d autoimmune LDSC files", len(ldsc_files))

    results: list[dict] = []

    for ldsc_file in ldsc_files:
        # Extract abbreviation from filename (e.g., "ra_ishigaki_2022.ldsc.tsv.gz" -> "ra")
        abbrev = ldsc_file.name.split("_")[0]
        meta = disease_meta.get(abbrev, {})
        disease_name = meta.get("disease", abbrev.upper())

        logger.info("Computing rg(TS, %s [%s])...", abbrev.upper(), disease_name)

        trait_df = munge_sumstats(load_sumstats(ldsc_file), abbrev)
        result = ldsc_regression(ts_df, trait_df, ld_scores, "TS", abbrev)

        results.append({
            "disease": disease_name,
            "abbreviation": abbrev.upper(),
            "rg": result.rg,
            "rg_se": result.rg_se,
            "p_value": result.p_value,
            "h2_ts": result.h2_trait1,
            "h2_ts_se": result.h2_trait1_se,
            "h2_disease": result.h2_trait2,
            "h2_disease_se": result.h2_trait2_se,
            "gcov_intercept": result.gcov_int,
            "n_snps": result.n_snps,
            "disease_n": meta.get("sample_size", 0),
            "disease_n_cases": meta.get("n_cases", 0),
            "gwas_id": meta.get("gwas_catalog_id", ""),
            "first_author": meta.get("first_author", ""),
            "year": meta.get("year", 0),
        })

        logger.info(
            "  rg=%.4f (SE=%.4f, p=%.2e, n_snps=%d)",
            result.rg, result.rg_se, result.p_value, result.n_snps,
        )

    df = pd.DataFrame(results)

    # FDR correction
    valid_mask = np.isfinite(df["p_value"].values)
    fdr_values = np.full(len(df), np.nan)
    if valid_mask.any():
        fdr_values[valid_mask] = fdr_correction(df.loc[valid_mask, "p_value"].values)
    df["fdr_q"] = fdr_values
    df["significant_fdr05"] = df["fdr_q"] < 0.05

    # Sort by p-value
    df = df.sort_values("p_value").reset_index(drop=True)

    # Save results
    out_path = OUTPUT_DIR / "ts_autoimmune_genetic_correlations.tsv"
    df.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    logger.info("Wrote %d correlations to %s", len(df), out_path)

    # Write summary JSON
    n_sig = int(df["significant_fdr05"].sum())
    summary = {
        "phase": "Phase 1: LDSC Cross-Trait Genetic Correlations",
        "reference_gwas": "Yu et al. 2019 PGC TS GWAS (GCST007277)",
        "ts_sample": {"n_cases": TS_N_CASES, "n_controls": TS_N_CONTROLS, "n_total": TS_N_TOTAL},
        "n_diseases_tested": len(df),
        "n_significant_fdr05": n_sig,
        "significant_diseases": df[df["significant_fdr05"]][["disease", "abbreviation", "rg", "rg_se", "p_value", "fdr_q"]].to_dict("records"),
        "power_note": "2019 GWAS (N=14,307) has lower power than 2024 TSAICG (N=19,138). Null results may become significant with larger sample.",
    }
    with open(OUTPUT_DIR / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Generate plots
    try:
        _generate_plots(df)
    except Exception as e:
        logger.warning("Plot generation failed (non-critical): %s", e)

    return df


def _generate_plots(df: pd.DataFrame) -> None:
    """Generate forest plot and heatmap of genetic correlations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Forest plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    df_plot = df.sort_values("rg")
    y_pos = range(len(df_plot))

    colors = ["#d62728" if sig else "#1f77b4" for sig in df_plot["significant_fdr05"]]
    ax.barh(y_pos, df_plot["rg"], xerr=1.96 * df_plot["rg_se"],
            color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row.disease} ({row.abbreviation})" for _, row in df_plot.iterrows()], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Genetic Correlation (rg) with 95% CI")
    ax.set_title("TS × Autoimmune Disease Genetic Correlations (LDSC)\nYu et al. 2019 PGC TS GWAS | Red = FDR < 0.05")
    plt.tight_layout()
    fig.savefig(fig_dir / "forest_plot_rg.png", dpi=150)
    plt.close(fig)
    logger.info("Saved forest plot")

    # Heatmap (single row)
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.7), 3))
    rg_vals = df.sort_values("abbreviation")["rg"].values.reshape(1, -1)
    labels = df.sort_values("abbreviation")["abbreviation"].values
    im = ax.imshow(rg_vals, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([0])
    ax.set_yticklabels(["TS"])
    for i, (rg, fdr) in enumerate(zip(df.sort_values("abbreviation")["rg"], df.sort_values("abbreviation")["fdr_q"])):
        marker = "**" if fdr < 0.01 else "*" if fdr < 0.05 else ""
        ax.text(i, 0, f"{rg:.2f}{marker}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Genetic Correlation (rg)")
    ax.set_title("TS × Autoimmune Genetic Correlations (* FDR < 0.05, ** FDR < 0.01)")
    plt.tight_layout()
    fig.savefig(fig_dir / "heatmap_rg.png", dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = run_phase1()

    print("\n=== Phase 1 Results ===")
    print(f"Diseases tested: {len(df)}")
    print(f"Significant (FDR < 0.05): {df['significant_fdr05'].sum()}")
    print("\nTop correlations:")
    for _, row in df.head(5).iterrows():
        sig = " ***" if row["fdr_q"] < 0.05 else ""
        print(f"  {row['abbreviation']:5s} ({row['disease']:35s}): rg={row['rg']:.4f} (SE={row['rg_se']:.4f}, p={row['p_value']:.2e}, FDR={row['fdr_q']:.2e}){sig}")
