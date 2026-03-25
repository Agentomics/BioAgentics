"""Phase 4a: Genome-wide PRISM drug sensitivity analysis for KEAP1/NRF2 cohort.

For each compound in PRISM Repurposing 24Q2 (~6790 treatments), computes
Cohen's d effect size comparing log2-fold-change sensitivity in KEAP1/NRF2-
altered vs WT lines. Applies Welch t-test + Benjamini-Hochberg FDR correction.

Usage:
    uv run python -m keap1_nrf2_pancancer_dependency_atlas.04_prism_drug_sensitivity
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase1"
)
OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase4"
)

# Significance thresholds
FDR_THRESHOLD = 0.1
EFFECT_SIZE_THRESHOLD = 0.3  # |d| > 0.3 per task spec
MIN_SAMPLES = 3


def _extract_compound_key(broad_id: str) -> str:
    """Extract compound key (BRD-XNNNNNNNNN) from a broad_id string."""
    m = re.search(r"BRD-[A-Za-z]\d{8}", str(broad_id))
    return m.group(0) if m else str(broad_id)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    fdr = np.empty(n)
    for i in range(n):
        fdr[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    # Enforce monotonicity
    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted
    return np.minimum(fdr, 1.0)


def build_drug_name_map(meta: pd.DataFrame) -> dict[str, str]:
    """Build compound_key -> name mapping from treatment metadata."""
    key_to_name: dict[str, str] = {}
    for _, row in meta.iterrows():
        bid = row.get("broad_id")
        name = row.get("name")
        if pd.notna(bid) and pd.notna(name):
            key = _extract_compound_key(str(bid))
            # Keep first non-empty name seen for each key
            if key not in key_to_name:
                key_to_name[key] = str(name)
    return key_to_name


def genomewide_drug_screen(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    drug_name_map: dict[str, str],
) -> pd.DataFrame:
    """Screen all PRISM drugs for KEAP1/NRF2-selective sensitivity (pan-cancer).

    Uses Welch t-test for p-values and Cohen's d for effect sizes.
    Negative d = more sensitive in altered lines (potential therapeutic).
    """
    altered_lines = classified[classified["KEAP1_NRF2_altered"] == True].index  # noqa: E712
    wt_lines = classified[classified["pathway_status"] == "WT"].index

    rows = []
    pvals = []

    for treatment_id in prism.index:
        drug_sens = prism.loc[treatment_id]
        alt_vals = drug_sens.reindex(altered_lines).dropna().values
        wt_vals = drug_sens.reindex(wt_lines).dropna().values

        if len(alt_vals) < MIN_SAMPLES or len(wt_vals) < MIN_SAMPLES:
            continue

        d = cohens_d(alt_vals, wt_vals)
        _, pval = stats.ttest_ind(alt_vals, wt_vals, equal_var=False)

        compound_key = _extract_compound_key(str(treatment_id))
        drug_name = drug_name_map.get(compound_key, "")

        rows.append({
            "treatment_id": treatment_id,
            "compound": drug_name,
            "compound_key": compound_key,
            "d": round(d, 4),
            "pvalue": float(pval),
            "n_mut": len(alt_vals),
            "n_wt": len(wt_vals),
            "median_sens_mut": round(float(np.median(alt_vals)), 4),
            "median_sens_wt": round(float(np.median(wt_vals)), 4),
        })
        pvals.append(pval)

    result = pd.DataFrame(rows)
    if len(result) > 0 and pvals:
        result["qvalue"] = fdr_correction(np.array(pvals))
    return result


def plot_volcano(results: pd.DataFrame, output_dir: Path) -> None:
    """Volcano plot of genome-wide drug screen."""
    if "qvalue" not in results.columns or len(results) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    x = results["d"].values
    y = -np.log10(results["qvalue"].values.clip(min=1e-50))

    sig = (results["qvalue"] < FDR_THRESHOLD) & (results["d"].abs() > EFFECT_SIZE_THRESHOLD)

    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)

    sensitizing = sig & (results["d"] < 0)
    resistant = sig & (results["d"] > 0)
    ax.scatter(x[sensitizing], y[sensitizing], c="#D95319", s=15, alpha=0.8,
               label="Sensitizing in altered")
    ax.scatter(x[resistant], y[resistant], c="#4DBEEE", s=15, alpha=0.8,
               label="Resistance in altered")

    # Label top sensitizing hits
    top = results[sensitizing].nsmallest(15, "d")
    for _, row in top.iterrows():
        label = row["compound"] if row["compound"] else row["compound_key"]
        ax.annotate(label, (row["d"], -np.log10(max(row["qvalue"], 1e-50))),
                    fontsize=6, ha="right")

    ax.axhline(-np.log10(FDR_THRESHOLD), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (KEAP1/NRF2-altered vs WT)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("PRISM Drug Screen: KEAP1/NRF2-Altered vs WT (Pan-Cancer)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "prism_drug_volcano.png", dpi=150)
    plt.close(fig)


def write_summary(
    results: pd.DataFrame,
    n_altered: int,
    n_wt: int,
    output_dir: Path,
) -> None:
    """Write Markdown summary of top hits."""
    lines = [
        "# KEAP1/NRF2 Pan-Cancer Dependency Atlas - Phase 4a: PRISM Drug Screen",
        "",
        "## Cohort",
        f"- KEAP1/NRF2-altered cell lines in PRISM: **{n_altered}**",
        f"- WT cell lines in PRISM: **{n_wt}**",
        f"- Compounds screened: **{len(results)}**",
        "",
    ]

    if "qvalue" not in results.columns:
        lines.append("No results with FDR correction available.")
        with open(output_dir / "phase4a_summary.md", "w") as f:
            f.write("\n".join(lines))
        return

    sig = results[(results["qvalue"] < FDR_THRESHOLD) & (results["d"].abs() > EFFECT_SIZE_THRESHOLD)]
    sensitizing = sig[sig["d"] < 0].sort_values("d")
    resistant = sig[sig["d"] > 0].sort_values("d", ascending=False)

    lines += [
        "## Summary Statistics",
        f"- Significant compounds (|d|>{EFFECT_SIZE_THRESHOLD}, FDR<{FDR_THRESHOLD}): **{len(sig)}**",
        f"  - Sensitizing in altered (d<0): **{len(sensitizing)}**",
        f"  - Resistance in altered (d>0): **{len(resistant)}**",
        "",
        "## Top 30 Sensitizing Hits (d<0, more sensitive in KEAP1/NRF2-altered)",
        "",
        "| Rank | Compound | d | p-value | FDR | n_mut | n_wt |",
        "|------|----------|---|---------|-----|-------|------|",
    ]

    for i, (_, row) in enumerate(sensitizing.head(30).iterrows(), 1):
        name = row["compound"] if row["compound"] else row["compound_key"]
        lines.append(
            f"| {i} | {name} | {row['d']:.3f} | {row['pvalue']:.2e} | "
            f"{row['qvalue']:.2e} | {row['n_mut']} | {row['n_wt']} |"
        )

    if len(resistant) > 0:
        lines += [
            "",
            "## Top Resistance Hits (d>0, less sensitive in KEAP1/NRF2-altered)",
            "",
            "| Rank | Compound | d | p-value | FDR | n_mut | n_wt |",
            "|------|----------|---|---------|-----|-------|------|",
        ]
        for i, (_, row) in enumerate(resistant.head(10).iterrows(), 1):
            name = row["compound"] if row["compound"] else row["compound_key"]
            lines.append(
                f"| {i} | {name} | {row['d']:.3f} | {row['pvalue']:.2e} | "
                f"{row['qvalue']:.2e} | {row['n_mut']} | {row['n_wt']} |"
            )

    lines += [
        "",
        "## Notes",
        "- MOA and target annotations are not in the PRISM 24Q2 treatment metadata.",
        "  Phase 4b targeted analysis will annotate specific compound classes.",
        "- Negative d = compound kills KEAP1/NRF2-altered cells more than WT (potential therapeutic).",
        "- Welch t-test used for p-values; Benjamini-Hochberg for FDR correction.",
        "",
    ]

    with open(output_dir / "phase4a_summary.md", "w") as f:
        f.write("\n".join(lines))


def smoke_test(results: pd.DataFrame) -> None:
    """Verify output schema and basic sanity checks."""
    required_cols = {"treatment_id", "compound", "compound_key", "d", "pvalue", "qvalue", "n_mut", "n_wt"}
    missing = required_cols - set(results.columns)
    assert not missing, f"Missing columns: {missing}"

    assert len(results) > 100, f"Too few compounds screened: {len(results)}"
    assert results["d"].between(-10, 10).all(), "Effect sizes out of range"
    assert results["pvalue"].between(0, 1).all(), "p-values out of range"
    assert results["qvalue"].between(0, 1).all(), "q-values out of range"
    assert (results["n_mut"] >= MIN_SAMPLES).all(), "n_mut below minimum"
    assert (results["n_wt"] >= MIN_SAMPLES).all(), "n_wt below minimum"

    # Check that some compounds have names
    named = (results["compound"] != "").sum()
    assert named > 100, f"Too few compounds with names: {named}"

    print("  Smoke test PASSED")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4a: PRISM Drug Sensitivity Screen (KEAP1/NRF2) ===\n")

    # Load Phase 1 classification
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "keap1_nrf2_classification.csv", index_col=0)
    n_altered = (classified["KEAP1_NRF2_altered"] == True).sum()  # noqa: E712
    n_wt = (classified["pathway_status"] == "WT").sum()
    print(f"  {n_altered} KEAP1/NRF2-altered, {n_wt} WT lines")

    # Load PRISM treatment metadata for name mapping
    print("\nLoading PRISM 24Q2 treatment metadata...")
    meta = pd.read_csv(DEPMAP_DIR / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")
    drug_name_map = build_drug_name_map(meta)
    print(f"  {len(drug_name_map)} unique compound keys mapped to names")

    # Load PRISM sensitivity matrix
    print("\nLoading PRISM data matrix...")
    prism = pd.read_csv(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )
    print(f"  {prism.shape[0]} treatments x {prism.shape[1]} cell lines")

    # Check overlap with classified lines
    prism_lines = set(prism.columns)
    classified_lines = set(classified.index)
    overlap = prism_lines & classified_lines
    altered_in_prism = set(classified[classified["KEAP1_NRF2_altered"] == True].index) & prism_lines  # noqa: E712
    wt_in_prism = set(classified[classified["pathway_status"] == "WT"].index) & prism_lines
    print(f"  Overlap: {len(overlap)} lines ({len(altered_in_prism)} altered, {len(wt_in_prism)} WT)")

    # Run genome-wide drug screen
    print("\nRunning genome-wide drug sensitivity screen...")
    results = genomewide_drug_screen(prism, classified, drug_name_map)
    print(f"  {len(results)} compounds screened")

    if "qvalue" in results.columns:
        sig = results[
            (results["qvalue"] < FDR_THRESHOLD) & (results["d"].abs() > EFFECT_SIZE_THRESHOLD)
        ]
        sensitizing = sig[sig["d"] < 0].sort_values("d")
        resistant = sig[sig["d"] > 0].sort_values("d", ascending=False)
        print(f"  Significant (|d|>{EFFECT_SIZE_THRESHOLD}, FDR<{FDR_THRESHOLD}): {len(sig)}")
        print(f"    Sensitizing in altered: {len(sensitizing)}")
        print(f"    Resistance in altered: {len(resistant)}")

        if len(sensitizing) > 0:
            print(f"\n  Top 10 sensitizing compounds:")
            for _, row in sensitizing.head(10).iterrows():
                name = row["compound"] if row["compound"] else row["compound_key"]
                print(f"    {name}: d={row['d']:.3f}, FDR={row['qvalue']:.2e}")

    # Smoke test
    print("\nRunning smoke test...")
    smoke_test(results)

    # Save outputs
    print("\nSaving outputs...")
    results_sorted = results.sort_values("d").reset_index(drop=True)

    # Parquet
    results_sorted.to_parquet(OUTPUT_DIR / "phase4_prism_genome_wide.parquet", index=False)
    print("  phase4_prism_genome_wide.parquet")

    # CSV
    results_sorted.to_csv(OUTPUT_DIR / "phase4_prism_genome_wide.csv", index=False)
    print("  phase4_prism_genome_wide.csv")

    # Volcano plot
    plot_volcano(results_sorted, OUTPUT_DIR)
    print("  prism_drug_volcano.png")

    # Markdown summary
    write_summary(results_sorted, len(altered_in_prism), len(wt_in_prism), OUTPUT_DIR)
    print("  phase4a_summary.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
