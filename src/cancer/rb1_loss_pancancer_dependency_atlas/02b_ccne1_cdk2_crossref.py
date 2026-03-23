"""Phase 2 Addendum: CCNE1-amp vs RB1-loss CDK2 dependency cross-reference.

Tests CDK2 dependency convergence hypothesis: CCNE1 amplification and RB1 loss
both drive CDK2-dependent G1/S transition. INX-315 (CDK2i) has FDA Fast Track
for CCNE1-amp ovarian — if CDK2 dependency overlaps in RB1-loss lines, INX-315
covers both populations.

Comparisons:
1. CCNE1-amp vs non-amp (all lines, regardless of RB1)
2. RB1-loss vs intact (all lines, regardless of CCNE1)
3. Four-group: RB1xCCNE1 (double-hit, RB1-only, CCNE1-only, neither)
4. Convergence test: overlap of CDK2 dependency distributions

Usage:
    uv run python src/cancer/rb1_loss_pancancer_dependency_atlas/02b_ccne1_cdk2_crossref.py
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

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "output" / "cancer" / "rb1-loss-pancancer-dependency-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "rb1-loss-pancancer-dependency-atlas" / "phase2"

N_BOOTSTRAP = 1000
SEED = 42


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def bootstrap_ci(
    group1: np.ndarray, group2: np.ndarray,
    n_boot: int = N_BOOTSTRAP, seed: int = SEED,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        b1 = rng.choice(group1, size=len(group1), replace=True)
        b2 = rng.choice(group2, size=len(group2), replace=True)
        ds[i] = cohens_d(b1, b2)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def compare_groups(
    vals1: np.ndarray, vals2: np.ndarray, label: str,
) -> dict:
    """Compute effect size and significance for two groups."""
    d = cohens_d(vals1, vals2)
    ci_lo, ci_hi = bootstrap_ci(vals1, vals2)
    _, pval = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
    return {
        "comparison": label,
        "n_group1": len(vals1),
        "n_group2": len(vals2),
        "median_group1": round(float(np.median(vals1)), 4),
        "median_group2": round(float(np.median(vals2)), 4),
        "mean_group1": round(float(np.mean(vals1)), 4),
        "mean_group2": round(float(np.mean(vals2)), 4),
        "cohens_d": round(d, 4),
        "ci_lower": round(ci_lo, 4),
        "ci_upper": round(ci_hi, 4),
        "mann_whitney_p": float(pval),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2 Addendum: CCNE1-amp vs RB1-loss CDK2 Cross-Reference ===\n")

    # Load data
    print("Loading Phase 1 classifications...")
    classified = pd.read_csv(PHASE1_DIR / "rb1_classification.csv", index_col=0)

    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    if "CDK2" not in crispr.columns:
        print("ERROR: CDK2 not in CRISPR data")
        return

    # Merge
    merged = classified[classified["has_crispr"]].join(crispr[["CDK2"]], how="inner")
    merged = merged[merged["RB1_status"].isin(["lost", "intact"])].copy()
    merged["CDK2"] = pd.to_numeric(merged["CDK2"], errors="coerce")
    merged = merged.dropna(subset=["CDK2"])

    print(f"  {len(merged)} lines with RB1 status and CDK2 CRISPR data")

    # Define groups
    rb1_loss = merged[merged["RB1_status"] == "lost"]
    rb1_intact = merged[merged["RB1_status"] == "intact"]
    ccne1_amp = merged[merged["CCNE1_amplified"] == True]
    ccne1_noamp = merged[merged["CCNE1_amplified"] == False]

    # Four-way groups
    double_hit = merged[(merged["RB1_status"] == "lost") & (merged["CCNE1_amplified"] == True)]
    rb1_only = merged[(merged["RB1_status"] == "lost") & (merged["CCNE1_amplified"] == False)]
    ccne1_only = merged[(merged["RB1_status"] == "intact") & (merged["CCNE1_amplified"] == True)]
    neither = merged[(merged["RB1_status"] == "intact") & (merged["CCNE1_amplified"] == False)]

    print(f"\n  Four-group sizes:")
    print(f"    Double-hit (RB1-loss + CCNE1-amp): {len(double_hit)}")
    print(f"    RB1-loss only:                     {len(rb1_only)}")
    print(f"    CCNE1-amp only:                    {len(ccne1_only)}")
    print(f"    Neither:                           {len(neither)}")

    rows = []

    # --- Comparison 1: CCNE1-amp vs non-amp (all lines) ---
    print("\n--- Comparison 1: CCNE1-amp vs non-amp (all lines) ---")
    r = compare_groups(
        ccne1_amp["CDK2"].values, ccne1_noamp["CDK2"].values,
        "CCNE1-amp vs non-amp (all)",
    )
    rows.append(r)
    print(f"  d={r['cohens_d']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] p={r['mann_whitney_p']:.3e}")

    # --- Comparison 2: RB1-loss vs intact (all lines) ---
    print("\n--- Comparison 2: RB1-loss vs intact (all lines) ---")
    r = compare_groups(
        rb1_loss["CDK2"].values, rb1_intact["CDK2"].values,
        "RB1-loss vs intact (all)",
    )
    rows.append(r)
    print(f"  d={r['cohens_d']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] p={r['mann_whitney_p']:.3e}")

    # --- Comparison 3: Double-hit vs neither ---
    if len(double_hit) >= 3 and len(neither) >= 3:
        print("\n--- Comparison 3: Double-hit vs neither ---")
        r = compare_groups(
            double_hit["CDK2"].values, neither["CDK2"].values,
            "Double-hit (RB1-loss+CCNE1-amp) vs neither",
        )
        rows.append(r)
        print(f"  d={r['cohens_d']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] p={r['mann_whitney_p']:.3e}")

    # --- Comparison 4: Double-hit vs RB1-only ---
    if len(double_hit) >= 3 and len(rb1_only) >= 3:
        print("\n--- Comparison 4: Double-hit vs RB1-loss only ---")
        r = compare_groups(
            double_hit["CDK2"].values, rb1_only["CDK2"].values,
            "Double-hit vs RB1-loss only",
        )
        rows.append(r)
        print(f"  d={r['cohens_d']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] p={r['mann_whitney_p']:.3e}")

    # --- Comparison 5: Double-hit vs CCNE1-only ---
    if len(double_hit) >= 3 and len(ccne1_only) >= 3:
        print("\n--- Comparison 5: Double-hit vs CCNE1-amp only ---")
        r = compare_groups(
            double_hit["CDK2"].values, ccne1_only["CDK2"].values,
            "Double-hit vs CCNE1-amp only",
        )
        rows.append(r)
        print(f"  d={r['cohens_d']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] p={r['mann_whitney_p']:.3e}")

    # --- Comparison 6: CCNE1-only vs neither (isolated CCNE1 effect) ---
    if len(ccne1_only) >= 3 and len(neither) >= 3:
        print("\n--- Comparison 6: CCNE1-amp only vs neither ---")
        r = compare_groups(
            ccne1_only["CDK2"].values, neither["CDK2"].values,
            "CCNE1-amp only vs neither",
        )
        rows.append(r)
        print(f"  d={r['cohens_d']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] p={r['mann_whitney_p']:.3e}")

    # --- Comparison 7: RB1-only vs neither (isolated RB1 effect) ---
    if len(rb1_only) >= 3 and len(neither) >= 3:
        print("\n--- Comparison 7: RB1-loss only vs neither ---")
        r = compare_groups(
            rb1_only["CDK2"].values, neither["CDK2"].values,
            "RB1-loss only vs neither",
        )
        rows.append(r)
        print(f"  d={r['cohens_d']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] p={r['mann_whitney_p']:.3e}")

    # Save comparison table
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_DIR / "ccne1_rb1_cdk2_crossref.csv", index=False)
    print(f"\nSaved comparison table to phase2/ccne1_rb1_cdk2_crossref.csv")

    # --- Plot: Four-group boxplot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    groups_data = []
    group_labels = []
    group_colors = []

    for name, data, color in [
        (f"Neither\n(n={len(neither)})", neither["CDK2"].values, "#999999"),
        (f"CCNE1-amp only\n(n={len(ccne1_only)})", ccne1_only["CDK2"].values, "#4DBEEE"),
        (f"RB1-loss only\n(n={len(rb1_only)})", rb1_only["CDK2"].values, "#D95319"),
        (f"Double-hit\n(n={len(double_hit)})", double_hit["CDK2"].values, "#7E2F8E"),
    ]:
        if len(data) > 0:
            groups_data.append(data)
            group_labels.append(name)
            group_colors.append(color)

    bp = ax.boxplot(groups_data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay strip points
    for i, data in enumerate(groups_data):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
        ax.scatter(np.full(len(data), i + 1) + jitter, data,
                   alpha=0.4, s=15, color="black", zorder=3)

    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylabel("CDK2 CRISPR Dependency Score\n(more negative = more essential)")
    ax.set_title("CDK2 Dependency: RB1 × CCNE1 Cross-Reference\n"
                 "(Convergence hypothesis: both drive CDK2-dependent G1/S)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ccne1_rb1_cdk2_crossref_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved boxplot to phase2/ccne1_rb1_cdk2_crossref_boxplot.png")

    # --- Summary text ---
    lines = [
        "=" * 70,
        "CCNE1-amp vs RB1-loss CDK2 Dependency Cross-Reference",
        "=" * 70,
        "",
        "HYPOTHESIS: CCNE1 amplification and RB1 loss both drive CDK2-dependent",
        "G1/S transition. If CDK2 dependency converges, INX-315 (CDK2i, FDA Fast",
        "Track for CCNE1-amp ovarian) may cover both populations.",
        "",
    ]

    for _, r in result_df.iterrows():
        sig = "***" if r["mann_whitney_p"] < 0.001 else "**" if r["mann_whitney_p"] < 0.01 else "*" if r["mann_whitney_p"] < 0.05 else "ns"
        lines.append(
            f"  {r['comparison']:50s} d={r['cohens_d']:+.3f} "
            f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] "
            f"p={r['mann_whitney_p']:.3e} {sig}"
        )

    lines.extend([
        "",
        "INTERPRETATION:",
    ])

    # Auto-generate interpretation
    ccne1_d = result_df[result_df["comparison"] == "CCNE1-amp vs non-amp (all)"]["cohens_d"].values
    rb1_d = result_df[result_df["comparison"] == "RB1-loss vs intact (all)"]["cohens_d"].values
    dh_row = result_df[result_df["comparison"].str.contains("Double-hit.*neither")]

    if len(ccne1_d) > 0 and len(rb1_d) > 0:
        lines.append(f"  - CCNE1-amp effect on CDK2: d={ccne1_d[0]:+.3f}")
        lines.append(f"  - RB1-loss effect on CDK2:  d={rb1_d[0]:+.3f}")
        if abs(ccne1_d[0]) > 0.2 and abs(rb1_d[0]) > 0.2 and ccne1_d[0] < 0 and rb1_d[0] < 0:
            lines.append("  - CONVERGENCE SUPPORTED: Both alterations increase CDK2 dependency")
        elif ccne1_d[0] < 0 and rb1_d[0] < 0:
            lines.append("  - Weak convergence: Both show negative trend but effect sizes differ")
        else:
            lines.append("  - Convergence NOT supported in DepMap CRISPR data")

    if len(dh_row) > 0:
        dh_d = dh_row["cohens_d"].values[0]
        lines.append(f"  - Double-hit vs neither: d={dh_d:+.3f} (additive/synergistic effect)")

    lines.append("")

    with open(OUTPUT_DIR / "ccne1_rb1_cdk2_crossref_summary.txt", "w") as f:
        f.write("\n".join(lines))
    print("Saved summary to phase2/ccne1_rb1_cdk2_crossref_summary.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
