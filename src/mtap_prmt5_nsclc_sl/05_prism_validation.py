"""Phase 3b: PRISM GSK3326595 (PRMT5 inhibitor) drug sensitivity validation.

Validates the MTAP/PRMT5 SL relationship using PRISM drug response data for
GSK3326595 — tests whether MTAP-deleted lines are more sensitive.

Usage:
    uv run python -m mtap_prmt5_nsclc_sl.05_prism_validation
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
OUTPUT_DIR = REPO_ROOT / "output" / "mtap-prmt5-nsclc-sl"
FIG_DIR = OUTPUT_DIR / "figures"

GSK_BRD_ID = "BRD:BRD-K00003421-001-01-9"
MTAP_CN_THRESHOLD = 0.5


def load_prism_gsk3326595() -> pd.Series:
    """Load GSK3326595 sensitivity scores from PRISM data matrix."""
    prism = pd.read_csv(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )
    if GSK_BRD_ID not in prism.index:
        raise ValueError(f"GSK3326595 ({GSK_BRD_ID}) not found in PRISM data")

    gsk = prism.loc[GSK_BRD_ID].astype(float)
    gsk.name = "GSK3326595_sensitivity"
    return gsk


def compare_sensitivity(
    deleted: np.ndarray, intact: np.ndarray, label: str
) -> dict:
    """Compare drug sensitivity between MTAP-deleted and intact groups."""
    stat, pval = stats.mannwhitneyu(deleted, intact, alternative="two-sided")

    n1, n2 = len(deleted), len(intact)
    var1, var2 = deleted.var(ddof=1), intact.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (deleted.mean() - intact.mean()) / pooled if pooled > 0 else 0.0

    result = {
        "label": label,
        "n_deleted": n1,
        "n_intact": n2,
        "median_deleted": float(np.median(deleted)),
        "median_intact": float(np.median(intact)),
        "mannwhitney_p": float(pval),
        "cohens_d": float(d),
    }

    print(f"\n{label}:")
    print(f"  N: deleted={n1}, intact={n2}")
    print(f"  Median sensitivity: deleted={np.median(deleted):.4f}, intact={np.median(intact):.4f}")
    print(f"  Wilcoxon p={pval:.2e}, Cohen's d={d:.3f}")

    return result


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load GSK3326595 sensitivity
    print("Loading PRISM GSK3326595 data...")
    gsk = load_prism_gsk3326595()
    print(f"  {gsk.notna().sum()} cell lines with sensitivity data")

    # Load classified NSCLC lines
    classified = pd.read_csv(OUTPUT_DIR / "nsclc_cell_lines_classified.csv", index_col=0)

    # Load CN for pan-cancer analysis
    cn_all = load_depmap_matrix(DEPMAP_DIR / "PortalOmicsCNGeneLog2.csv")
    mtap_cn_all = cn_all["MTAP"]

    # Load CRISPR for correlation
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    prmt5_dep = crispr["PRMT5"] if "PRMT5" in crispr.columns else None

    # === NSCLC analysis ===
    nsclc_merged = classified.join(gsk, how="inner")
    nsclc_merged = nsclc_merged.dropna(subset=["MTAP_deleted", "GSK3326595_sensitivity"])

    nsclc_del = nsclc_merged[nsclc_merged["MTAP_deleted"]]["GSK3326595_sensitivity"].values
    nsclc_int = nsclc_merged[~nsclc_merged["MTAP_deleted"]]["GSK3326595_sensitivity"].values

    nsclc_result = None
    if len(nsclc_del) >= 3 and len(nsclc_int) >= 3:
        nsclc_result = compare_sensitivity(nsclc_del, nsclc_int,
                                           "NSCLC: GSK3326595 sensitivity by MTAP status")
    else:
        print(f"\nNSCLC: insufficient data (deleted={len(nsclc_del)}, intact={len(nsclc_int)})")
        nsclc_result = {
            "label": "NSCLC",
            "n_deleted": len(nsclc_del),
            "n_intact": len(nsclc_int),
            "note": "Insufficient samples for test",
        }

    # === Pan-cancer analysis ===
    pan = pd.DataFrame({
        "MTAP_CN": mtap_cn_all,
        "GSK3326595": gsk,
    }).dropna()
    pan["MTAP_deleted"] = pan["MTAP_CN"] < MTAP_CN_THRESHOLD

    pan_del = pan[pan["MTAP_deleted"]]["GSK3326595"].values
    pan_int = pan[~pan["MTAP_deleted"]]["GSK3326595"].values
    pan_result = compare_sensitivity(pan_del, pan_int,
                                     "Pan-cancer: GSK3326595 sensitivity by MTAP status")

    # === Correlation: CRISPR dep vs drug sensitivity ===
    corr_result = None
    if prmt5_dep is not None:
        corr_df = pd.DataFrame({
            "PRMT5_dep": prmt5_dep,
            "GSK3326595": gsk,
        }).dropna()
        if len(corr_df) >= 10:
            r_s, p_s = stats.spearmanr(corr_df["PRMT5_dep"], corr_df["GSK3326595"])
            r_p, p_p = stats.pearsonr(corr_df["PRMT5_dep"], corr_df["GSK3326595"])
            corr_result = {
                "n": len(corr_df),
                "spearman_r": float(r_s), "spearman_p": float(p_s),
                "pearson_r": float(r_p), "pearson_p": float(p_p),
            }
            print(f"\nCRISPR-Drug correlation (n={len(corr_df)}):")
            print(f"  Spearman r={r_s:.3f} (p={p_s:.2e})")
            print(f"  Pearson  r={r_p:.3f} (p={p_p:.2e})")

    # === Visualizations ===
    print("\nGenerating plots...")

    # Box plot: pan-cancer
    fig, ax = plt.subplots(figsize=(5, 6))
    bp = ax.boxplot(
        [pan_int, pan_del],
        tick_labels=[f"MTAP intact\n(n={len(pan_int)})", f"MTAP deleted\n(n={len(pan_del)})"],
        widths=0.5, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    bp["boxes"][0].set_facecolor("#4DBEEE")
    bp["boxes"][1].set_facecolor("#D95319")
    for i, d in enumerate([pan_int, pan_del]):
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, alpha=0.3, s=15,
                   color="gray", zorder=3)
    ax.set_ylabel("GSK3326595 sensitivity (PRISM log fold-change)")
    ax.set_title("GSK3326595 (PRMT5i) by MTAP Status (Pan-cancer)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gsk3326595_pancancer_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: gsk3326595_pancancer_boxplot.png")

    # Scatter: CRISPR vs drug
    if corr_result and prmt5_dep is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(corr_df["PRMT5_dep"], corr_df["GSK3326595"], alpha=0.3, s=15)
        ax.set_xlabel("PRMT5 CRISPR dependency (GeneEffect)")
        ax.set_ylabel("GSK3326595 sensitivity (PRISM)")
        ax.set_title("PRMT5 CRISPR Dependency vs GSK3326595 Drug Sensitivity")
        ax.annotate(
            f"Spearman r={corr_result['spearman_r']:.3f}\np={corr_result['spearman_p']:.2e}",
            xy=(0.05, 0.95), xycoords="axes fraction", fontsize=9, va="top",
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "prmt5_crispr_vs_gsk3326595.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: prmt5_crispr_vs_gsk3326595.png")

    # === Save results ===
    results = {
        "drug": "GSK3326595 (PRMT5 inhibitor)",
        "brd_id": GSK_BRD_ID,
        "nsclc": nsclc_result,
        "pan_cancer": pan_result,
        "crispr_drug_correlation": corr_result,
    }
    out_path = OUTPUT_DIR / "prism_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
