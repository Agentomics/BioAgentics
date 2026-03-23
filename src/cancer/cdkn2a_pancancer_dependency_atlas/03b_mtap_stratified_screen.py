"""Phase 3b: MTAP-stratified genome-wide screen to deconfound CDKN2A vs MTAP effects.

CDKN2A and MTAP are both on 9p21 — 74% of CDKN2A-deleted lines have MTAP co-deletion.
This means the original Phase 3 screen conflates CDKN2A-specific and MTAP-driven dependencies.

Two key comparisons:
  A) MTAP-corrected CDKN2A screen:
     CDKN2A-del/RB1-intact/MTAP-intact vs CDKN2A-intact
     → Hits here are genuinely CDKN2A-specific (MTAP confounder removed)

  B) MTAP confound check (within CDKN2A-deleted/RB1-intact):
     MTAP-deleted vs MTAP-intact
     → Hits here are MTAP-driven, not CDKN2A-specific

Usage:
    uv run python -m cancer.cdkn2a_pancancer_dependency_atlas.03b_mtap_stratified_screen
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

HOMDEL_CN_THRESHOLD = 0.3

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase1"
PHASE3_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase3"
OUTPUT_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase3b_mtap_stratified"

FDR_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.3
MIN_SAMPLES = 3

# Genes of primary interest for MTAP confounding
KEY_GENES = {
    "CDK6", "CDK4", "WDR77", "PRMT5", "MAT2A", "MTAP",
    "CCND1", "CCND3", "CDK2", "MDM2", "E2F1",
    "KIF2C", "WWTR1", "TLN1", "ACTR3", "MSI2",
}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    fdr = np.empty(n)
    for i in range(n):
        fdr[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted
    return np.minimum(fdr, 1.0)


def screen_one_context(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    crispr_cols: list[str],
    context_name: str,
    comparison_label: str,
) -> list[dict]:
    """Run genome-wide differential dependency for one context."""
    rows = []
    pvals = []

    for gene in crispr_cols:
        a_vals = group_a[gene].dropna().values
        b_vals = group_b[gene].dropna().values

        if len(a_vals) < MIN_SAMPLES or len(b_vals) < MIN_SAMPLES:
            continue

        _, pval = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        d = cohens_d(a_vals, b_vals)

        rows.append({
            "comparison": comparison_label,
            "cancer_type": context_name,
            "gene": gene,
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_group_a": len(a_vals),
            "n_group_b": len(b_vals),
            "median_a": round(float(np.median(a_vals)), 4),
            "median_b": round(float(np.median(b_vals)), 4),
        })
        pvals.append(pval)

    if pvals:
        fdrs = fdr_correction(np.array(pvals))
        for i, row in enumerate(rows):
            row["fdr"] = float(fdrs[i])

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3b: MTAP-Stratified Genome-Wide Screen ===\n")

    # Load Phase 1 classification
    print("Loading Phase 1 classification...")
    classified = pd.read_csv(PHASE1_DIR / "cdkn2a_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()

    # Add MTAP annotation if missing from Phase 1 output
    if "MTAP_co_deleted" not in classified.columns:
        print("  MTAP annotation missing from Phase 1 CSV — loading from DepMap CN...")
        cn = load_depmap_matrix(DEPMAP_DIR / "PortalOmicsCNGeneLog2.csv")
        if "MTAP" in cn.columns:
            mtap_cn = cn["MTAP"].rename("MTAP_CN_log2")
            classified = classified.join(mtap_cn, how="left")
            classified["MTAP_co_deleted"] = classified["MTAP_CN_log2"] <= HOMDEL_CN_THRESHOLD
        else:
            raise RuntimeError("MTAP not found in PortalOmicsCNGeneLog2.csv")

    # Cohort counts
    cdkn2a_del = classified[classified["CDKN2A_status"] == "deleted"]
    n_mtap_codel = int(cdkn2a_del["MTAP_co_deleted"].fillna(False).sum())
    n_mtap_intact = len(cdkn2a_del) - n_mtap_codel
    print(f"  CDKN2A-deleted: {len(cdkn2a_del)} total")
    print(f"    MTAP co-deleted: {n_mtap_codel} ({n_mtap_codel/len(cdkn2a_del):.1%})")
    print(f"    MTAP intact: {n_mtap_intact} ({n_mtap_intact/len(cdkn2a_del):.1%})")

    # Load CRISPR
    print("\nLoading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    crispr_cols = list(crispr.columns)
    print(f"  {len(crispr_cols)} genes")

    # Build cohorts with RB1-intact filter (same as Phase 3 primary)
    classified_primary = classified[
        (classified["CDKN2A_status"] == "intact") |
        ((classified["CDKN2A_status"] == "deleted") & (classified["RB1_status"] == "intact"))
    ].copy()

    merged = classified_primary.join(crispr, how="inner")

    # Define strata
    intact_all = merged[merged["CDKN2A_status"] == "intact"]
    del_rb1intact = merged[
        (merged["CDKN2A_status"] == "deleted") & (merged["RB1_status"] == "intact")
    ]
    del_mtap_intact = del_rb1intact[del_rb1intact["MTAP_co_deleted"].fillna(False) == False]
    del_mtap_deleted = del_rb1intact[del_rb1intact["MTAP_co_deleted"].fillna(False) == True]

    print(f"\n  Cohort sizes (with CRISPR, RB1-intact filter):")
    print(f"    CDKN2A-intact (control): {len(intact_all)}")
    print(f"    CDKN2A-del/RB1-intact total: {len(del_rb1intact)}")
    print(f"    CDKN2A-del/RB1-intact/MTAP-intact: {len(del_mtap_intact)}")
    print(f"    CDKN2A-del/RB1-intact/MTAP-deleted: {len(del_mtap_deleted)}")

    all_rows = []

    # === Comparison A: MTAP-corrected CDKN2A screen (pan-cancer) ===
    print("\n--- Comparison A: MTAP-corrected CDKN2A screen ---")
    print("  (CDKN2A-del/MTAP-intact vs CDKN2A-intact)\n")

    contexts_a = qualifying_types + ["Pan-cancer (pooled)"]
    for context in contexts_a:
        if context == "Pan-cancer (pooled)":
            del_grp = del_mtap_intact
            intact_grp = intact_all
        else:
            del_grp = del_mtap_intact[del_mtap_intact["OncotreeLineage"] == context]
            intact_grp = intact_all[intact_all["OncotreeLineage"] == context]

        if len(del_grp) < MIN_SAMPLES or len(intact_grp) < MIN_SAMPLES:
            print(f"  {context}: SKIP (n_del={len(del_grp)}, n_intact={len(intact_grp)})")
            continue

        print(f"  {context}: {len(del_grp)} del/MTAP-intact vs {len(intact_grp)} intact")
        rows = screen_one_context(
            del_grp, intact_grp, crispr_cols, context,
            "CDKN2A-del_MTAP-intact_vs_intact"
        )
        all_rows.extend(rows)

    # === Comparison B: MTAP confound check (within CDKN2A-del/RB1-intact) ===
    print("\n--- Comparison B: MTAP confound check ---")
    print("  (CDKN2A-del/MTAP-del vs CDKN2A-del/MTAP-intact)\n")

    contexts_b = qualifying_types + ["Pan-cancer (pooled)"]
    for context in contexts_b:
        if context == "Pan-cancer (pooled)":
            mtap_del_grp = del_mtap_deleted
            mtap_int_grp = del_mtap_intact
        else:
            mtap_del_grp = del_mtap_deleted[del_mtap_deleted["OncotreeLineage"] == context]
            mtap_int_grp = del_mtap_intact[del_mtap_intact["OncotreeLineage"] == context]

        if len(mtap_del_grp) < MIN_SAMPLES or len(mtap_int_grp) < MIN_SAMPLES:
            print(f"  {context}: SKIP (n_mtap_del={len(mtap_del_grp)}, n_mtap_intact={len(mtap_int_grp)})")
            continue

        print(f"  {context}: {len(mtap_del_grp)} MTAP-del vs {len(mtap_int_grp)} MTAP-intact")
        rows = screen_one_context(
            mtap_del_grp, mtap_int_grp, crispr_cols, context,
            "MTAP-confound_within_CDKN2A-del"
        )
        all_rows.extend(rows)

    all_results = pd.DataFrame(all_rows)
    print(f"\nTotal tests: {len(all_results)}")
    all_results.to_csv(OUTPUT_DIR / "mtap_stratified_all_results.csv", index=False)

    # === Load original Phase 3 pan-cancer results for comparison ===
    print("\nLoading original Phase 3 results for comparison...")
    orig = pd.read_csv(PHASE3_DIR / "genomewide_all_results.csv")
    orig_pancancer = orig[orig["cancer_type"] == "Pan-cancer (pooled)"].set_index("gene")

    # === Analysis: Key gene comparison table ===
    comp_a_pc = all_results[
        (all_results["comparison"] == "CDKN2A-del_MTAP-intact_vs_intact") &
        (all_results["cancer_type"] == "Pan-cancer (pooled)")
    ].set_index("gene")

    comp_b_pc = all_results[
        (all_results["comparison"] == "MTAP-confound_within_CDKN2A-del") &
        (all_results["cancer_type"] == "Pan-cancer (pooled)")
    ].set_index("gene")

    print("\n" + "=" * 100)
    print("KEY GENE COMPARISON: Original vs MTAP-corrected (Pan-cancer)")
    print("=" * 100)
    print(f"{'Gene':<10} {'Orig d':>8} {'Orig FDR':>12} {'MTAP-corr d':>12} {'MTAP-corr FDR':>14} "
          f"{'Confound d':>12} {'Confound FDR':>14} {'Verdict':>20}")
    print("-" * 100)

    key_gene_rows = []
    for gene in sorted(KEY_GENES):
        orig_d = orig_pancancer.loc[gene, "cohens_d"] if gene in orig_pancancer.index else float("nan")
        orig_fdr = orig_pancancer.loc[gene, "fdr"] if gene in orig_pancancer.index else float("nan")
        corr_d = comp_a_pc.loc[gene, "cohens_d"] if gene in comp_a_pc.index else float("nan")
        corr_fdr = comp_a_pc.loc[gene, "fdr"] if gene in comp_a_pc.index else float("nan")
        conf_d = comp_b_pc.loc[gene, "cohens_d"] if gene in comp_b_pc.index else float("nan")
        conf_fdr = comp_b_pc.loc[gene, "fdr"] if gene in comp_b_pc.index else float("nan")

        # Verdict logic
        if np.isnan(orig_d):
            verdict = "NO DATA"
        elif abs(orig_d) < EFFECT_SIZE_THRESHOLD or (not np.isnan(orig_fdr) and orig_fdr >= FDR_THRESHOLD):
            verdict = "not significant"
        elif not np.isnan(conf_d) and abs(conf_d) > EFFECT_SIZE_THRESHOLD and (not np.isnan(conf_fdr) and conf_fdr < FDR_THRESHOLD):
            verdict = "MTAP-DRIVEN"
        elif not np.isnan(corr_d) and abs(corr_d) > EFFECT_SIZE_THRESHOLD and (not np.isnan(corr_fdr) and corr_fdr < FDR_THRESHOLD):
            verdict = "CDKN2A-SPECIFIC"
        elif not np.isnan(corr_d) and abs(corr_d) > EFFECT_SIZE_THRESHOLD:
            verdict = "trend (FDR ns)"
        else:
            verdict = "attenuated"

        print(f"{gene:<10} {orig_d:>8.3f} {orig_fdr:>12.3e} {corr_d:>12.3f} {corr_fdr:>14.3e} "
              f"{conf_d:>12.3f} {conf_fdr:>14.3e} {verdict:>20}")

        key_gene_rows.append({
            "gene": gene,
            "original_d": round(orig_d, 4) if not np.isnan(orig_d) else None,
            "original_fdr": orig_fdr if not np.isnan(orig_fdr) else None,
            "mtap_corrected_d": round(corr_d, 4) if not np.isnan(corr_d) else None,
            "mtap_corrected_fdr": corr_fdr if not np.isnan(corr_fdr) else None,
            "mtap_confound_d": round(conf_d, 4) if not np.isnan(conf_d) else None,
            "mtap_confound_fdr": conf_fdr if not np.isnan(conf_fdr) else None,
            "verdict": verdict,
        })

    pd.DataFrame(key_gene_rows).to_csv(OUTPUT_DIR / "key_gene_mtap_comparison.csv", index=False)

    # === Genome-wide: hits that survive MTAP correction ===
    print("\n" + "=" * 80)
    print("GENOME-WIDE MTAP CORRECTION SUMMARY (Pan-cancer)")
    print("=" * 80)

    # Original Phase 3 pan-cancer gained hits
    orig_gained = set(
        orig_pancancer[
            (orig_pancancer["fdr"] < FDR_THRESHOLD) &
            (orig_pancancer["cohens_d"] < -EFFECT_SIZE_THRESHOLD)
        ].index
    )

    # MTAP-corrected gained hits
    corr_gained = set(
        comp_a_pc[
            (comp_a_pc["fdr"] < FDR_THRESHOLD) &
            (comp_a_pc["cohens_d"] < -EFFECT_SIZE_THRESHOLD)
        ].index
    ) if len(comp_a_pc) > 0 else set()

    # MTAP confound hits (significant MTAP effect within CDKN2A-del)
    mtap_driven = set(
        comp_b_pc[
            (comp_b_pc["fdr"] < FDR_THRESHOLD) &
            (comp_b_pc["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
        ].index
    ) if len(comp_b_pc) > 0 else set()

    survived = orig_gained & corr_gained
    lost_after_correction = orig_gained - corr_gained
    new_after_correction = corr_gained - orig_gained
    confirmed_mtap = orig_gained & mtap_driven

    print(f"\n  Original gained dependencies: {len(orig_gained)}")
    print(f"  Survived MTAP correction: {len(survived)}")
    print(f"  Lost after MTAP correction: {len(lost_after_correction)}")
    print(f"  New after MTAP correction: {len(new_after_correction)}")
    print(f"  Confirmed MTAP-driven: {len(confirmed_mtap)}")

    print(f"\n  SURVIVED (genuinely CDKN2A-specific):")
    for gene in sorted(survived):
        d_corr = comp_a_pc.loc[gene, "cohens_d"]
        fdr_corr = comp_a_pc.loc[gene, "fdr"]
        print(f"    {gene}: d={d_corr:.3f}, FDR={fdr_corr:.3e}")

    print(f"\n  LOST after correction (likely MTAP-confounded):")
    for gene in sorted(lost_after_correction):
        d_orig = orig_pancancer.loc[gene, "cohens_d"]
        mtap_flag = " [confirmed MTAP-driven]" if gene in confirmed_mtap else ""
        corr_info = ""
        if gene in comp_a_pc.index:
            d_corr = comp_a_pc.loc[gene, "cohens_d"]
            fdr_corr = comp_a_pc.loc[gene, "fdr"]
            corr_info = f" → corrected d={d_corr:.3f}, FDR={fdr_corr:.3e}"
        print(f"    {gene}: orig d={d_orig:.3f}{corr_info}{mtap_flag}")

    print(f"\n  CONFIRMED MTAP-driven (significant in confound check):")
    for gene in sorted(confirmed_mtap):
        d_conf = comp_b_pc.loc[gene, "cohens_d"]
        fdr_conf = comp_b_pc.loc[gene, "fdr"]
        print(f"    {gene}: MTAP confound d={d_conf:.3f}, FDR={fdr_conf:.3e}")

    # Save summary sets
    summary_data = {
        "survived_cdkn2a_specific": sorted(survived),
        "lost_after_mtap_correction": sorted(lost_after_correction),
        "confirmed_mtap_driven": sorted(confirmed_mtap),
        "new_after_mtap_correction": sorted(new_after_correction),
    }

    # === Write summary text ===
    summary_lines = [
        "=" * 80,
        "CDKN2A Pan-Cancer Dependency Atlas - Phase 3b: MTAP Stratification",
        "=" * 80,
        "",
        "PURPOSE: Deconfound CDKN2A-specific from MTAP-driven dependencies.",
        f"MTAP co-deletion rate in CDKN2A-deleted lines: {int(n_mtap_codel)}/{len(cdkn2a_del)} ({n_mtap_codel/len(cdkn2a_del):.1%})",
        "",
        "COHORT SIZES (with CRISPR, RB1-intact filter, pan-cancer):",
        f"  CDKN2A-intact (control): {len(intact_all)}",
        f"  CDKN2A-del/MTAP-intact: {len(del_mtap_intact)}",
        f"  CDKN2A-del/MTAP-deleted: {len(del_mtap_deleted)}",
        "",
        "COMPARISON A: MTAP-corrected CDKN2A screen",
        "  (CDKN2A-del/MTAP-intact vs CDKN2A-intact)",
        f"  Tests: {len(comp_a_pc)}",
        "",
        "COMPARISON B: MTAP confound check",
        "  (MTAP-del vs MTAP-intact within CDKN2A-del/RB1-intact)",
        f"  Tests: {len(comp_b_pc)}",
        "",
        "GENOME-WIDE RESULTS (Pan-cancer):",
        f"  Original gained dependencies: {len(orig_gained)}",
        f"  Survived MTAP correction: {len(survived)}",
        f"  Lost after correction: {len(lost_after_correction)}",
        f"  Confirmed MTAP-driven: {len(confirmed_mtap)}",
        "",
        "SURVIVED (genuinely CDKN2A-specific):",
    ]
    for gene in sorted(survived):
        d_corr = comp_a_pc.loc[gene, "cohens_d"]
        fdr_corr = comp_a_pc.loc[gene, "fdr"]
        summary_lines.append(f"  {gene}: d={d_corr:.3f}, FDR={fdr_corr:.3e}")

    summary_lines += ["", "LOST AFTER CORRECTION (MTAP-confounded):"]
    for gene in sorted(lost_after_correction):
        d_orig = orig_pancancer.loc[gene, "cohens_d"]
        mtap_flag = " [confirmed MTAP-driven]" if gene in confirmed_mtap else ""
        summary_lines.append(f"  {gene}: orig d={d_orig:.3f}{mtap_flag}")

    summary_lines += ["", "KEY GENE VERDICTS:"]
    for row in key_gene_rows:
        summary_lines.append(f"  {row['gene']}: {row['verdict']}")

    summary_lines.append("")

    with open(OUTPUT_DIR / "mtap_stratified_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSaved summary to {OUTPUT_DIR / 'mtap_stratified_summary.txt'}")

    # Save sets as JSON for downstream use
    import json
    with open(OUTPUT_DIR / "mtap_correction_sets.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
