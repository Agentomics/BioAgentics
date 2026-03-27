"""Phase 4b: MTAP-stratified PRISM drug sensitivity reanalysis.

58% of Phase 3 genome-wide CRISPR hits were MTAP-confounded. This script applies
the same MTAP correction to Phase 4 PRISM drug sensitivity data.

Two comparisons:
  A) MTAP-corrected CDKN2A drug screen:
     CDKN2A-del/RB1-intact/MTAP-intact vs CDKN2A-intact
     → Drug sensitivities here are genuinely CDKN2A-specific

  B) MTAP confound check (within CDKN2A-deleted/RB1-intact):
     MTAP-deleted vs MTAP-intact
     → Drug sensitivities here are MTAP-driven

Usage:
    uv run python -m cancer.cdkn2a_pancancer_dependency_atlas.04b_mtap_stratified_prism
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

HOMDEL_CN_THRESHOLD = 0.3

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase1"
PHASE4_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase4"
PHASE3B_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase3b_mtap_stratified"
OUTPUT_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase4b_mtap_stratified"

FDR_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.3
MIN_SAMPLES = 3

# Target drugs of interest
TARGET_DRUGS = {
    "trilaciclib": ("CDK4/6 inhibitor", "CDK6"),
    "milademetan": ("MDM2 inhibitor", "MDM2"),
    "GSK3326595": ("PRMT5 inhibitor (type I)", "PRMT5"),
    "JNJ-64619178": ("PRMT5 inhibitor (type II)", "PRMT5"),
    "EPZ020411": ("PRMT5 inhibitor (SAM-competitive)", "PRMT5"),
    "palbociclib": ("CDK4/6 inhibitor", "CDK4"),
    "ribociclib": ("CDK4/6 inhibitor", "CDK4"),
    "abemaciclib": ("CDK4/6 inhibitor", "CDK4"),
    "idasanutlin": ("MDM2 inhibitor", "MDM2"),
    # WWTR1/Hippo pathway drugs
    "verteporfin": ("YAP/TEAD inhibitor", "WWTR1"),
}

# Drugs that are known CDK4/6 inhibitors (for filtering)
CDK46_DRUG_NAMES = {"trilaciclib", "palbociclib", "ribociclib", "abemaciclib"}


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


def find_available_drugs(meta: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Search PRISM metadata for target drugs."""
    all_names = set(meta["name"].dropna().unique())
    available = {}
    report_rows = []

    for drug_name, (mechanism, gene) in TARGET_DRUGS.items():
        in_prism = drug_name in all_names
        if in_prism:
            drug_rows = meta[meta["name"] == drug_name]
            broad_id = drug_rows["broad_id"].iloc[0]
            available[drug_name] = {
                "mechanism": mechanism,
                "crispr_gene": gene,
                "broad_id": broad_id,
            }
        report_rows.append({
            "drug_name": drug_name,
            "mechanism": mechanism,
            "crispr_gene": gene or "",
            "in_prism": in_prism,
        })

    return available, pd.DataFrame(report_rows)


def drug_sensitivity_comparison(
    drug_sens: pd.Series,
    group_a_idx: pd.Index,
    group_b_idx: pd.Index,
) -> dict | None:
    """Compare drug sensitivity between two groups. Returns stats or None."""
    a_vals = drug_sens.reindex(group_a_idx).dropna().values
    b_vals = drug_sens.reindex(group_b_idx).dropna().values

    if len(a_vals) < MIN_SAMPLES or len(b_vals) < MIN_SAMPLES:
        return None

    d = cohens_d(a_vals, b_vals)
    _, pval = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    return {
        "cohens_d": round(d, 4),
        "p_value": float(pval),
        "n_group_a": len(a_vals),
        "n_group_b": len(b_vals),
        "median_a": round(float(np.median(a_vals)), 4),
        "median_b": round(float(np.median(b_vals)), 4),
    }


def targeted_mtap_stratified(
    prism: pd.DataFrame,
    available_drugs: dict,
    del_mtap_intact_idx: pd.Index,
    del_mtap_deleted_idx: pd.Index,
    intact_idx: pd.Index,
) -> pd.DataFrame:
    """Run MTAP-stratified targeted drug sensitivity analysis."""
    rows = []

    for drug_name, info in available_drugs.items():
        broad_id = info["broad_id"]
        matching = [idx for idx in prism.index if broad_id in str(idx)]
        if not matching:
            continue

        drug_sens = prism.loc[matching[0]]

        # Comparison A: MTAP-corrected (del/MTAP-intact vs intact)
        result_a = drug_sensitivity_comparison(drug_sens, del_mtap_intact_idx, intact_idx)
        if result_a:
            rows.append({
                "comparison": "CDKN2A-del_MTAP-intact_vs_intact",
                "drug": drug_name,
                "mechanism": info["mechanism"],
                **result_a,
            })

        # Comparison B: MTAP confound (del/MTAP-del vs del/MTAP-intact)
        result_b = drug_sensitivity_comparison(drug_sens, del_mtap_deleted_idx, del_mtap_intact_idx)
        if result_b:
            rows.append({
                "comparison": "MTAP-confound_within_CDKN2A-del",
                "drug": drug_name,
                "mechanism": info["mechanism"],
                **result_b,
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        # FDR correction within each comparison
        for comp in result["comparison"].unique():
            mask = result["comparison"] == comp
            result.loc[mask, "fdr"] = fdr_correction(result.loc[mask, "p_value"].values)
    return result


def genomewide_mtap_stratified(
    prism: pd.DataFrame,
    drug_meta: pd.DataFrame,
    del_mtap_intact_idx: pd.Index,
    del_mtap_deleted_idx: pd.Index,
    intact_idx: pd.Index,
) -> pd.DataFrame:
    """Genome-wide MTAP-stratified drug screen."""
    bid_to_name = {}
    for _, row in drug_meta.iterrows():
        bid = row.get("broad_id")
        name = row.get("name")
        if pd.notna(bid) and pd.notna(name):
            bid_to_name[bid] = name

    rows = []
    pvals_a = []
    pvals_b = []

    for treatment_id in prism.index:
        drug_sens = prism.loc[treatment_id]
        broad_id = str(treatment_id).split("::")[0] if "::" in str(treatment_id) else str(treatment_id)
        drug_name = bid_to_name.get(broad_id, "")

        # Comparison A: MTAP-corrected
        result_a = drug_sensitivity_comparison(drug_sens, del_mtap_intact_idx, intact_idx)
        if result_a:
            rows.append({
                "comparison": "CDKN2A-del_MTAP-intact_vs_intact",
                "treatment_id": treatment_id,
                "drug_name": drug_name,
                **result_a,
            })
            pvals_a.append(result_a["p_value"])

        # Comparison B: MTAP confound
        result_b = drug_sensitivity_comparison(drug_sens, del_mtap_deleted_idx, del_mtap_intact_idx)
        if result_b:
            rows.append({
                "comparison": "MTAP-confound_within_CDKN2A-del",
                "treatment_id": treatment_id,
                "drug_name": drug_name,
                **result_b,
            })
            pvals_b.append(result_b["p_value"])

    result = pd.DataFrame(rows)
    if len(result) > 0:
        for comp in result["comparison"].unique():
            mask = result["comparison"] == comp
            result.loc[mask, "fdr"] = fdr_correction(result.loc[mask, "p_value"].values)

    return result


def compute_concordance(
    prism: pd.DataFrame,
    available_drugs: dict,
    crispr: pd.DataFrame,
    del_mtap_intact_idx: pd.Index,
    intact_idx: pd.Index,
) -> pd.DataFrame:
    """CRISPR-PRISM concordance for MTAP-corrected cohort."""
    valid_lines = del_mtap_intact_idx.union(intact_idx)
    rows = []

    for drug_name, info in available_drugs.items():
        gene = info["crispr_gene"]
        if not gene or gene not in crispr.columns:
            continue

        broad_id = info["broad_id"]
        matching = [idx for idx in prism.index if broad_id in str(idx)]
        if not matching:
            continue

        drug_sens = prism.loc[matching[0]]
        gene_dep = crispr[gene]

        common = drug_sens.dropna().index.intersection(gene_dep.dropna().index)
        common = common.intersection(valid_lines)

        if len(common) < 10:
            continue

        r, p = stats.spearmanr(drug_sens[common].values, gene_dep[common].values)
        rows.append({
            "drug": drug_name,
            "gene": gene,
            "mechanism": info["mechanism"],
            "spearman_r": round(float(r), 4),
            "p_value": float(p),
            "n_lines": len(common),
        })

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4b: MTAP-Stratified PRISM Drug Sensitivity ===\n")

    # Load Phase 1 classification
    print("Loading Phase 1 classification...")
    classified = pd.read_csv(PHASE1_DIR / "cdkn2a_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()

    # Verify MTAP column
    if "MTAP_co_deleted" not in classified.columns:
        print("  MTAP annotation missing — loading from DepMap CN...")
        cn = load_depmap_matrix(DEPMAP_DIR / "PortalOmicsCNGeneLog2.csv")
        if "MTAP" not in cn.columns:
            raise RuntimeError("MTAP not found in PortalOmicsCNGeneLog2.csv")
        mtap_cn = cn["MTAP"].rename("MTAP_CN_log2")
        classified = classified.join(mtap_cn, how="left")
        classified["MTAP_co_deleted"] = classified["MTAP_CN_log2"] <= HOMDEL_CN_THRESHOLD

    # Cohort counts
    cdkn2a_del = classified[classified["CDKN2A_status"] == "deleted"]
    n_mtap_codel = int(cdkn2a_del["MTAP_co_deleted"].fillna(False).sum())
    n_mtap_intact = len(cdkn2a_del) - n_mtap_codel
    print(f"  CDKN2A-deleted: {len(cdkn2a_del)} total")
    print(f"    MTAP co-deleted: {n_mtap_codel} ({n_mtap_codel/len(cdkn2a_del):.1%})")
    print(f"    MTAP intact: {n_mtap_intact} ({n_mtap_intact/len(cdkn2a_del):.1%})")

    # Build RB1-intact cohorts (same filter as Phase 3/3b)
    classified_primary = classified[
        (classified["CDKN2A_status"] == "intact") |
        ((classified["CDKN2A_status"] == "deleted") & (classified["RB1_status"] == "intact"))
    ].copy()

    intact_all = classified_primary[classified_primary["CDKN2A_status"] == "intact"]
    del_rb1intact = classified_primary[
        (classified_primary["CDKN2A_status"] == "deleted") & (classified_primary["RB1_status"] == "intact")
    ]
    del_mtap_intact = del_rb1intact[del_rb1intact["MTAP_co_deleted"].fillna(False) == False]
    del_mtap_deleted = del_rb1intact[del_rb1intact["MTAP_co_deleted"].fillna(False) == True]

    print(f"\n  Cohort sizes (RB1-intact filter):")
    print(f"    CDKN2A-intact (control): {len(intact_all)}")
    print(f"    CDKN2A-del/RB1-intact total: {len(del_rb1intact)}")
    print(f"    CDKN2A-del/RB1-intact/MTAP-intact: {len(del_mtap_intact)}")
    print(f"    CDKN2A-del/RB1-intact/MTAP-deleted: {len(del_mtap_deleted)}")

    # Load PRISM metadata and data
    print("\nLoading PRISM 24Q2 metadata...")
    meta = pd.read_csv(DEPMAP_DIR / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")

    available, avail_report = find_available_drugs(meta)
    avail_report.to_csv(OUTPUT_DIR / "drug_availability.csv", index=False)
    n_found = avail_report["in_prism"].sum()
    print(f"  {n_found}/{len(avail_report)} target drugs found in PRISM")
    for name in available:
        print(f"    {name} ({available[name]['mechanism']})")

    print("\nLoading PRISM data matrix...")
    prism = pd.read_csv(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )
    print(f"  {prism.shape[0]} treatments x {prism.shape[1]} cell lines")

    # === Targeted MTAP-stratified drug analysis ===
    print("\n--- Targeted MTAP-Stratified Drug Analysis ---")
    if available:
        targeted = targeted_mtap_stratified(
            prism, available,
            del_mtap_intact.index, del_mtap_deleted.index, intact_all.index,
        )
        targeted.to_csv(OUTPUT_DIR / "targeted_mtap_stratified.csv", index=False)

        # Load original Phase 4 targeted results for comparison
        orig_targeted = pd.read_csv(PHASE4_DIR / "cdk46i_sensitivity.csv")

        print("\n  KEY DRUG COMPARISON: Original vs MTAP-corrected")
        print(f"  {'Drug':<20} {'Mechanism':<28} {'Orig d':>8} {'Corr d':>8} {'Confound d':>10} {'Verdict':>20}")
        print("  " + "-" * 96)

        drug_verdict_rows = []
        for drug_name in available:
            info = available[drug_name]

            # Original (pan-cancer, stratum=all)
            orig_row = orig_targeted[
                (orig_targeted["drug"] == drug_name) &
                (orig_targeted["cancer_type"] == "Pan-cancer (pooled)") &
                (orig_targeted["stratum"] == "all")
            ]
            orig_d = float(orig_row["cohens_d"].iloc[0]) if len(orig_row) > 0 else float("nan")
            orig_fdr = float(orig_row["fdr"].iloc[0]) if len(orig_row) > 0 and "fdr" in orig_row.columns else float("nan")

            # MTAP-corrected
            corr_row = targeted[
                (targeted["drug"] == drug_name) &
                (targeted["comparison"] == "CDKN2A-del_MTAP-intact_vs_intact")
            ]
            corr_d = float(corr_row["cohens_d"].iloc[0]) if len(corr_row) > 0 else float("nan")
            corr_fdr = float(corr_row["fdr"].iloc[0]) if len(corr_row) > 0 else float("nan")

            # Confound check
            conf_row = targeted[
                (targeted["drug"] == drug_name) &
                (targeted["comparison"] == "MTAP-confound_within_CDKN2A-del")
            ]
            conf_d = float(conf_row["cohens_d"].iloc[0]) if len(conf_row) > 0 else float("nan")
            conf_fdr = float(conf_row["fdr"].iloc[0]) if len(conf_row) > 0 else float("nan")

            # Verdict
            if np.isnan(orig_d):
                verdict = "NO DATA"
            elif abs(orig_d) < EFFECT_SIZE_THRESHOLD:
                verdict = "not significant"
            elif not np.isnan(conf_d) and abs(conf_d) > EFFECT_SIZE_THRESHOLD and conf_fdr < FDR_THRESHOLD:
                verdict = "MTAP-DRIVEN"
            elif not np.isnan(corr_d) and abs(corr_d) > EFFECT_SIZE_THRESHOLD and corr_fdr < FDR_THRESHOLD:
                verdict = "CDKN2A-SPECIFIC"
            elif not np.isnan(corr_d) and abs(corr_d) > EFFECT_SIZE_THRESHOLD:
                verdict = "trend (FDR ns)"
            else:
                verdict = "attenuated"

            print(f"  {drug_name:<20} {info['mechanism']:<28} {orig_d:>8.3f} {corr_d:>8.3f} {conf_d:>10.3f} {verdict:>20}")

            drug_verdict_rows.append({
                "drug": drug_name,
                "mechanism": info["mechanism"],
                "original_d": round(orig_d, 4) if not np.isnan(orig_d) else None,
                "original_fdr": orig_fdr if not np.isnan(orig_fdr) else None,
                "mtap_corrected_d": round(corr_d, 4) if not np.isnan(corr_d) else None,
                "mtap_corrected_fdr": corr_fdr if not np.isnan(corr_fdr) else None,
                "mtap_confound_d": round(conf_d, 4) if not np.isnan(conf_d) else None,
                "mtap_confound_fdr": conf_fdr if not np.isnan(conf_fdr) else None,
                "verdict": verdict,
            })

        pd.DataFrame(drug_verdict_rows).to_csv(OUTPUT_DIR / "drug_verdict_comparison.csv", index=False)
    else:
        targeted = pd.DataFrame()
        drug_verdict_rows = []
        print("  No target drugs found — skipping.")

    # === Genome-wide MTAP-stratified drug screen ===
    print("\n--- Genome-Wide MTAP-Stratified Drug Screen ---")
    genomewide = genomewide_mtap_stratified(
        prism, meta,
        del_mtap_intact.index, del_mtap_deleted.index, intact_all.index,
    )
    genomewide.to_csv(OUTPUT_DIR / "genomewide_mtap_stratified.csv", index=False)

    # Load original Phase 4 genome-wide results for comparison
    orig_gw = pd.read_csv(PHASE4_DIR / "genomewide_drug_hits.csv")
    orig_gained = set(
        orig_gw[
            (orig_gw["fdr"] < FDR_THRESHOLD) &
            (orig_gw["cohens_d"] < -EFFECT_SIZE_THRESHOLD)
        ]["treatment_id"].values
    )

    # MTAP-corrected gained drugs
    gw_comp_a = genomewide[genomewide["comparison"] == "CDKN2A-del_MTAP-intact_vs_intact"]
    corr_gained = set(
        gw_comp_a[
            (gw_comp_a["fdr"] < FDR_THRESHOLD) &
            (gw_comp_a["cohens_d"] < -EFFECT_SIZE_THRESHOLD)
        ]["treatment_id"].values
    ) if len(gw_comp_a) > 0 else set()

    # MTAP confound drugs
    gw_comp_b = genomewide[genomewide["comparison"] == "MTAP-confound_within_CDKN2A-del"]
    mtap_driven = set(
        gw_comp_b[
            (gw_comp_b["fdr"] < FDR_THRESHOLD) &
            (gw_comp_b["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
        ]["treatment_id"].values
    ) if len(gw_comp_b) > 0 else set()

    survived = orig_gained & corr_gained
    lost = orig_gained - corr_gained
    new_hits = corr_gained - orig_gained
    confirmed_mtap = orig_gained & mtap_driven

    print(f"\n  Original CDKN2A-selective drugs: {len(orig_gained)}")
    print(f"  Survived MTAP correction: {len(survived)}")
    print(f"  Lost after correction: {len(lost)}")
    print(f"  New after correction: {len(new_hits)}")
    print(f"  Confirmed MTAP-driven: {len(confirmed_mtap)}")

    # Build name lookup
    bid_to_name = {}
    for _, row in meta.iterrows():
        bid = row.get("broad_id")
        name = row.get("name")
        if pd.notna(bid) and pd.notna(name):
            bid_to_name[bid] = name

    def get_drug_name(tid: str) -> str:
        bid = str(tid).split("::")[0] if "::" in str(tid) else str(tid)
        return bid_to_name.get(bid, str(tid))

    if survived:
        print(f"\n  SURVIVED (CDKN2A-specific drug sensitivities):")
        for tid in sorted(survived):
            row_a = gw_comp_a[gw_comp_a["treatment_id"] == tid].iloc[0]
            print(f"    {get_drug_name(tid)}: d={row_a['cohens_d']:.3f}, FDR={row_a['fdr']:.3e}")

    if lost:
        print(f"\n  LOST after correction:")
        for tid in sorted(lost):
            mtap_flag = " [confirmed MTAP-driven]" if tid in confirmed_mtap else ""
            orig_row = orig_gw[orig_gw["treatment_id"] == tid]
            orig_d = float(orig_row["cohens_d"].iloc[0]) if len(orig_row) > 0 else float("nan")
            print(f"    {get_drug_name(tid)}: orig d={orig_d:.3f}{mtap_flag}")

    # === CRISPR-PRISM concordance (MTAP-corrected) ===
    if available:
        print("\n--- CRISPR-PRISM Concordance (MTAP-corrected cohort) ---")
        crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
        concordance = compute_concordance(
            prism, available, crispr,
            del_mtap_intact.index, intact_all.index,
        )
        concordance.to_csv(OUTPUT_DIR / "crispr_prism_concordance_mtap_corrected.csv", index=False)

        # Load original concordance for comparison
        orig_conc = pd.read_csv(PHASE4_DIR / "crispr_prism_concordance.csv")

        if len(concordance) > 0:
            print(f"\n  {'Drug':<20} {'Gene':<8} {'Orig r':>8} {'Corr r':>8} {'p':>12} {'n':>5}")
            print("  " + "-" * 62)
            for _, row in concordance.iterrows():
                orig_r_row = orig_conc[orig_conc["drug"] == row["drug"]]
                orig_r = float(orig_r_row["spearman_r"].iloc[0]) if len(orig_r_row) > 0 else float("nan")
                print(f"  {row['drug']:<20} {row['gene']:<8} {orig_r:>8.3f} {row['spearman_r']:>8.3f} "
                      f"{row['p_value']:>12.3e} {row['n_lines']:>5}")
    else:
        concordance = pd.DataFrame()

    # === Search for WWTR1/Hippo pathway drugs ===
    print("\n--- WWTR1/Hippo Pathway Drug Search ---")
    hippo_keywords = ["verteporfin", "TEAD", "hippo", "YAP", "TAZ"]
    all_drug_names = meta["name"].dropna().unique()
    hippo_drugs_found = []
    for kw in hippo_keywords:
        matches = [n for n in all_drug_names if kw.lower() in n.lower()]
        hippo_drugs_found.extend(matches)
    hippo_drugs_found = sorted(set(hippo_drugs_found))

    if hippo_drugs_found:
        print(f"  Found {len(hippo_drugs_found)} WWTR1/Hippo-related drugs: {hippo_drugs_found}")
        # Test each for MTAP-corrected CDKN2A sensitivity
        for dname in hippo_drugs_found:
            drug_rows = meta[meta["name"] == dname]
            if len(drug_rows) == 0:
                continue
            broad_id = drug_rows["broad_id"].iloc[0]
            matching = [idx for idx in prism.index if broad_id in str(idx)]
            if not matching:
                print(f"    {dname}: no PRISM data")
                continue

            drug_sens = prism.loc[matching[0]]
            result = drug_sensitivity_comparison(drug_sens, del_mtap_intact.index, intact_all.index)
            if result:
                print(f"    {dname}: d={result['cohens_d']:.3f}, n={result['n_group_a']}+{result['n_group_b']}")
            else:
                print(f"    {dname}: insufficient data")
    else:
        print("  No WWTR1/Hippo pathway drugs found in PRISM 24Q2")

    # === Summary sets ===
    summary_data = {
        "survived_cdkn2a_specific": sorted(str(t) for t in survived),
        "lost_after_mtap_correction": sorted(str(t) for t in lost),
        "confirmed_mtap_driven": sorted(str(t) for t in confirmed_mtap),
        "new_after_mtap_correction": sorted(str(t) for t in new_hits),
        "drug_verdicts": drug_verdict_rows,
    }
    with open(OUTPUT_DIR / "mtap_correction_sets.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # === Write summary text ===
    summary_lines = [
        "=" * 80,
        "CDKN2A Pan-Cancer Dependency Atlas - Phase 4b: MTAP-Stratified PRISM Drug Sensitivity",
        "=" * 80,
        "",
        "PURPOSE: Deconfound CDKN2A-specific from MTAP-driven drug sensitivities.",
        f"MTAP co-deletion rate in CDKN2A-deleted lines: {n_mtap_codel}/{len(cdkn2a_del)} ({n_mtap_codel/len(cdkn2a_del):.1%})",
        "",
        "COHORT SIZES (RB1-intact filter):",
        f"  CDKN2A-intact (control): {len(intact_all)}",
        f"  CDKN2A-del/MTAP-intact: {len(del_mtap_intact)}",
        f"  CDKN2A-del/MTAP-deleted: {len(del_mtap_deleted)}",
        "",
        "TARGETED DRUG VERDICTS:",
    ]
    for dv in drug_verdict_rows:
        summary_lines.append(f"  {dv['drug']} ({dv['mechanism']}): {dv['verdict']}")

    summary_lines += [
        "",
        "GENOME-WIDE DRUG SCREEN RESULTS:",
        f"  Original CDKN2A-selective drugs: {len(orig_gained)}",
        f"  Survived MTAP correction: {len(survived)}",
        f"  Lost after correction: {len(lost)}",
        f"  Confirmed MTAP-driven: {len(confirmed_mtap)}",
        f"  New after correction: {len(new_hits)}",
    ]

    if survived:
        summary_lines.append("\nSURVIVED (genuinely CDKN2A-specific):")
        for tid in sorted(survived):
            row_a = gw_comp_a[gw_comp_a["treatment_id"] == tid].iloc[0]
            summary_lines.append(f"  {get_drug_name(tid)}: d={row_a['cohens_d']:.3f}, FDR={row_a['fdr']:.3e}")

    if hippo_drugs_found:
        summary_lines += ["", "WWTR1/HIPPO PATHWAY DRUGS IN PRISM:"]
        for dname in hippo_drugs_found:
            summary_lines.append(f"  {dname}")

    if len(concordance) > 0:
        summary_lines += ["", "CRISPR-PRISM CONCORDANCE (MTAP-corrected):"]
        for _, row in concordance.iterrows():
            summary_lines.append(
                f"  {row['drug']} vs {row['gene']}: r={row['spearman_r']:.3f}, p={row['p_value']:.3e}"
            )

    summary_lines.append("")

    with open(OUTPUT_DIR / "phase4b_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nOutputs saved to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
