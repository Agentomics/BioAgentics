"""Phase 4b: Targeted PRISM analysis for RD-prioritized axes.

1. DHODH inhibitors (BAY-2402234) — UXS1/pyrimidine vulnerability axis
2. ATR inhibitors (ATR inhibitor 2) — replication stress axis
3. KL-subtype (STK11+KRAS co-mutant) PRISM subgroup analysis
4. UGDH dependency from Phase 2 CRISPR data
5. NRF2 transcriptional activity score vs UXS1 Chronos dependency correlation

Usage:
    uv run python -m keap1_nrf2_pancancer_dependency_atlas.05_targeted_drug_analysis
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
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase1"
)
PHASE2_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase2"
)
OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase4"
)

MIN_SAMPLES = 3

# Target drugs present in PRISM 24Q2
TARGET_DRUGS = {
    "BAY-2402234": {
        "mechanism": "DHODH inhibitor",
        "axis": "UXS1/pyrimidine",
        "broad_id": "BRD-K22379844",
    },
    "ATR inhibitor 2": {
        "mechanism": "ATR inhibitor",
        "axis": "ATR/replication-stress",
        "broad_id": "BRD-K00091072",
    },
}

# Drugs we searched for but are NOT in PRISM 24Q2
NOT_IN_PRISM = [
    ("brequinar", "DHODH inhibitor"),
    ("ceralasertib (AZD6738)", "ATR inhibitor"),
    ("berzosertib (M6620)", "ATR inhibitor"),
    ("elimusertib (BAY1895344)", "ATR inhibitor"),
    ("prexasertib", "CHK1 inhibitor"),
    ("SRA737", "CHK1 inhibitor"),
]

# NRF2 transcriptional activity signature genes
NRF2_SIGNATURE = ["NQO1", "GCLM", "TXNRD1", "HMOX1", "AKR1C1"]


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


def _find_prism_row(prism: pd.DataFrame, broad_key: str) -> str | None:
    """Find the PRISM matrix row index matching a compound key."""
    for idx in prism.index:
        if broad_key in str(idx):
            return idx
    return None


def targeted_drug_sensitivity(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
) -> pd.DataFrame:
    """Test targeted drugs for KEAP1/NRF2-selective sensitivity."""
    altered_lines = classified[classified["KEAP1_NRF2_altered"] == True].index  # noqa: E712
    wt_lines = classified[classified["pathway_status"] == "WT"].index

    rows = []
    for drug_name, info in TARGET_DRUGS.items():
        prism_row = _find_prism_row(prism, info["broad_id"])
        if prism_row is None:
            print(f"  WARNING: {drug_name} not found in PRISM matrix")
            continue

        drug_sens = prism.loc[prism_row]

        # Pan-cancer
        alt_vals = drug_sens.reindex(altered_lines).dropna().values
        wt_vals = drug_sens.reindex(wt_lines).dropna().values

        if len(alt_vals) >= MIN_SAMPLES and len(wt_vals) >= MIN_SAMPLES:
            d = cohens_d(alt_vals, wt_vals)
            _, pval = stats.ttest_ind(alt_vals, wt_vals, equal_var=False)
            rows.append({
                "drug": drug_name,
                "mechanism": info["mechanism"],
                "axis": info["axis"],
                "context": "Pan-cancer",
                "cohens_d": round(d, 4),
                "p_value": float(pval),
                "n_mut": len(alt_vals),
                "n_wt": len(wt_vals),
                "median_sens_mut": round(float(np.median(alt_vals)), 4),
                "median_sens_wt": round(float(np.median(wt_vals)), 4),
            })

        # Per lineage (Lung is the key one)
        for lineage in ["Lung", "Bladder/Urinary Tract", "Head and Neck",
                        "Esophagus/Stomach", "Uterus"]:
            ct_data = classified[classified["OncotreeLineage"] == lineage]
            ct_alt = ct_data[ct_data["KEAP1_NRF2_altered"] == True].index  # noqa: E712
            ct_wt = ct_data[ct_data["pathway_status"] == "WT"].index

            ct_alt_vals = drug_sens.reindex(ct_alt).dropna().values
            ct_wt_vals = drug_sens.reindex(ct_wt).dropna().values

            if len(ct_alt_vals) >= MIN_SAMPLES and len(ct_wt_vals) >= MIN_SAMPLES:
                d = cohens_d(ct_alt_vals, ct_wt_vals)
                _, pval = stats.ttest_ind(ct_alt_vals, ct_wt_vals, equal_var=False)
                rows.append({
                    "drug": drug_name,
                    "mechanism": info["mechanism"],
                    "axis": info["axis"],
                    "context": lineage,
                    "cohens_d": round(d, 4),
                    "p_value": float(pval),
                    "n_mut": len(ct_alt_vals),
                    "n_wt": len(ct_wt_vals),
                    "median_sens_mut": round(float(np.median(ct_alt_vals)), 4),
                    "median_sens_wt": round(float(np.median(ct_wt_vals)), 4),
                })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["p_value"].values)
    return result


def kl_subtype_drug_screen(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    drug_name_map: dict[str, str],
) -> pd.DataFrame:
    """Genome-wide PRISM screen for KL-subtype (STK11+KRAS co-mutant) vs KEAP1-WT.

    Matches the HUDSON trial population for translational relevance.
    """
    kl_lines = classified[classified["KL_subtype"] == True].index  # noqa: E712
    wt_lines = classified[classified["pathway_status"] == "WT"].index

    rows = []
    pvals = []

    for treatment_id in prism.index:
        drug_sens = prism.loc[treatment_id]
        kl_vals = drug_sens.reindex(kl_lines).dropna().values
        wt_vals = drug_sens.reindex(wt_lines).dropna().values

        if len(kl_vals) < MIN_SAMPLES or len(wt_vals) < MIN_SAMPLES:
            continue

        d = cohens_d(kl_vals, wt_vals)
        _, pval = stats.ttest_ind(kl_vals, wt_vals, equal_var=False)

        compound_key = re.search(r"BRD-[A-Za-z]\d{8}", str(treatment_id))
        ckey = compound_key.group(0) if compound_key else str(treatment_id)
        drug_name = drug_name_map.get(ckey, "")

        rows.append({
            "treatment_id": treatment_id,
            "compound": drug_name,
            "compound_key": ckey,
            "d": round(d, 4),
            "pvalue": float(pval),
            "n_kl": len(kl_vals),
            "n_wt": len(wt_vals),
        })
        pvals.append(pval)

    result = pd.DataFrame(rows)
    if len(result) > 0 and pvals:
        result["qvalue"] = fdr_correction(np.array(pvals))
    return result


def check_ugdh_dependency(phase2_dir: Path) -> dict:
    """Check UGDH dependency scores from Phase 2 Chronos genome-wide screen."""
    gw = pd.read_csv(phase2_dir / "genomewide_all_results.csv")
    ugdh = gw[gw["gene"] == "UGDH"]
    results = {}
    for _, row in ugdh.iterrows():
        results[row["cancer_type"]] = {
            "cohens_d": row["cohens_d"],
            "fdr": row.get("fdr", float("nan")),
            "n_altered": row["n_altered"],
            "n_wt": row["n_wt"],
        }
    return results


def nrf2_activity_vs_uxs1(classified: pd.DataFrame) -> dict:
    """Correlate NRF2 transcriptional activity score with UXS1 Chronos dependency."""
    # Load expression data for NRF2 signature genes
    print("  Loading expression data for NRF2 signature...")
    expr = load_depmap_matrix(DEPMAP_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
    available_sig = [g for g in NRF2_SIGNATURE if g in expr.columns]
    if not available_sig:
        return {"error": "No NRF2 signature genes found in expression data"}

    # Compute NRF2 activity score (mean expression of signature genes)
    nrf2_score = expr[available_sig].mean(axis=1)
    nrf2_score.name = "NRF2_activity"

    # Load CRISPR data for UXS1
    print("  Loading CRISPR data for UXS1...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    if "UXS1" not in crispr.columns:
        return {"error": "UXS1 not found in CRISPR data"}

    uxs1_dep = crispr["UXS1"]

    # Find common lines
    common = nrf2_score.dropna().index.intersection(uxs1_dep.dropna().index)
    common = common.intersection(classified.index)

    if len(common) < 10:
        return {"error": f"Too few common lines: {len(common)}"}

    # Correlation (all lines)
    r_all, p_all = stats.spearmanr(nrf2_score[common].values, uxs1_dep[common].values)

    # Subset: only KEAP1/NRF2-altered lines
    alt_common = common.intersection(classified[classified["KEAP1_NRF2_altered"] == True].index)  # noqa: E712
    r_alt, p_alt = float("nan"), float("nan")
    if len(alt_common) >= 10:
        r_alt, p_alt = stats.spearmanr(nrf2_score[alt_common].values, uxs1_dep[alt_common].values)

    return {
        "n_all": len(common),
        "spearman_r_all": round(float(r_all), 4),
        "p_value_all": float(p_all),
        "n_altered": len(alt_common),
        "spearman_r_altered": round(float(r_alt), 4) if not np.isnan(r_alt) else None,
        "p_value_altered": float(p_alt) if not np.isnan(p_alt) else None,
        "signature_genes_used": available_sig,
    }


def plot_nrf2_vs_uxs1(classified: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of NRF2 activity vs UXS1 dependency."""
    expr = load_depmap_matrix(DEPMAP_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    available_sig = [g for g in NRF2_SIGNATURE if g in expr.columns]
    if not available_sig or "UXS1" not in crispr.columns:
        return

    nrf2_score = expr[available_sig].mean(axis=1)
    uxs1_dep = crispr["UXS1"]

    common = nrf2_score.dropna().index.intersection(uxs1_dep.dropna().index)
    common = common.intersection(classified.index)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by KEAP1/NRF2 status
    is_alt = classified.loc[common, "KEAP1_NRF2_altered"]
    wt_mask = ~is_alt
    alt_mask = is_alt

    ax.scatter(nrf2_score[common][wt_mask], uxs1_dep[common][wt_mask],
               c="#CCCCCC", s=10, alpha=0.5, label="WT")
    ax.scatter(nrf2_score[common][alt_mask], uxs1_dep[common][alt_mask],
               c="#D95319", s=20, alpha=0.8, label="KEAP1/NRF2-altered")

    r, p = stats.spearmanr(nrf2_score[common].values, uxs1_dep[common].values)
    ax.set_xlabel("NRF2 Activity Score (mean signature expression)")
    ax.set_ylabel("UXS1 Chronos Dependency")
    ax.set_title(f"NRF2 Activity vs UXS1 Dependency\n(Spearman r={r:.3f}, p={p:.2e}, n={len(common)})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "nrf2_activity_vs_uxs1.png", dpi=150)
    plt.close(fig)


def write_summary(
    targeted: pd.DataFrame,
    kl_screen: pd.DataFrame,
    ugdh_results: dict,
    nrf2_uxs1: dict,
    output_dir: Path,
) -> None:
    """Write Markdown summary."""
    lines = [
        "# KEAP1/NRF2 Pan-Cancer Dependency Atlas - Phase 4b: Targeted Drug Analysis",
        "",
        "## 1. Targeted Drug Sensitivity (KEAP1/NRF2-altered vs WT)",
        "",
    ]

    # Drug availability
    lines += [
        "### Drug Availability in PRISM 24Q2",
        "",
        "| Drug | Mechanism | Status |",
        "|------|-----------|--------|",
    ]
    for drug_name, info in TARGET_DRUGS.items():
        lines.append(f"| {drug_name} | {info['mechanism']} | **FOUND** |")
    for drug_name, mech in NOT_IN_PRISM:
        lines.append(f"| {drug_name} | {mech} | NOT FOUND |")

    if len(targeted) > 0:
        lines += [
            "",
            "### Sensitivity Results",
            "",
            "| Drug | Mechanism | Context | d | p-value | FDR | n_mut | n_wt |",
            "|------|-----------|---------|---|---------|-----|-------|------|",
        ]
        for _, row in targeted.iterrows():
            sig = " *" if row.get("fdr", 1) < 0.05 else ""
            lines.append(
                f"| {row['drug']} | {row['mechanism']} | {row['context']} | "
                f"{row['cohens_d']:.3f} | {row['p_value']:.2e} | "
                f"{row.get('fdr', float('nan')):.2e}{sig} | {row['n_mut']} | {row['n_wt']} |"
            )
    else:
        lines.append("\nNo targeted drug results available.")

    # KL-subtype
    lines += [
        "",
        "## 2. KL-Subtype PRISM Screen (STK11+KRAS co-mutant vs WT)",
        "",
    ]
    if len(kl_screen) > 0 and "qvalue" in kl_screen.columns:
        sig_kl = kl_screen[(kl_screen["qvalue"] < 0.1) & (kl_screen["d"].abs() > 0.3)]
        sensitizing_kl = sig_kl[sig_kl["d"] < 0].sort_values("d")
        lines.append(f"- Compounds screened: **{len(kl_screen)}**")
        lines.append(f"- Significant (|d|>0.3, FDR<0.1): **{len(sig_kl)}**")
        lines.append(f"- Sensitizing in KL-subtype: **{len(sensitizing_kl)}**")

        if len(sensitizing_kl) > 0:
            lines += [
                "",
                "| Rank | Compound | d | FDR | n_kl | n_wt |",
                "|------|----------|---|-----|------|------|",
            ]
            for i, (_, row) in enumerate(sensitizing_kl.head(20).iterrows(), 1):
                name = row["compound"] if row["compound"] else row["compound_key"]
                lines.append(
                    f"| {i} | {name} | {row['d']:.3f} | {row['qvalue']:.2e} | "
                    f"{row['n_kl']} | {row['n_wt']} |"
                )
        # Top hits by raw effect size
        top_raw = kl_screen[kl_screen["d"] < 0].nsmallest(10, "d")
        if len(top_raw) > 0:
            lines += [
                "",
                "### Top KL-subtype sensitizers (by effect size, uncorrected)",
                "",
                "| Compound | d | p-value | FDR | n_kl |",
                "|----------|---|---------|-----|------|",
            ]
            for _, row in top_raw.iterrows():
                name = row["compound"] if row["compound"] else row["compound_key"]
                lines.append(
                    f"| {name} | {row['d']:.3f} | {row['pvalue']:.2e} | "
                    f"{row['qvalue']:.2e} | {row['n_kl']} |"
                )
    else:
        lines.append("Insufficient KL-subtype lines in PRISM for screening.")

    # UGDH dependency
    lines += [
        "",
        "## 3. UGDH CRISPR Dependency (Phase 2 Chronos Data)",
        "",
    ]
    if ugdh_results:
        lines += [
            "| Context | d | FDR | n_altered | n_wt |",
            "|---------|---|-----|-----------|------|",
        ]
        for ctx, vals in ugdh_results.items():
            lines.append(
                f"| {ctx} | {vals['cohens_d']:.3f} | {vals['fdr']:.3e} | "
                f"{vals['n_altered']} | {vals['n_wt']} |"
            )
    else:
        lines.append("UGDH not found in Phase 2 results.")

    # NRF2 activity vs UXS1
    lines += [
        "",
        "## 4. NRF2 Transcriptional Activity vs UXS1 Dependency",
        "",
    ]
    if "error" in nrf2_uxs1:
        lines.append(f"Error: {nrf2_uxs1['error']}")
    else:
        lines += [
            f"- Signature genes: {', '.join(nrf2_uxs1['signature_genes_used'])}",
            f"- All lines: Spearman r={nrf2_uxs1['spearman_r_all']:.3f}, "
            f"p={nrf2_uxs1['p_value_all']:.2e}, n={nrf2_uxs1['n_all']}",
        ]
        if nrf2_uxs1.get("spearman_r_altered") is not None:
            lines.append(
                f"- KEAP1/NRF2-altered only: Spearman r={nrf2_uxs1['spearman_r_altered']:.3f}, "
                f"p={nrf2_uxs1['p_value_altered']:.2e}, n={nrf2_uxs1['n_altered']}"
            )

    lines += [
        "",
        "## 5. Convergence Summary",
        "",
        "The UXS1/pyrimidine and ATR/replication-stress axes are the two priority",
        "therapeutic hypotheses from Phases 1-3. This analysis tests them with",
        "pharmacological (PRISM drug sensitivity) and transcriptional (NRF2 activity",
        "score) data to assess drug-target convergence.",
        "",
    ]

    with open(output_dir / "phase4b_targeted_summary.md", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4b: Targeted Drug Analysis (KEAP1/NRF2) ===\n")

    # Load Phase 1 classification
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "keap1_nrf2_classification.csv", index_col=0)
    n_altered = (classified["KEAP1_NRF2_altered"] == True).sum()  # noqa: E712
    n_kl = (classified["KL_subtype"] == True).sum()  # noqa: E712
    print(f"  {n_altered} altered, {n_kl} KL-subtype lines")

    # Load PRISM treatment metadata for name mapping
    print("\nLoading PRISM metadata...")
    meta = pd.read_csv(DEPMAP_DIR / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")
    drug_name_map: dict[str, str] = {}
    for _, row in meta.iterrows():
        bid = row.get("broad_id")
        name = row.get("name")
        if pd.notna(bid) and pd.notna(name):
            m = re.search(r"BRD-[A-Za-z]\d{8}", str(bid))
            if m:
                drug_name_map.setdefault(m.group(0), str(name))

    # Load PRISM matrix
    print("Loading PRISM data matrix...")
    prism = pd.read_csv(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )
    print(f"  {prism.shape[0]} treatments x {prism.shape[1]} cell lines")

    # 1. Targeted drug sensitivity
    print("\n--- 1. Targeted Drug Sensitivity ---")
    targeted = targeted_drug_sensitivity(prism, classified)
    if len(targeted) > 0:
        targeted.to_csv(OUTPUT_DIR / "targeted_drug_sensitivity.csv", index=False)
        for _, row in targeted[targeted["context"] == "Pan-cancer"].iterrows():
            sig = " *" if row.get("fdr", 1) < 0.05 else ""
            print(f"  {row['drug']} ({row['axis']}): d={row['cohens_d']:.3f}, "
                  f"p={row['p_value']:.2e}{sig}")
    else:
        print("  No targeted drug results.")

    # 2. KL-subtype drug screen
    print("\n--- 2. KL-Subtype PRISM Screen ---")
    kl_in_prism = set(classified[classified["KL_subtype"] == True].index) & set(prism.columns)  # noqa: E712
    print(f"  KL-subtype lines in PRISM: {len(kl_in_prism)}")
    kl_screen = pd.DataFrame()
    if len(kl_in_prism) >= MIN_SAMPLES:
        kl_screen = kl_subtype_drug_screen(prism, classified, drug_name_map)
        if len(kl_screen) > 0:
            kl_screen.to_csv(OUTPUT_DIR / "kl_subtype_drug_screen.csv", index=False)
            if "qvalue" in kl_screen.columns:
                sig_kl = kl_screen[(kl_screen["qvalue"] < 0.1) & (kl_screen["d"].abs() > 0.3)]
                print(f"  {len(kl_screen)} compounds screened, {len(sig_kl)} significant")
                top_sens = kl_screen[kl_screen["d"] < 0].nsmallest(5, "d")
                for _, row in top_sens.iterrows():
                    name = row["compound"] if row["compound"] else row["compound_key"]
                    print(f"    {name}: d={row['d']:.3f}, FDR={row['qvalue']:.2e}")
    else:
        print(f"  Insufficient KL-subtype lines ({len(kl_in_prism)}) — skipping screen.")

    # 3. UGDH dependency check
    print("\n--- 3. UGDH CRISPR Dependency ---")
    ugdh_results = check_ugdh_dependency(PHASE2_DIR)
    if ugdh_results:
        for ctx, vals in ugdh_results.items():
            sig = " ***" if vals["fdr"] < 0.05 else ""
            print(f"  {ctx}: d={vals['cohens_d']:.3f}, FDR={vals['fdr']:.3e}{sig}")
    else:
        print("  UGDH not found in Phase 2 results.")

    # 4. NRF2 activity vs UXS1 dependency
    print("\n--- 4. NRF2 Activity vs UXS1 Dependency ---")
    nrf2_uxs1 = nrf2_activity_vs_uxs1(classified)
    if "error" not in nrf2_uxs1:
        print(f"  All lines: r={nrf2_uxs1['spearman_r_all']:.3f}, "
              f"p={nrf2_uxs1['p_value_all']:.2e}, n={nrf2_uxs1['n_all']}")
        if nrf2_uxs1.get("spearman_r_altered") is not None:
            print(f"  Altered only: r={nrf2_uxs1['spearman_r_altered']:.3f}, "
                  f"p={nrf2_uxs1['p_value_altered']:.2e}, n={nrf2_uxs1['n_altered']}")
        print("  Generating scatter plot...")
        plot_nrf2_vs_uxs1(classified, OUTPUT_DIR)
        print("  nrf2_activity_vs_uxs1.png")
    else:
        print(f"  {nrf2_uxs1['error']}")

    # Save targeted results as parquet
    if len(targeted) > 0:
        targeted.to_parquet(OUTPUT_DIR / "phase4_targeted_drugs.parquet", index=False)
        print("\n  phase4_targeted_drugs.parquet")

    # Write summary
    print("\nWriting summary...")
    write_summary(targeted, kl_screen, ugdh_results, nrf2_uxs1, OUTPUT_DIR)
    print("  phase4b_targeted_summary.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
