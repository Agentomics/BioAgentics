"""Phase 4: PRISM drug sensitivity validation for CDKN2A-deleted lines.

Tests CDK4/6 inhibitors, MDM2 inhibitors, and PRMT5 inhibitors for
CDKN2A-selective sensitivity. Runs genome-wide drug screen. Tests
CRISPR-PRISM concordance for CDK4/6 gene-drug pairs.

Usage:
    uv run python -m cdkn2a_pancancer_dependency_atlas.04_prism_drug_sensitivity
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
PHASE1_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase4"

# Target drugs to search for: name -> (mechanism, CRISPR gene)
TARGET_DRUGS = {
    "trilaciclib": ("CDK4/6 inhibitor", "CDK6"),
    "milademetan": ("MDM2 inhibitor", "MDM2"),
    "GSK3326595": ("PRMT5 inhibitor (type I)", "PRMT5"),
    "JNJ-64619178": ("PRMT5 inhibitor (type II)", "PRMT5"),
    "EPZ020411": ("PRMT5 inhibitor (SAM-competitive)", "PRMT5"),
    # Expected but may be absent
    "palbociclib": ("CDK4/6 inhibitor", "CDK4"),
    "ribociclib": ("CDK4/6 inhibitor", "CDK4"),
    "abemaciclib": ("CDK4/6 inhibitor", "CDK4"),
    "idasanutlin": ("MDM2 inhibitor", "MDM2"),
}

MIN_SAMPLES = 3


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


def compute_targeted_sensitivity(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    available_drugs: dict,
    qualifying_types: list[str],
) -> pd.DataFrame:
    """Compare targeted drug sensitivity in CDKN2A-deleted vs intact.

    Stratifies by RB1 status for CDK4/6 inhibitors.
    Restricts MDM2 inhibitor analysis to TP53-WT lines.
    """
    rows = []

    for drug_name, info in available_drugs.items():
        broad_id = info["broad_id"]
        matching_rows = [idx for idx in prism.index if broad_id in str(idx)]
        if not matching_rows:
            continue

        drug_sens = prism.loc[matching_rows[0]]

        # Determine strata based on drug type
        is_cdk46 = "CDK4/6" in info["mechanism"]
        is_mdm2 = "MDM2" in info["mechanism"]

        strata = [("all", classified)]
        if is_cdk46:
            # RB1 stratification
            rb1_intact = classified[
                (classified["CDKN2A_status"] == "intact") |
                ((classified["CDKN2A_status"] == "deleted") & (classified["RB1_status"] == "intact"))
            ]
            strata.append(("rb1_intact", rb1_intact))
        if is_mdm2:
            # TP53-WT only
            tp53_wt = classified[classified["TP53_status"] == "WT"]
            strata.append(("tp53_wt", tp53_wt))

        for stratum_name, stratum_data in strata:
            # Pan-cancer
            del_lines = stratum_data[stratum_data["CDKN2A_status"] == "deleted"].index
            intact_lines = stratum_data[stratum_data["CDKN2A_status"] == "intact"].index

            del_vals = drug_sens.reindex(del_lines).dropna().values
            intact_vals = drug_sens.reindex(intact_lines).dropna().values

            if len(del_vals) >= MIN_SAMPLES and len(intact_vals) >= MIN_SAMPLES:
                d = cohens_d(del_vals, intact_vals)
                _, pval = stats.mannwhitneyu(del_vals, intact_vals, alternative="two-sided")
                rows.append({
                    "cancer_type": "Pan-cancer (pooled)",
                    "drug": drug_name,
                    "mechanism": info["mechanism"],
                    "stratum": stratum_name,
                    "cohens_d": round(d, 4),
                    "p_value": float(pval),
                    "n_del": len(del_vals),
                    "n_intact": len(intact_vals),
                    "median_sens_del": round(float(np.median(del_vals)), 4),
                    "median_sens_intact": round(float(np.median(intact_vals)), 4),
                })

            # Per cancer type
            for ct in qualifying_types:
                ct_data = stratum_data[stratum_data["OncotreeLineage"] == ct]
                del_ct = ct_data[ct_data["CDKN2A_status"] == "deleted"].index
                intact_ct = ct_data[ct_data["CDKN2A_status"] == "intact"].index

                del_v = drug_sens.reindex(del_ct).dropna().values
                intact_v = drug_sens.reindex(intact_ct).dropna().values

                if len(del_v) >= MIN_SAMPLES and len(intact_v) >= MIN_SAMPLES:
                    d_ct = cohens_d(del_v, intact_v)
                    _, pval_ct = stats.mannwhitneyu(del_v, intact_v, alternative="two-sided")
                    rows.append({
                        "cancer_type": ct,
                        "drug": drug_name,
                        "mechanism": info["mechanism"],
                        "stratum": stratum_name,
                        "cohens_d": round(d_ct, 4),
                        "p_value": float(pval_ct),
                        "n_del": len(del_v),
                        "n_intact": len(intact_v),
                        "median_sens_del": round(float(np.median(del_v)), 4),
                        "median_sens_intact": round(float(np.median(intact_v)), 4),
                    })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["p_value"].values)
    return result


def genomewide_drug_screen(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    drug_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Screen all PRISM drugs for CDKN2A-selective sensitivity (pan-cancer)."""
    del_lines = classified[classified["CDKN2A_status"] == "deleted"].index
    intact_lines = classified[classified["CDKN2A_status"] == "intact"].index

    # Build broad_id -> name mapping
    bid_to_name = {}
    for _, row in drug_meta.iterrows():
        bid = row.get("broad_id")
        name = row.get("name")
        if pd.notna(bid) and pd.notna(name):
            bid_to_name[bid] = name

    rows = []
    pvals = []

    for treatment_id in prism.index:
        drug_sens = prism.loc[treatment_id]
        del_vals = drug_sens.reindex(del_lines).dropna().values
        intact_vals = drug_sens.reindex(intact_lines).dropna().values

        if len(del_vals) < MIN_SAMPLES or len(intact_vals) < MIN_SAMPLES:
            continue

        d = cohens_d(del_vals, intact_vals)
        _, pval = stats.mannwhitneyu(del_vals, intact_vals, alternative="two-sided")

        # Extract broad_id from treatment_id
        broad_id = str(treatment_id).split("::")[0] if "::" in str(treatment_id) else str(treatment_id)
        drug_name = bid_to_name.get(broad_id, "")

        rows.append({
            "treatment_id": treatment_id,
            "drug_name": drug_name,
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_del": len(del_vals),
            "n_intact": len(intact_vals),
            "median_sens_del": round(float(np.median(del_vals)), 4),
            "median_sens_intact": round(float(np.median(intact_vals)), 4),
        })
        pvals.append(pval)

    result = pd.DataFrame(rows)
    if len(result) > 0 and pvals:
        result["fdr"] = fdr_correction(np.array(pvals))
    return result


def compute_concordance(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    available_drugs: dict,
    crispr: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate CRISPR dependency with PRISM drug sensitivity per cell line."""
    rows = []

    for drug_name, info in available_drugs.items():
        gene = info["crispr_gene"]
        if not gene or gene not in crispr.columns:
            continue

        broad_id = info["broad_id"]
        matching_rows = [idx for idx in prism.index if broad_id in str(idx)]
        if not matching_rows:
            continue

        drug_sens = prism.loc[matching_rows[0]]
        gene_dep = crispr[gene]

        common = drug_sens.dropna().index.intersection(gene_dep.dropna().index)
        common = common.intersection(classified.index)

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


def plot_sensitivity(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of CDK4/6i sensitivity by CDKN2A status."""
    cdk46i = sensitivity[
        sensitivity["mechanism"].str.contains("CDK4/6") &
        (sensitivity["stratum"] == "rb1_intact")
    ].copy()
    if len(cdk46i) == 0:
        cdk46i = sensitivity[sensitivity["mechanism"].str.contains("CDK4/6")].copy()
    if len(cdk46i) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    cdk46i_sorted = cdk46i.sort_values("cohens_d")

    y_pos = np.arange(len(cdk46i_sorted))
    colors = ["#D95319" if row["fdr"] < 0.05 else "#999999" for _, row in cdk46i_sorted.iterrows()]

    ax.barh(y_pos, cdk46i_sorted["cohens_d"], color=colors, alpha=0.7, height=0.6)

    labels = [
        f"{row['cancer_type']} ({row['drug']}) n={row['n_del']}+{row['n_intact']}"
        for _, row in cdk46i_sorted.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (CDKN2A-del vs intact)")
    ax.set_title("CDK4/6 Inhibitor Sensitivity by CDKN2A Status\n(negative = more sensitive in CDKN2A-deleted)")

    fig.tight_layout()
    fig.savefig(output_dir / "cdk46i_sensitivity_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    avail_report: pd.DataFrame,
    sensitivity: pd.DataFrame,
    genomewide: pd.DataFrame,
    concordance: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write human-readable summary."""
    lines = [
        "=" * 60,
        "CDKN2A Pan-Cancer Dependency Atlas - Phase 4: PRISM Drug Sensitivity",
        "=" * 60,
        "",
        "DRUG AVAILABILITY IN PRISM 24Q2",
        "-" * 50,
    ]
    for _, row in avail_report.iterrows():
        status = "AVAILABLE" if row["in_prism"] else "NOT FOUND"
        lines.append(f"  {row['drug_name']} ({row['mechanism']}): {status}")

    if len(sensitivity) > 0:
        lines += ["", "TARGETED DRUG SENSITIVITY (CDKN2A-del vs intact)", "-" * 50]
        for _, row in sensitivity.sort_values("cohens_d").iterrows():
            sig = " *" if row.get("fdr", 1) < 0.05 else ""
            lines.append(
                f"  {row['cancer_type']} / {row['drug']} [{row['stratum']}]: "
                f"d={row['cohens_d']:.3f} FDR={row.get('fdr', float('nan')):.3e}{sig}"
            )
        lines.append("  (* = FDR < 0.05)")

    if len(genomewide) > 0:
        sig_gw = genomewide[genomewide["fdr"] < 0.05]
        gained = sig_gw[sig_gw["cohens_d"] < -0.3].sort_values("cohens_d")
        lines += [
            "", f"GENOME-WIDE DRUG SCREEN: {len(sig_gw)} significant (FDR<0.05)",
            f"  CDKN2A-selective sensitivity (d<-0.3, FDR<0.05): {len(gained)}",
            "-" * 50,
        ]
        for _, row in gained.head(20).iterrows():
            lines.append(
                f"  {row['drug_name'] or row['treatment_id']}: d={row['cohens_d']:.3f} "
                f"FDR={row['fdr']:.3e}"
            )

    if len(concordance) > 0:
        lines += ["", "CRISPR-PRISM CONCORDANCE", "-" * 50]
        for _, row in concordance.iterrows():
            lines.append(
                f"  {row['drug']} vs {row['gene']}: r={row['spearman_r']:.3f} "
                f"p={row['p_value']:.3e} n={row['n_lines']}"
            )
    lines.append("")

    with open(output_dir / "prism_drug_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4: PRISM Drug Sensitivity Validation ===\n")

    # Load Phase 1
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "cdkn2a_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()

    # Load PRISM metadata
    print("Loading PRISM 24Q2 metadata...")
    meta = pd.read_csv(DEPMAP_DIR / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")

    # Find available drugs
    available, avail_report = find_available_drugs(meta)
    avail_report.to_csv(OUTPUT_DIR / "drug_availability.csv", index=False)
    n_found = avail_report["in_prism"].sum()
    print(f"  {n_found}/{len(avail_report)} target drugs found in PRISM")
    for name in available:
        print(f"    {name} ({available[name]['mechanism']})")

    # Load PRISM matrix
    print("\nLoading PRISM data matrix...")
    prism = pd.read_csv(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )
    print(f"  {prism.shape[0]} treatments x {prism.shape[1]} cell lines")

    # Targeted drug sensitivity
    if available:
        print("\nComputing targeted drug sensitivity...")
        sensitivity = compute_targeted_sensitivity(prism, classified, available, qualifying_types)
        sensitivity.to_csv(OUTPUT_DIR / "cdk46i_sensitivity.csv", index=False)

        if len(sensitivity) > 0:
            print(f"  {len(sensitivity)} tests:")
            for _, row in sensitivity[sensitivity["stratum"] == "all"].sort_values("cohens_d").iterrows():
                sig = " *" if row.get("fdr", 1) < 0.05 else ""
                print(f"    {row['cancer_type']} / {row['drug']}: d={row['cohens_d']:.3f} "
                      f"FDR={row.get('fdr', float('nan')):.3e}{sig}")
    else:
        sensitivity = pd.DataFrame()
        print("\nNo target drugs found — skipping targeted analysis.")

    # Genome-wide drug screen
    print("\nRunning genome-wide drug screen (pan-cancer)...")
    genomewide = genomewide_drug_screen(prism, classified, meta)
    genomewide.to_csv(OUTPUT_DIR / "genomewide_drug_hits.csv", index=False)

    sig_gw = genomewide[genomewide.get("fdr", pd.Series(dtype=float)) < 0.05] if "fdr" in genomewide.columns else pd.DataFrame()
    gained = sig_gw[sig_gw["cohens_d"] < -0.3] if len(sig_gw) > 0 else pd.DataFrame()
    print(f"  {len(genomewide)} drugs screened, {len(sig_gw)} significant (FDR<0.05)")
    print(f"  CDKN2A-selective (d<-0.3, FDR<0.05): {len(gained)}")
    if len(gained) > 0:
        for _, row in gained.sort_values("cohens_d").head(10).iterrows():
            print(f"    {row['drug_name'] or row['treatment_id']}: d={row['cohens_d']:.3f} "
                  f"FDR={row['fdr']:.3e}")

    # Identify non-CDK4/6 drug hits
    non_cdk46 = gained.copy() if len(gained) > 0 else pd.DataFrame()
    cdk_drug_names = {"trilaciclib", "palbociclib", "ribociclib", "abemaciclib"}
    if len(non_cdk46) > 0:
        non_cdk46 = non_cdk46[~non_cdk46["drug_name"].isin(cdk_drug_names)]
        non_cdk46.to_csv(OUTPUT_DIR / "non_cdk46_drug_hits.csv", index=False)
        print(f"  Non-CDK4/6 selective drugs: {len(non_cdk46)}")

    # CRISPR-PRISM concordance
    if available:
        print("\nComputing CRISPR-PRISM concordance...")
        crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
        concordance = compute_concordance(prism, classified, available, crispr)
        concordance.to_csv(OUTPUT_DIR / "crispr_prism_concordance.csv", index=False)
        if len(concordance) > 0:
            for _, row in concordance.iterrows():
                print(f"    {row['drug']} vs {row['gene']}: r={row['spearman_r']:.3f} "
                      f"p={row['p_value']:.3e}")
    else:
        concordance = pd.DataFrame()

    # Sensitivity plot
    if len(sensitivity) > 0:
        plot_sensitivity(sensitivity, OUTPUT_DIR)
        print("  cdk46i_sensitivity_plot.png")

    # Summary
    write_summary(avail_report, sensitivity, genomewide, concordance, OUTPUT_DIR)
    print("  prism_drug_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
