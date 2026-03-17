"""Phase 4: PRISM drug sensitivity validation for RB1-loss cancers.

Tests CDK4/6 inhibitors (negative control — should lose efficacy),
Aurora kinase, CHK1/WEE1, PARPi, and platinum drugs for RB1-loss
selectivity. Runs genome-wide PRISM screen. Tests CRISPR-PRISM concordance.

Usage:
    uv run python -m rb1_loss_pancancer_dependency_atlas.04_prism_drug_sensitivity
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
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "rb1-loss-pancancer-dependency-atlas" / "phase4"

# Target drugs: name -> (mechanism, CRISPR gene, expected direction for RB1-loss)
# direction: "negative_control" = should NOT be selective, "selective" = should be more sensitive
TARGET_DRUGS = {
    # CDK4/6 inhibitors (negative control: RB1-loss = no target)
    "trilaciclib": ("CDK4/6 inhibitor", "CDK4", "negative_control"),
    "palbociclib": ("CDK4/6 inhibitor", "CDK4", "negative_control"),
    "ribociclib": ("CDK4/6 inhibitor", "CDK4", "negative_control"),
    "abemaciclib": ("CDK4/6 inhibitor", "CDK4", "negative_control"),
    # Aurora kinase inhibitors
    "alisertib": ("Aurora A inhibitor", "AURKA", "selective"),
    "barasertib": ("Aurora B inhibitor", "AURKB", "selective"),
    # CHK1/WEE1 inhibitors
    "prexasertib": ("CHK1 inhibitor", "CHEK1", "selective"),
    "adavosertib": ("WEE1 inhibitor", "WEE1", "selective"),
    "azenosertib": ("WEE1 inhibitor", "WEE1", "selective"),
    # Platinum/SCLC SOC
    "cisplatin": ("Platinum agent", None, "selective"),
    "carboplatin": ("Platinum agent", None, "selective"),
    "etoposide": ("Topoisomerase II inhibitor", "TOP2A", "selective"),
    # PARPi (cross-ref with CK2 SL)
    "olaparib": ("PARP inhibitor", "PARP1", "selective"),
    "talazoparib": ("PARP inhibitor", "PARP1", "selective"),
    "rucaparib": ("PARP inhibitor", "PARP1", "selective"),
    "niraparib": ("PARP inhibitor", "PARP1", "selective"),
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
    all_names = set(meta["name"].dropna().str.lower().unique())
    name_map = {n.lower(): n for n in meta["name"].dropna().unique()}
    available = {}
    report_rows = []

    for drug_name, (mechanism, gene, expected) in TARGET_DRUGS.items():
        in_prism = drug_name.lower() in all_names
        if in_prism:
            actual_name = name_map[drug_name.lower()]
            drug_rows = meta[meta["name"] == actual_name]
            broad_id = drug_rows["broad_id"].iloc[0]
            available[drug_name] = {
                "mechanism": mechanism,
                "crispr_gene": gene,
                "expected_direction": expected,
                "broad_id": broad_id,
            }
        report_rows.append({
            "drug_name": drug_name,
            "mechanism": mechanism,
            "crispr_gene": gene or "",
            "expected_for_rb1_loss": expected,
            "in_prism": in_prism,
        })

    return available, pd.DataFrame(report_rows)


def compute_targeted_sensitivity(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    available_drugs: dict,
    qualifying_types: list[str],
) -> pd.DataFrame:
    """Compare targeted drug sensitivity in RB1-loss vs RB1-intact."""
    rows = []

    for drug_name, info in available_drugs.items():
        broad_id = info["broad_id"]
        matching_rows = [idx for idx in prism.index if broad_id in str(idx)]
        if not matching_rows:
            continue

        drug_sens = prism.loc[matching_rows[0]]

        # Pan-cancer
        lost_lines = classified[classified["RB1_status"] == "lost"].index
        intact_lines = classified[classified["RB1_status"] == "intact"].index

        lost_vals = drug_sens.reindex(lost_lines).dropna().values
        intact_vals = drug_sens.reindex(intact_lines).dropna().values

        if len(lost_vals) >= MIN_SAMPLES and len(intact_vals) >= MIN_SAMPLES:
            d = cohens_d(lost_vals, intact_vals)
            _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")
            rows.append({
                "cancer_type": "Pan-cancer (pooled)",
                "drug": drug_name,
                "mechanism": info["mechanism"],
                "expected_direction": info["expected_direction"],
                "cohens_d": round(d, 4),
                "p_value": float(pval),
                "n_lost": len(lost_vals),
                "n_intact": len(intact_vals),
                "median_sens_lost": round(float(np.median(lost_vals)), 4),
                "median_sens_intact": round(float(np.median(intact_vals)), 4),
            })

        # Per cancer type
        for ct in qualifying_types:
            ct_data = classified[classified["OncotreeLineage"] == ct]
            lost_ct = ct_data[ct_data["RB1_status"] == "lost"].index
            intact_ct = ct_data[ct_data["RB1_status"] == "intact"].index

            lost_v = drug_sens.reindex(lost_ct).dropna().values
            intact_v = drug_sens.reindex(intact_ct).dropna().values

            if len(lost_v) >= MIN_SAMPLES and len(intact_v) >= MIN_SAMPLES:
                d_ct = cohens_d(lost_v, intact_v)
                _, pval_ct = stats.mannwhitneyu(lost_v, intact_v, alternative="two-sided")
                rows.append({
                    "cancer_type": ct,
                    "drug": drug_name,
                    "mechanism": info["mechanism"],
                    "expected_direction": info["expected_direction"],
                    "cohens_d": round(d_ct, 4),
                    "p_value": float(pval_ct),
                    "n_lost": len(lost_v),
                    "n_intact": len(intact_v),
                    "median_sens_lost": round(float(np.median(lost_v)), 4),
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
    """Screen all PRISM drugs for RB1-loss-selective sensitivity (pan-cancer)."""
    lost_lines = classified[classified["RB1_status"] == "lost"].index
    intact_lines = classified[classified["RB1_status"] == "intact"].index

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
        lost_vals = drug_sens.reindex(lost_lines).dropna().values
        intact_vals = drug_sens.reindex(intact_lines).dropna().values

        if len(lost_vals) < MIN_SAMPLES or len(intact_vals) < MIN_SAMPLES:
            continue

        d = cohens_d(lost_vals, intact_vals)
        _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")

        broad_id = str(treatment_id).split("::")[0] if "::" in str(treatment_id) else str(treatment_id)
        drug_name = bid_to_name.get(broad_id, "")

        rows.append({
            "treatment_id": treatment_id,
            "drug_name": drug_name,
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_lost": len(lost_vals),
            "n_intact": len(intact_vals),
            "median_sens_lost": round(float(np.median(lost_vals)), 4),
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


def plot_genomewide_volcano(genomewide: pd.DataFrame, output_dir: Path) -> None:
    """Volcano plot for genome-wide drug screen."""
    if "fdr" not in genomewide.columns or len(genomewide) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x = genomewide["cohens_d"].values
    y = -np.log10(genomewide["fdr"].values.clip(min=1e-50))

    sig = (genomewide["fdr"] < 0.05) & (genomewide["cohens_d"].abs() > 0.3)
    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)

    gained = sig & (genomewide["cohens_d"] < 0)
    lost_dep = sig & (genomewide["cohens_d"] > 0)
    ax.scatter(x[gained], y[gained], c="#D95319", s=15, alpha=0.8, label="RB1-loss selective")
    ax.scatter(x[lost_dep], y[lost_dep], c="#4DBEEE", s=15, alpha=0.8, label="RB1-loss resistant")

    top_gained = genomewide[gained].nsmallest(10, "cohens_d")
    for _, row in top_gained.iterrows():
        name = row["drug_name"] if row["drug_name"] else str(row["treatment_id"])[:20]
        ax.annotate(name, (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                     fontsize=7, ha="right")

    ax.axhline(-np.log10(0.05), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-0.3, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(0.3, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cohen's d (RB1-loss vs intact)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("Genome-wide PRISM Drug Screen: RB1-Loss Selectivity")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "genomewide_drug_volcano.png", dpi=150)
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
        "=" * 70,
        "RB1-Loss Pan-Cancer Dependency Atlas - Phase 4: PRISM Drug Sensitivity",
        "=" * 70,
        "",
        "DRUG AVAILABILITY IN PRISM 24Q2",
        "-" * 60,
    ]
    for _, row in avail_report.iterrows():
        status = "AVAILABLE" if row["in_prism"] else "NOT FOUND"
        lines.append(f"  {row['drug_name']} ({row['mechanism']}): {status}")

    if len(sensitivity) > 0:
        lines += ["", "TARGETED DRUG SENSITIVITY (RB1-loss vs intact)", "-" * 60]

        # Negative controls first
        neg_ctrl = sensitivity[sensitivity["expected_direction"] == "negative_control"]
        if len(neg_ctrl) > 0:
            lines.append("  CDK4/6 INHIBITORS (negative control: should LOSE efficacy in RB1-loss)")
            for _, row in neg_ctrl.sort_values("cohens_d").iterrows():
                sig = " *" if row.get("fdr", 1) < 0.05 else ""
                direction = "VALIDATES" if row["cohens_d"] > 0 else "unexpected"
                lines.append(
                    f"    {row['cancer_type']} / {row['drug']}: d={row['cohens_d']:.3f} "
                    f"FDR={row.get('fdr', float('nan')):.3e} ({direction}){sig}"
                )

        # Selective drugs
        sel = sensitivity[sensitivity["expected_direction"] == "selective"]
        if len(sel) > 0:
            lines.append("  EXPECTED SELECTIVE DRUGS")
            for _, row in sel.sort_values("cohens_d").iterrows():
                sig = " *" if row.get("fdr", 1) < 0.05 else ""
                lines.append(
                    f"    {row['cancer_type']} / {row['drug']}: d={row['cohens_d']:.3f} "
                    f"FDR={row.get('fdr', float('nan')):.3e}{sig}"
                )

    if len(genomewide) > 0 and "fdr" in genomewide.columns:
        sig_gw = genomewide[genomewide["fdr"] < 0.05]
        gained = sig_gw[sig_gw["cohens_d"] < -0.3].sort_values("cohens_d")
        resistant = sig_gw[sig_gw["cohens_d"] > 0.3].sort_values("cohens_d", ascending=False)
        lines += [
            "",
            f"GENOME-WIDE DRUG SCREEN: {len(sig_gw)} significant (FDR<0.05)",
            f"  RB1-loss selective (d<-0.3): {len(gained)}",
            f"  RB1-loss resistant (d>0.3): {len(resistant)}",
            "-" * 60,
        ]
        if len(gained) > 0:
            lines.append("  Top RB1-loss selective drugs:")
            for _, row in gained.head(15).iterrows():
                lines.append(
                    f"    {row['drug_name'] or row['treatment_id']}: d={row['cohens_d']:.3f} "
                    f"FDR={row['fdr']:.3e}"
                )
        if len(resistant) > 0:
            lines.append("  Top RB1-loss resistant drugs (CDK4/6i expected here):")
            for _, row in resistant.head(10).iterrows():
                lines.append(
                    f"    {row['drug_name'] or row['treatment_id']}: d={row['cohens_d']:.3f} "
                    f"FDR={row['fdr']:.3e}"
                )

    if len(concordance) > 0:
        lines += ["", "CRISPR-PRISM CONCORDANCE", "-" * 60]
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
    classified = pd.read_csv(PHASE1_DIR / "rb1_classification.csv", index_col=0)
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
        sensitivity.to_csv(OUTPUT_DIR / "targeted_drug_sensitivity.csv", index=False)

        if len(sensitivity) > 0:
            print(f"  {len(sensitivity)} tests:")
            for _, row in sensitivity[sensitivity["cancer_type"] == "Pan-cancer (pooled)"].sort_values("cohens_d").iterrows():
                sig = " *" if row.get("fdr", 1) < 0.05 else ""
                print(f"    {row['drug']} ({row['mechanism']}): d={row['cohens_d']:.3f} "
                      f"FDR={row.get('fdr', float('nan')):.3e}{sig}")
    else:
        sensitivity = pd.DataFrame()
        print("\nNo target drugs found — skipping targeted analysis.")

    # Genome-wide drug screen
    print("\nRunning genome-wide drug screen (pan-cancer)...")
    genomewide = genomewide_drug_screen(prism, classified, meta)
    genomewide.to_csv(OUTPUT_DIR / "genomewide_drug_screen.csv", index=False)

    if "fdr" in genomewide.columns:
        sig_gw = genomewide[genomewide["fdr"] < 0.05]
        gained = sig_gw[sig_gw["cohens_d"] < -0.3].sort_values("cohens_d")
        resistant = sig_gw[sig_gw["cohens_d"] > 0.3].sort_values("cohens_d", ascending=False)
        print(f"  {len(genomewide)} drugs screened, {len(sig_gw)} significant (FDR<0.05)")
        print(f"  RB1-loss selective (d<-0.3): {len(gained)}")
        print(f"  RB1-loss resistant (d>0.3): {len(resistant)}")

        if len(gained) > 0:
            print("  Top RB1-loss selective:")
            for _, row in gained.head(10).iterrows():
                print(f"    {row['drug_name'] or row['treatment_id']}: d={row['cohens_d']:.3f} "
                      f"FDR={row['fdr']:.3e}")
        if len(resistant) > 0:
            print("  Top RB1-loss resistant:")
            for _, row in resistant.head(5).iterrows():
                print(f"    {row['drug_name'] or row['treatment_id']}: d={row['cohens_d']:.3f} "
                      f"FDR={row['fdr']:.3e}")

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

    # Plots
    plot_genomewide_volcano(genomewide, OUTPUT_DIR)

    # Summary
    write_summary(avail_report, sensitivity, genomewide, concordance, OUTPUT_DIR)
    print("  prism_drug_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
