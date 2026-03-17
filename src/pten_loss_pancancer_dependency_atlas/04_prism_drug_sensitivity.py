"""Phase 4: PRISM drug sensitivity validation for PTEN-lost cell lines.

Tests PI3K pathway inhibitors (AKT, PI3K, PI3Kβ, mTOR) for PTEN-selective
sensitivity. Runs genome-wide drug screen. Tests CRISPR-PRISM concordance
for AKT/mTOR axis gene-drug pairs.

Usage:
    uv run python -m pten_loss_pancancer_dependency_atlas.04_prism_drug_sensitivity
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
PHASE1_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase4"

# Target drugs: name -> (mechanism, CRISPR gene for concordance)
TARGET_DRUGS = {
    # AKT inhibitors
    "capivasertib": ("AKT inhibitor", "AKT1"),
    "ipatasertib": ("AKT inhibitor", "AKT1"),
    "MK-2206": ("AKT inhibitor (allosteric)", "AKT1"),
    "AZD5363": ("AKT inhibitor", "AKT1"),
    # PI3K inhibitors (pan-class I)
    "alpelisib": ("PI3Kα inhibitor", "PIK3CA"),
    "pictilisib": ("PI3K inhibitor (pan)", "PIK3CA"),
    "buparlisib": ("PI3K inhibitor (pan)", "PIK3CA"),
    "copanlisib": ("PI3K inhibitor (pan)", "PIK3CA"),
    "inavolisib": ("PI3Kα inhibitor (mutant-selective)", "PIK3CA"),
    # PI3Kβ-selective (key for PTEN-null biology)
    "AZD8186": ("PI3Kβ inhibitor", "PIK3CB"),
    "GSK2636771": ("PI3Kβ inhibitor", "PIK3CB"),
    # mTOR inhibitors
    "everolimus": ("mTOR inhibitor (rapalog)", "MTOR"),
    "rapamycin": ("mTOR inhibitor (rapalog)", "MTOR"),
    "sirolimus": ("mTOR inhibitor (rapalog)", "MTOR"),
    "temsirolimus": ("mTOR inhibitor (rapalog)", "MTOR"),
    "AZD8055": ("mTOR inhibitor (catalytic)", "MTOR"),
    "INK-128": ("mTOR inhibitor (catalytic)", "MTOR"),
    # Dual PI3K/mTOR
    "dactolisib": ("PI3K/mTOR dual inhibitor", "MTOR"),
    "apitolisib": ("PI3K/mTOR dual inhibitor", "MTOR"),
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
    """Compare targeted drug sensitivity in PTEN-lost vs intact.

    For PI3Kα inhibitors, also stratifies by PIK3CA hotspot co-mutation
    (PIK3CA-mutant lines may respond to PI3Kα inhibitors regardless of PTEN).
    """
    rows = []

    for drug_name, info in available_drugs.items():
        broad_id = info["broad_id"]
        matching_rows = [idx for idx in prism.index if broad_id in str(idx)]
        if not matching_rows:
            continue

        drug_sens = prism.loc[matching_rows[0]]

        is_pi3ka = "PI3Kα" in info["mechanism"]

        strata = [("all", classified)]
        if is_pi3ka:
            # Exclude PIK3CA hotspot lines (they respond via PIK3CA, not PTEN)
            no_pik3ca = classified[classified["PIK3CA_hotspot"] == False]  # noqa: E712
            strata.append(("excl_PIK3CA_hotspot", no_pik3ca))

        for stratum_name, stratum_data in strata:
            # Pan-cancer
            lost_lines = stratum_data[stratum_data["PTEN_status"] == "lost"].index
            intact_lines = stratum_data[stratum_data["PTEN_status"] == "intact"].index

            lost_vals = drug_sens.reindex(lost_lines).dropna().values
            intact_vals = drug_sens.reindex(intact_lines).dropna().values

            if len(lost_vals) >= MIN_SAMPLES and len(intact_vals) >= MIN_SAMPLES:
                d = cohens_d(lost_vals, intact_vals)
                _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")
                rows.append({
                    "cancer_type": "Pan-cancer (pooled)",
                    "drug": drug_name,
                    "mechanism": info["mechanism"],
                    "stratum": stratum_name,
                    "cohens_d": round(d, 4),
                    "p_value": float(pval),
                    "n_lost": len(lost_vals),
                    "n_intact": len(intact_vals),
                    "median_sens_lost": round(float(np.median(lost_vals)), 4),
                    "median_sens_intact": round(float(np.median(intact_vals)), 4),
                })

            # Per cancer type
            for ct in qualifying_types:
                ct_data = stratum_data[stratum_data["OncotreeLineage"] == ct]
                lost_ct = ct_data[ct_data["PTEN_status"] == "lost"].index
                intact_ct = ct_data[ct_data["PTEN_status"] == "intact"].index

                lost_v = drug_sens.reindex(lost_ct).dropna().values
                intact_v = drug_sens.reindex(intact_ct).dropna().values

                if len(lost_v) >= MIN_SAMPLES and len(intact_v) >= MIN_SAMPLES:
                    d_ct = cohens_d(lost_v, intact_v)
                    _, pval_ct = stats.mannwhitneyu(lost_v, intact_v, alternative="two-sided")
                    rows.append({
                        "cancer_type": ct,
                        "drug": drug_name,
                        "mechanism": info["mechanism"],
                        "stratum": stratum_name,
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
    """Screen all PRISM drugs for PTEN-selective sensitivity (pan-cancer)."""
    lost_lines = classified[classified["PTEN_status"] == "lost"].index
    intact_lines = classified[classified["PTEN_status"] == "intact"].index

    # Build broad_id -> name mapping
    bid_to_name = {}
    bid_to_moa = {}
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
        # Strip BRD: prefix for lookup
        lookup_bid = broad_id.replace("BRD:", "") if broad_id.startswith("BRD:") else broad_id
        drug_name = bid_to_name.get(lookup_bid, bid_to_name.get(broad_id, ""))

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


def identify_combination_candidates(
    genomewide: pd.DataFrame,
) -> pd.DataFrame:
    """Identify non-PI3K-pathway drugs with PTEN-selective sensitivity as
    potential combination partners."""
    if len(genomewide) == 0 or "fdr" not in genomewide.columns:
        return pd.DataFrame()

    sig = genomewide[genomewide["fdr"] < 0.05].copy()
    # Negative d = more sensitive in PTEN-lost (lower viability)
    pten_selective = sig[sig["cohens_d"] < -0.3].sort_values("cohens_d")

    # Filter out known PI3K/AKT/mTOR pathway drugs
    pi3k_terms = [
        "pi3k", "akt", "mtor", "rapamycin", "everolimus", "sirolimus",
        "alpelisib", "capivasertib", "ipatasertib", "pictilisib",
        "buparlisib", "copanlisib", "dactolisib", "apitolisib",
        "temsirolimus", "inavolisib",
    ]

    def is_pi3k_drug(name: str) -> bool:
        name_lower = str(name).lower()
        return any(term in name_lower for term in pi3k_terms)

    if len(pten_selective) > 0:
        non_pi3k = pten_selective[~pten_selective["drug_name"].apply(is_pi3k_drug)]
        return non_pi3k
    return pd.DataFrame()


def plot_targeted_sensitivity(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of PI3K pathway drug sensitivity by PTEN status."""
    plot_data = sensitivity[sensitivity["stratum"] == "all"].copy()
    if len(plot_data) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.35)))
    plot_sorted = plot_data.sort_values("cohens_d")

    y_pos = np.arange(len(plot_sorted))
    colors = ["#D95319" if row["fdr"] < 0.05 else "#999999" for _, row in plot_sorted.iterrows()]

    ax.barh(y_pos, plot_sorted["cohens_d"], color=colors, alpha=0.7, height=0.6)

    labels = [
        f"{row['cancer_type']} / {row['drug']} (n={row['n_lost']}+{row['n_intact']})"
        for _, row in plot_sorted.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (PTEN-lost vs intact)")
    ax.set_title(
        "PI3K/AKT/mTOR Drug Sensitivity by PTEN Status\n"
        "(negative = more sensitive in PTEN-lost, orange = FDR < 0.05)"
    )

    fig.tight_layout()
    fig.savefig(output_dir / "pi3k_drug_sensitivity_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    avail_report: pd.DataFrame,
    sensitivity: pd.DataFrame,
    genomewide: pd.DataFrame,
    concordance: pd.DataFrame,
    combo_candidates: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write human-readable summary."""
    lines = [
        "=" * 70,
        "PTEN Loss Pan-Cancer Dependency Atlas - Phase 4: PRISM Drug Sensitivity",
        "=" * 70,
        "",
        "DRUG AVAILABILITY IN PRISM 24Q2",
        "-" * 50,
    ]
    for _, row in avail_report.iterrows():
        status = "AVAILABLE" if row["in_prism"] else "NOT FOUND"
        lines.append(f"  {row['drug_name']} ({row['mechanism']}): {status}")

    if len(sensitivity) > 0:
        lines += ["", "TARGETED DRUG SENSITIVITY (PTEN-lost vs intact)", "-" * 50]

        # Group by mechanism for clarity
        for mech in sensitivity["mechanism"].unique():
            mech_data = sensitivity[
                (sensitivity["mechanism"] == mech) & (sensitivity["stratum"] == "all")
            ].sort_values("cohens_d")
            if len(mech_data) == 0:
                continue
            lines.append(f"\n  {mech}:")
            for _, row in mech_data.iterrows():
                sig = " *" if row.get("fdr", 1) < 0.05 else ""
                lines.append(
                    f"    {row['cancer_type']} / {row['drug']}: "
                    f"d={row['cohens_d']:.3f} FDR={row.get('fdr', float('nan')):.3e}{sig}"
                )
        lines.append("  (* = FDR < 0.05)")

    if len(genomewide) > 0:
        sig_gw = genomewide[genomewide["fdr"] < 0.05] if "fdr" in genomewide.columns else pd.DataFrame()
        gained = sig_gw[sig_gw["cohens_d"] < -0.3].sort_values("cohens_d") if len(sig_gw) > 0 else pd.DataFrame()
        lines += [
            "",
            f"GENOME-WIDE DRUG SCREEN: {len(sig_gw)} significant (FDR<0.05)",
            f"  PTEN-selective sensitivity (d<-0.3, FDR<0.05): {len(gained)}",
            "-" * 50,
        ]
        for _, row in gained.head(20).iterrows():
            lines.append(
                f"  {row['drug_name'] or row['treatment_id']}: d={row['cohens_d']:.3f} "
                f"FDR={row['fdr']:.3e}"
            )

    if len(combo_candidates) > 0:
        lines += [
            "",
            f"COMBINATION CANDIDATES (non-PI3K drugs, PTEN-selective): {len(combo_candidates)}",
            "-" * 50,
        ]
        for _, row in combo_candidates.head(15).iterrows():
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

    print("=== Phase 4: PRISM Drug Sensitivity Validation (PTEN) ===\n")

    # Load Phase 1
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "pten_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying[qualifying["qualifies"] == True]["cancer_type"].tolist()  # noqa: E712

    n_lost = (classified["PTEN_status"] == "lost").sum()
    n_intact = (classified["PTEN_status"] == "intact").sum()
    print(f"  {n_lost} PTEN-lost, {n_intact} PTEN-intact lines")
    print(f"  {len(qualifying_types)} qualifying cancer types: {', '.join(qualifying_types)}")

    # Load PRISM metadata
    print("\nLoading PRISM 24Q2 metadata...")
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
            print(f"  {len(sensitivity)} tests performed")
            pancancer = sensitivity[
                (sensitivity["cancer_type"] == "Pan-cancer (pooled)") &
                (sensitivity["stratum"] == "all")
            ].sort_values("cohens_d")
            for _, row in pancancer.iterrows():
                sig = " *" if row.get("fdr", 1) < 0.05 else ""
                print(f"    {row['drug']} ({row['mechanism']}): d={row['cohens_d']:.3f} "
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
    print(f"  PTEN-selective (d<-0.3, FDR<0.05): {len(gained)}")
    if len(gained) > 0:
        for _, row in gained.sort_values("cohens_d").head(10).iterrows():
            print(f"    {row['drug_name'] or row['treatment_id']}: d={row['cohens_d']:.3f} "
                  f"FDR={row['fdr']:.3e}")

    # Combination candidates (non-PI3K drugs with PTEN selectivity)
    combo = identify_combination_candidates(genomewide)
    if len(combo) > 0:
        combo.to_csv(OUTPUT_DIR / "combination_candidates.csv", index=False)
        print(f"\n  Non-PI3K combination candidates: {len(combo)}")
    else:
        combo = pd.DataFrame()

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

    # Plot
    if len(sensitivity) > 0:
        plot_targeted_sensitivity(sensitivity, OUTPUT_DIR)
        print("\n  pi3k_drug_sensitivity_plot.png")

    # Summary
    write_summary(avail_report, sensitivity, genomewide, concordance, combo, OUTPUT_DIR)
    print("  prism_drug_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
