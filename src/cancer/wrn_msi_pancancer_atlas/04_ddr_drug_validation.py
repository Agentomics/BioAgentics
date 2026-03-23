"""Phase 4: PRISM DDR drug sensitivity validation.

Tests whether MSI-H cell lines show differential sensitivity to DDR-targeting
drugs in PRISM 24Q2. Tests CRISPR-PRISM concordance for gene-drug pairs.

Note: WRN-specific inhibitors (VVD-214, HRO-761) are not in PRISM 24Q2.
This analysis focuses on available DDR compounds.

Usage:
    uv run python -m wrn_msi_pancancer_atlas.04_ddr_drug_validation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix
from cancer.wrn_msi_pancancer_atlas.stats_utils import cohens_d, fdr_correction

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "output" / "wrn-msi-pancancer-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "wrn-msi-pancancer-atlas" / "phase4"

# DDR drugs to search for in PRISM (name -> (mechanism, CRISPR gene))
DDR_DRUG_TARGETS = {
    # Available in PRISM 24Q2
    "AZD0156": ("ATM inhibitor", "ATM"),
    "AZD1390": ("ATM inhibitor (brain-penetrant)", "ATM"),
    "AZD-7648": ("DNA-PK inhibitor", "PRKDC"),
    "AZD4573": ("CDK9 inhibitor", "CDK9"),
    "AZD5582": ("IAP antagonist/SMAC mimetic", "BIRC2"),
    # Expected but likely absent — document gap
    "ceralasertib": ("ATR inhibitor (AZD6738)", "ATR"),
    "berzosertib": ("ATR inhibitor (VX-970)", "ATR"),
    "adavosertib": ("WEE1 inhibitor (AZD1775)", "WEE1"),
    "olaparib": ("PARP inhibitor", "PARP1"),
    "talazoparib": ("PARP inhibitor (trapping)", "PARP1"),
    "rucaparib": ("PARP inhibitor", "PARP1"),
    "niraparib": ("PARP inhibitor", "PARP1"),
    "prexasertib": ("CHK1 inhibitor (LY2606368)", "CHEK1"),
    "temozolomide": ("Alkylating agent (MMR-dependent)", None),
    "cisplatin": ("Platinum crosslinker", None),
}


def find_available_drugs(meta: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Search PRISM metadata for DDR drugs. Return available + availability report."""
    all_names = set(meta["name"].dropna().unique())
    available = {}
    report_rows = []

    for drug_name, (mechanism, gene) in DDR_DRUG_TARGETS.items():
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


def compute_drug_sensitivity(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    available_drugs: dict,
) -> pd.DataFrame:
    """Compare drug sensitivity (MSI-H vs MSS) pan-cancer."""
    rows = []
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")

    for drug_name, info in available_drugs.items():
        # Find the drug rows in PRISM matrix (may have multiple doses/profiles)
        broad_id = info["broad_id"]
        matching_rows = [idx for idx in prism.index if broad_id in str(idx)]

        if not matching_rows:
            continue

        # Use first matching row
        drug_sens = prism.loc[matching_rows[0]]

        # Pan-cancer comparison
        msi_lines = classified[classified["msi_status"] == "MSI-H"].index
        mss_lines = classified[classified["msi_status"] == "MSS"].index

        msi_vals = drug_sens.reindex(msi_lines).dropna().values
        mss_vals = drug_sens.reindex(mss_lines).dropna().values

        if len(msi_vals) < 3 or len(mss_vals) < 3:
            continue

        d = cohens_d(msi_vals, mss_vals)
        _, pval = stats.mannwhitneyu(msi_vals, mss_vals, alternative="two-sided")

        rows.append({
            "cancer_type": "Pan-cancer (pooled)",
            "drug": drug_name,
            "mechanism": info["mechanism"],
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_msi": len(msi_vals),
            "n_mss": len(mss_vals),
            "median_sens_msi": round(float(np.median(msi_vals)), 4),
            "median_sens_mss": round(float(np.median(mss_vals)), 4),
        })

        # Per qualifying cancer type
        for ct in qualifying["cancer_type"]:
            ct_lines = classified[classified["OncotreeLineage"] == ct]
            msi_ct = ct_lines[ct_lines["msi_status"] == "MSI-H"].index
            mss_ct = ct_lines[ct_lines["msi_status"] == "MSS"].index

            msi_v = drug_sens.reindex(msi_ct).dropna().values
            mss_v = drug_sens.reindex(mss_ct).dropna().values

            if len(msi_v) < 3 or len(mss_v) < 3:
                continue

            d_ct = cohens_d(msi_v, mss_v)
            _, pval_ct = stats.mannwhitneyu(msi_v, mss_v, alternative="two-sided")

            rows.append({
                "cancer_type": ct,
                "drug": drug_name,
                "mechanism": info["mechanism"],
                "cohens_d": round(d_ct, 4),
                "p_value": float(pval_ct),
                "n_msi": len(msi_v),
                "n_mss": len(mss_v),
                "median_sens_msi": round(float(np.median(msi_v)), 4),
                "median_sens_mss": round(float(np.median(mss_v)), 4),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["p_value"].values)
    return result


def compute_crispr_prism_concordance(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    available_drugs: dict,
    crispr: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate CRISPR gene dependency with PRISM drug sensitivity per cell line."""
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
            "context": "pan-cancer",
        })

    return pd.DataFrame(rows)


def write_summary(
    avail_report: pd.DataFrame,
    sensitivity: pd.DataFrame,
    concordance: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write human-readable summary."""
    lines = [
        "=" * 60,
        "WRN-MSI Pan-Cancer Atlas — Phase 4: DDR Drug Validation",
        "=" * 60,
        "",
        "DRUG AVAILABILITY IN PRISM 24Q2",
        "-" * 50,
    ]
    for _, row in avail_report.iterrows():
        status = "AVAILABLE" if row["in_prism"] else "NOT FOUND"
        lines.append(f"  {row['drug_name']} ({row['mechanism']}): {status}")

    lines += [
        "",
        "NOTE: WRN-specific inhibitors (VVD-214/VVD-130037, HRO-761)",
        "are NOT in PRISM 24Q2. This is a critical data gap.",
        "",
    ]

    if len(sensitivity) > 0:
        lines.append("DDR DRUG SENSITIVITY (MSI-H vs MSS)")
        lines.append("-" * 50)
        for _, row in sensitivity.sort_values("cohens_d").iterrows():
            sig = " *" if row.get("fdr", 1) < 0.05 else ""
            lines.append(
                f"  {row['cancer_type']} / {row['drug']}: d={row['cohens_d']:.3f} "
                f"FDR={row.get('fdr', float('nan')):.3e}{sig}"
            )
        lines.append("  (* = FDR < 0.05)")
    else:
        lines.append("No DDR drug sensitivity tests possible.")
    lines.append("")

    if len(concordance) > 0:
        lines.append("CRISPR-PRISM CONCORDANCE")
        lines.append("-" * 50)
        for _, row in concordance.iterrows():
            lines.append(
                f"  {row['drug']} vs {row['gene']} ({row['context']}): "
                f"r={row['spearman_r']:.3f} p={row['p_value']:.3e} n={row['n_lines']}"
            )
    lines.append("")

    with open(output_dir / "ddr_drug_validation_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4: PRISM DDR Drug Validation ===\n")

    # Load Phase 1
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "msi_classification.csv", index_col=0)

    # Load PRISM metadata
    print("Loading PRISM 24Q2 metadata...")
    meta = pd.read_csv(DEPMAP_DIR / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")

    # Find available drugs
    available, avail_report = find_available_drugs(meta)
    avail_report.to_csv(OUTPUT_DIR / "ddr_drug_availability.csv", index=False)

    n_found = avail_report["in_prism"].sum()
    n_total = len(avail_report)
    print(f"  {n_found}/{n_total} DDR drugs found in PRISM")
    for name in available:
        print(f"    {name} ({available[name]['mechanism']})")

    if not available:
        print("\nNo DDR drugs found. Writing summary and exiting.")
        pd.DataFrame().to_csv(OUTPUT_DIR / "ddr_drug_sensitivity.csv", index=False)
        pd.DataFrame().to_csv(OUTPUT_DIR / "crispr_prism_concordance.csv", index=False)
        write_summary(avail_report, pd.DataFrame(), pd.DataFrame(), OUTPUT_DIR)
        print("Done.")
        return

    # Load PRISM matrix
    print("\nLoading PRISM data matrix...")
    prism = pd.read_csv(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )
    print(f"  {prism.shape[0]} treatments x {prism.shape[1]} cell lines")

    # Compute drug sensitivity
    print("\nComputing DDR drug sensitivity (MSI-H vs MSS)...")
    sensitivity = compute_drug_sensitivity(prism, classified, available)
    sensitivity.to_csv(OUTPUT_DIR / "ddr_drug_sensitivity.csv", index=False)

    if len(sensitivity) > 0:
        print(f"  {len(sensitivity)} tests:")
        for _, row in sensitivity.sort_values("cohens_d").iterrows():
            sig = " *" if row.get("fdr", 1) < 0.05 else ""
            print(f"    {row['cancer_type']} / {row['drug']}: d={row['cohens_d']:.3f} "
                  f"FDR={row.get('fdr', float('nan')):.3e}{sig}")

    # CRISPR-PRISM concordance
    print("\nComputing CRISPR-PRISM concordance...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    concordance = compute_crispr_prism_concordance(prism, classified, available, crispr)
    concordance.to_csv(OUTPUT_DIR / "crispr_prism_concordance.csv", index=False)

    if len(concordance) > 0:
        for _, row in concordance.iterrows():
            print(f"    {row['drug']} vs {row['gene']}: r={row['spearman_r']:.3f} "
                  f"p={row['p_value']:.3e}")

    # Summary
    write_summary(avail_report, sensitivity, concordance, OUTPUT_DIR)
    print("\n  ddr_drug_validation_summary.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
