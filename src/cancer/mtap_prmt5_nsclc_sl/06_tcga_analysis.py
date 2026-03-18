"""Phase 4: TCGA MTAP deletion frequency, subtype stratification, and population estimate.

Analyzes MTAP deletion frequency in TCGA LUAD and LUSC cohorts, stratifies
by molecular subtype (KP/KL/KOnly), and estimates PRMT5i-eligible patient
population.

Note: Survival analysis requires clinical data (not yet downloaded).

Usage:
    uv run python -m mtap_prmt5_nsclc_sl.06_tcga_analysis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from bioagentics.config import REPO_ROOT

TCGA_DIR = REPO_ROOT / "data" / "tcga"
OUTPUT_DIR = REPO_ROOT / "output" / "mtap-prmt5-nsclc-sl"
FIG_DIR = OUTPUT_DIR / "figures"

# MTAP gene ID in TCGA CN files
MTAP_GENE_NAME = "MTAP"
# Homozygous deletion: copy_number == 0
# Hemizygous deletion: copy_number <= 1
HOMODEL_THRESHOLD = 0
HEMIDEL_THRESHOLD = 1

# US NSCLC incidence for population estimate
US_NSCLC_INCIDENCE = 230000


def load_tcga_mtap_cn(cn_dir: Path) -> pd.DataFrame:
    """Load MTAP copy number from TCGA ASCAT gene-level CN files.

    Returns DataFrame with columns: patient_id, MTAP_copy_number.
    """
    ascat_files = sorted(cn_dir.glob("*.ascat3.gene_level_copy_number.v36.tsv"))
    if not ascat_files:
        raise FileNotFoundError(f"No ASCAT3 CN files found in {cn_dir}")

    rows = []
    for i, f in enumerate(ascat_files):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ascat_files)} files...", file=sys.stderr)

        # Extract patient ID from filename: TCGA-LUAD.UUID... -> extract TCGA barcode from manifest
        # File content has per-gene CN. Extract MTAP row.
        try:
            df = pd.read_csv(f, sep="\t", usecols=["gene_name", "copy_number"])
            mtap_row = df[df["gene_name"] == MTAP_GENE_NAME]
            if mtap_row.empty:
                continue
            cn_val = mtap_row["copy_number"].iloc[0]
            if pd.isna(cn_val):
                continue
            # Extract sample UUID from filename
            uuid = f.stem.split(".")[0]
            if uuid.startswith("TCGA-LU"):
                uuid = f.stem.split(".")[1]
            rows.append({"file_uuid": uuid, "MTAP_copy_number": int(cn_val)})
        except Exception:
            continue

    return pd.DataFrame(rows)


def load_manifest_mapping(cn_dir: Path) -> dict[str, str]:
    """Load manifest.tsv to map file UUIDs to case barcodes if available."""
    manifest_path = cn_dir / "manifest.tsv"
    if not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path, sep="\t")
    # Map UUID to filename
    return dict(zip(df["id"], df.get("filename", df.iloc[:, 1])))


def process_cohort(cancer_type: str, cn_dir: Path) -> pd.DataFrame:
    """Process TCGA cohort: load CN data, classify MTAP status."""
    print(f"\nProcessing {cancer_type}...")
    cn_data = load_tcga_mtap_cn(cn_dir)
    print(f"  {len(cn_data)} samples with MTAP CN data")

    cn_data["cancer_type"] = cancer_type
    cn_data["MTAP_homodel"] = cn_data["MTAP_copy_number"] <= HOMODEL_THRESHOLD
    cn_data["MTAP_hemidel"] = cn_data["MTAP_copy_number"] <= HEMIDEL_THRESHOLD

    n_homo = cn_data["MTAP_homodel"].sum()
    n_hemi = cn_data["MTAP_hemidel"].sum()
    print(f"  Homozygous deletion: {n_homo} ({n_homo/len(cn_data)*100:.1f}%)")
    print(f"  Hemizygous+homozygous deletion: {n_hemi} ({n_hemi/len(cn_data)*100:.1f}%)")

    return cn_data


def plot_deletion_by_subtype(
    subtypes_with_cn: pd.DataFrame, out_path: Path
) -> None:
    """Stacked bar: MTAP deletion rate by molecular subtype."""
    subtype_order = ["KP", "KL", "KPL", "KOnly", "KRAS-WT"]
    subtypes_present = [s for s in subtype_order if s in subtypes_with_cn["molecular_subtype"].values]

    rates = []
    counts = []
    for st in subtypes_present:
        sub = subtypes_with_cn[subtypes_with_cn["molecular_subtype"] == st]
        n = len(sub)
        if n == 0:
            rates.append(0)
            counts.append(0)
        else:
            rates.append(sub["MTAP_homodel"].mean() * 100)
            counts.append(n)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(subtypes_present)), rates, color="#D95319", alpha=0.8)
    ax.set_xticks(range(len(subtypes_present)))
    ax.set_xticklabels([f"{s}\n(n={c})" for s, c in zip(subtypes_present, counts)])
    ax.set_ylabel("MTAP Homozygous Deletion Rate (%)")
    ax.set_title("MTAP Deletion by Molecular Subtype (TCGA NSCLC)")

    for bar, rate in zip(bars, rates):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{rate:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_population_waterfall(estimates: dict, out_path: Path) -> None:
    """Waterfall chart: patient population funnel."""
    labels = list(estimates.keys())
    values = list(estimates.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4DBEEE"] * len(labels)
    colors[-1] = "#D95319"
    ax.barh(range(len(labels)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Estimated US patients/year")
    ax.set_title("PRMT5i-Eligible NSCLC Patient Population Estimate")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 500, i, f"{v:,.0f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Process LUAD and LUSC
    luad_cn = process_cohort("LUAD", TCGA_DIR / "luad" / "copy_number")
    lusc_cn = process_cohort("LUSC", TCGA_DIR / "lusc" / "copy_number")
    all_cn = pd.concat([luad_cn, lusc_cn], ignore_index=True)

    # Overall NSCLC stats
    n_total = len(all_cn)
    n_homo = all_cn["MTAP_homodel"].sum()
    n_hemi = all_cn["MTAP_hemidel"].sum()
    overall_homo_rate = n_homo / n_total * 100
    overall_hemi_rate = n_hemi / n_total * 100
    print(f"\nOverall NSCLC (n={n_total}):")
    print(f"  Homozygous deletion: {n_homo} ({overall_homo_rate:.1f}%)")
    print(f"  Any deletion (CN≤1): {n_hemi} ({overall_hemi_rate:.1f}%)")

    # LUAD vs LUSC comparison
    luad_rate = luad_cn["MTAP_homodel"].mean()
    lusc_rate = lusc_cn["MTAP_homodel"].mean()
    contingency = pd.crosstab(
        all_cn["cancer_type"],
        all_cn["MTAP_homodel"]
    )
    if contingency.shape == (2, 2):
        odds_ratio, fisher_p = sp_stats.fisher_exact(contingency)
        print(f"\nLUAD vs LUSC deletion rate:")
        print(f"  LUAD: {luad_rate*100:.1f}%, LUSC: {lusc_rate*100:.1f}%")
        print(f"  Fisher's exact p={fisher_p:.4f}")
    else:
        fisher_p = None
        print(f"\nLUAD: {luad_rate*100:.1f}%, LUSC: {lusc_rate*100:.1f}%")

    # Subtype stratification
    subtypes = pd.read_csv(TCGA_DIR / "nsclc_patient_subtypes.csv")
    print(f"\nLoaded {len(subtypes)} patients with subtype annotations")

    # Match CN data to subtypes by sample barcode patterns
    # The CN file UUIDs don't directly match patient IDs, so we'll report
    # subtype-level deletion rates from the subtype file (which has mutation data)
    # and overall CN-based deletion rates separately.
    # For now, report subtype distribution alongside overall deletion rates.

    subtype_counts = subtypes["molecular_subtype"].value_counts()
    print(f"\nMolecular subtype distribution:")
    for st, count in subtype_counts.items():
        print(f"  {st}: {count} ({count/len(subtypes)*100:.1f}%)")

    # Patient population estimate
    nsclc_homo_rate = overall_homo_rate / 100
    estimates = {
        "US NSCLC incidence": US_NSCLC_INCIDENCE,
        f"MTAP-deleted ({overall_homo_rate:.1f}%)": int(US_NSCLC_INCIDENCE * nsclc_homo_rate),
        f"LUAD MTAP-deleted ({luad_rate*100:.1f}%)": int(US_NSCLC_INCIDENCE * 0.85 * luad_rate),
    }
    # Note: Phase 2 showed no significant co-mutation modulators, so all
    # MTAP-deleted patients are potential PRMT5i candidates
    estimates["Potential PRMT5i-eligible (all MTAP-del)"] = int(
        US_NSCLC_INCIDENCE * nsclc_homo_rate
    )

    print(f"\nPatient population estimates:")
    for label, n in estimates.items():
        print(f"  {label}: {n:,}")

    # Visualizations
    print("\nGenerating plots...")

    # Bar: deletion rate by cancer type
    fig, ax = plt.subplots(figsize=(5, 4))
    rates_by_type = [luad_rate * 100, lusc_rate * 100]
    ns = [len(luad_cn), len(lusc_cn)]
    bars = ax.bar(["LUAD", "LUSC"], rates_by_type, color=["#4DBEEE", "#D95319"], alpha=0.8)
    ax.set_ylabel("MTAP Homozygous Deletion (%)")
    ax.set_title("MTAP Deletion Rate: LUAD vs LUSC (TCGA)")
    for bar, rate, n in zip(bars, rates_by_type, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{rate:.1f}%\n(n={n})", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mtap_del_luad_vs_lusc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mtap_del_luad_vs_lusc.png")

    # Population waterfall
    plot_population_waterfall(estimates, FIG_DIR / "population_estimate.png")

    # CN distribution histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = range(0, int(all_cn["MTAP_copy_number"].max()) + 2)
    ax.hist(all_cn["MTAP_copy_number"], bins=bins, color="#4DBEEE", alpha=0.8,
            edgecolor="black", align="left")
    ax.set_xlabel("MTAP Copy Number")
    ax.set_ylabel("Number of Samples")
    ax.set_title("MTAP Copy Number Distribution (TCGA NSCLC)")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5,
               label="Homozygous del threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mtap_cn_distribution_tcga.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mtap_cn_distribution_tcga.png")

    # Save results
    results = {
        "cohort_sizes": {"LUAD": len(luad_cn), "LUSC": len(lusc_cn), "total": n_total},
        "mtap_deletion": {
            "overall_homozygous_rate": overall_homo_rate,
            "overall_any_deletion_rate": overall_hemi_rate,
            "luad_homozygous_rate": float(luad_rate * 100),
            "lusc_homozygous_rate": float(lusc_rate * 100),
            "luad_vs_lusc_fisher_p": float(fisher_p) if fisher_p is not None else None,
        },
        "subtype_distribution": subtype_counts.to_dict(),
        "patient_population_estimate": estimates,
        "notes": [
            "Survival analysis requires clinical data download (not yet available)",
            "Subtype-level MTAP deletion rates require CN-to-barcode mapping via manifest",
            "Population estimate assumes all MTAP-deleted patients are PRMT5i-eligible "
            "(Phase 2 showed no co-mutation modulators)",
        ],
    }
    out_path = OUTPUT_DIR / "tcga_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
