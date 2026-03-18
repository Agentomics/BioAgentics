"""Phase 4: TCGA allele frequency integration and addressable populations.

Cross-validates DepMap cell line allele distributions against TCGA patient
populations. Estimates annual US patients per TP53 allele per cancer type.
Analyzes co-mutation landscape using DepMap mutation data.

Usage:
    uv run python -m tp53_hotspot_allele_dependencies.05_tcga_integration
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_mutations
from bioagentics.data.tp53_common import HOTSPOT_ALLELES

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "tp53_hotspot_allele_dependencies"
FIG_DIR = OUTPUT_DIR / "figures"
TCGA_PATH = REPO_ROOT / "data" / "tcga" / "pancancer_tp53" / "tp53_mutation_frequencies.csv"

# TCGA allele columns in the frequency file
TCGA_ALLELE_COLS = ["R175H", "R248W", "R273H", "G245S", "R249S", "Y220C", "R282W"]

# Approximate annual US cancer incidence by TCGA cancer type (SEER 2023)
US_ANNUAL_INCIDENCE = {
    "UCS": 4_000,
    "ESCA": 22_370,
    "LUSC": 70_000,
    "OV": 19_710,
    "HNSC": 66_920,
    "PAAD": 64_050,
    "COADREAD": 152_810,
    "LUAD": 117_000,
    "BLCA": 83_190,
    "STAD": 26_500,
    "LGG": 5_000,
    "UCEC": 67_880,
    "SARC": 16_890,
    "BRCA": 310_720,
    "KICH": 5_100,
    "GBM": 13_410,
    "LIHC": 41_210,
    "ACC": 600,
    "SKCM": 97_610,
    "MESO": 3_000,
    "DLBC": 18_000,
    "PRAD": 288_300,
    "CHOL": 8_000,
    "LAML": 20_380,
    "CESC": 13_820,
}

# Map TCGA cancer types to DepMap OncotreePrimaryDisease
TCGA_TO_DEPMAP = {
    "OV": "Ovarian Epithelial Tumor",
    "BRCA": "Invasive Breast Carcinoma",
    "COADREAD": "Colorectal Adenocarcinoma",
    "LUAD": "Non-Small Cell Lung Cancer",
    "LUSC": "Non-Small Cell Lung Cancer",
    "HNSC": "Head and Neck Squamous Cell Carcinoma",
    "BLCA": "Bladder Urothelial Carcinoma",
    "STAD": "Esophagogastric Adenocarcinoma",
    "ESCA": "Esophagogastric Adenocarcinoma",
    "GBM": "Diffuse Glioma",
    "LGG": "Diffuse Glioma",
    "UCEC": "Endometrial Carcinoma",
    "PAAD": "Pancreatic Adenocarcinoma",
    "LIHC": "Hepatocellular Carcinoma",
    "SKCM": "Melanoma",
    "LAML": "Acute Myeloid Leukemia",
}

# Key co-mutated genes to check
COMUTATION_GENES = ["KRAS", "PIK3CA", "PTEN", "APC", "RB1", "CDKN2A", "MYC", "BRCA1", "BRCA2"]


def load_tcga(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_depmap_allele_dist(output_dir: Path) -> pd.DataFrame:
    """DepMap allele distributions per cancer type from Phase 1 classified lines."""
    classified = pd.read_csv(output_dir / "tp53_classified_lines.csv", index_col=0)
    classified["TP53_mutated"] = classified["TP53_mutated"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    mutant = classified[classified["TP53_mutated"]]

    rows = []
    for ct, group in mutant.groupby("OncotreePrimaryDisease"):
        ac = group["TP53_allele"].value_counts()
        n = len(group)
        row = {"depmap_cancer_type": ct, "depmap_n_mutant": n}
        for allele in TCGA_ALLELE_COLS:
            row[f"depmap_{allele}"] = int(ac.get(allele, 0))
            row[f"depmap_{allele}_frac"] = row[f"depmap_{allele}"] / n if n > 0 else 0
        rows.append(row)

    return pd.DataFrame(rows)


def cross_validate(tcga: pd.DataFrame, depmap: pd.DataFrame) -> pd.DataFrame:
    """Compare allele distributions between TCGA and DepMap."""
    rows = []
    for _, tr in tcga.iterrows():
        ct = tr["cancer_type"]
        depmap_ct = TCGA_TO_DEPMAP.get(ct)
        if depmap_ct is None:
            continue

        dm_match = depmap[depmap["depmap_cancer_type"] == depmap_ct]
        if dm_match.empty:
            continue

        dm = dm_match.iloc[0]
        n_tcga = tr["TP53_mutated"]

        row = {
            "tcga_type": ct,
            "depmap_type": depmap_ct,
            "tcga_n_mutant": int(n_tcga),
            "depmap_n_mutant": int(dm["depmap_n_mutant"]),
            "tcga_structural_frac": round(tr["structural_total"] / n_tcga, 3) if n_tcga > 0 else 0,
            "tcga_contact_frac": round(tr["contact_total"] / n_tcga, 3) if n_tcga > 0 else 0,
        }
        for allele in TCGA_ALLELE_COLS:
            tcga_frac = tr[allele] / n_tcga if n_tcga > 0 else 0
            depmap_frac = dm[f"depmap_{allele}_frac"]
            row[f"tcga_{allele}_frac"] = round(tcga_frac, 3)
            row[f"depmap_{allele}_frac"] = round(depmap_frac, 3)
            row[f"diff_{allele}"] = round(depmap_frac - tcga_frac, 3)
        rows.append(row)

    return pd.DataFrame(rows)


def estimate_populations(tcga: pd.DataFrame) -> pd.DataFrame:
    """Estimate annual US patients per TP53 allele per cancer type."""
    rows = []
    for _, tr in tcga.iterrows():
        ct = tr["cancer_type"]
        incidence = US_ANNUAL_INCIDENCE.get(ct, 0)
        if incidence == 0:
            continue

        tp53_rate = tr["TP53_pct"] / 100.0
        n_mutated = tr["TP53_mutated"]

        for allele in TCGA_ALLELE_COLS:
            allele_count = tr[allele]
            allele_frac = allele_count / n_mutated if n_mutated > 0 else 0
            est = int(incidence * tp53_rate * allele_frac)
            rows.append({
                "cancer_type": ct,
                "allele": allele,
                "us_annual_incidence": incidence,
                "tp53_mutation_rate": round(tp53_rate, 3),
                "allele_fraction_of_mutant": round(allele_frac, 3),
                "estimated_annual_patients": est,
            })

    df = pd.DataFrame(rows)
    return df.sort_values("estimated_annual_patients", ascending=False)


def comutation_analysis(classified_path: Path, mutations_path: Path) -> pd.DataFrame:
    """Analyze co-mutation landscape per TP53 allele using DepMap data."""
    classified = pd.read_csv(classified_path, index_col=0)
    classified["TP53_mutated"] = classified["TP53_mutated"].map(
        {"True": True, "False": False, True: True, False: False}
    )

    muts = load_depmap_mutations(mutations_path)
    # Filter to HIGH/MODERATE impact in co-mutation genes
    comut = muts[
        (muts["HugoSymbol"].isin(COMUTATION_GENES))
        & (muts["VepImpact"].isin(["HIGH", "MODERATE"]))
    ]
    # Binary: which models have mutations in each gene
    gene_mutated = comut.groupby(["ModelID", "HugoSymbol"]).size().unstack(fill_value=0) > 0

    allele_groups = HOTSPOT_ALLELES + ["other_missense", "truncating", "TP53_WT"]
    rows = []
    for allele in allele_groups:
        model_ids = classified[classified["TP53_allele"] == allele].index
        n = len(model_ids)
        if n < 5:
            continue
        row = {"allele": allele, "n_lines": n}
        for gene in COMUTATION_GENES:
            if gene in gene_mutated.columns:
                n_comut = gene_mutated[gene].reindex(model_ids).fillna(False).sum()
                row[f"{gene}_n"] = int(n_comut)
                row[f"{gene}_pct"] = round(100 * n_comut / n, 1)
            else:
                row[f"{gene}_n"] = 0
                row[f"{gene}_pct"] = 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def plot_allele_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    """TCGA vs DepMap allele fractions side-by-side."""
    alleles = ["R175H", "R248W", "R273H"]
    cts = comparison["tcga_type"].tolist()
    if not cts:
        return

    fig, axes = plt.subplots(1, len(alleles), figsize=(5 * len(alleles), max(4, len(cts) * 0.45)))
    if len(alleles) == 1:
        axes = [axes]

    for ax, allele in zip(axes, alleles):
        y = np.arange(len(cts))
        w = 0.35
        tcga_v = comparison[f"tcga_{allele}_frac"].values
        depmap_v = comparison[f"depmap_{allele}_frac"].values
        ax.barh(y - w / 2, tcga_v, w, label="TCGA", color="#0072BD", alpha=0.7)
        ax.barh(y + w / 2, depmap_v, w, label="DepMap", color="#D95319", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(cts, fontsize=8)
        ax.set_xlabel("Fraction of TP53-mutant")
        ax.set_title(allele)
        ax.legend(fontsize=8)

    fig.suptitle("TP53 Allele Distribution: TCGA vs DepMap", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_patient_estimates(estimates: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar chart of estimated patients per cancer type by allele."""
    colors = {
        "R175H": "#D95319", "R248W": "#0072BD", "R273H": "#4DBEEE",
        "G245S": "#77AC30", "R249S": "#EDB120", "Y220C": "#7E2F8E", "R282W": "#A2142F",
    }
    pivot = estimates.pivot_table(
        index="cancer_type", columns="allele", values="estimated_annual_patients", fill_value=0
    )
    pivot = pivot[[a for a in TCGA_ALLELE_COLS if a in pivot.columns]]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).drop(columns="total")
    pivot = pivot.tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    bottom = np.zeros(len(pivot))
    for allele in pivot.columns:
        vals = pivot[allele].values
        ax.barh(range(len(pivot)), vals, left=bottom, label=allele,
                color=colors.get(allele, "gray"), alpha=0.8)
        bottom += vals

    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Estimated Annual US Patients")
    ax.set_title("TP53-Mutant Patient Population by Hotspot Allele")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_structural_vs_contact(tcga: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of structural vs contact allele fraction per cancer type."""
    # Filter to cancer types with enough mutations
    tcga_filt = tcga[tcga["TP53_mutated"] >= 20].copy()
    tcga_filt["structural_frac"] = tcga_filt["structural_total"] / tcga_filt["TP53_mutated"]
    tcga_filt["contact_frac"] = tcga_filt["contact_total"] / tcga_filt["TP53_mutated"]
    tcga_filt = tcga_filt.sort_values("structural_frac", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(tcga_filt) * 0.35)))
    y = np.arange(len(tcga_filt))
    w = 0.35
    ax.barh(y - w / 2, tcga_filt["structural_frac"], w, label="Structural", color="#D95319", alpha=0.7)
    ax.barh(y + w / 2, tcga_filt["contact_frac"], w, label="Contact", color="#0072BD", alpha=0.7)
    ax.set_yticks(y)
    labels = [f"{ct} (n={n})" for ct, n in zip(tcga_filt["cancer_type"], tcga_filt["TP53_mutated"])]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Fraction of TP53-mutant tumors")
    ax.set_title("Structural vs Contact TP53 Alleles Across TCGA Cancer Types")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comutation_heatmap(comut_df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of co-mutation rates by TP53 allele."""
    alleles = comut_df["allele"].tolist()
    pct_cols = [c for c in comut_df.columns if c.endswith("_pct")]
    gene_names = [c.replace("_pct", "") for c in pct_cols]

    data = comut_df[pct_cols].values

    fig, ax = plt.subplots(figsize=(10, max(4, len(alleles) * 0.5)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(gene_names)))
    ax.set_xticklabels(gene_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(alleles)))
    labels = [f"{a} (n={n})" for a, n in zip(alleles, comut_df["n_lines"])]
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(alleles)):
        for j in range(len(gene_names)):
            val = data[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Co-mutation rate (%)")
    ax.set_title("Co-mutation Landscape by TP53 Allele (DepMap)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- TCGA frequencies ---
    print("Loading TCGA TP53 frequencies...")
    tcga = load_tcga(TCGA_PATH)
    total_patients = tcga["n_sequenced"].sum()
    total_mutated = tcga["TP53_mutated"].sum()
    print(f"  {len(tcga)} cancer types, {total_patients} patients, "
          f"{total_mutated} TP53-mutated ({100 * total_mutated / total_patients:.1f}%)")

    # Pan-cancer allele totals from TCGA
    print("\n  Pan-cancer hotspot allele totals (TCGA):")
    for allele in TCGA_ALLELE_COLS:
        total = tcga[allele].sum()
        frac = total / total_mutated
        print(f"    {allele}: {total} ({frac:.1%})")
    structural = tcga["structural_total"].sum()
    contact = tcga["contact_total"].sum()
    print(f"  Structural total: {structural}, Contact total: {contact}")

    # Validate known enrichments
    print("\n  Validating known allele enrichments:")
    lihc = tcga[tcga["cancer_type"] == "LIHC"]
    if not lihc.empty:
        r249s_lihc = lihc.iloc[0]["R249S"]
        lihc_mut = lihc.iloc[0]["TP53_mutated"]
        print(f"    R249S in LIHC (HCC): {r249s_lihc}/{lihc_mut} "
              f"({100 * r249s_lihc / lihc_mut:.1f}%) — aflatoxin signature")
    ov = tcga[tcga["cancer_type"] == "OV"]
    if not ov.empty:
        r175h_ov = ov.iloc[0]["R175H"]
        ov_mut = ov.iloc[0]["TP53_mutated"]
        print(f"    R175H in OV (HGSOC): {r175h_ov}/{ov_mut} "
              f"({100 * r175h_ov / ov_mut:.1f}%)")

    # SKCM UV-passenger flag
    skcm = tcga[tcga["cancer_type"] == "SKCM"]
    if not skcm.empty:
        skcm_pct = skcm.iloc[0]["TP53_pct"]
        skcm_mut = skcm.iloc[0]["TP53_mutated"]
        hotspot_in_skcm = sum(skcm.iloc[0][a] for a in TCGA_ALLELE_COLS)
        print(f"\n  WARNING: SKCM TP53 rate ({skcm_pct}%) likely inflated by UV-passengers. "
              f"Only {hotspot_in_skcm}/{skcm_mut} are hotspot alleles.")

    # --- DepMap allele distributions ---
    print("\nLoading DepMap allele distributions...")
    depmap = load_depmap_allele_dist(OUTPUT_DIR)
    print(f"  {len(depmap)} cancer types with TP53-mutant lines")

    # --- Cross-validation ---
    print("\nCross-validating TCGA vs DepMap...")
    comparison = cross_validate(tcga, depmap)
    comparison.to_csv(OUTPUT_DIR / "tcga_depmap_comparison.csv", index=False)
    print(f"  {len(comparison)} cancer types compared")

    for _, row in comparison.iterrows():
        flags = []
        for allele in ["R175H", "R248W", "R273H"]:
            diff = row[f"diff_{allele}"]
            if abs(diff) > 0.05:
                direction = "over" if diff > 0 else "under"
                flags.append(f"{allele} {direction}-represented ({abs(diff):.0%})")
        if flags:
            print(f"  {row['tcga_type']}: {', '.join(flags)}")

    # --- Patient population estimates ---
    print("\nEstimating addressable patient populations...")
    estimates = estimate_populations(tcga)

    # Summary by allele
    allele_totals = estimates.groupby("allele")["estimated_annual_patients"].sum()
    total_tp53 = allele_totals.sum()
    print(f"\n  Estimated annual US TP53-hotspot-mutant patients: ~{total_tp53:,}")
    for allele, count in allele_totals.sort_values(ascending=False).items():
        print(f"    {allele}: ~{count:,} ({count / total_tp53:.1%})")

    # Top cancer types
    ct_totals = estimates.groupby("cancer_type")["estimated_annual_patients"].sum()
    print(f"\n  Top cancer types by TP53-hotspot patient volume:")
    for ct, count in ct_totals.sort_values(ascending=False).head(8).items():
        print(f"    {ct}: ~{count:,}")

    # Y220C-specific (rezatapopt addressable)
    y220c_est = estimates[estimates["allele"] == "Y220C"]
    y220c_total = y220c_est["estimated_annual_patients"].sum()
    print(f"\n  Y220C (rezatapopt-addressable): ~{y220c_total:,} annual US patients")
    for _, row in y220c_est[y220c_est["estimated_annual_patients"] > 0].head(5).iterrows():
        print(f"    {row['cancer_type']}: ~{row['estimated_annual_patients']:,}")

    # Save estimates
    with open(OUTPUT_DIR / "patient_population_estimates.json", "w") as f:
        json.dump({
            "total_hotspot_patients": int(total_tp53),
            "allele_totals": {str(k): int(v) for k, v in allele_totals.items()},
            "cancer_type_totals": {str(k): int(v) for k, v in ct_totals.sort_values(ascending=False).items()},
            "y220c_rezatapopt_addressable": int(y220c_total),
            "per_cancer_per_allele": estimates.to_dict(orient="records"),
        }, f, indent=2)

    # --- Co-mutation landscape ---
    print("\nAnalyzing co-mutation landscape (DepMap)...")
    comut = comutation_analysis(
        OUTPUT_DIR / "tp53_classified_lines.csv",
        DEPMAP_DIR / "OmicsSomaticMutations.csv",
    )
    comut.to_csv(OUTPUT_DIR / "comutation_by_allele.csv", index=False)
    print(f"  {len(comut)} allele groups analyzed")
    for _, row in comut.iterrows():
        top_genes = []
        for gene in COMUTATION_GENES:
            pct = row.get(f"{gene}_pct", 0)
            if pct > 20:
                top_genes.append(f"{gene}={pct:.0f}%")
        if top_genes:
            print(f"  {row['allele']} (n={row['n_lines']}): {', '.join(top_genes)}")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_allele_comparison(comparison, FIG_DIR / "tcga_vs_depmap_alleles.png")
    plot_patient_estimates(estimates, FIG_DIR / "patient_population_by_allele.png")
    plot_structural_vs_contact(tcga, FIG_DIR / "structural_vs_contact_tcga.png")
    if len(comut) > 0:
        plot_comutation_heatmap(comut, FIG_DIR / "comutation_heatmap.png")

    print(f"\nDone. Results in {OUTPUT_DIR.name}/")


if __name__ == "__main__":
    main()
