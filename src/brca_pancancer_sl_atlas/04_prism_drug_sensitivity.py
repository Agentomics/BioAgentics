"""Phase 4: PRISM drug sensitivity analysis for BRCA-deficient cell lines.

Screens all PRISM 24Q2 drugs for BRCA-selective sensitivity, tests available
DDR drugs (ATR inhibitor 2), stratifies by BRCA1 vs BRCA2, 53BP1/SHLD status,
and pre-RC covariate. Cross-references Phase 3 genetic hits for gene-drug
concordance.

Note: PARPi drugs (olaparib, talazoparib, niraparib, rucaparib) and most
named DDR agents (ceralasertib, adavosertib, prexasertib) are not in the
PRISM 24Q2 repurposing library. The genome-wide drug screen identifies
repurposing candidates with BRCA-selective sensitivity.

Usage:
    uv run python -m brca_pancancer_sl_atlas.04_prism_drug_sensitivity
"""

from __future__ import annotations

import json
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
PHASE1_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase1"
PHASE3_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase3"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase4"

BRCA_GROUPS = ["any_brca", "brca1_only", "brca2_only"]
MIN_SAMPLES = 3

# Named drugs to search for in PRISM (searched by name and known BRD keys)
NAMED_DRUGS = {
    # PARPi
    "olaparib": {"search": ["olaparib", "BRD-K82859696"], "mechanism": "PARP1/2 inhibitor", "gene": "PARP1"},
    "talazoparib": {"search": ["talazoparib", "BRD-K01337880"], "mechanism": "PARP1/2 trapping", "gene": "PARP1"},
    "niraparib": {"search": ["niraparib", "BRD-K28907609"], "mechanism": "PARP1/2 inhibitor", "gene": "PARP1"},
    "rucaparib": {"search": ["rucaparib"], "mechanism": "PARP1/2 inhibitor", "gene": "PARP1"},
    "veliparib": {"search": ["veliparib"], "mechanism": "PARP1/2 inhibitor", "gene": "PARP1"},
    # DDR
    "ATR inhibitor 2": {"search": ["ATR inhibitor 2", "BRD-K00091072"], "mechanism": "ATR inhibitor", "gene": "ATR"},
    "ceralasertib": {"search": ["ceralasertib", "AZD6738"], "mechanism": "ATR inhibitor", "gene": "ATR"},
    "adavosertib": {"search": ["adavosertib", "AZD1775"], "mechanism": "WEE1 inhibitor", "gene": "WEE1"},
    "prexasertib": {"search": ["prexasertib", "LY2606368"], "mechanism": "CHK1 inhibitor", "gene": "CHEK1"},
    "novobiocin": {"search": ["novobiocin"], "mechanism": "POLθ ATPase inhibitor", "gene": "POLQ"},
}


def _extract_compound_key(broad_id: str) -> str:
    """Extract compound key (BRD-KXXXXXXXX) from a broad_id string."""
    m = re.search(r"BRD-[A-Za-z]\d{8}", str(broad_id))
    return m.group(0) if m else str(broad_id)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD): group1 - group2."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_sd)


def fdr_correction(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    fdr = pvals * n / ranks
    sorted_idx = np.argsort(pvals)[::-1]
    sorted_fdr = fdr[sorted_idx]
    for i in range(1, len(sorted_fdr)):
        sorted_fdr[i] = min(sorted_fdr[i], sorted_fdr[i - 1])
    fdr[sorted_idx] = sorted_fdr
    return np.minimum(fdr, 1.0)


def find_named_drugs(
    meta: pd.DataFrame,
    prism: pd.DataFrame,
) -> dict[str, tuple[str, pd.Series]]:
    """Search PRISM for named drugs. Returns {drug_name: (treatment_id, sensitivity)}."""
    found = {}
    all_names = set(meta["name"].dropna().unique())
    name_to_brd = {}
    for _, row in meta.drop_duplicates("name").iterrows():
        if pd.notna(row["name"]) and pd.notna(row.get("broad_id")):
            name_to_brd[row["name"]] = _extract_compound_key(str(row["broad_id"]))

    for drug_name, info in NAMED_DRUGS.items():
        for term in info["search"]:
            # Try exact name match
            if term in all_names:
                brd = name_to_brd.get(term, "")
                matching = [idx for idx in prism.index if brd in str(idx)]
                if matching:
                    found[drug_name] = (matching[0], prism.loc[matching[0]])
                    break
            # Try BRD key in matrix index
            if "BRD-" in term:
                matching = [idx for idx in prism.index if term in str(idx)]
                if matching:
                    found[drug_name] = (matching[0], prism.loc[matching[0]])
                    break

    return found


def genomewide_drug_screen(
    prism: pd.DataFrame,
    classified: pd.DataFrame,
    meta: pd.DataFrame,
    brca_group: str = "any_brca",
) -> pd.DataFrame:
    """Screen all PRISM drugs for BRCA-selective sensitivity."""
    if brca_group == "any_brca":
        def_ids = classified[classified["brca_combined_status"] == "deficient"].index
    elif brca_group == "brca1_only":
        def_ids = classified[classified["brca1_status"] == "deficient"].index
    elif brca_group == "brca2_only":
        def_ids = classified[classified["brca2_status"] == "deficient"].index
    else:
        raise ValueError(f"Unknown brca_group: {brca_group}")

    prof_ids = classified[classified["brca_combined_status"] == "proficient"].index

    # Build name lookup
    name_lookup = {}
    for _, row in meta.drop_duplicates("broad_id").iterrows():
        if pd.notna(row.get("broad_id")) and pd.notna(row.get("name")):
            key = _extract_compound_key(str(row["broad_id"]))
            name_lookup[key] = str(row["name"])

    rows = []
    pvals = []

    for treatment_id in prism.index:
        drug_sens = prism.loc[treatment_id]
        def_vals = drug_sens.reindex(def_ids).dropna().values
        prof_vals = drug_sens.reindex(prof_ids).dropna().values

        if len(def_vals) < MIN_SAMPLES or len(prof_vals) < MIN_SAMPLES:
            continue

        d = cohens_d(def_vals, prof_vals)
        _, pval = stats.mannwhitneyu(def_vals, prof_vals, alternative="two-sided")

        compound_key = _extract_compound_key(str(treatment_id))
        drug_name = name_lookup.get(compound_key, "")

        rows.append({
            "treatment_id": treatment_id,
            "drug_name": drug_name,
            "brca_group": brca_group,
            "cohens_d": round(d, 4),
            "pvalue": float(pval),
            "n_mut": len(def_vals),
            "n_wt": len(prof_vals),
            "median_def": round(float(np.median(def_vals)), 4),
            "median_prof": round(float(np.median(prof_vals)), 4),
        })
        pvals.append(pval)

    result = pd.DataFrame(rows)
    if pvals:
        result["fdr"] = fdr_correction(np.array(pvals))
    return result


def targeted_drug_analysis(
    found_drugs: dict[str, tuple[str, pd.Series]],
    classified: pd.DataFrame,
    qualifying_types: list[str],
) -> pd.DataFrame:
    """Analyze found drugs across BRCA groups and cancer types."""
    rows = []

    for drug_name, (treatment_id, drug_sens) in found_drugs.items():
        info = NAMED_DRUGS.get(drug_name, {})
        mechanism = info.get("mechanism", "")

        for brca_group in BRCA_GROUPS:
            if brca_group == "any_brca":
                def_mask = classified["brca_combined_status"] == "deficient"
            elif brca_group == "brca1_only":
                def_mask = classified["brca1_status"] == "deficient"
            elif brca_group == "brca2_only":
                def_mask = classified["brca2_status"] == "deficient"
            else:
                continue

            prof_mask = classified["brca_combined_status"] == "proficient"

            # Pan-cancer and per cancer type
            for ct in [None] + qualifying_types:
                if ct:
                    ct_mask = classified["OncotreeLineage"] == ct
                    d_ids = classified[def_mask & ct_mask].index
                    p_ids = classified[prof_mask & ct_mask].index
                else:
                    d_ids = classified[def_mask].index
                    p_ids = classified[prof_mask].index

                d_vals = drug_sens.reindex(d_ids).dropna().values
                p_vals = drug_sens.reindex(p_ids).dropna().values

                if len(d_vals) < MIN_SAMPLES or len(p_vals) < MIN_SAMPLES:
                    continue

                d = cohens_d(d_vals, p_vals)
                _, pval = stats.mannwhitneyu(d_vals, p_vals, alternative="two-sided")

                rows.append({
                    "drug": drug_name,
                    "mechanism": mechanism,
                    "cancer_type": ct or "Pan-Cancer",
                    "brca_group": brca_group,
                    "cohens_d": round(d, 4),
                    "pvalue": float(pval),
                    "n_mut": len(d_vals),
                    "n_wt": len(p_vals),
                    "median_def": round(float(np.median(d_vals)), 4),
                    "median_prof": round(float(np.median(p_vals)), 4),
                })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
    return result


def shld_stratify_drugs(
    found_drugs: dict[str, tuple[str, pd.Series]],
    classified: pd.DataFrame,
) -> pd.DataFrame:
    """Stratify drug response by 53BP1/SHLD status within BRCA-deficient lines."""
    brca_def = classified[classified["brca_combined_status"] == "deficient"]
    intact_ids = brca_def[brca_def["shld_complex_status"] == "SHLD-intact"].index
    lost_ids = brca_def[brca_def["shld_complex_status"] == "SHLD-lost"].index

    rows = []
    for drug_name, (_, drug_sens) in found_drugs.items():
        intact_vals = drug_sens.reindex(intact_ids).dropna().values
        lost_vals = drug_sens.reindex(lost_ids).dropna().values

        if len(intact_vals) < 2 or len(lost_vals) < 2:
            continue

        d = cohens_d(lost_vals, intact_vals)
        _, pval = stats.mannwhitneyu(intact_vals, lost_vals, alternative="two-sided")

        rows.append({
            "drug": drug_name,
            "mechanism": NAMED_DRUGS.get(drug_name, {}).get("mechanism", ""),
            "comparison": "SHLD-lost vs SHLD-intact (BRCA-def only)",
            "cohens_d": round(d, 4),
            "pvalue": pval,
            "n_shld_intact": len(intact_vals),
            "n_shld_lost": len(lost_vals),
            "median_intact": round(float(np.median(intact_vals)), 4),
            "median_lost": round(float(np.median(lost_vals)), 4),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
    return result


def pre_rc_drug_covariate(
    found_drugs: dict[str, tuple[str, pd.Series]],
    classified: pd.DataFrame,
) -> pd.DataFrame:
    """Check if pre-RC status affects drug sensitivity within BRCA-deficient lines."""
    brca_def = classified[classified["brca_combined_status"] == "deficient"]
    normal_ids = brca_def[brca_def["pre_rc_status"] == "normal"].index
    compromised_ids = brca_def[brca_def["pre_rc_status"] != "normal"].index

    rows = []
    for drug_name, (_, drug_sens) in found_drugs.items():
        normal_vals = drug_sens.reindex(normal_ids).dropna().values
        comp_vals = drug_sens.reindex(compromised_ids).dropna().values

        if len(normal_vals) < 2 or len(comp_vals) < 2:
            continue

        d = cohens_d(comp_vals, normal_vals)
        _, pval = stats.mannwhitneyu(normal_vals, comp_vals, alternative="two-sided")

        rows.append({
            "drug": drug_name,
            "comparison": "pre-RC-compromised vs normal (BRCA-def only)",
            "cohens_d": round(d, 4),
            "pvalue": pval,
            "n_normal": len(normal_vals),
            "n_compromised": len(comp_vals),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
    return result


def gene_drug_concordance(
    found_drugs: dict[str, tuple[str, pd.Series]],
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    phase3_hits: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate CRISPR dependency with PRISM sensitivity for found drugs."""
    rows = []

    for drug_name, (_, drug_sens) in found_drugs.items():
        gene = NAMED_DRUGS.get(drug_name, {}).get("gene", "")
        if not gene or gene not in crispr.columns:
            continue

        gene_dep = crispr[gene]
        common = drug_sens.dropna().index.intersection(gene_dep.dropna().index)
        common = common.intersection(classified.index)

        if len(common) < 10:
            continue

        r, p = stats.spearmanr(drug_sens[common].values, gene_dep[common].values)
        in_phase3 = gene in phase3_hits["gene"].values if len(phase3_hits) > 0 else False

        rows.append({
            "drug": drug_name,
            "gene": gene,
            "mechanism": NAMED_DRUGS.get(drug_name, {}).get("mechanism", ""),
            "spearman_r": round(float(r), 4),
            "pvalue": float(p),
            "n_lines": len(common),
            "gene_is_phase3_sl_hit": in_phase3,
        })

    return pd.DataFrame(rows)


def plot_genomewide_volcano(results: pd.DataFrame, output_dir: Path) -> None:
    """Volcano plot of genome-wide drug screen."""
    if len(results) == 0 or "fdr" not in results.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    x = results["cohens_d"].values
    y = -np.log10(np.clip(results["fdr"].values, 1e-50, None))

    sig = (results["fdr"] < 0.05) & (results["cohens_d"].abs() > 0.3)
    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)
    ax.scatter(x[sig], y[sig], c="#D95319", s=15, alpha=0.8)

    # Label top BRCA-selective drugs
    sl_drugs = results[sig & (results["cohens_d"] < 0)].nsmallest(10, "cohens_d")
    for _, row in sl_drugs.iterrows():
        label = row["drug_name"] or str(row["treatment_id"])[:20]
        ax.annotate(label, (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                    fontsize=6, ha="right")

    ax.axhline(-np.log10(0.05), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-0.3, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(0.3, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (negative = more sensitive in BRCA-deficient)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("Genome-wide Drug Screen: BRCA-Selective Sensitivity")

    fig.tight_layout()
    fig.savefig(output_dir / "genomewide_drug_volcano.png", dpi=150)
    plt.close(fig)


def plot_targeted_results(results: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of targeted drug sensitivity."""
    if len(results) == 0:
        return

    pan = results[results["cancer_type"] == "Pan-Cancer"]
    if len(pan) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(pan) * 0.35)))
    pan_sorted = pan.sort_values("cohens_d")

    y_pos = range(len(pan_sorted))
    colors = []
    for _, row in pan_sorted.iterrows():
        if row.get("fdr", 1) < 0.05 and row["cohens_d"] < 0:
            colors.append("#E53935")
        elif row.get("fdr", 1) < 0.05:
            colors.append("#1E88E5")
        else:
            colors.append("#9E9E9E")

    ax.barh(y_pos, pan_sorted["cohens_d"], color=colors, alpha=0.8, height=0.6)
    labels = [
        f"{row['drug']} | {row['brca_group']} (n={row['n_mut']}v{row['n_wt']})"
        for _, row in pan_sorted.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Cohen's d (negative = more sensitive in BRCA-deficient)")
    ax.set_title("Targeted Drug Sensitivity by BRCA Status")

    fig.tight_layout()
    fig.savefig(output_dir / "targeted_drug_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_top_brca_selective_drugs(results: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of top BRCA-selective drugs across BRCA groups."""
    if len(results) == 0 or "fdr" not in results.columns:
        return

    # Get top drugs with BRCA-selective signal
    sig = results[(results["pvalue"] < 0.05) & (results["cohens_d"] < -0.3)]
    if len(sig) == 0:
        return

    # Aggregate by drug across groups
    drug_summary = (
        sig.groupby("drug_name")
        .agg(mean_d=("cohens_d", "mean"), n_groups=("brca_group", "nunique"))
        .sort_values("mean_d")
    )
    top_drugs = drug_summary.head(20).index.tolist()
    top_drugs = [d for d in top_drugs if d]  # filter empty names

    if not top_drugs:
        return

    pivot = sig[sig["drug_name"].isin(top_drugs)].pivot_table(
        index="drug_name", columns="brca_group", values="cohens_d", aggfunc="first"
    )
    pivot = pivot.reindex(top_drugs)

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.7)
    ax.set_title("Top BRCA-Selective Drugs by Group")
    fig.tight_layout()
    fig.savefig(output_dir / "ddr_drug_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4: PRISM Drug Sensitivity Analysis ===\n")

    # --- Load data ---
    print("Loading Phase 1 classifier output...")
    classified = pd.read_csv(PHASE1_DIR / "brca_classified_lines.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "cancer_type_summary.csv")
    qualifying = summary[
        summary["qualifies_primary"] | summary["qualifies_exploratory"]
    ]["cancer_type"].tolist()

    n_def = (classified["brca_combined_status"] == "deficient").sum()
    n_prof = (classified["brca_combined_status"] == "proficient").sum()
    print(f"  {n_def} BRCA-deficient, {n_prof} proficient")

    # Load Phase 3 hits
    phase3_path = PHASE3_DIR / "genomewide_sl_hits.csv"
    phase3_hits = pd.read_csv(phase3_path) if phase3_path.exists() else pd.DataFrame()
    if len(phase3_hits) > 0:
        print(f"  Loaded {len(phase3_hits)} Phase 3 SL hits")

    print("\nLoading PRISM 24Q2 data...")
    meta = pd.read_csv(DEPMAP_DIR / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")
    prism = pd.read_csv(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )
    print(f"  {prism.shape[0]} treatments x {prism.shape[1]} cell lines")

    # --- Search for named drugs ---
    print("\n--- Named Drug Search ---")
    found_drugs = find_named_drugs(meta, prism)
    drug_availability = []
    for drug_name in NAMED_DRUGS:
        status = "FOUND" if drug_name in found_drugs else "NOT IN PRISM 24Q2"
        drug_availability.append({
            "drug": drug_name,
            "mechanism": NAMED_DRUGS[drug_name]["mechanism"],
            "found": drug_name in found_drugs,
        })
        print(f"  {drug_name} ({NAMED_DRUGS[drug_name]['mechanism']}): {status}")

    pd.DataFrame(drug_availability).to_csv(
        OUTPUT_DIR / "drug_availability.csv", index=False
    )

    # --- Targeted analysis of found drugs ---
    targeted_results = pd.DataFrame()
    if found_drugs:
        print("\n--- Targeted Drug Analysis ---")
        targeted_results = targeted_drug_analysis(found_drugs, classified, qualifying)
        targeted_results.to_csv(OUTPUT_DIR / "parpi_sensitivity.csv", index=False)

        if len(targeted_results) > 0:
            pan = targeted_results[targeted_results["cancer_type"] == "Pan-Cancer"]
            for _, row in pan.iterrows():
                sig = " *" if row.get("fdr", 1) < 0.05 else ""
                print(f"  {row['drug']} | {row['brca_group']}: "
                      f"d={row['cohens_d']:.3f} p={row['pvalue']:.4f}{sig}")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "parpi_sensitivity.csv", index=False)
        print("\n  No named drugs found in PRISM — targeted analysis skipped")

    # --- Genome-wide drug screen ---
    print("\n--- Genome-Wide Drug Screen ---")
    all_gw_results = []
    for brca_group in BRCA_GROUPS:
        print(f"  Screening {brca_group}...")
        gw = genomewide_drug_screen(prism, classified, meta, brca_group)
        all_gw_results.append(gw)

        if len(gw) > 0 and "fdr" in gw.columns:
            sig = gw[(gw["fdr"] < 0.05) & (gw["cohens_d"] < -0.3)]
            nominal = gw[(gw["pvalue"] < 0.05) & (gw["cohens_d"] < -0.3)]
            print(f"    {len(gw)} drugs tested, {len(sig)} FDR-sig, {len(nominal)} nominal")
            for _, row in sig.nsmallest(5, "cohens_d").iterrows():
                print(f"      {row['drug_name'] or row['treatment_id']}: "
                      f"d={row['cohens_d']:.3f} FDR={row['fdr']:.4f}")

    gw_combined = pd.concat(all_gw_results, ignore_index=True) if all_gw_results else pd.DataFrame()
    gw_combined.to_csv(OUTPUT_DIR / "ddr_drug_sensitivity.csv", index=False)

    # --- 53BP1/SHLD stratification ---
    print("\n--- 53BP1/SHLD Drug Stratification ---")
    if found_drugs:
        shld_results = shld_stratify_drugs(found_drugs, classified)
        shld_results.to_csv(OUTPUT_DIR / "shld_stratified_drug_response.csv", index=False)
        if len(shld_results) > 0:
            for _, row in shld_results.iterrows():
                direction = "less sensitive" if row["cohens_d"] > 0 else "more sensitive"
                print(f"  {row['drug']}: SHLD-lost {direction}, d={row['cohens_d']:.3f}")
        else:
            print("  Insufficient SHLD-lost lines for drug stratification")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "shld_stratified_drug_response.csv", index=False)
        print("  No drugs available for SHLD stratification")

    # --- Pre-RC covariate ---
    print("\n--- Pre-RC Drug Covariate ---")
    if found_drugs:
        pre_rc = pre_rc_drug_covariate(found_drugs, classified)
        pre_rc.to_csv(OUTPUT_DIR / "pre_rc_drug_covariate.csv", index=False)
        if len(pre_rc) > 0:
            for _, row in pre_rc.iterrows():
                print(f"  {row['drug']}: d={row['cohens_d']:.3f} p={row['pvalue']:.4f}")
        else:
            print("  Insufficient pre-RC-compromised lines")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "pre_rc_drug_covariate.csv", index=False)
        print("  No drugs available for pre-RC analysis")

    # --- Gene-drug concordance ---
    print("\n--- Gene-Drug Concordance ---")
    if found_drugs:
        crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
        concordance = gene_drug_concordance(found_drugs, classified, crispr, phase3_hits)
        concordance.to_csv(OUTPUT_DIR / "gene_drug_concordance.csv", index=False)
        if len(concordance) > 0:
            for _, row in concordance.iterrows():
                ph3 = " [Phase 3 SL hit]" if row["gene_is_phase3_sl_hit"] else ""
                print(f"  {row['drug']} vs {row['gene']}: r={row['spearman_r']:.3f}{ph3}")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "gene_drug_concordance.csv", index=False)
        print("  No drugs available for concordance analysis")

    # --- Plots ---
    print("\nGenerating plots...")
    # Genome-wide volcano for any_brca
    any_brca_gw = gw_combined[gw_combined["brca_group"] == "any_brca"] if len(gw_combined) > 0 else pd.DataFrame()
    plot_genomewide_volcano(any_brca_gw, OUTPUT_DIR)
    print("  genomewide_drug_volcano.png")

    plot_targeted_results(targeted_results, OUTPUT_DIR)
    print("  targeted_drug_sensitivity.png")

    plot_top_brca_selective_drugs(gw_combined, OUTPUT_DIR)
    print("  ddr_drug_heatmap.png")

    # --- Combination candidates ---
    combo = {}
    if len(gw_combined) > 0 and "fdr" in gw_combined.columns:
        brca_selective = gw_combined[
            (gw_combined["brca_group"] == "any_brca")
            & (gw_combined["cohens_d"] < -0.3)
            & (gw_combined["pvalue"] < 0.05)
        ].nsmallest(20, "cohens_d")

        for _, row in brca_selective.iterrows():
            name = row["drug_name"] or str(row["treatment_id"])
            combo[name] = {
                "cohens_d": float(row["cohens_d"]),
                "pvalue": float(row["pvalue"]),
                "fdr": float(row.get("fdr", 1)),
                "n_mut": int(row["n_mut"]),
                "n_wt": int(row["n_wt"]),
                "rationale": f"BRCA-selective drug sensitivity (d={row['cohens_d']:.3f})",
            }

    with open(OUTPUT_DIR / "combination_candidates.json", "w") as f:
        json.dump(combo, f, indent=2)

    # --- Summary ---
    n_found = sum(1 for d in drug_availability if d["found"])
    n_gw_sig = 0
    n_gw_nominal = 0
    if len(any_brca_gw) > 0:
        if "fdr" in any_brca_gw.columns:
            n_gw_sig = int(((any_brca_gw["fdr"] < 0.05) & (any_brca_gw["cohens_d"] < -0.3)).sum())
        n_gw_nominal = int(((any_brca_gw["pvalue"] < 0.05) & (any_brca_gw["cohens_d"] < -0.3)).sum())

    print(f"\n=== Phase 4 Complete ===")
    print(f"  Named drugs found: {n_found}/{len(NAMED_DRUGS)}")
    print(f"  Genome-wide: {n_gw_sig} FDR-significant, {n_gw_nominal} nominal BRCA-selective drugs")
    print(f"  Combination candidates: {len(combo)}")


if __name__ == "__main__":
    main()
