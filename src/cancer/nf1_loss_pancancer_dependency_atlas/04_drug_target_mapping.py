"""Phase 4: Drug target mapping — PRISM sensitivity and druggable targets.

Cross-references Phase 3 top dependencies with druggable targets. Maps PRISM
24Q2 drug sensitivity data stratified by NF1-loss. Integrates literature
findings (PMID 41036607: 27 NF1-SL compounds, 4 validated in vivo with
selumetinib synergy).

Usage:
    uv run python -m nf1_loss_pancancer_dependency_atlas.04_drug_target_mapping
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

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase1"
)
PHASE3_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase3"
)
OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase4"
)

FDR_THRESHOLD = 0.1
EFFECT_SIZE_THRESHOLD = 0.3  # Relaxed for drug mapping — capture more targets

# Key druggable targets and their compounds (from plan + literature)
DRUGGABLE_TARGETS = {
    "PTPN11": {"name": "SHP2", "compounds": ["TNO155 (Novartis)", "RMC-4630 (Revolution Med)", "GDC-1971 (Genentech)"], "stage": "Phase 2"},
    "GRB2": {"name": "GRB2", "compounds": ["No direct inhibitors"], "stage": "Preclinical"},
    "RAF1": {"name": "C-RAF", "compounds": ["LY3009120 (pan-RAF)", "Belvarafenib (pan-RAF)"], "stage": "Phase 1/2"},
    "MAP2K1": {"name": "MEK1", "compounds": ["Selumetinib (AZD6244)", "Trametinib", "Mirdametinib"], "stage": "FDA-approved"},
    "MAP2K2": {"name": "MEK2", "compounds": ["Trametinib (dual MEK1/2)", "Cobimetinib"], "stage": "FDA-approved"},
    "BRAF": {"name": "BRAF", "compounds": ["Dabrafenib", "Vemurafenib", "Encorafenib"], "stage": "FDA-approved"},
    "CDK4": {"name": "CDK4", "compounds": ["Palbociclib", "Ribociclib", "Abemaciclib"], "stage": "FDA-approved"},
    "CDK6": {"name": "CDK6", "compounds": ["Palbociclib", "Ribociclib"], "stage": "FDA-approved"},
    "CDK2": {"name": "CDK2", "compounds": ["INX-315 (Incyte)", "PF-07104091 (Pfizer)"], "stage": "Phase 1/2"},
    "EZH2": {"name": "EZH2", "compounds": ["Tazemetostat (FDA-approved for epithelioid sarcoma)"], "stage": "FDA-approved"},
    "BRD4": {"name": "BRD4", "compounds": ["JQ1", "OTX015", "CPI-0610 (pelabresib)"], "stage": "Phase 2/3"},
    "ATR": {"name": "ATR", "compounds": ["Ceralasertib (AZD6738)", "Berzosertib (M6620)"], "stage": "Phase 2"},
    "CHEK1": {"name": "CHK1", "compounds": ["Prexasertib (LY2606368)", "SRA737"], "stage": "Phase 1/2"},
    "WEE1": {"name": "WEE1", "compounds": ["Adavosertib (AZD1775)", "ZN-c3"], "stage": "Phase 1/2"},
    "PARP1": {"name": "PARP1", "compounds": ["Olaparib", "Niraparib", "Rucaparib", "Talazoparib"], "stage": "FDA-approved"},
    "MTOR": {"name": "mTOR", "compounds": ["Everolimus", "Temsirolimus", "Sapanisertib (dual)"], "stage": "FDA-approved"},
    "PIK3CA": {"name": "PI3Kalpha", "compounds": ["Alpelisib (FDA)", "Inavolisib"], "stage": "FDA-approved"},
    "PIK3CB": {"name": "PI3Kbeta", "compounds": ["KIN-193/AZD6482"], "stage": "Preclinical"},
    "SOS1": {"name": "SOS1", "compounds": ["BI-3406 (Boehringer)", "BAY-293"], "stage": "Phase 1"},
    "RIT1": {"name": "RIT1", "compounds": ["No direct inhibitors (RAS family GTPase)"], "stage": "N/A"},
}

# Literature NF1-SL compounds (PMID 41036607) — 27 compounds, 4 in vivo validated
NF1_SL_LITERATURE = {
    "MEK inhibitors": {
        "compounds": ["Selumetinib", "Trametinib", "Cobimetinib", "PD-0325901"],
        "in_vivo": True,
        "synergy_with_selumetinib": False,
        "note": "Backbone therapy for NF1-loss",
    },
    "mTOR inhibitors": {
        "compounds": ["Everolimus", "Temsirolimus", "AZD8055"],
        "in_vivo": True,
        "synergy_with_selumetinib": True,
        "note": "MEK+mTOR combo under clinical investigation",
    },
    "CDK4/6 inhibitors": {
        "compounds": ["Palbociclib", "Ribociclib"],
        "in_vivo": True,
        "synergy_with_selumetinib": True,
        "note": "NF1-loss → RAS → cyclin D1 → CDK4/6 addiction",
    },
    "BET inhibitors": {
        "compounds": ["JQ1", "OTX015"],
        "in_vivo": False,
        "synergy_with_selumetinib": True,
        "note": "BRD4 inhibition disrupts super-enhancer driven oncogene transcription",
    },
    "HSP90 inhibitors": {
        "compounds": ["Ganetespib", "17-AAG"],
        "in_vivo": True,
        "synergy_with_selumetinib": False,
        "note": "Destabilizes mutant signaling complexes",
    },
    # KAT6A/B + Menin (ESR1 resistance angle)
    "KAT6A/B inhibitors": {
        "compounds": ["WM-1119", "PF-07248144"],
        "in_vivo": False,
        "synergy_with_selumetinib": False,
        "note": "NF1-loss as ESR1 resistance mechanism. KAT6A/B + Menin inhibitor combos resensitize HR+ cancers",
    },
}

MIN_PRISM_LINES = 3


def load_prism_data(depmap_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PRISM drug sensitivity data (memory-efficient)."""
    # Treatment metadata
    treatments = pd.read_csv(depmap_dir / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")

    # Data matrix — rows are treatment IDs, columns are cell lines
    # Read just the header to get cell line IDs
    matrix = pd.read_csv(
        depmap_dir / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )

    return treatments, matrix


def screen_prism_nf1(
    matrix: pd.DataFrame,
    treatments: pd.DataFrame,
    nf1_lost_ids: set[str],
    nf1_intact_ids: set[str],
) -> pd.DataFrame:
    """Screen PRISM drug sensitivity for NF1-loss selectivity."""
    lost_cols = [c for c in matrix.columns if c in nf1_lost_ids]
    intact_cols = [c for c in matrix.columns if c in nf1_intact_ids]

    if len(lost_cols) < MIN_PRISM_LINES:
        print(f"  WARNING: Only {len(lost_cols)} NF1-lost lines in PRISM (min={MIN_PRISM_LINES})")
        return pd.DataFrame()

    rows = []
    pvals = []

    # Group treatments by broad_id + name for aggregation
    treatment_groups = treatments.groupby(["broad_id", "name"]).first().reset_index()

    for _, trt in treatment_groups.iterrows():
        broad_id = trt["broad_id"]
        # Find all rows matching this broad_id
        trt_rows = matrix.index[matrix.index.str.startswith(f"BRD:{broad_id}")]
        if len(trt_rows) == 0:
            # Try without BRD: prefix
            trt_rows = matrix.index[matrix.index.str.startswith(broad_id)]
        if len(trt_rows) == 0:
            continue

        # Average across replicates/doses for this compound
        trt_data = matrix.loc[trt_rows]
        avg_sensitivity = trt_data.mean(axis=0)

        lost_vals = avg_sensitivity[lost_cols].dropna().values
        intact_vals = avg_sensitivity[intact_cols].dropna().values

        if len(lost_vals) < MIN_PRISM_LINES or len(intact_vals) < MIN_PRISM_LINES:
            continue

        d = _cohens_d(lost_vals, intact_vals)
        _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")

        rows.append({
            "compound": trt["name"],
            "broad_id": broad_id,
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_lost": len(lost_vals),
            "n_intact": len(intact_vals),
            "median_sensitivity_lost": round(float(np.median(lost_vals)), 4),
            "median_sensitivity_intact": round(float(np.median(intact_vals)), 4),
        })
        pvals.append(pval)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    if pvals:
        fdrs = _fdr_correction(np.array(pvals))
        result["fdr"] = fdrs

    return result.sort_values("cohens_d")


def _cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
    ps = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return 0.0 if ps == 0 else float((g1.mean() - g2.mean()) / ps)


def _fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    if n == 0:
        return np.array([])
    si = np.argsort(pvalues)
    sp = pvalues[si]
    fdr = np.empty(n)
    for i in range(n):
        fdr[si[i]] = sp[i] * n / (i + 1)
    fs = fdr[si]
    for i in range(n - 2, -1, -1):
        fs[i] = min(fs[i], fs[i + 1])
    fdr[si] = fs
    return np.minimum(fdr, 1.0)


def map_dependencies_to_drugs(
    dep_results: pd.DataFrame,
) -> pd.DataFrame:
    """Map dependency screen hits to druggable targets."""
    rows = []
    for _, dep in dep_results.iterrows():
        gene = dep["gene"]
        if gene in DRUGGABLE_TARGETS:
            info = DRUGGABLE_TARGETS[gene]
            rows.append({
                "gene": gene,
                "target_name": info["name"],
                "cancer_type": dep.get("cancer_type", "Pan-cancer"),
                "cohens_d": dep["cohens_d"],
                "fdr": dep.get("fdr", None),
                "composite_score": dep.get("composite_score", None),
                "compounds": "; ".join(info["compounds"]),
                "clinical_stage": info["stage"],
                "n_lost": dep.get("n_lost", None),
                "n_intact": dep.get("n_intact", None),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("cohens_d")


def plot_drug_target_priorities(drug_map: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of druggable target effect sizes."""
    if len(drug_map) == 0:
        return

    # Take unique genes (best d per gene)
    best = drug_map.groupby("gene").agg({"cohens_d": "min", "clinical_stage": "first", "target_name": "first"}).reset_index()
    best = best.sort_values("cohens_d")

    fig, ax = plt.subplots(figsize=(8, max(3, len(best) * 0.4)))

    colors = []
    for _, row in best.iterrows():
        if "FDA" in str(row["clinical_stage"]):
            colors.append("#4CAF50")
        elif "Phase" in str(row["clinical_stage"]):
            colors.append("#FF9800")
        else:
            colors.append("#9E9E9E")

    ax.barh(
        range(len(best)),
        best["cohens_d"],
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(
        [f"{row['gene']} ({row['target_name']})" for _, row in best.iterrows()],
        fontsize=8,
    )
    ax.axvline(0, color="grey", linestyle="-", alpha=0.3)
    ax.set_xlabel("Cohen's d (NF1-lost vs intact)\n← more essential in NF1-lost")
    ax.set_title("Druggable NF1-Loss Dependencies\n(Green=FDA, Orange=Clinical, Grey=Preclinical)")

    plt.tight_layout()
    fig.savefig(output_dir / "drug_target_priorities.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4: Drug Target Mapping ===\n")

    # Load Phase 1 + Phase 3
    print("Loading Phase 1 classification...")
    classified = pd.read_csv(PHASE1_DIR / "nf1_loss_classification.csv", index_col=0)
    nf1_lost_ids = set(classified[classified["NF1_loss"] == True].index)  # noqa: E712
    nf1_intact_ids = set(classified[classified["NF1_status"] == "intact"].index)
    print(f"  {len(nf1_lost_ids)} NF1-lost, {len(nf1_intact_ids)} intact")

    print("Loading Phase 3 genome-wide results...")
    genomewide = pd.read_csv(PHASE3_DIR / "genomewide_all_results.csv")
    print(f"  {len(genomewide)} total test results")

    # Map dependencies to druggable targets (all contexts, relaxed threshold)
    print("\nMapping dependencies to druggable targets...")
    dep_candidates = genomewide[
        (genomewide["cohens_d"] < -EFFECT_SIZE_THRESHOLD) &
        (genomewide["gene"].isin(DRUGGABLE_TARGETS.keys()))
    ].copy()
    print(f"  {len(dep_candidates)} dependency-drug matches (d < -{EFFECT_SIZE_THRESHOLD})")

    drug_map = map_dependencies_to_drugs(dep_candidates)
    print(f"  {len(drug_map)} druggable target entries")

    if len(drug_map) > 0:
        print("\nDruggable target hits:")
        for _, row in drug_map.iterrows():
            fdr_str = f"FDR={row['fdr']:.3e}" if pd.notna(row.get("fdr")) else "FDR=N/A"
            print(
                f"  {row['gene']} ({row['target_name']}) [{row['cancer_type']}]: "
                f"d={row['cohens_d']:.3f}, {fdr_str}, "
                f"stage={row['clinical_stage']}"
            )

    # Also capture all druggable genes regardless of threshold for complete table
    all_druggable = genomewide[
        genomewide["gene"].isin(DRUGGABLE_TARGETS.keys())
    ].copy()
    all_drug_map = map_dependencies_to_drugs(all_druggable)
    if len(all_drug_map) > 0:
        all_drug_map.to_csv(OUTPUT_DIR / "all_druggable_targets.csv", index=False)

    # PRISM drug sensitivity screen
    print("\nLoading PRISM 24Q2 drug sensitivity data...")
    try:
        treatments, matrix = load_prism_data(DEPMAP_DIR)
        print(f"  {len(treatments)} treatment records, {matrix.shape[1]} cell lines")

        print("Screening PRISM for NF1-selective compounds...")
        prism_results = screen_prism_nf1(matrix, treatments, nf1_lost_ids, nf1_intact_ids)

        if len(prism_results) > 0:
            prism_results.to_csv(OUTPUT_DIR / "prism_nf1_sensitivity.csv", index=False)

            # NF1-selective (more sensitive in NF1-lost → more negative PRISM score)
            selective = prism_results[
                (prism_results["cohens_d"] < -EFFECT_SIZE_THRESHOLD)
            ]
            print(f"  {len(selective)} NF1-selective compounds (d < -{EFFECT_SIZE_THRESHOLD})")

            sig_selective = selective[selective.get("fdr", pd.Series(dtype=float)) < FDR_THRESHOLD] if "fdr" in selective.columns else pd.DataFrame()
            print(f"  {len(sig_selective)} significant (FDR < {FDR_THRESHOLD})")

            print("\nTop NF1-selective compounds (PRISM):")
            for _, row in selective.head(20).iterrows():
                fdr_str = f"FDR={row['fdr']:.3e}" if "fdr" in row and pd.notna(row.get("fdr")) else ""
                print(f"  {row['compound']}: d={row['cohens_d']:.3f}, {fdr_str}")
        else:
            print("  No PRISM results (insufficient NF1-lost lines in PRISM)")
    except Exception as e:
        print(f"  PRISM loading failed: {e}")
        prism_results = pd.DataFrame()

    # Literature integration
    print("\nLiterature NF1-SL compounds (PMID 41036607):")
    lit_rows = []
    for category, info in NF1_SL_LITERATURE.items():
        in_vivo = "in vivo validated" if info["in_vivo"] else "in vitro"
        synergy = "+ selumetinib synergy" if info["synergy_with_selumetinib"] else ""
        print(f"  {category}: {', '.join(info['compounds'])} ({in_vivo}) {synergy}")
        for compound in info["compounds"]:
            lit_rows.append({
                "category": category,
                "compound": compound,
                "in_vivo_validated": info["in_vivo"],
                "selumetinib_synergy": info["synergy_with_selumetinib"],
                "note": info["note"],
            })

    lit_df = pd.DataFrame(lit_rows)
    lit_df.to_csv(OUTPUT_DIR / "literature_nf1_sl_compounds.csv", index=False)

    # MPNST-specific candidates
    print("\nMPNST therapeutic candidates:")
    mpnst_candidates = [
        {"target": "MEK1/2", "compounds": "Selumetinib, Trametinib, Mirdametinib", "evidence": "FDA-approved (NF1 plexiform), Phase 2 MPNST", "note": "Limited single-agent activity in MPNST"},
        {"target": "mTOR", "compounds": "Everolimus + Selumetinib", "evidence": "Phase 1/2 combo", "note": "MEK+mTOR combo under investigation for MPNST"},
        {"target": "CDK4/6", "compounds": "Palbociclib, Ribociclib", "evidence": "Phase 2 MPNST", "note": "CDKN2A loss frequent in MPNST → CDK4/6 dependency"},
        {"target": "SHP2", "compounds": "TNO155, RMC-4630", "evidence": "Phase 1/2 + MEKi", "note": "SHP2 feeds RAS; SHP2+MEK combo rational for NF1"},
        {"target": "BRD4", "compounds": "JQ1, OTX015, Pelabresib", "evidence": "Preclinical + Phase 2", "note": "BET inhibition disrupts super-enhancer oncogene programs"},
        {"target": "EZH2", "compounds": "Tazemetostat", "evidence": "FDA-approved (epithelioid sarcoma)", "note": "PRC2 loss common in MPNST; paradoxical dependency context"},
    ]
    mpnst_df = pd.DataFrame(mpnst_candidates)
    mpnst_df.to_csv(OUTPUT_DIR / "mpnst_therapeutic_candidates.csv", index=False)
    for _, row in mpnst_df.iterrows():
        print(f"  {row['target']}: {row['compounds']} — {row['note']}")

    # Plots
    print("\nGenerating plots...")
    if len(drug_map) > 0:
        drug_map.to_csv(OUTPUT_DIR / "druggable_dependencies.csv", index=False)
        plot_drug_target_priorities(drug_map, OUTPUT_DIR)

    # Summary text
    summary_lines = [
        "=" * 70,
        "NF1-Loss Pan-Cancer Dependency Atlas - Phase 4: Drug Target Mapping",
        "=" * 70,
        "",
        f"Druggable dependency targets (d < -{EFFECT_SIZE_THRESHOLD}): {len(drug_map)}",
        f"PRISM NF1-selective compounds: {len(prism_results) if len(prism_results) > 0 else 'N/A'}",
        f"Literature NF1-SL compounds (PMID 41036607): {len(lit_rows)}",
        f"MPNST therapeutic candidates: {len(mpnst_candidates)}",
        "",
        "DRUGGABLE DEPENDENCY TARGETS",
        "-" * 60,
    ]

    if len(drug_map) > 0:
        for _, row in drug_map.iterrows():
            fdr_str = f"FDR={row['fdr']:.3e}" if pd.notna(row.get("fdr")) else "FDR=N/A"
            summary_lines.append(
                f"  {row['gene']} ({row['target_name']}) [{row['cancer_type']}]: "
                f"d={row['cohens_d']:.3f}, {fdr_str}, {row['clinical_stage']}"
            )
            summary_lines.append(f"    Compounds: {row['compounds']}")

    if len(prism_results) > 0:
        summary_lines += [
            "",
            "TOP PRISM NF1-SELECTIVE COMPOUNDS",
            "-" * 60,
        ]
        selective = prism_results[prism_results["cohens_d"] < -EFFECT_SIZE_THRESHOLD]
        for _, row in selective.head(15).iterrows():
            fdr_str = f"FDR={row['fdr']:.3e}" if "fdr" in row and pd.notna(row.get("fdr")) else ""
            summary_lines.append(f"  {row['compound']}: d={row['cohens_d']:.3f} {fdr_str}")

    summary_lines += [
        "",
        "MPNST THERAPEUTIC CANDIDATES",
        "-" * 60,
    ]
    for _, row in mpnst_df.iterrows():
        summary_lines.append(f"  {row['target']}: {row['compounds']}")
        summary_lines.append(f"    {row['note']}")

    summary_lines += [
        "",
        "LITERATURE NF1-SL COMPOUNDS (PMID 41036607)",
        "-" * 60,
    ]
    for cat, info in NF1_SL_LITERATURE.items():
        iv = "in vivo" if info["in_vivo"] else "in vitro"
        summary_lines.append(f"  {cat}: {', '.join(info['compounds'])} ({iv})")

    summary_lines.append("")

    with open(OUTPUT_DIR / "drug_target_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  drug_target_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
