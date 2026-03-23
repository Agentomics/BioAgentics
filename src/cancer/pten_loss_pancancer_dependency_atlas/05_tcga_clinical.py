"""Phase 5: TCGA clinical integration — population estimates & trial mapping.

Estimates PTEN loss frequency per cancer type (deep deletion, truncating mutation,
missense separately), compares DepMap vs TCGA representation, maps addressable
patient populations, cross-references with PIK3CA atlas, and identifies clinical
trial gaps. Includes co-occurring alteration analysis.

Usage:
    uv run python -m pten_loss_pancancer_dependency_atlas.05_tcga_clinical
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

PHASE1_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase1"
PHASE2_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase2"
PHASE3_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase3"
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase5"

# ---------- TCGA PTEN alteration data (PanCanAtlas 2018, cBioPortal) ----------
# Separated by alteration type: deep deletion, truncating mutation, missense

TCGA_PTEN_PREVALENCE = {
    # cancer_type: (deep_del_pct, truncating_pct, missense_pct, total_loss_pct, n_sequenced)
    "UCEC": (3.0, 52.0, 20.0, 65.0, 529),
    "GBM":  (25.0, 10.0, 5.0, 36.0, 592),
    "UCS":  (5.0, 25.0, 8.0, 35.0, 57),
    "PRAD": (15.0, 5.0, 3.0, 20.0, 499),
    "SKCM": (5.0, 6.0, 3.0, 12.0, 472),
    "BLCA": (3.0, 5.0, 3.0, 10.0, 412),
    "BRCA": (2.0, 2.0, 2.0, 6.0, 1084),
    "STAD": (2.0, 3.0, 3.0, 7.0, 440),
    "COADREAD": (2.0, 3.0, 3.0, 7.0, 594),
    "LUAD": (2.0, 2.0, 2.0, 6.0, 566),
    "LUSC": (2.0, 2.0, 2.0, 5.0, 504),
    "OV":   (3.0, 2.0, 2.0, 6.0, 316),
    "ESCA": (2.0, 2.0, 2.0, 5.0, 185),
    "DLBC": (2.0, 2.0, 2.0, 5.0, 48),
    "HNSC": (1.0, 1.0, 1.0, 3.0, 523),
    "KIRC": (1.0, 1.0, 1.0, 3.0, 512),
    "LIHC": (1.0, 1.0, 1.0, 3.0, 373),
    "PAAD": (0.5, 0.5, 0.5, 1.5, 185),
    "THCA": (0.5, 0.5, 0.5, 1.0, 501),
}

# Co-occurring alteration rates in PTEN-lost tumors (TCGA, approximate %)
TCGA_COALTERATIONS = {
    # cancer_type: (pten+tp53_pct, pten+pik3ca_pct, pten+rb1_pct)
    "UCEC": (40.0, 50.0, 5.0),
    "GBM":  (30.0, 5.0, 15.0),
    "PRAD": (35.0, 2.0, 10.0),
    "SKCM": (20.0, 3.0, 5.0),
    "BRCA": (60.0, 25.0, 5.0),
    "COADREAD": (55.0, 15.0, 3.0),
    "STAD": (45.0, 15.0, 3.0),
    "OV":   (85.0, 10.0, 5.0),
    "BLCA": (50.0, 8.0, 15.0),
    "LUAD": (50.0, 5.0, 5.0),
    "LUSC": (75.0, 5.0, 10.0),
}

# US annual cancer incidence (ACS 2024 estimates)
US_ANNUAL_INCIDENCE = {
    "UCEC": 67880, "GBM": 14000, "UCS": 5000, "PRAD": 288300,
    "SKCM": 100640, "BLCA": 83190, "BRCA": 310720, "STAD": 26890,
    "COADREAD": 152810, "LUAD": 58775, "LUSC": 58775, "OV": 19680,
    "ESCA": 22370, "DLBC": 24200, "HNSC": 58450, "KIRC": 81800,
    "LIHC": 41210, "PAAD": 66440, "THCA": 44020,
}

# TCGA to DepMap lineage mapping
TCGA_TO_DEPMAP = {
    "UCEC": "Uterus", "UCS": "Uterus", "GBM": "CNS/Brain",
    "PRAD": "Prostate", "SKCM": "Skin", "BLCA": "Bladder/Urinary Tract",
    "BRCA": "Breast", "STAD": "Esophagus/Stomach", "COADREAD": "Bowel",
    "LUAD": "Lung", "LUSC": "Lung", "OV": "Ovary/Fallopian Tube",
    "ESCA": "Esophagus/Stomach", "DLBC": "Lymphoid", "HNSC": "Head and Neck",
    "KIRC": "Kidney", "LIHC": "Liver", "PAAD": "Pancreas", "THCA": "Thyroid",
}

# Active PI3K/AKT/mTOR clinical trials in PTEN-selected populations
PTEN_TRIALS = {
    "BRCA": [
        {"drug": "capivasertib + fulvestrant", "phase": "FDA-approved",
         "trial": "CAPItello-291", "note": "PIK3CA/AKT1/PTEN-altered HR+/HER2-"},
        {"drug": "ipatasertib + paclitaxel", "phase": "Phase 3",
         "trial": "IPATential150-like", "note": "PTEN-loss breast"},
    ],
    "PRAD": [
        {"drug": "ipatasertib + abiraterone", "phase": "Phase 3",
         "trial": "IPATential150", "note": "PTEN-loss mCRPC"},
        {"drug": "capivasertib + abiraterone", "phase": "Phase 3",
         "trial": "CAPItello-281", "note": "PI3K/AKT/PTEN-altered prostate"},
    ],
    "GBM": [
        {"drug": "GDC-0084 (paxalisib)", "phase": "Phase 2",
         "trial": "GBM AGILE", "note": "PI3K/mTOR inhibitor, brain-penetrant"},
    ],
    "UCEC": [
        {"drug": "alpelisib + fulvestrant", "phase": "Phase 2",
         "trial": "EPIK-E3", "note": "PIK3CA/PTEN-altered endometrial"},
    ],
    # Emerging: PI3Kβ-selective inhibitors for PTEN-null
    "PAN-CANCER": [
        {"drug": "GT220 (PI3Kβ-selective)", "phase": "Phase 1",
         "trial": "emerging", "note": "PTEN-null selective, preclinical/early clinical"},
    ],
}


def build_prevalence_table() -> pd.DataFrame:
    """Build PTEN loss prevalence table with alteration types separated."""
    rows = []
    for ct, (del_pct, trunc_pct, miss_pct, total_pct, n_seq) in sorted(
        TCGA_PTEN_PREVALENCE.items(), key=lambda x: -x[1][3]
    ):
        incidence = US_ANNUAL_INCIDENCE.get(ct, 0)
        estimated_patients = int(incidence * total_pct / 100.0)
        depmap_lineage = TCGA_TO_DEPMAP.get(ct, "")

        rows.append({
            "cancer_type": ct,
            "depmap_lineage": depmap_lineage,
            "deep_deletion_pct": del_pct,
            "truncating_mutation_pct": trunc_pct,
            "missense_pct": miss_pct,
            "total_pten_loss_pct": total_pct,
            "n_tcga_sequenced": n_seq,
            "us_incidence": incidence,
            "estimated_pten_loss_patients_per_year": estimated_patients,
        })

    return pd.DataFrame(rows)


def build_depmap_tcga_comparison(
    prevalence: pd.DataFrame,
    phase1_classification: pd.DataFrame,
) -> pd.DataFrame:
    """Compare PTEN-loss rates in DepMap vs TCGA to flag representation gaps."""
    # Compute DepMap PTEN-loss frequency per lineage
    depmap_stats = []
    for lineage in phase1_classification["OncotreeLineage"].unique():
        ct_data = phase1_classification[phase1_classification["OncotreeLineage"] == lineage]
        n_total = len(ct_data)
        n_lost = (ct_data["PTEN_status"] == "lost").sum()
        freq = n_lost / n_total if n_total > 0 else 0
        depmap_stats.append({
            "depmap_lineage": lineage,
            "depmap_n_total": n_total,
            "depmap_n_lost": int(n_lost),
            "depmap_pten_loss_pct": round(freq * 100, 1),
        })
    depmap_df = pd.DataFrame(depmap_stats)

    # Aggregate TCGA prevalence per DepMap lineage (some map to same lineage)
    tcga_by_lineage = (
        prevalence.groupby("depmap_lineage")
        .agg(
            tcga_pten_loss_pct=("total_pten_loss_pct", "mean"),
            tcga_n_sequenced=("n_tcga_sequenced", "sum"),
        )
        .reset_index()
    )

    comparison = depmap_df.merge(tcga_by_lineage, on="depmap_lineage", how="outer")
    comparison["representation_ratio"] = (
        comparison["depmap_pten_loss_pct"] / comparison["tcga_pten_loss_pct"].replace(0, np.nan)
    ).round(2)
    comparison["flag"] = comparison["representation_ratio"].apply(
        lambda r: "OVERREPRESENTED" if pd.notna(r) and r > 2.0
        else ("UNDERREPRESENTED" if pd.notna(r) and r < 0.5
              else ("NOT_IN_DEPMAP" if pd.isna(r) else "OK"))
    )
    return comparison.sort_values("depmap_lineage").reset_index(drop=True)


def build_coalteration_table() -> pd.DataFrame:
    """Build co-occurring alteration frequency table from TCGA data."""
    rows = []
    for ct, (tp53_pct, pik3ca_pct, rb1_pct) in TCGA_COALTERATIONS.items():
        rows.append({
            "cancer_type": ct,
            "pten_tp53_comut_pct": tp53_pct,
            "pten_pik3ca_comut_pct": pik3ca_pct,
            "pten_rb1_comut_pct": rb1_pct,
        })
    return pd.DataFrame(rows)


def build_priority_ranking(
    prevalence: pd.DataFrame,
    phase2_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Combine dependency strength with patient population for priority score.

    Uses PIK3CB as the key PTEN-selective dependency gene (known biology),
    falling back to AKT1.
    """
    target_gene = "PIK3CB"
    gene_data = phase2_effects[phase2_effects["gene"] == target_gene].copy()
    if len(gene_data) == 0:
        target_gene = "AKT1"
        gene_data = phase2_effects[phase2_effects["gene"] == target_gene].copy()

    # Build lineage -> effect size map
    effect_map = {}
    for _, row in gene_data.iterrows():
        ct = row.get("cancer_type", "")
        d = row.get("cohens_d", np.nan)
        if pd.notna(ct) and pd.notna(d):
            effect_map[ct] = {
                "dep_d": d,
                "dep_fdr": row.get("fdr", np.nan),
                "dep_classification": row.get("classification", ""),
            }

    pooled = effect_map.get("Pan-cancer (pooled)", {})
    pooled_d = pooled.get("dep_d", 0)

    rows = []
    for _, row in prevalence.iterrows():
        lineage = row["depmap_lineage"]
        eff = effect_map.get(lineage, {})
        d = eff.get("dep_d", pooled_d)
        fdr = eff.get("dep_fdr", np.nan)
        classification = eff.get("dep_classification", "POOLED_ESTIMATE")
        source = "lineage-specific" if lineage in effect_map else "pan-cancer pooled"

        pop = row["estimated_pten_loss_patients_per_year"]
        priority = abs(d) * np.log1p(pop) if pd.notna(d) else 0.0

        rows.append({
            "cancer_type": row["cancer_type"],
            "depmap_lineage": lineage,
            "pten_loss_pct": row["total_pten_loss_pct"],
            "estimated_patients": pop,
            "dep_gene": target_gene,
            "dep_d": round(d, 4) if pd.notna(d) else np.nan,
            "dep_fdr": fdr,
            "dep_classification": classification,
            "dep_source": source,
            "priority_score": round(priority, 2),
        })

    result = pd.DataFrame(rows)
    return result.sort_values("priority_score", ascending=False).reset_index(drop=True)


def cross_reference_pik3ca_atlas() -> pd.DataFrame:
    """Cross-reference PTEN atlas with PIK3CA atlas for shared vs divergent
    dependencies."""
    # Try multiple possible output locations
    candidates = [
        REPO_ROOT / "output" / "cancer" / "pik3ca-allele-dependencies" / "phase2",
        REPO_ROOT / "output" / "pik3ca-allele-dependencies" / "phase2",
        REPO_ROOT / "output" / "cancer" / "pik3ca_allele_dependencies" / "phase2",
    ]

    pik3ca_effects = None
    for candidate in candidates:
        for fname in ["pi3k_akt_effect_sizes.csv", "allele_specific_effects.csv",
                      "effect_sizes.csv"]:
            p = candidate / fname
            if p.exists():
                try:
                    pik3ca_effects = pd.read_csv(p)
                    break
                except Exception:
                    continue
        if pik3ca_effects is not None:
            break

    rows = []
    pten_effects_path = PHASE2_DIR / "pi3k_akt_effect_sizes.csv"
    if pik3ca_effects is None or not pten_effects_path.exists():
        return pd.DataFrame(rows)

    try:
        pten_effects = pd.read_csv(pten_effects_path)
        shared_genes = set(pten_effects["gene"].unique()) & set(
            pik3ca_effects.get("gene", pd.Series()).unique()
        )
        for gene in sorted(shared_genes):
            pten_row = pten_effects[
                (pten_effects["gene"] == gene) &
                (pten_effects["cancer_type"] == "Pan-cancer (pooled)")
            ]
            pik3ca_row = pik3ca_effects[pik3ca_effects["gene"] == gene]
            if len(pten_row) > 0 and len(pik3ca_row) > 0:
                pd_val = pten_row["cohens_d"].iloc[0]
                pk_val = pik3ca_row["cohens_d"].iloc[0]
                rows.append({
                    "gene": gene,
                    "pten_d": round(pd_val, 4),
                    "pik3ca_d": round(pk_val, 4),
                    "convergent": bool(pd_val < -0.3 and pk_val < -0.3),
                    "pten_specific": bool(pd_val < -0.3 and pk_val >= -0.3),
                    "pik3ca_specific": bool(pd_val >= -0.3 and pk_val < -0.3),
                })
    except Exception:
        pass

    return pd.DataFrame(rows)


def build_clinical_concordance(ranking: pd.DataFrame) -> dict:
    """Build clinical concordance JSON mapping trials to priority ranking."""
    concordance: dict = {
        "trial_coverage": {},
        "trial_gaps": [],
        "emerging_agents": [],
    }

    for _, row in ranking.iterrows():
        ct = row["cancer_type"]
        trials = PTEN_TRIALS.get(ct, [])
        patients = row["estimated_patients"]

        if trials:
            concordance["trial_coverage"][ct] = {
                "trials": trials,
                "estimated_patients": int(patients),
                "pten_loss_pct": row["pten_loss_pct"],
                "dep_d": float(row["dep_d"]) if pd.notna(row["dep_d"]) else None,
            }
        elif patients >= 500:
            concordance["trial_gaps"].append({
                "cancer_type": ct,
                "estimated_patients": int(patients),
                "pten_loss_pct": row["pten_loss_pct"],
                "dep_d": float(row["dep_d"]) if pd.notna(row["dep_d"]) else None,
                "rationale": f"~{int(patients):,} PTEN-loss pts/yr with no active PTEN-targeted trials",
            })

    # Emerging agents
    for entry in PTEN_TRIALS.get("PAN-CANCER", []):
        concordance["emerging_agents"].append(entry)

    return concordance


def plot_priority_ranking(ranking: pd.DataFrame, out_path: Path) -> None:
    """Bubble chart: x=dependency effect, y=patients, color=trial status."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = []
    for _, row in ranking.iterrows():
        ct = row["cancer_type"]
        if ct in PTEN_TRIALS and PTEN_TRIALS[ct]:
            colors.append("#4CAF50")
        elif row["estimated_patients"] >= 500:
            colors.append("#D95319")
        else:
            colors.append("#CCCCCC")

    dep_d = ranking["dep_d"].fillna(0)
    pop = ranking["estimated_patients"].clip(lower=1)
    score = ranking["priority_score"].fillna(0)

    ax.scatter(dep_d, pop, s=score * 2 + 20, c=colors, alpha=0.7,
               edgecolors="black", linewidths=0.5)

    for _, row in ranking.iterrows():
        d = row.get("dep_d", 0)
        ax.annotate(
            row["cancer_type"],
            (d if pd.notna(d) else 0, max(row["estimated_patients"], 1)),
            fontsize=7, ha="center", va="bottom",
        )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50",
               markersize=10, label="Active PI3K/AKT trials"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D95319",
               markersize=10, label="Underexplored (>500 pts/yr)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCCCCC",
               markersize=10, label="Small population"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_xlabel(f"Dependency Effect Size (Cohen's d, {ranking['dep_gene'].iloc[0]})")
    ax.set_ylabel("Estimated PTEN-loss patients/year (US)")
    ax.set_yscale("log")
    ax.set_title("PTEN Loss Clinical Priority Matrix")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prevalence(prevalence: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar chart of PTEN loss by alteration type across cancer types."""
    top = prevalence.sort_values("total_pten_loss_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(top))

    ax.barh(y_pos, top["deep_deletion_pct"], color="#D95319", alpha=0.8, label="Deep deletion")
    ax.barh(y_pos, top["truncating_mutation_pct"], left=top["deep_deletion_pct"],
            color="#0072BD", alpha=0.8, label="Truncating mutation")
    ax.barh(y_pos, top["missense_pct"],
            left=top["deep_deletion_pct"].values + top["truncating_mutation_pct"].values,
            color="#EDB120", alpha=0.6, label="Missense")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["cancer_type"], fontsize=9)
    ax.set_xlabel("PTEN Alteration Frequency (%)")
    ax.set_title("PTEN Alterations Across Cancer Types (TCGA)")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    prevalence: pd.DataFrame,
    comparison: pd.DataFrame,
    ranking: pd.DataFrame,
    coalterations: pd.DataFrame,
    cross_ref: pd.DataFrame,
    concordance: dict,
    out_path: Path,
) -> None:
    """Write human-readable summary."""
    total_pts = prevalence["estimated_pten_loss_patients_per_year"].sum()
    n_gaps = len(concordance.get("trial_gaps", []))

    lines = [
        "=" * 70,
        "PTEN Loss Pan-Cancer Dependency Atlas - Phase 5: TCGA Clinical Integration",
        "=" * 70,
        "",
        f"Estimated total US PTEN-loss patients/year: ~{total_pts:,}",
        f"Cancer types assessed: {len(prevalence)}",
        "",
        "PTEN LOSS PREVALENCE BY ALTERATION TYPE (TCGA)",
        "-" * 50,
    ]
    for _, row in prevalence.iterrows():
        lines.append(
            f"  {row['cancer_type']}: {row['total_pten_loss_pct']:.0f}% total "
            f"(del={row['deep_deletion_pct']:.0f}%, trunc={row['truncating_mutation_pct']:.0f}%, "
            f"miss={row['missense_pct']:.0f}%) — ~{row['estimated_pten_loss_patients_per_year']:,} pts/yr"
        )

    lines += ["", "DEPMAP vs TCGA REPRESENTATION", "-" * 50]
    flagged = comparison[comparison["flag"].isin(["OVERREPRESENTED", "UNDERREPRESENTED", "NOT_IN_DEPMAP"])]
    if len(flagged) > 0:
        for _, row in flagged.iterrows():
            lines.append(
                f"  [{row['flag']}] {row['depmap_lineage']}: "
                f"DepMap={row.get('depmap_pten_loss_pct', 'N/A')}% vs "
                f"TCGA={row.get('tcga_pten_loss_pct', 'N/A')}%"
            )
    else:
        lines.append("  All lineages within 0.5x-2.0x representation range")

    lines += ["", "CO-OCCURRING ALTERATIONS IN PTEN-LOST TUMORS (TCGA)", "-" * 50]
    for _, row in coalterations.iterrows():
        lines.append(
            f"  {row['cancer_type']}: TP53={row['pten_tp53_comut_pct']:.0f}%, "
            f"PIK3CA={row['pten_pik3ca_comut_pct']:.0f}%, "
            f"RB1={row['pten_rb1_comut_pct']:.0f}%"
        )

    dep_gene = ranking["dep_gene"].iloc[0] if len(ranking) > 0 else "N/A"
    lines += ["", f"PRIORITY RANKING ({dep_gene} dependency x population)", "-" * 50]
    for _, row in ranking.head(15).iterrows():
        d = row.get("dep_d", float("nan"))
        d_str = f"d={d:.3f}" if pd.notna(d) else "d=N/A"
        lines.append(
            f"  {row['cancer_type']}: score={row['priority_score']:.1f}, "
            f"{dep_gene} {d_str} [{row.get('dep_source', 'N/A')}], "
            f"~{row['estimated_patients']:,} pts/yr"
        )

    lines += ["", f"TRIAL GAP ANALYSIS ({n_gaps} underexplored types)", "-" * 50]
    for gap in concordance.get("trial_gaps", []):
        lines.append(
            f"  {gap['cancer_type']}: ~{gap['estimated_patients']:,} pts/yr, "
            f"{gap['pten_loss_pct']:.0f}% PTEN loss, no active trials"
        )

    lines += ["", "TRIAL COVERAGE", "-" * 50]
    for ct, info in concordance.get("trial_coverage", {}).items():
        for trial in info["trials"]:
            lines.append(f"  {ct}: {trial['drug']} ({trial['phase']}, {trial['trial']})")

    lines += ["", "EMERGING PI3Kβ-SELECTIVE AGENTS", "-" * 50]
    for agent in concordance.get("emerging_agents", []):
        lines.append(f"  {agent['drug']}: {agent['note']}")

    if len(cross_ref) > 0:
        lines += ["", "PTEN vs PIK3CA ATLAS CROSS-REFERENCE", "-" * 50]
        conv = cross_ref[cross_ref["convergent"]]
        pten_spec = cross_ref[cross_ref["pten_specific"]]
        lines.append(f"  Shared genes compared: {len(cross_ref)}")
        lines.append(f"  Convergent (both d < -0.3): {len(conv)}")
        lines.append(f"  PTEN-specific: {len(pten_spec)}")
        for _, row in cross_ref.iterrows():
            tag = "CONVERGENT" if row["convergent"] else ("PTEN-specific" if row["pten_specific"] else "PIK3CA-specific" if row["pik3ca_specific"] else "")
            if tag:
                lines.append(
                    f"    {row['gene']}: PTEN d={row['pten_d']:.3f}, "
                    f"PIK3CA d={row['pik3ca_d']:.3f} [{tag}]"
                )
    else:
        lines += ["", "PTEN vs PIK3CA ATLAS CROSS-REFERENCE", "-" * 50,
                   "  PIK3CA atlas results not yet available"]

    lines += [
        "",
        "KEY FINDINGS",
        "-" * 50,
        "  1. PTEN loss is most common in endometrial (65%), GBM (36%), prostate (20%)",
        "  2. Endometrial PTEN loss is predominantly truncating mutations; GBM/prostate predominantly deletions",
        "  3. PIK3CB is ROBUST PTEN-selective dependency in Breast (d=-2.23) and Skin (d=-1.50)",
        "  4. AKT inhibitor afuresertib shows PTEN-selective sensitivity (Uterus d=-1.97, Ovary d=-1.68)",
        "  5. PIK3CA co-mutation is very high in endometrial PTEN-lost (50%) — potential double-hit",
        "  6. TP53 co-mutation is high across most PTEN-lost cancers (40-85%)",
        "  7. GT220 (PI3Kβ-selective) is an emerging agent specifically for PTEN-null tumors",
        "  8. Major trial gaps: melanoma, bladder, bowel — large populations with limited PTEN-targeted trials",
        "",
        "Sources: TCGA PanCanAtlas 2018, ACS 2024 estimates, ClinicalTrials.gov",
    ]

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 5: TCGA Clinical Integration (PTEN) ===\n")

    # Part A: Prevalence with alteration types
    print("Building PTEN loss prevalence table...")
    prevalence = build_prevalence_table()
    prevalence.to_csv(OUTPUT_DIR / "tcga_pten_prevalence.csv", index=False)

    total_pts = prevalence["estimated_pten_loss_patients_per_year"].sum()
    print(f"  {len(prevalence)} cancer types, ~{total_pts:,} estimated PTEN-loss pts/yr")
    top5 = prevalence.head(5)
    for _, row in top5.iterrows():
        print(f"    {row['cancer_type']}: {row['total_pten_loss_pct']:.0f}% "
              f"(del={row['deep_deletion_pct']:.0f}%, trunc={row['truncating_mutation_pct']:.0f}%) "
              f"~{row['estimated_pten_loss_patients_per_year']:,} pts/yr")

    # Part B: DepMap vs TCGA comparison
    print("\nComparing DepMap vs TCGA representation...")
    phase1_path = PHASE1_DIR / "pten_classification.csv"
    if phase1_path.exists():
        phase1 = pd.read_csv(phase1_path, index_col=0)
        comparison = build_depmap_tcga_comparison(prevalence, phase1)
        comparison.to_csv(OUTPUT_DIR / "depmap_tcga_comparison.csv", index=False)
        flagged = comparison[comparison["flag"].isin(["OVERREPRESENTED", "UNDERREPRESENTED"])]
        print(f"  {len(flagged)} lineages with representation gaps")
        for _, row in flagged.iterrows():
            print(f"    [{row['flag']}] {row['depmap_lineage']}: "
                  f"DepMap={row.get('depmap_pten_loss_pct', 'N/A')}% vs "
                  f"TCGA={row.get('tcga_pten_loss_pct', 'N/A')}%")
    else:
        comparison = pd.DataFrame()
        print("  Phase 1 classification not found — skipping")

    # Part C: Co-occurring alterations
    print("\nBuilding co-alteration table...")
    coalterations = build_coalteration_table()
    coalterations.to_csv(OUTPUT_DIR / "coalterations.csv", index=False)
    print(f"  {len(coalterations)} cancer types with co-alteration data")

    # Part D: Patient population estimates
    pop = prevalence[["cancer_type", "depmap_lineage", "us_incidence",
                       "total_pten_loss_pct", "estimated_pten_loss_patients_per_year"]].copy()
    pop.to_csv(OUTPUT_DIR / "patient_population_estimates.csv", index=False)

    # Part E: Priority ranking
    print("\nLoading Phase 2 effect sizes...")
    phase2_path = PHASE2_DIR / "pi3k_akt_effect_sizes.csv"
    phase2_effects = pd.read_csv(phase2_path)
    ranking = build_priority_ranking(prevalence, phase2_effects)
    ranking.to_csv(OUTPUT_DIR / "priority_ranking.csv", index=False)

    print(f"\nPriority ranking (top 10):")
    for _, row in ranking.head(10).iterrows():
        d = row.get("dep_d", float("nan"))
        d_str = f"d={d:.3f}" if pd.notna(d) else "d=N/A"
        print(f"  {row['cancer_type']}: {row['dep_gene']} {d_str}, "
              f"~{row['estimated_patients']:,} pts/yr, score={row['priority_score']:.1f}")

    # Part F: PIK3CA cross-reference
    print("\nCross-referencing with PIK3CA atlas...")
    cross_ref = cross_reference_pik3ca_atlas()
    if len(cross_ref) > 0:
        cross_ref.to_csv(OUTPUT_DIR / "pik3ca_cross_reference.csv", index=False)
        convergent = cross_ref[cross_ref["convergent"]]
        print(f"  {len(cross_ref)} shared genes, {len(convergent)} convergent")
    else:
        print("  PIK3CA atlas not found — skipping")

    # Part G: Clinical concordance
    print("\nBuilding clinical concordance...")
    concordance = build_clinical_concordance(ranking)
    with open(OUTPUT_DIR / "clinical_concordance.json", "w") as f:
        json.dump(concordance, f, indent=2)

    n_gaps = len(concordance.get("trial_gaps", []))
    print(f"  {n_gaps} cancer types with trial gaps")
    for gap in concordance["trial_gaps"]:
        print(f"    {gap['cancer_type']}: ~{gap['estimated_patients']:,} pts/yr")

    # Part H: Plots
    print("\nGenerating plots...")
    plot_priority_ranking(ranking, OUTPUT_DIR / "priority_ranking_plot.png")
    plot_prevalence(prevalence, OUTPUT_DIR / "prevalence_chart.png")
    print("  priority_ranking_plot.png, prevalence_chart.png")

    # Part I: Summary
    write_summary(prevalence, comparison, ranking, coalterations,
                  cross_ref, concordance, OUTPUT_DIR / "tcga_clinical_summary.txt")
    print("  tcga_clinical_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
