"""Phase 1: Classify BRCA1/2 status across all DepMap 25Q3 cell lines.

Classifies cell lines by BRCA1 and BRCA2 status SEPARATELY:
  - Deficient: LOF mutation (LikelyLoF or VepImpact HIGH) OR homozygous deletion
  - Proficient: No damaging mutations, no deep deletions
  - Excluded: missense-only VUS (neither deficient nor proficient)

Annotates 53BP1/SHLD complex status (PARPi resistance biomarker), pre-replication
complex (pre-RC) gene status, and co-occurring mutations in key pathway genes.

Usage:
    uv run python -m brca_pancancer_sl_atlas.01_brca_classifier
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import (
    load_depmap_matrix,
    load_depmap_model_metadata,
)

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase1"

# CN threshold for homozygous deletion (log2 scale, 0 = diploid)
HOMDEL_CN_THRESHOLD = -1.0

# Expression threshold for "expression loss" (log2 TPM+1 scale)
EXPRESSION_LOSS_THRESHOLD = 1.0  # ~1 TPM

# Minimum samples for powered analysis
MIN_DEFICIENT_PRIMARY = 10
MIN_DEFICIENT_EXPLORATORY = 5
MIN_PROFICIENT = 10

# 53BP1/SHLD complex components
SHLD_GENES = ["TP53BP1", "RIF1", "SHLD1", "SHLD2", "SHLD3", "MAD2L2"]
# Alias mapping: gene aliases used in DepMap → canonical name
SHLD_ALIASES = {
    "FAM35A": "SHLD1",
    "RINN2": "SHLD1",
    "FAM35B": "SHLD2",
    "CTC-534A2.2": "SHLD3",
    "REV7": "MAD2L2",
}

# Pre-replication complex genes
PRE_RC_GENES = [
    "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
    "ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
    "CDC6", "CDT1",
]

# Co-mutation genes
COMUTATION_GENES = [
    "TP53", "PTEN", "PALB2", "RAD51C", "RAD51D",
    "ATM", "ATR", "CHEK2", "KRAS", "PIK3CA",
]


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap Model.csv for all cell lines with cancer type annotation."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def load_crispr_lines(depmap_dir: Path) -> set[str]:
    """Return set of ModelIDs that have CRISPR data."""
    crispr = pd.read_csv(depmap_dir / "CRISPRGeneEffect.csv", usecols=[0])
    return set(crispr.iloc[:, 0])


def load_mutations(depmap_dir: Path) -> pd.DataFrame:
    """Load all somatic mutations (filtered to default entries)."""
    cols = [
        "ModelID", "HugoSymbol", "VariantType", "VariantInfo",
        "ProteinChange", "VepImpact", "LikelyLoF", "IsDefaultEntryForModel",
    ]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    # Keep default entries only
    if "IsDefaultEntryForModel" in mutations.columns:
        mutations = mutations[mutations["IsDefaultEntryForModel"] == "Yes"].copy()
        mutations = mutations.drop(columns=["IsDefaultEntryForModel"])
    return mutations


def classify_gene_status(
    mutations: pd.DataFrame,
    cn_df: pd.DataFrame,
    gene: str,
) -> pd.DataFrame:
    """Classify a single gene's status across all cell lines.

    Returns DataFrame indexed by ModelID with columns:
        has_lof, has_homdel, status (deficient/proficient/excluded_VUS),
        lof_mutations, protein_changes
    """
    # --- LOF mutations ---
    gene_muts = mutations[mutations["HugoSymbol"] == gene].copy()
    gene_lof = gene_muts[
        (gene_muts["LikelyLoF"] == True) | (gene_muts["VepImpact"] == "HIGH")
    ]

    lof_per_line = (
        gene_lof.groupby("ModelID")
        .agg(
            n_lof=("VariantInfo", "count"),
            lof_mutations=("VariantInfo", lambda x: ";".join(
                sorted(set(str(v) for v in x if pd.notna(v)))
            )),
            protein_changes=("ProteinChange", lambda x: ";".join(
                sorted(set(str(v) for v in x if pd.notna(v)))
            )),
        )
        .reset_index()
    )
    lof_ids = set(lof_per_line["ModelID"])

    # --- Missense-only VUS ---
    all_mutated = set(gene_muts["ModelID"].unique())
    vus_ids = all_mutated - lof_ids  # lines with mutations but no LOF

    # --- Homozygous deletion ---
    if gene in cn_df.columns:
        gene_cn = cn_df[gene]
        homdel_ids = set(gene_cn[gene_cn < HOMDEL_CN_THRESHOLD].index)
    else:
        homdel_ids = set()

    return lof_per_line, lof_ids, vus_ids, homdel_ids


def annotate_shld_complex(
    mutations: pd.DataFrame,
    cn_df: pd.DataFrame,
    brca_deficient_ids: set[str],
) -> pd.DataFrame:
    """Annotate 53BP1/SHLD complex status for BRCA-deficient lines.

    LOF or homdel in ANY component = SHLD-lost.
    """
    # Build full gene list including aliases
    all_shld_names = set(SHLD_GENES) | set(SHLD_ALIASES.keys())

    # Find LOF mutations in SHLD genes
    shld_muts = mutations[
        (mutations["HugoSymbol"].isin(all_shld_names))
        & ((mutations["LikelyLoF"] == True) | (mutations["VepImpact"] == "HIGH"))
    ].copy()

    # Map aliases to canonical names
    shld_muts["canonical"] = shld_muts["HugoSymbol"].map(
        lambda g: SHLD_ALIASES.get(g, g)
    )

    lof_per_line = (
        shld_muts.groupby("ModelID")["canonical"]
        .apply(lambda x: sorted(set(x)))
        .rename("shld_lof_genes")
        .reset_index()
    )
    lof_lines = set(lof_per_line["ModelID"])

    # Check homdel in SHLD genes
    homdel_lines: dict[str, list[str]] = {}
    for gene in SHLD_GENES:
        # Check canonical name and aliases
        names_to_check = [gene] + [a for a, c in SHLD_ALIASES.items() if c == gene]
        for name in names_to_check:
            if name in cn_df.columns:
                homdel = cn_df[cn_df[name] < HOMDEL_CN_THRESHOLD].index
                for mid in homdel:
                    homdel_lines.setdefault(mid, []).append(gene)
                break  # found the gene in CN data

    homdel_df = pd.DataFrame([
        {"ModelID": mid, "shld_homdel_genes": sorted(set(genes))}
        for mid, genes in homdel_lines.items()
    ])

    # Combine LOF and homdel
    all_model_ids = list(brca_deficient_ids)
    result = pd.DataFrame({"ModelID": all_model_ids})

    if len(lof_per_line) > 0:
        result = result.merge(lof_per_line, on="ModelID", how="left")
    else:
        result["shld_lof_genes"] = None

    if len(homdel_df) > 0:
        result = result.merge(homdel_df, on="ModelID", how="left")
    else:
        result["shld_homdel_genes"] = None

    # Determine SHLD status
    def get_shld_status(row):
        lost_genes = set()
        if isinstance(row.get("shld_lof_genes"), list):
            lost_genes.update(row["shld_lof_genes"])
        if isinstance(row.get("shld_homdel_genes"), list):
            lost_genes.update(row["shld_homdel_genes"])
        if lost_genes:
            return "SHLD-lost", ";".join(sorted(lost_genes))
        return "SHLD-intact", ""

    statuses = result.apply(get_shld_status, axis=1, result_type="expand")
    result["shld_complex_status"] = statuses[0]
    result["shld_lost_genes"] = statuses[1]

    return result[["ModelID", "shld_complex_status", "shld_lost_genes"]]


def annotate_pre_rc(
    mutations: pd.DataFrame,
    cn_df: pd.DataFrame,
    expr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate pre-replication complex gene status.

    LOF mutation, homdel, or very low expression in any component
    flags the line as pre-RC-compromised.
    """
    # LOF mutations in pre-RC genes
    pre_rc_lof = mutations[
        (mutations["HugoSymbol"].isin(PRE_RC_GENES))
        & ((mutations["LikelyLoF"] == True) | (mutations["VepImpact"] == "HIGH"))
    ]
    lof_per_line = (
        pre_rc_lof.groupby("ModelID")["HugoSymbol"]
        .apply(lambda x: sorted(set(x)))
        .rename("pre_rc_lof_genes")
        .reset_index()
    )
    lof_lines = dict(zip(lof_per_line["ModelID"], lof_per_line["pre_rc_lof_genes"]))

    # Homdel in pre-RC genes
    homdel_lines: dict[str, list[str]] = {}
    for gene in PRE_RC_GENES:
        if gene in cn_df.columns:
            homdel = cn_df[cn_df[gene] < HOMDEL_CN_THRESHOLD].index
            for mid in homdel:
                homdel_lines.setdefault(mid, []).append(gene)

    # Expression loss in pre-RC genes
    expr_loss_lines: dict[str, list[str]] = {}
    for gene in PRE_RC_GENES:
        if gene in expr_df.columns:
            low_expr = expr_df[expr_df[gene] < EXPRESSION_LOSS_THRESHOLD].index
            for mid in low_expr:
                expr_loss_lines.setdefault(mid, []).append(gene)

    # Combine: any line with LOF, homdel, or expression loss in any pre-RC gene
    all_model_ids = set(lof_lines.keys()) | set(homdel_lines.keys()) | set(expr_loss_lines.keys())

    rows = []
    for mid in all_model_ids:
        affected = set()
        if mid in lof_lines:
            affected.update(lof_lines[mid])
        if mid in homdel_lines:
            affected.update(homdel_lines[mid])
        if mid in expr_loss_lines:
            affected.update(expr_loss_lines[mid])
        rows.append({
            "ModelID": mid,
            "pre_rc_status": "compromised",
            "pre_rc_affected_genes": ";".join(sorted(affected)),
        })

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["ModelID", "pre_rc_status", "pre_rc_affected_genes"])


def load_comutations(mutations: pd.DataFrame) -> pd.DataFrame:
    """Identify co-occurring LOF mutations in key genes per cell line."""
    comut = mutations[
        (mutations["HugoSymbol"].isin(COMUTATION_GENES))
        & ((mutations["LikelyLoF"] == True) | (mutations["VepImpact"] == "HIGH"))
    ].copy()

    per_line = (
        comut.groupby("ModelID")["HugoSymbol"]
        .apply(lambda x: ";".join(sorted(set(x))))
        .rename("co_mutations")
        .reset_index()
    )
    return per_line


def build_cancer_type_summary(df: pd.DataFrame, crispr_lines: set[str]) -> pd.DataFrame:
    """Summarize BRCA status per cancer type with qualification assessment."""
    # Only include lines with definitive status (not excluded VUS)
    analyzable = df[
        (df["brca1_status"].isin(["deficient", "proficient"]))
        | (df["brca2_status"].isin(["deficient", "proficient"]))
    ]

    summary_rows = []
    for lineage, group in analyzable.groupby("OncotreeLineage"):
        has_crispr = group.index.isin(crispr_lines)

        n_brca1_def = int((group["brca1_status"] == "deficient").sum())
        n_brca2_def = int((group["brca2_status"] == "deficient").sum())
        n_any_def = int((group["brca_combined_status"] == "deficient").sum())
        n_proficient = int((group["brca_combined_status"] == "proficient").sum())

        # CRISPR-available counts
        n_brca1_def_crispr = int(
            (has_crispr & (group["brca1_status"] == "deficient")).sum()
        )
        n_brca2_def_crispr = int(
            (has_crispr & (group["brca2_status"] == "deficient")).sum()
        )
        n_any_def_crispr = int(
            (has_crispr & (group["brca_combined_status"] == "deficient")).sum()
        )
        n_prof_crispr = int(
            (has_crispr & (group["brca_combined_status"] == "proficient")).sum()
        )

        n_total = len(group)
        qualifies_primary = (
            n_any_def_crispr >= MIN_DEFICIENT_PRIMARY
            and n_prof_crispr >= MIN_PROFICIENT
        )
        qualifies_exploratory = (
            n_any_def_crispr >= MIN_DEFICIENT_EXPLORATORY
            and n_prof_crispr >= MIN_PROFICIENT
        )

        # SHLD status among BRCA-deficient lines
        brca_def = group[group["brca_combined_status"] == "deficient"]
        n_shld_lost = int(
            (brca_def["shld_complex_status"] == "SHLD-lost").sum()
        ) if "shld_complex_status" in brca_def.columns else 0

        summary_rows.append({
            "cancer_type": lineage,
            "N_brca1_deficient": n_brca1_def,
            "N_brca2_deficient": n_brca2_def,
            "N_brca_any_deficient": n_any_def,
            "N_proficient": n_proficient,
            "N_total": n_total,
            "N_brca1_def_crispr": n_brca1_def_crispr,
            "N_brca2_def_crispr": n_brca2_def_crispr,
            "N_any_def_crispr": n_any_def_crispr,
            "N_prof_crispr": n_prof_crispr,
            "N_shld_lost": n_shld_lost,
            "qualifies_primary": qualifies_primary,
            "qualifies_exploratory": qualifies_exploratory,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("N_brca_any_deficient", ascending=False).reset_index(drop=True)
    return summary


def plot_classification_summary(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    """Generate bar plots of BRCA classification by cancer type."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: cancer types with BRCA-deficient lines (qualifying + exploratory)
    show = summary[summary["N_brca_any_deficient"] > 0].sort_values(
        "N_brca_any_deficient", ascending=True
    ).tail(20)

    if len(show) > 0:
        ax = axes[0]
        y = range(len(show))
        ax.barh(y, show["N_brca1_deficient"], color="#E53935", alpha=0.8, label="BRCA1-def")
        ax.barh(
            y, show["N_brca2_deficient"],
            left=show["N_brca1_deficient"],
            color="#1E88E5", alpha=0.8, label="BRCA2-def",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(show["cancer_type"], fontsize=8)
        ax.set_xlabel("Number of cell lines")
        ax.set_title("BRCA-Deficient Lines by Cancer Type")
        ax.legend(fontsize=8)

        # Mark qualifying types
        for i, (_, row) in enumerate(show.iterrows()):
            marker = ""
            if row["qualifies_primary"]:
                marker = " **"
            elif row["qualifies_exploratory"]:
                marker = " *"
            if marker:
                total = row["N_brca1_deficient"] + row["N_brca2_deficient"]
                ax.text(total + 0.3, i, marker, va="center", fontsize=9, fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "No BRCA-deficient lines found", ha="center", va="center",
                     transform=axes[0].transAxes)

    # Right: BRCA1 vs BRCA2 classification breakdown
    ax = axes[1]
    brca_def = df[df["brca_combined_status"] == "deficient"]
    categories = {
        "BRCA1-only": ((brca_def["brca1_status"] == "deficient") & (brca_def["brca2_status"] != "deficient")).sum(),
        "BRCA2-only": ((brca_def["brca2_status"] == "deficient") & (brca_def["brca1_status"] != "deficient")).sum(),
        "Both": ((brca_def["brca1_status"] == "deficient") & (brca_def["brca2_status"] == "deficient")).sum(),
    }
    labels = [f"{k}\n(n={v})" for k, v in categories.items() if v > 0]
    values = [v for v in categories.values() if v > 0]
    colors = ["#E53935", "#1E88E5", "#7B1FA2"][:len(values)]

    if values:
        ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("BRCA1 vs BRCA2 Deficiency")
    else:
        ax.text(0.5, 0.5, "No deficient lines", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(output_dir / "classification_summary.png", dpi=150)
    plt.close(fig)


def plot_comutation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate co-mutation heatmap for BRCA-deficient lines."""
    brca_def = df[df["brca_combined_status"] == "deficient"].copy()
    if len(brca_def) == 0:
        return

    # Build binary matrix for co-mutation genes
    comut_matrix = pd.DataFrame(index=brca_def.index)
    for gene in COMUTATION_GENES:
        comut_matrix[gene] = brca_def["co_mutations"].str.contains(gene, na=False).astype(int)

    # Add BRCA status columns
    comut_matrix["BRCA1-def"] = (brca_def["brca1_status"] == "deficient").astype(int)
    comut_matrix["BRCA2-def"] = (brca_def["brca2_status"] == "deficient").astype(int)

    # Sort by cancer type for visual grouping
    if "OncotreeLineage" in brca_def.columns:
        sort_order = brca_def["OncotreeLineage"].fillna("Unknown")
        comut_matrix = comut_matrix.loc[sort_order.sort_values().index]

    if len(comut_matrix) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(comut_matrix) * 0.15)))
    sns.heatmap(
        comut_matrix.T,
        cmap="YlOrRd",
        cbar_kws={"label": "Mutated"},
        ax=ax,
        yticklabels=True,
        xticklabels=False,
        linewidths=0.1,
    )
    ax.set_title(f"Co-mutation Landscape in BRCA-Deficient Lines (n={len(brca_def)})")
    ax.set_xlabel(f"Cell lines (n={len(brca_def)})")

    plt.tight_layout()
    fig.savefig(output_dir / "brca_comutation_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: BRCA1/2 Pan-Cancer Cell Line Classifier ===\n")

    # --- Step 1: Load all cell lines ---
    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    # --- Step 2: Load CRISPR line set ---
    print("Loading CRISPR line IDs...")
    crispr_lines = load_crispr_lines(DEPMAP_DIR)
    print(f"  {len(crispr_lines)} cell lines with CRISPR data")

    # --- Step 3: Load all mutations ---
    print("Loading somatic mutations...")
    mutations = load_mutations(DEPMAP_DIR)
    print(f"  {len(mutations)} total mutations loaded")

    # --- Step 4: Load copy number data ---
    print("Loading copy number data...")
    cn_df = load_depmap_matrix(DEPMAP_DIR / "PortalOmicsCNGeneLog2.csv")
    print(f"  {cn_df.shape[0]} cell lines, {cn_df.shape[1]} genes")

    # --- Step 5: Load expression data ---
    print("Loading expression data...")
    expr_df = load_depmap_matrix(
        DEPMAP_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    print(f"  {expr_df.shape[0]} cell lines, {expr_df.shape[1]} genes")

    # --- Step 6: Classify BRCA1 status ---
    print("\nClassifying BRCA1 status...")
    brca1_lof, brca1_lof_ids, brca1_vus_ids, brca1_homdel_ids = classify_gene_status(
        mutations, cn_df, "BRCA1"
    )
    brca1_def_ids = brca1_lof_ids | brca1_homdel_ids
    print(f"  BRCA1 LOF mutations: {len(brca1_lof_ids)} lines")
    print(f"  BRCA1 homdel: {len(brca1_homdel_ids)} lines")
    print(f"  BRCA1 deficient (LOF or homdel): {len(brca1_def_ids)} lines")
    print(f"  BRCA1 missense-only VUS (excluded): {len(brca1_vus_ids)} lines")

    # --- Step 7: Classify BRCA2 status ---
    print("\nClassifying BRCA2 status...")
    brca2_lof, brca2_lof_ids, brca2_vus_ids, brca2_homdel_ids = classify_gene_status(
        mutations, cn_df, "BRCA2"
    )
    brca2_def_ids = brca2_lof_ids | brca2_homdel_ids
    print(f"  BRCA2 LOF mutations: {len(brca2_lof_ids)} lines")
    print(f"  BRCA2 homdel: {len(brca2_homdel_ids)} lines")
    print(f"  BRCA2 deficient (LOF or homdel): {len(brca2_def_ids)} lines")
    print(f"  BRCA2 missense-only VUS (excluded): {len(brca2_vus_ids)} lines")

    # --- Step 8: Build classification columns ---
    print("\nBuilding classification table...")

    # BRCA1 status
    df["has_lof_brca1"] = df.index.isin(brca1_lof_ids)
    df["has_homdel_brca1"] = df.index.isin(brca1_homdel_ids)
    df["brca1_status"] = "proficient"
    df.loc[df.index.isin(brca1_def_ids), "brca1_status"] = "deficient"
    df.loc[df.index.isin(brca1_vus_ids) & ~df.index.isin(brca1_def_ids), "brca1_status"] = "excluded_VUS"

    # BRCA2 status
    df["has_lof_brca2"] = df.index.isin(brca2_lof_ids)
    df["has_homdel_brca2"] = df.index.isin(brca2_homdel_ids)
    df["brca2_status"] = "proficient"
    df.loc[df.index.isin(brca2_def_ids), "brca2_status"] = "deficient"
    df.loc[df.index.isin(brca2_vus_ids) & ~df.index.isin(brca2_def_ids), "brca2_status"] = "excluded_VUS"

    # Combined status: deficient if EITHER is deficient
    any_def = brca1_def_ids | brca2_def_ids
    any_vus = (brca1_vus_ids | brca2_vus_ids) - any_def
    df["brca_combined_status"] = "proficient"
    df.loc[df.index.isin(any_def), "brca_combined_status"] = "deficient"
    df.loc[df.index.isin(any_vus) & ~df.index.isin(any_def), "brca_combined_status"] = "excluded_VUS"

    # Merge LOF mutation details
    if len(brca1_lof) > 0:
        brca1_details = brca1_lof.set_index("ModelID")[["lof_mutations", "protein_changes"]].rename(
            columns={"lof_mutations": "brca1_lof_mutations", "protein_changes": "brca1_protein_changes"}
        )
        df = df.join(brca1_details, how="left")
    else:
        df["brca1_lof_mutations"] = ""
        df["brca1_protein_changes"] = ""

    if len(brca2_lof) > 0:
        brca2_details = brca2_lof.set_index("ModelID")[["lof_mutations", "protein_changes"]].rename(
            columns={"lof_mutations": "brca2_lof_mutations", "protein_changes": "brca2_protein_changes"}
        )
        df = df.join(brca2_details, how="left")
    else:
        df["brca2_lof_mutations"] = ""
        df["brca2_protein_changes"] = ""

    n_def = (df["brca_combined_status"] == "deficient").sum()
    n_prof = (df["brca_combined_status"] == "proficient").sum()
    n_excl = (df["brca_combined_status"] == "excluded_VUS").sum()
    print(f"  {n_def} BRCA-deficient (any), {n_prof} proficient, {n_excl} excluded VUS")

    n_b1_only = ((df["brca1_status"] == "deficient") & (df["brca2_status"] != "deficient")).sum()
    n_b2_only = ((df["brca2_status"] == "deficient") & (df["brca1_status"] != "deficient")).sum()
    n_both = ((df["brca1_status"] == "deficient") & (df["brca2_status"] == "deficient")).sum()
    print(f"  BRCA1-only: {n_b1_only}, BRCA2-only: {n_b2_only}, both: {n_both}")

    # --- Step 9: 53BP1/SHLD complex annotation ---
    print("\nAnnotating 53BP1/SHLD complex status...")
    shld_df = annotate_shld_complex(mutations, cn_df, any_def)
    df = df.join(shld_df.set_index("ModelID"), how="left")
    df["shld_complex_status"] = df["shld_complex_status"].fillna("")
    df["shld_lost_genes"] = df["shld_lost_genes"].fillna("")

    brca_def_lines = df[df["brca_combined_status"] == "deficient"]
    n_shld_lost = (brca_def_lines["shld_complex_status"] == "SHLD-lost").sum()
    n_shld_intact = (brca_def_lines["shld_complex_status"] == "SHLD-intact").sum()
    print(f"  Among {len(brca_def_lines)} BRCA-deficient lines:")
    print(f"    SHLD-intact: {n_shld_intact}, SHLD-lost: {n_shld_lost}")

    # --- Step 10: Pre-RC annotation ---
    print("\nAnnotating pre-replication complex status...")
    pre_rc_df = annotate_pre_rc(mutations, cn_df, expr_df)
    if len(pre_rc_df) > 0:
        df = df.join(pre_rc_df.set_index("ModelID"), how="left")
    else:
        df["pre_rc_status"] = pd.NA
        df["pre_rc_affected_genes"] = ""
    df["pre_rc_status"] = df["pre_rc_status"].fillna("intact")
    df["pre_rc_affected_genes"] = df["pre_rc_affected_genes"].fillna("")

    n_pre_rc = (df["pre_rc_status"] == "compromised").sum()
    n_pre_rc_brca = (brca_def_lines.index.isin(
        df[df["pre_rc_status"] == "compromised"].index
    )).sum()
    print(f"  {n_pre_rc} total lines with pre-RC compromise")
    print(f"  {n_pre_rc_brca} among BRCA-deficient lines")

    # --- Step 11: Co-mutation annotation ---
    print("\nAnnotating co-mutations (TP53, PTEN, PALB2, RAD51C/D, ATM, ATR, CHEK2, KRAS, PIK3CA)...")
    comut_df = load_comutations(mutations)
    if len(comut_df) > 0:
        df = df.join(comut_df.set_index("ModelID"), how="left")
    else:
        df["co_mutations"] = ""
    df["co_mutations"] = df["co_mutations"].fillna("")

    n_comut = (df["co_mutations"] != "").sum()
    print(f"  {n_comut} cell lines with co-occurring LOF/HIGH-impact mutations in key genes")

    # TP53 co-mutation frequency in BRCA-deficient
    brca_def_lines = df[df["brca_combined_status"] == "deficient"]
    n_tp53_comut = brca_def_lines["co_mutations"].str.contains("TP53", na=False).sum()
    print(f"  TP53 co-mutation in BRCA-deficient: {n_tp53_comut}/{len(brca_def_lines)} "
          f"({n_tp53_comut/max(1,len(brca_def_lines))*100:.1f}%)")

    # --- Step 12: Cancer type summary ---
    print("\nBuilding cancer type summary...")
    summary = build_cancer_type_summary(df, crispr_lines)

    n_primary = summary["qualifies_primary"].sum()
    n_exploratory = summary["qualifies_exploratory"].sum()
    print(f"  {n_primary} cancer types qualify for primary analysis (>={MIN_DEFICIENT_PRIMARY} def + >={MIN_PROFICIENT} prof with CRISPR)")
    print(f"  {n_exploratory} cancer types qualify for exploratory analysis (>={MIN_DEFICIENT_EXPLORATORY} def)")

    qualifying = summary[summary["qualifies_primary"] | summary["qualifies_exploratory"]]
    for _, row in qualifying.iterrows():
        tag = "PRIMARY" if row["qualifies_primary"] else "EXPLORATORY"
        print(f"    [{tag}] {row['cancer_type']}: "
              f"BRCA1={row['N_brca1_deficient']}, BRCA2={row['N_brca2_deficient']}, "
              f"any={row['N_brca_any_deficient']}, prof={row['N_proficient']} "
              f"(CRISPR: {row['N_any_def_crispr']}/{row['N_prof_crispr']})")

    # --- Step 13: Save outputs ---
    print("\nSaving outputs...")

    # Classified lines
    output_cols = [
        "OncotreeLineage", "OncotreeSubtype",
        "brca1_status", "brca2_status", "brca_combined_status",
        "has_lof_brca1", "has_lof_brca2", "has_homdel_brca1", "has_homdel_brca2",
        "brca1_lof_mutations", "brca1_protein_changes",
        "brca2_lof_mutations", "brca2_protein_changes",
        "shld_complex_status", "shld_lost_genes",
        "pre_rc_status", "pre_rc_affected_genes",
        "co_mutations",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()
    result.to_csv(OUTPUT_DIR / "brca_classified_lines.csv")
    print(f"  brca_classified_lines.csv ({len(result)} lines)")

    # Cancer type summary
    summary.to_csv(OUTPUT_DIR / "cancer_type_summary.csv", index=False)
    print(f"  cancer_type_summary.csv ({len(summary)} cancer types)")

    # --- Step 14: Plots ---
    print("\nGenerating plots...")
    plot_classification_summary(df, summary, OUTPUT_DIR)
    print("  classification_summary.png")

    plot_comutation_heatmap(df, OUTPUT_DIR)
    print("  brca_comutation_heatmap.png")

    # --- Step 15: Classification summary JSON ---
    summary_json = {
        "total_lines": len(df),
        "brca1_deficient": int((df["brca1_status"] == "deficient").sum()),
        "brca2_deficient": int((df["brca2_status"] == "deficient").sum()),
        "brca_any_deficient": int((df["brca_combined_status"] == "deficient").sum()),
        "brca_proficient": int((df["brca_combined_status"] == "proficient").sum()),
        "brca_excluded_vus": int((df["brca_combined_status"] == "excluded_VUS").sum()),
        "brca1_only_deficient": int(n_b1_only),
        "brca2_only_deficient": int(n_b2_only),
        "both_deficient": int(n_both),
        "shld_lost_among_deficient": int(n_shld_lost),
        "shld_intact_among_deficient": int(n_shld_intact),
        "pre_rc_compromised_total": int(n_pre_rc),
        "qualifying_primary": int(n_primary),
        "qualifying_exploratory": int(n_exploratory),
        "qualifying_types": [
            {
                "cancer_type": row["cancer_type"],
                "tier": "primary" if row["qualifies_primary"] else "exploratory",
                "N_brca1_def": int(row["N_brca1_deficient"]),
                "N_brca2_def": int(row["N_brca2_deficient"]),
                "N_any_def": int(row["N_brca_any_deficient"]),
                "N_proficient": int(row["N_proficient"]),
            }
            for _, row in qualifying.iterrows()
        ],
    }
    with open(OUTPUT_DIR / "classification_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print("  classification_summary.json")

    print("\n=== Phase 1 Complete ===")


if __name__ == "__main__":
    main()
