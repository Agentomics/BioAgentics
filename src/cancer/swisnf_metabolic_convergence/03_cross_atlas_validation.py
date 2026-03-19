"""Phase 2: Cross-atlas validation of metabolic dependencies.

Tests whether metabolic vulnerabilities found in one SWI/SNF subunit extend
to the other:
  - HMGCR (ARID1A atlas) → test in SMARCA4-mutant lines
  - Mitochondrial genes (SMARCA4 atlas) → test in ARID1A-mutant lines
  - ADCK5 (ARID1A atlas) → test in SMARCA4-mutant lines
  - Broad convergence: metabolic genes with SL in both ARID1A & SMARCA4

Usage:
    uv run python -m swisnf_metabolic_convergence.03_cross_atlas_validation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1A_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase1a"
PHASE1B_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase1b"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase2"

# Convergence thresholds (nominal — not FDR)
P_THRESHOLD = 0.05
D_THRESHOLD = 0.3  # absolute value
MIN_SAMPLES = 3

# Key genes from each atlas for targeted cross-validation
HMGCR_GENES = ["HMGCR"]
ADCK5_GENES = ["ADCK5"]

# Mitochondrial genes from SMARCA4 atlas (ovarian cancer hits)
SMARCA4_MITO_GENES = [
    "MTERF4", "MICOS13", "DMAC2", "MRPS35", "COX6C", "WARS2", "HIGD2A", "TIMM22",
]

# All targeted genes for direct CRISPR testing
TARGETED_GENES = HMGCR_GENES + ADCK5_GENES + SMARCA4_MITO_GENES


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (group1 - group2)."""
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


def load_phase1b_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 1b screen results for ARID1A and SMARCA4."""
    arid1a = pd.read_csv(PHASE1B_DIR / "screen_arid1a_vs_wt.csv")
    smarca4 = pd.read_csv(PHASE1B_DIR / "screen_smarca4_vs_wt.csv")
    return arid1a, smarca4


def find_convergent_genes_from_screens(
    arid1a_screen: pd.DataFrame,
    smarca4_screen: pd.DataFrame,
) -> pd.DataFrame:
    """Find metabolic genes with SL signal in BOTH ARID1A and SMARCA4 screens.

    Uses nominal p < 0.05 and d < -D_THRESHOLD (SL direction).
    A gene converges if it has at least one qualifying hit in each screen.
    """
    # Filter to SL hits (negative d = more essential in mutant)
    a_hits = arid1a_screen[
        (arid1a_screen["p_value"] < P_THRESHOLD)
        & (arid1a_screen["cohens_d"] < -D_THRESHOLD)
    ]
    s_hits = smarca4_screen[
        (smarca4_screen["p_value"] < P_THRESHOLD)
        & (smarca4_screen["cohens_d"] < -D_THRESHOLD)
    ]

    a_genes = set(a_hits["gene"].unique())
    s_genes = set(s_hits["gene"].unique())
    convergent = sorted(a_genes & s_genes)

    rows = []
    for gene in convergent:
        a_gene = a_hits[a_hits["gene"] == gene]
        s_gene = s_hits[s_hits["gene"] == gene]

        # Best (most negative d) hit in each screen
        a_best_idx = a_gene["cohens_d"].idxmin()
        s_best_idx = s_gene["cohens_d"].idxmin()
        a_best = a_gene.loc[a_best_idx]
        s_best = s_gene.loc[s_best_idx]

        rows.append({
            "gene": gene,
            "arid1a_best_cancer_type": a_best["cancer_type"],
            "arid1a_best_d": round(a_best["cohens_d"], 4),
            "arid1a_best_p": a_best["p_value"],
            "arid1a_best_fdr": a_best["fdr"],
            "arid1a_n_cancer_types": a_gene["cancer_type"].nunique(),
            "smarca4_best_cancer_type": s_best["cancer_type"],
            "smarca4_best_d": round(s_best["cohens_d"], 4),
            "smarca4_best_p": s_best["p_value"],
            "smarca4_best_fdr": s_best["fdr"],
            "smarca4_n_cancer_types": s_gene["cancer_type"].nunique(),
        })

    return pd.DataFrame(rows)


def test_targeted_genes(
    crispr: pd.DataFrame,
    classified: pd.DataFrame,
    genes: list[str],
    source_atlas: str,
) -> pd.DataFrame:
    """Test specific genes from one atlas in the other atlas's mutant lines.

    source_atlas: which atlas the gene was discovered in ("ARID1A" or "SMARCA4").
    Tests the gene in the OTHER atlas's mutant lines.
    """
    summary = pd.read_csv(PHASE1A_DIR / "cancer_type_summary.csv")

    if source_atlas == "ARID1A":
        # Gene found in ARID1A → test in SMARCA4-mutant lines
        target_col = "SMARCA4_disrupted"
        qual_col = "qualifies_smarca4"
        target_name = "SMARCA4"
    else:
        # Gene found in SMARCA4 → test in ARID1A-mutant lines
        target_col = "ARID1A_disrupted"
        qual_col = "qualifies_arid1a"
        target_name = "ARID1A"

    qualifying_types = summary[summary[qual_col]]["cancer_type"].tolist()
    available_genes = [g for g in genes if g in crispr.columns]
    missing_genes = [g for g in genes if g not in crispr.columns]

    if missing_genes:
        print(f"  Warning: genes not in CRISPR data: {missing_genes}")

    results = []
    for cancer_type in qualifying_types:
        ct_lines = classified[classified["OncotreeLineage"] == cancer_type]

        mut_ids = ct_lines[ct_lines[target_col] == True].index.intersection(crispr.index)
        wt_ids = ct_lines[ct_lines["swisnf_any_mutant"] == False].index.intersection(crispr.index)

        if len(mut_ids) < MIN_SAMPLES or len(wt_ids) < MIN_SAMPLES:
            continue

        for gene in available_genes:
            mut_vals = crispr.loc[mut_ids, gene].dropna().values
            wt_vals = crispr.loc[wt_ids, gene].dropna().values

            if len(mut_vals) < MIN_SAMPLES or len(wt_vals) < MIN_SAMPLES:
                continue

            d = cohens_d(mut_vals, wt_vals)
            _, pval = stats.mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")

            results.append({
                "gene": gene,
                "source_atlas": source_atlas,
                "tested_in": target_name,
                "cancer_type": cancer_type,
                "cohens_d": round(d, 4),
                "p_value": pval,
                "n_mut": len(mut_vals),
                "n_wt": len(wt_vals),
                "median_dep_mut": round(float(np.median(mut_vals)), 4),
                "median_dep_wt": round(float(np.median(wt_vals)), 4),
                "is_sl": d < -D_THRESHOLD and pval < P_THRESHOLD,
            })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.sort_values(["gene", "cohens_d"]).reset_index(drop=True)


def build_convergence_matrix(
    arid1a_screen: pd.DataFrame,
    smarca4_screen: pd.DataFrame,
    convergent_genes: list[str],
) -> pd.DataFrame:
    """Build a gene × cancer_type convergence matrix showing effect sizes.

    For each convergent gene, shows Cohen's d in every tested cancer type
    for both ARID1A and SMARCA4 comparisons.
    """
    rows = []
    for gene in convergent_genes:
        a_gene = arid1a_screen[arid1a_screen["gene"] == gene]
        s_gene = smarca4_screen[smarca4_screen["gene"] == gene]

        all_types = sorted(
            set(a_gene["cancer_type"].tolist()) | set(s_gene["cancer_type"].tolist())
        )

        for ct in all_types:
            a_row = a_gene[a_gene["cancer_type"] == ct]
            s_row = s_gene[s_gene["cancer_type"] == ct]

            rows.append({
                "gene": gene,
                "cancer_type": ct,
                "arid1a_d": round(a_row["cohens_d"].iloc[0], 4) if len(a_row) > 0 else None,
                "arid1a_p": a_row["p_value"].iloc[0] if len(a_row) > 0 else None,
                "smarca4_d": round(s_row["cohens_d"].iloc[0], 4) if len(s_row) > 0 else None,
                "smarca4_p": s_row["p_value"].iloc[0] if len(s_row) > 0 else None,
                "both_sl": (
                    len(a_row) > 0
                    and len(s_row) > 0
                    and a_row["cohens_d"].iloc[0] < -D_THRESHOLD
                    and s_row["cohens_d"].iloc[0] < -D_THRESHOLD
                    and a_row["p_value"].iloc[0] < P_THRESHOLD
                    and s_row["p_value"].iloc[0] < P_THRESHOLD
                ),
            })

    return pd.DataFrame(rows)


def categorize_convergent_genes(
    convergent_df: pd.DataFrame,
    metabolic_gene_list: pd.DataFrame,
) -> pd.DataFrame:
    """Add pathway annotations to convergent genes."""
    pathway_map = dict(
        zip(metabolic_gene_list["gene"], metabolic_gene_list["pathways"])
    )
    convergent_df = convergent_df.copy()
    convergent_df["pathways"] = convergent_df["gene"].map(pathway_map).fillna("")

    # Assign broad category based on pathway
    def categorize(pathways: str) -> str:
        p = pathways.lower()
        if "oxidative phosphorylation" in p:
            return "OXPHOS"
        if any(k in p for k in ["steroid", "terpenoid backbone", "biosynthesis of unsaturated"]):
            return "Cholesterol/Lipid"
        if "fatty acid" in p:
            return "Fatty acid metabolism"
        if "carbon metabolism" in p or "glycolysis" in p or "tca" in p or "citrate" in p:
            return "Central carbon metabolism"
        if "amino acid" in p or any(aa in p for aa in [
            "alanine", "arginine", "cysteine", "glycine", "histidine",
            "lysine", "phenylalanine", "tryptophan", "tyrosine", "valine",
        ]):
            return "Amino acid metabolism"
        if "nucleotide" in p or "purine" in p or "pyrimidine" in p:
            return "Nucleotide metabolism"
        if "glycan" in p or "glycosaminoglycan" in p or "glycosphingolipid" in p:
            return "Glycan biosynthesis"
        if "sphingolipid" in p or "glycerolipid" in p or "glycerophospholipid" in p:
            return "Lipid metabolism"
        if "one carbon" in p or "folate" in p:
            return "One-carbon metabolism"
        if "glutathione" in p or "porphyrin" in p or "selenium" in p:
            return "Redox/cofactor"
        return "Other metabolism"

    convergent_df["category"] = convergent_df["pathways"].apply(categorize)
    return convergent_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: Cross-Atlas Validation ===\n")

    # Load Phase 1b screen results
    print("Loading Phase 1b results...")
    arid1a_screen, smarca4_screen = load_phase1b_results()
    print(f"  ARID1A screen: {len(arid1a_screen)} tests")
    print(f"  SMARCA4 screen: {len(smarca4_screen)} tests")

    # Load classified cell lines and CRISPR data for targeted tests
    print("Loading classified cell lines...")
    classified = pd.read_csv(PHASE1A_DIR / "swisnf_classified_lines.csv", index_col=0)

    print("Loading CRISPR gene effect data...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {crispr.shape[0]} cell lines, {crispr.shape[1]} genes")

    # === Part 1: Broad convergence from Phase 1b screens ===
    print("\n--- Part 1: Broad metabolic gene convergence ---")
    convergent_df = find_convergent_genes_from_screens(arid1a_screen, smarca4_screen)
    print(f"  {len(convergent_df)} genes with SL signal in BOTH ARID1A and SMARCA4")

    # Add pathway annotations
    gene_list = pd.read_csv(PHASE1B_DIR / "metabolic_gene_list.csv")
    convergent_df = categorize_convergent_genes(convergent_df, gene_list)

    # Category summary
    cat_counts = convergent_df["category"].value_counts()
    print("\n  Convergent genes by pathway category:")
    for cat, count in cat_counts.items():
        print(f"    {cat}: {count}")

    convergent_df.to_csv(OUTPUT_DIR / "convergent_metabolic_genes.csv", index=False)

    # Top convergent genes (strongest combined signal)
    convergent_df["combined_d"] = (
        convergent_df["arid1a_best_d"] + convergent_df["smarca4_best_d"]
    )
    top = convergent_df.sort_values("combined_d").head(20)
    print("\n  Top 20 convergent genes (strongest combined SL signal):")
    for _, row in top.iterrows():
        print(
            f"    {row['gene']:12s} [{row['category']:25s}] "
            f"ARID1A d={row['arid1a_best_d']:+.2f} ({row['arid1a_best_cancer_type']}), "
            f"SMARCA4 d={row['smarca4_best_d']:+.2f} ({row['smarca4_best_cancer_type']})"
        )

    # === Part 2: Targeted cross-validation of key genes ===
    print("\n--- Part 2: Targeted cross-validation ---")

    # Test HMGCR + ADCK5 (ARID1A atlas genes) in SMARCA4 lines
    print("\nTesting ARID1A atlas genes (HMGCR, ADCK5) in SMARCA4-mutant lines...")
    arid1a_to_smarca4 = test_targeted_genes(
        crispr, classified,
        HMGCR_GENES + ADCK5_GENES,
        source_atlas="ARID1A",
    )
    if len(arid1a_to_smarca4) > 0:
        print(f"  {len(arid1a_to_smarca4)} tests")
        for _, row in arid1a_to_smarca4.iterrows():
            sl_flag = " ***SL***" if row["is_sl"] else ""
            print(
                f"    {row['gene']:8s} in SMARCA4-{row['cancer_type']:25s} "
                f"d={row['cohens_d']:+.3f} p={row['p_value']:.4f}{sl_flag}"
            )
        arid1a_to_smarca4.to_csv(
            OUTPUT_DIR / "targeted_arid1a_genes_in_smarca4.csv", index=False,
        )
    else:
        print("  No results (insufficient qualifying cancer types)")

    # Test SMARCA4 mito genes in ARID1A lines
    print("\nTesting SMARCA4 atlas mito genes in ARID1A-mutant lines...")
    smarca4_to_arid1a = test_targeted_genes(
        crispr, classified,
        SMARCA4_MITO_GENES,
        source_atlas="SMARCA4",
    )
    if len(smarca4_to_arid1a) > 0:
        print(f"  {len(smarca4_to_arid1a)} tests")
        sl_hits = smarca4_to_arid1a[smarca4_to_arid1a["is_sl"]]
        if len(sl_hits) > 0:
            print(f"  {len(sl_hits)} SL hits:")
            for _, row in sl_hits.iterrows():
                print(
                    f"    {row['gene']:8s} in ARID1A-{row['cancer_type']:25s} "
                    f"d={row['cohens_d']:+.3f} p={row['p_value']:.4f}"
                )
        else:
            print("  No SL hits at nominal threshold")

        # Show all results for key genes
        for gene in SMARCA4_MITO_GENES:
            gene_df = smarca4_to_arid1a[smarca4_to_arid1a["gene"] == gene]
            if len(gene_df) > 0:
                best = gene_df.loc[gene_df["cohens_d"].idxmin()]
                print(
                    f"    {gene:8s} best: ARID1A-{best['cancer_type']:25s} "
                    f"d={best['cohens_d']:+.3f} p={best['p_value']:.4f}"
                )
        smarca4_to_arid1a.to_csv(
            OUTPUT_DIR / "targeted_smarca4_mito_in_arid1a.csv", index=False,
        )
    else:
        print("  No results (insufficient qualifying cancer types)")

    # === Part 3: Convergence matrix ===
    print("\n--- Part 3: Convergence matrix ---")
    convergent_gene_list = convergent_df["gene"].tolist()
    matrix = build_convergence_matrix(
        arid1a_screen, smarca4_screen, convergent_gene_list,
    )
    matrix.to_csv(OUTPUT_DIR / "convergence_matrix.csv", index=False)
    n_both_sl = matrix["both_sl"].sum()
    print(f"  {len(matrix)} gene-cancer_type pairs in matrix")
    print(f"  {n_both_sl} pairs with SL in both ARID1A and SMARCA4")

    # === Summary ===
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    print(f"\nConvergent metabolic genes: {len(convergent_df)}")
    print(f"Success criterion (>=3): {'PASS' if len(convergent_df) >= 3 else 'FAIL'}")

    # Check OXPHOS convergence specifically
    oxphos_genes = convergent_df[convergent_df["category"] == "OXPHOS"]
    print(f"\nOXPHOS convergent genes: {len(oxphos_genes)}")
    if len(oxphos_genes) > 0:
        print("  Genes: " + ", ".join(oxphos_genes["gene"].tolist()))

    # Check pathway concentration
    top2_cats = cat_counts.head(2)
    top2_count = top2_cats.sum()
    print(f"\nTop 2 pathway categories contain {top2_count}/{len(convergent_df)} "
          f"convergent genes ({top2_count / len(convergent_df):.0%})")
    print(f"  {top2_cats.index[0]}: {top2_cats.iloc[0]}")
    if len(top2_cats) > 1:
        print(f"  {top2_cats.index[1]}: {top2_cats.iloc[1]}")

    # HMGCR cross-validation result
    if len(arid1a_to_smarca4) > 0:
        hmgcr_results = arid1a_to_smarca4[arid1a_to_smarca4["gene"] == "HMGCR"]
        if len(hmgcr_results) > 0:
            hmgcr_sl = hmgcr_results[hmgcr_results["is_sl"]]
            print(f"\nHMGCR in SMARCA4: {len(hmgcr_sl)}/{len(hmgcr_results)} cancer types show SL")
        else:
            print("\nHMGCR: not testable in SMARCA4 lines")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
