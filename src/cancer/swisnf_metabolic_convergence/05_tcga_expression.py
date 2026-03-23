"""Phase 4: TCGA expression analysis of metabolic genes in SWI/SNF-mutant tumors.

Tests whether SWI/SNF-selective metabolic dependency genes show transcriptional
deregulation in patient tumors (TCGA).

Usage:
    PYTHONPATH=src/cancer:src uv run python src/cancer/swisnf_metabolic_convergence/05_tcga_expression.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import (
    build_uuid_to_patient_map,
    load_tcga_expression_matrix,
    load_tcga_mutations,
)

PHASE2_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase2"
PHASE1B_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase1b"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase4"
TCGA_DIR = REPO_ROOT / "data" / "tcga"

# LOF variant classifications (same as cell line classifier)
LOF_CLASSES = {
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Splice_Region",
    "Translation_Start_Site",
    "Nonstop_Mutation",
}

# SWI/SNF target genes
SWISNF_GENES = ["ARID1A", "SMARCA4"]

# Available TCGA cancer types with expression + mutation data
TCGA_COHORTS = {
    "LUAD": {"expr": TCGA_DIR / "luad" / "expression", "mut": TCGA_DIR / "luad" / "mutations"},
    "LUSC": {"expr": TCGA_DIR / "lusc" / "expression", "mut": TCGA_DIR / "lusc" / "mutations"},
}


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


def classify_swisnf_tcga(mutations: pd.DataFrame) -> dict[str, set[str]]:
    """Classify TCGA patients as SWI/SNF-mutant from mutation data.

    Returns dict with keys 'ARID1A', 'SMARCA4', 'any' mapping to sets of
    patient barcodes (TCGA-XX-XXXX).
    """
    # Filter to SWI/SNF genes with LOF mutations
    swisnf_lof = mutations[
        (mutations["Hugo_Symbol"].isin(SWISNF_GENES))
        & (mutations["Variant_Classification"].isin(LOF_CLASSES))
    ]

    result: dict[str, set[str]] = {"ARID1A": set(), "SMARCA4": set(), "any": set()}

    for _, row in swisnf_lof.iterrows():
        # Extract patient barcode (first 12 chars: TCGA-XX-XXXX)
        barcode = str(row["Tumor_Sample_Barcode"])
        patient = "-".join(barcode.split("-")[:3])
        gene = row["Hugo_Symbol"]
        if gene in result:
            result[gene].add(patient)
        result["any"].add(patient)

    return result


def compute_differential_expression(
    expr_matrix: pd.DataFrame,
    mut_patients: set[str],
    wt_patients: set[str],
    genes: list[str],
) -> pd.DataFrame:
    """Compute differential expression for given genes between mutant and WT.

    expr_matrix: samples × genes DataFrame (TPM values, patient barcode index).
    Returns DataFrame with log2FC, p-value per gene.
    """
    available_genes = [g for g in genes if g in expr_matrix.columns]

    results = []
    for gene in available_genes:
        mut_vals = expr_matrix.loc[expr_matrix.index.isin(mut_patients), gene].dropna()
        wt_vals = expr_matrix.loc[expr_matrix.index.isin(wt_patients), gene].dropna()

        if len(mut_vals) < 3 or len(wt_vals) < 3:
            continue

        # Log2 fold change (add pseudocount for TPM=0)
        mut_mean = mut_vals.mean() + 1e-3
        wt_mean = wt_vals.mean() + 1e-3
        log2fc = np.log2(mut_mean / wt_mean)

        # Mann-Whitney U test
        _, pval = stats.mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")

        results.append({
            "gene": gene,
            "log2fc": round(float(log2fc), 4),
            "p_value": pval,
            "mean_tpm_mut": round(float(mut_vals.mean()), 4),
            "mean_tpm_wt": round(float(wt_vals.mean()), 4),
            "median_tpm_mut": round(float(mut_vals.median()), 4),
            "median_tpm_wt": round(float(wt_vals.median()), 4),
            "n_mut": len(mut_vals),
            "n_wt": len(wt_vals),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["fdr"] = fdr_correction(df["p_value"].values)
    return df.sort_values("p_value").reset_index(drop=True)


def process_cohort(
    cohort_name: str,
    expr_dir: Path,
    mut_dir: Path,
    convergent_genes: list[str],
) -> pd.DataFrame:
    """Process a single TCGA cohort: load data, classify, compute DE."""
    print(f"\n--- {cohort_name} ---")

    # Load expression matrix
    print(f"  Loading expression data from {expr_dir.name}/...")
    expr_matrix = load_tcga_expression_matrix(expr_dir)
    print(f"    {expr_matrix.shape[0]} samples, {expr_matrix.shape[1]} genes")

    # Build UUID-to-patient mapping
    print("  Building UUID-to-patient mapping...")
    uuid_to_patient = build_uuid_to_patient_map(expr_dir)
    print(f"    {len(uuid_to_patient)} UUID-patient mappings")

    # Remap expression index from UUID to patient barcode
    expr_matrix.index = expr_matrix.index.map(lambda x: uuid_to_patient.get(x, x))
    # Drop samples without patient mapping
    unmapped = expr_matrix.index.str.len() > 15  # UUIDs are longer than patient barcodes
    if unmapped.any():
        print(f"    Dropping {unmapped.sum()} unmapped samples")
        expr_matrix = expr_matrix[~unmapped]

    # Remove duplicate patients (keep first)
    if expr_matrix.index.duplicated().any():
        n_dup = expr_matrix.index.duplicated().sum()
        print(f"    Removing {n_dup} duplicate patient entries")
        expr_matrix = expr_matrix[~expr_matrix.index.duplicated(keep="first")]

    # Load mutations
    print(f"  Loading mutations from {mut_dir.name}/...")
    mutations = load_tcga_mutations(mut_dir)
    print(f"    {len(mutations)} mutation records")

    # Classify SWI/SNF status
    swisnf = classify_swisnf_tcga(mutations)
    print(f"    ARID1A-mutant: {len(swisnf['ARID1A'])} patients")
    print(f"    SMARCA4-mutant: {len(swisnf['SMARCA4'])} patients")
    print(f"    Any SWI/SNF-mutant: {len(swisnf['any'])} patients")

    # Define WT patients (in expression matrix but not SWI/SNF-mutant)
    all_patients = set(expr_matrix.index)
    wt_patients = all_patients - swisnf["any"]
    mut_patients = swisnf["any"] & all_patients
    print(f"    Patients with expression: {len(all_patients)}")
    print(f"    SWI/SNF-mutant with expression: {len(mut_patients)}")
    print(f"    WT with expression: {len(wt_patients)}")

    if len(mut_patients) < 3:
        print(f"    SKIP: insufficient SWI/SNF-mutant patients (<3)")
        return pd.DataFrame()

    # Compute differential expression
    print("  Computing differential expression...")
    de_results = compute_differential_expression(
        expr_matrix, mut_patients, wt_patients, convergent_genes,
    )

    if len(de_results) > 0:
        de_results["cohort"] = cohort_name
        sig = de_results[de_results["fdr"] < 0.05]
        print(f"    {len(de_results)} genes tested")
        print(f"    {len(sig)} genes with FDR < 0.05")

        # Top hits
        for _, row in de_results.head(10).iterrows():
            sig_flag = " *" if row["fdr"] < 0.05 else ""
            print(
                f"      {row['gene']:12s} log2FC={row['log2fc']:+.3f} "
                f"p={row['p_value']:.2e} FDR={row['fdr']:.2e}{sig_flag}"
            )

    # Free memory
    del expr_matrix, mutations
    return de_results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4: TCGA Expression Analysis ===\n")

    # Load convergent genes from Phase 2
    print("Loading convergent metabolic genes from Phase 2...")
    convergent_df = pd.read_csv(PHASE2_DIR / "convergent_metabolic_genes.csv")
    convergent_genes = convergent_df["gene"].tolist()
    print(f"  {len(convergent_genes)} convergent genes to test")

    # Also load Phase 1b screen results for correlation analysis
    arid1a_screen = pd.read_csv(PHASE1B_DIR / "screen_arid1a_vs_wt.csv")
    smarca4_screen = pd.read_csv(PHASE1B_DIR / "screen_smarca4_vs_wt.csv")

    # Process each TCGA cohort
    all_results = []
    for cohort_name, paths in TCGA_COHORTS.items():
        if not paths["expr"].exists() or not paths["mut"].exists():
            print(f"\n--- {cohort_name}: SKIP (data not available) ---")
            continue

        de_results = process_cohort(
            cohort_name, paths["expr"], paths["mut"], convergent_genes,
        )
        if len(de_results) > 0:
            all_results.append(de_results)

    if not all_results:
        print("\nNo results — no TCGA cohorts had sufficient data.")
        return

    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "tcga_differential_expression.csv", index=False)
    print(f"\n\n{'='*60}")
    print("COMBINED RESULTS")
    print(f"{'='*60}")

    # Per-gene summary: significant in ANY cohort
    gene_sig = combined[combined["fdr"] < 0.05].groupby("gene").agg(
        n_cohorts_sig=("cohort", "nunique"),
        best_log2fc=("log2fc", lambda x: x.iloc[np.argmin(np.abs(x))] if len(x) > 0 else 0),
        best_fdr=("fdr", "min"),
    ).sort_values("best_fdr").reset_index()

    gene_sig.to_csv(OUTPUT_DIR / "gene_significance_summary.csv", index=False)

    n_tested = combined["gene"].nunique()
    n_sig = len(gene_sig)
    pct_sig = n_sig / n_tested * 100 if n_tested > 0 else 0

    print(f"\nGenes tested: {n_tested}")
    print(f"Genes with FDR < 0.05 in any cohort: {n_sig} ({pct_sig:.0f}%)")
    print(f"Success criterion (>=50%): {'PASS' if pct_sig >= 50 else 'FAIL'}")

    # Top significant genes
    if len(gene_sig) > 0:
        print(f"\nTop significant genes (FDR < 0.05):")
        for _, row in gene_sig.head(20).iterrows():
            print(
                f"  {row['gene']:12s} log2FC={row['best_log2fc']:+.3f} "
                f"FDR={row['best_fdr']:.2e} ({row['n_cohorts_sig']} cohort(s))"
            )

    # Direction analysis: upregulated vs downregulated
    sig_genes = combined[combined["fdr"] < 0.05]
    if len(sig_genes) > 0:
        n_up = (sig_genes["log2fc"] > 0).sum()
        n_down = (sig_genes["log2fc"] < 0).sum()
        print(f"\nDirection: {n_up} upregulated, {n_down} downregulated in SWI/SNF-mutant")

    # Correlation with dependency (do depleted genes show expression changes?)
    print("\n--- Dependency-expression correlation ---")
    # Get min Cohen's d per gene from combined screen
    combined_screen = pd.concat([
        arid1a_screen[arid1a_screen["cancer_type"].str.contains("Lung", na=False)],
        smarca4_screen[smarca4_screen["cancer_type"].str.contains("Lung", na=False)],
    ])

    if len(combined_screen) > 0:
        dep_min_d = combined_screen.groupby("gene")["cohens_d"].min()

        # Get expression log2fc per gene (use lung cohorts)
        lung_de = combined[combined["cohort"].isin(["LUAD", "LUSC"])]
        if len(lung_de) > 0:
            expr_fc = lung_de.groupby("gene")["log2fc"].mean()

            # Merge
            shared = set(dep_min_d.index) & set(expr_fc.index)
            if len(shared) > 10:
                dep_vals = dep_min_d.loc[sorted(shared)].values
                expr_vals = expr_fc.loc[sorted(shared)].values
                r, p = stats.spearmanr(dep_vals, expr_vals)
                print(f"  Spearman correlation (dependency d vs expression log2FC): r={r:.3f}, p={p:.2e}")
                print(f"  N genes: {len(shared)}")
                pd.DataFrame({
                    "gene": sorted(shared),
                    "dependency_d": dep_vals,
                    "expression_log2fc": expr_vals,
                }).to_csv(OUTPUT_DIR / "dependency_expression_correlation.csv", index=False)

    # OXPHOS-specific analysis
    print("\n--- OXPHOS gene expression in SWI/SNF-mutant ---")
    oxphos_convergent = convergent_df[convergent_df["category"] == "OXPHOS"]["gene"].tolist()
    oxphos_de = combined[combined["gene"].isin(oxphos_convergent)]
    if len(oxphos_de) > 0:
        oxphos_sig = oxphos_de[oxphos_de["fdr"] < 0.05]
        print(f"  OXPHOS genes tested: {oxphos_de['gene'].nunique()}")
        print(f"  OXPHOS genes with FDR < 0.05: {oxphos_sig['gene'].nunique()}")
        if len(oxphos_sig) > 0:
            mean_fc = oxphos_sig["log2fc"].mean()
            print(f"  Mean log2FC of significant OXPHOS genes: {mean_fc:+.3f}")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
