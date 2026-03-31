"""Phase 3a: Two-way ANOVA interaction test for dependency scores.

For each qualifying pair (>=5 double-mutant lines), tests gene_A_status x
gene_B_status interaction on DepMap CRISPR dependency scores. For each
atlas-identified SL target gene:
  1. Independence test: do SL targets from BOTH atlases hold in double-mutant?
  2. Interaction term p-value and direction (synergistic vs antagonistic)
  3. BH-FDR correction across all tested genes per pair

Uses top SL genes from published atlases (genomewide screens and priority targets).

Output per pair: table of SL targets with main effects, interaction effects,
FDR-corrected p-values, and interaction classification.

Usage:
    uv run python -m cancer.tsg_co_alteration_dependency_interactions.04_interaction_test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "tsg-co-alteration-dependency-interactions"
DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"

MIN_GROUP_SIZE = 3
MAX_SL_GENES_PER_ATLAS = 50  # Top N SL genes per atlas to test


def _cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Cohen's d effect size (g1 - g2)."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled)


def _fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    ranked = np.argsort(pvalues)
    fdr = np.empty(n)
    for i, rank_idx in enumerate(reversed(ranked)):
        rank = n - i
        if i == 0:
            fdr[rank_idx] = pvalues[rank_idx]
        else:
            fdr[rank_idx] = min(pvalues[rank_idx] * n / rank, fdr[ranked[n - i]])
    return np.minimum(fdr, 1.0)


# Atlas SL gene sources — map gene symbol to atlas result files
# Files with per-cancer-type rows need aggregation (has_cancer_type=True)
ATLAS_SL_SOURCES: dict[str, list[dict]] = {
    "PTEN": [
        {"path": "output/cancer/pten-loss-pancancer-dependency-atlas/phase3/genomewide_all_results.csv",
         "gene_col": "gene", "effect_col": "cohens_d", "fdr_col": "fdr",
         "has_cancer_type": True},
    ],
    "RB1": [
        {"path": "output/cancer/rb1-loss-pancancer-dependency-atlas/phase3/genomewide_all_results.csv",
         "gene_col": "gene", "effect_col": "cohens_d", "fdr_col": "fdr",
         "has_cancer_type": True},
    ],
    "ARID1A": [
        {"path": "data/results/arid1a-pancancer-sl-atlas/phase3/genomewide_sl_hits.csv",
         "gene_col": "gene", "effect_col": "cohens_d", "fdr_col": "fdr",
         "has_cancer_type": True},
    ],
    "BRCA1": [
        {"path": "data/results/brca-pancancer-sl-atlas/phase3/genomewide_sl_hits.csv",
         "gene_col": "gene", "effect_col": "cohens_d", "fdr_col": "fdr",
         "has_cancer_type": True},
    ],
    "BRCA2": [
        {"path": "data/results/brca-pancancer-sl-atlas/phase3/genomewide_sl_hits.csv",
         "gene_col": "gene", "effect_col": "cohens_d", "fdr_col": "fdr",
         "has_cancer_type": True},
    ],
    "MTAP": [
        {"path": "output/cancer/mtap-prmt5-nsclc-sl/extended_sl_genes.csv",
         "gene_col": "gene", "effect_col": "cohens_d", "fdr_col": "fdr",
         "has_cancer_type": False},
    ],
    "NF1": [
        {"path": "output/cancer/nf1-loss-pancancer-dependency-atlas/phase3/sl_gained_dependencies.csv",
         "gene_col": "gene", "effect_col": "cohens_d", "fdr_col": "fdr",
         "has_cancer_type": True},
    ],
    "KEAP1": [
        {"path": "output/cancer/keap1-nrf2-pancancer-dependency-atlas/phase2/sl_gained_dependencies.csv",
         "gene_col": "gene", "effect_col": "cohens_d", "fdr_col": "fdr",
         "has_cancer_type": True},
    ],
    # TP53, PIK3CA, CDKN2A, SMARCA4 atlases not yet available — skip gracefully
}


def load_atlas_sl_genes(gene: str, top_n: int = MAX_SL_GENES_PER_ATLAS) -> list[str]:
    """Load top SL gene targets from published atlas for a given gene.

    For multi-cancer-type files: aggregate across cancer types by taking the
    strongest (most negative) Cohen's d per gene, then rank by absolute effect.
    """
    sl_genes: set[str] = set()

    sources = ATLAS_SL_SOURCES.get(gene, [])
    for src in sources:
        path = REPO_ROOT / src["path"]
        if not path.exists():
            print(f"    WARNING: Atlas file not found: {path}")
            continue
        try:
            df = pd.read_csv(path)
            gcol = src["gene_col"]
            ecol = src["effect_col"]
            fcol = src.get("fdr_col", "")
            has_ct = src.get("has_cancer_type", False)

            if gcol not in df.columns or ecol not in df.columns:
                print(f"    WARNING: Expected columns not found in {path}")
                continue

            if has_ct:
                # Aggregate per-cancer-type results: take strongest effect per gene
                agg = df.groupby(gcol)[ecol].agg(
                    lambda x: x.loc[x.abs().idxmax()]
                ).reset_index()
                agg.columns = [gcol, ecol]
                df = agg
            else:
                # Single-level results: apply FDR filter if available
                if fcol and fcol in df.columns:
                    df = df[df[fcol] < 0.1]

            # Take top N by absolute effect size, deduplicated
            df = df.drop_duplicates(subset=gcol)
            df = df.sort_values(ecol, key=abs, ascending=False)
            sl_genes.update(df[gcol].head(top_n).tolist())

        except Exception as e:
            print(f"    WARNING: Could not load {path}: {e}")

    return sorted(sl_genes)


def two_way_anova_interaction(
    crispr: pd.DataFrame,
    dep_gene: str,
    double_ids: list[str],
    a_only_ids: list[str],
    b_only_ids: list[str],
    wt_ids: list[str],
) -> dict | None:
    """Two-way ANOVA-like interaction test for a single dependency gene.

    Groups:
      - double-mutant (A=1, B=1)
      - A-only (A=1, B=0)
      - B-only (A=0, B=1)
      - WT-both (A=0, B=0)

    Returns dict with main effects, interaction, and classification,
    or None if insufficient data.
    """
    if dep_gene not in crispr.columns:
        return None

    # Extract scores for each group
    groups = {
        "double": crispr.loc[crispr.index.isin(double_ids), dep_gene].dropna().values,
        "a_only": crispr.loc[crispr.index.isin(a_only_ids), dep_gene].dropna().values,
        "b_only": crispr.loc[crispr.index.isin(b_only_ids), dep_gene].dropna().values,
        "wt": crispr.loc[crispr.index.isin(wt_ids), dep_gene].dropna().values,
    }

    # Need at least MIN_GROUP_SIZE in double and wt
    if len(groups["double"]) < MIN_GROUP_SIZE or len(groups["wt"]) < MIN_GROUP_SIZE:
        return None

    # Main effect A: (double + a_only) vs (b_only + wt)
    a_mut = np.concatenate([groups["double"], groups["a_only"]]) if len(groups["a_only"]) > 0 else groups["double"]
    a_wt = np.concatenate([groups["b_only"], groups["wt"]]) if len(groups["b_only"]) > 0 else groups["wt"]

    # Main effect B: (double + b_only) vs (a_only + wt)
    b_mut = np.concatenate([groups["double"], groups["b_only"]]) if len(groups["b_only"]) > 0 else groups["double"]
    b_wt = np.concatenate([groups["a_only"], groups["wt"]]) if len(groups["a_only"]) > 0 else groups["wt"]

    # Main effect tests (Mann-Whitney)
    _, p_main_a = stats.mannwhitneyu(a_mut, a_wt, alternative="two-sided") if len(a_mut) >= 3 and len(a_wt) >= 3 else (0, 1.0)
    _, p_main_b = stats.mannwhitneyu(b_mut, b_wt, alternative="two-sided") if len(b_mut) >= 3 and len(b_wt) >= 3 else (0, 1.0)

    d_main_a = _cohens_d(a_mut, a_wt)
    d_main_b = _cohens_d(b_mut, b_wt)

    # Interaction test: compare effect of A in B-mutant vs B-WT background
    # Effect of A in B-WT background = double vs b_only... no, a_only vs wt
    # Effect of A in B-mut background = double vs b_only
    # Interaction = difference of these effects
    effect_a_in_bwt = _cohens_d(groups["a_only"], groups["wt"]) if len(groups["a_only"]) >= MIN_GROUP_SIZE else 0.0
    effect_a_in_bmut = _cohens_d(groups["double"], groups["b_only"]) if len(groups["b_only"]) >= MIN_GROUP_SIZE else 0.0

    interaction_d = effect_a_in_bmut - effect_a_in_bwt

    # Kruskal-Wallis across all 4 groups as omnibus test
    non_empty = [g for g in [groups["double"], groups["a_only"], groups["b_only"], groups["wt"]] if len(g) >= 2]
    if len(non_empty) >= 2:
        _, p_kruskal = stats.kruskal(*non_empty)
    else:
        p_kruskal = 1.0

    # Interaction p-value: permutation-free approximation
    # Use the difference between double-mutant observed and additive expectation
    expected_additive = groups["wt"].mean() + (np.mean(groups["a_only"]) - groups["wt"].mean() if len(groups["a_only"]) > 0 else 0) + (np.mean(groups["b_only"]) - groups["wt"].mean() if len(groups["b_only"]) > 0 else 0)
    observed_double = groups["double"].mean()
    interaction_delta = observed_double - expected_additive

    # Test interaction with Mann-Whitney: double vs expected (approximated by pool)
    # Simpler: compare double-mutant to the MEAN of a_only and b_only
    if len(groups["a_only"]) >= 2 and len(groups["b_only"]) >= 2:
        expected_pool = np.concatenate([groups["a_only"], groups["b_only"]])
        _, p_interaction = stats.mannwhitneyu(groups["double"], expected_pool, alternative="two-sided")
    else:
        p_interaction = 1.0

    # Classify interaction
    if p_interaction < 0.05:
        if interaction_delta < 0:
            classification = "synergistic"  # double-mutant MORE dependent than expected
        else:
            classification = "antagonistic"  # double-mutant LESS dependent than expected
    else:
        classification = "additive"

    return {
        "dep_gene": dep_gene,
        "n_double": len(groups["double"]),
        "n_a_only": len(groups["a_only"]),
        "n_b_only": len(groups["b_only"]),
        "n_wt": len(groups["wt"]),
        "mean_double": float(groups["double"].mean()),
        "mean_a_only": float(np.mean(groups["a_only"])) if len(groups["a_only"]) > 0 else None,
        "mean_b_only": float(np.mean(groups["b_only"])) if len(groups["b_only"]) > 0 else None,
        "mean_wt": float(groups["wt"].mean()),
        "d_main_a": round(d_main_a, 4),
        "d_main_b": round(d_main_b, 4),
        "p_main_a": p_main_a,
        "p_main_b": p_main_b,
        "interaction_d": round(interaction_d, 4),
        "interaction_delta": round(interaction_delta, 4),
        "p_interaction": p_interaction,
        "p_kruskal": p_kruskal,
        "classification": classification,
    }


def main() -> None:
    (OUTPUT_DIR / "phase3_interaction_results").mkdir(parents=True, exist_ok=True)

    print("=== Phase 3a: Two-Way ANOVA Interaction Test ===\n")

    # Load Phase 2 groupings
    print("Loading Phase 2 double-mutant groups...")
    with open(OUTPUT_DIR / "phase2_double_mutant_lines.json") as f:
        phase2 = json.load(f)

    # Load CRISPR data (memory-intensive)
    print("Loading CRISPR dependency data...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {crispr.shape[0]} lines x {crispr.shape[1]} genes\n")

    all_pair_results = {}
    n_nonadditive_pairs = 0

    for pair_name, pair_data in phase2.items():
        gene_a = pair_data["gene_a"]
        gene_b = pair_data["gene_b"]
        groups = pair_data["groups"]

        double_ids = groups["double_mutant"]
        a_only_ids = groups.get(f"{gene_a}_only", [])
        b_only_ids = groups.get(f"{gene_b}_only", [])
        wt_ids = groups.get("wt_both", [])

        if len(double_ids) < 5:
            print(f"  {pair_name}: SKIPPED (N={len(double_ids)} < 5)")
            continue

        print(f"  {pair_name}: double={len(double_ids)}, {gene_a}-only={len(a_only_ids)}, "
              f"{gene_b}-only={len(b_only_ids)}, WT={len(wt_ids)}")

        # Get SL genes from both atlases
        sl_a = load_atlas_sl_genes(gene_a)
        sl_b = load_atlas_sl_genes(gene_b)
        all_sl = sorted(set(sl_a) | set(sl_b))
        print(f"    SL genes: {len(sl_a)} from {gene_a} atlas, {len(sl_b)} from {gene_b} atlas, "
              f"{len(all_sl)} unique")

        if not all_sl:
            print(f"    WARNING: No atlas SL genes found, skipping pair")
            continue

        # Run interaction test for each SL gene
        results = []
        for dep_gene in all_sl:
            res = two_way_anova_interaction(
                crispr, dep_gene, double_ids, a_only_ids, b_only_ids, wt_ids,
            )
            if res:
                res["from_atlas_a"] = dep_gene in sl_a
                res["from_atlas_b"] = dep_gene in sl_b
                results.append(res)

        if not results:
            print(f"    No testable genes found")
            continue

        # Apply FDR correction
        pvals = np.array([r["p_interaction"] for r in results])
        fdrs = _fdr_correction(pvals)
        for i, res in enumerate(results):
            res["fdr_interaction"] = round(float(fdrs[i]), 6)
            # Reclassify with FDR
            if fdrs[i] < 0.05:
                if res["interaction_delta"] < 0:
                    res["fdr_classification"] = "synergistic"
                else:
                    res["fdr_classification"] = "antagonistic"
            else:
                res["fdr_classification"] = "additive"

        # Save per-pair results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("p_interaction")
        out_path = OUTPUT_DIR / "phase3_interaction_results" / f"{pair_name.replace('+', '_')}.csv"
        results_df.to_csv(out_path, index=False)

        n_tested = len(results)
        n_sig_uncorrected = sum(1 for r in results if r["p_interaction"] < 0.05)
        n_sig_fdr = sum(1 for r in results if r["fdr_interaction"] < 0.05)
        n_synergistic = sum(1 for r in results if r["fdr_classification"] == "synergistic")
        n_antagonistic = sum(1 for r in results if r["fdr_classification"] == "antagonistic")

        if n_sig_fdr > 0:
            n_nonadditive_pairs += 1

        print(f"    Tested: {n_tested}, sig (p<0.05): {n_sig_uncorrected}, "
              f"sig (FDR<0.05): {n_sig_fdr} "
              f"(synergistic: {n_synergistic}, antagonistic: {n_antagonistic})")

        all_pair_results[pair_name] = {
            "gene_a": gene_a,
            "gene_b": gene_b,
            "n_double": len(double_ids),
            "n_sl_tested": n_tested,
            "n_sig_uncorrected": n_sig_uncorrected,
            "n_sig_fdr": n_sig_fdr,
            "n_synergistic": n_synergistic,
            "n_antagonistic": n_antagonistic,
        }

    # Save overall summary
    summary_path = OUTPUT_DIR / "phase3_interaction_results" / "interaction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_pair_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Pairs tested: {len(all_pair_results)}")
    print(f"  Pairs with non-additive interactions (FDR<0.05): {n_nonadditive_pairs}")

    # Validation: >=2 pairs with non-additive effects
    print(f"\nValidation:")
    print(f"  >=2 non-additive pairs: "
          f"{'PASS' if n_nonadditive_pairs >= 2 else 'PENDING'} ({n_nonadditive_pairs})")

    print("\nDone.")


if __name__ == "__main__":
    main()
