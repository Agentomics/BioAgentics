"""Phase 3b: Emergence test for novel double-mutant dependencies.

For each qualifying pair, runs genome-wide CRISPR dependency screen comparing
double-mutant lines vs all other lines. Identifies dependencies significant in
double-mutant lines but NOT in either individual single-mutant atlas.

Uses Mann-Whitney U test with Cohen's d effect sizes and BH-FDR correction.
Filter: FDR<0.05, |d|>0.5, gene NOT in either single-atlas SL list.

Output per pair: ranked list of emergent dependencies.

Usage:
    uv run python -m cancer.tsg_co_alteration_dependency_interactions.05_emergent_dependencies
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

MIN_DOUBLE = 10  # Minimum double-mutant lines for genome-wide screen
FDR_THRESHOLD = 0.05
EFFECT_THRESHOLD = 0.5  # |Cohen's d| > 0.5


def _cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled)


def _fdr_correction(pvalues: np.ndarray) -> np.ndarray:
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


def genomewide_screen(
    crispr: pd.DataFrame,
    test_ids: list[str],
    control_ids: list[str],
) -> pd.DataFrame:
    """Mann-Whitney U genome-wide screen: test vs control group."""
    test_set = set(test_ids) & set(crispr.index)
    ctrl_set = set(control_ids) & set(crispr.index)

    results = []
    for gene in crispr.columns:
        test_vals = crispr.loc[list(test_set), gene].dropna().values
        ctrl_vals = crispr.loc[list(ctrl_set), gene].dropna().values

        if len(test_vals) < 3 or len(ctrl_vals) < 3:
            continue

        _, pval = stats.mannwhitneyu(test_vals, ctrl_vals, alternative="two-sided")
        d = _cohens_d(test_vals, ctrl_vals)
        results.append({
            "gene": gene,
            "cohens_d": round(d, 4),
            "p_value": pval,
            "mean_test": round(float(test_vals.mean()), 4),
            "mean_control": round(float(ctrl_vals.mean()), 4),
            "n_test": len(test_vals),
            "n_control": len(ctrl_vals),
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["fdr"] = _fdr_correction(df["p_value"].values)
    return df


def load_atlas_known_genes(gene: str) -> set[str]:
    """Load known SL genes from published atlas (same sources as Phase 3a)."""
    import importlib
    mod = importlib.import_module(
        "cancer.tsg_co_alteration_dependency_interactions.04_interaction_test"
    )
    return set(mod.load_atlas_sl_genes(gene, top_n=200))


def main() -> None:
    out_dir = OUTPUT_DIR / "phase3_emergent_dependencies"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3b: Emergent Double-Mutant Dependencies ===\n")

    # Load Phase 2 groupings
    with open(OUTPUT_DIR / "phase2_double_mutant_lines.json") as f:
        phase2 = json.load(f)

    # Load CRISPR data
    print("Loading CRISPR dependency data...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    all_lines = set(crispr.index)
    print(f"  {crispr.shape[0]} lines x {crispr.shape[1]} genes\n")

    summary_results = {}

    for pair_name, pair_data in phase2.items():
        gene_a = pair_data["gene_a"]
        gene_b = pair_data["gene_b"]
        groups = pair_data["groups"]

        double_ids = groups["double_mutant"]

        if len(double_ids) < MIN_DOUBLE:
            print(f"  {pair_name}: SKIPPED (N={len(double_ids)} < {MIN_DOUBLE})")
            continue

        # Control = all lines NOT in double-mutant group
        control_ids = sorted(all_lines - set(double_ids))

        print(f"  {pair_name}: screening {len(double_ids)} double vs {len(control_ids)} other lines")

        # Run genome-wide screen
        results = genomewide_screen(crispr, double_ids, control_ids)

        if results.empty:
            print(f"    No testable genes")
            continue

        # Filter to significant hits
        sig = results[(results["fdr"] < FDR_THRESHOLD) & (results["cohens_d"].abs() > EFFECT_THRESHOLD)]
        print(f"    Genome-wide hits: {len(sig)} (FDR<{FDR_THRESHOLD}, |d|>{EFFECT_THRESHOLD})")

        # Load known SL genes from both atlases
        known_a = load_atlas_known_genes(gene_a)
        known_b = load_atlas_known_genes(gene_b)
        known_all = known_a | known_b

        # Filter to NOVEL emergent dependencies
        if not sig.empty:
            novel = sig[~sig["gene"].isin(known_all)].copy()
        else:
            novel = pd.DataFrame()

        n_novel = len(novel)
        print(f"    Novel (not in either atlas): {n_novel}")

        # Save all results
        results_sorted = results.sort_values("p_value")
        results_sorted.to_csv(out_dir / f"{pair_name.replace('+', '_')}_all.csv", index=False)

        if not novel.empty:
            novel = novel.sort_values("cohens_d")
            novel.to_csv(out_dir / f"{pair_name.replace('+', '_')}_emergent.csv", index=False)

            # Print top 5 emergent dependencies
            print(f"    Top emergent dependencies:")
            for _, row in novel.head(5).iterrows():
                print(f"      {row['gene']:15s}  d={row['cohens_d']:+.3f}  FDR={row['fdr']:.2e}")

        summary_results[pair_name] = {
            "gene_a": gene_a,
            "gene_b": gene_b,
            "n_double": len(double_ids),
            "n_genomewide_hits": len(sig),
            "n_novel_emergent": n_novel,
            "n_known_a": len(known_a),
            "n_known_b": len(known_b),
        }

    # Save summary
    with open(out_dir / "emergent_summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Summary:")
    total_novel = sum(r["n_novel_emergent"] for r in summary_results.values())
    pairs_with_novel = sum(1 for r in summary_results.values() if r["n_novel_emergent"] > 0)
    print(f"  Pairs screened: {len(summary_results)}")
    print(f"  Pairs with novel emergent dependencies: {pairs_with_novel}")
    print(f"  Total novel emergent genes: {total_novel}")
    print("\nDone.")


if __name__ == "__main__":
    main()
