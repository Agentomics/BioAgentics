"""Validation: robustness testing for ARID1A pan-cancer SL atlas.

Runs leave-one-out stability analysis and permutation tests to confirm
that key findings are not driven by single outlier lines and that the
ARID1B binomial pattern is unlikely by chance.

Usage:
    uv run python -m arid1a_pancancer_sl_atlas.validation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase1"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase2"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "validation"

N_PERMUTATIONS = 1000
SEED = 42

# Top SL genes to robustness-test per cancer type
LOO_GENES = ["ARID1B", "EZH2", "HMGCR", "ADCK5", "MDM2"]


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def leave_one_out(
    mutant_vals: np.ndarray,
    wt_vals: np.ndarray,
    full_d: float,
) -> dict:
    """Leave-one-out robustness: remove each mutant line and recompute d.

    Returns summary with worst-case drop, direction flips, and per-line results.
    """
    n = len(mutant_vals)
    if n < 3:
        return {"n_mut": n, "skipped": True, "reason": "n_mut < 3"}

    loo_ds = []
    for i in range(n):
        reduced = np.delete(mutant_vals, i)
        if len(reduced) < 2:
            continue
        d_loo = cohens_d(reduced, wt_vals)
        loo_ds.append({"dropped_idx": i, "dropped_val": float(mutant_vals[i]), "d_loo": d_loo})

    if not loo_ds:
        return {"n_mut": n, "skipped": True, "reason": "no valid LOO iterations"}

    ds = [r["d_loo"] for r in loo_ds]
    direction_flips = sum(1 for d in ds if (d > 0) != (full_d > 0)) if full_d != 0 else 0
    pct_changes = [abs(d - full_d) / abs(full_d) * 100 if full_d != 0 else 0 for d in ds]
    max_pct_change = max(pct_changes)
    lost_50pct = sum(1 for p in pct_changes if p > 50)

    return {
        "n_mut": n,
        "skipped": False,
        "full_d": full_d,
        "mean_loo_d": float(np.mean(ds)),
        "min_loo_d": float(np.min(ds)),
        "max_loo_d": float(np.max(ds)),
        "direction_flips": direction_flips,
        "lost_gt50pct": lost_50pct,
        "max_pct_change": round(max_pct_change, 1),
        "stable": direction_flips == 0 and lost_50pct == 0,
    }


def permutation_test(
    mutant_vals: np.ndarray,
    wt_vals: np.ndarray,
    observed_p: float,
    n_perm: int = N_PERMUTATIONS,
) -> dict:
    """Permutation test: shuffle ARID1A labels and recompute Mann-Whitney.

    Returns fraction of permuted p-values <= observed.
    """
    rng = np.random.default_rng(SEED)
    all_vals = np.concatenate([mutant_vals, wt_vals])
    n_mut = len(mutant_vals)

    perm_pvals = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(all_vals)
        perm_mut = shuffled[:n_mut]
        perm_wt = shuffled[n_mut:]
        _, p = stats.mannwhitneyu(perm_mut, perm_wt, alternative="two-sided")
        perm_pvals[i] = p

    empirical_p = float(np.mean(perm_pvals <= observed_p))
    return {
        "observed_p": observed_p,
        "empirical_p": empirical_p,
        "n_permutations": n_perm,
        "perm_p_median": float(np.median(perm_pvals)),
        "perm_p_5th": float(np.percentile(perm_pvals, 5)),
    }


def binomial_pattern_test(
    n_sig: int,
    n_total: int,
    alpha: float = 0.05,
) -> dict:
    """Test whether observing n_sig/n_total types with p<alpha is unlikely by chance.

    Under null: each type has probability alpha of being significant by chance.
    """
    binom_p = 1.0 - stats.binom.cdf(n_sig - 1, n_total, alpha)
    return {
        "n_significant": n_sig,
        "n_total": n_total,
        "alpha": alpha,
        "expected_by_chance": round(n_total * alpha, 1),
        "binomial_p": float(binom_p),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Validation: Robustness Testing ===\n")

    # Load data
    classified = pd.read_csv(PHASE1_DIR / "all_cell_lines_classified.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "cancer_type_summary.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()
    phase2 = pd.read_csv(PHASE2_DIR / "known_sl_effect_sizes.csv")

    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    merged = classified.join(crispr, how="inner")

    available_genes = [g for g in LOO_GENES if g in crispr.columns]
    print(f"Testing genes: {available_genes}")
    print(f"Qualifying cancer types: {len(qualifying)}\n")

    # ========================================
    # Part 1: Leave-one-out robustness
    # ========================================
    print("--- Part 1: Leave-One-Out Robustness ---\n")
    loo_results = []

    for cancer_type in qualifying:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        mutant = ct_data[ct_data["ARID1A_status"] == "mutant"]
        wt = ct_data[ct_data["ARID1A_status"] == "WT"]

        for gene in available_genes:
            mut_vals = mutant[gene].dropna().values
            wt_vals = wt[gene].dropna().values

            if len(mut_vals) < 3 or len(wt_vals) < 3:
                continue

            full_d = cohens_d(mut_vals, wt_vals)
            loo = leave_one_out(mut_vals, wt_vals, full_d)

            loo["cancer_type"] = cancer_type
            loo["gene"] = gene
            loo_results.append(loo)

            status = "STABLE" if loo.get("stable") else "UNSTABLE"
            if not loo.get("skipped"):
                flips = loo["direction_flips"]
                lost = loo["lost_gt50pct"]
                print(f"  {cancer_type}/{gene}: d={full_d:.3f}, LOO range=[{loo['min_loo_d']:.3f}, {loo['max_loo_d']:.3f}] "
                      f"max_change={loo['max_pct_change']}% flips={flips} [{status}]")

    loo_df = pd.DataFrame(loo_results)
    loo_df.to_csv(OUTPUT_DIR / "leave_one_out_results.csv", index=False)

    n_unstable = len(loo_df[loo_df.get("stable") == False]) if "stable" in loo_df.columns else 0
    n_tested = len(loo_df[loo_df.get("skipped") == False]) if "skipped" in loo_df.columns else 0
    print(f"\n  LOO summary: {n_tested} tests, {n_unstable} unstable")

    # ========================================
    # Part 2: Permutation tests for ARID1B
    # ========================================
    print("\n--- Part 2: Permutation Tests (ARID1B, 1000 permutations) ---\n")
    perm_results = []
    n_nominal_sig = 0

    for cancer_type in qualifying:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        mutant = ct_data[ct_data["ARID1A_status"] == "mutant"]
        wt = ct_data[ct_data["ARID1A_status"] == "WT"]

        mut_vals = mutant["ARID1B"].dropna().values
        wt_vals = wt["ARID1B"].dropna().values

        if len(mut_vals) < 3 or len(wt_vals) < 3:
            continue

        _, obs_p = stats.mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
        d = cohens_d(mut_vals, wt_vals)
        perm = permutation_test(mut_vals, wt_vals, obs_p)
        perm["cancer_type"] = cancer_type
        perm["cohens_d"] = d
        perm["n_mut"] = len(mut_vals)
        perm["n_wt"] = len(wt_vals)
        perm_results.append(perm)

        if obs_p < 0.05:
            n_nominal_sig += 1

        sig_str = "*" if obs_p < 0.05 else ""
        print(f"  {cancer_type}: d={d:.3f}, obs_p={obs_p:.4f}{sig_str}, "
              f"empirical_p={perm['empirical_p']:.4f} (n_mut={len(mut_vals)})")

    perm_df = pd.DataFrame(perm_results)
    perm_df.to_csv(OUTPUT_DIR / "permutation_test_arid1b.csv", index=False)

    # Binomial test for the pattern
    print(f"\n  ARID1B nominal p<0.05 in {n_nominal_sig}/{len(perm_results)} cancer types")
    binom = binomial_pattern_test(n_nominal_sig, len(perm_results))
    print(f"  Expected by chance: {binom['expected_by_chance']}")
    print(f"  Binomial p-value: {binom['binomial_p']:.4e}")

    # ========================================
    # Part 3: Effect size verification
    # ========================================
    print("\n--- Part 3: Effect Size Verification ---\n")
    claimed = {
        ("Lymphoid", "ARID1B"): -2.26,
        ("Skin", "ARID1B"): -2.14,
        ("Breast", "ARID1B"): -1.86,
        ("Ovary/Fallopian Tube", "ARID1B"): -1.55,
        ("Esophagus/Stomach", "ARID1B"): -1.39,
        ("Uterus", "ARID1B"): -1.23,
        ("Breast", "EZH2"): -0.68,
        ("Breast", "HMGCR"): -1.68,
        ("Uterus", "HMGCR"): -0.78,
        ("Lung", "HMGCR"): -0.75,
        ("Biliary Tract", "ADCK5"): -1.96,
        ("Skin", "ADCK5"): -1.35,
        ("Pancreas", "ADCK5"): -0.92,
        ("Uterus", "ADCK5"): -0.68,
        ("Ovary/Fallopian Tube", "MDM2"): -1.43,
        ("Esophagus/Stomach", "MDM2"): -1.33,
        ("Bowel", "MDM2"): -0.91,
    }

    verification = []
    all_match = True
    for (ct, gene), expected in claimed.items():
        ct_data = merged[merged["OncotreeLineage"] == ct]
        mutant = ct_data[ct_data["ARID1A_status"] == "mutant"]
        wt = ct_data[ct_data["ARID1A_status"] == "WT"]

        if gene not in crispr.columns:
            print(f"  {ct}/{gene}: GENE NOT IN CRISPR DATA")
            verification.append({"cancer_type": ct, "gene": gene, "expected": expected,
                                 "computed": None, "match": False})
            all_match = False
            continue

        mut_vals = mutant[gene].dropna().values
        wt_vals = wt[gene].dropna().values

        if len(mut_vals) < 2 or len(wt_vals) < 2:
            print(f"  {ct}/{gene}: insufficient data")
            verification.append({"cancer_type": ct, "gene": gene, "expected": expected,
                                 "computed": None, "match": False})
            all_match = False
            continue

        computed = cohens_d(mut_vals, wt_vals)
        match = abs(computed - expected) < 0.02
        status = "OK" if match else "MISMATCH"
        print(f"  {ct}/{gene}: expected={expected:.2f}, computed={computed:.3f} [{status}]")
        verification.append({"cancer_type": ct, "gene": gene, "expected": expected,
                             "computed": round(computed, 4), "match": match})
        if not match:
            all_match = False

    pd.DataFrame(verification).to_csv(OUTPUT_DIR / "effect_size_verification.csv", index=False)
    print(f"\n  All effect sizes verified: {all_match}")

    # ========================================
    # Part 4: Summary
    # ========================================
    print("\n=== Validation Summary ===\n")

    n_loo_unstable = len(loo_df[(loo_df.get("stable") == False) & (loo_df.get("skipped") == False)]) if "stable" in loo_df.columns else 0
    critical_unstable = loo_df[
        (loo_df.get("stable") == False) &
        (loo_df.get("skipped") == False) &
        (loo_df["gene"].isin(["ARID1B", "EZH2"]))
    ] if "stable" in loo_df.columns else pd.DataFrame()

    summary_dict = {
        "loo_tests": n_tested,
        "loo_unstable": n_loo_unstable,
        "loo_critical_unstable": len(critical_unstable),
        "arid1b_nominal_sig_types": n_nominal_sig,
        "arid1b_binomial_p": binom["binomial_p"],
        "effect_sizes_verified": all_match,
        "pass": (
            len(critical_unstable) == 0
            and binom["binomial_p"] < 0.05
            and all_match
        ),
    }

    with open(OUTPUT_DIR / "validation_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    for k, v in summary_dict.items():
        print(f"  {k}: {v}")

    if summary_dict["pass"]:
        print("\n  RESULT: VALIDATION PASSED")
    else:
        print("\n  RESULT: VALIDATION ISSUES DETECTED")
        if len(critical_unstable) > 0:
            print("    - Critical LOO instability in ARID1B/EZH2:")
            for _, row in critical_unstable.iterrows():
                print(f"      {row['cancer_type']}/{row['gene']}: "
                      f"flips={row['direction_flips']}, lost>50%={row['lost_gt50pct']}")
        if binom["binomial_p"] >= 0.05:
            print(f"    - ARID1B binomial pattern not significant (p={binom['binomial_p']:.4f})")
        if not all_match:
            print("    - Effect size mismatches detected")


if __name__ == "__main__":
    main()
