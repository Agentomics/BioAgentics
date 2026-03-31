"""Phase 1b: Compute pairwise co-alteration frequency matrix and rank pairs.

Using the binary alteration matrix from Phase 1a, computes all pairwise
co-alteration frequencies for the 11 target genes (BRCA1/2 combined = 11 genes,
66 pairwise combinations). For each pair: co-occurrence count, Fisher exact test
for co-occurrence vs mutual exclusivity, and composite ranking by frequency,
actionability, and DepMap representativeness.

Output:
  - phase1_coalteration_matrix.csv: all pairwise combinations with stats
  - phase1_coalteration_summary.txt: human-readable summary

Usage:
    uv run python -m cancer.tsg_co_alteration_dependency_interactions.02_coalteration_frequency_matrix
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "tsg-co-alteration-dependency-interactions"

# Genes for pairwise analysis (BRCA1/2 combined into one group)
ANALYSIS_GENES = [
    "MTAP", "ARID1A", "SMARCA4", "PIK3CA", "PTEN", "RB1",
    "BRCA1_2", "TP53", "KEAP1", "NF1", "CDKN2A",
]

# Drug targets from published atlases — genes where both partners have
# atlas-identified drug targets score higher on clinical actionability.
DRUGGABLE_GENES = {
    "MTAP": "PRMT5i (JNJ-64619178, GSK3326595)",
    "ARID1A": "EZH2i (tazemetostat), statins",
    "PIK3CA": "PI3Kalpha-i (inavolisib, alpelisib)",
    "PTEN": "AKTi (capivasertib), PI3Kbeta-i",
    "RB1": "CDK2i (PF-07104091), CHK1i",
    "BRCA1_2": "PARPi (olaparib, talazoparib)",
    "TP53": "MDM2i (KT-253), WEE1i, p53 reactivators",
    "NF1": "MEKi (trametinib)",
    "KEAP1": "NRF2 pathway inhibitors (investigational)",
    "CDKN2A": "CDK4/6i (palbociclib)",
    "SMARCA4": "CDK4/6i, EZH2i (investigational)",
}

# Estimated DepMap double-mutant line availability (rough estimates from atlas work)
DEPMAP_AVAILABILITY = {
    ("TP53", "PTEN"): 5,
    ("TP53", "RB1"): 4,
    ("TP53", "PIK3CA"): 4,
    ("TP53", "ARID1A"): 4,
    ("TP53", "NF1"): 3,
    ("TP53", "BRCA1_2"): 3,
    ("TP53", "CDKN2A"): 4,
    ("TP53", "MTAP"): 4,
    ("PTEN", "RB1"): 3,
    ("ARID1A", "PIK3CA"): 3,
    ("CDKN2A", "MTAP"): 5,  # 9p21 co-deletion
}


def load_alteration_matrix() -> pd.DataFrame:
    """Load binary alteration matrix from Phase 1a and add combined BRCA1/2."""
    df = pd.read_csv(OUTPUT_DIR / "phase1_alteration_matrix.csv")
    df["BRCA1_2"] = ((df["BRCA1"] == 1) | (df["BRCA2"] == 1)).astype(int)
    return df


def compute_pairwise_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise co-alteration stats for all gene pairs."""
    n_total = len(df)
    rows = []

    for g1, g2 in combinations(ANALYSIS_GENES, 2):
        a = df[g1].values
        b = df[g2].values

        co_alt = int(((a == 1) & (b == 1)).sum())
        g1_only = int(((a == 1) & (b == 0)).sum())
        g2_only = int(((a == 0) & (b == 1)).sum())
        neither = int(((a == 0) & (b == 0)).sum())

        # Fisher exact test: co-occurrence vs independence
        contingency = [[co_alt, g1_only], [g2_only, neither]]
        odds_ratio, fisher_p = fisher_exact(contingency)

        # Co-occurrence rate: fraction of the smaller group also altered in the other
        n_g1 = g1_only + co_alt
        n_g2 = g2_only + co_alt
        smaller = min(n_g1, n_g2)
        co_rate_of_smaller = co_alt / smaller if smaller > 0 else 0.0

        # Clinical actionability: both genes druggable?
        both_druggable = g1 in DRUGGABLE_GENES and g2 in DRUGGABLE_GENES
        actionability_score = 2 if both_druggable else (1 if g1 in DRUGGABLE_GENES or g2 in DRUGGABLE_GENES else 0)

        # DepMap availability estimate
        pair_key = tuple(sorted([g1, g2]))
        depmap_est = DEPMAP_AVAILABILITY.get(pair_key, 1)

        rows.append({
            "gene_1": g1,
            "gene_2": g2,
            "co_altered": co_alt,
            "gene_1_only": g1_only,
            "gene_2_only": g2_only,
            "neither": neither,
            "n_gene_1": n_g1,
            "n_gene_2": n_g2,
            "co_alt_pct": round(co_alt / n_total * 100, 3),
            "co_rate_of_smaller": round(co_rate_of_smaller * 100, 2),
            "odds_ratio": round(odds_ratio, 4),
            "fisher_p": fisher_p,
            "tendency": "co-occurrence" if odds_ratio > 1 else "mutual_exclusivity",
            "both_druggable": both_druggable,
            "actionability_score": actionability_score,
            "depmap_est_score": depmap_est,
        })

    result = pd.DataFrame(rows)

    # Composite rank: weighted combination of frequency, actionability, DepMap availability
    result["freq_rank"] = result["co_altered"].rank(ascending=False)
    result["action_rank"] = result["actionability_score"].rank(ascending=False, method="min")
    result["depmap_rank"] = result["depmap_est_score"].rank(ascending=False, method="min")
    result["composite_rank"] = (
        result["freq_rank"] * 0.5
        + result["action_rank"] * 0.3
        + result["depmap_rank"] * 0.2
    )
    result = result.sort_values("composite_rank").reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result


def write_summary(df_matrix: pd.DataFrame, pairs_df: pd.DataFrame) -> None:
    """Write human-readable summary."""
    n_total = len(df_matrix)
    lines = []
    lines.append("=" * 70)
    lines.append("TSG Co-Alteration Dependency Interactions — Phase 1b Summary")
    lines.append("=" * 70)
    lines.append(f"\nTotal patients: {n_total}")
    lines.append(f"Cancer types:   {df_matrix['cancer_type'].nunique()}")
    lines.append(f"Gene pairs:     {len(pairs_df)}")
    lines.append("")

    # Gene alteration frequencies
    lines.append("Per-gene alteration frequencies:")
    for gene in ANALYSIS_GENES:
        n = int(df_matrix[gene].sum())
        pct = n / n_total * 100
        lines.append(f"  {gene:10s}  {n:5d} ({pct:5.1f}%)")
    lines.append("")

    # Top 15 pairs
    lines.append("Top 15 co-alteration pairs (composite rank):")
    lines.append(f"{'Rank':>4s}  {'Gene1':>10s}  {'Gene2':>10s}  {'Co-alt':>6s}  "
                 f"{'%total':>6s}  {'%smaller':>8s}  {'OR':>7s}  {'Fisher p':>10s}  "
                 f"{'Tendency':>16s}  {'Druggable':>9s}")
    lines.append("-" * 100)
    for _, row in pairs_df.head(15).iterrows():
        lines.append(
            f"{int(row['rank']):4d}  {row['gene_1']:>10s}  {row['gene_2']:>10s}  "
            f"{int(row['co_altered']):6d}  {row['co_alt_pct']:6.2f}  "
            f"{row['co_rate_of_smaller']:7.1f}%  {row['odds_ratio']:7.2f}  "
            f"{row['fisher_p']:10.2e}  {row['tendency']:>16s}  "
            f"{'Yes' if row['both_druggable'] else 'No':>9s}"
        )
    lines.append("")

    # Validation: top 5 should include TP53+PTEN, TP53+RB1
    top5_pairs = set()
    for _, row in pairs_df.head(5).iterrows():
        top5_pairs.add((row["gene_1"], row["gene_2"]))

    tp53_pten = ("PTEN", "TP53") in top5_pairs or ("TP53", "PTEN") in top5_pairs
    tp53_rb1 = ("RB1", "TP53") in top5_pairs or ("TP53", "RB1") in top5_pairs
    lines.append("Validation:")
    lines.append(f"  TP53+PTEN in top 5: {'PASS' if tp53_pten else 'FAIL'}")
    lines.append(f"  TP53+RB1 in top 5:  {'PASS' if tp53_rb1 else 'FAIL'}")

    text = "\n".join(lines)
    (OUTPUT_DIR / "phase1_coalteration_summary.txt").write_text(text)
    print(text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1b: Pairwise Co-Alteration Frequency Matrix ===\n")

    # Load Phase 1a data
    print("Loading alteration matrix from Phase 1a...")
    df = load_alteration_matrix()
    print(f"  {len(df)} patients, {df['cancer_type'].nunique()} cancer types\n")

    # Compute pairwise stats
    print("Computing pairwise co-alteration statistics...")
    pairs = compute_pairwise_stats(df)

    # Save
    out_path = OUTPUT_DIR / "phase1_coalteration_matrix.csv"
    pairs.to_csv(out_path, index=False)
    print(f"Saved co-alteration matrix: {out_path}\n")

    # Summary
    write_summary(df, pairs)

    print("\nDone.")


if __name__ == "__main__":
    main()
