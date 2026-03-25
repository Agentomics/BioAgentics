"""Task #1082: Pathway enrichment statistical validation.

Validates KEGG and Reactome pathway enrichment with independent statistical
tests, FDR correction, effect sizes, and cross-database concordance analysis.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path

DATA_DIR = Path("data/pandas_pans/autoantibody_network")
RESULTS_DIR = DATA_DIR  # output alongside input data

# Approximate human protein-coding genome size for background
GENOME_BACKGROUND = 20_000


def _enrichment_stats(k: int, K: int, n: int, N: int) -> dict:
    """Compute hypergeometric/Fisher enrichment statistics for one pathway.

    Parameters: k=overlap, K=pathway size, n=network size, N=genome background.
    """
    hyper_pval = stats.hypergeom.sf(k - 1, N, K, n)

    a, b = k, n - k
    c = max(K - k, 0)
    d = max(N - n - c, 0)
    _, fisher_pval = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

    if b > 0 and c > 0 and d > 0:
        odds_ratio = (a * d) / (b * c)
    else:
        odds_ratio = np.inf

    expected = (K * n) / N
    fold_enrich = k / expected if expected > 0 else np.inf

    return {
        "expected_genes": round(expected, 2),
        "fold_enrichment_validated": round(fold_enrich, 3),
        "hypergeometric_pval": hyper_pval,
        "fisher_pval": fisher_pval,
        "odds_ratio": round(odds_ratio, 3) if odds_ratio != np.inf else "Inf",
    }


def _apply_fdr(result_df: pd.DataFrame) -> pd.DataFrame:
    """Add BH FDR columns and lost-significance flag to a validation result."""
    reject_hyper, fdr_hyper, _, _ = multipletests(
        result_df["hypergeometric_pval"].clip(lower=1e-300), method="fdr_bh"
    )
    reject_fisher, fdr_fisher, _, _ = multipletests(
        result_df["fisher_pval"].clip(lower=1e-300), method="fdr_bh"
    )
    result_df["hypergeometric_fdr"] = fdr_hyper
    result_df["fisher_fdr"] = fdr_fisher
    result_df["significant_hyper_fdr05"] = reject_hyper
    result_df["significant_fisher_fdr05"] = reject_fisher
    result_df["lost_significance"] = (result_df["original_fdr"] < 0.05) & ~reject_hyper
    return result_df


def validate_kegg_enrichment():
    """Re-compute KEGG enrichment with Fisher exact test and hypergeometric test."""
    df = pd.read_csv(DATA_DIR / "kegg_pathway_enrichment.tsv", sep="\t")

    with open(DATA_DIR / "kegg_mapping_stats.json") as f:
        stats_info = json.load(f)

    network_size = stats_info["nodes_mapped_to_kegg"]

    results = []
    for _, row in df.iterrows():
        k = int(row["network_genes"])
        K = int(row["background_genes"])
        rec = {
            "pathway_id": row["pathway_id"],
            "pathway_name": row["pathway_name"],
            "network_genes": k,
            "background_genes": K,
            "network_size": network_size,
            "genome_background": GENOME_BACKGROUND,
            **_enrichment_stats(k, K, network_size, GENOME_BACKGROUND),
            "fold_enrichment_original": round(row["fold_enrichment"], 3),
            "original_pval": row["pvalue"],
            "original_fdr": row["fdr"],
            "seed_proteins_in_pathway": row["seed_proteins_in_pathway"],
            "seed_list": row["seed_list"],
            "is_focus_pathway": row["is_focus_pathway"],
        }
        results.append(rec)

    return _apply_fdr(pd.DataFrame(results))


def validate_reactome_enrichment():
    """Re-compute Reactome enrichment with Fisher exact test and hypergeometric test."""
    df = pd.read_csv(DATA_DIR / "reactome_pathway_enrichment.tsv", sep="\t")

    with open(DATA_DIR / "reactome_mapping_stats.json") as f:
        stats_info = json.load(f)

    network_size = stats_info["genes_mapped_to_pathways"]

    results = []
    for _, row in df.iterrows():
        k = int(row["network_genes"])
        K = int(row["pathway_total_genes"])
        rec = {
            "pathway_id": row["pathway_id"],
            "pathway_name": row["pathway_name"],
            "network_genes": k,
            "pathway_total_genes": K,
            "network_size": network_size,
            "genome_background": GENOME_BACKGROUND,
            **_enrichment_stats(k, K, network_size, GENOME_BACKGROUND),
            "original_pval": row["pvalue"],
            "original_fdr": row["fdr"],
            "is_focus_pathway": row["is_focus_pathway"],
        }
        results.append(rec)

    return _apply_fdr(pd.DataFrame(results))


def seed_neighborhood_enrichment(kegg_node_map, extended_net):
    """Test if pathways are enriched specifically in seed-protein neighborhoods."""
    seeds = extended_net[extended_net["source"].isin(
        ["DRD1", "DRD2", "CAMK2A", "TUBB3", "PKM", "ALDOC", "ENO1", "ENO2", "ENO3"]
    ) | extended_net["target"].isin(
        ["DRD1", "DRD2", "CAMK2A", "TUBB3", "PKM", "ALDOC", "ENO1", "ENO2", "ENO3"]
    )]
    seed_neighbors = set(seeds["source"].unique()) | set(seeds["target"].unique())

    all_network_genes = set(kegg_node_map["gene_symbol"].unique())
    non_seed_genes = all_network_genes - seed_neighbors

    results = []
    for pathway_id, grp in kegg_node_map.groupby("pathway_id"):
        pathway_genes = set(grp["gene_symbol"].unique())
        seed_neigh_in_pw = len(pathway_genes & seed_neighbors)
        non_seed_in_pw = len(pathway_genes & non_seed_genes)
        seed_neigh_not_pw = len(seed_neighbors - pathway_genes)
        non_seed_not_pw = len(non_seed_genes - pathway_genes)

        if seed_neigh_in_pw + non_seed_in_pw == 0:
            continue

        _, pval = stats.fisher_exact(
            [[seed_neigh_in_pw, seed_neigh_not_pw],
             [non_seed_in_pw, non_seed_not_pw]],
            alternative="greater"
        )

        pathway_name = grp["pathway_name"].iloc[0]
        results.append({
            "pathway_id": pathway_id,
            "pathway_name": pathway_name,
            "seed_neighborhood_genes": seed_neigh_in_pw,
            "non_seed_genes": non_seed_in_pw,
            "total_pathway_genes_in_network": seed_neigh_in_pw + non_seed_in_pw,
            "fisher_pval_seed_enrichment": pval,
        })

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        _, fdr, _, _ = multipletests(
            result_df["fisher_pval_seed_enrichment"].clip(lower=1e-300), method="fdr_bh"
        )
        result_df["seed_enrichment_fdr"] = fdr
        result_df["seed_specific"] = fdr < 0.05
    return result_df


def kegg_reactome_concordance(kegg_validated, reactome_validated):
    """Assess concordance between KEGG and Reactome significant pathways."""
    kegg_sig = set(
        kegg_validated[kegg_validated["significant_hyper_fdr05"]]["pathway_name"]
        .str.lower().str.strip()
    )
    reactome_sig = set(
        reactome_validated[reactome_validated["significant_hyper_fdr05"]]["pathway_name"]
        .str.lower().str.strip()
    )

    # Keyword-based matching (pathways have different naming conventions)
    keywords = [
        "dopamin", "calcium", "jak-stat", "jak stat", "interleukin", "interferon",
        "mapk", "pi3k", "akt", "camp", "chemokine", "cytokine", "synap",
        "glutamat", "gaba", "serotonin", "cholinerg", "nmda", "potentiation",
        "depression", "axon", "folate", "mtor", "toll", "nf-kb", "nfkb",
        "notch", "wnt", "hedgehog", "apoptosis", "autophagy"
    ]

    concordance = {}
    for kw in keywords:
        kegg_has = any(kw in p for p in kegg_sig)
        reactome_has = any(kw in p for p in reactome_sig)
        if kegg_has or reactome_has:
            concordance[kw] = {
                "kegg_significant": kegg_has,
                "reactome_significant": reactome_has,
                "concordant": kegg_has == reactome_has,
            }

    return concordance


def main():
    print("=" * 60)
    print("Task #1082: Pathway Enrichment Statistical Validation")
    print("=" * 60)

    # 1. Validate KEGG
    print("\n[1/5] Validating KEGG pathway enrichment...")
    kegg_validated = validate_kegg_enrichment()
    kegg_validated.to_csv(RESULTS_DIR / "kegg_enrichment_validated.tsv", sep="\t", index=False)

    kegg_sig_orig = (kegg_validated["original_fdr"] < 0.05).sum()
    kegg_sig_hyper = kegg_validated["significant_hyper_fdr05"].sum()
    kegg_lost = kegg_validated["lost_significance"].sum()

    print(f"  KEGG pathways tested: {len(kegg_validated)}")
    print(f"  Originally significant (FDR<0.05): {kegg_sig_orig}")
    print(f"  Validated significant (hypergeometric FDR<0.05): {kegg_sig_hyper}")
    print(f"  Lost significance after revalidation: {kegg_lost}")

    # 2. Validate Reactome
    print("\n[2/5] Validating Reactome pathway enrichment...")
    reactome_validated = validate_reactome_enrichment()
    reactome_validated.to_csv(RESULTS_DIR / "reactome_enrichment_validated.tsv", sep="\t", index=False)

    react_sig_orig = (reactome_validated["original_fdr"] < 0.05).sum()
    react_sig_hyper = reactome_validated["significant_hyper_fdr05"].sum()
    react_lost = reactome_validated["lost_significance"].sum()

    print(f"  Reactome pathways tested: {len(reactome_validated)}")
    print(f"  Originally significant (FDR<0.05): {react_sig_orig}")
    print(f"  Validated significant (hypergeometric FDR<0.05): {react_sig_hyper}")
    print(f"  Lost significance after revalidation: {react_lost}")

    # 3. Seed neighborhood enrichment
    print("\n[3/5] Testing seed-protein neighborhood enrichment...")
    kegg_node_map = pd.read_csv(DATA_DIR / "kegg_node_pathway_mapping.tsv", sep="\t")
    extended_net = pd.read_csv(DATA_DIR / "extended_network.tsv", sep="\t")
    seed_enrich = seed_neighborhood_enrichment(kegg_node_map, extended_net)
    seed_enrich.to_csv(RESULTS_DIR / "seed_neighborhood_enrichment.tsv", sep="\t", index=False)

    seed_specific_count = seed_enrich["seed_specific"].sum() if "seed_specific" in seed_enrich else 0
    print(f"  Pathways tested: {len(seed_enrich)}")
    print(f"  Seed-specific pathways (FDR<0.05): {seed_specific_count}")

    # 4. KEGG-Reactome concordance
    print("\n[4/5] Assessing KEGG-Reactome concordance...")
    concordance = kegg_reactome_concordance(kegg_validated, reactome_validated)

    concordant = sum(1 for v in concordance.values() if v["concordant"])
    total_kw = len(concordance)
    print(f"  Keyword themes tested: {total_kw}")
    pct = 100 * concordant / total_kw if total_kw > 0 else 0
    print(f"  Concordant: {concordant}/{total_kw} ({pct:.0f}%)")

    # 5. Focus pathway analysis
    print("\n[5/5] Focus pathway validation summary...")
    focus = kegg_validated[kegg_validated["is_focus_pathway"] == True].copy()
    focus_sig = focus["significant_hyper_fdr05"].sum()
    focus_lost = focus["lost_significance"].sum()
    print(f"  Focus pathways: {len(focus)}")
    print(f"  Still significant: {focus_sig}")
    print(f"  Lost significance: {focus_lost}")

    if focus_lost > 0:
        lost_names = focus[focus["lost_significance"]]["pathway_name"].tolist()
        print(f"  Lost: {lost_names}")

    # Top validated KEGG pathways by fold enrichment
    top_kegg = kegg_validated[kegg_validated["significant_hyper_fdr05"]].nlargest(
        15, "fold_enrichment_validated"
    )

    # Compile summary stats
    summary = {
        "kegg": {
            "total_tested": len(kegg_validated),
            "originally_significant_fdr05": int(kegg_sig_orig),
            "validated_significant_fdr05": int(kegg_sig_hyper),
            "lost_significance": int(kegg_lost),
            "gained_significance": int((~(kegg_validated["original_fdr"] < 0.05) & kegg_validated["significant_hyper_fdr05"]).sum()),
            "top_pathways": top_kegg[["pathway_name", "fold_enrichment_validated", "hypergeometric_fdr", "odds_ratio"]].to_dict("records")[:10],
        },
        "reactome": {
            "total_tested": len(reactome_validated),
            "originally_significant_fdr05": int(react_sig_orig),
            "validated_significant_fdr05": int(react_sig_hyper),
            "lost_significance": int(react_lost),
        },
        "seed_neighborhood": {
            "total_tested": len(seed_enrich),
            "seed_specific_fdr05": int(seed_specific_count),
        },
        "concordance": concordance,
        "focus_pathways": {
            "total": len(focus),
            "still_significant": int(focus_sig),
            "lost_significance": int(focus_lost),
            "details": focus[["pathway_name", "fold_enrichment_validated", "hypergeometric_fdr", "significant_hyper_fdr05"]].to_dict("records"),
        },
        "methods": {
            "background_genome_size": GENOME_BACKGROUND,
            "tests_used": ["hypergeometric (scipy.stats.hypergeom)", "Fisher exact (scipy.stats.fisher_exact)"],
            "multiple_testing_correction": "Benjamini-Hochberg FDR",
            "significance_threshold": 0.05,
        },
    }

    with open(RESULTS_DIR / "pathway_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print(f"Output files:")
    print(f"  - kegg_enrichment_validated.tsv")
    print(f"  - reactome_enrichment_validated.tsv")
    print(f"  - seed_neighborhood_enrichment.tsv")
    print(f"  - pathway_validation_summary.json")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    summary = main()
