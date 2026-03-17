"""Phase 3 Step 2: SL enrichment analysis.

Cross-reference 223 subtype-significant genes against 1,476 validated SL
benchmark pairs. Calculate enrichment per subtype, check positive controls,
and identify anti-correlated dependency pairs as candidate novel SL pairs.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


DATA_DIR = Path("data/results")
OUT_DIR = DATA_DIR / "sl_enrichment"


def load_data():
    """Load all input datasets."""
    sig_genes = (
        pd.read_csv(DATA_DIR / "subtype_dependencies" / "significant_genes.txt", header=None)[0]
        .tolist()
    )
    predictable_genes = (
        pd.read_csv(DATA_DIR / "predictable_genes.txt", header=None)[0].tolist()
    )
    sl_bench = pd.read_csv(DATA_DIR / "sl_benchmark_combined.csv")
    pairwise = pd.read_csv(DATA_DIR / "subtype_dependencies" / "pairwise_comparisons.csv")
    pred_deps = pd.read_csv(DATA_DIR / "tcga_predicted_dependencies.csv", index_col=0)
    # Map UUID index to TCGA patient barcodes
    uuid_map_path = Path("data/tcga/luad/uuid_patient_map.json")
    with open(uuid_map_path) as f:
        uuid_map = json.load(f)
    pred_deps.index = pred_deps.index.map(lambda x: uuid_map.get(x, x))
    metadata = pd.read_csv(DATA_DIR / "tcga_patient_metadata.csv")
    return sig_genes, predictable_genes, sl_bench, pairwise, pred_deps, metadata


def assign_subtype_genes(sig_genes, pairwise):
    """Assign each significant gene to the subtype(s) where it shows highest dependency.

    A gene is 'subtype-enriched' if it has significantly higher dependency
    (FDR < 0.05, direction=a_higher) in that subtype vs the majority of others.
    """
    subtypes = ["KL", "KP", "KOnly", "KRAS-WT"]
    gene_subtypes = {s: set() for s in subtypes}

    for gene in sig_genes:
        gene_pw = pairwise[pairwise["gene"] == gene]
        if gene_pw.empty:
            continue

        # For each subtype, count significant wins (a_higher with FDR < 0.05)
        for subtype in subtypes:
            # Comparisons where this subtype is group_a and it's higher
            wins_a = gene_pw[
                (gene_pw["group_a"] == subtype)
                & (gene_pw["direction"] == "a_higher")
                & (gene_pw["fdr"] < 0.05)
            ]
            # Comparisons where this subtype is group_b and it's higher (b_higher)
            wins_b = gene_pw[
                (gene_pw["group_b"] == subtype)
                & (gene_pw["direction"] == "b_higher")
                & (gene_pw["fdr"] < 0.05)
            ]
            n_wins = len(wins_a) + len(wins_b)
            # Enriched if significantly higher than at least 2 other subtypes
            if n_wins >= 2:
                gene_subtypes[subtype].add(gene)

    return gene_subtypes


def cross_reference_sl(gene_subtypes, sl_bench, sig_genes):
    """Cross-reference subtype genes against SL benchmark pairs."""
    all_sl_genes = set(sl_bench["gene_a"].unique()) | set(sl_bench["gene_b"].unique())
    sig_set = set(sig_genes)

    rows = []
    for subtype, genes in gene_subtypes.items():
        for gene in genes:
            # Check if gene appears as either partner in SL pairs
            matches_a = sl_bench[sl_bench["gene_a"] == gene]
            matches_b = sl_bench[sl_bench["gene_b"] == gene]

            for _, row in matches_a.iterrows():
                rows.append({
                    "subtype": subtype,
                    "gene": gene,
                    "sl_partner": row["gene_b"],
                    "sources": row["sources"],
                    "confidence_tier": row["confidence_tier"],
                    "driver_context": row.get("driver_context", ""),
                    "n_sources": row["n_sources"],
                    "partner_in_sig_genes": row["gene_b"] in sig_set,
                })
            for _, row in matches_b.iterrows():
                rows.append({
                    "subtype": subtype,
                    "gene": gene,
                    "sl_partner": row["gene_a"],
                    "sources": row["sources"],
                    "confidence_tier": row["confidence_tier"],
                    "driver_context": row.get("driver_context", ""),
                    "n_sources": row["n_sources"],
                    "partner_in_sig_genes": row["gene_a"] in sig_set,
                })

    hits_df = pd.DataFrame(rows)
    if not hits_df.empty:
        hits_df = hits_df.drop_duplicates(subset=["subtype", "gene", "sl_partner"])
    return hits_df, all_sl_genes


def fisher_enrichment(gene_subtypes, sl_bench, sig_genes, predictable_genes):
    """Fisher's exact test: are subtype-enriched genes over-represented in SL databases?"""
    all_sl_genes = set(sl_bench["gene_a"].unique()) | set(sl_bench["gene_b"].unique())
    background = set(predictable_genes)

    rows = []
    for subtype, genes in gene_subtypes.items():
        genes_in_sl = len(genes & all_sl_genes)
        genes_not_in_sl = len(genes - all_sl_genes)
        bg_in_sl = len((background - genes) & all_sl_genes)
        bg_not_in_sl = len((background - genes) - all_sl_genes)

        table = [[genes_in_sl, genes_not_in_sl],
                 [bg_in_sl, bg_not_in_sl]]
        odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")

        rows.append({
            "subtype": subtype,
            "n_subtype_genes": len(genes),
            "n_in_sl_db": genes_in_sl,
            "n_not_in_sl_db": genes_not_in_sl,
            "bg_in_sl": bg_in_sl,
            "bg_not_in_sl": bg_not_in_sl,
            "odds_ratio": odds_ratio,
            "fisher_p": p_value,
        })

    enrich_df = pd.DataFrame(rows)
    # FDR correction
    if len(enrich_df) > 0:
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(enrich_df["fisher_p"], method="fdr_bh")
        enrich_df["fisher_fdr"] = fdr
    return enrich_df


def check_positive_controls(hits_df, sl_bench, sig_genes):
    """Check for priority positive control SL pairs."""
    controls = [
        ("STK11", "MARK2"), ("STK11", "MARK3"),
        ("KEAP1", "TGIF1"), ("KEAP1", "TFRC"), ("KEAP1", "NCOA4"),
    ]
    sig_set = set(sig_genes)
    results = []
    for gene_a, gene_b in controls:
        # Check if pair exists in SL benchmark
        in_bench = sl_bench[
            ((sl_bench["gene_a"] == gene_a) & (sl_bench["gene_b"] == gene_b))
            | ((sl_bench["gene_a"] == gene_b) & (sl_bench["gene_b"] == gene_a))
        ]
        in_benchmark = len(in_bench) > 0
        a_in_sig = gene_a in sig_set
        b_in_sig = gene_b in sig_set

        # Check if found in subtype hits
        in_hits = False
        hit_subtypes = []
        if not hits_df.empty:
            hit_rows = hits_df[
                ((hits_df["gene"] == gene_a) & (hits_df["sl_partner"] == gene_b))
                | ((hits_df["gene"] == gene_b) & (hits_df["sl_partner"] == gene_a))
            ]
            in_hits = len(hit_rows) > 0
            if in_hits:
                hit_subtypes = hit_rows["subtype"].unique().tolist()

        results.append({
            "gene_a": gene_a,
            "gene_b": gene_b,
            "in_sl_benchmark": in_benchmark,
            "gene_a_in_sig_genes": a_in_sig,
            "gene_b_in_sig_genes": b_in_sig,
            "found_in_subtype_hits": in_hits,
            "hit_subtypes": ";".join(hit_subtypes) if hit_subtypes else "",
            "benchmark_sources": in_bench["sources"].iloc[0] if in_benchmark else "",
        })
    return pd.DataFrame(results)


def find_anticorrelated_pairs(pred_deps, metadata, predictable_genes, top_n=100):
    """Find anti-correlated dependency pairs within each subtype.

    Gene A has high dependency and gene B has low dependency in the same
    patients — candidates for novel SL interactions.
    """
    subtypes = ["KL", "KP", "KOnly", "KRAS-WT"]
    meta = metadata[["patient_id", "molecular_subtype"]].drop_duplicates()

    # Filter to predictable genes that are in the matrix
    available_genes = [g for g in predictable_genes if g in pred_deps.columns]
    dep_matrix = pred_deps[available_genes]

    all_pairs = []
    for subtype in subtypes:
        subtype_patients = meta[meta["molecular_subtype"] == subtype]["patient_id"]
        sub_deps = dep_matrix.loc[dep_matrix.index.isin(subtype_patients)]

        if len(sub_deps) < 10:
            continue

        # Compute pairwise correlation among genes (use Spearman for robustness)
        # For efficiency, compute on means and find anti-correlated via correlation matrix
        corr_matrix = sub_deps.corr(method="spearman")

        # Extract lower triangle anti-correlated pairs
        genes = corr_matrix.columns.tolist()
        pairs = []
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                r = corr_matrix.iloc[i, j]
                if r < -0.15:  # relaxed threshold to capture top candidates
                    pairs.append({
                        "subtype": subtype,
                        "gene_a": genes[i],
                        "gene_b": genes[j],
                        "spearman_r": r,
                        "n_patients": len(sub_deps),
                    })

        all_pairs.extend(pairs)

    pairs_df = pd.DataFrame(all_pairs)
    if not pairs_df.empty:
        # Add p-values for top pairs
        enriched_pairs = []
        for _, row in pairs_df.iterrows():
            subtype_patients = meta[meta["molecular_subtype"] == row["subtype"]]["patient_id"]
            sub_deps_local = dep_matrix.loc[dep_matrix.index.isin(subtype_patients)]
            r, p = stats.spearmanr(
                sub_deps_local[row["gene_a"]],
                sub_deps_local[row["gene_b"]],
            )
            enriched_pairs.append({**row.to_dict(), "spearman_p": p})

        pairs_df = pd.DataFrame(enriched_pairs)
        pairs_df = pairs_df.sort_values("spearman_r").head(top_n)

    return pairs_df


def main():
    print("Loading data...")
    sig_genes, predictable_genes, sl_bench, pairwise, pred_deps, metadata = load_data()

    print(f"Significant genes: {len(sig_genes)}")
    print(f"Predictable genes: {len(predictable_genes)}")
    print(f"SL benchmark pairs: {len(sl_bench)}")

    print("\nAssigning subtype-enriched genes...")
    gene_subtypes = assign_subtype_genes(sig_genes, pairwise)
    for s, g in gene_subtypes.items():
        print(f"  {s}: {len(g)} enriched genes")

    print("\nCross-referencing against SL benchmarks...")
    hits_df, all_sl_genes = cross_reference_sl(gene_subtypes, sl_bench, sig_genes)
    print(f"  Total SL hits: {len(hits_df)}")

    print("\nFisher's exact enrichment test...")
    enrich_df = fisher_enrichment(gene_subtypes, sl_bench, sig_genes, predictable_genes)
    print(enrich_df[["subtype", "n_subtype_genes", "n_in_sl_db", "odds_ratio", "fisher_p"]].to_string(index=False))

    print("\nChecking positive controls...")
    controls_df = check_positive_controls(hits_df, sl_bench, sig_genes)
    print(controls_df.to_string(index=False))

    print("\nFinding anti-correlated dependency pairs...")
    anticorr_df = find_anticorrelated_pairs(pred_deps, metadata, predictable_genes)
    print(f"  Anti-correlated pairs (r < -0.15): {len(anticorr_df)}")
    if not anticorr_df.empty:
        for subtype in anticorr_df["subtype"].unique():
            n = len(anticorr_df[anticorr_df["subtype"] == subtype])
            print(f"    {subtype}: {n} pairs")

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hits_df.to_csv(OUT_DIR / "subtype_sl_hits.csv", index=False)
    enrich_df.to_csv(OUT_DIR / "enrichment_stats.csv", index=False)
    anticorr_df.to_csv(OUT_DIR / "anticorrelated_pairs.csv", index=False)
    controls_df.to_csv(OUT_DIR / "positive_controls.csv", index=False)

    print(f"\nOutputs saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
