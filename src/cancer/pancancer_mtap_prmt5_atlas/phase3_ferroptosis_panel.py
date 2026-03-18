"""Phase 3 Step 4: FSP1/ferroptosis gene panel validation for KEAP1 subtype.

Check ferroptosis/metabolic vulnerability genes for dependency enrichment
in KEAP1-mutant NSCLC using both predicted dependencies (TCGA) and raw
DepMap CRISPR scores in NSCLC cell lines.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


DATA_DIR = Path("data/results")
DEPMAP_DIR = Path("data/depmap/25q3")
OUT_DIR = DATA_DIR / "ferroptosis_panel"

FERROPTOSIS_GENES = {
    "AIFM2": "FSP1/CoQ-dependent ferroptosis suppressor. CRITICAL: Nature Jan 2026 showed "
              "FSP1 deletion suppresses tumorigenesis ~80% in vivo but NOT essential in vitro. "
              "icFSP1 inhibitor exists. Flag as priority target even if DepMap score is modest.",
    "GPX4": "GSH-dependent ferroptosis suppressor",
    "SAT1": "Polyamine catabolism -> ferroptosis",
    "GLS": "Glutaminase -> glutathione synthesis (GLS1)",
    "SLC7A11": "Cystine transporter -> glutathione (xCT)",
    "NCOA4": "Ferritinophagy receptor",
    "TFRC": "Transferrin receptor (iron import)",
    "SHMT1": "Serine hydroxymethyltransferase 1 (SIK-NRF2 axis)",
    "SHMT2": "Mitochondrial SHMT (SIK-NRF2 axis)",
}

NRF2_SIGNATURE = ["NQO1", "HMOX1", "GCLC", "GCLM", "TXNRD1"]
ASTUTE_SIGNATURE = ["SRXN1", "CABYR", "TRIM16"]


def load_data():
    """Load datasets."""
    # CRISPR dependency (raw gene effect)
    crispr = pd.read_csv(DEPMAP_DIR / "CRISPRGeneEffect.csv", index_col=0)

    # NSCLC cell lines with mutations
    nsclc_cl = pd.read_csv(DEPMAP_DIR / "nsclc_cell_lines_annotated.csv")

    # Predicted dependencies (TCGA)
    pred_deps = pd.read_csv(DATA_DIR / "tcga_predicted_dependencies.csv", index_col=0)
    uuid_map_path = Path("data/tcga/luad/uuid_patient_map.json")
    with open(uuid_map_path) as f:
        uuid_map = json.load(f)
    pred_deps.index = pred_deps.index.map(lambda x: uuid_map.get(x, x))

    # TCGA patient metadata
    metadata = pd.read_csv(DATA_DIR / "tcga_patient_metadata.csv")

    # Predictable genes
    predictable = set(
        pd.read_csv(DATA_DIR / "predictable_genes.txt", header=None)[0].tolist()
    )

    return crispr, nsclc_cl, pred_deps, metadata, predictable


def _find_crispr_col(crispr, gene):
    """Find CRISPR column for a gene (format: 'GENE (ENTREZID)')."""
    for col in crispr.columns:
        if col.split(" (")[0] == gene:
            return col
    return None


def analyze_crispr_by_mutation(crispr, nsclc_cl, genes):
    """Analyze CRISPR dependency scores for genes stratified by KEAP1/STK11 status."""
    nsclc_in_crispr = nsclc_cl[nsclc_cl["ModelID"].isin(crispr.index)].copy()

    # Define mutation groups
    nsclc_in_crispr["keap1_status"] = nsclc_in_crispr["KEAP1_mutated"].map(
        {True: "KEAP1-mut", False: "KEAP1-wt"}
    )
    nsclc_in_crispr["stk11_status"] = nsclc_in_crispr["STK11_mutated"].map(
        {True: "STK11-mut", False: "STK11-wt"}
    )

    rows = []
    for gene in genes:
        col = _find_crispr_col(crispr, gene)
        if col is None:
            continue

        for grouping, label in [("keap1_status", "KEAP1"), ("stk11_status", "STK11"),
                                 ("molecular_subtype", "subtype")]:
            for group_val, group_df in nsclc_in_crispr.groupby(grouping):
                model_ids = group_df["ModelID"].tolist()
                scores = crispr.loc[
                    crispr.index.isin(model_ids), col
                ].dropna()

                rows.append({
                    "gene": gene,
                    "annotation": FERROPTOSIS_GENES.get(gene, ""),
                    "grouping": label,
                    "group": group_val,
                    "n_cell_lines": len(scores),
                    "mean_dependency": scores.mean(),
                    "median_dependency": scores.median(),
                    "std_dependency": scores.std(),
                    "in_predictable_genes": gene in predictable_set,
                })

    return pd.DataFrame(rows)


def keap1_enrichment_tests(crispr, nsclc_cl, genes):
    """Mann-Whitney U test: KEAP1-mutant vs KEAP1-wt for each gene."""
    nsclc_in_crispr = nsclc_cl[nsclc_cl["ModelID"].isin(crispr.index)]
    keap1_mut_ids = nsclc_in_crispr[nsclc_in_crispr["KEAP1_mutated"] == True]["ModelID"]
    keap1_wt_ids = nsclc_in_crispr[nsclc_in_crispr["KEAP1_mutated"] == False]["ModelID"]
    stk11_mut_ids = nsclc_in_crispr[nsclc_in_crispr["STK11_mutated"] == True]["ModelID"]
    stk11_wt_ids = nsclc_in_crispr[nsclc_in_crispr["STK11_mutated"] == False]["ModelID"]

    rows = []
    for gene in genes:
        col = _find_crispr_col(crispr, gene)
        if col is None:
            continue

        # KEAP1 mut vs wt
        mut_scores = crispr.loc[crispr.index.isin(keap1_mut_ids), col].dropna()
        wt_scores = crispr.loc[crispr.index.isin(keap1_wt_ids), col].dropna()

        if len(mut_scores) >= 3 and len(wt_scores) >= 3:
            u_stat, p_val = stats.mannwhitneyu(mut_scores, wt_scores, alternative="two-sided")
            effect = mut_scores.mean() - wt_scores.mean()
            rows.append({
                "gene": gene,
                "comparison": "KEAP1_mut_vs_wt",
                "n_mut": len(mut_scores),
                "n_wt": len(wt_scores),
                "mean_mut": mut_scores.mean(),
                "mean_wt": wt_scores.mean(),
                "effect_size": effect,
                "direction": "more_dependent" if effect < 0 else "less_dependent",
                "u_stat": u_stat,
                "p_value": p_val,
            })

        # STK11 mut vs wt
        mut_scores_s = crispr.loc[crispr.index.isin(stk11_mut_ids), col].dropna()
        wt_scores_s = crispr.loc[crispr.index.isin(stk11_wt_ids), col].dropna()

        if len(mut_scores_s) >= 3 and len(wt_scores_s) >= 3:
            u_stat, p_val = stats.mannwhitneyu(mut_scores_s, wt_scores_s, alternative="two-sided")
            effect = mut_scores_s.mean() - wt_scores_s.mean()
            rows.append({
                "gene": gene,
                "comparison": "STK11_mut_vs_wt",
                "n_mut": len(mut_scores_s),
                "n_wt": len(wt_scores_s),
                "mean_mut": mut_scores_s.mean(),
                "mean_wt": wt_scores_s.mean(),
                "effect_size": effect,
                "direction": "more_dependent" if effect < 0 else "less_dependent",
                "u_stat": u_stat,
                "p_value": p_val,
            })

    enrich_df = pd.DataFrame(rows)
    if len(enrich_df) > 0:
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(enrich_df["p_value"], method="fdr_bh")
        enrich_df["fdr"] = fdr
    return enrich_df


def nrf2_signature_analysis(crispr, nsclc_cl):
    """Check NRF2 target gene signature across subtypes."""
    all_nrf2 = NRF2_SIGNATURE + ASTUTE_SIGNATURE
    nsclc_in_crispr = nsclc_cl[nsclc_cl["ModelID"].isin(crispr.index)]

    rows = []
    for gene in all_nrf2:
        col = _find_crispr_col(crispr, gene)
        if col is None:
            continue

        for subtype, sub_df in nsclc_in_crispr.groupby("molecular_subtype"):
            scores = crispr.loc[crispr.index.isin(sub_df["ModelID"]), col].dropna()
            rows.append({
                "gene": gene,
                "signature": "NRF2" if gene in NRF2_SIGNATURE else "ASTUTE",
                "molecular_subtype": subtype,
                "n_cell_lines": len(scores),
                "mean_dependency": scores.mean(),
                "median_dependency": scores.median(),
            })

    return pd.DataFrame(rows)


def predicted_dependency_subtype(pred_deps, metadata, genes, predictable):
    """Check predicted dependency for ferroptosis genes across TCGA subtypes."""
    rows = []
    for gene in genes:
        if gene not in predictable or gene not in pred_deps.columns:
            continue

        for subtype in ["KL", "KP", "KOnly", "KRAS-WT"]:
            pts = metadata[metadata["molecular_subtype"] == subtype]["patient_id"]
            vals = pred_deps.loc[pred_deps.index.isin(pts), gene].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "gene": gene,
                "source": "predicted_dependency",
                "subtype": subtype,
                "n_patients": len(vals),
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
            })

    return pd.DataFrame(rows)


# Module-level variable set during main()
predictable_set = set()


def main():
    global predictable_set
    print("Loading data...")
    crispr, nsclc_cl, pred_deps, metadata, predictable = load_data()
    predictable_set = predictable

    ferroptosis_genes = list(FERROPTOSIS_GENES.keys())
    print(f"NSCLC cell lines in CRISPR: {len(nsclc_cl[nsclc_cl['ModelID'].isin(crispr.index)])}")
    print(f"Ferroptosis panel: {len(ferroptosis_genes)} genes")
    in_pred = [g for g in ferroptosis_genes if g in predictable]
    print(f"In predictable genes: {in_pred}")

    print("\nCRISPR dependency by KEAP1/STK11/subtype...")
    dep_df = analyze_crispr_by_mutation(crispr, nsclc_cl, ferroptosis_genes)
    # Show KEAP1 grouping summary
    keap1_summary = dep_df[dep_df["grouping"] == "KEAP1"][
        ["gene", "group", "n_cell_lines", "mean_dependency"]
    ].pivot(index="gene", columns="group", values="mean_dependency")
    print(keap1_summary.to_string())

    print("\nKEAP1/STK11 enrichment tests...")
    enrich_df = keap1_enrichment_tests(crispr, nsclc_cl, ferroptosis_genes)
    sig = enrich_df[enrich_df["fdr"] < 0.05]
    print(f"  Significant (FDR<0.05): {len(sig)}")
    if not sig.empty:
        print(sig[["gene", "comparison", "effect_size", "direction", "p_value", "fdr"]].to_string(index=False))

    print("\nNRF2 signature analysis...")
    nrf2_df = nrf2_signature_analysis(crispr, nsclc_cl)
    # Pivot for readability
    nrf2_pivot = nrf2_df.pivot_table(
        index="gene", columns="molecular_subtype", values="mean_dependency"
    )
    print(nrf2_pivot.to_string())

    print("\nPredicted dependency (TCGA subtypes)...")
    pred_df = predicted_dependency_subtype(pred_deps, metadata, ferroptosis_genes, predictable)
    if not pred_df.empty:
        print(pred_df.to_string(index=False))
    else:
        print("  No ferroptosis genes among predictable genes with TCGA data (only NCOA4)")

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dep_df.to_csv(OUT_DIR / "ferroptosis_gene_dependencies.csv", index=False)
    nrf2_df.to_csv(OUT_DIR / "nrf2_signature.csv", index=False)
    enrich_df.to_csv(OUT_DIR / "keap1_enrichment_stats.csv", index=False)
    if not pred_df.empty:
        pred_df.to_csv(OUT_DIR / "predicted_dependency_subtypes.csv", index=False)

    print(f"\nOutputs saved to {OUT_DIR}/")

    # Flag FSP1/AIFM2
    aifm2_keap1 = enrich_df[
        (enrich_df["gene"] == "AIFM2") & (enrich_df["comparison"] == "KEAP1_mut_vs_wt")
    ]
    if not aifm2_keap1.empty:
        row = aifm2_keap1.iloc[0]
        print(f"\n** FSP1/AIFM2 KEAP1 enrichment: effect={row['effect_size']:.4f}, "
              f"p={row['p_value']:.4f}, FDR={row['fdr']:.4f}")
        print("   NOTE: DepMap may underestimate FSP1 importance (in vivo ~80% tumor suppression).")
        print("   icFSP1 inhibitor available. Flagged as priority target regardless of DepMap score.")

    print("\nDone.")


if __name__ == "__main__":
    main()
