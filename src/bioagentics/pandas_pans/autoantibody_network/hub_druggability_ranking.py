"""Task #1085: Hub druggability and therapeutic target ranking.

Evaluates novel hub proteins for therapeutic potential, ranks by composite score,
and assesses repurposing potential for neuropsychiatric/autoimmune indications.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/pandas_pans/autoantibody_network")

# Known neuropsychiatric and autoimmune drug targets for repurposing check
NEUROPSYCH_AUTOIMMUNE_DRUGS = {
    "AKT1": {"indications": ["cancer (multiple)", "tuberous sclerosis"], "drugs": ["everolimus", "capivasertib"], "relevance": "PI3K-Akt pathway modulator; capivasertib is AKT inhibitor. mTOR inhibitors (everolimus) are used in neurological conditions (tuberous sclerosis with seizures). PI3K-Akt pathway implicated in neuroinflammation."},
    "PIK3R1": {"indications": ["cancer", "activated PI3KD syndrome (APDS)"], "drugs": ["alpelisib", "copanlisib", "leniolisib"], "relevance": "PI3K pathway is targetable. Leniolisib is approved for APDS (autoimmune/lymphoproliferative). Direct link to immune dysregulation."},
    "NFKB1": {"indications": ["autoimmune/inflammatory"], "drugs": ["bortezomib (indirect)", "sulfasalazine (indirect)"], "relevance": "NF-kB is a master regulator of inflammation. Indirect inhibitors used in autoimmune conditions. Direct NFKB1 inhibition would be highly relevant to PANDAS neuroinflammation."},
    "SRC": {"indications": ["cancer", "chronic myeloid leukemia"], "drugs": ["dasatinib", "bosutinib", "ponatinib"], "relevance": "Src kinase inhibitors (dasatinib) have shown neuroprotective effects in preclinical models. Dasatinib crosses BBB. Potential for neuroinflammation modulation."},
    "JAK1": {"indications": ["rheumatoid arthritis", "atopic dermatitis", "alopecia areata", "myelofibrosis"], "drugs": ["tofacitinib", "baricitinib", "upadacitinib", "ruxolitinib"], "relevance": "JAK inhibitors are approved for multiple autoimmune conditions. Baricitinib showed benefit in COVID neurological complications. Directly targets JAK-STAT pathway enriched in this network."},
    "JAK2": {"indications": ["myelofibrosis", "polycythemia vera"], "drugs": ["ruxolitinib", "fedratinib"], "relevance": "JAK2 inhibitors modulate inflammatory cytokine signaling. JAK-STAT is the top enriched pathway in the PANDAS network."},
    "STAT3": {"indications": ["cancer"], "drugs": ["napabucasin (investigational)"], "relevance": "STAT3 is downstream of multiple cytokines (IL-6, IFNg). Cytokine amplification layer hub. Modulating STAT3 could dampen neuroinflammatory cascade."},
    "CREB1": {"indications": ["none approved"], "drugs": ["none approved"], "relevance": "CREB1 is a convergence point for cAMP and calcium signaling. Maps to anxiety, cognitive, dopaminergic, and autoimmune symptom domains. Druggable but no approved drugs yet."},
    "PRKACA": {"indications": ["cancer (adrenal)"], "drugs": ["H89 (tool compound)"], "relevance": "PKA catalytic subunit. Downstream of cAMP-dopamine signaling. Tool compounds exist but no clinical drugs for neuropsychiatric use."},
    "MAPK1": {"indications": ["cancer"], "drugs": ["trametinib", "cobimetinib", "binimetinib (MEK inhibitors, upstream)"], "relevance": "MAPK/ERK pathway. MEK inhibitors approved for cancer. Neuro-inflammatory role emerging in literature."},
}


def compute_composite_score(targets_df, brain_expr_df):
    """Compute composite therapeutic ranking score."""
    # Normalize components to [0, 1]
    df = targets_df.copy()

    # Centrality component (hub_score already normalized-ish, but let's use convergence_score)
    cs = df["convergence_score"]
    df["centrality_norm"] = (cs - cs.min()) / (cs.max() - cs.min()) if cs.max() > cs.min() else 0

    # Druggability component
    df["druggability_norm"] = df["is_druggable"].astype(float)
    # Bonus for having approved drugs
    df["has_approved"] = df["approved_drugs"].fillna("").str.strip().str.len() > 0
    df["druggability_score"] = df["druggability_norm"] * 0.5 + df["has_approved"].astype(float) * 0.5

    # Brain expression component (average across PANDAS-relevant regions)
    pandas_regions = ["basal_ganglia", "prefrontal_cortex", "thalamus"]
    brain_expr_pivot = brain_expr_df.pivot_table(
        index="gene_symbol", columns="region", values="z_score", aggfunc="mean"
    )
    available_regions = [r for r in pandas_regions if r in brain_expr_pivot.columns]
    if available_regions:
        brain_expr_pivot["brain_score"] = brain_expr_pivot[available_regions].mean(axis=1)
        brain_scores = brain_expr_pivot["brain_score"].to_dict()
    else:
        brain_scores = {}

    df["brain_expression_score"] = df["gene_symbol"].map(brain_scores).fillna(0)
    bs = df["brain_expression_score"]
    df["brain_norm"] = (bs - bs.min()) / (bs.max() - bs.min()) if bs.max() > bs.min() else 0

    # Pathway convergence (total_enriched_pathways)
    ep = df["total_enriched_pathways"]
    df["pathway_norm"] = (ep - ep.min()) / (ep.max() - ep.min()) if ep.max() > ep.min() else 0

    # Cross-layer bonus
    df["cross_layer_bonus"] = df["is_cross_layer"].astype(float) * 0.15

    # Composite score: centrality 25%, druggability 25%, brain expression 20%, pathway convergence 20%, cross-layer 10%
    df["therapeutic_rank_score"] = (
        0.25 * df["centrality_norm"] +
        0.25 * df["druggability_score"] +
        0.20 * df["brain_norm"] +
        0.20 * df["pathway_norm"] +
        0.10 * df["cross_layer_bonus"] / 0.15  # normalize back
    )

    return df.sort_values("therapeutic_rank_score", ascending=False)


def assess_repurposing(ranked_df, drug_annotations):
    """Assess repurposing potential for top candidates."""
    top_candidates = ranked_df.head(10)

    assessments = []
    for _, row in top_candidates.iterrows():
        gene = row["gene_symbol"]

        # Get drug annotations for this gene
        gene_drugs = drug_annotations[drug_annotations["gene_symbol"] == gene]
        approved = gene_drugs[gene_drugs["approved"] == True]
        n_approved = len(approved)
        approved_list = approved["drug_name"].tolist()[:10]
        interaction_types = gene_drugs["interaction_type"].dropna().unique().tolist()

        # Check known neuropsych/autoimmune relevance
        known_info = NEUROPSYCH_AUTOIMMUNE_DRUGS.get(gene, {})

        assessments.append({
            "rank": len(assessments) + 1,
            "gene_symbol": gene,
            "convergence_score": round(row["convergence_score"], 4),
            "therapeutic_rank_score": round(row["therapeutic_rank_score"], 4),
            "is_cross_layer": bool(row["is_cross_layer"]),
            "total_enriched_pathways": int(row["total_enriched_pathways"]),
            "symptom_domains": row.get("symptom_domains", ""),
            "n_approved_drugs": n_approved,
            "approved_drugs": "; ".join(approved_list) if approved_list else "none",
            "interaction_types": "; ".join(interaction_types) if interaction_types else "",
            "known_indications": known_info.get("indications", []),
            "repurposing_rationale": known_info.get("relevance", "No known neuropsych/autoimmune indication data"),
            "safety_data_available": n_approved > 0,
        })

    return pd.DataFrame(assessments)


def main():
    print("=" * 60)
    print("Task #1085: Hub Druggability and Therapeutic Target Ranking")
    print("=" * 60)

    # Load data
    targets = pd.read_csv(DATA_DIR / "novel_therapeutic_targets.tsv", sep="\t")
    drugs = pd.read_csv(DATA_DIR / "druggability_annotations.tsv", sep="\t")
    brain_expr = pd.read_csv(DATA_DIR / "allen_expression_by_region.tsv", sep="\t")
    hub_metrics = pd.read_csv(DATA_DIR / "hub_centrality_metrics.tsv", sep="\t")

    print(f"\n  Novel targets: {len(targets)}")
    print(f"  Drug annotations: {len(drugs)}")
    print(f"  Druggable targets: {targets['is_druggable'].sum()}")

    # 1. Validate DGIdb annotations
    print("\n[1/4] Cross-validating DGIdb annotations...")
    drug_gene_counts = drugs.groupby("gene_symbol").agg(
        n_drugs=("drug_name", "nunique"),
        n_approved=("approved", "sum"),
        interaction_types=("interaction_type", lambda x: "; ".join(x.dropna().unique()))
    ).reset_index()

    target_drug_info = targets.merge(drug_gene_counts, on="gene_symbol", how="left")
    targets_with_approved = target_drug_info[target_drug_info["n_approved"] > 0]
    print(f"  Targets with approved drugs: {len(targets_with_approved)}/{len(targets)}")
    print(f"  Total unique drugs across targets: {drugs[drugs['gene_symbol'].isin(targets['gene_symbol'])]['drug_name'].nunique()}")

    # 2. Compute composite ranking
    print("\n[2/4] Computing composite therapeutic ranking score...")
    ranked = compute_composite_score(targets, brain_expr)
    ranked.to_csv(DATA_DIR / "therapeutic_target_ranking.tsv", sep="\t", index=False)

    print("\n  Top 10 ranked therapeutic targets:")
    for i, (_, row) in enumerate(ranked.head(10).iterrows()):
        cl = "cross-layer" if row["is_cross_layer"] else ""
        print(f"  {i+1:2d}. {row['gene_symbol']:10s} score={row['therapeutic_rank_score']:.3f} "
              f"pathways={int(row['total_enriched_pathways']):3d} "
              f"druggable={'Y' if row['is_druggable'] else 'N'} {cl}")

    # 3. Repurposing assessment
    print("\n[3/4] Assessing repurposing potential for top candidates...")
    repurposing = assess_repurposing(ranked, drugs)
    repurposing.to_csv(DATA_DIR / "repurposing_assessment.tsv", sep="\t", index=False)

    for _, row in repurposing.head(5).iterrows():
        print(f"\n  #{row['rank']} {row['gene_symbol']} (score={row['therapeutic_rank_score']:.3f}):")
        print(f"    Cross-layer: {row['is_cross_layer']}, Enriched pathways: {row['total_enriched_pathways']}")
        print(f"    Approved drugs: {row['approved_drugs']}")
        print(f"    Known indications: {row['known_indications']}")
        print(f"    Rationale: {row['repurposing_rationale'][:120]}...")

    # 4. Summary
    print("\n[4/4] Compiling summary...")
    summary = {
        "total_novel_targets": len(targets),
        "druggable_targets": int(targets["is_druggable"].sum()),
        "targets_with_approved_drugs": len(targets_with_approved),
        "composite_score_weights": {
            "centrality": 0.25,
            "druggability": 0.25,
            "brain_expression": 0.20,
            "pathway_convergence": 0.20,
            "cross_layer": 0.10,
        },
        "top5_candidates": repurposing.head(5).to_dict("records"),
        "repurposing_highlights": [
            r.to_dict() for _, r in repurposing.iterrows()
            if r["safety_data_available"] and r["repurposing_rationale"] != "No known neuropsych/autoimmune indication data"
        ],
    }

    with open(DATA_DIR / "druggability_ranking_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("DRUGGABILITY RANKING COMPLETE")
    print(f"Output: therapeutic_target_ranking.tsv, repurposing_assessment.tsv,")
    print(f"        druggability_ranking_summary.json")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    summary = main()
