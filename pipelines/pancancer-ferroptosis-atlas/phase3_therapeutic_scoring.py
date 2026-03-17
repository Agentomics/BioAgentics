"""Phase 3: Therapeutic opportunity scoring with combination vulnerabilities and PRISM validation.

Computes single-agent and combination therapy opportunity scores per cancer type,
with in vivo evidence weighting (FSP1 2x), FSP1+SLC7A11 expression signature,
and PRISM drug sensitivity validation.

Usage:
    uv run python -m pipelines.pancancer-ferroptosis-atlas.phase3_therapeutic_scoring
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix, load_depmap_model_metadata

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase1"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase2"
TCGA_DIR = REPO_ROOT / "data" / "tcga" / "pancancer_nrf2_keap1"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase3"

# PRISM ferroptosis compounds to search for
PRISM_FERROPTOSIS_COMPOUNDS = [
    "erastin", "RSL3", "ML162", "ML210", "FIN56", "FINO2",
    "CB-839", "telaglenastat", "icFSP1", "auranofin",
]

# TCGA cancer type mapping (TCGA abbreviation → OncotreeLineage)
TCGA_TO_ONCOTREE = {
    "LUAD": "Lung", "LUSC": "Lung", "SCLC": "Lung",
    "BRCA": "Breast", "OV": "Ovary/Fallopian Tube",
    "SKCM": "Skin", "UVM": "Eye",
    "KIRC": "Kidney", "KIRP": "Kidney", "KICH": "Kidney",
    "BLCA": "Bladder/Urinary Tract",
    "HNSC": "Head and Neck",
    "ESCA": "Esophagus/Stomach", "STAD": "Esophagus/Stomach",
    "COAD": "Bowel", "READ": "Bowel",
    "UCEC": "Uterus", "UCS": "Uterus",
    "CESC": "Cervix",
    "PAAD": "Pancreas",
    "LIHC": "Liver",
    "CHOL": "Biliary Tract",
    "GBM": "CNS/Brain", "LGG": "CNS/Brain",
    "THCA": "Thyroid",
    "PRAD": "Prostate",
    "SARC": "Soft Tissue",
    "PCPG": "Peripheral Nervous System",
    "ACC": "Adrenal Gland",
    "TGCT": "Testis",
    "DLBC": "Lymphoid",
    "LAML": "Myeloid", "AML": "Myeloid",
    "THYM": "Pleura",
    "MESO": "Pleura",
}


def load_tcga_nrf2_frequencies(tcga_dir: Path) -> pd.DataFrame | None:
    """Load TCGA pan-cancer NRF2/KEAP1 mutation frequencies."""
    freq_path = tcga_dir / "nfe2l2_keap1_mutation_frequencies.csv"
    if not freq_path.exists():
        print("  TCGA mutation frequencies not available — scoring without patient fractions")
        return None

    tcga = pd.read_csv(freq_path)

    # Map to OncotreeLineage and aggregate
    tcga["OncotreeLineage"] = tcga["cancer_type"].map(TCGA_TO_ONCOTREE)
    tcga = tcga.dropna(subset=["OncotreeLineage"])

    agg = tcga.groupby("OncotreeLineage").agg(
        n_sequenced=("n_sequenced", "sum"),
        NFE2L2_mutated=("NFE2L2_mutated", "sum"),
        KEAP1_mutated=("KEAP1_mutated", "sum"),
        double_wt=("double_wt", "sum"),
    ).reset_index()

    agg["nrf2_active_pct"] = (agg["NFE2L2_mutated"] + agg["KEAP1_mutated"]) / agg["n_sequenced"] * 100
    agg["wt_fraction"] = agg["double_wt"] / agg["n_sequenced"]

    print(f"  Loaded TCGA frequencies for {len(agg)} lineages")
    return agg


def compute_single_agent_scores(stats: pd.DataFrame, tcga: pd.DataFrame | None) -> pd.DataFrame:
    """Compute single-agent therapeutic opportunity scores with in vivo weighting."""
    # Pivot stats to get mean dependency per cancer type per gene
    pivot = stats.pivot_table(
        index="cancer_type", columns="gene", values="mean_dependency", aggfunc="first"
    )

    rows = []
    for ct in pivot.index:
        row = {"cancer_type": ct}

        # Get NRF2-WT patient fraction from TCGA
        wt_frac = 1.0  # default if no TCGA data
        if tcga is not None and ct in tcga["OncotreeLineage"].values:
            wt_frac = tcga.loc[tcga["OncotreeLineage"] == ct, "wt_fraction"].iloc[0]

        # FSP1i score (Tier A, 2x weight)
        aifm2_dep = pivot.loc[ct, "AIFM2"] if "AIFM2" in pivot.columns else 0
        row["FSP1i_raw_dependency"] = aifm2_dep
        row["FSP1i_score"] = abs(aifm2_dep) * 2 * wt_frac  # 2x weight, more negative = better
        row["FSP1i_tier"] = "A"

        # GPX4i score (Tier B, 1x weight)
        gpx4_dep = pivot.loc[ct, "GPX4"] if "GPX4" in pivot.columns else 0
        row["GPX4i_raw_dependency"] = gpx4_dep
        row["GPX4i_score"] = abs(gpx4_dep) * 1 * wt_frac
        row["GPX4i_tier"] = "B"

        # GLS1i score (untiered, 1x weight)
        gls_dep = pivot.loc[ct, "GLS"] if "GLS" in pivot.columns else 0
        row["GLS1i_raw_dependency"] = gls_dep
        row["GLS1i_score"] = abs(gls_dep) * 1 * wt_frac
        row["GLS1i_tier"] = "untiered"

        row["nrf2_wt_fraction"] = wt_frac
        rows.append(row)

    scores = pd.DataFrame(rows)
    return scores


def compute_combination_scores(stats: pd.DataFrame) -> pd.DataFrame:
    """Compute combination vulnerability scores."""
    pivot = stats.pivot_table(
        index="cancer_type", columns="gene", values="mean_dependency", aggfunc="first"
    )

    rows = []
    for ct in pivot.index:
        row = {"cancer_type": ct}

        aifm2 = pivot.loc[ct].get("AIFM2", 0)
        gpx4 = pivot.loc[ct].get("GPX4", 0)
        txnrd1 = pivot.loc[ct].get("TXNRD1", 0)
        gclc = pivot.loc[ct].get("GCLC", 0)

        # GPX4+FSP1 dual score (Salem et al. ChRCC)
        row["GPX4_FSP1_dual_score"] = abs(gpx4) + abs(aifm2)
        row["GPX4_dep"] = gpx4
        row["AIFM2_dep"] = aifm2

        # TrxR1+GCLC combination (Tier C — non-ferroptotic cell death)
        row["TrxR1_GCLC_dual_score"] = abs(txnrd1) + abs(gclc)
        row["TXNRD1_dep"] = txnrd1
        row["GCLC_dep"] = gclc
        row["TrxR1_GCLC_tier"] = "C"

        # Flag for HDACi+ferroptosis (high defense dependency overall)
        defense_genes = ["AIFM2", "GPX4", "SLC7A11", "GLS", "GCLC"]
        defense_deps = [abs(pivot.loc[ct].get(g, 0)) for g in defense_genes]
        row["ferroptosis_defense_burden"] = np.mean(defense_deps)
        row["HDACi_ferroptosis_candidate"] = np.mean(defense_deps) > 0.2

        rows.append(row)

    return pd.DataFrame(rows)


def compute_expression_signature(depmap_dir: Path, meta: pd.DataFrame) -> pd.DataFrame:
    """Compute FSP1+SLC7A11 expression signature per cancer type."""
    print("Loading expression data for FSP1+SLC7A11 signature...")
    expr = load_depmap_matrix(depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")

    sig_genes = ["AIFM2", "SLC7A11"]
    missing = [g for g in sig_genes if g not in expr.columns]
    if missing:
        print(f"  WARNING: missing expression genes: {missing}")
        return pd.DataFrame()

    # Merge with metadata for cancer type
    common = expr.index.intersection(meta.index)
    expr_sub = expr.loc[common, sig_genes].copy()
    expr_sub["OncotreeLineage"] = meta.loc[common, "OncotreeLineage"]
    expr_sub = expr_sub.dropna(subset=["OncotreeLineage"])

    # Mean expression signature per cancer type
    sig = expr_sub.groupby("OncotreeLineage")[sig_genes].mean()
    sig["FSP1_SLC7A11_signature"] = sig.mean(axis=1)
    sig = sig.sort_values("FSP1_SLC7A11_signature", ascending=False).reset_index()
    sig.columns = ["cancer_type", "AIFM2_expression", "SLC7A11_expression", "FSP1_SLC7A11_signature"]

    print(f"  Computed expression signature for {len(sig)} cancer types")
    return sig


def search_prism_compounds(depmap_dir: Path) -> pd.DataFrame:
    """Search PRISM 24Q2 for ferroptosis-related compounds."""
    meta_path = depmap_dir / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv"
    if not meta_path.exists():
        print("  PRISM metadata not found")
        return pd.DataFrame()

    prism_meta = pd.read_csv(meta_path)
    if "name" not in prism_meta.columns:
        print("  PRISM metadata missing 'name' column")
        return pd.DataFrame()

    pattern = "|".join(rf"\b{compound}\b" for compound in PRISM_FERROPTOSIS_COMPOUNDS)
    matches = prism_meta[prism_meta["name"].str.contains(pattern, case=False, na=False, regex=True)]

    if matches.empty:
        print("  No ferroptosis compounds found in PRISM 24Q2")
    else:
        print(f"  Found {len(matches)} PRISM entries for ferroptosis compounds")

    return matches


def build_top5_summary(
    single_scores: pd.DataFrame,
    combo_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Top 5 cancer types per therapy class."""
    rows = []

    # FSP1i (Tier A)
    top_fsp1 = single_scores.nlargest(5, "FSP1i_score")
    for rank, (_, r) in enumerate(top_fsp1.iterrows(), 1):
        rows.append({
            "therapy": "FSP1 inhibitor (icFSP1)",
            "evidence_tier": "A",
            "rank": rank,
            "cancer_type": r["cancer_type"],
            "score": r["FSP1i_score"],
            "raw_dependency": r["FSP1i_raw_dependency"],
        })

    # GPX4i (Tier B)
    top_gpx4 = single_scores.nlargest(5, "GPX4i_score")
    for rank, (_, r) in enumerate(top_gpx4.iterrows(), 1):
        rows.append({
            "therapy": "GPX4 inhibitor (CAUTION: in vitro only)",
            "evidence_tier": "B",
            "rank": rank,
            "cancer_type": r["cancer_type"],
            "score": r["GPX4i_score"],
            "raw_dependency": r["GPX4i_raw_dependency"],
        })

    # GLS1i
    top_gls = single_scores.nlargest(5, "GLS1i_score")
    for rank, (_, r) in enumerate(top_gls.iterrows(), 1):
        rows.append({
            "therapy": "GLS1 inhibitor (CB-839/telaglenastat)",
            "evidence_tier": "untiered",
            "rank": rank,
            "cancer_type": r["cancer_type"],
            "score": r["GLS1i_score"],
            "raw_dependency": r["GLS1i_raw_dependency"],
        })

    # GPX4+FSP1 dual
    top_dual = combo_scores.nlargest(5, "GPX4_FSP1_dual_score")
    for rank, (_, r) in enumerate(top_dual.iterrows(), 1):
        rows.append({
            "therapy": "GPX4+FSP1 dual targeting (Salem et al.)",
            "evidence_tier": "A+B",
            "rank": rank,
            "cancer_type": r["cancer_type"],
            "score": r["GPX4_FSP1_dual_score"],
            "raw_dependency": np.nan,
        })

    # TrxR1+GCLC (Tier C)
    top_trxr = combo_scores.nlargest(5, "TrxR1_GCLC_dual_score")
    for rank, (_, r) in enumerate(top_trxr.iterrows(), 1):
        rows.append({
            "therapy": "TrxR1+GCLC (NON-ferroptotic cell death)",
            "evidence_tier": "C",
            "rank": rank,
            "cancer_type": r["cancer_type"],
            "score": r["TrxR1_GCLC_dual_score"],
            "raw_dependency": np.nan,
        })

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Therapeutic opportunity scoring")
    parser.add_argument("--depmap-dir", type=Path, default=DEPMAP_DIR)
    parser.add_argument("--phase1-dir", type=Path, default=PHASE1_DIR)
    parser.add_argument("--tcga-dir", type=Path, default=TCGA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 1 stats
    print("Loading Phase 1 cancer type stats...")
    stats = pd.read_csv(args.phase1_dir / "cancer_type_stats.csv")

    # Load TCGA NRF2/KEAP1 frequencies (optional)
    tcga = load_tcga_nrf2_frequencies(args.tcga_dir)

    # Step 1: Single-agent scores
    print("\nComputing single-agent therapeutic opportunity scores...")
    single_scores = compute_single_agent_scores(stats, tcga)
    single_path = args.results_dir / "therapeutic_opportunity_scores.csv"
    single_scores.to_csv(single_path, index=False)
    print(f"Saved: {single_path}")

    # Step 2: Combination scores
    print("\nComputing combination vulnerability scores...")
    combo_scores = compute_combination_scores(stats)
    combo_path = args.results_dir / "combination_vulnerability_scores.csv"
    combo_scores.to_csv(combo_path, index=False)
    print(f"Saved: {combo_path}")

    # Step 3: FSP1+SLC7A11 expression signature
    print("\nComputing FSP1+SLC7A11 expression signature...")
    meta = load_depmap_model_metadata(args.depmap_dir / "Model.csv")
    expr_sig = compute_expression_signature(args.depmap_dir, meta)
    if not expr_sig.empty:
        expr_path = args.results_dir / "fsp1_slc7a11_expression_signature.csv"
        expr_sig.to_csv(expr_path, index=False)
        print(f"Saved: {expr_path}")

    # Step 4: PRISM validation
    print("\nSearching PRISM 24Q2 for ferroptosis compounds...")
    prism_hits = search_prism_compounds(args.depmap_dir)
    prism_path = args.results_dir / "prism_validation.csv"
    if not prism_hits.empty:
        prism_hits.to_csv(prism_path, index=False)
    else:
        pd.DataFrame({
            "note": ["No ferroptosis compounds found in PRISM 24Q2"],
            "searched_compounds": [", ".join(PRISM_FERROPTOSIS_COMPOUNDS)],
        }).to_csv(prism_path, index=False)
    print(f"Saved: {prism_path}")

    # Step 5: Top 5 summary
    top5 = build_top5_summary(single_scores, combo_scores)
    top5_path = args.results_dir / "top5_cancer_types_per_therapy.csv"
    top5.to_csv(top5_path, index=False)
    print(f"\nTop 5 cancer types per therapy class:")
    print(top5.to_string(index=False))

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
