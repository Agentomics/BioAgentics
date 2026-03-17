"""PRISM drug sensitivity analysis by CRC KRAS allele.

Computes differential drug sensitivity across KRAS allele groups and
cross-references with CRISPR dependency data.

Usage:
    uv run python -m bioagentics.models.crc_drug_sensitivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.crc_depmap import annotate_crc_lines, DEFAULT_DEPMAP_DIR
from bioagentics.data.gene_ids import load_depmap_matrix
from bioagentics.models.crc_dependency import _cohens_d, _fdr_correction

DEFAULT_DEST = REPO_ROOT / "output" / "crc-kras-dependencies" / "prism_drug_results.json"

# Key drug classes to prioritize in output
KEY_DRUG_TERMS = {
    "MEK": ["mek", "trametinib", "selumetinib", "binimetinib", "cobimetinib"],
    "ERK": ["erk"],
    "SHP2": ["shp2", "shp-2"],
    "mTOR": ["mtor", "rapamycin", "everolimus", "temsirolimus"],
    "EGFR": ["egfr", "cetuximab", "erlotinib", "gefitinib", "afatinib"],
    "KRAS": ["kras", "ras", "sotorasib", "adagrasib", "mrtx"],
    "PI3K": ["pi3k", "bkm120", "alpelisib", "pictilisib"],
    "RAF": ["raf", "sorafenib", "vemurafenib", "dabrafenib"],
}


def _load_prism_data(depmap_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PRISM sensitivity matrix and treatment metadata.

    Returns (sensitivity_matrix, drug_info) where:
    - sensitivity_matrix: rows=drugs (broad_id), columns=cell lines (ACH-*)
    - drug_info: broad_id -> drug name mapping
    """
    dm = pd.read_csv(depmap_dir / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv", index_col=0)
    # Strip "BRD:" prefix from index
    dm.index = dm.index.str.replace("^BRD:", "", regex=True)

    treat = pd.read_csv(depmap_dir / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")
    # Aggregate: one row per broad_id with drug name and dose
    drug_info = treat.groupby("broad_id").agg(
        name=("name", "first"),
        dose=("dose", "first"),
    )

    return dm, drug_info


def _classify_drug(name: str) -> str | None:
    """Classify drug into a key drug class, or None."""
    name_lower = str(name).lower()
    for cls, terms in KEY_DRUG_TERMS.items():
        if any(t in name_lower for t in terms):
            return cls
    return None


def _drug_allele_comparison(
    prism: pd.DataFrame,
    allele_ids: list[str],
    wt_ids: list[str],
    drug_info: pd.DataFrame,
) -> pd.DataFrame:
    """Test each drug for differential sensitivity between allele and WT."""
    results = []
    allele_cols = [c for c in allele_ids if c in prism.columns]
    wt_cols = [c for c in wt_ids if c in prism.columns]

    for broad_id in prism.index:
        allele_vals = prism.loc[broad_id, allele_cols].dropna().values.astype(float)
        wt_vals = prism.loc[broad_id, wt_cols].dropna().values.astype(float)

        if len(allele_vals) < 3 or len(wt_vals) < 3:
            continue

        _, pval = stats.mannwhitneyu(allele_vals, wt_vals, alternative="two-sided")
        d = _cohens_d(allele_vals, wt_vals)
        name = drug_info.loc[broad_id, "name"] if broad_id in drug_info.index else broad_id
        drug_class = _classify_drug(name)

        results.append({
            "broad_id": broad_id,
            "drug_name": str(name),
            "drug_class": drug_class,
            "pvalue": pval,
            "cohens_d": d,
            "mean_allele": float(allele_vals.mean()),
            "mean_wt": float(wt_vals.mean()),
            "n_allele": len(allele_vals),
            "n_wt": len(wt_vals),
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["fdr"] = _fdr_correction(df["pvalue"].values)
    return df


def _drug_dependency_correlation(
    prism: pd.DataFrame,
    crispr: pd.DataFrame,
    crc_ids: list[str],
    drug_info: pd.DataFrame,
    dep_results: dict,
) -> list[dict]:
    """Cross-reference drug sensitivity with CRISPR dependency.

    For top dependencies from Phase 2a, check if target genes have
    corresponding drugs and compute Spearman correlation.
    """
    shared_ids = list(set(prism.columns) & set(crispr.index) & set(crc_ids))
    if len(shared_ids) < 5:
        return []

    # Get top dependency genes from all allele comparisons
    top_genes = set()
    for allele_data in dep_results.get("allele_comparisons", {}).values():
        for entry in allele_data.get("top_dependencies", [])[:20]:
            top_genes.add(entry["gene"])

    # Map drug names to broad_ids for quick lookup
    name_to_brd = {}
    for brd, row in drug_info.iterrows():
        name_lower = str(row["name"]).lower()
        name_to_brd[name_lower] = brd

    correlations = []
    for gene in top_genes:
        if gene not in crispr.columns:
            continue
        dep_scores = crispr.loc[shared_ids, gene].dropna()
        valid_ids = list(dep_scores.index)
        if len(valid_ids) < 5:
            continue

        # Check if any drug targets this gene (by name match)
        gene_lower = gene.lower()
        matching_drugs = []
        for drug_name, brd in name_to_brd.items():
            if gene_lower in drug_name and brd in prism.index:
                matching_drugs.append((brd, drug_name))

        for brd, drug_name in matching_drugs:
            drug_vals = prism.loc[brd, valid_ids].dropna()
            shared = list(set(dep_scores.index) & set(drug_vals.index))
            if len(shared) < 5:
                continue
            rho, pval = stats.spearmanr(dep_scores[shared], drug_vals[shared])
            correlations.append({
                "gene": gene,
                "drug_name": drug_name,
                "broad_id": brd,
                "spearman_rho": float(rho),
                "pvalue": float(pval),
                "n_lines": len(shared),
            })

    return correlations


def compute_drug_sensitivity(depmap_dir: str | Path) -> dict:
    """Run PRISM drug sensitivity analysis."""
    depmap_dir = Path(depmap_dir)

    classified = annotate_crc_lines(depmap_dir)
    prism, drug_info = _load_prism_data(depmap_dir)

    # Filter to CRC lines present in PRISM
    crc_in_prism = [mid for mid in classified.index if mid in prism.columns]
    classified_prism = classified.loc[crc_in_prism]
    print(f"CRC lines in PRISM: {len(classified_prism)}")

    wt_ids = list(classified_prism[classified_prism["KRAS_allele"] == "WT"].index)
    all_mut_ids = list(classified_prism[classified_prism["KRAS_allele"] != "WT"].index)
    print(f"KRAS-WT: {len(wt_ids)}, KRAS-mutant: {len(all_mut_ids)}")
    print(f"Drugs in PRISM: {len(prism)}")

    results = {"allele_drug_comparisons": {}, "key_drugs": {}, "correlations": []}

    # All KRAS-mut vs WT
    print("\nAll KRAS-mutant vs WT drug sensitivity...")
    df_all = _drug_allele_comparison(prism, all_mut_ids, wt_ids, drug_info)
    sig_all = df_all[(df_all["fdr"] < 0.05) & (df_all["cohens_d"].abs() > 0.5)] if len(df_all) > 0 else pd.DataFrame()
    print(f"  Drugs tested: {len(df_all)}, Significant: {len(sig_all)}")
    results["allele_drug_comparisons"]["all_KRAS_mut"] = {
        "n_allele": len(all_mut_ids),
        "n_wt": len(wt_ids),
        "n_drugs_tested": len(df_all),
        "n_significant": len(sig_all),
        "top_drugs": df_all.nsmallest(30, "fdr").to_dict(orient="records") if len(df_all) > 0 else [],
    }

    # Per allele with >=4 lines in PRISM
    for allele in ["G12D", "G13D", "G12V", "G12C"]:
        allele_ids = list(classified_prism[classified_prism["KRAS_allele"] == allele].index)
        n = len(allele_ids)
        print(f"\n{allele}: {n} lines in PRISM")
        if n < 3:
            print("  Skipped — too few lines")
            continue

        df = _drug_allele_comparison(prism, allele_ids, wt_ids, drug_info)
        sig = df[(df["fdr"] < 0.05) & (df["cohens_d"].abs() > 0.5)] if len(df) > 0 else pd.DataFrame()
        print(f"  Drugs tested: {len(df)}, Significant: {len(sig)}")
        results["allele_drug_comparisons"][allele] = {
            "n_allele": n,
            "n_wt": len(wt_ids),
            "n_drugs_tested": len(df),
            "n_significant": len(sig),
            "top_drugs": df.nsmallest(30, "fdr").to_dict(orient="records") if len(df) > 0 else [],
        }

    # Key drug class results
    print("\nKey drug classes:")
    all_drug_results = df_all if len(df_all) > 0 else pd.DataFrame()
    for cls in KEY_DRUG_TERMS:
        cls_drugs = all_drug_results[all_drug_results["drug_class"] == cls] if len(all_drug_results) > 0 else pd.DataFrame()
        if len(cls_drugs) > 0:
            print(f"  {cls}: {len(cls_drugs)} drugs")
            results["key_drugs"][cls] = cls_drugs.sort_values("fdr").to_dict(orient="records")
        else:
            print(f"  {cls}: no drugs found")

    # Drug-dependency cross-reference
    print("\nDrug-dependency cross-reference...")
    dep_file = REPO_ROOT / "output" / "crc-kras-dependencies" / "allele_dependency_results.json"
    if dep_file.exists():
        with open(dep_file) as f:
            dep_results = json.load(f)
        crispr = load_depmap_matrix(depmap_dir / "CRISPRGeneEffect.csv")
        correlations = _drug_dependency_correlation(
            prism, crispr, crc_in_prism, drug_info, dep_results
        )
        results["correlations"] = correlations
        print(f"  Correlations computed: {len(correlations)}")
        for c in correlations:
            if abs(c["spearman_rho"]) > 0.3:
                print(f"  Strong: {c['gene']} ~ {c['drug_name']}: rho={c['spearman_rho']:.3f}")
    else:
        print("  Dependency results not found — skipping")

    # Summary
    results["summary"] = {
        "crc_lines_in_prism": len(classified_prism),
        "total_drugs": len(prism),
        "allele_counts_in_prism": classified_prism["KRAS_allele"].value_counts().to_dict(),
    }

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PRISM drug sensitivity analysis by CRC KRAS allele",
    )
    parser.add_argument(
        "--depmap-dir", type=Path, default=DEFAULT_DEPMAP_DIR,
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
    )
    args = parser.parse_args(argv)

    results = compute_drug_sensitivity(args.depmap_dir)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dest, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.dest}")


if __name__ == "__main__":
    main()
