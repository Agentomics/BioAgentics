"""TCGA COAD/READ KRAS allele validation and survival analysis.

Loads pre-processed TCGA colorectal cancer data (mutations + clinical),
validates KRAS allele frequencies against DepMap, computes co-mutation
patterns by allele, runs survival analysis, and estimates patient populations.

Usage:
    uv run python -m bioagentics.data.crc_tcga
    uv run python -m bioagentics.data.crc_tcga --dest output/crc-kras-dependencies/tcga_crc_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import fisher_exact

from bioagentics.config import REPO_ROOT
from bioagentics.data.crc_common import DRIVER_GENES, classify_kras_allele

DEFAULT_TCGA_DIR = REPO_ROOT / "data" / "tcga" / "coadread"
DEFAULT_DEST = REPO_ROOT / "output" / "crc-kras-dependencies" / "tcga_crc_validation.json"

# CRC epidemiology constants
CRC_US_ANNUAL_CASES = 153_000
KRAS_MUTANT_FRACTION = 0.45

# Co-mutation genes to test (exclude KRAS itself)
COMUTATION_GENES = ["APC", "TP53", "PIK3CA", "BRAF", "SMAD4"]


def load_mutations(tcga_dir: Path) -> pd.DataFrame:
    """Load pre-processed TCGA COAD/READ mutation data."""
    path = tcga_dir / "mutations_driver_genes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Mutation file not found: {path}")
    return pd.read_csv(path)


def load_clinical(tcga_dir: Path) -> pd.DataFrame:
    """Load TCGA COAD/READ clinical patient data."""
    path = tcga_dir / "clinical_patient.csv"
    if not path.exists():
        raise FileNotFoundError(f"Clinical file not found: {path}")
    return pd.read_csv(path)


def classify_patients(muts: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    """Build per-patient table with KRAS allele, co-mutations, and clinical data.

    Returns DataFrame indexed by patient_id with driver mutation status,
    KRAS allele classification, and clinical annotations.
    """
    # Start from clinical data (all patients)
    patients = clinical[["patient_id", "CANCER_TYPE_ACRONYM", "AGE", "SEX",
                          "AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE",
                          "OS_MONTHS", "OS_STATUS", "PFS_MONTHS", "PFS_STATUS"]].copy()

    # Filter mutations to damaging types
    damaging = muts[muts["mutation_type"].isin({
        "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
        "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    })].copy()

    # Per-gene mutation annotation
    for gene in DRIVER_GENES:
        gene_muts = damaging[damaging["gene"] == gene]
        mutated_patients = set(gene_muts["patient_id"])
        patients[f"{gene}_mutated"] = patients["patient_id"].isin(mutated_patients)

        pc = gene_muts.groupby("patient_id")["protein_change"].apply(
            lambda x: ";".join(x.dropna().unique())
        )
        patients[f"{gene}_protein_change"] = patients["patient_id"].map(pc).fillna("")

    # BRAF V600E specific flag
    braf_muts = damaging[damaging["gene"] == "BRAF"]
    v600e_patients = set(braf_muts[braf_muts["protein_change"] == "V600E"]["patient_id"])
    patients["BRAF_V600E"] = patients["patient_id"].isin(v600e_patients)

    # KRAS allele classification — add p. prefix for classify_kras_allele
    kras_muts = damaging[damaging["gene"] == "KRAS"]
    kras_alleles = kras_muts.groupby("patient_id")["protein_change"].apply(
        lambda x: [f"p.{v}" for v in x.dropna().unique()]
    )

    def get_allele(pid: str) -> str:
        if pid in kras_alleles.index:
            return classify_kras_allele(kras_alleles[pid])
        return "WT"

    patients["KRAS_allele"] = [get_allele(p) for p in patients["patient_id"]]

    return patients


def compute_allele_frequencies(patients: pd.DataFrame) -> dict:
    """Compute KRAS allele frequencies overall and by cancer type."""
    kras_mut = patients[patients["KRAS_mutated"]].copy()

    overall = kras_mut["KRAS_allele"].value_counts()
    overall_pct = (overall / len(kras_mut) * 100).round(1)

    result = {
        "total_patients": len(patients),
        "kras_mutant_patients": len(kras_mut),
        "kras_mutation_rate_pct": round(len(kras_mut) / len(patients) * 100, 1),
        "allele_counts": overall.to_dict(),
        "allele_pct": overall_pct.to_dict(),
    }

    # By cancer type
    by_type = {}
    for ct in sorted(patients["CANCER_TYPE_ACRONYM"].dropna().unique()):
        ct_patients = patients[patients["CANCER_TYPE_ACRONYM"] == ct]
        ct_kras = ct_patients[ct_patients["KRAS_mutated"]]
        if len(ct_kras) > 0:
            counts = ct_kras["KRAS_allele"].value_counts()
            by_type[ct] = {
                "total": len(ct_patients),
                "kras_mutant": len(ct_kras),
                "kras_rate_pct": round(len(ct_kras) / len(ct_patients) * 100, 1),
                "allele_counts": counts.to_dict(),
                "allele_pct": (counts / len(ct_kras) * 100).round(1).to_dict(),
            }
    result["by_cancer_type"] = by_type

    return result


def compute_comutation_patterns(patients: pd.DataFrame) -> dict:
    """Compute co-mutation rates per KRAS allele with Fisher's exact test."""
    kras_mut = patients[patients["KRAS_mutated"]].copy()
    kras_wt = patients[~patients["KRAS_mutated"]].copy()

    alleles = [a for a in kras_mut["KRAS_allele"].unique() if a not in ("WT", "KRAS_other")]
    alleles = sorted(alleles, key=lambda a: -kras_mut["KRAS_allele"].value_counts().get(a, 0))

    results = {}
    for gene in COMUTATION_GENES:
        col = f"{gene}_mutated"
        gene_results = {
            "overall_kras_mut_rate": round(kras_mut[col].mean() * 100, 1),
            "kras_wt_rate": round(kras_wt[col].mean() * 100, 1),
            "by_allele": {},
        }

        for allele in alleles:
            allele_patients = kras_mut[kras_mut["KRAS_allele"] == allele]
            other_patients = kras_mut[kras_mut["KRAS_allele"] != allele]
            n_allele = len(allele_patients)
            if n_allele < 3:
                continue

            rate = allele_patients[col].mean() * 100

            # Fisher's exact: this allele vs all other KRAS-mutant
            a = allele_patients[col].sum()
            b = n_allele - a
            c = other_patients[col].sum()
            d = len(other_patients) - c
            table = [[int(a), int(b)], [int(c), int(d)]]
            odds_ratio, p_value = fisher_exact(table)

            gene_results["by_allele"][allele] = {
                "n": n_allele,
                "mutated": int(a),
                "rate_pct": round(rate, 1),
                "odds_ratio_vs_other_kras": round(odds_ratio, 2) if np.isfinite(odds_ratio) else None,
                "p_value": round(p_value, 4),
            }

        results[gene] = gene_results

    return results


def compute_survival(patients: pd.DataFrame) -> dict:
    """Kaplan-Meier survival analysis by KRAS allele for OS and PFS."""
    results = {}

    for endpoint, time_col, status_col in [
        ("OS", "OS_MONTHS", "OS_STATUS"),
        ("PFS", "PFS_MONTHS", "PFS_STATUS"),
    ]:
        # Parse status: "1:DECEASED" -> 1, "0:LIVING" -> 0
        df = patients.copy()
        df["_time"] = pd.to_numeric(df[time_col], errors="coerce")
        df["_event"] = df[status_col].apply(
            lambda x: int(str(x).split(":")[0]) if pd.notna(x) and ":" in str(x) else np.nan
        )
        df = df.dropna(subset=["_time", "_event"])
        df = df[df["_time"] > 0]

        # Compare KRAS-mut vs WT
        kras_mut_mask = df["KRAS_mutated"]
        kras_wt_mask = ~df["KRAS_mutated"]

        endpoint_result = {"n_evaluable": len(df)}

        if kras_mut_mask.sum() >= 5 and kras_wt_mask.sum() >= 5:
            lr = logrank_test(
                df.loc[kras_mut_mask, "_time"], df.loc[kras_wt_mask, "_time"],
                df.loc[kras_mut_mask, "_event"], df.loc[kras_wt_mask, "_event"],
            )
            kmf = KaplanMeierFitter()

            kmf.fit(df.loc[kras_mut_mask, "_time"], df.loc[kras_mut_mask, "_event"], label="KRAS_mut")
            median_mut = kmf.median_survival_time_
            median_mut = round(float(median_mut), 1) if np.isfinite(median_mut) else None

            kmf.fit(df.loc[kras_wt_mask, "_time"], df.loc[kras_wt_mask, "_event"], label="KRAS_WT")
            median_wt = kmf.median_survival_time_
            median_wt = round(float(median_wt), 1) if np.isfinite(median_wt) else None

            endpoint_result["kras_mut_vs_wt"] = {
                "n_mut": int(kras_mut_mask.sum()),
                "n_wt": int(kras_wt_mask.sum()),
                "median_months_mut": median_mut,
                "median_months_wt": median_wt,
                "logrank_p": round(float(lr.p_value), 4),
            }

        # Per-allele survival (only alleles with >=10 patients)
        allele_counts = df.loc[kras_mut_mask, "KRAS_allele"].value_counts()
        evaluable_alleles = [a for a, n in allele_counts.items() if n >= 10 and a != "KRAS_other"]

        if len(evaluable_alleles) >= 2:
            allele_results = {}
            for allele in evaluable_alleles:
                mask = df["KRAS_allele"] == allele
                n = int(mask.sum())
                kmf = KaplanMeierFitter()
                kmf.fit(df.loc[mask, "_time"], df.loc[mask, "_event"], label=allele)
                median = kmf.median_survival_time_
                allele_results[allele] = {
                    "n": n,
                    "events": int(df.loc[mask, "_event"].sum()),
                    "median_months": round(float(median), 1) if np.isfinite(median) else None,
                }

            # Pairwise log-rank between alleles
            pairwise = {}
            for i, a1 in enumerate(evaluable_alleles):
                for a2 in evaluable_alleles[i + 1:]:
                    m1 = df["KRAS_allele"] == a1
                    m2 = df["KRAS_allele"] == a2
                    lr = logrank_test(
                        df.loc[m1, "_time"], df.loc[m2, "_time"],
                        df.loc[m1, "_event"], df.loc[m2, "_event"],
                    )
                    pairwise[f"{a1}_vs_{a2}"] = round(float(lr.p_value), 4)

            endpoint_result["by_allele"] = allele_results
            endpoint_result["pairwise_logrank_p"] = pairwise

        results[endpoint] = endpoint_result

    return results


def compute_population_estimates(allele_freq: dict) -> dict:
    """Estimate US patient populations per KRAS allele for trial enrollment."""
    kras_mutant_annual = int(CRC_US_ANNUAL_CASES * KRAS_MUTANT_FRACTION)
    allele_pcts = allele_freq["allele_pct"]

    estimates = {
        "crc_annual_us": CRC_US_ANNUAL_CASES,
        "kras_mutant_fraction": KRAS_MUTANT_FRACTION,
        "kras_mutant_annual": kras_mutant_annual,
        "per_allele": {},
    }

    for allele, pct in sorted(allele_pcts.items(), key=lambda x: -x[1]):
        annual = int(kras_mutant_annual * pct / 100)
        estimates["per_allele"][allele] = {
            "tcga_pct": pct,
            "estimated_annual_us": annual,
        }

    return estimates


def run_validation(tcga_dir: Path) -> dict:
    """Run complete TCGA CRC validation pipeline."""
    print("Loading TCGA COAD/READ data...")
    muts = load_mutations(tcga_dir)
    clinical = load_clinical(tcga_dir)
    print(f"  Mutations: {len(muts)} entries, {muts['gene'].nunique()} genes")
    print(f"  Clinical: {len(clinical)} patients")

    print("Classifying patients...")
    patients = classify_patients(muts, clinical)

    print("\n--- KRAS Allele Frequencies ---")
    allele_freq = compute_allele_frequencies(patients)
    print(f"  Total patients: {allele_freq['total_patients']}")
    print(f"  KRAS mutant: {allele_freq['kras_mutant_patients']} ({allele_freq['kras_mutation_rate_pct']}%)")
    for allele, pct in sorted(allele_freq["allele_pct"].items(), key=lambda x: -x[1]):
        n = allele_freq["allele_counts"][allele]
        print(f"    {allele}: {n} ({pct}%)")

    print("\n--- Co-mutation Patterns ---")
    comutation = compute_comutation_patterns(patients)
    for gene, data in comutation.items():
        print(f"  {gene}: {data['overall_kras_mut_rate']}% in KRAS-mut vs {data['kras_wt_rate']}% in WT")

    print("\n--- Survival Analysis ---")
    survival = compute_survival(patients)
    for ep, data in survival.items():
        print(f"  {ep}: {data['n_evaluable']} evaluable")
        if "kras_mut_vs_wt" in data:
            mv = data["kras_mut_vs_wt"]
            print(f"    KRAS-mut median: {mv['median_months_mut']}m, WT: {mv['median_months_wt']}m, p={mv['logrank_p']}")

    print("\n--- Population Estimates ---")
    population = compute_population_estimates(allele_freq)
    for allele, est in population["per_allele"].items():
        print(f"  {allele}: ~{est['estimated_annual_us']:,}/yr ({est['tcga_pct']}%)")

    return {
        "allele_frequencies": allele_freq,
        "comutation_patterns": comutation,
        "survival_analysis": survival,
        "population_estimates": population,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TCGA COAD/READ KRAS allele validation and survival analysis",
    )
    parser.add_argument(
        "--tcga-dir", type=Path, default=DEFAULT_TCGA_DIR,
        help=f"TCGA COAD/READ data directory (default: {DEFAULT_TCGA_DIR})",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"Output JSON path (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    results = run_validation(args.tcga_dir)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dest, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.dest}")


if __name__ == "__main__":
    main()
