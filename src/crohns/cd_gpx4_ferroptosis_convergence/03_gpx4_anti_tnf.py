"""Extract GPX4 from anti-TNF bulk cohorts and test association with treatment response.

Phase 1, Task 3 of cd-gpx4-ferroptosis-convergence.
Cohorts: GSE16879, GSE12251, GSE73661.
Tests responder vs non-responder using Wilcoxon rank-sum and Cohen's d.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROCESSED_DIR = Path("output/crohns/anti-tnf-response-prediction/processed")
OUTPUT_DIR = Path("output/crohns/cd-gpx4-ferroptosis-convergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STUDIES = ["GSE16879", "GSE12251", "GSE73661"]


def load_gpx4_per_study(study: str) -> pd.DataFrame:
    """Load GPX4 expression and merge with metadata for a study."""
    expr = pd.read_csv(PROCESSED_DIR / f"{study}_expression.csv")
    gpx4_row = expr[expr["gene_symbol"] == "GPX4"].drop(columns="gene_symbol")
    if gpx4_row.empty:
        raise ValueError(f"GPX4 not found in {study}")

    # Melt: sample_id → GPX4 expression
    gpx4_long = gpx4_row.T.reset_index()
    gpx4_long.columns = ["sample_id", "GPX4"]
    gpx4_long["GPX4"] = gpx4_long["GPX4"].astype(float)

    meta = pd.read_csv(PROCESSED_DIR / "combined_metadata.csv")
    meta = meta[meta["study"] == study]

    merged = gpx4_long.merge(meta, on="sample_id", how="inner")
    return merged


def test_response(df: pd.DataFrame, study: str) -> dict:
    """Wilcoxon rank-sum test: responder vs non-responder GPX4."""
    resp = df.loc[df["response_status"] == "responder", "GPX4"]
    nresp = df.loc[df["response_status"] == "non_responder", "GPX4"]

    stat, pval = sp_stats.mannwhitneyu(resp, nresp, alternative="two-sided")
    pooled_std = np.sqrt(
        ((len(resp) - 1) * resp.std() ** 2 + (len(nresp) - 1) * nresp.std() ** 2)
        / (len(resp) + len(nresp) - 2)
    )
    cohens_d = (resp.mean() - nresp.mean()) / pooled_std if pooled_std > 0 else 0.0

    return {
        "study": study,
        "n_responder": len(resp),
        "n_non_responder": len(nresp),
        "mean_responder": resp.mean(),
        "mean_non_responder": nresp.mean(),
        "log2FC_R_vs_NR": np.log2((resp.mean() + 1e-6) / (nresp.mean() + 1e-6)),
        "cohens_d": cohens_d,
        "mann_whitney_U": stat,
        "pvalue": pval,
    }


def main():
    results = []
    all_data = []

    for study in STUDIES:
        print(f"\n=== {study} ===")
        df = load_gpx4_per_study(study)
        print(f"  Samples: {len(df)} ({(df['response_status']=='responder').sum()} R, "
              f"{(df['response_status']=='non_responder').sum()} NR)")

        res = test_response(df, study)
        results.append(res)
        all_data.append(df)

        print(f"  GPX4 mean R={res['mean_responder']:.2f}, NR={res['mean_non_responder']:.2f}")
        print(f"  log2FC(R/NR)={res['log2FC_R_vs_NR']:.3f}, Cohen's d={res['cohens_d']:.3f}, p={res['pvalue']:.4f}")

    results_df = pd.DataFrame(results)

    # BH FDR
    n = len(results_df)
    sorted_p = results_df["pvalue"].sort_values()
    fdr = sorted_p * n / (np.arange(1, n + 1))
    fdr = fdr.clip(upper=1.0)
    fdr_vals = fdr.values.copy()
    for i in range(len(fdr_vals) - 2, -1, -1):
        fdr_vals[i] = min(fdr_vals[i], fdr_vals[i + 1])
    results_df.loc[sorted_p.index, "FDR"] = fdr_vals

    results_df.to_csv(OUTPUT_DIR / "gpx4_anti_tnf_response.csv", index=False)

    print("\n=== Summary ===")
    print(results_df[["study", "n_responder", "n_non_responder", "mean_responder",
                       "mean_non_responder", "log2FC_R_vs_NR", "cohens_d", "pvalue", "FDR"]].to_string(index=False))

    # Check success criterion: |Cohen's d| > 0.5 and p < 0.05 in at least 2/3
    sig = results_df[(results_df["pvalue"] < 0.05) & (results_df["cohens_d"].abs() > 0.5)]
    print(f"\nCohorts meeting criterion (p<0.05, |d|>0.5): {len(sig)}/3")
    if len(sig) >= 2:
        print("SUCCESS: GPX4 differentially expressed in >= 2/3 cohorts.")
    else:
        print("NOTE: Criterion not met in >= 2 cohorts. See plan for mitigation.")

    print(f"\nOutput saved to {OUTPUT_DIR}/gpx4_anti_tnf_response.csv")


if __name__ == "__main__":
    main()
