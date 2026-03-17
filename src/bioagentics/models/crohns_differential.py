"""Single-omic differential analysis (CD vs non-IBD controls) for HMP2 data.

Performs differential abundance testing for each feature in metagenomic
and metabolomic layers to validate known CD biology before multi-omic
integration.

Methods:
- Wilcoxon rank-sum test (Mann-Whitney U) for each feature
- Benjamini-Hochberg FDR correction
- Cliff's delta effect size
- Volcano plot visualization
- Extraction of known CD-relevant features

Usage::

    from bioagentics.models.crohns_differential import DifferentialAnalysis

    da = DifferentialAnalysis()
    results = da.run(species_qc, labels)
    da.volcano_plot(results, output_path="volcano.png")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "microbiome-metabolome-subtyping"

# Known CD-relevant features to specifically report
KNOWN_CD_SPECIES = [
    "Faecalibacterium_prausnitzii",
    "Roseburia_hominis",
    "Roseburia_intestinalis",
    "Escherichia_coli",
    "Bifidobacterium_longum",
    "Bifidobacterium_catenulatum",
    "Ruminococcus_gnavus",
    "Ruminococcus_torques",
    "Bacteroides_vulgatus",
    "Fusobacterium_nucleatum",
]

KNOWN_CD_METABOLITES = [
    # SCFAs
    "butyrate",
    "propionate",
    "acetate",
    # Bile acids
    "taurochenodeoxycholic acid",
    "TCDCA",
    "lithocholic acid",
    "deoxycholic acid",
    # Tryptophan pathway
    "tryptophan",
    "kynurenine",
    "quinolinate",
    "kynurenate",
    "indole",
    # Acylcarnitines
    "carnitine",
    "acetylcarnitine",
    # BCAAs
    "leucine",
    "valine",
    "isoleucine",
]


# ── Statistical Tests ──


def wilcoxon_ranksum_test(
    df: pd.DataFrame,
    labels: pd.Series,
    group_a: str = "CD",
    group_b: str = "nonIBD",
) -> pd.DataFrame:
    """Wilcoxon rank-sum test for each feature between two groups.

    Parameters
    ----------
    df : DataFrame
        Normalized feature matrix (samples × features).
    labels : Series
        Group labels indexed by sample ID (e.g., "CD", "nonIBD").
    group_a, group_b : str
        Group labels to compare.

    Returns
    -------
    DataFrame with columns: feature, statistic, p_value, mean_a, mean_b, log2fc
    """
    # Align labels to df index
    common = df.index.intersection(labels.index)
    df = df.loc[common]
    labels = labels.loc[common]

    mask_a = labels == group_a
    mask_b = labels == group_b
    n_a = mask_a.sum()
    n_b = mask_b.sum()

    if n_a == 0 or n_b == 0:
        raise ValueError(f"No samples in group '{group_a}' ({n_a}) or '{group_b}' ({n_b})")

    logger.info(
        "Wilcoxon rank-sum: %s (n=%d) vs %s (n=%d), %d features",
        group_a, n_a, group_b, n_b, df.shape[1],
    )

    results = []
    for col in df.columns:
        vals_a = df.loc[mask_a, col].dropna().values
        vals_b = df.loc[mask_b, col].dropna().values

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        stat, pval = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))

        # Log2 fold change (for already log-transformed data, this is difference)
        log2fc = mean_a - mean_b

        results.append({
            "feature": col,
            "statistic": stat,
            "p_value": pval,
            f"mean_{group_a}": mean_a,
            f"mean_{group_b}": mean_b,
            "log2fc": log2fc,
        })

    result_df = pd.DataFrame(results)

    # FDR correction (Benjamini-Hochberg)
    if len(result_df) > 0:
        result_df = result_df.sort_values("p_value")
        result_df["fdr"] = _benjamini_hochberg(result_df["p_value"].values)

    return result_df


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return pvalues

    # Sort p-values and compute adjusted values
    order = np.argsort(pvalues)
    sorted_pvals = pvalues[order]

    # BH formula: q_i = p_i * n / rank_i (1-indexed)
    adjusted = sorted_pvals * n / (np.arange(1, n + 1))

    # Enforce monotonicity: walk from end to start, take cumulative minimum
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    # Unsort back to original order
    fdr = np.empty(n)
    fdr[order] = adjusted
    return fdr


def cliffs_delta(
    df: pd.DataFrame,
    labels: pd.Series,
    group_a: str = "CD",
    group_b: str = "nonIBD",
) -> pd.Series:
    """Compute Cliff's delta effect size for each feature.

    Cliff's delta ranges from -1 to 1:
    - |d| < 0.147: negligible
    - |d| < 0.33: small
    - |d| < 0.474: medium
    - |d| >= 0.474: large

    Returns Series indexed by feature name.
    """
    common = df.index.intersection(labels.index)
    df = df.loc[common]
    labels = labels.loc[common]

    mask_a = labels == group_a
    mask_b = labels == group_b

    deltas = {}
    for col in df.columns:
        vals_a = df.loc[mask_a, col].dropna().values
        vals_b = df.loc[mask_b, col].dropna().values
        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        # Cliff's delta: proportion of concordant - discordant pairs
        n_pairs = len(vals_a) * len(vals_b)
        if n_pairs == 0:
            continue
        greater = sum(1 for a in vals_a for b in vals_b if a > b)
        less = sum(1 for a in vals_a for b in vals_b if a < b)
        deltas[col] = (greater - less) / n_pairs

    return pd.Series(deltas, name="cliffs_delta")


# ── Known Feature Extraction ──


def extract_known_features(
    results: pd.DataFrame,
    known_list: list[str],
    omic_type: str = "species",
) -> pd.DataFrame:
    """Extract results for known CD-relevant features.

    Performs fuzzy matching on feature names (case-insensitive substring).
    """
    matched = []
    for known in known_list:
        for _, row in results.iterrows():
            feature = str(row["feature"])
            if known.lower() in feature.lower():
                row_dict = row.to_dict()
                row_dict["known_feature"] = known
                matched.append(row_dict)
                break  # Take first match per known feature

    if matched:
        df = pd.DataFrame(matched)
        logger.info(
            "%s: matched %d / %d known features",
            omic_type, len(df), len(known_list),
        )
        return df

    logger.warning("%s: no known features matched", omic_type)
    return pd.DataFrame()


# ── Visualization ──


def volcano_plot(
    results: pd.DataFrame,
    fdr_threshold: float = 0.05,
    fc_threshold: float = 1.0,
    title: str = "Differential Abundance",
    output_path: Path | None = None,
    top_n_labels: int = 15,
) -> None:
    """Volcano plot of differential analysis results.

    Parameters
    ----------
    results : DataFrame
        Must contain 'log2fc', 'fdr', 'feature' columns.
    fdr_threshold : float
        FDR significance threshold (default: 0.05).
    fc_threshold : float
        Log2 fold-change threshold for coloring (default: 1.0).
    title : str
        Plot title.
    output_path : Path, optional
        Save plot if provided.
    top_n_labels : int
        Number of top features to label.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "fdr" not in results.columns or "log2fc" not in results.columns:
        logger.warning("Missing 'fdr' or 'log2fc' columns — cannot plot volcano")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    neg_log10_fdr = -np.log10(results["fdr"].clip(lower=1e-300))
    log2fc = results["log2fc"]

    # Color by significance and fold change
    sig_up = (results["fdr"] < fdr_threshold) & (log2fc > fc_threshold)
    sig_down = (results["fdr"] < fdr_threshold) & (log2fc < -fc_threshold)
    ns = ~sig_up & ~sig_down

    ax.scatter(log2fc[ns], neg_log10_fdr[ns], c="gray", alpha=0.4, s=20, label="NS")
    ax.scatter(log2fc[sig_up], neg_log10_fdr[sig_up], c="red", alpha=0.6, s=30, label="Up in CD")
    ax.scatter(log2fc[sig_down], neg_log10_fdr[sig_down], c="blue", alpha=0.6, s=30, label="Down in CD")

    # Label top features
    top = results.nsmallest(top_n_labels, "fdr")
    for _, row in top.iterrows():
        feature_name = str(row["feature"])
        # Shorten long names
        if len(feature_name) > 30:
            feature_name = feature_name[:27] + "..."
        ax.annotate(
            feature_name,
            (row["log2fc"], -np.log10(max(row["fdr"], 1e-300))),
            fontsize=7,
            alpha=0.8,
        )

    ax.axhline(-np.log10(fdr_threshold), ls="--", color="gray", alpha=0.5)
    ax.axvline(fc_threshold, ls="--", color="gray", alpha=0.3)
    ax.axvline(-fc_threshold, ls="--", color="gray", alpha=0.3)

    ax.set_xlabel("Log2 Fold Change (CD vs nonIBD)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(title)
    ax.legend()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved volcano plot: %s", output_path)
    plt.close(fig)


# ── Differential Analysis Pipeline ──


class DifferentialAnalysis:
    """Run differential analysis on a single omic layer."""

    def __init__(
        self,
        group_a: str = "CD",
        group_b: str = "nonIBD",
        fdr_threshold: float = 0.05,
    ) -> None:
        self.group_a = group_a
        self.group_b = group_b
        self.fdr_threshold = fdr_threshold

    def run(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """Run Wilcoxon rank-sum test with FDR correction and effect sizes.

        Parameters
        ----------
        df : DataFrame
            Normalized feature matrix (samples × features).
        labels : Series
            Diagnosis labels indexed by sample ID.

        Returns
        -------
        DataFrame with test results sorted by FDR.
        """
        # Wilcoxon rank-sum test
        results = wilcoxon_ranksum_test(
            df, labels, group_a=self.group_a, group_b=self.group_b
        )

        if len(results) == 0:
            logger.warning("No features tested")
            return results

        # Add Cliff's delta effect sizes
        deltas = cliffs_delta(
            df, labels, group_a=self.group_a, group_b=self.group_b
        )
        results = results.merge(
            deltas.reset_index().rename(columns={"index": "feature"}),
            on="feature",
            how="left",
        )

        # Classify effect size
        results["effect_magnitude"] = results["cliffs_delta"].abs().apply(
            lambda d: "large" if d >= 0.474
            else "medium" if d >= 0.33
            else "small" if d >= 0.147
            else "negligible"
        )

        n_sig = (results["fdr"] < self.fdr_threshold).sum()
        logger.info(
            "Results: %d features tested, %d significant (FDR < %.2f)",
            len(results), n_sig, self.fdr_threshold,
        )

        return results.sort_values("fdr")

    def run_all_omics(
        self,
        species: pd.DataFrame,
        metabolomics: pd.DataFrame,
        labels: pd.Series,
        output_dir: Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run differential analysis on species and metabolomics.

        Applies feature importance weighting: bacterial features are
        given proportionally higher weight per the pediatric 6-omics
        importance hierarchy (bacteria 40%, metabolites 22%).

        Returns dict with 'species' and 'metabolomics' result DataFrames.
        """
        output_dir = output_dir or OUTPUT_DIR

        results: dict[str, pd.DataFrame] = {}

        # Species differential analysis
        logger.info("=== Species Differential Analysis ===")
        species_results = self.run(species, labels)
        species_results["omic"] = "species"
        species_results["importance_weight"] = 0.40  # 40% per hierarchy
        results["species"] = species_results

        # Extract known CD species
        known_species = extract_known_features(
            species_results, KNOWN_CD_SPECIES, omic_type="species"
        )
        if len(known_species) > 0:
            logger.info("Known CD species found:\n%s", known_species[["known_feature", "log2fc", "fdr", "cliffs_delta"]].to_string())

        # Metabolomics differential analysis
        logger.info("=== Metabolomics Differential Analysis ===")
        metab_results = self.run(metabolomics, labels)
        metab_results["omic"] = "metabolomics"
        metab_results["importance_weight"] = 0.22  # 22% per hierarchy
        results["metabolomics"] = metab_results

        # Extract known CD metabolites
        known_metab = extract_known_features(
            metab_results, KNOWN_CD_METABOLITES, omic_type="metabolomics"
        )
        if len(known_metab) > 0:
            logger.info("Known CD metabolites found:\n%s", known_metab[["known_feature", "log2fc", "fdr", "cliffs_delta"]].to_string())

        # Generate volcano plots
        volcano_plot(
            species_results,
            title="Species: CD vs nonIBD",
            output_path=output_dir / "volcano_species.png",
        )
        volcano_plot(
            metab_results,
            title="Metabolomics: CD vs nonIBD",
            output_path=output_dir / "volcano_metabolomics.png",
        )

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        species_results.to_csv(output_dir / "differential_species.csv", index=False)
        metab_results.to_csv(output_dir / "differential_metabolomics.csv", index=False)
        if len(known_species) > 0:
            known_species.to_csv(output_dir / "known_cd_species.csv", index=False)
        if len(known_metab) > 0:
            known_metab.to_csv(output_dir / "known_cd_metabolites.csv", index=False)

        logger.info("Results saved to %s", output_dir)
        return results
