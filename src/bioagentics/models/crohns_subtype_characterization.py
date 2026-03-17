"""Subtype characterization: features, clinical associations, pathways, networks.

For each discovered CD subtype, characterizes:
1. Top driving species and metabolites (from integration loadings + differential)
2. Clinical feature associations (Montreal, CRP, calprotectin, treatment)
3. Pathway enrichment (MetaCyc/KEGG mapping)
4. Metabolic axis mapping (Bifidobacterium-TCDCA, Tryptophan-NAD, BCAA)
5. Species-metabolite correlation networks per subtype
6. Subtype-specific biomarker panels

Usage::

    from bioagentics.models.crohns_subtype_characterization import SubtypeCharacterization

    charact = SubtypeCharacterization()
    results = charact.characterize(
        subtypes=labels, species=species_qc, metabolomics=metab_qc,
        metadata=metadata, species_loadings=sp_load, metabolite_loadings=mb_load,
    )
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

# Metabolic axis marker features
METABOLIC_AXES = {
    "Bifidobacterium-TCDCA": {
        "species_markers": ["bifidobacterium", "bacteroides"],
        "metabolite_markers": [
            "tcdca", "taurochenodeoxychol", "bile", "lithochol",
            "deoxychol",
        ],
        "therapeutic": "FMT/probiotics",
    },
    "Tryptophan-NAD": {
        "species_markers": ["tryptophan", "kynurenine"],
        "metabolite_markers": [
            "tryptophan", "kynurenine", "quinolinate", "kynurenate",
            "nad", "nicotinamide", "indole",
        ],
        "therapeutic": "Dietary saccharide modification",
    },
    "BCAA": {
        "species_markers": [],
        "metabolite_markers": [
            "leucine", "valine", "isoleucine",
        ],
        "therapeutic": "BCAA supplementation",
    },
}


# ── Feature-Level Characterization ──


def top_features_per_subtype(
    data: pd.DataFrame,
    subtypes: pd.Series,
    n_top: int = 20,
) -> dict[int, pd.DataFrame]:
    """Identify top differentially expressed features per subtype.

    For each subtype, performs Kruskal-Wallis test against all others
    and ranks by effect size.
    """
    results: dict[int, pd.DataFrame] = {}
    shared = data.index.intersection(subtypes.index)
    data = data.loc[shared]
    subtypes = subtypes.loc[shared]

    for sub in sorted(subtypes.unique()):
        mask = subtypes == sub
        features_list = []
        for col in data.columns:
            in_group = data.loc[mask, col].dropna()
            out_group = data.loc[~mask, col].dropna()
            if len(in_group) < 2 or len(out_group) < 2:
                continue

            _, pval = stats.mannwhitneyu(
                in_group, out_group, alternative="two-sided"
            )
            mean_diff = float(in_group.mean() - out_group.mean())
            features_list.append({
                "feature": col,
                "mean_in_subtype": float(in_group.mean()),
                "mean_out_subtype": float(out_group.mean()),
                "mean_diff": mean_diff,
                "p_value": pval,
            })

        if features_list:
            df = pd.DataFrame(features_list)
            # FDR correction
            n = len(df)
            order = np.argsort(df["p_value"].values)
            fdr = np.clip(
                np.minimum.accumulate(
                    (df["p_value"].values[order] * n / (np.arange(1, n + 1)))[::-1]
                )[::-1],
                0, 1,
            )
            df.loc[df.index[order], "fdr"] = fdr
            df = df.sort_values("p_value")
            results[sub] = df.head(n_top)

    return results


# ── Clinical Associations ──


def clinical_associations(
    subtypes: pd.Series,
    metadata: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    continuous_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Test associations between subtypes and clinical features.

    Categorical: chi-square test.
    Continuous: Kruskal-Wallis test.

    Returns dict with 'categorical' and 'continuous' result DataFrames.
    """
    if categorical_cols is None:
        categorical_cols = []
        for col in metadata.columns:
            cl = col.lower()
            if any(
                kw in cl
                for kw in [
                    "montreal", "location", "behavior", "sex", "gender",
                    "smoking", "treatment", "medication",
                ]
            ):
                categorical_cols.append(col)

    if continuous_cols is None:
        continuous_cols = []
        for col in metadata.columns:
            cl = col.lower()
            if any(
                kw in cl
                for kw in [
                    "crp", "calprotectin", "age", "bmi", "hbi",
                    "albumin", "esr",
                ]
            ):
                continuous_cols.append(col)

    shared = subtypes.index.intersection(metadata.index)
    subtypes = subtypes.loc[shared]
    metadata = metadata.loc[shared]

    # Categorical tests (chi-square)
    cat_results = []
    for col in categorical_cols:
        if col not in metadata.columns:
            continue
        ct = pd.crosstab(subtypes, metadata[col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, pval, dof, _ = stats.chi2_contingency(ct)
        cat_results.append({
            "feature": col,
            "test": "chi-square",
            "statistic": float(chi2),
            "p_value": float(pval),
            "dof": int(dof),
        })

    # Continuous tests (Kruskal-Wallis)
    cont_results = []
    for col in continuous_cols:
        if col not in metadata.columns:
            continue
        groups = []
        for sub in sorted(subtypes.unique()):
            vals = metadata.loc[subtypes == sub, col].dropna()
            if len(vals) > 0:
                groups.append(vals.values)

        if len(groups) < 2:
            continue

        h_stat, pval = stats.kruskal(*groups)
        cont_results.append({
            "feature": col,
            "test": "kruskal-wallis",
            "statistic": float(h_stat),
            "p_value": float(pval),
        })

    return {
        "categorical": pd.DataFrame(cat_results) if cat_results else pd.DataFrame(),
        "continuous": pd.DataFrame(cont_results) if cont_results else pd.DataFrame(),
    }


# ── Metabolic Axis Mapping ──


def map_subtypes_to_axes(
    subtypes: pd.Series,
    species: pd.DataFrame,
    metabolomics: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """Map each subtype to metabolic axis scores.

    For each subtype, computes the mean abundance of axis marker
    features to determine which metabolic axis is dominant.
    """
    shared = subtypes.index.intersection(species.index).intersection(
        metabolomics.index
    )
    subtypes = subtypes.loc[shared]
    species = species.loc[shared]
    metabolomics = metabolomics.loc[shared]

    result: dict[int, dict[str, float]] = {}

    for sub in sorted(subtypes.unique()):
        mask = subtypes == sub
        axis_scores: dict[str, float] = {}

        for axis_name, markers in METABOLIC_AXES.items():
            score = 0.0
            n_matched = 0

            # Species markers
            for sp_col in species.columns:
                sp_lower = sp_col.lower()
                if any(m in sp_lower for m in markers["species_markers"]):
                    score += float(species.loc[mask, sp_col].mean())
                    n_matched += 1

            # Metabolite markers
            for mb_col in metabolomics.columns:
                mb_lower = mb_col.lower()
                if any(m in mb_lower for m in markers["metabolite_markers"]):
                    score += float(metabolomics.loc[mask, mb_col].mean())
                    n_matched += 1

            axis_scores[axis_name] = score / n_matched if n_matched > 0 else 0.0

        result[sub] = axis_scores

    return result


# ── Correlation Networks ──


def subtype_correlation_network(
    species: pd.DataFrame,
    metabolomics: pd.DataFrame,
    subtypes: pd.Series,
    subtype: int,
    method: str = "spearman",
    fdr_threshold: float = 0.05,
    min_corr: float = 0.3,
) -> pd.DataFrame:
    """Compute species-metabolite correlation network for a subtype.

    Returns DataFrame with columns: species, metabolite, correlation,
    p_value, fdr.
    """
    shared = subtypes.index.intersection(species.index).intersection(
        metabolomics.index
    )
    mask = subtypes.loc[shared] == subtype
    sp_sub = species.loc[shared].loc[mask]
    mb_sub = metabolomics.loc[shared].loc[mask]

    if len(sp_sub) < 5:
        logger.warning(
            "Subtype %d: only %d samples, skipping network", subtype, len(sp_sub)
        )
        return pd.DataFrame()

    edges = []
    for sp_col in sp_sub.columns:
        for mb_col in mb_sub.columns:
            sp_vals = sp_sub[sp_col].dropna()
            mb_vals = mb_sub[mb_col].dropna()
            common = sp_vals.index.intersection(mb_vals.index)
            if len(common) < 5:
                continue

            if method == "spearman":
                corr, pval = stats.spearmanr(
                    sp_vals.loc[common], mb_vals.loc[common]
                )
            else:
                corr, pval = stats.pearsonr(
                    sp_vals.loc[common], mb_vals.loc[common]
                )

            if abs(corr) >= min_corr:
                edges.append({
                    "species": sp_col,
                    "metabolite": mb_col,
                    "correlation": float(corr),
                    "p_value": float(pval),
                })

    if not edges:
        return pd.DataFrame()

    df = pd.DataFrame(edges)

    # FDR correction
    n = len(df)
    order = np.argsort(df["p_value"].values)
    fdr = np.clip(
        np.minimum.accumulate(
            (df["p_value"].values[order] * n / (np.arange(1, n + 1)))[::-1]
        )[::-1],
        0, 1,
    )
    df.loc[df.index[order], "fdr"] = fdr

    # Filter by FDR
    df = df[df["fdr"] < fdr_threshold].sort_values("fdr")

    return df


# ── Biomarker Panel ──


def build_biomarker_panel(
    top_features: dict[int, pd.DataFrame],
    axis_mapping: dict[int, dict[str, float]],
    max_markers: int = 10,
) -> dict[int, dict]:
    """Build subtype-specific biomarker panels.

    Combines top differential features with metabolic axis classification.
    """
    panels: dict[int, dict] = {}

    for sub in top_features:
        tf = top_features[sub]
        sig = tf[tf["fdr"] < 0.05] if "fdr" in tf.columns else tf.head(max_markers)
        markers = list(sig.head(max_markers)["feature"])

        # Dominant axis
        axes = axis_mapping.get(sub, {})
        dominant_axis = max(axes, key=lambda a: abs(axes[a])) if axes else "unknown"

        panels[sub] = {
            "markers": markers,
            "n_markers": len(markers),
            "dominant_axis": dominant_axis,
            "therapeutic_suggestion": METABOLIC_AXES.get(
                dominant_axis, {}
            ).get("therapeutic", ""),
            "axis_scores": axes,
        }

    return panels


# ── Full Characterization Pipeline ──


class SubtypeCharacterization:
    """Complete subtype characterization pipeline."""

    def __init__(
        self,
        n_top_features: int = 20,
        fdr_threshold: float = 0.05,
        network_min_corr: float = 0.3,
    ) -> None:
        self.n_top_features = n_top_features
        self.fdr_threshold = fdr_threshold
        self.network_min_corr = network_min_corr

    def characterize(
        self,
        subtypes: pd.Series,
        species: pd.DataFrame,
        metabolomics: pd.DataFrame,
        metadata: pd.DataFrame,
        species_loadings: pd.DataFrame | None = None,
        metabolite_loadings: pd.DataFrame | None = None,
        output_dir: Path | None = None,
    ) -> dict:
        """Run full characterization pipeline.

        Returns dict with all characterization results.
        """
        output_dir = output_dir or OUTPUT_DIR
        n_subtypes = subtypes.nunique()
        logger.info(
            "Characterizing %d subtypes across %d participants",
            n_subtypes, len(subtypes),
        )

        # 1. Top features per subtype
        logger.info("=== Top Features ===")
        top_species = top_features_per_subtype(
            species, subtypes, n_top=self.n_top_features
        )
        top_metab = top_features_per_subtype(
            metabolomics, subtypes, n_top=self.n_top_features
        )

        # 2. Clinical associations
        logger.info("=== Clinical Associations ===")
        clinical = clinical_associations(subtypes, metadata)

        # 3. Metabolic axis mapping
        logger.info("=== Metabolic Axis Mapping ===")
        axis_map = map_subtypes_to_axes(subtypes, species, metabolomics)

        # 4. Correlation networks per subtype
        logger.info("=== Correlation Networks ===")
        networks: dict[int, pd.DataFrame] = {}
        for sub in sorted(subtypes.unique()):
            net = subtype_correlation_network(
                species, metabolomics, subtypes,
                subtype=sub,
                min_corr=self.network_min_corr,
                fdr_threshold=self.fdr_threshold,
            )
            networks[sub] = net
            logger.info(
                "Subtype %d network: %d edges", sub, len(net),
            )

        # 5. Biomarker panels
        logger.info("=== Biomarker Panels ===")
        # Combine species and metabolite top features
        combined_top: dict[int, pd.DataFrame] = {}
        for sub in sorted(subtypes.unique()):
            parts = []
            if sub in top_species:
                sp = top_species[sub].copy()
                sp["omic"] = "species"
                parts.append(sp)
            if sub in top_metab:
                mb = top_metab[sub].copy()
                mb["omic"] = "metabolomics"
                parts.append(mb)
            if parts:
                combined_top[sub] = pd.concat(parts).sort_values("p_value")

        panels = build_biomarker_panel(combined_top, axis_map)

        results = {
            "top_species": top_species,
            "top_metabolites": top_metab,
            "clinical_associations": clinical,
            "axis_mapping": axis_map,
            "networks": networks,
            "biomarker_panels": panels,
        }

        # Save results
        self._save(results, output_dir)

        # Log summary
        for sub, panel in panels.items():
            logger.info(
                "Subtype %d: %d markers, axis=%s, therapeutic=%s",
                sub,
                panel["n_markers"],
                panel["dominant_axis"],
                panel["therapeutic_suggestion"],
            )

        return results

    def _save(self, results: dict, output_dir: Path) -> None:
        """Save characterization results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Top features
        for sub, df in results["top_species"].items():
            df.to_csv(output_dir / f"top_species_subtype_{sub}.csv", index=False)
        for sub, df in results["top_metabolites"].items():
            df.to_csv(output_dir / f"top_metabolites_subtype_{sub}.csv", index=False)

        # Clinical associations
        for key, df in results["clinical_associations"].items():
            if len(df) > 0:
                df.to_csv(output_dir / f"clinical_{key}.csv", index=False)

        # Axis mapping
        axis_df = pd.DataFrame(results["axis_mapping"]).T
        axis_df.index.name = "subtype"
        axis_df.to_csv(output_dir / "metabolic_axis_mapping.csv")

        # Networks
        for sub, df in results["networks"].items():
            if len(df) > 0:
                df.to_csv(
                    output_dir / f"network_subtype_{sub}.csv", index=False
                )

        # Biomarker panels
        panels_summary = []
        for sub, panel in results["biomarker_panels"].items():
            panels_summary.append({
                "subtype": sub,
                "n_markers": panel["n_markers"],
                "dominant_axis": panel["dominant_axis"],
                "therapeutic": panel["therapeutic_suggestion"],
                "markers": "; ".join(panel["markers"]),
            })
        pd.DataFrame(panels_summary).to_csv(
            output_dir / "biomarker_panels.csv", index=False
        )

        logger.info("Saved characterization results to %s", output_dir)
