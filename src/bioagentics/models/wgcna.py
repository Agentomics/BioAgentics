"""Weighted Gene Co-expression Network Analysis (WGCNA).

Identifies gene co-expression modules correlated with autoimmune
neuropsychiatric phenotype. Uses AnnData input to match the pipeline.

Steps:
1. Filter to top variable genes (memory-safe for 8GB systems)
2. Soft-thresholding power selection (scale-free topology)
3. Compute adjacency and Topological Overlap Matrix (TOM)
4. Hierarchical clustering + dynamic tree cutting -> modules
5. Compute module eigengenes (first PC per module)
6. Correlate module eigengenes with phenotype traits
7. Identify hub genes in trait-correlated modules

Usage:
    uv run python -m bioagentics.models.wgcna input.h5ad \\
        --condition-key condition --dest output/
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "transcriptomic-biomarker-panel"

# Cap genes to keep memory under control on 8GB systems.
# 5000 genes -> ~200MB per n_genes^2 matrix, ~600MB peak total.
MAX_GENES = 5000


@dataclass
class WGCNAResult:
    """Container for WGCNA analysis results."""

    power: int
    fit_table: pd.DataFrame
    modules: pd.DataFrame
    eigengenes: pd.DataFrame
    trait_correlations: pd.DataFrame
    hub_genes: dict[int, list[dict]]
    n_modules: int = 0
    n_trait_significant: int = 0

    def __post_init__(self):
        if self.n_modules == 0:
            self.n_modules = len(set(self.modules["module"]) - {0})


def select_soft_threshold(
    expr: np.ndarray,
    gene_names: list[str],
    powers: list[int] | None = None,
    r2_cutoff: float = 0.85,
) -> tuple[int, pd.DataFrame]:
    """Select soft-thresholding power for scale-free topology.

    Parameters
    ----------
    expr : np.ndarray
        Samples x genes expression matrix.
    gene_names : list[str]
        Gene names (for logging only).
    powers : list[int], optional
        Candidate powers to test. Default 1-20.
    r2_cutoff : float
        R^2 threshold for scale-free fit.

    Returns
    -------
    Tuple of (selected_power, fit_table DataFrame).
    """
    if powers is None:
        powers = list(range(1, 21))

    cor_matrix = np.corrcoef(expr, rowvar=False)
    cor_matrix = np.nan_to_num(cor_matrix, nan=0.0)

    records = []
    for beta in powers:
        adj = np.abs(cor_matrix) ** beta
        k = adj.sum(axis=1) - 1  # connectivity excluding self
        k = k[k > 0]

        if len(k) < 10:
            records.append({"power": beta, "r2": 0.0, "mean_k": 0.0, "median_k": 0.0})
            continue

        hist, bin_edges = np.histogram(np.log10(k), bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = hist > 0
        if mask.sum() < 3:
            records.append({
                "power": beta, "r2": 0.0,
                "mean_k": float(k.mean()), "median_k": float(np.median(k)),
            })
            continue

        log_p = np.log10(hist[mask] / hist[mask].sum())
        slope, _intercept, r_value, _, _ = stats.linregress(bin_centers[mask], log_p)
        r2 = r_value ** 2 if slope < 0 else 0.0

        records.append({
            "power": beta,
            "r2": float(r2),
            "slope": float(slope),
            "mean_k": float(k.mean()),
            "median_k": float(np.median(k)),
        })

    fit_df = pd.DataFrame(records)

    above = fit_df[fit_df["r2"] >= r2_cutoff]
    if not above.empty:
        selected = int(above.iloc[0]["power"])
    else:
        selected = int(fit_df.loc[fit_df["r2"].idxmax(), "power"])

    logger.info(
        "Selected soft-threshold power: %d (R^2=%.3f, %d genes)",
        selected, fit_df[fit_df["power"] == selected]["r2"].values[0], len(gene_names),
    )
    return selected, fit_df


def compute_tom(cor_matrix: np.ndarray, power: int) -> np.ndarray:
    """Compute Topological Overlap Matrix from correlation matrix.

    TOM(i,j) = (l_ij + a_ij) / (min(k_i, k_j) + 1 - a_ij)
    where l_ij = sum_u(a_iu * a_uj), a = |cor|^power.

    Parameters
    ----------
    cor_matrix : np.ndarray
        Gene-gene correlation matrix.
    power : int
        Soft-thresholding power.

    Returns
    -------
    TOM matrix (symmetric, values in [0, 1]).
    """
    adj = np.abs(cor_matrix) ** power
    np.fill_diagonal(adj, 0)
    k = adj.sum(axis=1)

    l_matrix = adj @ adj
    numerator = l_matrix + adj

    k_min = np.minimum(k[:, None], k[None, :])
    denominator = k_min + 1 - adj
    denominator = np.maximum(denominator, 1e-10)

    tom = numerator / denominator
    np.fill_diagonal(tom, 1.0)
    return np.clip(tom, 0, 1)


def identify_modules(
    tom: np.ndarray,
    gene_names: list[str],
    min_module_size: int = 30,
    cut_height: float = 0.0,
) -> pd.DataFrame:
    """Cluster genes into modules using hierarchical clustering on 1-TOM.

    Parameters
    ----------
    tom : np.ndarray
        Topological Overlap Matrix.
    gene_names : list[str]
        Gene names matching TOM dimensions.
    min_module_size : int
        Minimum genes per module. Smaller clusters go to module 0 (unassigned).
    cut_height : float
        Fixed cut height. If <= 0, uses adaptive search.

    Returns
    -------
    DataFrame with columns: gene_symbol, module.
    """
    dist = 1 - tom
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    if cut_height <= 0:
        best_labels = None
        best_n_modules = 0
        for h in np.arange(0.95, 0.4, -0.05):
            labels = fcluster(Z, t=h, criterion="distance")
            _unique, counts = np.unique(labels, return_counts=True)
            valid = (counts >= min_module_size).sum()
            if valid > best_n_modules:
                best_n_modules = valid
                best_labels = labels.copy()
                if valid >= 3:
                    break
        if best_labels is None:
            best_labels = fcluster(Z, t=0.7, criterion="distance")
        labels = best_labels
    else:
        labels = fcluster(Z, t=cut_height, criterion="distance")

    # Merge small modules into module 0
    unique, counts = np.unique(labels, return_counts=True)
    for sm in unique[counts < min_module_size]:
        labels[labels == sm] = 0

    # Renumber from 1
    unique_labels = sorted(set(labels) - {0})
    label_map = {old: new for new, old in enumerate(unique_labels, 1)}
    label_map[0] = 0
    labels = np.array([label_map.get(lab, 0) for lab in labels])

    result = pd.DataFrame({"gene_symbol": gene_names, "module": labels})
    n_modules = len(set(labels) - {0})
    n_unassigned = (labels == 0).sum()
    logger.info("Identified %d modules (%d genes unassigned)", n_modules, n_unassigned)
    return result


def compute_module_eigengenes(
    expr: np.ndarray,
    gene_names: list[str],
    modules: pd.DataFrame,
) -> pd.DataFrame:
    """Compute module eigengenes (first PC of each module).

    Parameters
    ----------
    expr : np.ndarray
        Samples x genes expression matrix.
    gene_names : list[str]
        Gene names matching expr columns.
    modules : pd.DataFrame
        Module assignments from identify_modules.

    Returns
    -------
    DataFrame: samples x modules (columns named ME1, ME2, ...).
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    eigengenes = {}

    for mod_id in sorted(modules["module"].unique()):
        if mod_id == 0:
            continue
        mod_genes = modules[modules["module"] == mod_id]["gene_symbol"].tolist()
        idx = [gene_to_idx[g] for g in mod_genes if g in gene_to_idx]
        if len(idx) < 3:
            continue

        mod_expr = expr[:, idx]
        mod_expr = np.nan_to_num(mod_expr, nan=0.0)

        pca = PCA(n_components=1)
        me = pca.fit_transform(mod_expr)[:, 0]
        eigengenes[f"ME{mod_id}"] = me

    return pd.DataFrame(eigengenes)


def correlate_with_traits(
    eigengenes: pd.DataFrame,
    traits: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate module eigengenes with phenotype traits.

    Parameters
    ----------
    eigengenes : pd.DataFrame
        Module eigengenes (samples x modules).
    traits : pd.DataFrame
        Numeric trait matrix (samples x traits). Non-numeric columns skipped.

    Returns
    -------
    DataFrame with columns: module, trait, correlation, pvalue.
    """
    rows = []
    for me_col in eigengenes.columns:
        me = eigengenes[me_col].values
        for trait_col in traits.columns:
            t = traits[trait_col].values.astype(float)
            valid = np.isfinite(me) & np.isfinite(t)
            if valid.sum() < 5:
                continue
            r, p = stats.pearsonr(me[valid], t[valid])
            rows.append({
                "module": me_col,
                "trait": trait_col,
                "correlation": float(r),
                "pvalue": float(p),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pvalue").reset_index(drop=True)
    return df


def find_hub_genes(
    expr: np.ndarray,
    gene_names: list[str],
    modules: pd.DataFrame,
    eigengenes: pd.DataFrame,
    module_id: int,
    n_top: int = 10,
) -> list[dict]:
    """Find hub genes (highest module membership) for a module.

    Module membership = |correlation with module eigengene|.

    Parameters
    ----------
    expr : np.ndarray
        Samples x genes expression matrix.
    gene_names : list[str]
        Gene names.
    modules : pd.DataFrame
        Module assignments.
    eigengenes : pd.DataFrame
        Module eigengenes.
    module_id : int
        Module to analyze.
    n_top : int
        Number of top hub genes to return.

    Returns
    -------
    List of dicts with gene_symbol, module_membership, mm_pvalue.
    """
    me_col = f"ME{module_id}"
    if me_col not in eigengenes.columns:
        return []

    me = eigengenes[me_col].values
    mod_genes = modules[modules["module"] == module_id]["gene_symbol"].tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    hubs = []
    for gene in mod_genes:
        if gene not in gene_to_idx:
            continue
        g_expr = expr[:, gene_to_idx[gene]]
        valid = np.isfinite(g_expr) & np.isfinite(me)
        if valid.sum() < 5:
            continue
        r, p = stats.pearsonr(g_expr[valid], me[valid])
        hubs.append({
            "gene_symbol": gene,
            "module_membership": abs(float(r)),
            "mm_pvalue": float(p),
        })

    hubs.sort(key=lambda x: -x["module_membership"])
    return hubs[:n_top]


def _filter_variable_genes(
    expr: np.ndarray,
    gene_names: list[str],
    max_genes: int = MAX_GENES,
) -> tuple[np.ndarray, list[str]]:
    """Keep top variable genes to control memory usage."""
    if expr.shape[1] <= max_genes:
        return expr, gene_names

    variances = np.nanvar(expr, axis=0)
    top_idx = np.argsort(variances)[-max_genes:]
    top_idx.sort()

    logger.info(
        "Filtered to top %d/%d variable genes for WGCNA (memory constraint)",
        max_genes, expr.shape[1],
    )
    return expr[:, top_idx], [gene_names[i] for i in top_idx]


def run_wgcna(
    adata: ad.AnnData,
    condition_key: str = "condition",
    min_module_size: int = 30,
    max_genes: int = MAX_GENES,
    dest_dir: Path | None = None,
) -> WGCNAResult:
    """Full WGCNA pipeline on AnnData input.

    Parameters
    ----------
    adata : AnnData
        Normalized expression data.
    condition_key : str
        Column in obs for phenotype trait correlation.
    min_module_size : int
        Minimum genes per module.
    max_genes : int
        Maximum genes to include (top by variance).
    dest_dir : Path, optional
        Output directory. If None, uses default OUTPUT_DIR.

    Returns
    -------
    WGCNAResult with all analysis outputs.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Prepare expression matrix
    X = np.array(adata.X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    gene_names = list(adata.var_names)

    # Filter to top variable genes
    X, gene_names = _filter_variable_genes(X, gene_names, max_genes)

    logger.info("WGCNA: %d samples x %d genes", X.shape[0], X.shape[1])

    # 1. Select soft threshold
    power, fit_table = select_soft_threshold(X, gene_names)
    fit_table.to_csv(dest_dir / "wgcna_soft_threshold.csv", index=False)

    # 2. Compute correlation and TOM
    logger.info("Computing correlation matrix...")
    cor = np.corrcoef(X, rowvar=False).astype(np.float32)
    cor = np.nan_to_num(cor, nan=0.0)

    logger.info("Computing TOM (power=%d)...", power)
    tom = compute_tom(cor, power)

    # Free correlation matrix
    del cor

    # 3. Identify modules
    modules = identify_modules(tom, gene_names, min_module_size=min_module_size)
    modules.to_csv(dest_dir / "wgcna_modules.csv", index=False)

    # Free TOM
    del tom

    # 4. Compute module eigengenes
    eigengenes = compute_module_eigengenes(X, gene_names, modules)
    if not eigengenes.empty:
        eigengenes.to_csv(dest_dir / "wgcna_eigengenes.csv", index=False)

    # 5. Correlate with phenotype traits
    traits = _build_trait_matrix(adata, condition_key)
    trait_corr = correlate_with_traits(eigengenes, traits)
    if not trait_corr.empty:
        trait_corr.to_csv(dest_dir / "wgcna_trait_correlations.csv", index=False)

    n_sig = (trait_corr["pvalue"] < 0.05).sum() if not trait_corr.empty else 0
    logger.info("Module-trait correlations: %d significant (p < 0.05)", n_sig)

    # 6. Find hub genes for trait-correlated modules
    hub_genes: dict[int, list[dict]] = {}
    if not trait_corr.empty:
        sig_modules = trait_corr[trait_corr["pvalue"] < 0.05]["module"].unique()
        for me_col in sig_modules:
            mod_id = int(me_col.replace("ME", ""))
            hubs = find_hub_genes(X, gene_names, modules, eigengenes, mod_id)
            if hubs:
                hub_genes[mod_id] = hubs

    if hub_genes:
        hub_path = dest_dir / "wgcna_hub_genes.json"
        with open(hub_path, "w") as f:
            json.dump(hub_genes, f, indent=2, default=str)
        logger.info("Saved hub genes for %d modules", len(hub_genes))

    # 7. Generate module-trait heatmap
    _plot_module_trait_heatmap(trait_corr, dest_dir / "wgcna_trait_heatmap.png")

    result = WGCNAResult(
        power=power,
        fit_table=fit_table,
        modules=modules,
        eigengenes=eigengenes,
        trait_correlations=trait_corr,
        hub_genes=hub_genes,
        n_trait_significant=n_sig,
    )
    logger.info(
        "WGCNA complete: %d modules, %d trait-correlated",
        result.n_modules, n_sig,
    )
    return result


def _build_trait_matrix(adata: ad.AnnData, condition_key: str) -> pd.DataFrame:
    """Build numeric trait matrix from AnnData obs for module-trait correlation.

    Encodes the condition as a binary variable. Also includes any numeric
    columns already present in obs (e.g., age, severity scores).
    """
    traits = pd.DataFrame(index=adata.obs_names)

    if condition_key in adata.obs.columns:
        cond = adata.obs[condition_key].astype(str)
        unique_vals = sorted(cond.unique())
        if len(unique_vals) == 2:
            traits[condition_key] = (cond == unique_vals[1]).astype(float)
        else:
            for val in unique_vals:
                traits[f"{condition_key}_{val}"] = (cond == val).astype(float)

    # Include existing numeric columns
    for col in adata.obs.columns:
        if col == condition_key:
            continue
        if adata.obs[col].dtype.kind in ("i", "f"):
            traits[col] = adata.obs[col].astype(float).values

    return traits


def _plot_module_trait_heatmap(
    trait_corr: pd.DataFrame,
    save_path: Path,
) -> None:
    """Generate heatmap of module-trait correlations."""
    if trait_corr.empty:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = trait_corr.pivot_table(
        index="module", columns="trait", values="correlation", aggfunc="first",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Module-Trait Correlations")

    # Annotate with correlation values
    pval_pivot = trait_corr.pivot_table(
        index="module", columns="trait", values="pvalue", aggfunc="first",
    )
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            r = pivot.values[i, j]
            p = pval_pivot.values[i, j] if not np.isnan(pval_pivot.values[i, j]) else 1.0
            star = "*" if p < 0.05 else ""
            ax.text(j, i, f"{r:.2f}{star}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WGCNA co-expression network analysis"
    )
    parser.add_argument("input", type=Path, help="Input h5ad file")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--min-module-size", type=int, default=30)
    parser.add_argument("--max-genes", type=int, default=MAX_GENES)
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    adata = ad.read_h5ad(args.input)
    result = run_wgcna(
        adata,
        condition_key=args.condition_key,
        min_module_size=args.min_module_size,
        max_genes=args.max_genes,
        dest_dir=args.dest,
    )

    print(f"\nWGCNA complete:")
    print(f"  Power: {result.power}")
    print(f"  Modules: {result.n_modules}")
    print(f"  Trait-correlated: {result.n_trait_significant}")
    for mod_id, hubs in result.hub_genes.items():
        top3 = ", ".join(h["gene_symbol"] for h in hubs[:3])
        print(f"  Module {mod_id} hubs: {top3}")


if __name__ == "__main__":
    main()
