"""Weighted Gene Co-expression Network Analysis (WGCNA) on AHBA CSTC data.

Pure-Python implementation of WGCNA to identify co-expression modules
in CSTC brain regions and test enrichment for TS risk genes.

Pipeline:
1. Extract AHBA expression matrix for CSTC samples (all genes)
2. Soft-thresholding power selection (scale-free topology)
3. Compute adjacency and Topological Overlap Matrix (TOM)
4. Hierarchical clustering + dynamic tree cutting → modules
5. Test TS gene enrichment per module (hypergeometric + permutation)
6. Identify hub genes in enriched modules

Usage:
    uv run python -m bioagentics.analysis.tourettes.wgcna_cstc
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import get_gene_set
from bioagentics.data.tourettes.hmba_reference import (
    get_taxonomy,
    get_marker_panel,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "cstc-circuit-expression-atlas"


def select_soft_threshold(
    expr_matrix: pd.DataFrame,
    powers: list[int] | None = None,
    r2_cutoff: float = 0.85,
) -> tuple[int, pd.DataFrame]:
    """Select soft-thresholding power for scale-free topology.

    For each candidate power, compute the signed adjacency matrix and
    assess scale-free topology fit (R^2 of log(k) vs log(p(k))).

    Returns (selected_power, fit_table).
    """
    if powers is None:
        powers = list(range(1, 21))

    cor_matrix = expr_matrix.corr().values
    n = cor_matrix.shape[0]

    records = []
    for beta in powers:
        adj = np.abs(cor_matrix) ** beta
        k = adj.sum(axis=1) - 1  # connectivity (exclude self)
        k = k[k > 0]

        if len(k) < 10:
            records.append({"power": beta, "r2": 0, "mean_k": 0, "median_k": 0})
            continue

        # Scale-free fit: log-log regression of connectivity distribution
        hist, bin_edges = np.histogram(np.log10(k), bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = hist > 0
        if mask.sum() < 3:
            records.append({"power": beta, "r2": 0, "mean_k": float(k.mean()), "median_k": float(np.median(k))})
            continue

        log_p = np.log10(hist[mask] / hist[mask].sum())
        slope, intercept, r_value, _, _ = stats.linregress(bin_centers[mask], log_p)
        r2 = r_value ** 2

        # Scale-free networks have negative slope
        if slope > 0:
            r2 = 0

        records.append({
            "power": beta,
            "r2": float(r2),
            "slope": float(slope),
            "mean_k": float(k.mean()),
            "median_k": float(np.median(k)),
        })

    fit_df = pd.DataFrame(records)

    # Select first power that exceeds R^2 cutoff
    above = fit_df[fit_df["r2"] >= r2_cutoff]
    if not above.empty:
        selected = int(above.iloc[0]["power"])
    else:
        # Fall back to power with highest R^2
        selected = int(fit_df.loc[fit_df["r2"].idxmax(), "power"])

    logger.info("Selected soft-threshold power: %d (R^2=%.3f)",
                selected, fit_df[fit_df["power"] == selected]["r2"].values[0])

    return selected, fit_df


def compute_adjacency(cor_matrix: np.ndarray, power: int) -> np.ndarray:
    """Compute signed adjacency matrix: adj = |cor|^power."""
    return np.abs(cor_matrix) ** power


def compute_tom(adjacency: np.ndarray) -> np.ndarray:
    """Compute Topological Overlap Matrix (TOM).

    TOM(i,j) = (sum_u(a_iu * a_uj) + a_ij) / (min(k_i, k_j) + 1 - a_ij)
    where k_i = sum_u(a_iu) (connectivity of node i, excluding self).
    """
    n = adjacency.shape[0]
    np.fill_diagonal(adjacency, 0)
    k = adjacency.sum(axis=1)

    # Numerator: l_ij + a_ij where l_ij = sum_u(a_iu * a_uj)
    l_matrix = adjacency @ adjacency
    numerator = l_matrix + adjacency

    # Denominator: min(k_i, k_j) + 1 - a_ij
    k_min = np.minimum(k[:, None], k[None, :])
    denominator = k_min + 1 - adjacency

    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-10)

    tom = numerator / denominator
    np.fill_diagonal(tom, 1.0)
    return np.clip(tom, 0, 1)


def identify_modules(
    tom: np.ndarray,
    gene_names: list[str],
    min_module_size: int = 30,
    cut_height: float = 0.0,
    deep_split: int = 2,
) -> pd.DataFrame:
    """Identify co-expression modules via hierarchical clustering on 1-TOM.

    Uses dynamic tree cutting approximation (fcluster with distance criterion).

    Returns DataFrame with gene_symbol and module columns.
    """
    dist = 1 - tom
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 0)

    # Condensed distance matrix for linkage
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    # Dynamic tree cutting approximation:
    # Try different cut heights to find one giving reasonable module sizes
    if cut_height <= 0:
        # Adaptive: try cuts from 0.9 to 0.5
        best_labels = None
        best_n_modules = 0
        for h in np.arange(0.95, 0.4, -0.05):
            labels = fcluster(Z, t=h, criterion="distance")
            # Count modules meeting minimum size
            unique, counts = np.unique(labels, return_counts=True)
            valid_modules = (counts >= min_module_size).sum()
            if valid_modules > best_n_modules:
                best_n_modules = valid_modules
                best_labels = labels.copy()
                if valid_modules >= 3:
                    break

        if best_labels is None:
            best_labels = fcluster(Z, t=0.7, criterion="distance")
        labels = best_labels
    else:
        labels = fcluster(Z, t=cut_height, criterion="distance")

    # Assign small modules to module 0 (grey/unassigned)
    unique, counts = np.unique(labels, return_counts=True)
    small_modules = unique[counts < min_module_size]
    for sm in small_modules:
        labels[labels == sm] = 0

    # Renumber modules starting from 1
    unique_labels = sorted(set(labels) - {0})
    label_map = {old: new for new, old in enumerate(unique_labels, 1)}
    label_map[0] = 0
    labels = np.array([label_map.get(l, 0) for l in labels])

    result = pd.DataFrame({
        "gene_symbol": gene_names,
        "module": labels,
    })

    n_modules = len(set(labels) - {0})
    n_unassigned = (labels == 0).sum()
    logger.info("Identified %d modules (%d genes unassigned)", n_modules, n_unassigned)

    return result


def compute_enrichment(
    modules: pd.DataFrame,
    ts_genes: set[str],
    n_permutations: int = 1000,
) -> list[dict]:
    """Test enrichment of TS genes in each module.

    Uses hypergeometric test and permutation-based p-value.
    """
    all_genes = set(modules["gene_symbol"])
    N = len(all_genes)  # total genes
    K = len(ts_genes & all_genes)  # TS genes in dataset

    if K == 0:
        logger.warning("No TS genes found in expression data")
        return []

    rng = np.random.default_rng(42)
    results = []

    for mod_id in sorted(modules["module"].unique()):
        if mod_id == 0:
            continue  # skip unassigned

        mod_genes = set(modules[modules["module"] == mod_id]["gene_symbol"])
        n = len(mod_genes)  # module size
        k = len(ts_genes & mod_genes)  # TS genes in module

        # Hypergeometric p-value
        hyper_p = 1 - stats.hypergeom.cdf(k - 1, N, K, n)

        # Permutation p-value
        perm_count = 0
        all_genes_list = list(all_genes)
        for _ in range(n_permutations):
            random_ts = set(rng.choice(all_genes_list, size=K, replace=False))
            random_k = len(random_ts & mod_genes)
            if random_k >= k:
                perm_count += 1
        perm_p = (perm_count + 1) / (n_permutations + 1)

        ts_in_module = sorted(ts_genes & mod_genes)

        results.append({
            "module": int(mod_id),
            "module_size": n,
            "ts_genes_in_module": k,
            "ts_genes_expected": round(K * n / N, 2),
            "fold_enrichment": round(k / max(K * n / N, 0.01), 2),
            "hypergeometric_p": float(hyper_p),
            "permutation_p": float(perm_p),
            "significant": bool(perm_p < 0.01),
            "ts_gene_list": ts_in_module,
        })

    return results


def find_hub_genes(
    expr_matrix: pd.DataFrame,
    modules: pd.DataFrame,
    module_id: int,
    n_top: int = 10,
) -> list[dict]:
    """Find hub genes for a module (highest intramodular connectivity).

    Hub genes are those with highest correlation to the module eigengene.
    """
    mod_genes = modules[modules["module"] == module_id]["gene_symbol"].tolist()
    mod_expr = expr_matrix[[g for g in mod_genes if g in expr_matrix.columns]]

    if mod_expr.shape[1] < 3:
        return []

    # Module eigengene = first PC of module expression
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    me = pca.fit_transform(mod_expr.fillna(0).values)[:, 0]

    # Module membership = correlation of each gene with module eigengene
    hub_genes = []
    for gene in mod_expr.columns:
        r, p = stats.pearsonr(mod_expr[gene].fillna(0).values, me)
        hub_genes.append({
            "gene_symbol": gene,
            "module_membership": abs(float(r)),
            "mm_pvalue": float(p),
        })

    hub_genes.sort(key=lambda x: -x["module_membership"])
    return hub_genes[:n_top]


def annotate_modules_cell_types(
    modules: pd.DataFrame,
) -> list[dict]:
    """Annotate each WGCNA module with HMBA cell-type marker overlap.

    For each module, checks how many of its genes overlap with each
    HMBA cell type's marker panel.  Returns a list of per-module
    annotations sorted by best cell-type match.
    """
    taxonomy = get_taxonomy()
    # Build marker -> cell_type_label mapping
    marker_to_ct: dict[str, list[str]] = {}
    for label in taxonomy:
        for marker in get_marker_panel(label):
            marker_to_ct.setdefault(marker, []).append(label)

    results: list[dict] = []
    for mod_id in sorted(modules["module"].unique()):
        if mod_id == 0:
            continue
        mod_genes = set(modules[modules["module"] == mod_id]["gene_symbol"])
        ct_hits: dict[str, list[str]] = {}
        for gene in mod_genes:
            for ct_label in marker_to_ct.get(gene, []):
                ct_hits.setdefault(ct_label, []).append(gene)

        ct_annotations = sorted(
            [
                {"cell_type": ct, "marker_genes": genes, "n_markers": len(genes)}
                for ct, genes in ct_hits.items()
            ],
            key=lambda x: -x["n_markers"],
        )
        results.append({
            "module": int(mod_id),
            "module_size": len(mod_genes),
            "cell_type_annotations": ct_annotations,
            "top_cell_type": ct_annotations[0]["cell_type"] if ct_annotations else None,
        })

    return results


def run_analysis(
    expr_matrix: pd.DataFrame,
    gene_set_name: str = "ts_combined",
    output_dir: Path = OUTPUT_DIR,
    min_module_size: int = 30,
    n_permutations: int = 1000,
) -> dict:
    """Run the full WGCNA pipeline.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Genes (columns) x samples (rows) expression matrix.
        Should contain CSTC region samples.
    gene_set_name : str
        TS gene set to test for enrichment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_genes = set(get_gene_set(gene_set_name).keys())
    logger.info("WGCNA on %d genes x %d samples, testing %d TS genes",
                expr_matrix.shape[1], expr_matrix.shape[0], len(ts_genes))

    # 1. Select soft threshold
    power, fit_table = select_soft_threshold(expr_matrix)
    fit_table.to_csv(output_dir / "wgcna_soft_threshold.csv", index=False)

    # 2. Compute adjacency and TOM
    logger.info("Computing correlation matrix...")
    cor = expr_matrix.corr().values
    logger.info("Computing adjacency (power=%d)...", power)
    adj = compute_adjacency(cor, power)
    logger.info("Computing TOM...")
    tom = compute_tom(adj)

    # 3. Identify modules
    gene_names = list(expr_matrix.columns)
    modules = identify_modules(tom, gene_names, min_module_size=min_module_size)
    modules.to_csv(output_dir / "wgcna_modules.csv", index=False)

    # 4. Test enrichment
    enrichment = compute_enrichment(modules, ts_genes, n_permutations=n_permutations)

    enrichment_path = output_dir / "wgcna_enrichment.json"
    with open(enrichment_path, "w") as f:
        json.dump(enrichment, f, indent=2)

    # 5. Find hub genes for enriched modules
    hub_results = {}
    for e in enrichment:
        if e["significant"]:
            hubs = find_hub_genes(expr_matrix, modules, e["module"])
            hub_results[e["module"]] = hubs

    if hub_results:
        hub_path = output_dir / "wgcna_hub_genes.json"
        with open(hub_path, "w") as f:
            json.dump(hub_results, f, indent=2, default=str)

    # Annotate modules with HMBA cell-type markers
    logger.info("Annotating modules with HMBA cell-type markers...")
    cell_type_annotations = annotate_modules_cell_types(modules)
    ct_path = output_dir / "wgcna_module_cell_types.json"
    with open(ct_path, "w") as f:
        json.dump(cell_type_annotations, f, indent=2)

    # Generate enrichment plot
    _generate_enrichment_plot(enrichment, output_dir / "wgcna_enrichment_plot.png")

    n_significant = sum(1 for e in enrichment if e["significant"])
    logger.info("WGCNA complete: %d modules, %d significantly enriched for TS genes",
                len(enrichment), n_significant)

    return {
        "soft_threshold_power": power,
        "n_modules": len(enrichment),
        "n_significant": n_significant,
        "enrichment": enrichment,
        "hub_genes": hub_results,
        "cell_type_annotations": cell_type_annotations,
    }


def _generate_enrichment_plot(enrichment: list[dict], output_path: Path) -> None:
    """Generate bar plot of TS gene enrichment per module."""
    if not enrichment:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modules = [f"M{e['module']}" for e in enrichment]
    fold = [e["fold_enrichment"] for e in enrichment]
    colors = ["#e74c3c" if e["significant"] else "#95a5a6" for e in enrichment]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(modules, fold, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, label="Expected")
    ax.set_ylabel("Fold Enrichment")
    ax.set_xlabel("Module")
    ax.set_title("TS Gene Enrichment in WGCNA Co-expression Modules\n"
                 "(red = permutation p < 0.01)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WGCNA co-expression analysis on AHBA CSTC data"
    )
    parser.add_argument("--gene-set", default="ts_combined")
    parser.add_argument("--min-module-size", type=int, default=30)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # For full analysis, would load AHBA CSTC expression matrix here.
    # Placeholder: generate synthetic data to validate pipeline.
    logger.info("NOTE: Full analysis requires AHBA CSTC expression matrix.")
    logger.info("Run ahba_spatial.py first to fetch AHBA data, then pass the "
                "expression matrix to run_analysis().")

    # Demo with synthetic data
    rng = np.random.default_rng(42)
    n_samples, n_genes = 50, 200
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    # Include some TS genes
    ts_genes = list(get_gene_set(args.gene_set).keys())
    for i, g in enumerate(ts_genes[:20]):
        gene_names[i] = g

    expr = pd.DataFrame(
        rng.normal(size=(n_samples, n_genes)),
        columns=gene_names,
    )

    results = run_analysis(
        expr,
        gene_set_name=args.gene_set,
        output_dir=args.output,
        min_module_size=args.min_module_size,
        n_permutations=args.permutations,
    )

    print(f"\nWGCNA complete:")
    print(f"  Power: {results['soft_threshold_power']}")
    print(f"  Modules: {results['n_modules']}")
    print(f"  Significantly enriched: {results['n_significant']}")


if __name__ == "__main__":
    main()
