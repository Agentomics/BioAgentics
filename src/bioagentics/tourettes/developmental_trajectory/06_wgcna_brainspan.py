"""Step 06 — WGCNA co-expression analysis on BrainSpan developmental data.

Applies Weighted Gene Co-expression Network Analysis (WGCNA) to BrainSpan
RNA-seq data in CSTC brain regions to identify developmentally dynamic
co-expression modules and test TS risk gene enrichment.

This adapts the existing WGCNA pipeline (wgcna_cstc.py, designed for AHBA
adult data) to the BrainSpan developmental transcriptome, enabling
identification of modules with temporal expression dynamics.

Task: #783 (WGCNA on BrainSpan)
Project: ts-developmental-trajectory-modeling

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.06_wgcna_brainspan
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.analysis.tourettes.brainspan_trajectories import (
    CACHE_DIR,
    DEV_STAGES,
    download_brainspan,
    match_cstc_region,
    parse_age,
    classify_dev_stage,
)
from bioagentics.analysis.tourettes.wgcna_cstc import (
    select_soft_threshold,
    compute_adjacency,
    compute_tom,
    identify_modules,
    compute_enrichment,
    find_hub_genes,
)
from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import get_gene_set

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-developmental-trajectory-modeling"
STAGE_ORDER = list(DEV_STAGES.keys())


def build_brainspan_cstc_matrix(
    cache_dir: Path = CACHE_DIR,
    min_expression: float = 1.0,
    min_samples: int = 5,
    max_genes: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a genes-as-columns, samples-as-rows expression matrix from BrainSpan.

    Filters to CSTC-relevant brain regions and retains genes with
    sufficient expression and sample coverage.

    Returns (expression_matrix, sample_metadata).
    """
    expression, rows_meta, cols_meta = download_brainspan(cache_dir)

    # Find column names
    gene_col = next(
        (c for c in rows_meta.columns if "gene_symbol" in c.lower() or "symbol" in c.lower()),
        None,
    )
    age_col = next((c for c in cols_meta.columns if "age" in c.lower()), None)
    struct_col = next(
        (c for c in cols_meta.columns if "structure" in c.lower()),
        None,
    )

    if not all([gene_col, age_col, struct_col]):
        raise ValueError("Cannot find required columns in BrainSpan metadata")

    # Build sample metadata with CSTC region assignment
    sample_info = []
    cstc_indices = []
    for idx, row in cols_meta.iterrows():
        cstc = match_cstc_region(str(row[struct_col]))
        if cstc is not None:
            period, value = parse_age(str(row[age_col]))
            stage = classify_dev_stage(period, value)
            sample_info.append({
                "sample_idx": idx,
                "age": str(row[age_col]),
                "structure": str(row[struct_col]),
                "cstc_region": cstc,
                "period": period,
                "age_value": value,
                "dev_stage": stage,
            })
            cstc_indices.append(idx)

    if not cstc_indices:
        raise ValueError("No CSTC samples found in BrainSpan")

    sample_meta = pd.DataFrame(sample_info)
    logger.info("Found %d CSTC samples across %d regions",
                len(cstc_indices), sample_meta["cstc_region"].nunique())

    # Extract expression for CSTC samples
    cstc_expr = expression.iloc[:, cstc_indices].copy()

    # Set gene symbols as index
    gene_symbols = rows_meta[gene_col].values
    cstc_expr.index = gene_symbols

    # Remove duplicate gene symbols (keep highest mean)
    cstc_expr["_mean"] = cstc_expr.mean(axis=1)
    cstc_expr = cstc_expr.sort_values("_mean", ascending=False)
    cstc_expr = cstc_expr[~cstc_expr.index.duplicated(keep="first")]
    cstc_expr = cstc_expr.drop(columns=["_mean"])

    # Filter: minimum expression
    gene_means = cstc_expr.mean(axis=1)
    expressed = gene_means >= min_expression
    cstc_expr = cstc_expr[expressed]
    logger.info("After expression filter (>= %.1f): %d genes", min_expression, len(cstc_expr))

    # Filter: expressed in minimum number of samples
    n_expressing = (cstc_expr > 0).sum(axis=1)
    cstc_expr = cstc_expr[n_expressing >= min_samples]
    logger.info("After sample coverage filter (>= %d samples): %d genes",
                min_samples, len(cstc_expr))

    # Limit to most variable genes for WGCNA (computational tractability)
    if len(cstc_expr) > max_genes:
        gene_var = cstc_expr.var(axis=1)
        top_var = gene_var.nlargest(max_genes).index
        cstc_expr = cstc_expr.loc[top_var]
        logger.info("Selected top %d most variable genes", max_genes)

    # Transpose: samples as rows, genes as columns (WGCNA convention)
    expr_matrix = cstc_expr.T
    expr_matrix.index = range(len(expr_matrix))

    return expr_matrix, sample_meta


def characterize_module_dynamics(
    expr_matrix: pd.DataFrame,
    modules: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> list[dict]:
    """Characterize the developmental dynamics of each WGCNA module.

    For each module, computes the module eigengene and tracks it across
    developmental stages to identify temporally dynamic modules.
    """
    from sklearn.decomposition import PCA

    dynamics = []

    for mod_id in sorted(modules["module"].unique()):
        if mod_id == 0:
            continue

        mod_genes = modules[modules["module"] == mod_id]["gene_symbol"].tolist()
        mod_expr = expr_matrix[[g for g in mod_genes if g in expr_matrix.columns]]

        if mod_expr.shape[1] < 3:
            continue

        # Module eigengene
        pca = PCA(n_components=1)
        me = pca.fit_transform(mod_expr.fillna(0).values)[:, 0]
        var_explained = float(pca.explained_variance_ratio_[0])

        # Track eigengene across stages
        stage_means: dict[str, float] = {}
        for stage in STAGE_ORDER:
            stage_mask = sample_meta["dev_stage"] == stage
            if stage_mask.any():
                stage_idx = sample_meta[stage_mask].index.tolist()
                valid_idx = [i for i in stage_idx if i < len(me)]
                if valid_idx:
                    stage_means[stage] = float(np.mean([me[i] for i in valid_idx]))

        if not stage_means:
            continue

        # Find peak and trough stages
        peak_stage = max(stage_means, key=stage_means.get)  # type: ignore[arg-type]
        trough_stage = min(stage_means, key=stage_means.get)  # type: ignore[arg-type]
        amplitude = stage_means[peak_stage] - stage_means[trough_stage]

        # Classify temporal pattern
        stage_list = [s for s in STAGE_ORDER if s in stage_means]
        if stage_list:
            peak_pos = stage_list.index(peak_stage) / max(len(stage_list) - 1, 1)
        else:
            peak_pos = 0.5

        if amplitude < 0.3:
            pattern = "stable"
        elif peak_pos < 0.3:
            pattern = "early_declining"
        elif peak_pos < 0.6:
            pattern = "childhood_peak"
        elif peak_pos < 0.8:
            pattern = "adolescent_rising"
        else:
            pattern = "adult_peak"

        dynamics.append({
            "module": int(mod_id),
            "n_genes": len(mod_genes),
            "variance_explained": round(var_explained, 4),
            "peak_stage": peak_stage,
            "trough_stage": trough_stage,
            "amplitude": round(amplitude, 4),
            "temporal_pattern": pattern,
            "stage_eigengene": {k: round(v, 4) for k, v in stage_means.items()},
        })

    return dynamics


def generate_module_dynamics_plot(
    dynamics: list[dict],
    output_path: Path,
) -> Path:
    """Plot module eigengene trajectories across development."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not dynamics:
        return output_path

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i / max(len(dynamics) - 1, 1)) for i in range(len(dynamics))]

    all_stages: list[str] = []
    for i, mod in enumerate(dynamics):
        stages = [s for s in STAGE_ORDER if s in mod["stage_eigengene"]]
        if not all_stages:
            all_stages = stages
        values = [mod["stage_eigengene"][s] for s in stages]
        label = f"M{mod['module']} ({mod['temporal_pattern']}, n={mod['n_genes']})"
        ax.plot(range(len(stages)), values, color=colors[i],
                linewidth=2, marker="o", markersize=4, label=label)

    ax.set_xticks(range(len(all_stages)))
    ax.set_xticklabels([s.replace("_", "\n") for s in all_stages],
                       fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Module Eigengene")
    ax.set_title("WGCNA Module Developmental Dynamics (BrainSpan CSTC)")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved module dynamics plot to %s", output_path)
    return output_path


def run(
    output_dir: Path = OUTPUT_DIR,
    cache_dir: Path = CACHE_DIR,
    min_module_size: int = 20,
    max_genes: int = 5000,
    n_permutations: int = 1000,
) -> dict:
    """Run WGCNA on BrainSpan CSTC developmental data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wgcna_dir = output_dir / "wgcna_brainspan"
    wgcna_dir.mkdir(parents=True, exist_ok=True)

    # Build expression matrix
    logger.info("Building BrainSpan CSTC expression matrix...")
    expr_matrix, sample_meta = build_brainspan_cstc_matrix(
        cache_dir, max_genes=max_genes,
    )
    sample_meta.to_csv(wgcna_dir / "sample_metadata.csv", index=False)
    logger.info("Expression matrix: %d samples x %d genes", *expr_matrix.shape)

    # TS genes for enrichment
    ts_genes = set(get_gene_set("ts_combined").keys())
    ts_in_matrix = ts_genes & set(expr_matrix.columns)
    logger.info("TS genes in matrix: %d/%d", len(ts_in_matrix), len(ts_genes))

    # 1. Select soft threshold
    logger.info("Selecting soft threshold power...")
    power, fit_table = select_soft_threshold(expr_matrix)
    fit_table.to_csv(wgcna_dir / "soft_threshold.csv", index=False)

    # 2. Compute adjacency and TOM
    logger.info("Computing correlation matrix (%d genes)...", expr_matrix.shape[1])
    cor = expr_matrix.corr().values

    logger.info("Computing adjacency (power=%d)...", power)
    adj = compute_adjacency(cor, power)

    logger.info("Computing TOM...")
    tom = compute_tom(adj)

    # 3. Identify modules
    gene_names = list(expr_matrix.columns)
    modules = identify_modules(tom, gene_names, min_module_size=min_module_size)
    modules.to_csv(wgcna_dir / "modules.csv", index=False)

    n_modules = len(set(modules["module"]) - {0})
    logger.info("Found %d modules", n_modules)

    # 4. Enrichment testing
    logger.info("Testing TS gene enrichment...")
    enrichment = compute_enrichment(modules, ts_genes, n_permutations=n_permutations)

    with open(wgcna_dir / "enrichment.json", "w") as f:
        json.dump(enrichment, f, indent=2)

    # 5. Characterize module developmental dynamics
    logger.info("Characterizing module dynamics...")
    dynamics = characterize_module_dynamics(expr_matrix, modules, sample_meta)

    with open(wgcna_dir / "module_dynamics.json", "w") as f:
        json.dump(dynamics, f, indent=2)

    # 6. Find hub genes in enriched modules
    hub_results: dict[int, list[dict]] = {}
    for e in enrichment:
        if e["significant"]:
            hubs = find_hub_genes(expr_matrix, modules, e["module"])
            hub_results[e["module"]] = hubs
            # Check which hub genes are TS genes
            for h in hubs:
                h["is_ts_gene"] = h["gene_symbol"] in ts_genes

    if hub_results:
        with open(wgcna_dir / "hub_genes.json", "w") as f:
            json.dump(hub_results, f, indent=2, default=str)

    # 7. Generate plots
    generate_module_dynamics_plot(dynamics, wgcna_dir / "module_dynamics_plot.png")

    # Cross-reference: which enriched modules have interesting dynamics?
    enriched_dynamic = []
    dynamics_map = {d["module"]: d for d in dynamics}
    for e in enrichment:
        if e["significant"] and e["module"] in dynamics_map:
            d = dynamics_map[e["module"]]
            enriched_dynamic.append({
                "module": e["module"],
                "module_size": e["module_size"],
                "ts_genes_in_module": e["ts_genes_in_module"],
                "fold_enrichment": e["fold_enrichment"],
                "permutation_p": e["permutation_p"],
                "ts_gene_list": e["ts_gene_list"],
                "temporal_pattern": d["temporal_pattern"],
                "peak_stage": d["peak_stage"],
                "amplitude": d["amplitude"],
                "hub_genes": [h["gene_symbol"] for h in hub_results.get(e["module"], [])[:5]],
            })

    # Summary
    n_significant = sum(1 for e in enrichment if e["significant"])
    summary = {
        "expression_matrix": {
            "n_samples": int(expr_matrix.shape[0]),
            "n_genes": int(expr_matrix.shape[1]),
            "ts_genes_in_matrix": len(ts_in_matrix),
        },
        "soft_threshold_power": int(power),
        "n_modules": n_modules,
        "n_significant_enriched": n_significant,
        "enriched_dynamic_modules": enriched_dynamic,
        "all_module_dynamics": [
            {"module": d["module"], "pattern": d["temporal_pattern"],
             "peak": d["peak_stage"], "n_genes": d["n_genes"]}
            for d in dynamics
        ],
    }

    with open(wgcna_dir / "wgcna_brainspan_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WGCNA co-expression analysis on BrainSpan developmental data"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--min-module-size", type=int, default=20)
    parser.add_argument("--max-genes", type=int, default=5000,
                        help="Maximum genes for WGCNA (most variable selected)")
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = run(
        output_dir=args.output,
        cache_dir=args.cache,
        min_module_size=args.min_module_size,
        max_genes=args.max_genes,
        n_permutations=args.permutations,
    )

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    em = summary["expression_matrix"]
    print(f"\nWGCNA on BrainSpan CSTC Data")
    print(f"  Matrix: {em['n_samples']} samples x {em['n_genes']} genes")
    print(f"  TS genes in matrix: {em['ts_genes_in_matrix']}")
    print(f"  Soft threshold power: {summary['soft_threshold_power']}")
    print(f"  Modules found: {summary['n_modules']}")
    print(f"  Enriched for TS genes: {summary['n_significant_enriched']}")

    if summary["enriched_dynamic_modules"]:
        print(f"\n  TS-enriched modules with developmental dynamics:")
        for m in summary["enriched_dynamic_modules"]:
            print(f"    Module {m['module']}: {m['temporal_pattern']} "
                  f"(peak {m['peak_stage']}, {m['fold_enrichment']}x enriched)")
            print(f"      TS genes: {', '.join(m['ts_gene_list'])}")
            if m["hub_genes"]:
                print(f"      Top hubs: {', '.join(m['hub_genes'])}")


if __name__ == "__main__":
    main()
