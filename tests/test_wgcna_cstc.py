"""Tests for WGCNA co-expression network analysis module."""

import numpy as np
import pandas as pd

from bioagentics.analysis.tourettes.wgcna_cstc import (
    compute_adjacency,
    compute_tom,
    find_hub_genes,
    identify_modules,
    select_soft_threshold,
    compute_enrichment,
)


def _make_expr_matrix(n_samples: int = 50, n_genes: int = 100) -> pd.DataFrame:
    """Create synthetic expression matrix with correlated gene blocks."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(n_samples, n_genes))

    # Create correlated blocks (simulate modules)
    for start in range(0, min(60, n_genes), 20):
        block = rng.normal(size=(n_samples, 1))
        for j in range(start, min(start + 20, n_genes)):
            data[:, j] += block[:, 0] * 2  # strong correlation within block

    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    return pd.DataFrame(data, columns=gene_names)


def test_select_soft_threshold():
    expr = _make_expr_matrix()
    power, fit_table = select_soft_threshold(expr, powers=list(range(1, 11)))
    assert 1 <= power <= 10
    assert "power" in fit_table.columns
    assert "r2" in fit_table.columns
    assert len(fit_table) == 10


def test_compute_adjacency():
    cor = np.array([[1.0, 0.8], [0.8, 1.0]])
    adj = compute_adjacency(cor, power=6)
    assert adj.shape == (2, 2)
    assert np.isclose(adj[0, 1], 0.8 ** 6)
    assert np.isclose(adj[0, 0], 1.0)


def test_compute_tom_symmetric():
    rng = np.random.default_rng(42)
    adj = np.abs(rng.normal(size=(10, 10)))
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 1)
    adj = np.clip(adj, 0, 1)

    tom = compute_tom(adj)
    assert tom.shape == (10, 10)
    assert np.allclose(tom, tom.T, atol=1e-10)
    assert np.all(tom >= 0) and np.all(tom <= 1)
    assert np.allclose(np.diag(tom), 1.0)


def test_identify_modules():
    rng = np.random.default_rng(42)
    n = 100
    adj = np.abs(rng.normal(size=(n, n))) * 0.1
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)
    adj = np.clip(adj, 0, 1)
    tom = compute_tom(adj)

    gene_names = [f"GENE_{i}" for i in range(n)]
    modules = identify_modules(tom, gene_names, min_module_size=5)
    assert len(modules) == n
    assert "gene_symbol" in modules.columns
    assert "module" in modules.columns


def test_compute_enrichment_basic():
    modules = pd.DataFrame({
        "gene_symbol": [f"GENE_{i}" for i in range(100)],
        "module": [1] * 40 + [2] * 40 + [0] * 20,
    })
    # Put 10 TS genes in module 1, 2 in module 2
    ts_genes = {f"GENE_{i}" for i in range(10)}

    results = compute_enrichment(modules, ts_genes, n_permutations=100)
    assert len(results) == 2  # modules 1 and 2

    mod1 = next(r for r in results if r["module"] == 1)
    assert mod1["ts_genes_in_module"] == 10
    assert mod1["fold_enrichment"] > 1.0

    mod2 = next(r for r in results if r["module"] == 2)
    assert mod2["ts_genes_in_module"] == 0


def test_compute_enrichment_no_ts_genes():
    modules = pd.DataFrame({
        "gene_symbol": [f"GENE_{i}" for i in range(50)],
        "module": [1] * 25 + [2] * 25,
    })
    results = compute_enrichment(modules, {"NONEXISTENT"}, n_permutations=10)
    assert results == []


def test_find_hub_genes():
    rng = np.random.default_rng(42)
    genes = [f"G{i}" for i in range(20)]
    expr = pd.DataFrame(rng.normal(size=(30, 20)), columns=genes)
    modules = pd.DataFrame({
        "gene_symbol": genes,
        "module": [1] * 15 + [2] * 5,
    })
    hubs = find_hub_genes(expr, modules, module_id=1, n_top=5)
    assert len(hubs) == 5
    assert "gene_symbol" in hubs[0]
    assert "module_membership" in hubs[0]
    # Hub genes should be sorted by module membership (descending)
    mms = [h["module_membership"] for h in hubs]
    assert mms == sorted(mms, reverse=True)
