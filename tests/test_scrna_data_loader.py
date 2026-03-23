"""Tests for scRNA-seq data_loader module — flat tar, empty-droplet filtering, Arrow compat."""

import gzip
import io
import os
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
import scipy.sparse as sp

from bioagentics.scrna.data_loader import (
    _filter_empty_droplets,
    load_10x_from_flat_tar,
    discover_and_load_supplementary,
)


def _write_10x_triplet(tar: tarfile.TarFile, gsm: str, n_cells: int, n_genes: int, seed: int = 0, density: float = 0.05):
    """Write a fake 10x triplet (barcodes, genes, matrix) into an open tar."""
    rng = np.random.default_rng(seed)
    mat = sp.random(n_cells, n_genes, density=density, format="coo", random_state=rng).T  # genes x cells for mtx

    # matrix.mtx.gz
    buf = io.BytesIO()
    sio.mmwrite(buf, mat)
    mtx_gz = gzip.compress(buf.getvalue())
    info = tarfile.TarInfo(name=f"{gsm}_matrix.mtx.gz")
    info.size = len(mtx_gz)
    tar.addfile(info, io.BytesIO(mtx_gz))

    # barcodes.tsv.gz
    barcodes = "\n".join(f"BARCODE{i}" for i in range(n_cells))
    bc_gz = gzip.compress(barcodes.encode())
    info = tarfile.TarInfo(name=f"{gsm}_barcodes.tsv.gz")
    info.size = len(bc_gz)
    tar.addfile(info, io.BytesIO(bc_gz))

    # genes.tsv.gz
    genes = "\n".join(f"ENSG{i:05d}\tGENE{i}" for i in range(n_genes))
    genes_gz = gzip.compress(genes.encode())
    info = tarfile.TarInfo(name=f"{gsm}_genes.tsv.gz")
    info.size = len(genes_gz)
    tar.addfile(info, io.BytesIO(genes_gz))


class TestLoadFromFlatTar:
    def test_loads_two_samples(self, tmp_path):
        tar_path = tmp_path / "GSE999999_RAW.tar"
        with tarfile.open(tar_path, "w") as tar:
            _write_10x_triplet(tar, "GSM0001", n_cells=50, n_genes=30, seed=1)
            _write_10x_triplet(tar, "GSM0002", n_cells=40, n_genes=30, seed=2)

        adatas = load_10x_from_flat_tar(tar_path, min_genes=0)
        assert len(adatas) == 2
        assert adatas[0].obs["sample"].iloc[0] == "GSM0001"
        assert adatas[1].obs["sample"].iloc[0] == "GSM0002"
        assert adatas[0].n_vars == 30
        assert adatas[1].n_vars == 30

    def test_min_genes_filters_empty_droplets(self, tmp_path):
        tar_path = tmp_path / "GSE999999_RAW.tar"
        with tarfile.open(tar_path, "w") as tar:
            _write_10x_triplet(tar, "GSM0001", n_cells=100, n_genes=50, seed=42)

        # With min_genes=0, get all cells
        all_cells = load_10x_from_flat_tar(tar_path, min_genes=0)
        n_all = all_cells[0].n_obs

        # With min_genes=5, should filter some sparse cells
        filtered = load_10x_from_flat_tar(tar_path, min_genes=5)
        assert filtered[0].n_obs <= n_all

    def test_skips_incomplete_sample(self, tmp_path):
        tar_path = tmp_path / "GSE999999_RAW.tar"
        with tarfile.open(tar_path, "w") as tar:
            _write_10x_triplet(tar, "GSM0001", n_cells=50, n_genes=30, seed=1)
            # Add only barcodes for GSM0002 (incomplete)
            bc_gz = gzip.compress(b"BC1\nBC2")
            info = tarfile.TarInfo(name="GSM0002_barcodes.tsv.gz")
            info.size = len(bc_gz)
            tar.addfile(info, io.BytesIO(bc_gz))

        adatas = load_10x_from_flat_tar(tar_path, min_genes=0)
        assert len(adatas) == 1
        assert adatas[0].obs["sample"].iloc[0] == "GSM0001"

    def test_handles_tar_gz(self, tmp_path):
        tar_path = tmp_path / "GSE999999_RAW.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            _write_10x_triplet(tar, "GSM0001", n_cells=20, n_genes=10, seed=7)

        adatas = load_10x_from_flat_tar(tar_path, min_genes=0)
        assert len(adatas) == 1


class TestFilterEmptyDroplets:
    def test_filters_low_gene_cells(self):
        rng = np.random.default_rng(42)
        # 10 cells, 50 genes; first 5 cells have many genes, last 5 have very few
        data = np.zeros((10, 50), dtype=np.float32)
        data[:5, :] = rng.negative_binomial(5, 0.3, size=(5, 50))
        data[5:, :3] = 1  # only 3 genes expressed
        adata = ad.AnnData(X=sp.csr_matrix(data))

        result = _filter_empty_droplets(adata, min_genes=10)
        assert result.n_obs == 5

    def test_no_op_when_all_pass(self):
        data = np.ones((10, 50), dtype=np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(data))
        result = _filter_empty_droplets(adata, min_genes=1)
        assert result.n_obs == 10

    def test_no_op_when_min_genes_zero(self):
        data = np.zeros((10, 50), dtype=np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(data))
        result = _filter_empty_droplets(adata, min_genes=0)
        assert result.n_obs == 10


class TestDiscoverFlatTar:
    def test_discovers_plain_tar(self, tmp_path):
        gse_dir = tmp_path / "GSE134809"
        suppl = gse_dir / "supplementary"
        suppl.mkdir(parents=True)

        # High density so cells pass default min_genes=200 filter
        tar_path = suppl / "GSE134809_RAW.tar"
        with tarfile.open(tar_path, "w") as tar:
            _write_10x_triplet(tar, "GSM0001", n_cells=30, n_genes=300, seed=3, density=0.9)

        adatas = discover_and_load_supplementary(gse_dir, "GSE134809")
        assert len(adatas) >= 1
        assert adatas[0].obs["sample"].iloc[0] == "GSM0001"


class TestArrowStringCompat:
    def test_env_var_is_set(self):
        assert os.environ.get("PANDAS_FUTURE_INFER_STRING") == "0"
