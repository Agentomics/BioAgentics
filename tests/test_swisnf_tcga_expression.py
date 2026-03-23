"""Tests for SWI/SNF metabolic convergence Phase 4 TCGA expression analysis."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "cancer"))

_mod = importlib.import_module(
    "swisnf_metabolic_convergence.05_tcga_expression"
)
fdr_correction = _mod.fdr_correction
classify_swisnf_tcga = _mod.classify_swisnf_tcga
compute_differential_expression = _mod.compute_differential_expression


class TestClassifySWISNFTCGA:
    """Test TCGA SWI/SNF classification from mutation data."""

    def test_identifies_arid1a_lof(self):
        """ARID1A LOF mutations are detected."""
        mutations = pd.DataFrame([
            {"Hugo_Symbol": "ARID1A", "Variant_Classification": "Nonsense_Mutation",
             "Tumor_Sample_Barcode": "TCGA-AA-1234-01A-11D"},
            {"Hugo_Symbol": "TP53", "Variant_Classification": "Missense_Mutation",
             "Tumor_Sample_Barcode": "TCGA-BB-5678-01A-11D"},
        ])
        result = classify_swisnf_tcga(mutations)
        assert "TCGA-AA-1234" in result["ARID1A"]
        assert "TCGA-AA-1234" in result["any"]
        assert "TCGA-BB-5678" not in result["any"]

    def test_ignores_missense(self):
        """Missense mutations in ARID1A are not classified as LOF."""
        mutations = pd.DataFrame([
            {"Hugo_Symbol": "ARID1A", "Variant_Classification": "Missense_Mutation",
             "Tumor_Sample_Barcode": "TCGA-CC-9999-01A-11D"},
        ])
        result = classify_swisnf_tcga(mutations)
        assert len(result["ARID1A"]) == 0
        assert len(result["any"]) == 0

    def test_smarca4_detected(self):
        """SMARCA4 mutations are classified separately."""
        mutations = pd.DataFrame([
            {"Hugo_Symbol": "SMARCA4", "Variant_Classification": "Frame_Shift_Del",
             "Tumor_Sample_Barcode": "TCGA-DD-1111-01A-11D"},
        ])
        result = classify_swisnf_tcga(mutations)
        assert "TCGA-DD-1111" in result["SMARCA4"]
        assert "TCGA-DD-1111" in result["any"]
        assert len(result["ARID1A"]) == 0


class TestComputeDifferentialExpression:
    """Test differential expression computation."""

    def test_basic_de(self):
        """Detects significant DE when expression differs."""
        rng = np.random.default_rng(42)
        n_mut, n_wt = 20, 80
        genes = ["GENE_UP", "GENE_FLAT"]
        patients = [f"P{i:03d}" for i in range(n_mut + n_wt)]

        data = {
            "GENE_UP": np.concatenate([
                rng.normal(100, 10, n_mut),  # high in mutant
                rng.normal(50, 10, n_wt),    # low in WT
            ]),
            "GENE_FLAT": rng.normal(50, 10, n_mut + n_wt),
        }
        expr = pd.DataFrame(data, index=patients)
        mut_patients = set(patients[:n_mut])
        wt_patients = set(patients[n_mut:])

        result = compute_differential_expression(expr, mut_patients, wt_patients, genes)
        assert len(result) == 2

        gene_up = result[result["gene"] == "GENE_UP"].iloc[0]
        assert gene_up["log2fc"] > 0.5  # upregulated in mutant
        assert gene_up["p_value"] < 0.001

    def test_empty_when_insufficient_samples(self):
        """Returns empty when too few samples."""
        expr = pd.DataFrame({"GENE": [1, 2]}, index=["P1", "P2"])
        result = compute_differential_expression(expr, {"P1"}, {"P2"}, ["GENE"])
        assert len(result) == 0
