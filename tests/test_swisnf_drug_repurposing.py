"""Tests for SWI/SNF metabolic convergence Phase 5 drug repurposing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "cancer"))

_mod = importlib.import_module(
    "swisnf_metabolic_convergence.06_drug_repurposing"
)
DRUG_TARGET_MAP = _mod.DRUG_TARGET_MAP
compute_drug_sensitivity = _mod.compute_drug_sensitivity


class TestDrugTargetMap:
    """Test the curated drug-target mapping."""

    def test_metformin_targets_oxphos(self):
        """Metformin should target Complex I genes."""
        met = [d for d in DRUG_TARGET_MAP if d["drug"] == "Metformin"][0]
        assert "OXPHOS" in met["target_pathway"]
        assert "NDUFA2" in met["target_genes"]
        assert "FDA-approved" in met["approval_status"]

    def test_statins_target_hmgcr(self):
        """Statins should target HMGCR."""
        statins = [d for d in DRUG_TARGET_MAP if "statin" in d["drug"].lower()]
        assert len(statins) >= 2
        for s in statins:
            assert s["target_genes"] == ["HMGCR"]

    def test_eprenetapopt_targets_gsh(self):
        """Eprenetapopt targets GSH genes."""
        epr = [d for d in DRUG_TARGET_MAP if "Eprenetapopt" in d["drug"]][0]
        assert "GCLC" in epr["target_genes"]
        assert "Weak" in epr["evidence_strength"]


class TestDrugSensitivity:
    """Test drug sensitivity computation."""

    def test_detects_sensitivity(self):
        """More negative values in mutant = more sensitive."""
        rng = np.random.default_rng(42)
        vals = pd.Series(
            np.concatenate([rng.normal(-2, 0.5, 20), rng.normal(-0.5, 0.5, 80)]),
            index=[f"ACH-{i:06d}" for i in range(100)],
        )
        mut_ids = {f"ACH-{i:06d}" for i in range(20)}
        wt_ids = {f"ACH-{i:06d}" for i in range(20, 100)}

        result = compute_drug_sensitivity(vals, mut_ids, wt_ids)
        assert result is not None
        assert result["cohens_d"] < -0.5
        assert result["p_value"] < 0.001
        assert result["is_sl"] == True

    def test_returns_none_insufficient_data(self):
        """Returns None when too few samples."""
        vals = pd.Series([1.0, 2.0], index=["ACH-000001", "ACH-000002"])
        result = compute_drug_sensitivity(vals, {"ACH-000001"}, {"ACH-000002"})
        assert result is None
