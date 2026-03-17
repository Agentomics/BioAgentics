"""Tests for cytokine_treatment_response module."""

from __future__ import annotations

from bioagentics.cytokine_extraction import CytokineDataset, CytokineRecord
from bioagentics.cytokine_treatment_response import TreatmentResponseAnalyzer


def _make_dataset() -> CytokineDataset:
    """Create a test dataset with treatment annotations."""
    records = []
    studies = [
        ("S1", "IVIG"),
        ("S2", "IVIG"),
        ("S3", "IVIG"),
        ("S4", "plasmapheresis"),
    ]
    for sid, treatment in studies:
        for analyte, flare_m, rem_m in [("IL-6", 15, 8), ("TNF-α", 10, 6), ("IL-10", 4, 7)]:
            records.append(CytokineRecord(
                study_id=sid, analyte_name=analyte, measurement_method="ELISA",
                sample_type="serum", condition="flare", sample_size_n=20,
                mean_or_median=flare_m + (hash(sid) % 3), sd_or_iqr=3.0,
                notes=f"Treatment: {treatment}",
            ))
            records.append(CytokineRecord(
                study_id=sid, analyte_name=analyte, measurement_method="ELISA",
                sample_type="serum", condition="remission", sample_size_n=20,
                mean_or_median=rem_m + (hash(sid) % 2), sd_or_iqr=2.5,
                notes=f"Treatment: {treatment}",
            ))
    return CytokineDataset(records)


def test_analyze_treatment():
    ds = _make_dataset()
    analyzer = TreatmentResponseAnalyzer(ds)
    result = analyzer.analyze_treatment("IVIG")
    assert result.treatment == "IVIG"
    assert len(result.effects) > 0


def test_analyze_all():
    ds = _make_dataset()
    analyzer = TreatmentResponseAnalyzer(ds)
    results = analyzer.analyze_all()
    assert len(results) > 0
    treatments_found = {r.treatment for r in results}
    assert "IVIG" in treatments_found


def test_heatmap(tmp_path):
    ds = _make_dataset()
    analyzer = TreatmentResponseAnalyzer(ds)
    results = analyzer.analyze_all()
    out = analyzer.plot_heatmap(results, output_path=tmp_path / "heatmap.png")
    assert out is not None
    assert out.exists()


def test_results_to_dataframe():
    ds = _make_dataset()
    analyzer = TreatmentResponseAnalyzer(ds)
    results = analyzer.analyze_all()
    df = analyzer.results_to_dataframe(results)
    assert len(df) > 0
    assert "treatment" in df.columns
    assert "analyte" in df.columns
    assert "mean_effect" in df.columns
