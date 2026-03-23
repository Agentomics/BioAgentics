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
                treatment=treatment,
                notes=f"Treatment: {treatment}",
            ))
            records.append(CytokineRecord(
                study_id=sid, analyte_name=analyte, measurement_method="ELISA",
                sample_type="serum", condition="remission", sample_size_n=20,
                mean_or_median=rem_m + (hash(sid) % 2), sd_or_iqr=2.5,
                treatment=treatment,
                notes=f"Treatment: {treatment}",
            ))
    return CytokineDataset(records)


def _make_dataset_no_treatment() -> CytokineDataset:
    """Create a test dataset WITHOUT treatment annotations."""
    records = []
    for sid in ["S1", "S2"]:
        for analyte, flare_m, rem_m in [("IL-6", 15, 8), ("TNF-α", 10, 6)]:
            records.append(CytokineRecord(
                study_id=sid, analyte_name=analyte, measurement_method="ELISA",
                sample_type="serum", condition="flare", sample_size_n=20,
                mean_or_median=flare_m, sd_or_iqr=3.0,
            ))
            records.append(CytokineRecord(
                study_id=sid, analyte_name=analyte, measurement_method="ELISA",
                sample_type="serum", condition="remission", sample_size_n=20,
                mean_or_median=rem_m, sd_or_iqr=2.5,
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


def test_treatment_column_filtering():
    """Verify analyzer filters by the treatment column, not just notes."""
    ds = _make_dataset()
    analyzer = TreatmentResponseAnalyzer(ds)
    ivig_result = analyzer.analyze_treatment("IVIG")
    plas_result = analyzer.analyze_treatment("plasmapheresis")
    # IVIG has 3 studies, plasmapheresis has 1 — results should differ
    if ivig_result.effects and plas_result.effects:
        ivig_n = ivig_result.effects["IL-6"].n_studies
        plas_n = plas_result.effects["IL-6"].n_studies
        assert ivig_n == 3
        assert plas_n == 1


def test_no_fallback_to_full_dataset():
    """When no treatment annotations exist, analyzer should return empty results."""
    ds = _make_dataset_no_treatment()
    analyzer = TreatmentResponseAnalyzer(ds)
    result = analyzer.analyze_treatment("IVIG")
    assert len(result.effects) == 0
