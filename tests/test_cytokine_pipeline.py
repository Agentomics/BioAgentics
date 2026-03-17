"""Integration test for the full cytokine pipeline orchestrator."""

from __future__ import annotations

import pandas as pd

from bioagentics.run_cytokine_pipeline import run_pipeline


def _create_test_data(tmp_path):
    """Create a synthetic extracted cytokine CSV for pipeline testing."""
    rows = []
    studies = [
        ("Frankovich2015", "12345"),
        ("Swedo2012", "23456"),
        ("Hesselmark2021", "34567"),
        ("Calaprice2017", "45678"),
    ]
    analytes = {
        "IL-6": (15, 3, 8, 2.5),
        "IL-1β": (12, 4, 6, 3),
        "TNF-α": (10, 2.5, 5, 2),
        "IFN-γ": (8, 3, 4, 2),
        "IL-17A": (6, 2, 3, 1.5),
        "IL-10": (3, 1, 6, 2),
        "IL-4": (2, 0.8, 2.5, 1),
    }

    for sid, pmid in studies:
        for analyte, (flare_m, flare_s, rem_m, rem_s) in analytes.items():
            # Add some noise per study
            noise = hash(sid + analyte) % 3
            rows.append({
                "study_id": sid, "pmid": pmid, "analyte_name": analyte,
                "measurement_method": "ELISA", "sample_type": "serum",
                "condition": "flare", "sample_size_n": 20,
                "mean_or_median": flare_m + noise, "sd_or_iqr": flare_s,
                "p_value": 0.01, "notes": "Treatment: IVIG",
            })
            rows.append({
                "study_id": sid, "pmid": pmid, "analyte_name": analyte,
                "measurement_method": "ELISA", "sample_type": "serum",
                "condition": "remission", "sample_size_n": 20,
                "mean_or_median": rem_m + noise * 0.5, "sd_or_iqr": rem_s,
                "p_value": 0.01, "notes": "Treatment: IVIG",
            })

    csv_path = tmp_path / "test_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_full_pipeline(tmp_path):
    csv_path = _create_test_data(tmp_path)
    output_dir = tmp_path / "output"

    summary = run_pipeline(data_path=csv_path, output_dir=output_dir)

    # Validate summary structure
    assert summary["pipeline"] == "cytokine-network-flare-prediction"
    assert summary["dataset"]["n_records"] > 0
    assert summary["dataset"]["n_studies"] == 4

    # Meta-analysis ran
    assert "meta_analysis" in summary
    assert summary["meta_analysis"]["n_analytes_analyzed"] > 0

    # Network built
    assert "network" in summary
    assert summary["network"]["n_nodes"] > 10

    # Axis classification
    assert "axis_classification" in summary
    assert summary["axis_classification"]["dominant_axis"] != ""

    # Prediction ran
    assert "prediction" in summary

    # Treatment response
    assert "treatment_response" in summary

    # Output files exist
    assert (output_dir / "cytokine_network.png").exists()
    assert (output_dir / "cytokine_network.json").exists()
    assert (output_dir / "immune_axis_radar.png").exists()
    assert (output_dir / "pipeline_summary.json").exists()
    assert (output_dir / "meta_analysis_summary.csv").exists()
