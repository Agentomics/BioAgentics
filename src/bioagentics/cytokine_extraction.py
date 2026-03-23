"""Cytokine/chemokine data extraction schema and loader for PANDAS/PANS meta-analysis.

Provides a Pydantic-validated data model for systematically extracted cytokine
measurements from published PANDAS/PANS studies, plus CSV/JSON loaders and a
curated analyte vocabulary covering standard cytokines, CIRS biomarkers, and
BBB markers.

Usage::

    from bioagentics.cytokine_extraction import CytokineDataset

    ds = CytokineDataset.from_csv("data/pandas_pans/cytokine-network-flare-prediction/extracted_cytokines.csv")
    print(ds.summary())
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "pandas_pans" / "cytokine-network-flare-prediction"
OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "cytokine-network-flare-prediction"


# ---------------------------------------------------------------------------
# Enums for controlled vocabularies
# ---------------------------------------------------------------------------


class MeasurementMethod(str, Enum):
    ELISA = "ELISA"
    LUMINEX = "Luminex"
    CBA = "CBA"  # cytometric bead array
    MSD = "MSD"  # meso scale discovery
    SIMOA = "Simoa"
    IMMUNOASSAY = "immunoassay"
    NEPHELOMETRY = "nephelometry"
    OTHER = "other"


class SampleType(str, Enum):
    SERUM = "serum"
    PLASMA = "plasma"
    CSF = "CSF"
    WHOLE_BLOOD = "whole_blood"
    PBMC = "PBMC"
    OTHER = "other"


class Condition(str, Enum):
    FLARE = "flare"
    REMISSION = "remission"
    HEALTHY_CONTROL = "healthy_control"
    ACTIVE = "active"  # active disease (not specifically flare)
    BASELINE = "baseline"


# ---------------------------------------------------------------------------
# Analyte vocabulary
# ---------------------------------------------------------------------------

# Standard cytokines and chemokines expected in PANDAS/PANS studies
STANDARD_CYTOKINES = [
    "IL-1β", "IL-2", "IL-4", "IL-5", "IL-6", "IL-8", "IL-10",
    "IL-12", "IL-12p70", "IL-13", "IL-17A", "IL-17F", "IL-22",
    "IL-23", "IL-33", "IFN-γ", "TNF-α", "TGF-β",
    "CXCL10", "CCL2", "CCL3", "CCL4", "CCL5", "CXCL8",
    "GM-CSF", "G-CSF",
]

# CIRS (Chronic Inflammatory Response Syndrome) biomarkers — task #281
CIRS_BIOMARKERS = [
    "TGF-β1", "MMP-9", "C4a", "α-MSH", "C3",
]

# BBB (Blood-Brain Barrier) markers — task #338
BBB_MARKERS = [
    "S100B", "MMP-9",  # MMP-9 is shared with CIRS
]

# Combined vocabulary (deduplicated)
ANALYTE_VOCABULARY = sorted(set(STANDARD_CYTOKINES + CIRS_BIOMARKERS + BBB_MARKERS))


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class CytokineRecord(BaseModel):
    """A single cytokine measurement extracted from a published study."""

    study_id: str = Field(description="Short identifier for the study (e.g., 'Frankovich2015')")
    pmid: str | None = Field(default=None, description="PubMed ID", coerce_numbers_to_str=True)
    analyte_name: str = Field(description="Cytokine/chemokine/biomarker name")
    measurement_method: str = Field(description="Assay platform (ELISA, Luminex, etc.)")
    sample_type: str = Field(description="Sample matrix (serum, CSF, plasma, etc.)")
    condition: str = Field(description="Patient condition (flare, remission, healthy_control, etc.)")
    sample_size_n: int = Field(ge=1, description="Number of subjects in this group")
    mean_or_median: float = Field(description="Central tendency value (mean or median)")
    sd_or_iqr: float | None = Field(default=None, ge=0, description="Spread measure (SD or IQR)")
    p_value: float | None = Field(default=None, ge=0, le=1, description="Reported p-value")
    treatment: str | None = Field(default=None, description="Treatment received (IVIG, plasmapheresis, antibiotics, etc.)")
    notes: str | None = Field(default=None, description="Free-text notes")

    @field_validator("analyte_name")
    @classmethod
    def warn_unknown_analyte(cls, v: str) -> str:
        if v not in ANALYTE_VOCABULARY:
            logger.warning("Analyte '%s' not in standard vocabulary — will be included but may need mapping", v)
        return v

    @field_validator("measurement_method")
    @classmethod
    def normalize_method(cls, v: str) -> str:
        try:
            return MeasurementMethod(v).value
        except ValueError:
            logger.warning("Non-standard measurement method: '%s'", v)
            return v

    @field_validator("sample_type")
    @classmethod
    def normalize_sample_type(cls, v: str) -> str:
        try:
            return SampleType(v).value
        except ValueError:
            logger.warning("Non-standard sample type: '%s'", v)
            return v

    @field_validator("condition")
    @classmethod
    def normalize_condition(cls, v: str) -> str:
        try:
            return Condition(v).value
        except ValueError:
            logger.warning("Non-standard condition: '%s'", v)
            return v


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


class CytokineDataset:
    """Collection of validated cytokine records with loading and query helpers."""

    def __init__(self, records: list[CytokineRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    # -- Loaders -----------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str | Path) -> CytokineDataset:
        """Load and validate records from a CSV file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        return cls._from_dataframe(df)

    @classmethod
    def from_json(cls, path: str | Path) -> CytokineDataset:
        """Load and validate records from a JSON file (list of objects)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {path}")
        with open(path) as f:
            raw = json.load(f)
        records = [CytokineRecord(**r) for r in raw]
        return cls(records)

    @classmethod
    def _from_dataframe(cls, df: pd.DataFrame) -> CytokineDataset:
        """Validate each row as a CytokineRecord."""
        records: list[CytokineRecord] = []
        errors: list[str] = []
        for i, row in df.iterrows():
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            try:
                records.append(CytokineRecord(**row_dict))
            except Exception as e:
                errors.append(f"Row {i}: {e}")
        if errors:
            logger.warning("Validation errors in %d rows:\n%s", len(errors), "\n".join(errors[:10]))
        logger.info("Loaded %d valid records (%d errors)", len(records), len(errors))
        return cls(records)

    # -- Conversion --------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to a pandas DataFrame."""
        return pd.DataFrame([r.model_dump() for r in self.records])

    def to_json(self, path: str | Path) -> None:
        """Save records to a JSON file."""
        with open(path, "w") as f:
            json.dump([r.model_dump() for r in self.records], f, indent=2)

    # -- Query helpers -----------------------------------------------------

    def filter_analyte(self, analyte: str) -> CytokineDataset:
        """Return a subset with only the given analyte."""
        return CytokineDataset([r for r in self.records if r.analyte_name == analyte])

    def filter_condition(self, condition: str) -> CytokineDataset:
        """Return a subset with only the given condition."""
        return CytokineDataset([r for r in self.records if r.condition == condition])

    def filter_treatment(self, treatment: str) -> CytokineDataset:
        """Return a subset where the treatment field matches (case-insensitive)."""
        t_lower = treatment.lower()
        return CytokineDataset([r for r in self.records if r.treatment and r.treatment.lower() == t_lower])

    def analytes(self) -> list[str]:
        """List unique analyte names across all records."""
        return sorted({r.analyte_name for r in self.records})

    def studies(self) -> list[str]:
        """List unique study IDs."""
        return sorted({r.study_id for r in self.records})

    def summary(self) -> dict:
        """Return a summary of the dataset."""
        df = self.to_dataframe()
        return {
            "n_records": len(self.records),
            "n_studies": df["study_id"].nunique(),
            "n_analytes": df["analyte_name"].nunique(),
            "analytes": self.analytes(),
            "conditions": sorted(df["condition"].unique().tolist()),
            "sample_types": sorted(df["sample_type"].unique().tolist()),
        }

    # -- Paired extraction for meta-analysis -------------------------------

    def paired_effects(self, analyte: str, condition_a: str = "flare", condition_b: str = "remission") -> pd.DataFrame:
        """Extract paired study-level summaries for a single analyte.

        Returns a DataFrame with columns: study_id, n_a, mean_a, sd_a, n_b,
        mean_b, sd_b — ready for meta-analysis effect-size computation.
        """
        df = self.to_dataframe()
        df = df[df["analyte_name"] == analyte]
        a = df[df["condition"] == condition_a].set_index("study_id")
        b = df[df["condition"] == condition_b].set_index("study_id")
        shared = sorted(set(a.index) & set(b.index))
        if not shared:
            return pd.DataFrame()
        rows = []
        for sid in shared:
            ra, rb = a.loc[sid], b.loc[sid]
            # Handle case where study has multiple rows per condition
            if isinstance(ra, pd.DataFrame):
                ra = ra.iloc[0]
            if isinstance(rb, pd.DataFrame):
                rb = rb.iloc[0]
            rows.append({
                "study_id": sid,
                "n_a": int(ra["sample_size_n"]),
                "mean_a": float(ra["mean_or_median"]),
                "sd_a": float(ra["sd_or_iqr"]) if ra["sd_or_iqr"] is not None else None,
                "n_b": int(rb["sample_size_n"]),
                "mean_b": float(rb["mean_or_median"]),
                "sd_b": float(rb["sd_or_iqr"]) if rb["sd_or_iqr"] is not None else None,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Template CSV generator
# ---------------------------------------------------------------------------


TEMPLATE_COLUMNS = [
    "study_id", "pmid", "analyte_name", "measurement_method", "sample_type",
    "condition", "sample_size_n", "mean_or_median", "sd_or_iqr", "p_value",
    "treatment", "notes",
]


def write_template_csv(path: str | Path | None = None) -> Path:
    """Write an empty template CSV with the correct column headers.

    If *path* is None, writes to the default data directory.
    """
    if path is None:
        path = DATA_DIR / "cytokine_extraction_template.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(path, index=False)
    logger.info("Wrote template CSV to %s", path)
    return path
