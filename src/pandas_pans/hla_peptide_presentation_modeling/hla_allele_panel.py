"""Phase 1: HLA susceptibility allele panel compilation.

Defines the PANDAS/PANS-associated HLA allele panel with categories
(susceptibility, protective, control, exploratory) based on published
literature from rheumatic fever, CIRS/PANS-ASD, and OCD GWAS studies.

Population frequencies sourced from AFND/NMDP.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "pandas_pans" / "hla-peptide-presentation"


@dataclass(frozen=True)
class HLAAllele:
    """An HLA allele with disease association metadata."""

    allele_name: str
    four_digit_resolution: str
    category: str  # susceptibility, protective, control, exploratory
    hla_class: str  # I or II
    evidence_source: str
    odds_ratio: Optional[float]
    population_frequency: float
    frequency_population: str
    notes: str = ""


ALLELE_CATEGORIES = ("susceptibility", "protective", "control", "exploratory")

HLA_PANEL: list[HLAAllele] = [
    # --- Susceptibility alleles ---
    HLAAllele(
        allele_name="HLA-DRB1*07:01",
        four_digit_resolution="DRB1*07:01",
        category="susceptibility",
        hla_class="II",
        evidence_source="Rheumatic fever meta-analysis; PANDAS HLA association studies",
        odds_ratio=1.68,
        population_frequency=0.1342,
        frequency_population="USA NMDP European Caucasian (n=1,242,890)",
        notes="Strongest RF association among DRB1 alleles. Part of DR53 supratype. "
        "Key allele for MHC-II binding predictions.",
    ),
    HLAAllele(
        allele_name="HLA-DRB1*04:01",
        four_digit_resolution="DRB1*04:01",
        category="susceptibility",
        hla_class="II",
        evidence_source="Rheumatic fever HLA associations; autoimmune polyendocrinopathy",
        odds_ratio=None,
        population_frequency=0.0878,
        frequency_population="USA NMDP European Caucasian (n=1,242,890)",
        notes="Classic rheumatic fever susceptibility allele. Associated with RA and T1D. "
        "DRB1*04 supertypes broadly autoimmune-prone.",
    ),
    HLAAllele(
        allele_name="HLA-DRB1*01:01",
        four_digit_resolution="DRB1*01:01",
        category="susceptibility",
        hla_class="II",
        evidence_source="Rheumatic fever HLA associations",
        odds_ratio=None,
        population_frequency=0.0860,
        frequency_population="USA NMDP European Caucasian (n=1,242,890)",
        notes="Rheumatic fever susceptibility allele. Part of DR1 serological group.",
    ),
    # --- Protective alleles ---
    HLAAllele(
        allele_name="HLA-DRB1*15:01",
        four_digit_resolution="DRB1*15:01",
        category="protective",
        hla_class="II",
        evidence_source="Rheumatic fever protective association",
        odds_ratio=0.60,
        population_frequency=0.1346,
        frequency_population="USA NMDP European Caucasian (n=1,242,890)",
        notes="Protective against RF (OR=0.60). Common worldwide. Serves as negative "
        "control — peptides binding this allele preferentially should NOT be enriched "
        "among differential presenters.",
    ),
    HLAAllele(
        allele_name="HLA-DQA1*03:01",
        four_digit_resolution="DQA1*03:01",
        category="protective",
        hla_class="II",
        evidence_source="CIRS/PANS-ASD overlap study; restricted DR-DQ haplotypes",
        odds_ratio=None,
        population_frequency=0.1320,
        frequency_population="USA San Diego Caucasian (n=496); range 0.099-0.159 in "
        "US/European Caucasians. No NMDP-scale data for DQA1.",
        notes="Part of restricted DR-DQ haplotype from CIRS/PANS-ASD study. "
        "DQA1 chain pairs with DQB1 for MHC-II presentation.",
    ),
    # --- Control alleles ---
    HLAAllele(
        allele_name="HLA-DRB1*03:01",
        four_digit_resolution="DRB1*03:01",
        category="control",
        hla_class="II",
        evidence_source="Common population control; no PANDAS/RF association",
        odds_ratio=None,
        population_frequency=0.1216,
        frequency_population="USA NMDP European Caucasian (n=1,242,890)",
        notes="Common allele without specific PANDAS/RF association. Part of DR52 "
        "supratype. Used as population control for differential binding analysis.",
    ),
    HLAAllele(
        allele_name="HLA-DRB1*13:01",
        four_digit_resolution="DRB1*13:01",
        category="control",
        hla_class="II",
        evidence_source="Common population control; no PANDAS/RF association",
        odds_ratio=None,
        population_frequency=0.0563,
        frequency_population="USA NMDP European Caucasian (n=1,242,890)",
        notes="Common allele without PANDAS/RF association. Part of DR52 supratype. "
        "Used as population control.",
    ),
    # --- Exploratory MHC-I alleles ---
    HLAAllele(
        allele_name="HLA-A*02:01",
        four_digit_resolution="A*02:01",
        category="exploratory",
        hla_class="I",
        evidence_source="Most common MHC-I allele globally; included for completeness",
        odds_ratio=None,
        population_frequency=0.2755,
        frequency_population="USA NMDP European Caucasian (n=1,242,890)",
        notes="Most common HLA-A allele worldwide. Included for MHC-I binding predictions. "
        "CD8+ T cell responses may contribute to tissue damage.",
    ),
    HLAAllele(
        allele_name="HLA-B*07:02",
        four_digit_resolution="B*07:02",
        category="exploratory",
        hla_class="I",
        evidence_source="Common MHC-I allele; included for completeness",
        odds_ratio=None,
        population_frequency=0.1306,
        frequency_population="USA NMDP European Caucasian (n=1,242,890)",
        notes="Common HLA-B allele in European populations. Included for MHC-I "
        "binding predictions alongside HLA-A*02:01.",
    ),
]

# Named allele groups for convenience
ALLELE_GROUPS = {
    "susceptibility": ["HLA-DRB1*07:01", "HLA-DRB1*04:01", "HLA-DRB1*01:01"],
    "protective": ["HLA-DRB1*15:01", "HLA-DQA1*03:01"],
    "control": ["HLA-DRB1*03:01", "HLA-DRB1*13:01"],
    "exploratory_class_i": ["HLA-A*02:01", "HLA-B*07:02"],
    "mhc_ii_all": [
        "HLA-DRB1*07:01", "HLA-DRB1*04:01", "HLA-DRB1*01:01",
        "HLA-DRB1*15:01", "HLA-DQA1*03:01",
        "HLA-DRB1*03:01", "HLA-DRB1*13:01",
    ],
    "mhc_i_all": ["HLA-A*02:01", "HLA-B*07:02"],
}


def get_alleles_by_category(category: str) -> list[HLAAllele]:
    """Return alleles filtered by category."""
    if category not in ALLELE_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Valid: {ALLELE_CATEGORIES}")
    return [a for a in HLA_PANEL if a.category == category]


def get_mhc_ii_alleles() -> list[HLAAllele]:
    """Return all MHC class II alleles."""
    return [a for a in HLA_PANEL if a.hla_class == "II"]


def get_mhc_i_alleles() -> list[HLAAllele]:
    """Return all MHC class I alleles."""
    return [a for a in HLA_PANEL if a.hla_class == "I"]


def get_allele_names(category: Optional[str] = None) -> list[str]:
    """Return allele names, optionally filtered by category."""
    alleles = get_alleles_by_category(category) if category else HLA_PANEL
    return [a.allele_name for a in alleles]


def export_panel_json(output_path: Optional[Path] = None) -> Path:
    """Export the HLA panel to JSON with metadata and allele groups."""
    if output_path is None:
        output_path = DATA_DIR / "hla_allele_panel.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel = {
        "metadata": {
            "description": "HLA allele panel for PANDAS/PANS peptide presentation modeling",
            "project": "hla-peptide-presentation-modeling",
            "division": "pandas_pans",
            "created": "2026-03-17",
            "notes": "Approved panel from research director. "
            "Population frequencies from AFND and published literature.",
        },
        "alleles": [
            {
                "allele_name": a.allele_name,
                "four_digit_resolution": a.four_digit_resolution,
                "category": a.category,
                "class": a.hla_class,
                "evidence_source": a.evidence_source,
                "odds_ratio": a.odds_ratio,
                "population_frequency": a.population_frequency,
                "frequency_population": a.frequency_population,
                "notes": a.notes,
            }
            for a in HLA_PANEL
        ],
        "allele_groups": ALLELE_GROUPS,
    }

    with open(output_path, "w") as f:
        json.dump(panel, f, indent=2)

    return output_path


if __name__ == "__main__":
    path = export_panel_json()
    print(f"Exported {len(HLA_PANEL)} alleles to {path}")
    for cat in ALLELE_CATEGORIES:
        alleles = get_alleles_by_category(cat)
        print(f"  {cat}: {len(alleles)} alleles")
