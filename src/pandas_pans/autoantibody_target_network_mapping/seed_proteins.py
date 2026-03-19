"""PANDAS autoantibody target seed proteins.

Defines the complete set of known PANDAS/PANS autoantibody targets with UniProt
identifiers, gene symbols, and mechanism categories. These seed proteins form
the starting nodes for the autoantibody target interaction network.

Targets sourced from:
- Cunningham Panel clinical targets (DRD1, DRD2, GM1 ganglioside, tubulin, CaMKII)
- Neuronal surface glycolytic enzymes identified in PANDAS autoantibody studies
  (pyruvate kinase M1, aldolase C, enolases)
- FOLR1 added per research_director task #304 (63.8% prevalence, distinct folate
  transport mechanism)
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SeedProtein:
    """A PANDAS autoantibody target protein."""

    name: str
    uniprot_id: str
    gene_symbol: str
    mechanism_category: str
    cunningham_panel: bool
    notes: str = ""


MECHANISM_CATEGORIES = (
    "dopaminergic",
    "calcium",
    "metabolic",
    "structural",
    "folate",
)

SEED_PROTEINS: list[SeedProtein] = [
    SeedProtein(
        name="Dopamine D1 receptor",
        uniprot_id="P21728",
        gene_symbol="DRD1",
        mechanism_category="dopaminergic",
        cunningham_panel=True,
        notes="Anti-DRD1 measured in Cunningham Panel",
    ),
    SeedProtein(
        name="Dopamine D2 receptor",
        uniprot_id="P14416",
        gene_symbol="DRD2",
        mechanism_category="dopaminergic",
        cunningham_panel=True,
        notes="Anti-DRD2 measured in Cunningham Panel",
    ),
    SeedProtein(
        name="Tubulin beta-III chain",
        uniprot_id="Q13509",
        gene_symbol="TUBB3",
        mechanism_category="structural",
        cunningham_panel=True,
        notes="Anti-tubulin measured in Cunningham Panel; neuron-specific beta-tubulin",
    ),
    SeedProtein(
        name="CaMKII-alpha",
        uniprot_id="Q9UQM7",
        gene_symbol="CAMK2A",
        mechanism_category="calcium",
        cunningham_panel=True,
        notes="CaMKII activity measured in Cunningham Panel; calcium/calmodulin-dependent protein kinase II",
    ),
    SeedProtein(
        name="Pyruvate kinase M1",
        uniprot_id="P14618",
        gene_symbol="PKM",
        mechanism_category="metabolic",
        cunningham_panel=False,
        notes="Neuronal surface glycolytic enzyme; PKM1 isoform",
    ),
    SeedProtein(
        name="Aldolase C",
        uniprot_id="P09972",
        gene_symbol="ALDOC",
        mechanism_category="metabolic",
        cunningham_panel=False,
        notes="Brain-type aldolase; neuronal surface glycolytic enzyme",
    ),
    SeedProtein(
        name="Alpha-enolase",
        uniprot_id="P06733",
        gene_symbol="ENO1",
        mechanism_category="metabolic",
        cunningham_panel=False,
        notes="Neuronal surface glycolytic enzyme; non-neuronal enolase",
    ),
    SeedProtein(
        name="Gamma-enolase",
        uniprot_id="P09104",
        gene_symbol="ENO2",
        mechanism_category="metabolic",
        cunningham_panel=False,
        notes="Neuron-specific enolase (NSE); neuronal surface glycolytic enzyme",
    ),
    SeedProtein(
        name="Folate receptor alpha",
        uniprot_id="P15328",
        gene_symbol="FOLR1",
        mechanism_category="folate",
        cunningham_panel=False,
        notes="63.8% prevalence in PANDAS; distinct folate transport mechanism; "
        "added per research_director task #304",
    ),
]

# Lysoganglioside GM1 is a glycolipid, not a protein — it does not have a UniProt
# entry. It is included as a Cunningham Panel target but cannot be used as a seed
# node for protein interaction network queries. We record it here for completeness.
GM1_NOTE = (
    "Lysoganglioside GM1 is a glycolipid antigen measured in the Cunningham Panel "
    "(anti-lysoganglioside GM1). It is not a protein and has no UniProt ID. It cannot "
    "serve as a seed node for STRING/BioGRID protein interaction queries. Downstream "
    "signaling effects of anti-GM1 antibodies should be modeled through known GM1-"
    "associated signaling proteins (e.g., RET, TrkA/NTRK1, integrins) in later "
    "analysis steps."
)


def get_seed_dataframe() -> pd.DataFrame:
    """Return seed proteins as a pandas DataFrame.

    Columns: name, uniprot_id, gene_symbol, mechanism_category,
             cunningham_panel, notes
    """
    return pd.DataFrame([vars(p) for p in SEED_PROTEINS])


def get_seed_dict() -> dict[str, dict]:
    """Return seed proteins as a dict keyed by gene symbol.

    Each value contains: uniprot_id, name, mechanism_category,
    cunningham_panel, notes.
    """
    return {
        p.gene_symbol: {
            "uniprot_id": p.uniprot_id,
            "name": p.name,
            "mechanism_category": p.mechanism_category,
            "cunningham_panel": p.cunningham_panel,
            "notes": p.notes,
        }
        for p in SEED_PROTEINS
    }


def get_gene_symbols() -> list[str]:
    """Return list of gene symbols for all seed proteins."""
    return [p.gene_symbol for p in SEED_PROTEINS]


def get_uniprot_ids() -> list[str]:
    """Return list of UniProt IDs for all seed proteins."""
    return [p.uniprot_id for p in SEED_PROTEINS]


def get_by_category(category: str) -> list[SeedProtein]:
    """Return seed proteins filtered by mechanism category."""
    if category not in MECHANISM_CATEGORIES:
        raise ValueError(
            f"Unknown category '{category}'. Valid: {MECHANISM_CATEGORIES}"
        )
    return [p for p in SEED_PROTEINS if p.mechanism_category == category]
