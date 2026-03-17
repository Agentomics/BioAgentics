"""Curate TS seed gene/protein set for network pharmacology.

Compiles Tourette syndrome disease-relevant genes from multiple evidence streams:
- TSAICG GWAS risk genes (2019 meta-analysis + 2024 update)
- Rare variant genes with strong TS association
- Differentially expressed genes from TS transcriptome studies
- Validated pharmacological targets (dopamine, serotonin, GABA, histamine,
  cannabinoid, PDE10A, muscarinic, immune/neuroinflammatory)

Output: ts_seed_genes.csv with columns [gene_symbol, uniprot_id, source, evidence_type]

Sources:
- Yu et al. (Am J Hum Genet 2019): TSAICG GWAS meta-analysis
- Willsey et al. (Cell 2024): updated GWAS + de novo variant analysis
- Ercan-Sencicek et al. (NEJM 2010): HDC rare variant
- Abelson et al. (Science 2005): SLITRK1 rare variant
- Fernandez et al. (PNAS 2012): NRXN1 deletions in TS
- Paschou et al. (Mol Psychiatry 2014): CNTN6 association
- Lennington et al. (Brain 2016): TS basal ganglia transcriptome
- Tischfield et al. (Neuron 2024): gemlapodect PDE10A validation
- Gerber et al. (AJP 2025): ecopipam D1R Phase 3

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.seed_genes
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import requests

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network"
OUTPUT_PATH = DATA_DIR / "ts_seed_genes.csv"

# ── TSAICG GWAS risk genes ──
# From 2019 meta-analysis (4,819 cases / 9,488 controls) and 2024 update.
# Genome-wide significant loci and suggestive loci with biological plausibility.
GWAS_GENES: dict[str, str] = {
    "FLT3":    "P36888",   # 13q12 — receptor tyrosine kinase, immune/neuro
    "MEIS1":   "O00470",   # 2p14 — homeobox TF, dopamine neuron development
    "PTPRD":   "P23468",   # 9p24 — receptor protein tyrosine phosphatase delta
    "SEMA6D":  "Q8NBL1",   # 15q21 — semaphorin, axon guidance
    "NTN4":    "Q9HB63",   # 12q22 — netrin-4, axon guidance
    "COL27A1": "Q8IZC6",   # 9q32 — collagen XXVII alpha-1, neural ECM
    "CADPS2":  "Q86UW7",   # 7q31 — Ca2+-dependent secretion activator
    "OPRD1":   "P41143",   # 1p36 — delta opioid receptor
    "ASH1L":   "Q9NR48",   # 1q22 — histone methyltransferase
    "CELSR3":  "Q9NYQ7",   # 3p21 — planar cell polarity, neural
    "MAOA":    "P21397",   # Xp11 — monoamine oxidase A
    "NRXN1":   "Q9ULB1",   # 2p16 — neurexin 1, synaptic adhesion
    "CNTNAP2": "Q9UHC6",   # 7q35 — contactin-associated protein-like 2
    "NEGR1":   "Q7Z3B1",   # 1p31 — neuronal growth regulator 1
    "LHX6":    "Q9UPM6",   # 9q33 — LIM homeobox 6, interneuron specification
    "RBFOX1":  "Q9NWB1",   # 16p13 — RNA-binding Fox-1, neuronal splicing
}

# ── Rare variant genes ──
# Genes with rare penetrant variants in TS families.
RARE_VARIANT_GENES: dict[str, str] = {
    "SLITRK1": "Q96PX8",   # SLIT/NTRK-like 1, axon guidance (Abelson 2005)
    "HDC":     "P19113",   # Histidine decarboxylase (Ercan-Sencicek 2010)
    "NRXN1":   "Q9ULB1",   # Neurexin 1, synaptic adhesion (Fernandez 2012)
    "CNTN6":   "Q9UQ52",   # Contactin 6 (Paschou 2014)
    "WWC1":    "Q8TAF3",   # WW domain-containing 1 / KIBRA (Willsey 2024)
}

# ── Differentially expressed genes from TS transcriptome studies ──
# From Lennington et al. (Brain 2016): basal ganglia (caudate/putamen)
# postmortem TS vs control.
TRANSCRIPTOME_DE_GENES: dict[str, str] = {
    "PENK":    "P01210",   # Proenkephalin — downregulated in TS striatum
    "TAC1":    "P20366",   # Tachykinin precursor 1 (substance P)
    "GAD1":    "Q99259",   # Glutamic acid decarboxylase 67 — GABA synthesis
    "GAD2":    "Q05329",   # Glutamic acid decarboxylase 65
    "NPY":     "P01303",   # Neuropeptide Y — striatal interneuron marker
    "SST":     "P61278",   # Somatostatin — interneuron marker
    "PVALB":   "P20472",   # Parvalbumin — fast-spiking interneuron marker
    "CHAT":    "P28329",   # Choline acetyltransferase — cholinergic interneuron
    "TH":      "P07101",   # Tyrosine hydroxylase — dopamine synthesis
    "SLC6A3":  "Q01959",   # Dopamine transporter (DAT)
    "SLC6A4":  "P31645",   # Serotonin transporter (SERT)
    "DRD3":    "P35462",   # Dopamine receptor D3
}

# ── Validated pharmacological targets ──
# Drugs with clinical evidence in TS, organized by pathway.

# Dopamine pathway
DOPAMINE_TARGETS: dict[str, str] = {
    "DRD1":    "P21728",   # D1 receptor — ecopipam (FDA breakthrough, Phase 3)
    "DRD2":    "P14416",   # D2 receptor — haloperidol, aripiprazole
    "DRD4":    "P21917",   # D4 receptor — clozapine
    "SLC18A2": "Q05940",   # VMAT2 — valbenazine, deutetrabenazine (negative ctrl)
}

# PDE10A / cAMP-cGMP intracellular signaling
PDE10A_TARGETS: dict[str, str] = {
    "PDE10A":  "Q9Y233",   # Phosphodiesterase 10A — gemlapodect Phase 2a
    "PDE1B":   "Q01064",   # PDE1B — striatal expression, calcium/calmodulin
}

# Muscarinic cholinergic (M4 receptor pathway)
MUSCARINIC_TARGETS: dict[str, str] = {
    "CHRM4":   "P08173",   # M4 muscarinic receptor — KarXT (orthosteric agonist)
    "CHRM1":   "P11229",   # M1 muscarinic receptor — KarXT secondary target
}

# Serotonin pathway
SEROTONIN_TARGETS: dict[str, str] = {
    "HTR2A":   "P28223",   # 5-HT2A — risperidone, clozapine
    "HTR2C":   "P28335",   # 5-HT2C — lorcaserin pilot
    "HTR1A":   "P08908",   # 5-HT1A — buspirone
}

# GABA / glutamate balance
GABA_GLUTAMATE_TARGETS: dict[str, str] = {
    "GABRA1":  "P14867",   # GABA-A alpha 1 — clonazepam
    "GABRG2":  "P18507",   # GABA-A gamma 2
    "GRIN2B":  "Q13224",   # NMDA receptor subunit 2B
    "SLC1A2":  "P43004",   # Glutamate transporter EAAT2
}

# Histamine H3 receptor
HISTAMINE_TARGETS: dict[str, str] = {
    "HRH3":    "Q9Y5N1",   # Histamine H3 receptor — pitolisant
}

# Endocannabinoid system
CANNABINOID_TARGETS: dict[str, str] = {
    "CNR1":    "P21554",   # CB1 receptor
    "CNR2":    "P34972",   # CB2 receptor
    "FAAH":    "O00519",   # Fatty acid amide hydrolase
    "MGLL":    "Q99NB1",   # Monoglyceride lipase (MAGL) — endocannabinoid
}

# Alpha-2 adrenergic (first-line for mild TS)
ADRENERGIC_TARGETS: dict[str, str] = {
    "ADRA2A":  "P08913",   # Alpha-2A adrenergic — clonidine, guanfacine
    "ADRA2C":  "P18825",   # Alpha-2C adrenergic
}

# Immune / neuroinflammatory module
# Cross-referenced with ts-neuroimmune-subtyping project
IMMUNE_TARGETS: dict[str, str] = {
    "TNF":     "P01375",   # Tumor necrosis factor
    "IL12A":   "P29459",   # Interleukin-12 subunit alpha
    "IL12B":   "P29460",   # Interleukin-12 subunit beta (shared with IL-23)
    "IL23A":   "Q9NPF7",   # Interleukin-23 subunit alpha
    "JAK1":    "P23458",   # Janus kinase 1
    "JAK2":    "O60674",   # Janus kinase 2
    "JAK3":    "P52333",   # Janus kinase 3
    "STAT3":   "P40763",   # Signal transducer and activator of transcription 3
    "CCL2":    "P13500",   # MCP-1 / monocyte chemoattractant protein 1
    "CCR2":    "P41597",   # C-C chemokine receptor type 2
    "IL1RN":   "P18510",   # Interleukin-1 receptor antagonist (anakinra)
    "TGFB1":   "P01137",   # TGF-beta 1 — neuroinflammation
}


def compile_seed_genes() -> list[dict[str, str]]:
    """Compile all seed genes into a unified list with provenance."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(genes: dict[str, str], source: str, evidence: str) -> None:
        for symbol, uniprot in genes.items():
            if symbol not in seen:
                rows.append({
                    "gene_symbol": symbol,
                    "uniprot_id": uniprot,
                    "source": source,
                    "evidence_type": evidence,
                })
                seen.add(symbol)

    _add(GWAS_GENES, "TSAICG_GWAS", "gwas")
    _add(RARE_VARIANT_GENES, "rare_variant_studies", "rare_variant")
    _add(TRANSCRIPTOME_DE_GENES, "Lennington_2016_Brain", "transcriptome_de")
    _add(DOPAMINE_TARGETS, "dopamine_pathway", "validated_target")
    _add(PDE10A_TARGETS, "PDE10A_cAMP_cGMP_pathway", "validated_target")
    _add(MUSCARINIC_TARGETS, "muscarinic_cholinergic_pathway", "validated_target")
    _add(SEROTONIN_TARGETS, "serotonin_pathway", "validated_target")
    _add(GABA_GLUTAMATE_TARGETS, "GABA_glutamate_pathway", "validated_target")
    _add(HISTAMINE_TARGETS, "histamine_H3_pathway", "validated_target")
    _add(CANNABINOID_TARGETS, "endocannabinoid_system", "validated_target")
    _add(ADRENERGIC_TARGETS, "alpha2_adrenergic_pathway", "validated_target")
    _add(IMMUNE_TARGETS, "immune_neuroinflammatory", "validated_target")

    return rows


def validate_uniprot_ids(genes: list[dict[str, str]], batch_size: int = 50) -> None:
    """Validate UniProt IDs via UniProt REST API (in-place update)."""
    ids = [g["uniprot_id"] for g in genes]
    # Query in batches
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        query = " OR ".join(f"accession:{uid}" for uid in batch)
        try:
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={"query": query, "fields": "accession,gene_primary", "size": batch_size},
                headers={"Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                found = {
                    r["primaryAccession"]
                    for r in data.get("results", [])
                }
                for g in genes[i : i + batch_size]:
                    if g["uniprot_id"] not in found:
                        print(f"  WARNING: UniProt ID {g['uniprot_id']} for {g['gene_symbol']} not found")
        except requests.RequestException as e:
            print(f"  UniProt validation skipped (batch {i}): {e}")


def save_seed_genes(genes: list[dict[str, str]], output: Path) -> None:
    """Save seed genes to CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gene_symbol", "uniprot_id", "source", "evidence_type"])
        writer.writeheader()
        writer.writerows(genes)
    print(f"  Saved {len(genes)} seed genes to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate TS seed gene set")
    parser.add_argument("--validate", action="store_true", help="Validate UniProt IDs via API")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print("Compiling TS seed gene set...")
    genes = compile_seed_genes()

    # Summary by evidence type
    from collections import Counter
    counts = Counter(g["evidence_type"] for g in genes)
    for etype, n in counts.most_common():
        print(f"  {etype}: {n} genes")
    print(f"  Total: {len(genes)} unique genes")

    if args.validate:
        print("Validating UniProt IDs...")
        validate_uniprot_ids(genes)

    save_seed_genes(genes, args.output)


if __name__ == "__main__":
    main()
