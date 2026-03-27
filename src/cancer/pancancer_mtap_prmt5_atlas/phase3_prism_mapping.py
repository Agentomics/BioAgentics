"""Phase 3 Step 3: PRISM drug mapping.

Map 263 predictable dependency genes to drug sensitivity data from PRISM
repurposing and curated clinical compounds. Stratify by NSCLC molecular
subtype and apply PRISM-CRISPR concordance filtering.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


DATA_DIR = Path("data/results")
DEPMAP_DIR = Path("data/depmap/25q3")
OUT_DIR = DATA_DIR / "prism_mapping"

# Curated drug-target mapping: gene -> list of (drug_name, mechanism, clinical_stage, caveats)
DRUG_TARGET_MAP = {
    "EGFR": [
        ("erlotinib", "EGFR TKI", "approved", ""),
        ("gefitinib", "EGFR TKI", "approved", ""),
        ("osimertinib", "EGFR TKI (3rd gen)", "approved", ""),
        ("afatinib", "pan-ERBB TKI", "approved", ""),
    ],
    "ERBB3": [
        ("afatinib", "pan-ERBB TKI", "approved", ""),
    ],
    "IGF1R": [
        ("linsitinib", "IGF1R inhibitor", "phase2", ""),
    ],
    "PTK2": [
        ("defactinib", "FAK inhibitor", "phase2", ""),
    ],
    "CDH1": [],  # No direct inhibitor
    "CCND1": [
        ("palbociclib", "CDK4/6 inhibitor", "approved", ""),
        ("ribociclib", "CDK4/6 inhibitor", "approved", ""),
        ("abemaciclib", "CDK4/6 inhibitor", "approved", ""),
    ],
    "CDC25A": [],
    "MDM2": [
        ("idasanutlin", "MDM2 inhibitor", "phase2", ""),
        ("navtemadlin", "MDM2 inhibitor", "phase2", ""),
    ],
    "MDM4": [
        ("ALRN-6924", "MDM2/MDMX stapled peptide", "phase1", ""),
    ],
    "PTEN": [],  # Tumor suppressor, not druggable target
    "TSC1": [],  # Upstream of mTOR
    "EZH2": [
        ("tazemetostat", "EZH2 inhibitor", "approved", ""),
    ],
    "EP300": [
        ("CCS1477", "p300/CBP inhibitor", "phase1", ""),
    ],
    "BRD9": [
        ("BI-7273", "BRD9 degrader", "preclinical", ""),
    ],
    "STAT3": [
        ("napabucasin", "STAT3 inhibitor", "phase3", ""),
    ],
    "PRKACA": [
        ("H89", "PKA inhibitor", "tool compound", ""),
    ],
    "CTNNB1": [],  # Difficult drug target
    "YAP1": [],  # Transcriptional co-activator
    "WWTR1": [],  # TAZ, similar to YAP1
    "LATS2": [],  # Hippo pathway kinase
    "NFE2L2": [],  # NRF2, transcription factor
    "SMARCA2": [
        ("FHD-286", "SMARCA2 degrader", "phase1", ""),
        ("AU-15330", "SMARCA2/4 degrader", "preclinical", ""),
    ],
    "FOSL1": [],  # AP-1 component
    "TXNRD1": [
        ("auranofin", "thioredoxin reductase inhibitor", "phase2", "NRF2 target gene"),
    ],
    "SLC16A3": [
        ("AZD0095", "MCT4 inhibitor", "preclinical",
         "AstraZeneca clinical candidate; IC50=1.3nM, >1000x MCT1 selectivity; PMID:36525250; no registered Phase 1 trial"),
    ],
    "SLC2A1": [],  # GLUT1 transporter
    "NCOA4": [],  # Ferritinophagy receptor
    "NMT1": [
        ("IMP-1088", "NMT1/2 inhibitor", "preclinical", ""),
    ],
    "PCNA": [],  # Essential replication factor
    "CSK": [],  # Src-family regulator
    "PSMB7": [
        ("bortezomib", "proteasome inhibitor", "approved", "pan-proteasome, not subunit-specific"),
    ],
}

# Clinical compounds from task spec (may not be in PRISM)
CLINICAL_COMPOUNDS = [
    {"drug": "bemcentinib", "target_gene": "AXL", "mechanism": "AXL inhibitor",
     "stage": "phase2", "caveat": ""},
    {"drug": "ceralasertib", "target_gene": "ATR", "mechanism": "ATR inhibitor",
     "stage": "phase3", "caveat": "LATIFY Phase 3 failure: DDR targets may not translate clinically in IO combinations"},
    {"drug": "TNG260", "target_gene": "RCOR1", "mechanism": "CoREST inhibitor",
     "stage": "phase1", "caveat": ""},
    {"drug": "TNO155", "target_gene": "PTPN11", "mechanism": "SHP2 inhibitor",
     "stage": "phase2", "caveat": ""},
    {"drug": "RMC-4630", "target_gene": "PTPN11", "mechanism": "SHP2 inhibitor",
     "stage": "phase2", "caveat": ""},
    {"drug": "vistusertib", "target_gene": "MTOR", "mechanism": "mTOR inhibitor",
     "stage": "phase2", "caveat": "Known negative: vistusertib failed in NSCLC trials"},
]


def load_data():
    """Load datasets."""
    predictable = pd.read_csv(DATA_DIR / "predictable_genes.txt", header=None)[0].tolist()

    # PRISM treatment meta
    treat_meta = pd.read_csv(DEPMAP_DIR / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")
    treat_meta = treat_meta[["broad_id", "name"]].drop_duplicates().dropna(subset=["name"])

    # PRISM sensitivity matrix
    prism_data = pd.read_csv(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )

    # NSCLC cell lines with subtype annotations
    nsclc_cl = pd.read_csv(DEPMAP_DIR / "nsclc_cell_lines_annotated.csv")

    # CRISPR dependency
    crispr = pd.read_csv(DEPMAP_DIR / "CRISPRGeneEffect.csv", index_col=0)

    return predictable, treat_meta, prism_data, nsclc_cl, crispr


def build_gene_drug_table(predictable, treat_meta):
    """Build gene-drug mapping table combining curated + PRISM search."""
    rows = []

    # From curated map
    for gene, drugs in DRUG_TARGET_MAP.items():
        if gene not in predictable:
            continue
        if not drugs:
            rows.append({
                "gene": gene, "drug": "", "mechanism": "no known inhibitor",
                "clinical_stage": "", "in_prism": False, "broad_id": "",
                "caveat": "",
            })
            continue
        for drug_name, mech, stage, caveat in drugs:
            # Check if in PRISM
            prism_match = treat_meta[treat_meta["name"].str.lower() == drug_name.lower()]
            in_prism = len(prism_match) > 0
            bid = prism_match["broad_id"].iloc[0] if in_prism else ""
            rows.append({
                "gene": gene, "drug": drug_name, "mechanism": mech,
                "clinical_stage": stage, "in_prism": in_prism, "broad_id": bid,
                "caveat": caveat,
            })

    # Clinical compounds
    for cc in CLINICAL_COMPOUNDS:
        prism_match = treat_meta[treat_meta["name"].str.lower() == cc["drug"].lower()]
        in_prism = len(prism_match) > 0
        bid = prism_match["broad_id"].iloc[0] if in_prism else ""
        rows.append({
            "gene": cc["target_gene"], "drug": cc["drug"],
            "mechanism": cc["mechanism"], "clinical_stage": cc["stage"],
            "in_prism": in_prism, "broad_id": bid,
            "caveat": cc["caveat"],
        })

    return pd.DataFrame(rows)


def extract_prism_sensitivity(gene_drug_df, prism_data, nsclc_cl):
    """Extract PRISM sensitivity for matched drugs in NSCLC cell lines by subtype."""
    prism_drugs = gene_drug_df[gene_drug_df["in_prism"] & (gene_drug_df["broad_id"] != "")]
    nsclc_ids = set(nsclc_cl["ModelID"])
    # PRISM columns are cell line IDs
    nsclc_prism_cols = [c for c in prism_data.columns if c in nsclc_ids]

    # Build core-ID lookup for PRISM index (handles suffix mismatches)
    def _core_bid(bid_str):
        return "-".join(bid_str.replace("BRD:", "").split("-")[:3])

    prism_core_map = {}
    for idx in prism_data.index:
        prism_core_map.setdefault(_core_bid(idx), []).append(idx)

    rows = []
    for _, drug_row in prism_drugs.iterrows():
        bid = drug_row["broad_id"]
        core = _core_bid(bid)
        matching_rows = prism_core_map.get(core, [])
        if not matching_rows:
            continue

        bid_key = matching_rows[0]
        sensitivities = prism_data.loc[bid_key, nsclc_prism_cols].dropna()
        if len(sensitivities) == 0:
            continue

        for cl_id, sens_val in sensitivities.items():
            cl_info = nsclc_cl[nsclc_cl["ModelID"] == cl_id]
            if cl_info.empty:
                continue
            subtype = cl_info["molecular_subtype"].iloc[0]
            rows.append({
                "gene": drug_row["gene"],
                "drug": drug_row["drug"],
                "cell_line": cl_id,
                "cell_line_name": cl_info["CellLineName"].iloc[0],
                "molecular_subtype": subtype,
                "prism_sensitivity": sens_val,
            })

    sens_df = pd.DataFrame(rows)
    if sens_df.empty:
        return sens_df, pd.DataFrame()

    # Summarize by subtype
    summary = (
        sens_df.groupby(["gene", "drug", "molecular_subtype"])["prism_sensitivity"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    summary.columns = ["gene", "drug", "molecular_subtype", "mean_sensitivity",
                       "median_sensitivity", "std_sensitivity", "n_cell_lines"]

    return sens_df, summary


def concordance_filter(gene_drug_df, prism_data, crispr, nsclc_cl):
    """Vermeulen PRISM-CRISPR concordance: keep drugs where sensitivity
    correlates with CRISPR dependency for the target gene in NSCLC."""
    prism_drugs = gene_drug_df[gene_drug_df["in_prism"] & (gene_drug_df["broad_id"] != "")]
    nsclc_ids = set(nsclc_cl["ModelID"])

    # CRISPR columns are "GENE (ENTREZID)" format — extract gene names
    crispr_gene_cols = {}
    for col in crispr.columns:
        gene_name = col.split(" (")[0] if " (" in col else col
        crispr_gene_cols[gene_name] = col

    def _core_bid(bid_str):
        return "-".join(bid_str.replace("BRD:", "").split("-")[:3])

    prism_core_map = {}
    for idx in prism_data.index:
        prism_core_map.setdefault(_core_bid(idx), []).append(idx)

    rows = []
    for _, drug_row in prism_drugs.iterrows():
        gene = drug_row["gene"]
        bid = drug_row["broad_id"]
        core = _core_bid(bid)
        matching_rows = prism_core_map.get(core, [])

        if not matching_rows:
            continue
        if gene not in crispr_gene_cols:
            continue

        crispr_col = crispr_gene_cols[gene]

        # Get overlapping NSCLC cell lines
        prism_cls = set(prism_data.columns) & nsclc_ids
        crispr_cls = set(crispr.index) & nsclc_ids
        overlap_cls = list(prism_cls & crispr_cls)

        if len(overlap_cls) < 10:
            continue

        bid_key = matching_rows[0]
        prism_vals = prism_data.loc[bid_key, overlap_cls].astype(float)
        crispr_vals = crispr.loc[overlap_cls, crispr_col].astype(float)

        # Drop NaN
        valid = prism_vals.notna() & crispr_vals.notna()
        if valid.sum() < 10:
            continue

        r, p = stats.spearmanr(
            prism_vals[valid].values,
            crispr_vals[valid].values,
        )

        rows.append({
            "gene": gene,
            "drug": drug_row["drug"],
            "broad_id": bid,
            "spearman_r": r,
            "spearman_p": p,
            "n_cell_lines": int(valid.sum()),
            "concordant": r > 0.2 and p < 0.05,
        })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    predictable, treat_meta, prism_data, nsclc_cl, crispr = load_data()
    print(f"Predictable genes: {len(predictable)}")
    print(f"PRISM drugs: {len(treat_meta)}")
    print(f"NSCLC cell lines: {len(nsclc_cl)}")

    print("\nBuilding gene-drug mapping...")
    gene_drug_df = build_gene_drug_table(predictable, treat_meta)
    print(f"  Total mappings: {len(gene_drug_df)}")
    print(f"  In PRISM: {gene_drug_df['in_prism'].sum()}")
    print(f"  Unique genes with drugs: {gene_drug_df[gene_drug_df['drug'] != '']['gene'].nunique()}")

    print("\nExtracting PRISM sensitivity for NSCLC cell lines...")
    sens_df, summary_df = extract_prism_sensitivity(gene_drug_df, prism_data, nsclc_cl)
    print(f"  Sensitivity measurements: {len(sens_df)}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    print("\nApplying PRISM-CRISPR concordance filter...")
    concordance_df = concordance_filter(gene_drug_df, prism_data, crispr, nsclc_cl)
    if not concordance_df.empty:
        print(f"  Tested: {len(concordance_df)}")
        print(f"  Concordant: {concordance_df['concordant'].sum()}")
        print(concordance_df.to_string(index=False))
    else:
        print("  No drugs with sufficient PRISM + CRISPR overlap for concordance testing")

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gene_drug_df.to_csv(OUT_DIR / "gene_drug_matches.csv", index=False)
    if not summary_df.empty:
        summary_df.to_csv(OUT_DIR / "subtype_drug_sensitivity.csv", index=False)
    else:
        pd.DataFrame(columns=["gene", "drug", "molecular_subtype", "mean_sensitivity",
                               "median_sensitivity", "std_sensitivity", "n_cell_lines"]).to_csv(
            OUT_DIR / "subtype_drug_sensitivity.csv", index=False)

    if not concordance_df.empty:
        concordance_df.to_csv(OUT_DIR / "concordance_filtered.csv", index=False)
    else:
        pd.DataFrame(columns=["gene", "drug", "broad_id", "spearman_r",
                               "spearman_p", "n_cell_lines", "concordant"]).to_csv(
            OUT_DIR / "concordance_filtered.csv", index=False)

    print(f"\nOutputs saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
