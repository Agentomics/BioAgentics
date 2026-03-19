"""Step 07 — Critical period gene module analysis (Phase 2).

Identifies gene modules whose expression dynamics match the clinical trajectory
of Tourette syndrome:
  - Module A (onset): genes upregulating in striatum at ~5-7 years
  - Module B (peak): genes peaking at 10-12 years in CSTC circuit
  - Module C (remission): genes increasing in late adolescence (cortex/striatum)

Integrates Phase 1 temporal clusters (step 04) and WGCNA modules (step 06) to
select candidate modules, tests TS GWAS gene enrichment, and performs pathway
enrichment analysis using curated neurodevelopmental gene sets.

Task: #784 (Critical Period Gene Module Analysis)
Project: ts-developmental-trajectory-modeling

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.07_critical_period_modules
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import get_gene_set

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-developmental-trajectory-modeling"

# ── Clinical trajectory windows ──────────────────────────────────────────
# Map TS clinical milestones to BrainSpan developmental stages.

CLINICAL_WINDOWS = {
    "onset": {
        "description": "Tic onset window (5-7 years)",
        "match_stages": ["early_childhood", "late_childhood"],
        "match_patterns_temporal": ["onset_window_peak"],
        "match_patterns_wgcna": ["childhood_peak"],
    },
    "peak_severity": {
        "description": "Peak tic severity (10-12 years)",
        "match_stages": ["late_childhood"],
        "match_patterns_temporal": ["onset_window_peak", "adolescent_peak"],
        "match_patterns_wgcna": ["childhood_peak", "adolescent_rising"],
    },
    "remission": {
        "description": "Spontaneous remission window (15-18 years)",
        "match_stages": ["adolescence"],
        "match_patterns_temporal": ["adolescent_peak"],
        "match_patterns_wgcna": ["adolescent_rising", "adult_peak"],
    },
}

# ── Curated neurodevelopmental pathway gene sets ─────────────────────────
# Used for pathway enrichment when GO/KEGG databases are unavailable locally.
# Curated from KEGG, Reactome, and literature on TS-relevant biology.

NEURODEVEL_PATHWAYS: dict[str, dict[str, str]] = {
    "dopamine_signaling": {
        "DRD1": "D1 dopamine receptor",
        "DRD2": "D2 dopamine receptor",
        "DRD3": "D3 dopamine receptor",
        "DRD4": "D4 dopamine receptor",
        "DRD5": "D5 dopamine receptor",
        "TH": "tyrosine hydroxylase",
        "DDC": "DOPA decarboxylase",
        "SLC6A3": "dopamine transporter (DAT)",
        "COMT": "catechol-O-methyltransferase",
        "MAOA": "monoamine oxidase A",
        "MAOB": "monoamine oxidase B",
    },
    "gaba_signaling": {
        "GAD1": "glutamic acid decarboxylase 67",
        "GAD2": "glutamic acid decarboxylase 65",
        "SLC32A1": "vesicular GABA transporter (VGAT)",
        "GABRA1": "GABA-A receptor alpha1",
        "GABRA2": "GABA-A receptor alpha2",
        "GABRB2": "GABA-A receptor beta2",
        "GABRG2": "GABA-A receptor gamma2",
        "SLC6A1": "GABA transporter 1 (GAT1)",
        "ABAT": "4-aminobutyrate aminotransferase",
    },
    "glutamate_signaling": {
        "GRIN1": "NMDA receptor subunit NR1",
        "GRIN2A": "NMDA receptor subunit NR2A",
        "GRIN2B": "NMDA receptor subunit NR2B",
        "GRIA1": "AMPA receptor subunit GluA1",
        "GRIA2": "AMPA receptor subunit GluA2",
        "GRM5": "metabotropic glutamate receptor 5",
        "SLC17A7": "vesicular glutamate transporter 1 (VGLUT1)",
        "SLC17A6": "vesicular glutamate transporter 2 (VGLUT2)",
    },
    "synaptic_transmission": {
        "SYP": "synaptophysin",
        "SYN1": "synapsin I",
        "SNAP25": "synaptosomal-associated protein 25",
        "STX1A": "syntaxin 1A",
        "VAMP2": "vesicle-associated membrane protein 2",
        "SYT1": "synaptotagmin 1",
        "NRXN1": "neurexin 1",
        "NLGN1": "neuroligin 1",
        "SHANK3": "SH3 and ankyrin repeat domains 3",
        "CADPS2": "Ca-dependent secretion activator 2",
    },
    "axon_guidance": {
        "SEMA6D": "semaphorin 6D",
        "SLITRK1": "SLIT and NTRK-like 1",
        "NTN4": "netrin 4",
        "ROBO1": "roundabout guidance receptor 1",
        "ROBO2": "roundabout guidance receptor 2",
        "SLIT1": "slit guidance ligand 1",
        "SLIT2": "slit guidance ligand 2",
        "DCC": "DCC netrin 1 receptor",
        "CNTN6": "contactin 6",
        "CNTNAP2": "contactin-associated protein-like 2",
    },
    "interneuron_development": {
        "LHX6": "LIM homeobox 6 (striatal interneuron spec)",
        "NKX2-1": "NK2 homeobox 1 (MGE patterning)",
        "DLX1": "distal-less homeobox 1",
        "DLX2": "distal-less homeobox 2",
        "DLX5": "distal-less homeobox 5",
        "DLX6": "distal-less homeobox 6",
        "ARX": "aristaless related homeobox",
        "SOX6": "SRY-box 6 (PV interneuron maturation)",
        "PVALB": "parvalbumin",
        "SST": "somatostatin",
        "CHAT": "choline acetyltransferase",
    },
    "striatal_development": {
        "BCL11B": "BAF chromatin remodeling (CTIP2, MSN diff)",
        "FOXP1": "forkhead box P1 (striatal identity)",
        "FOXP2": "forkhead box P2 (corticostriatal circuits)",
        "ISL1": "ISL LIM homeobox 1 (striatonigral)",
        "EBF1": "early B-cell factor 1 (striatopallidal)",
        "DARPP-32": "dopamine- and cAMP-regulated phosphoprotein",
        "PPP1R1B": "protein phosphatase 1 regulatory subunit 1B",
        "ADORA2A": "adenosine A2a receptor (indirect pathway)",
        "TAC1": "tachykinin precursor 1 (substance P)",
    },
    "chromatin_regulation": {
        "ASH1L": "histone methyltransferase H3K36",
        "HDAC1": "histone deacetylase 1",
        "HDAC2": "histone deacetylase 2",
        "KDM6A": "lysine demethylase 6A",
        "MECP2": "methyl CpG binding protein 2",
        "ARID1A": "AT-rich interaction domain 1A (BAF)",
        "SMARCA4": "SWI/SNF chromatin remodeling ATPase",
    },
    "myelination": {
        "MBP": "myelin basic protein",
        "PLP1": "proteolipid protein 1",
        "MOG": "myelin oligodendrocyte glycoprotein",
        "OLIG2": "oligodendrocyte TF 2",
        "SOX10": "SRY-box 10 (oligodendrocyte diff)",
        "CNP": "2',3'-cyclic nucleotide 3' phosphodiesterase",
        "MAG": "myelin-associated glycoprotein",
    },
}


# ── Phase 1 output loaders ───────────────────────────────────────────────

def load_temporal_clusters(output_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Load temporal cluster assignments and characterization from step 04.

    Returns (assignments_df, cluster_info_list).
    """
    assignments_path = output_dir / "temporal_clusters.csv"
    info_path = output_dir / "cluster_characterization.json"

    if not assignments_path.exists():
        logger.warning("Temporal clusters not found at %s", assignments_path)
        return pd.DataFrame(), []

    assignments = pd.read_csv(assignments_path)

    cluster_info: list[dict] = []
    if info_path.exists():
        with open(info_path) as f:
            cluster_info = json.load(f)

    return assignments, cluster_info


def load_wgcna_modules(output_dir: Path) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Load WGCNA module assignments, dynamics, and enrichment from step 06.

    Returns (modules_df, dynamics_list, enrichment_list).
    """
    wgcna_dir = output_dir / "wgcna_brainspan"

    modules_path = wgcna_dir / "modules.csv"
    dynamics_path = wgcna_dir / "module_dynamics.json"
    enrichment_path = wgcna_dir / "enrichment.json"

    modules = pd.DataFrame()
    dynamics: list[dict] = []
    enrichment: list[dict] = []

    if modules_path.exists():
        modules = pd.read_csv(modules_path)
    else:
        logger.warning("WGCNA modules not found at %s", modules_path)

    if dynamics_path.exists():
        with open(dynamics_path) as f:
            dynamics = json.load(f)

    if enrichment_path.exists():
        with open(enrichment_path) as f:
            enrichment = json.load(f)

    return modules, dynamics, enrichment


# ── Clinical window matching ─────────────────────────────────────────────

def match_temporal_clusters_to_windows(
    cluster_info: list[dict],
    assignments: pd.DataFrame,
) -> dict[str, dict]:
    """Match temporal clusters (step 04) to clinical trajectory windows.

    Returns dict keyed by clinical window name with matched cluster info
    and gene lists.
    """
    matched: dict[str, dict] = {}

    for window_name, window_def in CLINICAL_WINDOWS.items():
        target_patterns = window_def["match_patterns_temporal"]
        matching_clusters = [
            c for c in cluster_info
            if c.get("temporal_pattern") in target_patterns
        ]

        genes: list[str] = []
        for mc in matching_clusters:
            cid = mc["cluster"]
            cluster_genes = assignments[assignments["cluster"] == cid]["gene_symbol"].tolist()
            genes.extend(cluster_genes)

        matched[window_name] = {
            "window": window_def["description"],
            "source": "temporal_clustering",
            "matched_clusters": [
                {
                    "cluster": c["cluster"],
                    "temporal_pattern": c["temporal_pattern"],
                    "peak_stage": c["peak_stage"],
                    "amplitude": c.get("amplitude", 0),
                }
                for c in matching_clusters
            ],
            "genes": sorted(set(genes)),
            "n_genes": len(set(genes)),
        }

    return matched


def match_wgcna_modules_to_windows(
    dynamics: list[dict],
    modules: pd.DataFrame,
    enrichment: list[dict],
) -> dict[str, dict]:
    """Match WGCNA modules (step 06) to clinical trajectory windows.

    Prioritizes modules that are both developmentally dynamic AND enriched
    for TS genes.
    """
    enriched_modules = {
        e["module"] for e in enrichment if e.get("significant", False)
    }

    matched: dict[str, dict] = {}

    for window_name, window_def in CLINICAL_WINDOWS.items():
        target_patterns = window_def["match_patterns_wgcna"]
        matching_dynamics = [
            d for d in dynamics
            if d.get("temporal_pattern") in target_patterns
        ]

        matched_modules: list[dict] = []
        genes: list[str] = []

        for md in matching_dynamics:
            mod_id = md["module"]
            mod_genes = modules[modules["module"] == mod_id]["gene_symbol"].tolist()
            is_enriched = mod_id in enriched_modules

            matched_modules.append({
                "module": mod_id,
                "temporal_pattern": md["temporal_pattern"],
                "peak_stage": md["peak_stage"],
                "amplitude": md.get("amplitude", 0),
                "n_genes": md.get("n_genes", len(mod_genes)),
                "variance_explained": md.get("variance_explained", 0),
                "ts_enriched": is_enriched,
            })
            genes.extend(mod_genes)

        matched[window_name] = {
            "window": window_def["description"],
            "source": "wgcna",
            "matched_modules": matched_modules,
            "genes": sorted(set(genes)),
            "n_genes": len(set(genes)),
            "n_ts_enriched": sum(1 for m in matched_modules if m["ts_enriched"]),
        }

    return matched


def combine_window_genes(
    temporal_matched: dict[str, dict],
    wgcna_matched: dict[str, dict],
) -> dict[str, dict]:
    """Combine genes from temporal clusters and WGCNA modules for each window.

    Creates unified gene sets per clinical window with source tracking.
    """
    combined: dict[str, dict] = {}

    for window_name in CLINICAL_WINDOWS:
        t_genes = set(temporal_matched.get(window_name, {}).get("genes", []))
        w_genes = set(wgcna_matched.get(window_name, {}).get("genes", []))

        union = t_genes | w_genes
        intersection = t_genes & w_genes

        combined[window_name] = {
            "window": CLINICAL_WINDOWS[window_name]["description"],
            "genes_temporal_only": sorted(t_genes - w_genes),
            "genes_wgcna_only": sorted(w_genes - t_genes),
            "genes_both": sorted(intersection),
            "genes_union": sorted(union),
            "n_temporal": len(t_genes),
            "n_wgcna": len(w_genes),
            "n_union": len(union),
            "n_intersection": len(intersection),
        }

    return combined


# ── Enrichment testing ───────────────────────────────────────────────────

def test_ts_enrichment_in_window(
    window_genes: list[str],
    background_genes: list[str],
    ts_gene_set: set[str],
    n_permutations: int = 5000,
) -> dict:
    """Test enrichment of a TS gene subset in a clinical window gene set.

    Uses Fisher's exact test and permutation-based p-value.

    Parameters
    ----------
    window_genes : genes in the clinical window module
    background_genes : all genes tested (universe)
    ts_gene_set : TS genes to test for enrichment
    n_permutations : number of permutations for empirical p-value
    """
    bg = set(background_genes)
    wg = set(window_genes) & bg
    ts = ts_gene_set & bg

    N = len(bg)
    K = len(ts)
    n = len(wg)
    k = len(ts & wg)

    if N == 0 or K == 0 or n == 0:
        return {
            "observed": 0, "expected": 0, "fold_enrichment": 0,
            "fisher_p": 1.0, "permutation_p": 1.0, "significant": False,
            "ts_genes_found": [],
        }

    expected = K * n / N
    fold = k / max(expected, 1e-6)

    # Fisher's exact test (2x2 contingency)
    table = np.array([
        [k, n - k],
        [K - k, N - K - (n - k)],
    ])
    # Ensure no negative values from rounding
    table = np.maximum(table, 0)
    _, fisher_p_raw = stats.fisher_exact(table, alternative="greater")
    fisher_p = float(fisher_p_raw)

    # Permutation p-value
    rng = np.random.default_rng(42)
    bg_list = list(bg)
    perm_count = 0
    for _ in range(n_permutations):
        perm_set = set(rng.choice(bg_list, size=K, replace=False))
        if len(perm_set & wg) >= k:
            perm_count += 1
    perm_p = (perm_count + 1) / (n_permutations + 1)

    return {
        "n_background": N,
        "n_ts_genes": K,
        "n_window_genes": n,
        "observed": k,
        "expected": round(expected, 2),
        "fold_enrichment": round(fold, 2),
        "fisher_p": fisher_p,
        "permutation_p": perm_p,
        "significant": perm_p < 0.05,
        "ts_genes_found": sorted(ts & wg),
    }


def test_pathway_enrichment(
    window_genes: list[str],
    background_genes: list[str],
) -> list[dict]:
    """Test enrichment of a clinical window module against neurodevelopmental pathways.

    Uses Fisher's exact test for each curated pathway.
    """
    bg = set(background_genes)
    wg = set(window_genes) & bg
    N = len(bg)
    n = len(wg)

    if N == 0 or n == 0:
        return []

    results = []
    for pathway_name, pathway_genes in NEURODEVEL_PATHWAYS.items():
        pw = set(pathway_genes.keys()) & bg
        K = len(pw)
        k = len(pw & wg)

        if K == 0:
            continue

        expected = K * n / N
        fold = k / max(expected, 1e-6)

        table = np.array([
            [k, n - k],
            [K - k, N - K - (n - k)],
        ])
        table = np.maximum(table, 0)
        _, fisher_p_raw = stats.fisher_exact(table, alternative="greater")
        fisher_p = float(fisher_p_raw)

        results.append({
            "pathway": pathway_name,
            "pathway_size": K,
            "overlap": k,
            "expected": round(expected, 2),
            "fold_enrichment": round(fold, 2),
            "fisher_p": fisher_p,
            "significant_005": fisher_p < 0.05,
            "significant_bonf": fisher_p < 0.05 / len(NEURODEVEL_PATHWAYS),
            "genes_in_pathway": sorted(pw & wg),
        })

    # Sort by p-value
    results.sort(key=lambda r: r["fisher_p"])
    return results


# ── Visualization ────────────────────────────────────────────────────────

def generate_critical_period_plot(
    combined: dict[str, dict],
    ts_enrichment: dict[str, dict[str, dict]],
    output_path: Path,
) -> Path:
    """Generate summary plot of critical period module analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window_names = list(CLINICAL_WINDOWS.keys())
    n_windows = len(window_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: gene counts per window
    ax = axes[0]
    temporal_counts = [combined[w]["n_temporal"] for w in window_names]
    wgcna_counts = [combined[w]["n_wgcna"] for w in window_names]
    overlap_counts = [combined[w]["n_intersection"] for w in window_names]

    x = np.arange(n_windows)
    width = 0.25
    ax.bar(x - width, temporal_counts, width, label="Temporal clusters", color="#4C72B0")
    ax.bar(x, wgcna_counts, width, label="WGCNA modules", color="#DD8452")
    ax.bar(x + width, overlap_counts, width, label="Overlap", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([CLINICAL_WINDOWS[w]["description"] for w in window_names],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Gene count")
    ax.set_title("Genes per Clinical Window")
    ax.legend(fontsize=8)

    # Right: TS enrichment fold change per window
    ax2 = axes[1]
    gene_sets_tested = list(next(iter(ts_enrichment.values())).keys()) if ts_enrichment else []
    cmap = plt.colormaps["Set2"]
    colors = cmap(np.linspace(0, 1, max(len(gene_sets_tested), 1)))
    bar_width = 0.8 / max(len(gene_sets_tested), 1)

    for i, gs_name in enumerate(gene_sets_tested):
        folds = []
        sig_markers = []
        for w in window_names:
            result = ts_enrichment.get(w, {}).get(gs_name, {})
            folds.append(result.get("fold_enrichment", 0))
            sig_markers.append(result.get("significant", False))

        x_pos = np.arange(n_windows) + i * bar_width
        bars = ax2.bar(x_pos, folds, bar_width, label=gs_name.replace("_", " "),
                       color=colors[i], edgecolor="black", linewidth=0.5)
        for bar, is_sig in zip(bars, sig_markers):
            if is_sig:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.05, "*",
                         ha="center", fontsize=14, fontweight="bold")

    ax2.set_xticks(np.arange(n_windows) + bar_width * (len(gene_sets_tested) - 1) / 2)
    ax2.set_xticklabels([CLINICAL_WINDOWS[w]["description"] for w in window_names],
                        fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel("Fold Enrichment")
    ax2.set_title("TS Gene Enrichment per Window\n(* = permutation p < 0.05)")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    if gene_sets_tested:
        ax2.legend(fontsize=7)

    fig.suptitle("Phase 2: Critical Period Gene Module Analysis", fontsize=13, y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved critical period plot to %s", output_path)
    return output_path


# ── Main pipeline ────────────────────────────────────────────────────────

def run(
    output_dir: Path = OUTPUT_DIR,
    n_permutations: int = 5000,
) -> dict:
    """Run Phase 2 critical period gene module analysis."""
    phase2_dir = output_dir / "phase2_critical_periods"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Phase 1 outputs ─────────────────────────────────────────
    tc_assignments, cluster_info = load_temporal_clusters(output_dir)
    wgcna_modules, wgcna_dynamics, wgcna_enrichment = load_wgcna_modules(output_dir)

    has_temporal = not tc_assignments.empty and len(cluster_info) > 0
    has_wgcna = not wgcna_modules.empty and len(wgcna_dynamics) > 0

    if not has_temporal and not has_wgcna:
        return {"error": "No Phase 1 outputs found. Run steps 04 and/or 06 first."}

    logger.info("Phase 1 data: temporal=%s (%d clusters), wgcna=%s (%d modules)",
                has_temporal, len(cluster_info) if has_temporal else 0,
                has_wgcna, len(wgcna_dynamics) if has_wgcna else 0)

    # ── Match to clinical windows ────────────────────────────────────
    temporal_matched: dict[str, dict] = {}
    wgcna_matched: dict[str, dict] = {}

    if has_temporal:
        temporal_matched = match_temporal_clusters_to_windows(
            cluster_info, tc_assignments,
        )
        with open(phase2_dir / "temporal_window_matching.json", "w") as f:
            json.dump(temporal_matched, f, indent=2)

    if has_wgcna:
        wgcna_matched = match_wgcna_modules_to_windows(
            wgcna_dynamics, wgcna_modules, wgcna_enrichment,
        )
        with open(phase2_dir / "wgcna_window_matching.json", "w") as f:
            json.dump(wgcna_matched, f, indent=2)

    # ── Combine gene sets per window ─────────────────────────────────
    combined = combine_window_genes(temporal_matched, wgcna_matched)
    with open(phase2_dir / "combined_window_genes.json", "w") as f:
        json.dump(combined, f, indent=2)

    # ── Build background gene list ───────────────────────────────────
    all_bg_genes: set[str] = set()
    if has_temporal:
        all_bg_genes.update(tc_assignments["gene_symbol"].tolist())
    if has_wgcna:
        all_bg_genes.update(wgcna_modules["gene_symbol"].tolist())
    bg_list = sorted(all_bg_genes)

    # ── TS gene enrichment per window ────────────────────────────────
    ts_gene_sets = {
        "tsaicg_gwas": set(get_gene_set("tsaicg_gwas").keys()),
        "rare_variant": set(get_gene_set("rare_variant").keys()),
        "de_novo_variant": set(get_gene_set("de_novo_variant").keys()),
        "ts_combined": set(get_gene_set("ts_combined").keys()),
    }

    ts_enrichment: dict[str, dict[str, dict]] = {}
    for window_name in CLINICAL_WINDOWS:
        window_genes = combined[window_name]["genes_union"]
        ts_enrichment[window_name] = {}

        for gs_name, gs_genes in ts_gene_sets.items():
            logger.info("Testing %s enrichment in %s window (%d genes)...",
                        gs_name, window_name, len(window_genes))
            ts_enrichment[window_name][gs_name] = test_ts_enrichment_in_window(
                window_genes, bg_list, gs_genes, n_permutations,
            )

    with open(phase2_dir / "ts_enrichment_per_window.json", "w") as f:
        json.dump(ts_enrichment, f, indent=2)

    # ── Pathway enrichment per window ────────────────────────────────
    pathway_enrichment: dict[str, list[dict]] = {}
    for window_name in CLINICAL_WINDOWS:
        window_genes = combined[window_name]["genes_union"]
        logger.info("Testing pathway enrichment in %s window...", window_name)
        pathway_enrichment[window_name] = test_pathway_enrichment(
            window_genes, bg_list,
        )

    with open(phase2_dir / "pathway_enrichment_per_window.json", "w") as f:
        json.dump(pathway_enrichment, f, indent=2)

    # ── Generate plot ────────────────────────────────────────────────
    generate_critical_period_plot(
        combined, ts_enrichment,
        phase2_dir / "critical_period_summary_plot.png",
    )

    # ── Summary ──────────────────────────────────────────────────────
    significant_ts = []
    for window_name, gs_results in ts_enrichment.items():
        for gs_name, result in gs_results.items():
            if result.get("significant"):
                significant_ts.append({
                    "window": window_name,
                    "gene_set": gs_name,
                    "fold_enrichment": result["fold_enrichment"],
                    "permutation_p": result["permutation_p"],
                    "ts_genes_found": result["ts_genes_found"],
                })

    significant_pathways = []
    for window_name, pw_results in pathway_enrichment.items():
        for pw in pw_results:
            if pw.get("significant_005"):
                significant_pathways.append({
                    "window": window_name,
                    "pathway": pw["pathway"],
                    "fold_enrichment": pw["fold_enrichment"],
                    "fisher_p": pw["fisher_p"],
                    "genes": pw["genes_in_pathway"],
                })

    summary = {
        "phase1_inputs": {
            "temporal_clusters": has_temporal,
            "wgcna_modules": has_wgcna,
            "n_temporal_clusters": len(cluster_info) if has_temporal else 0,
            "n_wgcna_modules": len(wgcna_dynamics) if has_wgcna else 0,
        },
        "clinical_windows": {
            w: {
                "description": CLINICAL_WINDOWS[w]["description"],
                "n_genes_union": combined[w]["n_union"],
                "n_genes_temporal": combined[w]["n_temporal"],
                "n_genes_wgcna": combined[w]["n_wgcna"],
                "n_genes_overlap": combined[w]["n_intersection"],
            }
            for w in CLINICAL_WINDOWS
        },
        "ts_enrichment": {
            "n_tests": sum(len(gs) for gs in ts_enrichment.values()),
            "n_significant": len(significant_ts),
            "significant_findings": significant_ts,
        },
        "pathway_enrichment": {
            "n_pathways_tested": len(NEURODEVEL_PATHWAYS),
            "n_significant": len(significant_pathways),
            "significant_findings": significant_pathways,
        },
    }

    with open(phase2_dir / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Critical period gene module analysis"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = run(output_dir=args.output, n_permutations=args.permutations)

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print("\nPhase 2: Critical Period Gene Module Analysis")
    print(f"  Phase 1 inputs: temporal={summary['phase1_inputs']['temporal_clusters']}, "
          f"wgcna={summary['phase1_inputs']['wgcna_modules']}")

    print("\n  Clinical Windows:")
    for w, info in summary["clinical_windows"].items():
        print(f"    {info['description']}: {info['n_genes_union']} genes "
              f"(temporal={info['n_genes_temporal']}, wgcna={info['n_genes_wgcna']}, "
              f"overlap={info['n_genes_overlap']})")

    ts = summary["ts_enrichment"]
    print(f"\n  TS Enrichment: {ts['n_significant']}/{ts['n_tests']} significant")
    for f in ts["significant_findings"]:
        print(f"    {f['gene_set']} in {f['window']}: "
              f"{f['fold_enrichment']}x (p={f['permutation_p']:.4f})")
        print(f"      Genes: {', '.join(f['ts_genes_found'])}")

    pw = summary["pathway_enrichment"]
    print(f"\n  Pathway Enrichment: {pw['n_significant']}/{pw['n_pathways_tested']} "
          f"pathways significant")
    for f in pw["significant_findings"]:
        print(f"    {f['pathway']} in {f['window']}: "
              f"{f['fold_enrichment']}x (p={f['fisher_p']:.4f})")
        print(f"      Genes: {', '.join(f['genes'])}")


if __name__ == "__main__":
    main()
