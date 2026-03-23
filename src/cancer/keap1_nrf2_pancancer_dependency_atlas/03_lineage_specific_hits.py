"""Phase 3: Lineage-specific KEAP1/NRF2 dependency hits with UXS1/ATR highlighting.

Identifies lineage-specific hits (|d| > 0.5, minimum 3 mutant lines per lineage).
Prioritizes UXS1 and ATR based on journal evidence:
  - Journal #1679: UXS1 pyrimidine vulnerability in KEAP1-mutant lung cancer
  - Journal #1680: KEAP1/STK11 + ATR inhibitor vulnerability (Ceralasertib)

Usage:
    uv run python -m keap1_nrf2_pancancer_dependency_atlas.03_lineage_specific_hits
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

PHASE1_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase1"
)
PHASE2_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase2"
)
OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase3"
)

# Thresholds from the research plan
LINEAGE_EFFECT_SIZE = 0.5  # |d| > 0.5
LINEAGE_MIN_ALTERED = 3  # minimum 3 mutant lines per lineage
FDR_THRESHOLD = 0.1

# NRF2 pathway genes
NRF2_PATHWAY_GENES = {
    "KEAP1", "NFE2L2", "CUL3", "RBX1",
    "NQO1", "GCLM", "GCLC", "HMOX1", "TXNRD1", "AKR1C1", "AKR1C2", "AKR1C3",
    "GPX2", "GSR", "SLC7A11", "ABCC1", "ABCC2", "ABCG2",
    "ME1", "IDH1", "G6PD", "PGD", "TKT", "TALDO1",
    "FTH1", "FTL", "SQSTM1", "MAFG", "MAFK", "MAFF",
}

# Priority genes from journal evidence
JOURNAL_EVIDENCE = {
    "UXS1": {
        "journal_id": 1679,
        "rationale": "Pyrimidine vulnerability in KEAP1-mutant lung cancer. "
                     "UXS1 knockout selectively kills KEAP1-mut cells via UDP-glucuronate "
                     "depletion. Cancer Res Dec 2025.",
        "expected_lineages": ["Lung"],
    },
    "ATR": {
        "journal_id": 1680,
        "rationale": "KEAP1/STK11 co-mutant tumors sensitive to ATR inhibition. "
                     "Ceralasertib (ATR inhibitor) shows clinical validation in "
                     "KEAP1-mut NSCLC. Cancer Cell 2025.",
        "expected_lineages": ["Lung"],
    },
}


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Phase 1 classification and Phase 2 screen results."""
    classified = pd.read_csv(PHASE1_DIR / "keap1_nrf2_classification.csv", index_col=0)
    all_results = pd.read_csv(PHASE2_DIR / "genomewide_all_results.csv")
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    return classified, all_results, qualifying


def extract_lineage_hits(all_results: pd.DataFrame) -> pd.DataFrame:
    """Filter to lineage-specific hits meeting plan criteria."""
    lineage_results = all_results[
        all_results["cancer_type"] != "Pan-cancer (pooled)"
    ].copy()

    hits = lineage_results[
        (lineage_results["fdr"] < FDR_THRESHOLD)
        & (lineage_results["cohens_d"].abs() > LINEAGE_EFFECT_SIZE)
        & (lineage_results["n_altered"] >= LINEAGE_MIN_ALTERED)
    ].copy()

    # Separate gained (SL) vs lost dependencies
    hits["direction"] = np.where(hits["cohens_d"] < 0, "gained_SL", "lost")

    # Annotate pathway membership
    hits["is_nrf2_pathway"] = hits["gene"].isin(NRF2_PATHWAY_GENES)
    hits["is_priority"] = hits["gene"].isin(JOURNAL_EVIDENCE.keys())

    return hits.sort_values("cohens_d").reset_index(drop=True)


def build_gene_summary(
    lineage_hits: pd.DataFrame, all_results: pd.DataFrame
) -> pd.DataFrame:
    """Build per-gene summary across lineages."""
    # Only SL (gained dependency) hits
    sl_hits = lineage_hits[lineage_hits["direction"] == "gained_SL"]

    # Pan-cancer results for reference
    pancancer = all_results[all_results["cancer_type"] == "Pan-cancer (pooled)"].set_index("gene")

    gene_rows = []
    for gene, gene_data in sl_hits.groupby("gene"):
        lineages = gene_data["cancer_type"].unique().tolist()
        best_d = gene_data["cohens_d"].min()
        best_fdr = gene_data.loc[gene_data["cohens_d"].idxmin(), "fdr"]
        best_lineage = gene_data.loc[gene_data["cohens_d"].idxmin(), "cancer_type"]

        pan_d = pancancer.loc[gene, "cohens_d"] if gene in pancancer.index else None
        pan_fdr = pancancer.loc[gene, "fdr"] if gene in pancancer.index else None

        gene_rows.append({
            "gene": gene,
            "n_lineages": len(lineages),
            "lineages": ";".join(lineages),
            "best_cohens_d": round(best_d, 4),
            "best_fdr": float(best_fdr),
            "best_lineage": best_lineage,
            "pancancer_d": round(float(pan_d), 4) if pan_d is not None else None,
            "pancancer_fdr": float(pan_fdr) if pan_fdr is not None else None,
            "is_nrf2_pathway": gene in NRF2_PATHWAY_GENES,
            "is_priority": gene in JOURNAL_EVIDENCE.keys(),
        })

    gene_summary = pd.DataFrame(gene_rows)
    gene_summary = gene_summary.sort_values("best_cohens_d").reset_index(drop=True)
    return gene_summary


def evaluate_priority_genes(
    all_results: pd.DataFrame, classified: pd.DataFrame
) -> list[dict]:
    """Deep evaluation of UXS1 and ATR with journal context."""
    reports = []

    for gene, evidence in JOURNAL_EVIDENCE.items():
        gene_results = all_results[all_results["gene"] == gene]

        report = {
            "gene": gene,
            "journal_id": evidence["journal_id"],
            "rationale": evidence["rationale"],
            "expected_lineages": evidence["expected_lineages"],
            "results": [],
        }

        for _, row in gene_results.iterrows():
            is_sig = row.get("fdr", 1) < FDR_THRESHOLD and row["cohens_d"] < -LINEAGE_EFFECT_SIZE
            is_expected = row["cancer_type"] in evidence["expected_lineages"]

            report["results"].append({
                "cancer_type": row["cancer_type"],
                "cohens_d": round(row["cohens_d"], 4),
                "fdr": row.get("fdr", None),
                "n_altered": int(row["n_altered"]),
                "n_wt": int(row["n_wt"]),
                "significant": is_sig,
                "expected_lineage": is_expected,
            })

        # KL-subtype specific for ATR
        if gene == "ATR" and "KL_subtype" in classified.columns:
            n_kl = classified["KL_subtype"].sum()
            report["kl_subtype_note"] = (
                f"{int(n_kl)} KEAP1+STK11 co-mutant lines available. "
                "ATR inhibitor vulnerability may be enriched in KL subtype."
            )

        reports.append(report)

    return reports


def plot_priority_gene_heatmap(
    all_results: pd.DataFrame, output_dir: Path
) -> None:
    """Heatmap of priority gene effect sizes across lineages."""
    priority_genes = list(JOURNAL_EVIDENCE.keys())
    gene_results = all_results[all_results["gene"].isin(priority_genes)]

    if len(gene_results) == 0:
        return

    pivot = gene_results.pivot_table(
        index="gene", columns="cancer_type", values="cohens_d", aggfunc="first"
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8), 3))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Add values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, label="Cohen's d")
    ax.set_title("UXS1 & ATR: Effect Sizes Across Cancer Types")
    plt.tight_layout()
    fig.savefig(output_dir / "priority_gene_heatmap.png", dpi=150)
    plt.close(fig)


def plot_lineage_hit_counts(gene_summary: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of top SL genes by number of lineages hit."""
    top = gene_summary[~gene_summary["is_nrf2_pathway"]].head(30)
    if len(top) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.3)))
    colors = []
    for _, row in top.iterrows():
        if row["is_priority"]:
            colors.append("#E53935")
        else:
            colors.append("#1E88E5")

    ax.barh(range(len(top)), top["n_lineages"], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["gene"], fontsize=8)
    ax.set_xlabel("Number of lineages with significant SL")
    ax.set_title("Top Lineage-Specific SL Genes (Red = Priority)")
    ax.invert_yaxis()

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["n_lineages"] + 0.1, i, f"d={row['best_cohens_d']:.2f}", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / "lineage_hit_counts.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3: Lineage-Specific Hits + UXS1/ATR ===\n")

    # Load data
    print("Loading Phase 1 + Phase 2 results...")
    classified, all_results, qualifying = load_results()
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(all_results)} total screen results across {len(qualifying_types)} lineages + pan-cancer")

    # Extract lineage-specific hits
    print(f"\nFiltering lineage-specific hits (|d|>{LINEAGE_EFFECT_SIZE}, n>={LINEAGE_MIN_ALTERED}, FDR<{FDR_THRESHOLD})...")
    lineage_hits = extract_lineage_hits(all_results)
    sl_hits = lineage_hits[lineage_hits["direction"] == "gained_SL"]
    lost_hits = lineage_hits[lineage_hits["direction"] == "lost"]
    print(f"  {len(sl_hits)} lineage-specific SL hits")
    print(f"  {len(lost_hits)} lineage-specific lost dependencies")

    # Gene-level summary
    print("\nBuilding gene-level summary...")
    gene_summary = build_gene_summary(lineage_hits, all_results)
    novel_genes = gene_summary[~gene_summary["is_nrf2_pathway"]]
    print(f"  {len(gene_summary)} unique SL genes across lineages")
    print(f"  {len(novel_genes)} novel (non-NRF2-pathway) SL genes")

    # Top lineage-specific SL genes
    print(f"\nTop lineage-specific SL genes (novel):")
    for _, row in novel_genes.head(15).iterrows():
        label = " [PRIORITY]" if row["is_priority"] else ""
        pan_str = f"pan-d={row['pancancer_d']:.3f}" if row["pancancer_d"] is not None else "pan-d=N/A"
        print(f"  {row['gene']}: best d={row['best_cohens_d']:.3f} in {row['best_lineage']}, "
              f"{row['n_lineages']} lineage(s), {pan_str}{label}")

    # Priority gene evaluation
    print("\n" + "=" * 60)
    print("PRIORITY GENE EVALUATION")
    print("=" * 60)
    priority_reports = evaluate_priority_genes(all_results, classified)

    for report in priority_reports:
        print(f"\n  {report['gene']} (Journal #{report['journal_id']})")
        print(f"  Rationale: {report['rationale']}")
        for res in report["results"]:
            sig = " ***" if res["significant"] else ""
            expected = " (expected)" if res["expected_lineage"] else ""
            fdr_str = f"FDR={res['fdr']:.3e}" if res["fdr"] is not None else "FDR=N/A"
            print(f"    {res['cancer_type']}: d={res['cohens_d']:.3f}, {fdr_str}, "
                  f"n={res['n_altered']}v{res['n_wt']}{expected}{sig}")
        if "kl_subtype_note" in report:
            print(f"    KL note: {report['kl_subtype_note']}")

    # Plots
    print("\nGenerating plots...")
    plot_priority_gene_heatmap(all_results, OUTPUT_DIR)
    plot_lineage_hit_counts(gene_summary, OUTPUT_DIR)

    # Save outputs
    print("\nSaving outputs...")
    lineage_hits.to_csv(OUTPUT_DIR / "lineage_specific_hits.csv", index=False)
    print(f"  lineage_specific_hits.csv - {len(lineage_hits)} hits")

    gene_summary.to_csv(OUTPUT_DIR / "gene_summary.csv", index=False)
    print(f"  gene_summary.csv - {len(gene_summary)} genes")

    # Priority gene detail
    priority_rows = []
    for report in priority_reports:
        for res in report["results"]:
            priority_rows.append({
                "gene": report["gene"],
                "journal_id": report["journal_id"],
                **res,
            })
    pd.DataFrame(priority_rows).to_csv(OUTPUT_DIR / "priority_gene_details.csv", index=False)
    print(f"  priority_gene_details.csv")

    # Summary text
    summary_lines = [
        "=" * 70,
        "KEAP1/NRF2 Pan-Cancer Dependency Atlas - Phase 3: Lineage-Specific",
        "=" * 70,
        "",
        f"Lineage-specific SL hits (|d|>{LINEAGE_EFFECT_SIZE}, n>={LINEAGE_MIN_ALTERED}, FDR<{FDR_THRESHOLD}): {len(sl_hits)}",
        f"Unique SL genes: {len(gene_summary)}",
        f"Novel SL genes (non-pathway): {len(novel_genes)}",
        "",
        "TOP NOVEL LINEAGE-SPECIFIC SL GENES",
        "-" * 60,
    ]
    for _, row in novel_genes.head(20).iterrows():
        label = " [PRIORITY]" if row["is_priority"] else ""
        summary_lines.append(
            f"  {row['gene']}: d={row['best_cohens_d']:.3f} in {row['best_lineage']}, "
            f"{row['n_lineages']} lineage(s){label}"
        )

    summary_lines += [
        "",
        "PRIORITY GENE ASSESSMENT",
        "-" * 60,
    ]
    for report in priority_reports:
        summary_lines.append(f"  {report['gene']} (J#{report['journal_id']}): {report['rationale']}")
        for res in report["results"]:
            sig = " ***SIG***" if res["significant"] else ""
            summary_lines.append(
                f"    {res['cancer_type']}: d={res['cohens_d']:.3f}, "
                f"FDR={res['fdr']:.3e if res['fdr'] is not None else 'N/A'}{sig}"
            )

    summary_lines.append("")

    with open(OUTPUT_DIR / "lineage_hits_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  lineage_hits_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
