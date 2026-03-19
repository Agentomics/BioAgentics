"""Compile consolidated TS risk gene set for compartment analysis.

Combines genes from three evidence streams:
1. TSAICG GWAS significant genes (common variants)
2. Rare variant genes from ts-rare-variant-convergence (rare variants)
3. De novo variant genes from exome studies

Wang et al. snRNA-seq DE genes are not yet available (NDA-restricted data,
DE analysis pending). They will be added when available.

Output: results table with source annotation and evidence level.

Usage:
    uv run python -m bioagentics.tourettes.striosomal_matrix.compile_risk_genes
"""

from __future__ import annotations

import csv
from pathlib import Path

from bioagentics.data.tourettes.gene_sets import (
    DE_NOVO_VARIANT,
    RARE_VARIANT,
    TSAICG_GWAS,
)

# Extended rare variant data from ts-rare-variant-convergence Phase 1
RARE_VARIANT_PHASE1 = Path(
    "data/results/ts-rare-variant-convergence/phase1/rare_variant_genes.csv"
)

OUTPUT_DIR = Path("output/tourettes/ts-striosomal-matrix-subtypes")


def _load_extended_rare_variants() -> dict[str, dict[str, str]]:
    """Load extended rare variant gene annotations from Phase 1 results."""
    genes: dict[str, dict[str, str]] = {}
    if not RARE_VARIANT_PHASE1.exists():
        print(f"  WARN: {RARE_VARIANT_PHASE1} not found, using core rare variants only")
        return genes

    with open(RARE_VARIANT_PHASE1) as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["gene_symbol"]
            genes[symbol] = {
                "evidence_types": row.get("evidence_types", ""),
                "evidence_strength": row.get("evidence_strength", ""),
                "variant_types": row.get("variant_types", ""),
                "pathways": row.get("pathways", ""),
                "chromosome": row.get("chromosome", ""),
            }
    return genes


def compile_risk_genes() -> list[dict[str, str]]:
    """Compile all TS risk genes into a single annotated table.

    Returns list of dicts with keys:
        gene_symbol, source, evidence_level, description, variant_types, pathways
    """
    extended_rare = _load_extended_rare_variants()
    seen: set[str] = set()
    rows: list[dict[str, str]] = []

    # 1. GWAS genes
    for symbol, desc in sorted(TSAICG_GWAS.items()):
        sources = ["GWAS"]
        # Check if also in rare variants
        if symbol in RARE_VARIANT or symbol in extended_rare:
            sources.append("rare")
        if symbol in DE_NOVO_VARIANT:
            sources.append("de_novo")

        rare_info = extended_rare.get(symbol, {})
        rows.append({
            "gene_symbol": symbol,
            "source": ";".join(sources),
            "evidence_level": rare_info.get("evidence_strength", "genome_wide_significant"),
            "description": desc,
            "variant_types": rare_info.get("variant_types", "common"),
            "pathways": rare_info.get("pathways", ""),
            "chromosome": rare_info.get("chromosome", ""),
        })
        seen.add(symbol)

    # 2. Rare variant genes (not already added via GWAS)
    all_rare_symbols = set(RARE_VARIANT.keys()) | set(extended_rare.keys())
    for symbol in sorted(all_rare_symbols - seen):
        sources = ["rare"]
        if symbol in DE_NOVO_VARIANT:
            sources.append("de_novo")

        rare_info = extended_rare.get(symbol, {})
        desc = RARE_VARIANT.get(symbol, rare_info.get("chromosome", ""))
        rows.append({
            "gene_symbol": symbol,
            "source": ";".join(sources),
            "evidence_level": rare_info.get("evidence_strength", "strong"),
            "description": desc,
            "variant_types": rare_info.get("variant_types", "rare"),
            "pathways": rare_info.get("pathways", ""),
            "chromosome": rare_info.get("chromosome", ""),
        })
        seen.add(symbol)

    # 3. De novo variant genes (not already added)
    for symbol in sorted(set(DE_NOVO_VARIANT.keys()) - seen):
        rare_info = extended_rare.get(symbol, {})
        rows.append({
            "gene_symbol": symbol,
            "source": "de_novo",
            "evidence_level": rare_info.get("evidence_strength", "moderate"),
            "description": DE_NOVO_VARIANT[symbol],
            "variant_types": rare_info.get("variant_types", "de_novo"),
            "pathways": rare_info.get("pathways", ""),
            "chromosome": rare_info.get("chromosome", ""),
        })
        seen.add(symbol)

    return rows


def save_risk_genes(output_path: Path | None = None) -> Path:
    """Compile and save TS risk genes to CSV."""
    if output_path is None:
        output_path = OUTPUT_DIR / "ts_risk_genes.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = compile_risk_genes()

    fieldnames = [
        "gene_symbol", "source", "evidence_level", "description",
        "variant_types", "pathways", "chromosome",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    sources = {}
    for r in rows:
        for s in r["source"].split(";"):
            sources[s] = sources.get(s, 0) + 1

    print(f"Compiled {len(rows)} TS risk genes:")
    for s, count in sorted(sources.items()):
        print(f"  {s}: {count} genes")
    multi = sum(1 for r in rows if ";" in r["source"])
    print(f"  multi-source: {multi} genes")
    print(f"Saved to {output_path}")
    return output_path


def main() -> None:
    save_risk_genes()


if __name__ == "__main__":
    main()
