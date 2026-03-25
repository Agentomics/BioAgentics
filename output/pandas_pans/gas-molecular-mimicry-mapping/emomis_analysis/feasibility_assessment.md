# EMoMiS Feasibility Assessment

## Tool Overview

EMoMiS (Epitope-based Molecular Mimicry Search) is a pipeline for predicting
conformational B-cell epitope cross-reactivity between proteins using structural
and deep learning approaches.

**Publication:** Stebliankin et al., CSBJ Feb 2026 (bioRxiv: 10.1101/2022.02.05.479274)

### Pipeline Steps
1. **Sequence similarity search** — BLAST against SAbDab (Structural Antibody Database)
2. **Filtering** — ≥3 consecutive AA exact match, surface accessible (RASA >20%), in antibody-antigen contact region (<5Å)
3. **Structural alignment** — TM-align, Z-score thresholds for confidence
4. **Deep learning binding prediction** — MaSIF-Search pre-trained "sc05" model for antibody-antigen binding evaluation

## Hardware Requirements (from paper Section 2.7)

| Component | EMoMiS requirement | Our machine |
|-----------|-------------------|-------------|
| RAM | 128 GB (pipeline), 256 GB (DL) | 8 GB |
| GPU | 8× GeForce GTX 1080 Ti | None |
| CPU | 2×12-core Xeon E5-2670 v3 | ARM (Apple Silicon) |
| Disk | ~1.4 TB (MaSIF full dataset) | Limited |
| Container | Singularity 3.8.5 | Not available |

### MaSIF-Search Dependencies
- Python 3.6, TensorFlow 1.9 (legacy)
- PyMesh 0.1.14, MSMS 2.6.1, APBS 1.5, PDB2PQR 2.1.1
- Open3D 0.5.0.0, Dask 2.2.0
- Pre-computed surface meshes (~400 GB per application)

## Feasibility Verdict: NOT RUNNABLE

**Blocked on 3 independent constraints:**

1. **RAM**: 8 GB vs 128-256 GB minimum — 16-32× shortfall
2. **GPU**: MaSIF-Search requires CUDA GPU; 60× slower on CPU even if RAM were sufficient
3. **Disk/Dependencies**: 1.4 TB MaSIF dataset, legacy TF 1.9, PyMesh (complex build)

## Lightweight Alternative: EMoMiS-Lite

Built `emomis_conformational.py` — a CPU-only reimplementation that captures the
EMoMiS conceptual framework using pre-computed AlphaFold structures already
downloaded by the existing pipeline.

### Approach (maps to EMoMiS steps)
| EMoMiS Step | EMoMiS-Lite Equivalent |
|-------------|----------------------|
| BLAST vs SAbDab | Uses existing BLASTp mimicry hits |
| Surface accessibility filter | DSSP-proxy via neighbor count + protrusion index |
| Structural alignment | Uses existing TM-align results |
| MaSIF-Search DL binding | Physicochemical similarity of surface epitope patches |

### Scoring Components
- **Conformational epitope identification**: pLDDT ≥70, surface-exposed (≤14 CA neighbors within 10Å), protruding (ElliPro-inspired), hydrophilic
- **Epitope overlap**: fraction of aligned positions where both GAS and human residues are in conformational epitope patches
- **Physicochemical similarity**: hydrophobicity + charge + volume similarity at overlapping epitope positions
- **Structural confidence**: minimum pLDDT across the pair

### Results on Top Mimicry Pairs

| Human Target | Conf. Score | Epitope Overlap | Physchem Similarity |
|-------------|-------------|----------------|---------------------|
| GAPDH | 0.456 | 8.2% | 0.777 |
| ENO2 | 0.448 | 6.1% | 0.811 |
| B4DHW5 (HSPA6) | 0.448 | 9.0% | 0.802 |
| B4DHP5 (HSPA6-like) | 0.437 | 5.1% | 0.815 |

### Integration Plan

The conformational mimicry score should be added to the composite scoring in
`target_scoring.py` at 5-8% weight (taking from existing weights proportionally).
This adds a conformational B-cell epitope dimension that complements:
- BLASTp sequence similarity
- Linear B-cell epitope overlap
- MMPred MHC-II T-cell cross-reactivity
- Serotype conservation

The `emomis_target_summary.tsv` output follows the same format as
`mmpred_target_summary.tsv` for straightforward integration.
