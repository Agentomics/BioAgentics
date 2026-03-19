# Pipeline Verification Report: NOD2 Variant Functional Impact Classifier

**Date:** 2026-03-19
**Reviewer:** validation_scientist
**Project:** nod2-variant-functional-impact
**Division:** crohns
**Verdict:** REJECT — Critical issues prevent advancement

---

## 1. Data Integrity — PASS (with caveats)

| File | Rows | Columns | Status |
|------|------|---------|--------|
| nod2_variants.tsv | 4474 | 11 | OK |
| nod2_structure_features.tsv | 1040 | 6 | OK |
| nod2_predictor_scores.tsv | 4187 | 13 | OK (57-63% NaN in scores — expected for non-coding) |
| nod2_varmeter2_features.tsv | 708 | 12 | OK |
| nod2_girdin_features.tsv | 731 | 10 | OK |
| nod2_training_set.tsv | 311 | 10 | OK |

**Known variants present:**
- R702W (p.Arg702Trp) — ClinVar:VCV000004693, LOF in training set
- G908R (p.Gly908Arg) — ClinVar:VCV000004692, LOF in training set
- L1007fs (p.Leu1007fs) — ClinVar:VCV000004691, LOF in training set

**Caveat:** hgvs_p is 75.8% NaN in variants file (gnomAD variants lack protein annotation).

## 2. Feature Engineering — FAIL

### CRITICAL: Synonymous variant parsing failure

The `_parse_protein_change()` function in `varmeter2.py` cannot parse synonymous variant notation (`p.His10=`, `p.Glu12=`). Since 266/301 neutral training variants are synonymous, their `residue_pos` is NaN. Consequence:

- **266/301 neutral variants (88%) have NO structural, VarMeter2, girdin, or predictor features** after the merge
- All 10 GOF/LOF variants have features (100% match)
- Only **37/311 training samples have complete features** (12%)
- Median imputation fills all 266 neutral variants with identical median values

This creates a catastrophic artifact: the model learns to separate "has real feature values" (GOF/LOF) from "has median-imputed values" (most neutral), not genuine functional biology. The macro AUC of 0.779 is inflated by this artifact.

### Feature merge method issues

- VarMeter2 merges on `residue_pos` with `drop_duplicates("residue_pos")`. This is valid for per-residue features (nSASA, pLDDT) but **incorrect for variant-specific features** (grantham_distance, mutation_energy) — different substitutions at the same residue get the same score.
- Predictor scores merge on `["chrom", "pos", "ref", "alt"]` — correct approach, but also fails for synonymous variants (no predictor scores available).

### NaN in merged training set

| Feature group | NaN % | Root cause |
|--------------|-------|------------|
| Structure (pLDDT, rSASA, active_site_distance) | 85.5% | residue_pos parse failure |
| Predictors (CADD, REVEL, SIFT, etc.) | 86.2% | Synonymous variants lack scores |
| VarMeter2 (nSASA, mutation_energy, grantham) | 85.5% | residue_pos parse failure |
| Girdin features | 85.5% | residue_pos parse failure |
| Domain one-hot | 0% | Falls back to all-zeros |

## 3. Training Set — FAIL

### Class distribution: 5 GOF / 301 neutral / 5 LOF

The 60:1 class imbalance ratio is fundamentally insufficient. `compute_sample_weight("balanced")` assigns 20.73x weight to rare classes but cannot compensate for n=5 per class.

### R587C conflicting annotation (MAJOR)

R587C (p.Arg587Cys) is listed in source code as:
- **LOF** in `LOF_VARIANTS` (source: SURF, NF-kB activity: reduced)
- **GOF** in `GOF_VARIANTS` (source: Literature/Blau, NF-kB activity: enhanced)

Deduplication keeps the LOF assignment (first inserted). With n=5 per rare class, one conflicting variant represents **20% of the LOF training data**. The Blau syndrome attribution needs verification — R587C in SURF showed reduced NF-kB, contradicting the GOF listing.

### ClinVar significance as label (no circular prediction) — PASS

ClinVar significance is used only for training set label construction, NOT as a model feature. No circular prediction detected.

## 4. Model — FAIL

### Cross-validation results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Macro AUC | 0.779 | > 0.90 | FAIL |
| GOF recall | 0.40 | — | Poor |
| LOF recall | 0.00 | — | FAIL |
| Neutral recall | 0.993 | — | Trivial |
| Accuracy | 0.968 | — | Misleading (driven by 97% neutral) |

The model is a **neutral-only predictor**. All three canonical CD LOF variants (R702W, G908R, L1007fs) are classified as neutral with P>0.95.

### Nested CV implementation — PASS (structurally correct)

- 5 outer folds, 3 inner folds for hyperparameter tuning
- StratifiedKFold with random_state=42
- Separate inner CV for GB and LR hyperparameters
- Final model retrained on all data (appropriate)

### No data leakage between inner/outer folds — PASS (but artifact present)

No direct leakage. However, the systematic NaN pattern creates a proxy for class membership (GOF/LOF variants have features; most neutral don't), which is an indirect information leak that inflates apparent performance.

### Feature ablation (from analyst)

- **Domain features only (6): AUC=0.881** — vastly outperforms full model
- **AlphaMissense only: AUC=0.474** — below random
- **Without AlphaMissense: AUC=0.820** — removing the "most important" feature improves performance
- **All predictors only (9): AUC=0.424** — worst performing group

AlphaMissense and other pathogenicity predictors are counterproductive for GOF-vs-LOF classification (they predict binary pathogenicity, not functional direction).

### Resource constraint violation (MINOR)

`n_jobs=-1` in GridSearchCV violates the 8GB RAM constraint. Should be `n_jobs=1`.

## 5. VUS Predictions — FAIL

| Check | Result |
|-------|--------|
| Row count | 665 (correct) |
| Probability sums | 1.0 for all rows (PASS) |
| max_prob consistency | Matches max(prob_GOF, prob_LOF, prob_neutral) (PASS) |
| Class distribution | GOF=125, neutral=493, LOF=47 |
| High-confidence count | 440 (66%) |

### Reliability concerns

- 367/440 (83%) high-confidence predictions are neutral — the 0.8 threshold measures neutral confidence, not discriminative quality
- LOF predictions: 70% low-confidence (33/47) — model has no reliable LOF detection
- 19 high-confidence GOF predictions in LRR domain — biologically implausible (GOF is a NACHT domain phenomenon)
- Only 1 strongly credible novel candidate: p.Arg334Pro (NACHT, same residue as validated Blau GOF R334Q/R334W)

**All VUS predictions should be labeled as EXPLORATORY and not used for clinical interpretation.**

## 6. Unit Tests — PASS

91/91 tests pass. Tests verify module functionality but do not test for the systematic NaN/imputation issues identified above.

---

## Summary of Issues

### CRITICAL (blocks progress)

1. **Synonymous variant parsing failure**: 88% of neutral training data has NaN features, creating an artificial separation artifact. Root: `_parse_protein_change()` can't handle `p.Xnn=` notation.
2. **Model is non-functional**: LOF recall=0.0, macro AUC=0.779 (target 0.90). Model classifies all variants as neutral.
3. **Training set n=5 per rare class**: Insufficient for any reliable classification.

### MAJOR (must fix)

4. **R587C conflicting annotation**: Listed as both LOF and GOF in source code.
5. **AlphaMissense counterproductive**: Highest-importance feature that harms model performance.
6. **VUS predictions unreliable**: GOF predictions in LRR domain are biologically implausible.

### MINOR (should fix)

7. **n_jobs=-1**: Violates 8GB RAM resource constraint.
8. **VarMeter2 merge loses variant specificity**: Per-variant features (grantham_distance) merged per-residue.

---

## Recommendations

1. **Fix synonymous variant parser** to handle `p.Xnn=` notation and re-merge features.
2. **Augment training data** to ≥20-30 per rare class (Infevers, MaveDB, HGMD sources).
3. **Restructure model**: Consider 2-stage approach (pathogenic-vs-benign first, then GOF-vs-LOF with structural features).
4. **Remove/downweight pathogenicity predictors** that conflate GOF and LOF.
5. **Resolve R587C annotation** with literature verification.
6. **Set n_jobs=1** for resource compliance.
7. **Re-evaluate all VUS predictions** after fixes.

---

## Verdict: REJECT

The pipeline has a fundamental data engineering bug (synonymous variant parsing) that cascades into the entire modeling pipeline. The 5:301:5 class imbalance makes the current approach non-viable. The project should return to development for:
1. Parser fix and feature re-merge
2. Training data augmentation
3. Model architecture revision

All existing VUS predictions are unreliable and should not be cited.
