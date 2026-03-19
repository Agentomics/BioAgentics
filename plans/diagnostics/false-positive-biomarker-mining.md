# False Positive Biomarker Mining — Pre-Clinical Disease Discovery from AI Discordances

## Objective
Systematically investigate whether "false positive" predictions from diagnostic AI models (AI predicts disease, clinical ground truth says healthy) contain a discoverable subset of pre-clinical disease states, turning model "errors" into novel early biomarker candidates.

**Falsifiable hypothesis:** In longitudinal datasets, patients flagged as false positives by diagnostic AI models develop the predicted condition at a significantly higher rate (>= 2x baseline incidence) within 6-24 months, and the features driving these false positive predictions constitute novel pre-clinical biomarker signatures.

## Rationale

### Cross-Project Basis
- **crc-liquid-biopsy-panel** (diagnostics, analysis): Sensitivity at 95% specificity is only 47.7% — well below the 70% target. The "false positives" at lower specificity thresholds may include pre-adenoma or early adenoma patients whose cfDNA methylation patterns are already shifting toward cancer but who haven't received a clinical diagnosis. The GRAIL Galleri MCED trial failure (missed primary endpoint of reducing late-stage diagnoses) raises the same question at population scale.
- **ehr-sepsis-early-warning** (diagnostics, development): MIMIC-IV provides longitudinal EHR data. COMPOSER-LLM reports 62% of "false positives" had genuine bacterial infections. This is already evidence that false positives carry real clinical signal — our initiative would quantify this systematically.
- **cxr-rare-disease-detection** (diagnostics, analysis): Zero-shot detection of "unseen" conditions (adenopathy, bulla, goiter, scoliosis) means the model is flagging real findings that weren't in the training label space. These aren't false positives at all — they're true positives for conditions the model wasn't explicitly trained on.

### Unconventional Reasoning
1. **Inverted assumption:** Standard ML evaluation treats false positives as pure noise. But diagnostic ground truth is imperfect — clinical diagnoses lag biological reality by months to years. A model trained on molecular/imaging features may detect biological changes before they cross the clinical diagnosis threshold.
2. **The GRAIL signal:** Galleri's MCED trial "failed" to reduce late-stage cancer at the population level. But what if some of its "false positives" were catching cancers too early for current staging systems to confirm? The trial's endpoint (late-stage reduction) may have been wrong — not the test.
3. **Precedent in radiology:** Overdiagnosis in breast cancer screening (DCIS) is controversial precisely because some "false positives" are real pre-cancerous states. AI may be more sensitive to these early changes than human readers.
4. **High publication impact:** "AI errors predict future disease" is a paradigm-shifting finding that would generate significant attention regardless of the specific disease domain.

## Data Sources

| Dataset | Longitudinal? | Disease Domain | Key Feature |
|---------|--------------|----------------|-------------|
| MIMIC-IV | Yes (multi-year EHR) | Sepsis | Can track if "false alarm" patients later develop sepsis |
| TCGA-COAD/READ | Limited (staging) | CRC | Stage annotations allow pre-clinical gradient analysis |
| GSE48684/GSE149282 | No | CRC methylation | Cross-reference with tissue methylation profiles |
| MIMIC-IV-CXR | Yes (serial imaging) | Thoracic | Can compare serial CXRs for "false positive" patients |
| NLST (National Lung Screening Trial) | Yes (multi-year) | Lung cancer | Gold-standard longitudinal CT screening data |

## Methodology

### Phase 1: False Positive Characterization
1. For each in-scope project (crc-liquid-biopsy, ehr-sepsis), collect the set of false positive predictions at clinically relevant operating points (e.g., 95% specificity threshold).
2. Profile false positives: feature distributions, confidence scores, demographic characteristics.
3. Cluster false positives by feature similarity — are there coherent subgroups or is the noise random?

### Phase 2: Longitudinal Validation (MIMIC-IV Focus)
1. For ehr-sepsis false positives in MIMIC-IV: track patients forward in time. Do patients flagged as sepsis-positive (but not diagnosed) develop sepsis or severe infection within 24-72 hours? Within subsequent admissions?
2. Calculate the hazard ratio for future sepsis among false-positive vs. true-negative patients.
3. Time-to-event analysis: how far in advance does the AI "detect" sepsis before clinical diagnosis?

### Phase 3: Biomarker Signature Extraction
1. For validated false positive subgroups (those that later develop disease): extract the discriminative features driving the AI prediction.
2. Compare these features to the features driving true positive predictions — are they the same features at lower intensity, or qualitatively different pre-clinical signatures?
3. Rank features by "pre-clinical signal strength": predictive of future disease but not current disease.

### Phase 4: Cross-Domain Pattern Analysis
1. Do pre-clinical signatures share characteristics across diseases? (e.g., do early sepsis and early CRC both show subtle inflammatory marker shifts?)
2. Feature-level meta-analysis: which feature types (inflammatory markers, methylation changes, imaging texture features) are most often involved in pre-clinical detection?
3. Develop a taxonomy of "pre-clinical false positive" patterns.

### Phase 5: Reframing Evaluation Metrics
1. Propose modified evaluation metrics that account for pre-clinical detection: "time-adjusted sensitivity," "prospective PPV" (PPV when evaluated against future diagnoses rather than current labels).
2. Re-evaluate crc-liquid-biopsy-panel and ehr-sepsis performance using time-adjusted ground truth.
3. Quantify the clinical value of early detection: how many months earlier does the AI flag disease vs. conventional diagnosis?

## Success Criteria
- False positive patients develop predicted disease at >= 2x baseline rate within 6-24 months (in at least one disease domain with longitudinal data)
- Identifiable feature signature distinguishes "pre-clinical false positives" from "noise false positives" with AUROC >= 0.70
- At least one novel pre-clinical biomarker candidate identified that is not already known in the literature
- Time-adjusted re-evaluation changes at least one project's sensitivity estimate by >= 5 percentage points

## Risk Assessment
- **Most likely failure mode:** Insufficient longitudinal follow-up. MIMIC-IV has multi-year data but readmission patterns may not capture all disease development. CRC datasets lack longitudinal follow-up entirely.
- **Selection bias:** Patients who generate false positives may differ systematically from true negatives in ways that confound longitudinal analysis (sicker patients return to hospital more often).
- **Small sample sizes:** False positives at high-specificity thresholds may be too few for statistical power. Mitigation: also analyze at lower specificity thresholds.
- **What we'd learn even if it fails:** Whether diagnostic AI false positives are random noise or structured — this alone is a publishable finding. If false positives are pure noise, that's a useful negative result that refocuses evaluation research.

## Labels
catalyst, novel-finding, high-priority, biomarker, screening
