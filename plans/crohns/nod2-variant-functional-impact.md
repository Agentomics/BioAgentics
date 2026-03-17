# NOD2 Variant Functional Impact Prediction

## Objective
Build a computational model to predict the functional impact of NOD2 variants on innate immune signaling and Crohn's disease risk, enabling classification of variants of uncertain significance (VUS).

## Background
NOD2 is the strongest single-gene risk factor for Crohn's disease. The three major risk variants (R702W, G908R, L1007fsinsC) are well-characterized, but hundreds of rare NOD2 variants exist with unknown clinical significance. Current GWAS approaches identify risk loci but cannot distinguish functional from benign rare variants. A predictive model combining structural, evolutionary, and clinical features could classify VUS and improve genetic counseling for IBD patients.

## Data Sources
- **ClinVar**: Annotated NOD2 variants with clinical significance ratings (accession: VCV000005321 and related)
- **gnomAD v4**: Population allele frequencies for NOD2 variants
- **IBDGC GWAS summary statistics**: Association data from the International IBD Genetics Consortium (doi:10.1038/ng.3359)
- **AlphaFold DB**: Predicted NOD2 protein structure (UniProt: Q9HC29)
- **UniProt/InterPro**: Domain annotations, evolutionary conservation scores
- **dbNSFP**: Pre-computed functional prediction scores (CADD, REVEL, PolyPhen-2, SIFT)

## Methodology
1. **Data collection**: Aggregate all known NOD2 variants from ClinVar, gnomAD, and IBDGC. Annotate with allele frequency, clinical significance, domain location.
2. **Feature engineering**: For each variant compute — evolutionary conservation (PhyloP, GERP++), structural features (distance to active site, solvent accessibility from AlphaFold), domain impact (LRR, NACHT, CARD domains), existing predictor scores. **VarMeter2 structure-based features** (Comp Struct Biotech J 2025, PMC11952791): normalized solvent-accessible surface area (nSASA), mutation energy, and pLDDT score computed from AlphaFold structure (Q9HC29). VarMeter2 achieved 82% accuracy using only these physical parameters — orthogonal to conservation-based scores (CADD, REVEL, AlphaMissense) and captures physicochemical mechanism of pathogenicity rather than evolutionary signal. **Girdin binding domain features** (Ghosh et al., JCI Oct 2025, doi:10.1172/JCI190851): Girdin identified as critical NOD2 binding partner controlling macrophage inflammatory/reparative balance. L1007fs (most common CD variant) deletes the girdin binding domain — mechanistic explanation for LOF in tissue repair context. For each variant, compute: (1) distance to girdin binding interface residues (from AF3 complex model or published structure), (2) binary feature: disrupts girdin binding domain (yes/no), (3) predicted effect on NOD2-girdin binding affinity. Girdin binding disruption is a high-priority pathogenicity feature — variants in this interface should be predicted as LOF for the inflammation-repair axis even if they retain muramyl dipeptide sensing. **Exploratory: AlphaFold3 complex modeling** — model NOD2-RIPK2 and NOD2-girdin complexes with AF3 to assess interface variant effects. NOD2-girdin now elevated to medium priority given mechanistic importance. No NOD2-specific AF3 study exists, making this novel. **NF-kB/RANK pathway crosstalk feature** (Sci Rep 2025, doi:10.1038/s41598-025-28395-7): Genomic analysis identified TNFSF11/RANKL as CD-linked pathway with denosumab as candidate. RANK signaling intersects with NF-kB (which NOD2 activates via RIPK2). For each variant, compute: (1) predicted effect on NF-kB activation capacity (using known reporter assay data where available), (2) proximity to RIPK2 interaction interface (CARD-CARD domain). Variants impairing NF-kB activation may have compounded effects when RANK/RANKL signaling is disrupted in the same tissue — include as exploratory interaction feature.
3. **Training set construction**: Construct a **functional spectrum** training set (GOF ↔ neutral ↔ LOF) rather than binary pathogenic/benign:
   - *Loss-of-function (LOF)*: ClinVar pathogenic CD-associated variants (R702W, G908R, L1007fsinsC and others) — primarily in LRR domain, impairing muramyl dipeptide sensing.
   - *Gain-of-function (GOF)*: **Blau syndrome variants** R334Q, R334W, L469F (Current Rheumatology Reports 2026, doi:10.1007/s11926-026-01212-4) — dominant GOF mutations in NACHT domain causing NF-κB hyperactivation → granulomatous arthritis, uveitis, dermatitis. Also include early-onset sarcoidosis GOF variants. Label as "gain_of_function" with disease phenotype "Blau syndrome" to distinguish from CD-associated LOF. Combined with RIPK2-activating GOF data (task #230).
   - *SURF functional characterization data* (Front Immunol 2025, PMID 41357180): 9 NOD2 variants functionally characterized with NF-κB activity and cytokine measurements (IL-2, TNF-α, IL-6, IL-8) in pediatric SURF patients. Enhanced/GOF: R702W, G908R, P268S/V955I compound, R702W/P268S compound. Reduced/LOF: L682F (NACHT domain), L1007fs, R587C. Moderate: A1000T, P268S alone. **Critical insight**: R702W and G908R show GOF in autoinflammation context but LOF in microbial sensing — context-dependent functional effects that the model must capture. L682F is notable: NACHT domain (near Blau GOF variants) yet shows LOF — a discriminating training example. Compound heterozygote data provides variant interaction examples. Combined with Blau variants, ~15 NOD2 variants now have functional measurements across the GOF-LOF spectrum.
   - *Benign/neutral*: ClinVar benign variants and common gnomAD variants with no disease association.
   - This three-class approach provides domain-specific training signal — Blau GOF variants cluster in NACHT domain (structurally distinct from CD-associated LRR variants), enabling the model to learn domain-dependent functional consequences. Apply stringent filtering to avoid circular predictions.
4. **Model development**: Train ensemble classifier (gradient boosting + logistic regression) to predict pathogenicity. Use nested cross-validation to avoid overfitting.
5. **Validation**: Hold-out validation against recently reclassified ClinVar variants. Compare performance to individual predictors (CADD, REVEL alone). **Functional readout benchmark**: Use the 53-gene macrophage polarization signature (Ghosh et al., JCI 2025) that discriminates inflammatory (M1) from reparative macrophages as an orthogonal validation metric — variants predicted as LOF for girdin binding should correlate with M1-skewed macrophage signatures in available expression datasets.
6. **VUS classification**: Apply model to all NOD2 VUS and rank by predicted pathogenicity.

## Expected Outputs
- Trained NOD2 variant **functional spectrum** classifier (GOF / neutral / LOF) with performance metrics (AUC, sensitivity, specificity per class)
- Ranked list of NOD2 VUS with predicted functional impact scores and directional classification (GOF vs LOF)
- Feature importance analysis showing which structural/evolutionary features drive pathogenicity and functional direction
- Visualization of variants mapped onto NOD2 3D structure colored by predicted functional class (GOF=red, neutral=gray, LOF=blue), highlighting NACHT vs LRR domain patterns

## Success Criteria
- Model AUC > 0.90 on held-out ClinVar variants
- At least 50 VUS classified with high confidence (probability > 0.8 or < 0.2)
- Novel structural insights into which NOD2 domains are most sensitive to variation

## Labels
genomic, immunology, biomarker, novel-finding, high-priority
