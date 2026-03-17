# Voice Biomarkers for Early Parkinson's Disease Detection

## Objective
Develop a phone-based voice analysis system that detects early Parkinson's disease from short speech recordings, combining traditional acoustic biomarkers with deep spectrogram analysis.

## Background
Parkinson's disease (PD) affects 10M+ people worldwide with an average diagnostic delay of 1-2 years from symptom onset. Early diagnosis enables earlier intervention (exercise programs, medication timing) that improves quality of life. Voice and speech changes are among the earliest motor symptoms — hypophonia (reduced volume), monotone speech, imprecise articulation, and vocal tremor — often appearing years before clinical diagnosis.

Current diagnostic pathway requires neurological examination by a specialist, creating access barriers in rural areas and low-income countries. A screening tool that runs on any smartphone could identify at-risk individuals for specialist referral.

Key gaps:
- Most voice-PD studies use small datasets (<200 subjects) and controlled recording conditions
- Limited work on combining classical acoustic features (jitter, shimmer, HNR) with deep learning on raw audio/spectrograms
- Phone-quality audio introduces noise that degrades acoustic feature extraction — needs explicit robustness
- Few studies evaluate generalization across languages, accents, and phone hardware

## Data Sources
- **mPower Study (Sage Bionetworks / Synapse)** — ~16,000 participants, smartphone voice recordings (sustained vowel phonation + short walk audio). Largest mobile PD dataset. Self-reported PD diagnosis and medication status.
- **UCI Parkinson's Speech Dataset** — 1,040 recordings from 40 subjects (20 PD, 20 healthy). Multiple phonation types: sustained vowels, words, short sentences.
- **UCI Parkinson's Telemonitoring Dataset** — 5,875 recordings from 42 early-stage PD patients. UPDRS scores for regression targets. 16 voice measures per recording.
- **Italian Parkinson's Voice and Speech** — 65 subjects, multiple speech tasks, professionally recorded.
- **PC-GITA (Colombian Spanish)** — 100 subjects (50 PD, 50 control), multiple speech tasks in Spanish. Critical for cross-language evaluation.

## Methodology
1. **Data acquisition and preprocessing**: Download and unify datasets. Standardize audio format (16kHz mono WAV). Apply noise profiling to characterize recording quality differences between datasets. Segment sustained vowel portions from longer recordings.
2. **Classical acoustic feature extraction**: Extract established voice biomarkers using Parselmouth/Praat:
   - Jitter (pitch perturbation), shimmer (amplitude perturbation)
   - Harmonics-to-noise ratio (HNR), noise-to-harmonics ratio (NHR)
   - Fundamental frequency (F0) statistics: mean, std, range
   - Formant frequencies (F1-F4) and bandwidths
   - MFCC (Mel-frequency cepstral coefficients) — 13 coefficients + deltas
   - Speech rate, pause patterns (from connected speech tasks)
3. **Deep spectrogram analysis**: Generate mel-spectrograms from raw audio. Train lightweight CNN (MobileNetV2-based) on spectrogram images. Also evaluate 1D-CNN on raw waveform and wav2vec 2.0 embeddings (pre-trained on speech).
4. **Ensemble fusion**: Combine classical features (via gradient boosting) with deep features (via CNN) in a late-fusion ensemble. Evaluate early fusion (concatenated features) vs. late fusion (prediction averaging) vs. stacked generalization.
5. **Robustness engineering**: Add noise augmentation (background noise, microphone artifacts, compression artifacts) during training. Evaluate performance degradation across recording quality tiers. Test cross-dataset generalization (train on mPower, test on UCI and PC-GITA).
6. **Cross-language evaluation**: Train on English datasets (mPower, UCI), evaluate on Spanish (PC-GITA) and Italian datasets. Assess which features are language-invariant.
7. **Mobile deployment prototype**: Convert best model to TFLite. Design 30-second voice task protocol (sustained "ahhh" + reading a standard sentence). Benchmark inference on mobile hardware.

## Expected Outputs
- Feature importance ranking of classical acoustic biomarkers for PD detection
- Trained ensemble model achieving high AUC for PD vs. healthy classification
- Cross-dataset and cross-language generalization analysis
- Noise robustness evaluation across recording quality levels
- Mobile-optimized model with defined voice task protocol
- Analysis of which voice features appear earliest in disease progression (using UPDRS scores from telemonitoring dataset)

## Success Criteria
- AUC >= 0.85 for PD detection on held-out mPower test set
- Cross-dataset AUC >= 0.80 when training on mPower and testing on UCI/PC-GITA
- Ensemble outperforms single-modality (classical-only or deep-only) by >= 3% AUC
- Model inference < 2 seconds on mobile hardware for a 30-second recording
- Identified at least 3 acoustic features significantly associated with early-stage PD (low UPDRS)

## Labels
novel-finding, ai-diagnostic, point-of-care, accessibility, screening
