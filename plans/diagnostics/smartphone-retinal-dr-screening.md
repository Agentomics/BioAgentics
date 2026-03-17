# Smartphone-Based Retinal Screening for Diabetic Retinopathy

## Objective
Build a lightweight deep learning model for 5-class diabetic retinopathy grading that runs on smartphone hardware, enabling mass screening by community health workers in low-resource settings.

## Background
Diabetic retinopathy (DR) affects ~100M people globally and is a leading cause of preventable blindness. Early detection and treatment can prevent >90% of vision loss, yet screening requires ophthalmologists and expensive fundus cameras — resources scarce in rural and low-income settings. Existing deep learning models achieve high accuracy but are too large for mobile deployment. A model optimized for smartphone inference with a low-cost clip-on lens adapter could dramatically expand screening access.

Key gaps:
- Most published models target accuracy on benchmark datasets, not real-world deployment constraints (latency, memory, image quality variability)
- Calibration — clinical deployment needs reliable confidence estimates, not just classifications
- Demographic generalization — models trained on one population often underperform on others

## Data Sources
- **EyePACS / Kaggle Diabetic Retinopathy Detection** — ~88,000 high-resolution fundus images, 5-class DR severity labels (0-4). Largest public DR dataset.
- **APTOS 2019 Blindness Detection** — ~5,590 images from Aravind Eye Hospital (India), 5-class grading. Important for South Asian population representation.
- **IDRiD (Indian Diabetic Retinopathy Image Dataset)** — 516 images with pixel-level lesion annotations (microaneurysms, hemorrhages, hard/soft exudates). Useful for interpretability.
- **Messidor-2** — 1,748 images from French hospitals, graded by ophthalmologists. European population.
- **ODIR-5K** — 5,000 patients, both eyes, 8 disease labels including DR. Multi-label setting.

## Methodology
1. **Data preparation**: Standardize images across datasets (crop, resize, normalize). Apply quality filtering to remove ungradable images. Stratified train/val/test splits preserving dataset-of-origin for cross-population evaluation.
2. **Baseline model**: Train EfficientNet-B0 and MobileNetV3-Small on combined training set. Standard augmentation (flips, rotations, color jitter, Gaussian blur to simulate phone camera noise).
3. **Knowledge distillation**: Train a larger EfficientNet-B4 teacher, then distill into MobileNetV3-Small student. Compare distilled vs. direct training.
4. **Calibration**: Apply temperature scaling and Platt scaling post-hoc. Evaluate with ECE (Expected Calibration Error) and reliability diagrams. Clinical deployment requires "refer" threshold where sensitivity for referable DR (grade >= 2) exceeds 95%.
5. **Robustness evaluation**: Test on held-out datasets not seen during training. Evaluate performance stratified by image quality score, patient demographics (age, ethnicity where available), and camera type.
6. **Mobile optimization**: Convert best model to TFLite/ONNX. Benchmark inference latency on representative mobile hardware. Target <500ms per image.
7. **Interpretability**: Generate Grad-CAM heatmaps for lesion localization. Compare against IDRiD pixel-level annotations for sanity checking.

## Expected Outputs
- Trained MobileNetV3-Small model achieving >=90% sensitivity and >=85% specificity for referable DR, with <500ms mobile inference
- Calibration analysis showing reliable confidence estimates
- Cross-population performance report across EyePACS, APTOS, Messidor-2
- Grad-CAM visualizations demonstrating lesion-appropriate attention
- Model artifacts in TFLite/ONNX format ready for mobile integration

## Success Criteria
- Referable DR sensitivity >= 90% with specificity >= 85% on held-out test sets
- Cross-population AUC drop < 5% between training and external validation sets
- ECE < 0.05 after calibration
- Inference latency < 500ms on mobile-class hardware
- Clinically interpretable Grad-CAM maps that align with known lesion locations

## Labels
ai-diagnostic, imaging, point-of-care, accessibility, cost-reduction, screening
