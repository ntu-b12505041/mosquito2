# MI ECG Model Report

Generated: 2026-07-08T22:54:19

## Architecture Background

This project trains a myocardial infarction (MI) detector from 12-lead ECG waveforms. The first reproducible public dataset is PTB-XL v1.0.3 because it provides 10-second 12-lead ECG waveforms, SCP-ECG labels, patient-stratified folds, and an MI diagnostic superclass.

Literature-driven design points:

- PTB-XL benchmark work supports ResNet- and Inception-based ECG time-series models as strong baselines.
- ECG12Net supports deep 12-lead CNN feature learning, although PMID 32134388 targets dyskalemia rather than MI.
- ProtoECGNet motivates future prototype-based interpretability.
- ECGFounder motivates future pretrained ECG encoder fine-tuning.
- The 2023 NYCU thesis on wearable 3-lead MI detection motivates a second experiment track using differential chest leads `(V1-V2, V3-V4, V5-V6)` plus STFT image modeling.

## Problem

- Input: 12-lead ECG waveform, shape `(12, time)`.
- Optional wearable setting: 3 differential chest leads, shape `(3, time)`.
- Output: probability of MI.
- Positive label: any PTB-XL SCP code mapped to `diagnostic_class == MI`.
- Negative label: no MI diagnostic-class SCP code.
- Split: PTB-XL folds 1-8 train, fold 9 validation, fold 10 test.

## Data

Primary source: PTB-XL v1.0.3 from PhysioNet.

Used files:

- `ptbxl_database.csv`
- `scp_statements.csv`
- `records100/` WFDB waveform files by default

## Method

1. Parse PTB-XL metadata and SCP labels.
2. Generate binary MI target.
3. Load waveform records.
4. Compare preprocessing recipes.
5. Train classical and deep baselines.
6. Select threshold on validation set using Youden's J statistic.
7. Evaluate once on test fold.

## Models

- Logistic regression on global waveform features.
- Random forest on global waveform features.
- Histogram gradient boosting on global waveform features.
- 1D ResNet on raw waveforms.
- InceptionTime-like 1D CNN on raw waveforms.
- Thesis-inspired STFT spectrogram 2D CNN for limited-lead ECG.

## Results

Best completed run: `logistic_regression` with `raw_zscore` (test AUROC=1.0000, test AUPRC=1.0000).

| family | model | lead_mode | preprocess | test_auroc | test_average_precision | test_sensitivity | test_specificity | test_f1 | test_tp | test_fp | test_tn | test_fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical | logistic_regression | wearable3_vdiff | raw_zscore | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 | 0 | 1 | 0 |
| classical | logistic_regression | all12 | raw_zscore | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.6667 | 1 | 1 | 0 | 0 |

## Test Confusion Matrices

Rows use the validation-selected threshold and fold-10 test split.

| experiment_id | family | model | lead_mode | preprocess | n | positives | tn | fp | fn | tp | sensitivity | specificity | precision | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | classical | logistic_regression | wearable3_vdiff | raw_zscore | 2 | 1 | 1 | 0 | 0 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| E002 | classical | logistic_regression | all12 | raw_zscore | 2 | 1 | 0 | 1 | 0 | 1 | 1.0000 | 0.0000 | 0.5000 | 0.6667 |

## Generated Artifacts

- `reports/all_model_results.csv`: full sortable model result table.
- `reports/all_model_results.md`: clean Markdown result table.
- `reports/confusion_matrices.csv`: train/validation/test TP, FP, TN, FN table.
- `reports/confusion_matrices.md`: Markdown confusion-matrix table.
- `reports/confusion_matrices/*.png`: per-experiment test confusion-matrix images.

## Explanation

Rows produced with a small `max_records` value are smoke tests only. They prove the pipeline runs, but they are not valid clinical or benchmark performance estimates. Full model selection should use all PTB-XL records100 and report patient-safe fold-10 test metrics.

## Conclusion

Use the best completed full-data run by test AUROC, then inspect AUPRC, sensitivity, specificity, and calibration before considering deployment.

## Future Work

- Add calibration and confidence intervals.
- Add MIMIC-IV-ECG external validation if credentialed access is available.
- Add LUDB/ISP delineation pretraining for QRS offset/J-point/ST-segment features.
- Add two-stage MI then STEMI/NSTEMI training when reliable acute MI subtype labels are available.
- Add Grad-CAM or prototype explanations.
- Fine-tune public ECG foundation model weights if licensing permits.

## Reproduction

Run:

```bash
python scripts/run_classical_experiments.py --lead-mode all12 --preprocess raw_zscore bandpass_0.5_40_zscore --models logistic_regression random_forest hist_gradient_boosting
python scripts/run_classical_experiments.py --lead-mode wearable3_vdiff --preprocess raw_zscore bandpass_0.5_40_zscore --models logistic_regression random_forest hist_gradient_boosting
python scripts/run_deep_experiments.py --lead-mode all12 --epochs 20 --preprocess raw_zscore bandpass_0.5_40_zscore --models resnet1d inception1d
python scripts/run_deep_experiments.py --lead-mode wearable3_vdiff --epochs 20 --preprocess bandpass_0.5_40_zscore --models resnet1d inception1d spectrogram2d
python scripts/make_report.py
```
