# Confusion Matrices

Rows are generated from stored TP/FP/TN/FN metrics. Matrix orientation is actual class by predicted class.

| experiment_id | split | family | model | lead_mode | preprocess | n | positives | tn | fp | fn | tp | sensitivity | specificity | precision | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | train | classical | logistic_regression | wearable3_vdiff | raw_zscore | 16 | 8 | 8 | 0 | 0 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| E001 | val | classical | logistic_regression | wearable3_vdiff | raw_zscore | 2 | 1 | 1 | 0 | 0 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| E001 | test | classical | logistic_regression | wearable3_vdiff | raw_zscore | 2 | 1 | 1 | 0 | 0 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| E002 | train | classical | logistic_regression | all12 | raw_zscore | 16 | 8 | 8 | 0 | 0 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| E002 | val | classical | logistic_regression | all12 | raw_zscore | 2 | 1 | 1 | 0 | 0 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| E002 | test | classical | logistic_regression | all12 | raw_zscore | 2 | 1 | 0 | 1 | 0 | 1 | 1.0000 | 0.0000 | 0.5000 | 0.6667 |
