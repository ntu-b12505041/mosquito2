# MI ECG Model Training Project

This project trains reproducible myocardial infarction (MI) detection baselines from 12-lead ECG waveforms.

## Status

Implemented:

- PTB-XL metadata download.
- PTB-XL WFDB waveform loading.
- MI binary target generation from SCP diagnostic superclass.
- Preprocessing recipes.
- Classical baselines.
- Deep learning baselines.
- 12-lead and thesis-inspired wearable 3-differential-lead experiment modes.
- STFT spectrogram 2D CNN baseline for limited-lead ECG.
- Metrics and report generation.
- Colab notebook for GPU/full training.

Executed locally:

- 20-record PTB-XL smoke test with logistic regression.

Not executed locally:

- Full PTB-XL training.
- Deep learning training.

Reason: this machine has no detected NVIDIA GPU and did not have PyTorch installed before project setup.

## Project Layout

```text
mi_model/
├── configs/
│   └── experiments.yaml
├── data/
│   └── ptbxl/
├── notebooks/
│   └── mi_ptbxl_colab.ipynb
├── reports/
│   ├── dataset_card.json
│   ├── label_summary.csv
│   └── metrics_classical.csv
├── scripts/
│   ├── make_report.py
│   ├── run_classical_experiments.py
│   └── run_deep_experiments.py
├── src/
│   └── mi_ecg/
│       ├── classical.py
│       ├── data.py
│       ├── deep.py
│       ├── features.py
│       ├── metrics.py
│       └── preprocess.py
├── README.md
├── report.md
└── requirements.txt
```

## Data Source

Primary data is PTB-XL v1.0.3 from PhysioNet:

- Source: https://physionet.org/content/ptb-xl/1.0.3/
- Signal: 10-second 12-lead ECG.
- Sampling: 500 Hz raw and 100 Hz downsampled release.
- Labels: SCP-ECG statements.
- MI target: any SCP code where `diagnostic_class == MI`.
- Split: official `strat_fold`, using folds 1-8 train, 9 validation, 10 test.

The scripts use `records100/` by default for faster training and lower memory use.

Additional reference added:

- NYCU thesis, 2023: "Myocardial Infraction Detection and ECG Synthesizing Deep Learning Algorithm Based on 3-Lead ECG of Wearable Devices."
- Local copy: `../knowledge_base/ecg_mi_stemi/papers/nycu_thesis_mi_ecg.pdf`
- Key idea adopted here: simulate wearable 3 differential chest leads from PTB-XL as `V1-V2`, `V3-V4`, `V5-V6`, then compare waveform CNNs and STFT image CNNs.

## Install

```bash
cd mi_model
python -m pip install -r requirements.txt
```

For local CPU smoke tests, `torch` is not needed if you only run classical models. Full deep learning requires PyTorch and should be run on GPU.

## Local Smoke Test

```bash
cd mi_model
python scripts/run_classical_experiments.py \
  --max-records 20 \
  --lead-mode wearable3_vdiff \
  --preprocess raw_zscore \
  --models logistic_regression
python scripts/make_report.py
```

This validates the pipeline only. Do not use the smoke result as model performance.

## Full Classical Training

Recommended after downloading full `records100/`:

```bash
python scripts/run_classical_experiments.py \
  --lead-mode all12 \
  --preprocess raw_zscore bandpass_0.5_40_zscore bandpass_0.5_40_zscore_downsample50 \
  --models logistic_regression random_forest hist_gradient_boosting

python scripts/run_classical_experiments.py \
  --lead-mode wearable3_vdiff \
  --preprocess raw_zscore bandpass_0.5_40_zscore \
  --models logistic_regression random_forest hist_gradient_boosting
```

## Full Deep Training

Recommended on Colab GPU:

```bash
python scripts/run_deep_experiments.py \
  --lead-mode all12 \
  --epochs 20 \
  --batch-size 128 \
  --preprocess raw_zscore bandpass_0.5_40_zscore \
  --models resnet1d inception1d

python scripts/run_deep_experiments.py \
  --lead-mode wearable3_vdiff \
  --epochs 20 \
  --batch-size 128 \
  --preprocess bandpass_0.5_40_zscore \
  --models resnet1d inception1d spectrogram2d
```

## SSH GPU Full Run

For a remote SSH server with an NVIDIA GPU, use:

```bash
export PTBXL_DATA_DIR=/absolute/path/to/ptb-xl
export DEVICE=cuda
export EPOCHS=20
export BATCH_SIZE=128
bash RUN_FULL_GPU_EXPERIMENTS.sh
```

This runner uses full PTB-XL `records100/` and does not pass `--max-records`.
See `README_GPU_SSH.md` for tmux/nohup usage, CUDA PyTorch install notes, and the complete experiment matrix.

## Colab

Use:

```text
notebooks/mi_ptbxl_colab.ipynb
```

The notebook:

1. Mounts Google Drive.
2. Installs requirements.
3. Runs a 20-record smoke test.
4. Downloads full PTB-XL `records100/` with `wget`.
5. Runs full classical and deep experiments for `all12` and `wearable3_vdiff`.
6. Regenerates `report.md`.

I could not directly operate Chrome/Colab from this session because the requested Chrome plugin was not available in the current tool list and was not available for installation.

## Model Selection Strategy

Start with these candidates:

1. `resnet1d + raw_zscore`
2. `resnet1d + bandpass_0.5_40_zscore`
3. `inception1d + raw_zscore`
4. `spectrogram2d + wearable3_vdiff + bandpass_0.5_40_zscore`
5. `logistic_regression + raw_zscore`
6. `random_forest + bandpass_0.5_40_zscore`

Primary metric:

- Test AUROC

Secondary metrics:

- Test AUPRC
- Sensitivity
- Specificity
- Balanced accuracy
- F1

Threshold is selected on validation fold 9 using Youden's J statistic, then evaluated once on fold 10.

## Literature Notes

- PTB-XL benchmark: ResNet and Inception-style CNNs are strong ECG time-series baselines.
- ECG12Net: 12-lead deep CNNs can learn subtle morphology, even though that paper targets potassium abnormalities rather than MI.
- ProtoECGNet: motivates prototype-based interpretability for future clinical review.
- ECGFounder: motivates future pretrained ECG encoder fine-tuning.
- NYCU wearable 3-lead MI thesis: motivates differential chest-lead simulation, two-stage MI/STEMI/NSTEMI thinking, and STFT image modeling.

## Current Caveat

The current `report.md` includes only the local smoke-test result. Run the Colab notebook for the actual full MI model comparison.
