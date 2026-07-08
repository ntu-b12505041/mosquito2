# Full Deep Learning Runs on Local CPU

This page is for running the full deep-learning experiment matrix one job at a time on a local Windows CPU machine.

## First Step

Edit:

```text
LOCAL_CPU_SETTINGS.ps1
```

Replace:

```powershell
$PTBXL_DATA_DIR = "<<<PTBXL_DATA_DIR>>>"
```

with your local PTB-XL root or `records100` path.

Default full-run settings:

```powershell
$DEEP_MAX_RECORDS = 0
$DEEP_EPOCHS = 20
$DEEP_BATCH_SIZE = 32
```

`$DEEP_MAX_RECORDS = 0` means full dataset.

## Run Order

Run these files one by one:

```text
RUN_DEEP_CPU_01_ALL12_RESNET_RAW.bat
RUN_DEEP_CPU_02_ALL12_RESNET_BANDPASS.bat
RUN_DEEP_CPU_03_ALL12_INCEPTION_RAW.bat
RUN_DEEP_CPU_04_ALL12_INCEPTION_BANDPASS.bat
RUN_DEEP_CPU_05_WEARABLE3_RESNET_BANDPASS.bat
RUN_DEEP_CPU_06_WEARABLE3_INCEPTION_BANDPASS.bat
RUN_DEEP_CPU_07_WEARABLE3_SPECTROGRAM_BANDPASS.bat
```

Each `.bat` runs exactly one deep-learning experiment and then regenerates `report.md`.

## Experiment Matrix

| File | Lead mode | Preprocess | Model |
|---|---|---|---|
| `RUN_DEEP_CPU_01_ALL12_RESNET_RAW.bat` | `all12` | `raw_zscore` | `resnet1d` |
| `RUN_DEEP_CPU_02_ALL12_RESNET_BANDPASS.bat` | `all12` | `bandpass_0.5_40_zscore` | `resnet1d` |
| `RUN_DEEP_CPU_03_ALL12_INCEPTION_RAW.bat` | `all12` | `raw_zscore` | `inception1d` |
| `RUN_DEEP_CPU_04_ALL12_INCEPTION_BANDPASS.bat` | `all12` | `bandpass_0.5_40_zscore` | `inception1d` |
| `RUN_DEEP_CPU_05_WEARABLE3_RESNET_BANDPASS.bat` | `wearable3_vdiff` | `bandpass_0.5_40_zscore` | `resnet1d` |
| `RUN_DEEP_CPU_06_WEARABLE3_INCEPTION_BANDPASS.bat` | `wearable3_vdiff` | `bandpass_0.5_40_zscore` | `inception1d` |
| `RUN_DEEP_CPU_07_WEARABLE3_SPECTROGRAM_BANDPASS.bat` | `wearable3_vdiff` | `bandpass_0.5_40_zscore` | `spectrogram2d` |

## Outputs

After each run, results are appended or updated in:

```text
reports/metrics_deep.csv
reports/metrics_deep.json
report.md
```

Model checkpoints are saved under `reports/`, for example:

```text
reports/all12_resnet1d_raw_zscore.pt
reports/wearable3_vdiff_spectrogram2d_bandpass_0.5_40_zscore.pt
```

## Notes

- Full deep learning on CPU can take a very long time.
- If you already ran `RUN_LOCAL_CPU_DEEP_TRY.bat`, that was only a small trial by default. It does not replace the full run unless you edited it to full dataset and 20 epochs.
- The spectrogram run can spend noticeable time building the STFT tensor before the first epoch starts.
- `metrics_deep.csv` includes `training_time_seconds`, `preprocess_time_seconds`, `input_transform_time_seconds`, `parameters`, and `trainable_parameters`.
