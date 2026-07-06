# MI ECG Local CPU Run Package

This package is for running the MI ECG pipeline on a local Windows machine with CPU only.

## What This Runs

The default one-click script runs full-dataset classical CPU experiments:

- `all12`
  - `raw_zscore`
  - `bandpass_0.5_40_zscore`
  - `bandpass_0.5_40_zscore_downsample50`
  - `logistic_regression`
  - `random_forest`
  - `hist_gradient_boosting`
- `wearable3_vdiff`
  - `raw_zscore`
  - `bandpass_0.5_40_zscore`
  - `logistic_regression`
  - `random_forest`
  - `hist_gradient_boosting`

Deep learning scripts are included, but they are not run by default because full deep training on CPU can take a very long time.

## Optional CPU Deep-Learning Trial

If you still want to test deep learning on local CPU, use:

```text
RUN_LOCAL_CPU_DEEP_TRY.bat
```

Before running it, edit:

```text
RUN_LOCAL_CPU_DEEP_TRY.ps1
```

and replace:

```powershell
$PTBXL_DATA_DIR = "<<<PTBXL_DATA_DIR>>>"
```

with your local PTB-XL path.

The default CPU deep trial intentionally uses a smaller setting:

```powershell
$MAX_RECORDS = 1000
$EPOCHS = 2
$BATCH_SIZE = 32
```

It runs:

```text
wearable3_vdiff + bandpass_0.5_40_zscore + resnet1d
```

To try the full dataset on CPU, set:

```powershell
$MAX_RECORDS = 0
```

Full deep learning on CPU can take a long time. Use this only if you want to test feasibility.

## Required Data Layout

Set the data path to either the PTB-XL root folder:

```text
<PTBXL_DATA_DIR>/
  ptbxl_database.csv
  scp_statements.csv
  records100/
    00000/
    01000/
    ...
```

or directly to:

```text
<PTBXL_DATA_DIR>/records100
```

The script accepts both.

## How To Run

1. Unzip this package.
2. Open `RUN_LOCAL_CPU.ps1`.
3. Replace this line:

```powershell
$PTBXL_DATA_DIR = "<<<PTBXL_DATA_DIR>>>"
```

with your local PTB-XL path, for example:

```powershell
$PTBXL_DATA_DIR = "D:\datasets\ptb-xl\1.0.3"
```

or:

```powershell
$PTBXL_DATA_DIR = "D:\datasets\ptb-xl\1.0.3\records100"
```

4. Double-click `RUN_LOCAL_CPU.bat`.

The script will:

1. Create `.venv`.
2. Install CPU requirements.
3. Check local `records100`.
4. Run full classical experiments on CPU.
5. Generate `report.md`.

## Outputs

After completion:

```text
reports/metrics_classical.csv
reports/metrics_classical.json
report.md
```

`metrics_classical.csv` includes:

```text
training_time_seconds
```

The optional deep CPU trial also creates:

```text
reports/metrics_deep.csv
reports/metrics_deep.json
reports/wearable3_vdiff_resnet1d_bandpass_0.5_40_zscore.pt
```

## Notes

- The zip does not include PTB-XL waveform data.
- The default script uses `--no-download`, so it expects waveform files to already exist locally.
- Metadata files `ptbxl_database.csv` and `scp_statements.csv` should be in the PTB-XL root folder. If missing, the underlying scripts may try to download them, but keeping them locally is recommended.
- Full classical CPU training can still take time because all ECG waveforms must be read from disk.
