# SSH GPU Full Run Guide

This guide is for running the full PTB-XL MI experiment matrix on a remote SSH server with an NVIDIA GPU.

The runner uses the full dataset. It does not pass `--max-records`.

## What You Need On The Server

The PTB-XL root folder should contain:

```text
ptbxl_database.csv
scp_statements.csv
records100/
```

If the two CSV files are missing but the server has internet, the training scripts may download those small metadata files automatically. The waveform folder `records100/` should already be present to avoid long downloads.

## Fast Setup

```bash
git clone https://github.com/ntu-b12505041/mosquito2.git
cd mosquito2

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your server already has the repo:

```bash
cd mosquito2
git pull
source .venv/bin/activate
```

## Install PyTorch With CUDA

First check the CUDA driver:

```bash
nvidia-smi
```

If normal `pip install -r requirements.txt` installs a CPU-only torch build, reinstall PyTorch from the official CUDA wheel index. Example for CUDA 12.1:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

## Point To Existing PTB-XL Data

Set `PTBXL_DATA_DIR` to the PTB-XL root, not only to an individual record folder.

```bash
export PTBXL_DATA_DIR=/absolute/path/to/ptb-xl
python scripts/check_local_ptbxl_data.py --data-dir "$PTBXL_DATA_DIR"
```

Passing the `records100` path to the checker is accepted, but the training scripts should use the PTB-XL root.

## Run Everything On GPU

Use `tmux` or `screen` so the run continues after SSH disconnects:

```bash
tmux new -s mi_gpu
```

Then run:

```bash
export PTBXL_DATA_DIR=/absolute/path/to/ptb-xl
export DEVICE=cuda
export EPOCHS=20
export BATCH_SIZE=128

bash RUN_FULL_GPU_EXPERIMENTS.sh
```

If GPU memory is not enough:

```bash
export BATCH_SIZE=64
bash RUN_FULL_GPU_EXPERIMENTS.sh
```

Detach from tmux:

```text
Ctrl-b then d
```

Reattach later:

```bash
tmux attach -t mi_gpu
```

## What This Runs

Classical full matrix:

- `all12`: `raw_zscore`, `bandpass_0.5_40_zscore`, `bandpass_0.5_40_zscore_downsample50`
- `wearable3_vdiff`: `raw_zscore`, `bandpass_0.5_40_zscore`
- Models: `logistic_regression`, `random_forest`, `hist_gradient_boosting`

Deep GPU matrix:

- `all12 + raw_zscore + resnet1d`
- `all12 + bandpass_0.5_40_zscore + resnet1d`
- `all12 + raw_zscore + inception1d`
- `all12 + bandpass_0.5_40_zscore + inception1d`
- `wearable3_vdiff + bandpass_0.5_40_zscore + resnet1d`
- `wearable3_vdiff + bandpass_0.5_40_zscore + inception1d`
- `wearable3_vdiff + bandpass_0.5_40_zscore + spectrogram2d`

Every job is called separately so completed jobs write metrics before the next job starts.

## Outputs

```text
reports/metrics_classical.csv
reports/metrics_classical.json
reports/metrics_deep.csv
reports/metrics_deep.json
reports/all_model_results.csv
reports/all_model_results.md
reports/confusion_matrices.csv
reports/confusion_matrices.md
reports/confusion_matrices/*.png
reports/*.pt
report.md
logs/full_gpu_YYYYMMDD_HHMMSS/
```

Model checkpoints are intentionally ignored by git.

## Direct Parameter Version

If you do not want to use the shell runner, the key parameters are:

```bash
python scripts/run_deep_experiments.py \
  --data-dir "$PTBXL_DATA_DIR" \
  --lead-mode all12 \
  --preprocess raw_zscore \
  --models resnet1d \
  --epochs 20 \
  --batch-size 128 \
  --device cuda \
  --no-download
```

Important points:

- Use `--device cuda` for GPU.
- Do not use `--max-records`.
- Keep `configs/experiments.yaml` as `max_records: null`.
- Use `--no-download` only when `records100/` is already complete.
