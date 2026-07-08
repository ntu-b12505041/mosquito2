#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PTBXL_DATA_DIR="${PTBXL_DATA_DIR:-data/ptbxl}"
OUT_DIR="${OUT_DIR:-reports}"
LOG_ROOT="${LOG_ROOT:-logs}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_ROOT}/full_gpu_${RUN_ID}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "Mi ECG MI full GPU experiment run"
echo "Project: $(pwd)"
echo "Data: ${PTBXL_DATA_DIR}"
echo "Out: ${OUT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Device: ${DEVICE}"
echo

python - <<'PY' | tee "${LOG_DIR}/00_torch_device.log"
try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}")

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
PY

if [[ "${DEVICE}" == "cuda" ]]; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("DEVICE=cuda was requested, but torch.cuda.is_available() is false.")
PY
fi

python scripts/check_local_ptbxl_data.py --data-dir "${PTBXL_DATA_DIR}" | tee "${LOG_DIR}/01_check_ptbxl_data.log"

run_job() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "Command: $*"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

# Full planned classical matrix on full PTB-XL records100.
# No --max-records is used anywhere in this file.
for model in logistic_regression random_forest hist_gradient_boosting; do
  for preprocess in raw_zscore bandpass_0.5_40_zscore bandpass_0.5_40_zscore_downsample50; do
    run_job "classical_all12_${model}_${preprocess}" \
      python scripts/run_classical_experiments.py \
        --data-dir "${PTBXL_DATA_DIR}" \
        --out-dir "${OUT_DIR}" \
        --sampling-rate 100 \
        --lead-mode all12 \
        --preprocess "${preprocess}" \
        --models "${model}" \
        --no-download
  done
done

for model in logistic_regression random_forest hist_gradient_boosting; do
  for preprocess in raw_zscore bandpass_0.5_40_zscore; do
    run_job "classical_wearable3_vdiff_${model}_${preprocess}" \
      python scripts/run_classical_experiments.py \
        --data-dir "${PTBXL_DATA_DIR}" \
        --out-dir "${OUT_DIR}" \
        --sampling-rate 100 \
        --lead-mode wearable3_vdiff \
        --preprocess "${preprocess}" \
        --models "${model}" \
        --no-download
  done
done

# Full planned deep matrix on GPU.
for model in resnet1d inception1d; do
  for preprocess in raw_zscore bandpass_0.5_40_zscore; do
    run_job "deep_all12_${model}_${preprocess}" \
      python scripts/run_deep_experiments.py \
        --data-dir "${PTBXL_DATA_DIR}" \
        --out-dir "${OUT_DIR}" \
        --sampling-rate 100 \
        --lead-mode all12 \
        --preprocess "${preprocess}" \
        --models "${model}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --no-download
  done
done

for model in resnet1d inception1d spectrogram2d; do
  run_job "deep_wearable3_vdiff_${model}_bandpass_0.5_40_zscore" \
    python scripts/run_deep_experiments.py \
      --data-dir "${PTBXL_DATA_DIR}" \
      --out-dir "${OUT_DIR}" \
      --sampling-rate 100 \
      --lead-mode wearable3_vdiff \
      --preprocess bandpass_0.5_40_zscore \
      --models "${model}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --no-download
done

run_job "make_report" python scripts/make_report.py --reports-dir "${OUT_DIR}" --out report.md

echo
echo "Full GPU experiment run finished."
echo "Metrics:"
echo "  ${OUT_DIR}/metrics_classical.csv"
echo "  ${OUT_DIR}/metrics_deep.csv"
echo "  ${OUT_DIR}/all_model_results.csv"
echo "  ${OUT_DIR}/confusion_matrices.csv"
echo "Report:"
echo "  report.md"
echo "  ${OUT_DIR}/all_model_results.md"
echo "  ${OUT_DIR}/confusion_matrices.md"
echo "  ${OUT_DIR}/confusion_matrices/*.png"
echo "Logs:"
echo "  ${LOG_DIR}"
