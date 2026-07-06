from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report.md from experiment metrics.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out", default="report.md")
    return parser.parse_args()


def load_metrics(reports_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ["metrics_classical.csv", "metrics_deep.csv"]:
        path = reports_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def metrics_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No completed training metrics yet._"
    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:
        return "_No completed training metrics yet._"
    cols = [
        "family",
        "model",
        "training_time_seconds",
        "epochs_run",
        "device",
        "lead_mode",
        "preprocess",
        "test_auroc",
        "test_average_precision",
        "test_sensitivity",
        "test_specificity",
        "test_f1",
    ]
    available = [c for c in cols if c in ok.columns]
    ok = ok.sort_values("test_auroc", ascending=False, na_position="last")
    view = ok[available].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / args.reports_dir
    df = load_metrics(reports_dir)
    table = metrics_table(df)
    best = ""
    if not df.empty and "test_auroc" in df.columns:
        ok = df[df["status"].eq("ok")].sort_values("test_auroc", ascending=False)
        if not ok.empty:
            row = ok.iloc[0]
            best = (
                f"Best completed run: `{row['model']}` with `{row['preprocess']}` "
                f"(test AUROC={row['test_auroc']:.4f}, test AUPRC={row['test_average_precision']:.4f})."
            )

    content = f"""# MI ECG Model Report

Generated: {datetime.now().isoformat(timespec="seconds")}

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

{best or "No completed full run has been recorded yet."}

{table}

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
"""
    (PROJECT_ROOT / args.out).write_text(content, encoding="utf-8")
    print(f"Wrote {PROJECT_ROOT / args.out}")


if __name__ == "__main__":
    main()
