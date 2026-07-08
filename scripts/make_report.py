from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report.md from experiment metrics.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out", default="report.md")
    parser.add_argument("--skip-plots", action="store_true", help="Skip confusion-matrix PNG generation.")
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


def ok_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    if "test_auroc" in ok.columns:
        ok = ok.sort_values("test_auroc", ascending=False, na_position="last")
    return ok


def format_value(value: object, column: str) -> str:
    if pd.isna(value):
        return ""
    if column in {"n", "positives", "tp", "fp", "tn", "fn"}:
        return str(int(round(float(value))))
    if column.endswith(("_n", "_positives", "_tp", "_fp", "_tn", "_fn")):
        return str(int(round(float(value))))
    if column in {"parameters", "trainable_parameters", "epochs_run", "epochs_requested", "sampling_rate"}:
        return str(int(round(float(value))))
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return ""
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    available = [col for col in columns if col in df.columns]
    if df.empty or not available:
        return "_No completed training metrics yet._"
    view = df[available].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = [
        "| " + " | ".join(format_value(value, column) for column, value in row.items()) + " |"
        for row in view.to_dict(orient="records")
    ]
    return "\n".join([header, sep, *rows])


def results_columns(compact: bool = False) -> list[str]:
    common = [
        "family",
        "model",
        "lead_mode",
        "preprocess",
    ]
    if compact:
        return [
            *common,
            "training_time_seconds",
            "epochs_run",
            "device",
            "test_auroc",
            "test_average_precision",
            "test_sensitivity",
            "test_specificity",
            "test_f1",
            "test_tp",
            "test_fp",
            "test_tn",
            "test_fn",
        ]
    return [
        *common,
        "sampling_rate",
        "max_records",
        "status",
        "device",
        "epochs_requested",
        "epochs_run",
        "training_time_seconds",
        "preprocess_time_seconds",
        "input_transform_time_seconds",
        "parameters",
        "trainable_parameters",
        "train_n",
        "train_positives",
        "val_n",
        "val_positives",
        "test_n",
        "test_positives",
        "val_threshold",
        "test_auroc",
        "test_average_precision",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
        "test_sensitivity",
        "test_specificity",
        "test_f1",
        "test_tp",
        "test_fp",
        "test_tn",
        "test_fn",
        "val_auroc",
        "val_average_precision",
        "train_auroc",
        "train_average_precision",
    ]


def write_results_artifacts(ok: pd.DataFrame, reports_dir: Path) -> str:
    if ok.empty:
        return "_No completed training metrics yet._"
    columns = [col for col in results_columns(compact=False) if col in ok.columns]
    full = ok[columns].copy()
    full.to_csv(reports_dir / "all_model_results.csv", index=False)

    compact_columns = results_columns(compact=True)
    compact_table = markdown_table(ok, compact_columns)
    (reports_dir / "all_model_results.md").write_text(
        "# All Model Results\n\n"
        "Sorted by test AUROC descending. Counts are from the selected validation threshold applied to the test split.\n\n"
        f"{compact_table}\n",
        encoding="utf-8",
    )
    return compact_table


def confusion_matrix_rows(ok: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if ok.empty:
        return pd.DataFrame()
    metadata_cols = ["family", "model", "lead_mode", "preprocess", "sampling_rate", "max_records", "device"]
    for index, row in ok.reset_index(drop=True).iterrows():
        experiment_id = f"E{index + 1:03d}"
        base = {
            "experiment_id": experiment_id,
            **{col: row[col] for col in metadata_cols if col in row.index},
        }
        for split in ("train", "val", "test"):
            required = [f"{split}_tn", f"{split}_fp", f"{split}_fn", f"{split}_tp"]
            if not all(col in row.index and not pd.isna(row[col]) for col in required):
                continue
            split_row = {
                **base,
                "split": split,
                "n": row.get(f"{split}_n"),
                "positives": row.get(f"{split}_positives"),
                "threshold": row.get(f"{split}_threshold"),
                "tn": row[f"{split}_tn"],
                "fp": row[f"{split}_fp"],
                "fn": row[f"{split}_fn"],
                "tp": row[f"{split}_tp"],
                "sensitivity": row.get(f"{split}_sensitivity"),
                "specificity": row.get(f"{split}_specificity"),
                "precision": row.get(f"{split}_precision"),
                "recall": row.get(f"{split}_recall"),
                "f1": row.get(f"{split}_f1"),
                "auroc": row.get(f"{split}_auroc"),
                "average_precision": row.get(f"{split}_average_precision"),
            }
            rows.append(split_row)
    return pd.DataFrame(rows)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "experiment"


def write_confusion_matrix_artifacts(cm: pd.DataFrame, reports_dir: Path, skip_plots: bool = False) -> str:
    if cm.empty:
        return "_No confusion-matrix counts are available yet._"
    cm.to_csv(reports_dir / "confusion_matrices.csv", index=False)
    columns = [
        "experiment_id",
        "split",
        "family",
        "model",
        "lead_mode",
        "preprocess",
        "n",
        "positives",
        "tn",
        "fp",
        "fn",
        "tp",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
    ]
    full_table = markdown_table(cm, columns)
    (reports_dir / "confusion_matrices.md").write_text(
        "# Confusion Matrices\n\n"
        "Rows are generated from stored TP/FP/TN/FN metrics. Matrix orientation is actual class by predicted class.\n\n"
        f"{full_table}\n",
        encoding="utf-8",
    )

    test_cm = cm[cm["split"].eq("test")].copy()
    test_table = markdown_table(test_cm, [col for col in columns if col != "split"])
    if not skip_plots:
        write_confusion_matrix_plots(test_cm, reports_dir / "confusion_matrices")
    return test_table


def write_confusion_matrix_plots(test_cm: pd.DataFrame, out_dir: Path) -> None:
    if test_cm.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped confusion-matrix PNG generation.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in test_cm.iterrows():
        matrix = [
            [int(round(float(row["tn"]))), int(round(float(row["fp"])))],
            [int(round(float(row["fn"]))), int(round(float(row["tp"])))],
        ]
        title = f"{row['experiment_id']} {row['model']} | {row['lead_mode']} | {row['preprocess']}"
        filename = slugify(title) + ".png"

        fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=160)
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], labels=["No MI", "MI"])
        ax.set_yticks([0, 1], labels=["No MI", "MI"])
        max_value = max(max(row_values) for row_values in matrix) or 1
        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i][j] > max_value * 0.55 else "black"
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color=color, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(reports_dir)
    ok = ok_metrics(df)
    table = write_results_artifacts(ok, reports_dir)
    cm = confusion_matrix_rows(ok)
    test_confusion_table = write_confusion_matrix_artifacts(cm, reports_dir, skip_plots=args.skip_plots)
    best = ""
    if not ok.empty and "test_auroc" in ok.columns:
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

## Test Confusion Matrices

Rows use the validation-selected threshold and fold-10 test split.

{test_confusion_table}

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
"""
    (PROJECT_ROOT / args.out).write_text(content, encoding="utf-8")
    print(f"Wrote {PROJECT_ROOT / args.out}")


if __name__ == "__main__":
    main()
