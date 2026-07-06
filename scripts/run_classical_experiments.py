from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_ecg.classical import make_classical_model, predict_scores
from mi_ecg.data import build_dataset, labels_summary, write_dataset_card
from mi_ecg.features import extract_global_features
from mi_ecg.io import write_metrics
from mi_ecg.metrics import choose_threshold_by_youden, evaluate_binary
from mi_ecg.preprocess import apply_lead_mode, apply_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PTB-XL MI classical baselines.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--sampling-rate", type=int, default=None, choices=[100, 500])
    parser.add_argument("--lead-mode", default=None, choices=["all12", "limb6", "precordial6", "wearable3_vdiff"])
    parser.add_argument("--max-records", type=int, default=None, help="Use a stratified subset for smoke tests.")
    parser.add_argument("--preprocess", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_dir = args.data_dir or cfg["data"]["data_dir"]
    sampling_rate = args.sampling_rate or int(cfg["data"]["sampling_rate"])
    lead_mode = args.lead_mode or cfg["data"].get("lead_mode", "all12")
    max_records = args.max_records if args.max_records is not None else cfg["data"].get("max_records")
    recipes = args.preprocess or cfg["preprocessing"]
    model_names = args.models or cfg["classical_models"]
    seed = int(cfg.get("seed", 42))
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, records, masks = build_dataset(
        data_dir=data_dir,
        sampling_rate=sampling_rate,
        max_records=max_records,
        seed=seed,
        download=not args.no_download,
        overwrite=args.overwrite,
    )
    write_dataset_card(records, out_dir / "dataset_card.json")
    labels_summary(records).to_csv(out_dir / "label_summary.csv", index=False)
    x = apply_lead_mode(x, lead_mode)

    rows: list[dict[str, object]] = []
    for recipe in recipes:
        x_processed, fs = apply_preprocessing(x, recipe, sampling_rate)
        features = extract_global_features(x_processed, fs)
        for model_name in model_names:
            row: dict[str, object] = {
                "family": "classical",
                "model": model_name,
                "preprocess": recipe,
                "lead_mode": lead_mode,
                "sampling_rate": fs,
                "max_records": max_records,
            }
            y_train = y[masks["train"]]
            y_val = y[masks["val"]]
            y_test = y[masks["test"]]
            if len(set(y_train.tolist())) < 2 or len(set(y_val.tolist())) < 2 or len(set(y_test.tolist())) < 2:
                row["status"] = "skipped_split_has_single_class"
                rows.append(row)
                continue

            model = make_classical_model(model_name, seed=seed)
            training_start = time.perf_counter()
            model.fit(features[masks["train"]], y_train)
            row["training_time_seconds"] = round(time.perf_counter() - training_start, 3)
            val_scores = predict_scores(model, features[masks["val"]])
            threshold = choose_threshold_by_youden(y_val, val_scores)
            train_scores = predict_scores(model, features[masks["train"]])
            test_scores = predict_scores(model, features[masks["test"]])

            row.update({"status": "ok", "threshold_source": "validation_youden"})
            row.update(evaluate_binary(y_train, train_scores, threshold, prefix="train"))
            row.update(evaluate_binary(y_val, val_scores, threshold, prefix="val"))
            row.update(evaluate_binary(y_test, test_scores, threshold, prefix="test"))
            rows.append(row)
            print(
                f"{model_name} / {recipe}: "
                f"val_auc={row['val_auroc']:.4f} test_auc={row['test_auroc']:.4f} "
                f"test_ap={row['test_average_precision']:.4f}"
            )

    df = write_metrics(rows, out_dir / "metrics_classical.csv", out_dir / "metrics_classical.json")
    print(f"Wrote {out_dir / 'metrics_classical.csv'}")


if __name__ == "__main__":
    main()
