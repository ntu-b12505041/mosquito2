from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_ecg.data import build_dataset, labels_summary, write_dataset_card
from mi_ecg.deep import make_deep_model
from mi_ecg.images import stft_multilead_tensor
from mi_ecg.io import write_metrics
from mi_ecg.metrics import choose_threshold_by_youden, evaluate_binary
from mi_ecg.preprocess import apply_lead_mode, apply_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PTB-XL MI deep learning baselines.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--sampling-rate", type=int, default=None, choices=[100, 500])
    parser.add_argument("--lead-mode", default=None, choices=["all12", "limb6", "precordial6", "wearable3_vdiff"])
    parser.add_argument("--max-records", type=int, default=None, help="Use a stratified subset for smoke tests.")
    parser.add_argument("--preprocess", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_loader(torch, x, y, batch_size: int, shuffle: bool):
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(torch, model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("torch is required. Install with `pip install -r requirements.txt`.") from exc

    cfg = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    data_dir = args.data_dir or cfg["data"]["data_dir"]
    sampling_rate = args.sampling_rate or int(cfg["data"]["sampling_rate"])
    lead_mode = args.lead_mode or cfg["data"].get("lead_mode", "all12")
    max_records = args.max_records if args.max_records is not None else cfg["data"].get("max_records")
    recipes = args.preprocess or cfg["preprocessing"]
    model_names = args.models or cfg["deep_models"]
    epochs = args.epochs or int(cfg["training"]["epochs"])
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    learning_rate = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    patience = int(cfg["training"]["patience"])
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []

    for recipe in recipes:
        x_processed, fs = apply_preprocessing(x, recipe, sampling_rate)
        for model_name in model_names:
            row: dict[str, object] = {
                "family": "deep",
                "model": model_name,
                "preprocess": recipe,
                "lead_mode": lead_mode,
                "sampling_rate": fs,
                "max_records": max_records,
                "device": str(device),
                "epochs_requested": epochs,
            }
            y_train = y[masks["train"]]
            y_val = y[masks["val"]]
            y_test = y[masks["test"]]
            if len(set(y_train.tolist())) < 2 or len(set(y_val.tolist())) < 2 or len(set(y_test.tolist())) < 2:
                row["status"] = "skipped_split_has_single_class"
                rows.append(row)
                continue

            if model_name == "spectrogram2d":
                model_input = stft_multilead_tensor(x_processed, fs, max_freq=min(119.0, fs / 2.0))
            else:
                model_input = x_processed

            train_loader = make_loader(torch, model_input[masks["train"]], y_train, batch_size, shuffle=True)
            val_loader = make_loader(torch, model_input[masks["val"]], y_val, batch_size, shuffle=False)
            test_loader = make_loader(torch, model_input[masks["test"]], y_test, batch_size, shuffle=False)
            full_train_loader = make_loader(torch, model_input[masks["train"]], y_train, batch_size, shuffle=False)

            model = make_deep_model(model_name, in_channels=model_input.shape[1]).to(device)
            classes = np.array([0, 1])
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            pos_weight = torch.tensor(weights[1] / weights[0], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_auc = -np.inf
            best_state = None
            bad_epochs = 0
            history = []
            training_start = time.perf_counter()
            for epoch in range(1, epochs + 1):
                epoch_start = time.perf_counter()
                model.train()
                losses = []
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))

                val_scores = predict(torch, model, val_loader, device)
                threshold = choose_threshold_by_youden(y_val, val_scores)
                val_metrics = evaluate_binary(y_val, val_scores, threshold)
                history.append(
                    {
                        "epoch": epoch,
                        "loss": float(np.mean(losses)),
                        "epoch_time_seconds": round(time.perf_counter() - epoch_start, 3),
                        **val_metrics,
                    }
                )
                val_auc = val_metrics["auroc"]
                print(f"{model_name}/{recipe} epoch={epoch} loss={np.mean(losses):.4f} val_auc={val_auc:.4f}")
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            val_scores = predict(torch, model, val_loader, device)
            threshold = choose_threshold_by_youden(y_val, val_scores)
            train_scores = predict(torch, model, full_train_loader, device)
            test_scores = predict(torch, model, test_loader, device)
            row.update(
                {
                    "status": "ok",
                    "threshold_source": "validation_youden",
                    "epochs_run": len(history),
                    "training_time_seconds": round(time.perf_counter() - training_start, 3),
                }
            )
            row.update(evaluate_binary(y_train, train_scores, threshold, prefix="train"))
            row.update(evaluate_binary(y_val, val_scores, threshold, prefix="val"))
            row.update(evaluate_binary(y_test, test_scores, threshold, prefix="test"))
            rows.append(row)

            model_path = out_dir / f"{model_name}_{recipe}.pt"
            torch.save({"model_state_dict": model.state_dict(), "history": history, "row": row}, model_path)
            print(
                f"{model_name} / {recipe}: "
                f"val_auc={row['val_auroc']:.4f} test_auc={row['test_auroc']:.4f} "
                f"test_ap={row['test_average_precision']:.4f}"
            )

    df = write_metrics(rows, out_dir / "metrics_deep.csv", out_dir / "metrics_deep.json")
    print(f"Wrote {out_dir / 'metrics_deep.csv'}")


if __name__ == "__main__":
    main()
