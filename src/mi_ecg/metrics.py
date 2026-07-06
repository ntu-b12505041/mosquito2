from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold_by_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 401)))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        score = sensitivity + specificity - 1
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    prefix: str = "",
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": float(len(y_true)),
        "positives": float(y_true.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
        out["average_precision"] = float("nan")
    if prefix:
        return {f"{prefix}_{k}": v for k, v in out.items()}
    return out


def flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for split, values in metrics.items():
        flat.update({f"{split}_{key}": value for key, value in values.items()})
    return flat

