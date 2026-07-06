from __future__ import annotations

import numpy as np
from scipy import signal


def zscore_per_record(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def robust_zscore_per_record(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    median = np.median(x, axis=-1, keepdims=True)
    q75 = np.percentile(x, 75, axis=-1, keepdims=True)
    q25 = np.percentile(x, 25, axis=-1, keepdims=True)
    iqr = q75 - q25
    return (x - median) / np.maximum(iqr / 1.349, eps)


def butter_bandpass(x: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    high = min(high, nyq - 1e-3)
    sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def downsample(x: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    if target_fs == original_fs:
        return x
    if original_fs % target_fs == 0:
        return x[..., :: original_fs // target_fs].astype(np.float32)
    target_len = int(round(x.shape[-1] * target_fs / original_fs))
    return signal.resample(x, target_len, axis=-1).astype(np.float32)


def apply_preprocessing(x: np.ndarray, name: str, fs: int) -> tuple[np.ndarray, int]:
    """Apply a named preprocessing recipe.

    Input shape is (n_records, n_leads, n_samples).
    """
    x = x.astype(np.float32, copy=False)
    current_fs = fs
    if name == "raw":
        return x, current_fs
    if name == "raw_zscore":
        return zscore_per_record(x).astype(np.float32), current_fs
    if name == "raw_robust_zscore":
        return robust_zscore_per_record(x).astype(np.float32), current_fs
    if name == "bandpass_0.5_40_zscore":
        return zscore_per_record(butter_bandpass(x, current_fs, 0.5, 40.0)).astype(np.float32), current_fs
    if name == "bandpass_0.5_40_robust_zscore":
        return robust_zscore_per_record(butter_bandpass(x, current_fs, 0.5, 40.0)).astype(np.float32), current_fs
    if name == "bandpass_0.5_40_zscore_downsample50":
        filtered = zscore_per_record(butter_bandpass(x, current_fs, 0.5, 40.0)).astype(np.float32)
        return downsample(filtered, current_fs, 50), 50
    raise ValueError(f"Unknown preprocessing recipe: {name}")


def lead_subset(x: np.ndarray, leads: list[int] | None = None) -> np.ndarray:
    if leads is None:
        return x
    return x[:, leads, :]


def apply_lead_mode(x: np.ndarray, mode: str) -> np.ndarray:
    """Select or derive ECG lead channels.

    PTB-XL/WFDB lead order is expected to be:
    I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
    """
    mode = mode.lower()
    if mode in {"all12", "12lead", "all"}:
        return x
    if mode == "limb6":
        return x[:, :6, :]
    if mode == "precordial6":
        return x[:, 6:12, :]
    if mode == "wearable3_vdiff":
        # Thesis-inspired differential chest leads:
        # Vdiff1 = V1 - V2, Vdiff2 = V3 - V4, Vdiff3 = V5 - V6.
        return np.stack(
            [
                x[:, 6, :] - x[:, 7, :],
                x[:, 8, :] - x[:, 9, :],
                x[:, 10, :] - x[:, 11, :],
            ],
            axis=1,
        ).astype(np.float32)
    raise ValueError(f"Unknown lead mode: {mode}")
