from __future__ import annotations

import numpy as np


DEFAULT_BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 15.0), (15.0, 40.0))


def _bandpower_features(x: np.ndarray, fs: int, bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    feats = []
    total = power.sum(axis=-1, keepdims=True) + 1e-8
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        feats.append((power[..., mask].sum(axis=-1, keepdims=True) / total).squeeze(-1))
    return np.concatenate(feats, axis=1)


def extract_global_features(x: np.ndarray, fs: int) -> np.ndarray:
    """Extract lightweight, reproducible features for classical baselines.

    Input shape: (n_records, n_leads, n_samples).
    Output shape: (n_records, n_features).
    """
    mean = x.mean(axis=-1)
    std = x.std(axis=-1)
    minimum = x.min(axis=-1)
    maximum = x.max(axis=-1)
    median = np.median(x, axis=-1)
    q25 = np.percentile(x, 25, axis=-1)
    q75 = np.percentile(x, 75, axis=-1)
    iqr = q75 - q25
    rms = np.sqrt(np.mean(x**2, axis=-1))
    ptp = maximum - minimum
    derivative_energy = np.sqrt(np.mean(np.diff(x, axis=-1) ** 2, axis=-1))
    bandpower = _bandpower_features(x, fs)
    per_lead = np.concatenate([mean, std, minimum, maximum, median, iqr, rms, ptp, derivative_energy], axis=1)
    return np.concatenate([per_lead, bandpower], axis=1).astype(np.float32)

