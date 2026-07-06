from __future__ import annotations

import numpy as np
from scipy import signal


def stft_multilead_tensor(
    x: np.ndarray,
    fs: int,
    nperseg_seconds: float = 1.0,
    step_seconds: float = 0.1,
    max_freq: float | None = None,
    include_phase: bool = True,
) -> np.ndarray:
    """Convert ECG waveform to a thesis-inspired STFT image tensor.

    Input shape: (n_records, n_leads, n_samples).
    Output shape: (n_records, channels, freq_bins, time_bins), where channels are
    lead-wise amplitude maps plus optional lead-wise phase maps.
    """
    nperseg = max(8, int(round(fs * nperseg_seconds)))
    step = max(1, int(round(fs * step_seconds)))
    noverlap = max(0, nperseg - step)
    out = []
    for record in x:
        lead_maps = []
        for lead in record:
            freqs, _, zxx = signal.stft(
                lead,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nperseg,
                boundary=None,
                padded=False,
            )
            if max_freq is not None:
                keep = freqs <= max_freq
                zxx = zxx[keep]
            amp = np.log1p(np.abs(zxx)).astype(np.float32)
            lead_maps.append(amp)
        if include_phase:
            for lead in record:
                freqs, _, zxx = signal.stft(
                    lead,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nperseg,
                    boundary=None,
                    padded=False,
                )
                if max_freq is not None:
                    keep = freqs <= max_freq
                    zxx = zxx[keep]
                phase = np.angle(zxx).astype(np.float32)
                lead_maps.append(phase)
        out.append(np.stack(lead_maps, axis=0))
    images = np.stack(out, axis=0).astype(np.float32)
    mean = images.mean(axis=(2, 3), keepdims=True)
    std = images.std(axis=(2, 3), keepdims=True)
    return (images - mean) / np.maximum(std, 1e-6)

