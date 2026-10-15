"""Training-only audio augmentation for the VCM.

Applied ONLY during training (never at eval/inference) to improve robustness to
unseen phrasings and noise. Kept NumPy-based so it has no hard dependency.

Augmentations (each independently toggleable):
    spec_aug      time-mask + freq-mask on the log-mel
    speed_perturb 0.9x / 1.0x / 1.1x (training-time resample)
    noise         additive white / pink noise at a SNR range
    gain          random linear gain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AugConfig:
    enabled: bool = True
    spec_aug_time_mask: int = 2      # number of time masks
    spec_aug_time_width: int = 20    # max width (frames)
    spec_aug_freq_mask: int = 2      # number of freq masks
    spec_aug_freq_width: int = 24    # max width (mels)
    speed_perturb: tuple = (0.9, 1.0, 1.1)
    noise_prob: float = 0.3
    snr_db: tuple = (10.0, 25.0)
    gain_db: tuple = (-6.0, 6.0)
    seed: Optional[int] = None


def augment_mel(mel: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply spec augmentation to a (1, n_mels, T) log-mel. Returns same shape."""
    x = mel.copy()
    if not cfg.enabled:
        return x
    x = _time_mask(x, cfg.spec_aug_time_mask, cfg.spec_aug_time_width, rng)
    x = _freq_mask(x, cfg.spec_aug_freq_mask, cfg.spec_aug_freq_width, rng)
    return x


def _time_mask(x, n_mask, max_w, rng):
    _, C, T = x.shape
    for _ in range(n_mask):
        w = int(rng.integers(0, max_w + 1))
        if w == 0:
            continue
        t0 = int(rng.integers(0, max(T - w, 1)))
        x[:, :, t0:t0 + w] = x[:, :, 0:1]  # fill with first-frame value
    return x


def _freq_mask(x, n_mask, max_w, rng):
    _, C, T = x.shape
    for _ in range(n_mask):
        w = int(rng.integers(0, max_w + 1))
        if w == 0:
            continue
        f0 = int(rng.integers(0, max(C - w, 1)))
        x[:, f0:f0 + w, :] = x[:, 0:1, :]
    return x


def perturb_speed(wav: np.ndarray, sr: int, factor: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Resample-free speed perturbation via linear-interp stride (approx)."""
    if factor == 1.0:
        return wav
    n_out = int(len(wav) / factor)
    idx = np.linspace(0, len(wav) - 1, n_out)
    idx_i = np.floor(idx).astype(int)
    idx_j = np.minimum(idx_i + 1, len(wav) - 1)
    frac = (idx - idx_i)[:, None]
    return (wav[idx_i] * (1 - frac) + wav[idx_j] * frac).astype(np.float32)


def add_noise(wav: np.ndarray, sr: int, cfg: AugConfig,
              rng: np.random.Generator) -> np.ndarray:
    if rng.random() > cfg.noise_prob:
        return wav
    snr = float(rng.uniform(*cfg.snr_db))
    sig_pow = np.mean(wav ** 2) + 1e-9
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    noise *= np.sqrt(sig_pow / (10 ** (snr / 10.0)))
    return (wav + noise).astype(np.float32)


def apply_gain(wav: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    db = float(rng.uniform(*cfg.gain_db))
    return (wav * (10 ** (db / 20.0))).astype(np.float32)
