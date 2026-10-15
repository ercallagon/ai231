"""VCM audio feature extraction: waveform -> log-mel spectrogram.

Produces the (1, n_mels, T) tensor the model expects. With the locked config
(16 kHz, 2.0 s window, 80-mel, hop 160) this yields (1, 80, 198).
Target frames = 1 + (window_samples - n_fft) // hop_length = 1 + 31600//160 = 198.

Implemented in NumPy/scipy so it runs identically in training (CPU/GPU host)
and on the Pi (no torch required for the feature path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Defaults mirror configs/vcm.yaml `audio:`. Overridable per call.
DEFAULTS = dict(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80,
    window_seconds=2.0,
    f_min=20.0,
    f_max=8000.0,
)


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = DEFAULTS["sample_rate"]
    n_fft: int = DEFAULTS["n_fft"]
    hop_length: int = DEFAULTS["hop_length"]
    n_mels: int = DEFAULTS["n_mels"]
    window_seconds: float = DEFAULTS["window_seconds"]
    f_min: float = DEFAULTS["f_min"]
    f_max: float = DEFAULTS["f_max"]

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.sample_rate))

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(**{k: d[k] for k in d if k in DEFAULTS})


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int,
                    f_min: float, f_max: float) -> np.ndarray:
    """Triangular mel filterbank (scipy-free, matches torchaudio closely)."""
    import scipy.signal as sig

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low, high = hz2mel(f_min), hz2mel(f_max)
    mel_pts = np.linspace(low, high, n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)  # n_fft//2+1 bins, matches rfft

    fb = np.zeros((n_mels, len(bins)))  # width == STFT rfft bins (n_fft//2+1)
    for i in range(1, n_mels + 1):
        l, c, r = hz_pts[i - 1], hz_pts[i], hz_pts[i + 1]
        for j in range(len(bins)):
            if bins[j] >= l and bins[j] <= c:
                fb[i - 1, j] = (bins[j] - l) / (c - l) if c > l else 0.0
            elif bins[j] > c and bins[j] <= r:
                fb[i - 1, j] = (r - bins[j]) / (r - c) if r > c else 0.0
    # normalize columns
    col_max = fb.max(axis=0, keepdims=True)
    fb = np.where(col_max > 0, fb / np.maximum(col_max, 1e-9), fb)
    return fb


class MelFeatures:
    """Waveform -> log-mel spectrogram with center-crop/pad to a fixed width."""

    def __init__(self, cfg: Optional[FeatureConfig] = None):
        self.cfg = cfg or FeatureConfig()
        self._fb = _mel_filterbank(
            self.cfg.sample_rate, self.cfg.n_fft, self.cfg.n_mels,
            self.cfg.f_min, self.cfg.f_max,
        )
        self._target_frames = self._expected_frames()

    def _expected_frames(self) -> int:
        return 1 + (self.cfg.window_samples - self.cfg.n_fft) // self.cfg.hop_length

    def _stft_power(self, wav: np.ndarray) -> np.ndarray:
        n_frames = 1 + max(0, (len(wav) - self.cfg.n_fft) // self.cfg.hop_length)
        window = np.hanning(self.cfg.n_fft)
        frames = np.zeros((n_frames, self.cfg.n_fft))
        for i in range(n_frames):
            start = i * self.cfg.hop_length
            seg = wav[start:start + self.cfg.n_fft]
            if len(seg) < self.cfg.n_fft:
                seg = np.pad(seg, (0, self.cfg.n_fft - len(seg)))
            frames[i] = seg * window
        spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        return spec

    def transform(self, wav: np.ndarray) -> np.ndarray:
        """Return (1, n_mels, T) float32, center-cropped/padded to target width."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.ndim != 1:
            wav = wav.mean(axis=-1)
        # normalize to [-1, 1] if peak is larger
        peak = np.max(np.abs(wav))
        if peak > 1.0:
            wav = wav / peak

        spec = self._stft_power(wav)
        mel = self._fb @ spec.T
        log_mel = np.log(mel + 1e-5)

        # center crop / pad to target frame count
        t = log_mel.shape[1]
        if t >= self._target_frames:
            start = (t - self._target_frames) // 2
            log_mel = log_mel[:, start:start + self._target_frames]
        else:
            pad = self._target_frames - t
            log_mel = np.pad(log_mel, ((0, 0), (pad // 2, pad - pad // 2)), mode="edge")

        return log_mel[np.newaxis, :, :].astype(np.float32)
