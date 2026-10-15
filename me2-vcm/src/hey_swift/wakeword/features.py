"""Wake-word feature extraction: 16 kHz mono PCM -> (1, 80, 100) float32.

Faithful port of the historical exp15 pipeline
(`me2-wakeword-swift/scripts/features_np_swift.py`). This module is the SINGLE
SOURCE OF TRUTH for the DSP and is NumPy-only so training (PyTorch) and live
inference (ONNX Runtime, Pi) are bit-identical.

Pipeline (16 kHz mono PCM -> (1, 80, 100)):
    Hann STFT (n_fft=512, hop=160, center, float64) -> power
    -> 40-band triangular log-Mel (20 Hz - 8 kHz) -> log10 -> dB
    -> per-frame CMVN (mean/std over the mel axis)
    -> delta (centered 1st derivative, pad=2)
    -> concat [static, delta] -> (1, 80, 100)
    -> center-crop / center-pad to 100 frames

Input  : (n_samples,) float32, 16 kHz mono
Output : (1, 80, 100) float32  -> feeds the wake-word ONNX / torch model
"""

from __future__ import annotations

import numpy as np

# --- Historical exp15 constants (do not change without re-validation) -------
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 512
TARGET_FRAMES = 100
FMIN = 20
FMAX = 8000
CMVN_EPS = 1e-5


def _mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float):
    def hz2mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels)
    freqs = mel2hz(mels)
    n_bins = n_fft // 2 + 1
    bins = np.clip(np.floor((n_bins - 1) * (freqs / (sr / 2))).astype(int), 0, n_bins - 1)
    left = np.concatenate([[0], bins[:-1]])
    center = bins
    right = np.concatenate([bins[1:], [n_bins - 1]])
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    up = center - left
    dn = right - center
    for i in range(n_mels):
        if up[i] > 0:
            fb[i, left[i]:center[i] + 1] = np.linspace(0, 1, center[i] - left[i] + 1)
        if dn[i] > 0:
            fb[i, center[i]:right[i] + 1] = np.linspace(1, 0, right[i] - center[i] + 1)
    return fb


_FB = None


def mel_filterbank():
    global _FB
    if _FB is None:
        _FB = _mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX)
    return _FB


_WIN = None


def hann_window():
    global _WIN
    if _WIN is None:
        n = np.arange(WIN_LENGTH)
        _WIN = (0.5 * (1.0 - np.cos(2.0 * np.pi * n / WIN_LENGTH))).astype(np.float32)
    return _WIN


def _stft_power(pcm: np.ndarray) -> np.ndarray:
    # float64 throughout so log10 + CMVN do not amplify float32 rounding on
    # near-silent frames (matches the historical pipeline exactly).
    pcm = np.asarray(pcm, dtype=np.float64).ravel()
    pad = N_FFT // 2
    x = np.concatenate([np.zeros(pad), pcm, np.zeros(pad)])
    n_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
    win = hann_window().astype(np.float64)
    power = np.empty((N_FFT // 2 + 1, n_frames), dtype=np.float64)
    for f in range(n_frames):
        seg = x[f * HOP_LENGTH: f * HOP_LENGTH + N_FFT] * win
        spec = np.fft.rfft(seg, n=N_FFT)
        power[:, f] = spec.real ** 2 + spec.imag ** 2
    return power.astype(np.float32)


def _cmvn(mel_db: np.ndarray) -> np.ndarray:
    """mel_db: (n_mels, frames) -> per-frame CMVN over the mel axis."""
    mu = mel_db.mean(axis=0, keepdims=True)
    var = mel_db.var(axis=0, keepdims=True)
    return (mel_db - mu) / np.sqrt(var + CMVN_EPS)


def _delta(x: np.ndarray, pad: int = 2) -> np.ndarray:
    """x: (C, F) -> centered 1st derivative along frames, replicate pad."""
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    return (xp[:, 2 * pad:] - xp[:, :-2 * pad]) / (2.0 * pad)


def _pad_or_truncate(mel: np.ndarray, target_frames: int) -> np.ndarray:
    f = mel.shape[1]
    if f >= target_frames:
        start = (f - target_frames) // 2
        return mel[:, start:start + target_frames]
    pad = target_frames - f
    left = pad // 2
    return np.pad(mel, ((0, 0), (left, pad - left)), mode="constant")


def mel_from_pcm(pcm: np.ndarray) -> np.ndarray:
    """16 kHz mono PCM (n_samples,) -> (1, 80, 100) float32 [static + delta]."""
    power = _stft_power(pcm)
    mel = mel_filterbank().astype(np.float64) @ power.astype(np.float64)
    mel = np.log10(np.clip(mel, 1e-5, None))
    mel_db = (mel * 10.0).astype(np.float32)
    static = _cmvn(mel_db)
    delta = _delta(static)
    feat = np.concatenate([static, delta], axis=0).astype(np.float32)
    feat = _pad_or_truncate(feat, TARGET_FRAMES)
    return feat[None, :, :]  # (1, 80, 100)


def resample_to_16k(data: np.ndarray, sr: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32).ravel()
    if sr == SAMPLE_RATE:
        return data
    try:
        from scipy.signal import resample
        n_out = int(round(len(data) * SAMPLE_RATE / sr))
        return resample(data, n_out).astype(np.float32)
    except Exception:  # pragma: no cover - scipy absent
        n_out = int(round(len(data) * SAMPLE_RATE / sr))
        xp = np.linspace(0.0, 1.0, num=len(data), dtype=np.float32)
        xo = np.linspace(0.0, 1.0, num=n_out, dtype=np.float32)
        return np.interp(xo, xp, data).astype(np.float32)


def load_wav_mono_16k(path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return resample_to_16k(data, sr)


def load_wav_to_mel(path) -> np.ndarray:
    """Convenience: WAV path -> (1, 80, 100) float32 features."""
    return mel_from_pcm(load_wav_mono_16k(path))
