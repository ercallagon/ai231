"""Deterministic experimental rolling-window frontends; no learned weights."""
import numpy as np
from scipy.signal import lfilter
from . import features as F


def window_features(pcm, frontend="cmvn2s_v1"):
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if pcm.size != 32000 or not np.isfinite(pcm).all():
        raise ValueError("Experimental wake frontend requires 32000 finite samples")
    power = F._stft_power(pcm)
    mel = F.mel_filterbank().astype(np.float64) @ power.astype(np.float64)
    if frontend == "cmvn2s_v1":
        static = F._cmvn((10*np.log10(np.clip(mel, 1e-5, None))).astype(np.float32))
    elif frontend == "pcen2s_v1":
        # Causal EMA within the already captured scoring window. Reset semantics
        # are identical for training and inference, not continuous-stream PCEN.
        smooth = lfilter([.025], [1., -.975], mel, axis=1,
                         zi=(.975*mel[:, :1]))[0]
        static = (np.sqrt(mel / (1e-6+smooth)**.98 + 2.) - np.sqrt(2.)).astype(np.float32)
    else:
        raise ValueError(f"Unknown wake frontend: {frontend}")
    output = F._pad_or_truncate(np.concatenate([static, F._delta(static)]), 200)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite wake features")
    return output[None].astype(np.float32)
