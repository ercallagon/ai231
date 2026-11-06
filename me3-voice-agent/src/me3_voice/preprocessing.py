"""Stateful PCM conversion and diagnostics; no pretend AEC or automatic gain tuning.

Echo cancellation must operate on the actual capture/render path upstream. This
module preserves continuity between callbacks and only converts for model input.
"""

from dataclasses import dataclass

import numpy as np
import soxr

from .contracts import RATE


@dataclass
class SignalStats:
    samples: int = 0
    clipped: int = 0
    peak: float = 0.0
    total: float = 0.0
    squares: float = 0.0

    def update(self, values):
        values = values.astype(np.float64).ravel()
        self.samples += len(values)
        self.clipped += int(np.count_nonzero(np.abs(values) >= 32767 / 32768))
        self.peak = max(self.peak, float(np.max(np.abs(values), initial=0)))
        self.total += float(values.sum())
        self.squares += float(np.dot(values, values))

    def report(self):
        return {
            "samples": self.samples,
            "peak": self.peak,
            "rms": (self.squares / self.samples) ** 0.5 if self.samples else 0,
            "mean_dc": self.total / self.samples if self.samples else 0,
            "clipped_fraction": self.clipped / self.samples if self.samples else 0,
        }


class PCMPreprocessor:
    """Explicit PCM16 mono/stereo -> mono PCM16 16 kHz, with persistent HQ resampling.

    No denoising, AGC, peak normalization or echo cancellation is silently applied.
    At most one second per push. Sequence gaps fail rather than inventing audio.
    """

    def __init__(self, sample_rate: int, channels: int = 1):
        if sample_rate not in (8000, 16000, 24000, 32000, 44100, 48000) or channels not in (1, 2):
            raise ValueError("unsupported PCM rate/channel count")
        self.sample_rate, self.channels = sample_rate, channels
        self.sequence = 0
        self.closed = False
        self.input_stats, self.output_stats = SignalStats(), SignalStats()
        self.resampler = soxr.ResampleStream(sample_rate, RATE, 1, dtype="float32", quality="HQ")

    def push(self, pcm: bytes, sequence: int) -> bytes:
        if self.closed:
            raise ValueError("preprocessor already finished")
        if sequence != self.sequence:
            raise ValueError("capture sequence gap or duplicate")
        if (
            not pcm
            or len(pcm) % (2 * self.channels)
            or len(pcm) > self.sample_rate * self.channels * 2
        ):
            raise ValueError("expected complete PCM16 frames, at most one second")
        values = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        self.input_stats.update(values)
        if self.channels == 2:
            values = values.reshape(-1, 2).mean(axis=1)
        self.sequence += 1
        return self._encode(self.resampler.resample_chunk(values))

    def finish(self) -> bytes:
        if self.closed:
            return b""
        self.closed = True
        return self._encode(self.resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))

    def _encode(self, values):
        self.output_stats.update(values)
        return np.clip(np.rint(values * 32768), -32768, 32767).astype("<i2").tobytes()


class PCMFrames:
    """Bounded frame assembly; leave the unpadded tail explicit at file EOF."""

    def __init__(self, samples):
        if not 1 <= samples <= RATE:
            raise ValueError("invalid model frame length")
        self.size = samples * 2
        self.pending = bytearray()

    def push(self, pcm):
        if len(pcm) % 2:
            raise ValueError("partial PCM16 sample")
        if len(pcm) > RATE * 2 + self.size:
            raise ValueError("input chunk too large")
        self.pending.extend(pcm)
        frames = []
        while len(self.pending) >= self.size:
            frames.append(bytes(self.pending[: self.size]))
            del self.pending[: self.size]
        return frames

    def finish(self):
        tail = bytes(self.pending)
        self.pending.clear()
        return tail
