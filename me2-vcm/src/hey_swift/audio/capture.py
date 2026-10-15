"""Microphone capture: streaming hops + post-wake command capture.

Imports `sounddevice` lazily. On a machine without audio hardware, constructing
CommandCapture raises a clear error (handled by app.main).

The Raspberry Pi 5 board microphone (ALSA device 0) only accepts 48 kHz and
rejects 16 kHz / 44.1 kHz, while the wake-word / VCM feature pipelines are fixed
at 16 kHz (see ``hey_swift.wakeword.features.SAMPLE_RATE``). So on the Pi we
capture at the device's native rate and resample to 16 kHz. The reusable pieces
for that live here and are shared by the production pipeline and
``scripts/wake_test.py``:

* :class:`ContinuousResampler` -- stateful, chunk-boundary-safe resampler
  (carries a tail forward so the stream resamples as one continuous signal).
* :func:`capture_supported_rate` -- probes a device for a rate it accepts.
"""

from __future__ import annotations

import time
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
import yaml

from hey_swift.wakeword.features import SAMPLE_RATE

# Rates a physical mic might accept, tried in order by capture_supported_rate().
# 48 kHz first (the Pi 5 board mic requirement); 16 kHz yields a no-resample
# fast path on desktop mics that accept it.
DEFAULT_RATE_PREFERENCES: Tuple[int, ...] = (48000, 16000, 44100, 8000)


def _resample_segment(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample one segment. Prefers scipy.signal.resample_poly (exact rational
    ratio, no length drift); falls back to the shared NumPy path in
    ``hey_swift.wakeword.features.resample_to_16k`` when scipy is absent."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if sr_in == sr_out or x.size == 0:
        return x
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(sr_out, sr_in)
        out = resample_poly(x, sr_out // g, sr_in // g)
        return out.astype(np.float32)
    except Exception:  # pragma: no cover - scipy absent
        from hey_swift.wakeword.features import resample_to_16k

        if sr_out == SAMPLE_RATE:
            return resample_to_16k(x, sr_in)
        n_out = int(round(len(x) * sr_out / sr_in))
        if n_out <= 1:
            return np.zeros(n_out, dtype=np.float32)
        xp = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
        xo = np.linspace(0.0, 1.0, num=n_out, dtype=np.float32)
        return np.interp(xo, xp, x).astype(np.float32)


class ContinuousResampler:
    """Resample a stream of chunks from ``sr_in`` to ``sr_out`` with carry-over.

    Resampling each 0.1 s chunk in isolation leaks phase across chunk edges and
    produces audible clicks / a smeared pitch. This class carries the tail of the
    previous chunk forward so the stream resamples as if it were one continuous
    signal -- the path the final integration should use.

    Parameters
    ----------
    sr_in:
        Sample rate of the incoming chunks (the device's native rate, e.g. 48000).
    sr_out:
        Target sample rate (default: the model rate, 16000).
    tail:
        Number of incoming samples to overlap between chunks. Larger = smoother
        transitions, slightly more latency. Default 1024 (~21 ms at 48 kHz).
    """

    def __init__(self, sr_in: int, sr_out: int = SAMPLE_RATE, tail: int = 1024):
        self.sr_in = int(sr_in)
        self.sr_out = int(sr_out)
        self.tail = max(0, int(tail))
        self._prev_in = np.zeros(0, dtype=np.float32)   # previous chunk (sr_in)
        self._prev_out = np.zeros(0, dtype=np.float32)  # resampled prev chunk (sr_out)

    def reset(self) -> None:
        """Drop carried-over state (call after a capture gap, e.g. after playing
        an acknowledgement tone while the mic is paused)."""
        self._prev_in = np.zeros(0, dtype=np.float32)
        self._prev_out = np.zeros(0, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Consume one ``sr_in`` chunk; return the *new* ``sr_out`` samples.

        The previous chunk's resampled output is kept so the boundary region is
        resampled as part of a continuous signal (no chunk-edge click); only the
        portion beyond the previous chunk's contribution is returned, so no
        samples are dropped or duplicated. When ``sr_in == sr_out`` this is a
        zero-cost passthrough so the common desktop 16 kHz case stays free.
        """
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        if self.sr_in == self.sr_out:
            self._prev_in = np.zeros(0, dtype=np.float32)
            self._prev_out = np.zeros(0, dtype=np.float32)
            return chunk
        if self._prev_in.size:
            combined = np.concatenate([self._prev_in, chunk])
        else:
            combined = chunk
        out = _resample_segment(combined, self.sr_in, self.sr_out)
        skip = self._prev_out.size
        new = out[skip:] if skip < out.size else np.zeros(0, dtype=np.float32)
        self._prev_in = chunk.copy()
        self._prev_out = _resample_segment(chunk, self.sr_in, self.sr_out)
        return new

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        return self.push(chunk)


def capture_supported_rate(
    device=None,
    preferences: Iterable[int] = DEFAULT_RATE_PREFERENCES,
    channels: int = 1,
) -> Tuple[int, int]:
    """Probe ``device`` and return ``(sample_rate, blocksize)`` it accepts.

    Tries each rate in ``preferences`` by opening and immediately closing a
    throwaway stream; the first rate the device accepts wins. On the Pi 5 board
    mic this yields 48 kHz (16 kHz raises an exception); on a desktop mic that
    accepts 16 kHz it yields 16 kHz and no resampling is needed downstream.

    ``blocksize`` is a 0.1 s frame at the chosen rate, matching the cadence the
    detector and command capture already use.
    """
    import sounddevice as sd  # lazy: only needed on a machine with audio

    last_err: Optional[Exception] = None
    for rate in preferences:
        blocksize = max(1, int(rate * 0.1))
        try:
            with sd.RawInputStream(
                samplerate=rate,
                blocksize=blocksize,
                dtype="float32",
                channels=channels,
                device=device,
            ):
                return int(rate), blocksize
        except Exception as exc:  # device rejects this rate; try the next
            last_err = exc
    raise RuntimeError(
        f"No supported capture rate found for device={device!r} among "
        f"{list(preferences)}; last error: {last_err}"
    )


class CommandCapture:
    """Streams microphone hops and captures a bounded command after a wake.

    Captures at the device's supported rate (48 kHz on the Pi 5 board mic) and
    yields 16 kHz mono float32 hops, resampling continuously so chunk boundaries
    do not introduce clicks.
    """

    def __init__(self, config: str = "configs/hardware.yaml"):
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Audio capture needs the 'audio' extra: pip install 'hey-swift[audio]'"
            ) from exc

        self.cfg = yaml.safe_load(open(config, encoding="utf-8")) or {}
        io_cfg = self.cfg.get("audio_io", {}) or {}
        self.sr = SAMPLE_RATE  # model rate; stream() yields at this rate
        self.block_seconds = float(io_cfg.get("capture_block_seconds", 3.0))
        self.pre_roll_ms = int(io_cfg.get("pre_roll_ms", 200))
        self.device = io_cfg.get("capture_device")
        # 0/None -> auto-probe the device for a rate it accepts (Pi = 48 kHz).
        self.capture_sr = int(io_cfg.get("capture_sample_rate", 0) or 0) or None
        self._sd = __import__("sounddevice")
        self._t0 = time.time()

    def timestamp(self) -> float:
        return time.time() - self._t0

    def _resolve_capture_rate(self) -> int:
        if self.capture_sr:
            return int(self.capture_sr)
        rate, _ = capture_supported_rate(self.device)
        return rate

    def stream(self) -> Iterator[np.ndarray]:
        """Yield 16 kHz mono float32 hops (0.1 s) from the microphone.

        Captures at the device's supported rate and resamples continuously to
        the model rate, so the Pi 5 board mic (48 kHz only) works without
        chunk-boundary artifacts.
        """
        rate = self._resolve_capture_rate()
        hop = max(1, int(rate * 0.1))
        rs = ContinuousResampler(rate, self.sr)
        q = self._sd.RawInputStream(
            samplerate=rate, blocksize=hop, dtype="float32",
            channels=1, device=self.device,
        )
        with q:
            while True:
                data, _overflow = q.read(hop)
                chunk = np.frombuffer(data, dtype=np.float32).reshape(-1)
                yield rs.push(chunk)  # -> 16 kHz, continuous

    def capture_command(self) -> np.ndarray:
        """Record up to block_seconds of audio after a wake, at 16 kHz."""
        rate = self._resolve_capture_rate()
        n = int(rate * self.block_seconds)
        frames = self._sd.rec(n, samplerate=rate, dtype="float32",
                              channels=1, device=self.device)
        self._sd.wait()
        frames = np.asarray(frames, dtype=np.float32).reshape(-1)
        if rate != self.sr:
            frames = _resample_segment(frames, rate, self.sr)
        return frames
