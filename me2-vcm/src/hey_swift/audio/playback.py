"""Speaker output: confirmation beeps, music, and TTS-style spoken feedback.

Imports `sounddevice` lazily. Provides a no-op fallback (NullPlayback) so the
pipeline runs headless (tests, CI) without audio hardware.

The Raspberry Pi 5 speaker rejects 16 kHz, so all output is generated / played
at the configured ``playback_sample_rate`` (48 kHz on the Pi). Files are
resampled to that rate before playback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample ``x`` from ``sr_in`` to ``sr_out`` (scipy if available, else
    linear interpolation). No-op when the rates already match."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if sr_in == sr_out or x.size == 0:
        return x
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(sr_out, sr_in)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:  # pragma: no cover - scipy absent
        n_out = int(round(len(x) * sr_out / sr_in))
        if n_out <= 1:
            return np.zeros(n_out, dtype=np.float32)
        xp = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
        xo = np.linspace(0.0, 1.0, num=n_out, dtype=np.float32)
        return np.interp(xo, xp, x).astype(np.float32)


class Playback:
    """Real speaker output via sounddevice (lazy import)."""

    def __init__(self, config: str = "configs/hardware.yaml",
                 sr: Optional[int] = None):
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Playback needs the 'audio' extra: pip install 'hey-swift[audio]'"
            ) from exc
        import yaml
        self.cfg = yaml.safe_load(open(config, encoding="utf-8")) or {}
        io_cfg = self.cfg.get("audio_io", {}) or {}
        # Pi speaker rejects 16 kHz -> use the configured rate (48k on the Pi).
        self.sr = int(sr or io_cfg.get("playback_sample_rate", 48000))
        self.device = io_cfg.get("playback_device")
        self._sd = __import__("sounddevice")

    def beep(self, freq_hz: float = 880.0, duration: float = 0.12,
             gain: float = 0.3) -> None:
        """Generated confirmation tone, rendered at the supported speaker rate
        (the Pi rejects 16 kHz, so this no longer assumes 16 kHz)."""
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        tone = (gain * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        self._sd.play(tone, self.sr, device=self.device)
        self._sd.wait()

    def play_file(self, path: str) -> None:
        """Play an audio file (wav/mp3/...), resampled to the supported speaker
        rate. Mono-downmixed if multichannel."""
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sr:
            audio = _resample_to(audio, sr, self.sr)
        self._sd.play(audio, self.sr, device=self.device)
        self._sd.wait()

    def play_ack(self, path: str) -> None:
        """Play the acknowledgement cue (the VOICY wake chime). Kept separate
        from :meth:`play_file` for semantic clarity at the call site; same
        behaviour (resample to the supported rate and play)."""
        self.play_file(path)

    def play_music(self, track: str) -> None:  # pragma: no cover - device dep
        self.play_file(track)


class NullPlayback:
    """Headless fallback: records intended output, plays nothing."""

    def __init__(self):
        self.log = []

    def beep(self, freq_hz: float = 880.0, duration: float = 0.12,
             gain: float = 0.3) -> None:
        self.log.append(("beep", freq_hz, duration))

    def play_file(self, path: str) -> None:
        self.log.append(("file", path))

    def play_ack(self, path: str) -> None:
        self.log.append(("ack", path))

    def play_music(self, track: str) -> None:
        self.log.append(("music", track))


def make_playback(config: str = "configs/hardware.yaml",
                  allow_null: bool = True) -> Optional[object]:
    """Return a Playback, or NullPlayback/None if audio is unavailable."""
    try:
        return Playback(config)
    except ImportError:
        if allow_null:
            return NullPlayback()
        raise
