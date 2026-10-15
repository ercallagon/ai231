"""Live / pretrained wake-word detection (faithful exp15 runtime gating).

Ports the historical `me2-wakeword-swift/inference/run_live.py` detector so the
canonical repository can drive the existing exp15 ONNX champion with the exact
validated semantics:

    * 1.0 s scoring window, 0.25 s hop (streaming ring buffer)
    * optional 4 overlapping 1.0 s sub-windows (0.25 s sub-hop), max P(wake)
    * RMS floor gate  - the scoring window must exceed -60 dBFS (silence guard)
    * model gate      - P(non-wake) < 0.90
    * probability gate- P(wake) >= threshold (0.30)
    * sustain         - N (2) consecutive qualifying hops before firing
    * cooldown        - suppress re-fire for 2.0 s after a fire

NumPy + ONNX Runtime only (Pi-clean, no torch). The feature path is the shared
single source of truth in `hey_swift.wakeword.features`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml

from .features import SAMPLE_RATE, mel_from_pcm

DEFAULT_ONNX = "models/wakeword/wakeword_exp15_mined_lp.onnx"

# --- Historical exp15 operating constants ----------------------------------
WIN_SECONDS = 1.0
HOP_SECONDS = 0.25
MULTIWIN_N = 4
SUBHOP_SECONDS = 0.25
RMS_FLOOR_DB = -60.0
GATE_NONWAKE = 0.90
DEFAULT_THRESHOLD = 0.30


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _db(x: float) -> float:
    return 20.0 * float(np.log10(x + 1e-9))


@dataclass(frozen=True)
class GateConfig:
    threshold: float = DEFAULT_THRESHOLD
    sustain: int = 2
    cooldown_seconds: float = 2.0
    rms_floor_db: float = RMS_FLOOR_DB
    hop_seconds: float = HOP_SECONDS
    multiwin: bool = True
    n_subwin: int = MULTIWIN_N
    subhop_seconds: float = SUBHOP_SECONDS
    gate_nonwake: float = GATE_NONWAKE


@dataclass
class _State:
    streak: int = 0
    last_fire: float = -1.0
    t: float = 0.0
    since_hop: int = 0
    primed: bool = False


class WakeWordDetector:
    """Streaming "Hey Swift" detector with the validated exp15 gating."""

    def __init__(self, config: str = "configs/wakeword.yaml",
                 onnx_path: Optional[str] = None):
        import onnxruntime as ort

        self.cfg = yaml.safe_load(open(config, encoding="utf-8")) or {}
        inf = self.cfg.get("inference", {}) or {}
        self.gate = GateConfig(
            threshold=float(inf.get("threshold", DEFAULT_THRESHOLD)),
            sustain=int(inf.get("sustain", 2)),
            cooldown_seconds=float(inf.get("cooldown_seconds", 2.0)),
            rms_floor_db=float(inf.get("rms_floor_db", RMS_FLOOR_DB)),
            hop_seconds=float(inf.get("hop_seconds", HOP_SECONDS)),
        )
        self.win = int(WIN_SECONDS * SAMPLE_RATE)
        self.hop_samples = int(self.gate.hop_seconds * SAMPLE_RATE)
        self.subhop_samples = int(self.gate.subhop_seconds * SAMPLE_RATE)
        self.n_subwin = self.gate.n_subwin if self.gate.multiwin else 1
        self.sustain = max(1, self.gate.sustain)

        path = onnx_path or self.cfg.get("model", {}).get("onnx", DEFAULT_ONNX)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.buf = np.zeros(0, dtype=np.float32)
        self._state = _State()

    # ------------------------------------------------------------------ #
    def _score_window(self, w: np.ndarray):
        """w: (WIN,) float32 -> (p_wake, p_nonwake, rms)."""
        feat = mel_from_pcm(w)
        logits = self.sess.run(None, {self.input_name: feat})[0][0]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        p_wake = float(p[0])
        p_nonwake = float(p[1] + p[2]) if p.size > 2 else float(1.0 - p_wake)
        return p_wake, p_nonwake, rms(w)

    def _score(self):
        """Score the most recent 1.0 s (or N overlapping sub-windows).

        Returns (p_wake, p_nonwake, window_ok).
        """
        if self.buf.size < self.win:
            return 0.0, 0.0, False
        if self.gate.multiwin and self.n_subwin > 1:
            pmax, nw_min, best_rms = 0.0, 1.0, 0.0
            for k in range(self.n_subwin - 1, -1, -1):
                start = len(self.buf) - self.win - k * self.subhop_samples
                if start < 0:
                    continue
                w = self.buf[start:start + self.win]
                pw, nw, wrms = self._score_window(w)
                if pw > pmax:
                    pmax, nw_min, best_rms = pw, nw, wrms
            return pmax, nw_min, (_db(best_rms) > self.gate.rms_floor_db)
        w = self.buf[-self.win:]
        pw, nw, wrms = self._score_window(w)
        return pw, nw, (_db(wrms) > self.gate.rms_floor_db)

    def _gate(self, p_nonwake: float, window_ok: bool) -> bool:
        """Allow a trigger only if the window has real audio AND the model is
        not confident it is non-wake."""
        return window_ok and (p_nonwake < self.gate.gate_nonwake)

    def process(self, chunk: np.ndarray):
        """Feed a chunk of 16 kHz mono float32. Returns (events, p_wake_last)."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        cap = self.win + 4 * self.subhop_samples
        events: List[dict] = []
        p_last = 0.0

        # Consume samples only as far as the next exact scoring boundary.
        # This keeps hop cadence independent of input chunk size and retains
        # the rolling history needed by the overlapping-window gate.
        offset = 0
        while offset < chunk.size:
            if self._state.primed:
                needed = self.hop_samples - self._state.since_hop
            else:
                needed = self.win - self.buf.size
            take = min(needed, chunk.size - offset)
            self.buf = np.concatenate((self.buf, chunk[offset:offset + take]))
            if self.buf.size > cap:
                self.buf = self.buf[-cap:]
            offset += take
            self._state.t += take / SAMPLE_RATE

            if self._state.primed:
                self._state.since_hop += take
                if self._state.since_hop < self.hop_samples:
                    continue
                self._state.since_hop = 0
            elif self.buf.size < self.win:
                continue
            else:
                self._state.primed = True

            p_wake, p_nonwake, window_ok = self._score()
            p_last = p_wake
            if p_wake >= self.gate.threshold and self._gate(p_nonwake, window_ok):
                self._state.streak += 1
            else:
                self._state.streak = 0
            if (self._state.streak >= self.sustain
                    and (self._state.t - self._state.last_fire) >= self.gate.cooldown_seconds):
                events.append({"t": self._state.t, "prob": p_wake})
                self._state.last_fire = self._state.t
                self._state.streak = 0
        return events, p_last

    def reset(self) -> None:
        """Clear all streaming state (buffer, streak, timing)."""
        self.buf = np.zeros(0, dtype=np.float32)
        self._state = _State()

    def detect_file(self, path: str, chunk_seconds: float = 0.5):
        """Replay a WAV through the detector. Returns (events, n_chunks)."""
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            from .features import resample_to_16k
            data = resample_to_16k(data, sr)
        chunk = int(chunk_seconds * SAMPLE_RATE)
        self.reset()
        events: List[dict] = []
        n = 0
        for i in range(0, len(data), chunk):
            ev, _ = self.process(data[i:i + chunk])
            events.extend(ev)
            n += 1
        return events, n
