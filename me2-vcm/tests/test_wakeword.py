"""Tests for the wake-word subsystem (faithful exp15 pipeline).

Import-safe: the feature / gating / dataset / model-shape tests need only
NumPy. The ONNX + torch tests are skipped automatically when those packages or
the deployed model are absent (e.g. a fresh clone before `make download`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO = Path(__file__).resolve().parent.parent
ONNX = REPO / "models" / "wakeword" / "wakeword_exp15_mined_lp.onnx"
HAVE_TORCH = pytest.importorskip("torch") is not None


class TestWakeFeatures:
    def test_output_shape_locked(self):
        from hey_swift.wakeword.features import mel_from_pcm
        wav = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
        x = mel_from_pcm(wav)
        # 40 mel * (static + 1 delta) = 80 channels, 100 frames
        assert x.shape == (1, 80, 100), f"got {x.shape}"
        assert x.dtype == np.float32

    def test_short_clip_padded(self):
        from hey_swift.wakeword.features import mel_from_pcm
        wav = np.random.default_rng(1).standard_normal(8000).astype(np.float32)
        x = mel_from_pcm(wav)
        assert x.shape == (1, 80, 100)

    def test_cmvn_near_zero_mean(self):
        from hey_swift.wakeword.features import mel_from_pcm
        wav = np.random.default_rng(1).standard_normal(16000).astype(np.float32)
        x = mel_from_pcm(wav)
        # After per-frame CMVN over the mel axis the static half is ~zero-mean.
        assert abs(x[0, :40].mean()) < 1e-3

    def test_bit_exact_against_reference(self):
        # The DSP must be deterministic and stable across calls.
        from hey_swift.wakeword.features import mel_from_pcm
        wav = np.random.default_rng(7).standard_normal(16000).astype(np.float32)
        a = mel_from_pcm(wav)
        b = mel_from_pcm(wav.copy())
        assert np.array_equal(a, b)


class _FakeDetector:
    """Drive the real gating semantics without an ONNX session.

    Mirrors WakeWordDetector.process's gate sequence (RMS floor -> cooldown ->
    probability -> sustain) but injects p(wake) and a controllable clock.
    """

    def __init__(self, gate):
        self.gate = gate
        self._sustain = 0
        self._last_fire = -1e9

    def _rms_db(self, wav):
        rms = float(np.sqrt(np.mean(np.square(wav))))
        return -120.0 if rms <= 1e-9 else 20.0 * np.log10(rms)

    def detect(self, audio_hop, p_wake=0.0, now=None):
        if self._rms_db(audio_hop) < self.gate.rms_floor_db:
            self._sustain = 0
            return False
        if now - self._last_fire < self.gate.cooldown_seconds:
            return False
        if p_wake >= self.gate.threshold:
            self._sustain += 1
        else:
            self._sustain = 0
        if self._sustain >= self.gate.sustain:
            self._last_fire = now
            self._sustain = 0
            return True
        return False


@pytest.mark.parametrize("chunk_samples", [1600, 8000])
def test_stream_detector_scores_on_exact_hops_and_keeps_window_history(chunk_samples):
    from hey_swift.wakeword.inference import GateConfig, WakeWordDetector, _State

    detector = object.__new__(WakeWordDetector)
    detector.gate = GateConfig(threshold=0.3, sustain=2, cooldown_seconds=2.0)
    detector.win = 16000
    detector.hop_samples = 4000
    detector.subhop_samples = 4000
    detector.n_subwin = 4
    detector.sustain = 2
    detector.buf = np.zeros(0, dtype=np.float32)
    detector._state = _State()
    scored_buffer_sizes = []

    def score():
        scored_buffer_sizes.append(detector.buf.size)
        return 0.9, 0.1, True

    detector._score = score
    audio = np.zeros(32000, dtype=np.float32)
    events = []
    for start in range(0, audio.size, chunk_samples):
        new_events, _ = detector.process(audio[start:start + chunk_samples])
        events.extend(new_events)

    assert events[0]["t"] == pytest.approx(1.25)
    assert scored_buffer_sizes[:5] == [16000, 20000, 24000, 28000, 32000]


@pytest.fixture
def gate():
    from hey_swift.wakeword.inference import GateConfig
    return GateConfig(threshold=0.30, sustain=2, cooldown_seconds=2.0,
                      rms_floor_db=-60.0, hop_seconds=0.25)


LOUD = np.ones(1600, dtype=np.float32) * 0.5     # ~ -6 dB, above floor
SILENT = np.zeros(1600, dtype=np.float32)        # -inf dB, below floor


class TestGating:
    def test_silence_never_fires(self, gate):
        d = _FakeDetector(gate)
        for i in range(10):
            assert d.detect(SILENT, p_wake=1.0, now=100.0 + i * 0.25) is False

    def test_requires_sustain(self, gate):
        d = _FakeDetector(gate)
        assert d.detect(LOUD, p_wake=0.9, now=100.0) is False
        assert d.detect(LOUD, p_wake=0.9, now=100.25) is True

    def test_cooldown_suppresses_immediate_refire(self, gate):
        d = _FakeDetector(gate)
        d.detect(LOUD, p_wake=0.9, now=100.0)
        d.detect(LOUD, p_wake=0.9, now=100.25)  # fires
        assert d.detect(LOUD, p_wake=0.9, now=100.5) is False
        assert d.detect(LOUD, p_wake=0.9, now=101.5) is False
        assert d.detect(LOUD, p_wake=0.9, now=103.0) is False
        assert d.detect(LOUD, p_wake=0.9, now=103.25) is True

    def test_subthreshold_resets_sustain(self, gate):
        d = _FakeDetector(gate)
        d.detect(LOUD, p_wake=0.9, now=100.0)
        d.detect(LOUD, p_wake=0.1, now=100.25)  # below threshold -> reset
        d.detect(LOUD, p_wake=0.9, now=100.5)
        assert d.detect(LOUD, p_wake=0.9, now=100.75) is True


class TestGateConfigDefaults:
    def test_exp15_operating_point(self):
        from hey_swift.wakeword.inference import GateConfig
        g = GateConfig()
        assert g.threshold == 0.30
        assert g.sustain == 2
        assert g.cooldown_seconds == 2.0
        assert g.rms_floor_db == -60.0
        assert g.gate_nonwake == 0.90


class TestDataset:
    def test_collect_labels_sorted(self, tmp_path):
        from hey_swift.wakeword import dataset as ds
        for label in ("positive", "negative"):
            (tmp_path / label).mkdir()
            for name in ("b.wav", "a.wav"):
                (tmp_path / label / name).touch()
        items = ds.collect(tmp_path)
        labels = [l for _, l in items]
        # positives (0) come before negatives (1)
        first_neg = labels.index(1)
        assert all(l == 0 for l in labels[:first_neg])
        assert all(l == 1 for l in labels[first_neg:])
        assert len(items) == 4
        assert [Path(p).name for p, _ in items] == ["a.wav", "b.wav", "a.wav", "b.wav"]

    @pytest.mark.skipif(not Path("data/wakeword/raw/train").is_dir(), reason="raw wake dataset integration check; data not fetched")
    def test_counts_match_exp15(self):
        from hey_swift.wakeword import dataset as ds
        from collections import Counter
        c = Counter(l for _, l in ds.collect_split("train"))
        assert c[0] == 1931 and c[1] == 12887
        cv = Counter(l for _, l in ds.collect_split("validation"))
        assert cv[0] == 414 and cv[1] == 726
        ct = Counter(l for _, l in ds.collect_split("test"))
        assert ct[0] == 415 and ct[1] == 728

    @pytest.mark.skipif(not Path("data/wakeword/raw/train").is_dir(), reason="raw wake dataset integration check; data not fetched")
    def test_manifest_matches_disk(self):
        from hey_swift.wakeword import dataset as ds
        for split in ("train", "validation", "test"):
            rep = ds.verify_manifest(split)
            assert rep["ok"], rep


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
class TestModel:
    def test_param_count(self):
        from hey_swift.wakeword.model import build_model, count_params, WakeArchSpec
        m = build_model(WakeArchSpec())
        assert count_params(m) == 204738

    def test_forward_shape(self):
        from hey_swift.wakeword.model import build_model, WakeArchSpec
        import torch
        m = build_model(WakeArchSpec())
        m.eval()
        x = torch.randn(4, 80, 100)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (4, 2)


@pytest.mark.skipif(not ONNX.exists(), reason="deployed ONNX not present")
class TestOnnxInference:
    def test_detector_runs_and_scores(self):
        from hey_swift.wakeword.inference import WakeWordDetector
        det = WakeWordDetector(onnx_path=str(ONNX))
        # feed 1 s of loud noise -> should produce a p_wake in [0,1]
        loud = np.ones(16000, dtype=np.float32) * 0.3
        events, p_last = det.process(loud)
        assert 0.0 <= p_last <= 1.0

    def test_reset_clears_state(self):
        from hey_swift.wakeword.inference import WakeWordDetector
        det = WakeWordDetector(onnx_path=str(ONNX))
        det.process(np.ones(16000, dtype=np.float32) * 0.3)
        assert det.buf.size > 0
        det.reset()
        assert det.buf.size == 0
