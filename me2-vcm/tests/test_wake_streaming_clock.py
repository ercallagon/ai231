"""Regression for chunk-dependent cooldown and experimental DSP contracts."""
import numpy as np
import pytest
from hey_swift.wakeword.inference import GateConfig, WakeWordDetector, _State
from hey_swift.wakeword.streaming_features import window_features


def probe(chunk):
    d = object.__new__(WakeWordDetector)
    d.gate = GateConfig(); d.win = 32000; d.hop_samples = 4000
    d.subhop_samples = 4000; d.sustain = 2; d.buf = np.zeros(0, dtype=np.float32)
    d._state = _State(); d._score = lambda: (1., 0., True)
    events = []
    for start in range(0, 160000, chunk):
        events.extend(d.process(np.ones(min(chunk, 160000-start), dtype=np.float32))[0])
    return events, d


@pytest.mark.parametrize("chunk", [1, 1733, 4000, 160000])
def test_integer_clock_exact_cooldown(chunk):
    expected, _ = probe(1600)
    actual, d = probe(chunk)
    assert actual == expected
    assert [e["t"] for e in actual] == [2.25, 4.25, 6.25, 8.25]
    assert d._state.samples == 160000 and d._state.t == 10.
    d.reset(); assert d._state.samples == 0 and d._state.last_fire_sample is None


@pytest.mark.parametrize("frontend", ["cmvn2s_v1", "pcen2s_v1"])
def test_frontend_determinism_shape_silence(frontend):
    x = np.random.default_rng(1).normal(0,.02,32000).astype(np.float32)
    a = window_features(x,frontend)
    assert a.shape == (1,80,200) and a.dtype == np.float32
    np.testing.assert_array_equal(a,window_features(x.copy(),frontend))
    assert np.isfinite(window_features(np.zeros(32000),frontend)).all()
    with pytest.raises(ValueError): window_features(np.zeros(32001),frontend)
    with pytest.raises(ValueError): window_features(np.full(32000,np.nan),frontend)
