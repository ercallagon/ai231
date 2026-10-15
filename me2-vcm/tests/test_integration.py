"""End-to-end integration test (import-safe: no torch/audio/hardware required).

Exercises the full *decision* path without models or a microphone:
    synthetic waveform -> VCM features -> (fake) prediction -> action router
        -> typed action result -> feedback

This proves the wiring is coherent: feature shapes feed the prediction contract,
the prediction contract feeds the router, and the router produces a well-formed
ActionResult with feedback. The real model is swapped in at deploy time; the
glue tested here is what must not break.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def wired(tmp_path):
    """Build a router + feature extractor from temp configs (no model needed)."""
    from hey_swift.actions.router import ActionRouter
    from hey_swift.vcm.features import FeatureConfig, MelFeatures

    actions = tmp_path / "actions.yaml"
    actions.write_text(
        "rejection:\n  oos_confidence_threshold: 0.5\n  on_oos:\n    led: error\n"
        "    speak: Sorry.\n"
        "actions:\n"
        "  PLAY_MUSIC:\n    handler: music.play\n"
        "  BRIGHTNESS:\n    handler: lights.set_brightness\n"
        "    slots: [BRIGHTNESS]\n    range: [0, 100]\n"
        "feedback:\n  led: success\n  speak_default: Done.\n"
    )
    vcm_cfg = tmp_path / "vcm.yaml"
    vcm_cfg.write_text(
        "schema:\n  n_intents: 19\n  oos_class: OUT_OF_SCOPE\n  n_way_head: 20\n"
        "  slot_reject: false\n  slotted_intents: [BRIGHTNESS]\n"
        "  slot_sets:\n    BRIGHTNESS: [20, 60, 100, REJECT]\n"
        "oos:\n  confidence_threshold: 0.5\n"
        "audio:\n  sample_rate: 16000\n  n_fft: 400\n  hop_length: 160\n"
        "  n_mels: 80\n  window_seconds: 2.0\n  f_min: 20.0\n  f_max: 8000.0\n"
    )
    router = ActionRouter(str(actions), str(vcm_cfg))
    feats = MelFeatures(FeatureConfig.from_dict(
        __import__("yaml").safe_load(open(vcm_cfg, encoding="utf-8")).get("audio", {})))
    return router, feats


def test_feature_to_prediction_to_action(wired):
    router, feats = wired

    # 1) A synthetic waveform produces the locked feature shape.
    wav = np.random.default_rng(0).standard_normal(32000).astype(np.float32)
    mel = feats.transform(wav)
    assert mel.shape == (1, 80, 198)

    # 2) A (fake) model prediction in the router's contract.
    prediction = {"intent": "BRIGHTNESS", "confidence": 0.92,
                  "slots": {"BRIGHTNESS": "60"}, "oos": False}

    # 3) The router produces a well-formed ActionResult with feedback.
    result = router.route(prediction)
    assert result.ok is True
    assert result.intent == "BRIGHTNESS"
    assert result.action == "lights.set_brightness"
    assert result.feedback.get("led") == "success"
    assert "60" in result.message


def test_oos_produces_error_feedback(wired):
    router, _ = wired
    result = router.route({"intent": "OUT_OF_SCOPE", "confidence": 0.9,
                           "oos": True, "slots": {}})
    assert result.ok is False
    assert result.feedback.get("led") == "error"


def test_app_build_pipeline_is_import_safe():
    """app.build_pipeline must not import torch/audio at call-construction time
    beyond the lazy subsystem imports; importing the module itself is always safe."""
    import hey_swift.app as app  # noqa: F401
    assert hasattr(app, "build_pipeline")
    assert hasattr(app, "PipelineState")
    assert hasattr(app, "dispatch")
