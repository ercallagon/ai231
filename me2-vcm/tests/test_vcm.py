"""Tests for the VCM subsystem (import-safe: no torch/audio required).

Covers the pure-logic core: taxonomy resolution (slot_reject on/off), feature
shape, dataset disjointness, and the action router (which depends only on the
taxonomy + handlers, not on models or hardware).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable without installation (tests run from repo root).
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestTaxonomy:
    def test_default_is_20_way(self):
        from hey_swift.vcm.taxonomy import default_taxonomy
        t = default_taxonomy()
        assert t.n_way_head == 20
        assert len(t.intents) == 19
        assert t.oos_class == "OUT_OF_SCOPE"
        assert t.oos_idx == 19

    def test_slot_reject_off_gives_3_way(self):
        from hey_swift.vcm.taxonomy import default_taxonomy
        t = default_taxonomy()  # slot_reject=False
        assert t.slot_classes("TIMER") == ["10s", "30s", "1m"]
        assert t.slot_width("COLOR") == 3
        assert "REJECT" not in t.slot_classes("ALARM")

    def test_non_slotted_has_no_slots(self):
        from hey_swift.vcm.taxonomy import default_taxonomy
        t = default_taxonomy()
        assert t.slot_classes("PLAY_MUSIC") == []
        assert not t.is_slotted("STOP")

    def test_load_from_config(self, tmp_path):
        # A minimal config exercising slot_reject=True -> 4-way.
        cfg = tmp_path / "vcm.yaml"
        cfg.write_text(
            "schema:\n"
            "  n_intents: 19\n"
            "  oos_class: OUT_OF_SCOPE\n"
            "  n_way_head: 20\n"
            "  slot_reject: true\n"
            "  slotted_intents: [TIMER]\n"
            "  slot_sets:\n"
            "    TIMER: [10s, 30s, 1m, REJECT]\n"
        )
        from hey_swift.vcm.taxonomy import load_taxonomy
        t = load_taxonomy(cfg)
        assert t.slot_reject is True
        assert t.slot_classes("TIMER") == ["10s", "30s", "1m", "REJECT"]
        assert t.slot_width("TIMER") == 4


class TestFeatures:
    def test_output_shape_locked(self):
        from hey_swift.vcm.features import FeatureConfig, MelFeatures
        cfg = FeatureConfig()  # 16kHz, 2.0s, 80-mel
        f = MelFeatures(cfg)
        wav = np.random.default_rng(0).standard_normal(cfg.window_samples)
        mel = f.transform(wav)
        assert mel.shape == (1, 80, 198), f"got {mel.shape}"
        assert mel.dtype == np.float32

    def test_handles_short_input_by_padding(self):
        from hey_swift.vcm.features import FeatureConfig, MelFeatures
        f = MelFeatures(FeatureConfig())
        short = np.zeros(16000, dtype=np.float32)  # 1.0 s, shorter than 2.0 s
        mel = f.transform(short)
        assert mel.shape[1] == 80
        assert mel.shape[2] == 198


class TestDatasetDisjointness:
    def test_verify_disjointness_detects_overlap(self, tmp_path):
        import pandas as pd
        from hey_swift.vcm import dataset as ds

        # train and test share an audio path -> overlap must be caught.
        (tmp_path / "train_19intent.csv").write_text(
            "audio_path,intent,slot,speaker\na.wav,PLAY_MUSIC,,s1\nb.wav,STOP,,s2\n")
        (tmp_path / "test_19intent.csv").write_text(
            "audio_path,intent,slot,speaker\na.wav,PLAY_MUSIC,,s3\nc.wav,TIME,,s4\n")
        (tmp_path / "val_19intent.csv").write_text(
            "audio_path,intent,slot,speaker\nd.wav,PAUSE,,s5\n")
        (tmp_path / "holdout_19intent.csv").write_text(
            "audio_path,intent,slot,speaker\ne.wav,NEXT,,s6\n")

        result = ds.verify_disjointness(tmp_path)
        assert result["overlaps"]["train|test"] == 1
        assert result["disjoint"] is False

    def test_verify_disjointness_clean(self, tmp_path):
        import pandas as pd
        from hey_swift.vcm import dataset as ds
        for name, paths in {
            "train": ["a.wav", "b.wav"],
            "val": ["c.wav"],
            "test": ["d.wav"],
            "holdout": ["e.wav"],
        }.items():
            rows = "".join(f"{p},PLAY_MUSIC,,s\n" for p in paths)
            (tmp_path / f"{name}_19intent.csv").write_text(
                "audio_path,intent,slot,speaker\n" + rows)
        result = ds.verify_disjointness(tmp_path)
        assert result["disjoint"] is True


class TestActionRouter:
    def _router(self, tmp_path):
        from hey_swift.actions.router import ActionRouter
        actions = tmp_path / "actions.yaml"
        actions.write_text(
            "rejection:\n  oos_confidence_threshold: 0.5\n"
            "  on_oos:\n    led: error\n    speak: Sorry.\n"
            "actions:\n"
            "  PLAY_MUSIC:\n    handler: music.play\n"
            "  BRIGHTNESS:\n    handler: lights.set_brightness\n"
            "    slots: [BRIGHTNESS]\n    range: [0, 100]\n"
            "  TIMER:\n    handler: timer.start\n    slots: [TIMER]\n"
            "feedback:\n  led: success\n  speak_default: Done.\n"
        )
        vcm_cfg = tmp_path / "vcm.yaml"
        vcm_cfg.write_text(
            "schema:\n  n_intents: 19\n  oos_class: OUT_OF_SCOPE\n  n_way_head: 20\n"
            "  slot_reject: false\n  slotted_intents: [BRIGHTNESS, TIMER]\n"
            "  slot_sets:\n    BRIGHTNESS: [20, 60, 100, REJECT]\n"
            "    TIMER: [10s, 30s, 1m, REJECT]\n"
            "oos:\n  confidence_threshold: 0.5\n"
        )
        return ActionRouter(str(actions), str(vcm_cfg))

    def test_routes_simple_intent(self, tmp_path):
        r = self._router(tmp_path)
        res = r.route({"intent": "PLAY_MUSIC", "confidence": 0.9, "slots": {}})
        assert res.ok is True
        assert res.action == "music.play"

    def test_rejects_low_confidence(self, tmp_path):
        r = self._router(tmp_path)
        res = r.route({"intent": "PLAY_MUSIC", "confidence": 0.2, "slots": {}})
        assert res.ok is False
        assert res.action == "reject"

    def test_rejects_oos(self, tmp_path):
        r = self._router(tmp_path)
        res = r.route({"intent": "OUT_OF_SCOPE", "confidence": 0.9, "oos": True, "slots": {}})
        assert res.ok is False
        assert res.action == "reject"

    def test_invalid_slot_rejected(self, tmp_path):
        r = self._router(tmp_path)
        res = r.route({"intent": "BRIGHTNESS", "confidence": 0.9,
                       "slots": {"BRIGHTNESS": "50"}})  # 50 not in {20,60,100}
        assert res.ok is False
        assert res.action == "invalid_slot"

    def test_valid_slot_accepted(self, tmp_path):
        r = self._router(tmp_path)
        res = r.route({"intent": "BRIGHTNESS", "confidence": 0.9,
                       "slots": {"BRIGHTNESS": "60"}})
        assert res.ok is True
        assert "60" in res.message
