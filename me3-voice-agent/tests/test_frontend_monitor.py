import json
from argparse import Namespace

import pytest

from me3_voice.audio import SileroVAD
from me3_voice.frontend import FrontendMonitor
from me3_voice.readiness import publish, replay
from me3_voice.wake import WakeDetection, WakeGate


class Detector:
    frame_samples = 320
    backend = "test"
    provenance = {}

    def __init__(self, at=(640,)):
        self.at, self.samples = set(at), 0

    def detect(self, pcm):
        self.samples += 320
        return WakeDetection("HEY_SWIFT") if self.samples in self.at else None


class VAD:
    def voiced(self, pcm):
        return pcm != bytes(1024)


def run(pcm, chunk):
    # Two wake sessions; follow-up speech requires no second wake.
    gate = WakeGate(Detector((640, 6400)), cooldown_samples=0)
    monitor = FrontendMonitor(gate, VAD(), silence_frames=2, idle_frames=3)
    for start in range(0, len(pcm), chunk):
        monitor.push(pcm[start : start + chunk])
    monitor.finish()
    return monitor.report()


def test_wake_vad_followup_timeout_rearm_and_callback_invariance():
    voiced, quiet = b"\1\0" * 512, bytes(1024)
    pcm = voiced * 2 + quiet * 2 + voiced * 2 + quiet * 6 + voiced * 2 + quiet * 5
    tiny, whole = run(pcm, 14), run(pcm, 16000)
    for key in ("frontend_events", "vad_segments", "wake_events", "state", "activation_count"):
        assert tiny[key] == whole[key]
    names = [e["event"] for e in whole["frontend_events"]]
    assert names == [
        "wake",
        "utterance_start",
        "endpoint",
        "utterance_start",
        "endpoint",
        "session_timeout",
        "wake",
        "utterance_start",
        "endpoint",
        "session_timeout",
    ]
    assert whole["activation_count"] == 2 and whole["state"] == "wake_listening"
    assert whole["frontend_events"][1]["pre_roll_start_seconds"] == 0
    ends = [e for e in whole["frontend_events"] if e["event"] == "endpoint"]
    assert ends[0]["audio_seconds"] == 2048 / 16000
    assert all(e["reason"] == "silence" for e in ends)


def test_tail_incomplete_speech_and_finish_are_explicit():
    monitor = FrontendMonitor(WakeGate(Detector()), VAD())
    monitor.push(b"\1\0" * 1234)
    monitor.finish()
    monitor.finish()
    report = monitor.report()
    assert report["incomplete_utterance"]
    assert report["vad_tail_samples_unscored"] == 210
    assert report["wake_tail_samples_unscored"] == 274
    assert report["vad_segments"] == [{"start": 0, "end": 1024 / 16000, "open": False}]
    with pytest.raises(ValueError, match="finished"):
        monitor.push(bytes(1024))


def test_vad_only_never_activates_a_session_and_bounded_event_retention():
    monitor = FrontendMonitor(None, VAD())
    for _ in range(1100):
        monitor.push(b"\1\0" * 512 + bytes(1024))
    monitor.finish()
    report = monitor.report()
    assert report["state"] == "vad_only" and report["activation_count"] == 0
    assert report["vad_segments_truncated"] and len(report["vad_segments"]) == 1000
    assert not report["wake_events"]
    assert report["compute_ms"]["vad"]["count_window"] == 2200


def test_atomic_publication_keeps_prior_report_on_invalid_json(tmp_path):
    output = tmp_path / "report.json"
    publish(output, {"state": "listening"})
    with pytest.raises(ValueError):
        publish(output, {"bad": float("nan")})
    assert json.loads(output.read_text()) == {"state": "listening"}
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("threshold", [0, 1, -1, float("nan"), float("inf")])
def test_invalid_vad_threshold_fails_before_model_load(threshold):
    with pytest.raises(ValueError, match="confidence"):
        SileroVAD(threshold)


def test_stdin_rate_conversion_and_error_publication(tmp_path, monkeypatch):
    import io
    import sys
    import me3_voice.audio
    import me3_voice.readiness

    monkeypatch.setattr(me3_voice.audio, "SileroVAD", lambda _: VAD())
    monkeypatch.setattr(
        me3_voice.readiness, "inventory", lambda: {"pi_detected": True, "architecture": "aarch64"}
    )
    args = Namespace(
        command="monitor",
        wav=None,
        output=tmp_path / "report.json",
        processed_wav=None,
        wake_model=None,
        mel_model=None,
        embedding_model=None,
        rate=48000,
        channels=2,
        max_seconds=1,
        vad_threshold=0.7,
        audio_origin="synthetic",
    )
    monkeypatch.setattr(sys, "stdin", Namespace(buffer=io.BytesIO(bytes(48000 * 4))))
    report = replay(args)
    assert report["mode"] == "Pi-local-inference"  # Host execution, never acoustic qualification.
    assert report["run_status"] == "limit_reached"
    assert report["source_kind"] == "pcm-stdin"
    assert report["vad_frames"] == 31 and report["vad_tail_samples_unscored"] == 128
    assert report["processing_rtf"] is None
    monkeypatch.setattr(sys, "stdin", Namespace(buffer=io.BytesIO(b"\0")))
    with pytest.raises(ValueError, match="PCM16"):
        replay(args)
    assert json.loads(args.output.read_text())["run_status"] == "error"
