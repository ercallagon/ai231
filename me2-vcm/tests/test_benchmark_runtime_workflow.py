"""Benchmark logging/endpoint overrides must not change scores or frozen models."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("endpoint", [None, "800", "1200"])
def test_launcher_preserves_filtered_log_and_keeps_capture_diagnostics(monkeypatch, tmp_path, endpoint):
    module = script("run_benchmark_assistant")
    lines = [json.dumps(x) + "\n" for x in [dict(status="ready"), dict(event="wake"),
        dict(event="benchmark_decision", intent="OUT_OF_SCOPE", infer_ms=0, audio_ms=0),
        dict(event="command", capture=dict(reason="speech_start_timeout"))]]
    commands = []
    def start(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=iter(lines), wait=lambda: 0, poll=lambda: 0)
    monkeypatch.setattr(module.subprocess, "Popen", start)
    argv = ["benchmark", "--profile", "champion", "--log-dir", str(tmp_path)]
    if endpoint:
        argv += ["--end-silence-ms", endpoint]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    command = commands[0]
    assert "--offline" in command and "--live" not in command and "--speak" not in command
    if endpoint:
        assert command[command.index("--end-silence-ms")+1] == str(float(endpoint))
    else:
        assert "--end-silence-ms" not in command
    records = [json.loads(x) for x in next(tmp_path.glob("*.log")).read_text().splitlines()]
    assert [x["event"] for x in records] == ["wake", "benchmark_decision"]
    assert next(tmp_path.glob("*.diagnostics")).read_text() == "".join(lines)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_endpoint_stops_before_start(monkeypatch, tmp_path, value):
    module = script("run_benchmark_assistant")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--log-dir", str(tmp_path), "--end-silence-ms", value])
    with pytest.raises(SystemExit):
        module.main()
    assert not list(tmp_path.iterdir())


def test_audit_preserves_early_and_zero_audio_failures(tmp_path):
    module = script("review_class_benchmark")
    row = dict(kind="wake", order="1", true_intent="TIME", pred_intent="OUT_OF_SCOPE",
        correct_command="False", is_synthetic="False", n_command_events="2", infer_ms="0",
        audio_ms="0", latency_s="-1", pred_raw=json.dumps(dict(accepted=False,
        rejection_reason="no_sustained_speech", speech_end_to_decision_ms=None)))
    with (tmp_path/"trials.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    before=(tmp_path/"trials.csv").read_bytes()
    result=module.review(tmp_path)
    assert result["early_decision_trials"]["n"] == 1
    assert result["multiple_decision_trials"]["n"] == 1
    assert result["zero_audio_decisions"] == 1 and result["missing_timing_fields"] == 0
    assert result["preprocessing_and_inference_ms_processed_only"]["n"] == 0
    assert result["supported"]["all"] == {"n": 1,"accepted_correct":0}
    assert before == (tmp_path/"trials.csv").read_bytes()


def test_wrapper_interrupt_reaps_child_without_traceback(monkeypatch, tmp_path):
    module = script("run_benchmark_assistant")
    stopped = []
    def lines():
        raise KeyboardInterrupt()
        yield ""
    child = SimpleNamespace(stdout=lines(), poll=lambda: None,
        terminate=lambda: stopped.append(True), wait=lambda **kwargs: 0)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--log-dir", str(tmp_path)])
    assert module.main() == 130 and stopped == [True]


def test_failed_capture_records_signal_evidence_without_lowering_gate(monkeypatch):
    import numpy as np
    from hey_swift.integration import e2e_loop as loop
    import hey_swift.audio.capture as capture
    clock = [0.]
    class Stream:
        def __init__(self, **kwargs): self.i = 0
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, n):
            level = .1 if self.i < 4 else .05
            self.i += 1
            clock[0] += .04
            return np.full(n, level, np.float32).tobytes(), False
    class Resampler:
        def __init__(self, *args): pass
        def push(self, x): return x
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(RawInputStream=Stream))
    monkeypatch.setattr(capture, "ContinuousResampler", Resampler)
    monkeypatch.setattr(loop.time, "monotonic", lambda: clock[0])
    args=SimpleNamespace(vad_frame_ms=40, command_seconds=6, min_command_seconds=.5,
        speech_start_ms=120, pre_roll_ms=200, end_silence_ms=800,
        speech_start_timeout=.4, vad_noise_seconds=.16, vad_margin_db=10, vad_min_speech_db=-45)
    _, meta=loop._capture_command_vad(0, args, rate=16000)
    assert meta["reason"] == "speech_start_timeout" and not meta["speech_started"]
    assert meta["noise_floor_db"] == -20.0 and meta["threshold_db"] == -10.0
    assert meta["peak_input_db"] == -26.02 and meta["peak_onset_sustain_ms"] == 0
