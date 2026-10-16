import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_runtime_latency import summarize


def write_log(path, records):
    path.write_text("startup text\n" + "\n".join(json.dumps(r) for r in records))
    return path


def test_summary_counts_rejections_missing_endpoints_and_avoids_duplicate_events(tmp_path):
    path = write_log(tmp_path / "live.log", [
        dict(event="wake"),
        dict(event="benchmark_decision", speech_end_to_decision_ms=1000, accepted=True,
             audio_ms=500, infer_ms=100, threads=1),
        dict(event="command", latency=dict(speech_end_to_decision_seconds=1)),
        dict(event="benchmark_decision", speech_end_to_decision_ms=2000, accepted=False,
             audio_ms=700, infer_ms=200, threads=1),
        dict(event="benchmark_decision", speech_end_to_decision_ms=None, accepted=False,
             audio_ms=0, infer_ms=0, rejection_reason="no_sustained_speech", threads=1),
    ])
    result = summarize([path])
    assert result["total_decisions"] == 3
    assert result["all_speech_decisions"]["n"] == 2
    assert result["all_speech_decisions"]["p50_ms"] == 1500
    assert result["all_speech_decisions"]["p95_ms"] == 1950
    assert result["accepted_speech_decisions"]["n"] == 1
    assert result["rejected_speech_decisions"]["n"] == 1
    assert result["missing_endpoint_or_legacy_timing"] == 1
    assert result["rejection_reason_counts"] == {"no_sustained_speech": 1}
    assert result["preprocessing_and_inference_processed_only"]["n"] == 2
    assert result["intra_threads"] == [1]


def test_command_logs_and_invalid_or_legacy_timings(tmp_path):
    path = write_log(tmp_path / "commands.log", [
        dict(event="command", latency=dict(speech_end_to_decision_seconds=1.25),
             prediction=dict(accepted=True)),
        dict(event="command", latency={}),
        dict(event="command", latency=dict(speech_end_to_decision_seconds=-1)),
        dict(event="command", latency=dict(speech_end_to_decision_seconds=float("nan"))),
    ])
    result = summarize([path])
    assert result["total_decisions"] == 4
    assert result["all_speech_decisions"]["p95_ms"] == 1250
    assert result["missing_endpoint_or_legacy_timing"] == 1
    assert result["invalid_latency"] == 2
    assert result["rejected_speech_decisions"]["p95_ms"] is None


def test_cli_writes_metrics_and_refuses_overwrite(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_log(logs / "trial.log", [dict(event="benchmark_decision",
        speech_end_to_decision_ms=1200, accepted=True, threads=1)])
    output = tmp_path / "metrics.json"
    command = [sys.executable, str(ROOT / "scripts/summarize_runtime_latency.py"),
               str(logs), "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(output.read_text())["all_speech_decisions"]["p95_ms"] == 1200
    before = output.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert "Output exists" in second.stderr
    assert output.read_bytes() == before


def test_cli_rejects_logs_without_decisions(tmp_path):
    path = write_log(tmp_path / "empty.log", [dict(event="wake")])
    output = tmp_path / "metrics.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/summarize_runtime_latency.py"),
        str(path), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.exists()
