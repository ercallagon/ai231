"""Read-only ME3 dashboard: local reports, loopback serving and offline export."""

import argparse
import hashlib
import json
import math
import re
import secrets
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .artifacts import output_path

MAX_FILE = 2 * 1024 * 1024
MAX_TOTAL = 16 * 1024 * 1024
ASSETS = Path(__file__).with_name("ui")


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def bounded(value, limit=300):
    return value[:limit] if isinstance(value, str) else ""


def project_file(root, path):
    """Refuse symlinks escaping the project's ignored artifact tree."""
    resolved = path.resolve()
    if not resolved.is_relative_to((root / "var").resolve()):
        raise ValueError("artifact outside var")
    if path.is_symlink() or not resolved.is_file() or resolved.stat().st_size > MAX_FILE:
        raise ValueError("artifact unavailable or oversized")
    return resolved


def normalize(data, *, include_transcript=False):
    """Allowlisted presentation fields; never expose arbitrary JSON or history."""
    if not isinstance(data, dict):
        return None
    if (
        data.get("kind") == "dgx3-discovery"
        and data.get("mode") == "read-only-service-inspection"
        and data.get("host") == "10.153.60.233"
        and data.get("inference_submitted") is False
    ):
        checks = data.get("checks")
        if not isinstance(checks, list):
            return None
        return (
            "services",
            "reported",
            {
                "label": "DGX3",
                "inference_submitted": False,
                "checks": [
                    {"port": row["port"], "path": row["path"], "status": number(row.get("status"))}
                    for row in checks[:6]
                    if isinstance(row, dict)
                    and row.get("port") in (8000, 8001)
                    and row.get("path") in {"/health", "/openapi.json", "/v1/models"}
                ],
            },
        )
    if data.get("kind") == "application-replay" and data.get("schema_version") == 1:
        if data.get("agent_mode") not in {"mock", "external-services"} or not isinstance(
            data.get("frontend"), dict
        ):
            return None
        frontend = data["frontend"]
        mode = data.get("frontend_mode")
        if mode not in {"HPC-live CPU file inference", "Pi-local CPU file inference"}:
            return None
        _, _, payload = normalize(
            {
                **frontend,
                "mode": "Pi-local-inference"
                if mode.startswith("Pi")
                else "HPC-live-file-inference",
                "source_kind": "wav-file",
                "audio_origin": data.get("audio_origin"),
                "input_rate": data.get("input_rate"),
                "input_channels": data.get("input_channels"),
                "model_rate": data.get("model_rate"),
                "input_stats": data.get("input_stats"),
                "wake_backend": data.get("wake_backend"),
                "wake_provenance": data.get("wake_provenance"),
                "input_sha256": data.get("input_sha256"),
                "aec": "not_run",
                "limitations": data.get("limitations"),
                "run_status": "eof",
                "incomplete_utterance": data.get("unfinished_input") is True,
                "noise_suppression": "not_run",
            }
        )
        diagnostics = normalize(data.get("diagnostics"))
        payload.update(
            {
                "agent_mode": data["agent_mode"],
                "input_pipeline": "pipecat-streaming-pcm"
                if data.get("input_pipeline") == "pipecat-streaming-pcm"
                else None,
                "input_pipeline_frames": number(data.get("input_pipeline_frames")),
                "run_id": bounded(data.get("run_id"), 80),
                "session_ids": [bounded(value, 80) for value in data.get("session_ids", [])[:100]]
                if isinstance(data.get("session_ids"), list)
                else [],
                "sink_chunks": number(data.get("sink_chunks")),
                "mock_effects": number(data.get("mock_effects")),
                "pace": number(data.get("pace")),
                "endpoint_silence_ms": number(data.get("endpoint_silence_ms")),
                "turn_policy": data.get("turn_policy")
                if data.get("turn_policy") in {"silence", "smart-turn"}
                else None,
                "max_replay_schedule_lag_ms": number(data.get("max_replay_schedule_lag_ms")),
                "speech_end_to_first_audible_response_ms": None,
                "interruption_to_audio_stop_ms": None,
                "verified_physical_task_completion_ms": None,
                "backends": {
                    role: value
                    for role, value in data.get("backends", {}).items()
                    if role in ("asr", "llm", "tts") and value in {"mock", "external"}
                }
                if isinstance(data.get("backends"), dict)
                else {},
                "backend_contracts": {
                    role: value
                    for role, value in data.get("backend_contracts", {}).items()
                    if value
                    in {
                        "asr": {
                            "mock",
                            "deepgram-listen-v1",
                            "groq-whisper-v1",
                            "vllm-qwen-audio-v1",
                            "vllm-qwen3-asr-v1",
                        },
                        "llm": {"mock", "chat-completions-v1"},
                        "tts": {"mock", "deepgram-speak-v1", "vllm-qwen-tts-v1"},
                    }.get(role, set())
                }
                if isinstance(data.get("backend_contracts"), dict)
                else {},
                "turns": diagnostics[2]["turns"]
                if diagnostics and diagnostics[0] == "trace"
                else [],
            }
        )
        return "replay", "reported", payload
    if (
        data.get("kind") == "scenario-suite"
        and data.get("schema_version") == 1
        and data.get("mode") == "mock"
    ):
        rows = data.get("scenarios")
        if not isinstance(rows, list):
            return None
        return (
            "scenarios",
            "mock",
            {
                "scenarios": [
                    {
                        "name": bounded(row.get("name"), 80),
                        "passed": row.get("passed") if type(row.get("passed")) is bool else None,
                        "run_id": bounded(row.get("run_id"), 80),
                        "elapsed_seconds": number(row.get("elapsed_seconds")),
                        "error": bounded(row.get("error")),
                        "observations": {
                            key: number(value)
                            if type(value) in (int, float)
                            else bounded(value, 160)
                            for key, value in row.get("observations", {}).items()
                            if key
                            in (
                                "frontend",
                                "800ms_pause_endpoint_640ms_utterances",
                                "800ms_pause_endpoint_1024ms_utterances",
                                "relaxed_profile_extra_wait_ms",
                            )
                        }
                        if isinstance(row.get("observations"), dict)
                        else {},
                        "controlled": {
                            key: number(value) if type(value) is not bool else value
                            for key, value in row.get("controlled", {}).items()
                            if key
                            in (
                                "reasoning_barrier",
                                "sink_delay_seconds",
                                "tts_chunk_delay_seconds",
                                "asr_delay_seconds",
                                "rpc_timeout_seconds",
                                "disconnect_after_partial",
                            )
                        }
                        if isinstance(row.get("controlled"), dict)
                        else {},
                    }
                    for row in rows[:100]
                    if isinstance(row, dict)
                ],
                "limitations": bounded(data.get("limitations"), 1000),
            },
        )
    if data.get("kind") == "frontend-score" and data.get("schema_version") == 1:
        return (
            "frontend_score",
            "HPC-live" if data.get("mode") == "HPC-live-file-inference" else "reported",
            {
                name: number(data.get(name))
                for name in (
                    "wake_expected",
                    "wake_detected",
                    "wake_missed",
                    "false_activations",
                    "negative_exposure_hours",
                    "false_activations_per_hour",
                    "vad_precision",
                    "vad_recall",
                )
            }
            | {
                name: bounded(data.get(name), 1000)
                for name in ("replay_sha256", "audio_origin", "split", "limitations")
            },
        )
    if data.get("kind") == "conversation-transcript":
        if not include_transcript:
            return None
        from .transcript import normalize_transcript

        payload = normalize_transcript(data)
        evidence = payload["mode"] if payload["mode"] in {"mock", "HPC-live"} else "reported"
        return "transcript", evidence, payload
    mode = data.get("mode", "")
    if mode == "read-only-inventory":
        return (
            "inventory",
            "inventory",
            {
                "architecture": bounded(data.get("architecture")),
                "python": bounded(data.get("python")),
                "pi_detected": data.get("pi_detected") is True,
                "audio_device_directory": data.get("audio_device_directory") is True,
                "verdict": bounded(data.get("verdict")),
                "limitations": bounded(data.get("limitations"), 1000),
            },
        )
    if mode in {"HPC-live-file-inference", "Pi-local-inference"}:
        stats = data.get("input_stats")
        stats = stats if isinstance(stats, dict) else {}
        events = data.get("wake_events")
        events = events if isinstance(events, list) else []
        provenance = data.get("wake_provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        return (
            "replay",
            "HPC-live" if mode == "HPC-live-file-inference" else "reported",
            {
                "execution_mode": mode,
                "wake_events_available": isinstance(data.get("wake_events"), list),
                "source_kind": bounded(data.get("source_kind")),
                "audio_origin": bounded(data.get("audio_origin")),
                "run_status": bounded(data.get("run_status")),
                "state": bounded(data.get("state")),
                "finished": data.get("finished") is True,
                "activation_count": number(data.get("activation_count")),
                "incomplete_utterance": data.get("incomplete_utterance") is True,
                "events_truncated": data.get("events_truncated") is True,
                "vad_threshold": number(data.get("vad_threshold")),
                "vad_segments": [
                    {"start": number(e.get("start")), "end": number(e.get("end"))}
                    for e in data.get("vad_segments", [])[:1000]
                    if isinstance(e, dict)
                ]
                if isinstance(data.get("vad_segments"), list)
                else [],
                "frontend_events": [
                    {
                        "event": bounded(e.get("event")),
                        "audio_seconds": number(e.get("audio_seconds")),
                        "reason": bounded(e.get("reason")),
                    }
                    for e in data.get("frontend_events", [])[-1000:]
                    if isinstance(e, dict)
                ]
                if isinstance(data.get("frontend_events"), list)
                else [],
                "compute_ms": {
                    name: {metric: number(values.get(metric)) for metric in ("p50", "p95", "max")}
                    for name, values in data.get("compute_ms", {}).items()
                    if name in ("wake", "vad") and isinstance(values, dict)
                }
                if isinstance(data.get("compute_ms"), dict)
                else {},
                "input_rate": number(data.get("input_rate")),
                "model_rate": number(data.get("model_rate")),
                "input_channels": number(data.get("input_channels")),
                "samples": number(stats.get("samples")),
                "peak": number(stats.get("peak")),
                "rms": number(stats.get("rms")),
                "clipped_fraction": number(stats.get("clipped_fraction")),
                "wake_backend": bounded(data.get("wake_backend")),
                "wake_frames": number(data.get("wake_frames")),
                "wake_peak_score": number(data.get("wake_peak_score")),
                "wake_events": [
                    {
                        "phrase": bounded(e.get("phrase")),
                        "score": number(e.get("score")),
                        "audio_seconds": number(e.get("audio_seconds")),
                    }
                    for e in events[:100]
                    if isinstance(e, dict) and e.get("triggered") is True
                ],
                "vad_frames": number(data.get("vad_frames")),
                "voiced_frames": number(data.get("voiced_frames")),
                "vad_tail_samples_unscored": number(data.get("vad_tail_samples_unscored")),
                "wake_tail_samples_unscored": number(data.get("wake_tail_samples_unscored")),
                "input_sha256": bounded(data.get("input_sha256")),
                "wake_version": bounded(provenance.get("version")),
                "threshold": number(provenance.get("threshold")),
                "boost": number(provenance.get("boost")),
                "aec": bounded(data.get("aec")),
                "noise_suppression": bounded(data.get("noise_suppression")),
                "limitations": bounded(data.get("limitations"), 1000),
            },
        )
    if isinstance(data.get("event_counts"), dict) and isinstance(data.get("turns"), list):
        from .telemetry import PAIRS

        turns = []
        for row in data["turns"][:1000]:
            if not isinstance(row, dict):
                continue
            durations = row.get("durations_ms")
            durations = durations if isinstance(durations, dict) else {}
            turns.append(
                {
                    "run_id": bounded(row.get("run_id"), 80),
                    "session_id": bounded(row.get("session_id"), 80),
                    "turn_id": number(row.get("turn_id")),
                    "mode": bounded(row.get("mode")),
                    "outcome": bounded(row.get("outcome")),
                    "asr_input_duration_ms": number(row.get("asr_input_duration_ms")),
                    "asr_input_incremental": row.get("asr_input_incremental") is True,
                    "asr_partial_observed": row.get("asr_partial_observed") is True,
                    "durations_ms": {
                        name: [n for value in values[:100] if (n := number(value)) is not None]
                        for name, values in durations.items()
                        if name in PAIRS and isinstance(values, list)
                    },
                }
            )
        # A label in a file is not physical verification. Unknown/mixed modes stay explicit.
        modes = {t["mode"] for t in turns}
        evidence = "mock" if modes == {"mock"} else "reported"
        return (
            "trace",
            evidence,
            {
                "turns": turns,
                "event_count": number(
                    sum(n for v in data["event_counts"].values() if (n := number(v)) is not None)
                ),
                "limitations": bounded(data.get("limitations"), 1000),
            },
        )
    if mode == "mock" and isinstance(data.get("sink_chunks"), int):
        return (
            "conversation",
            "mock",
            {
                "run_id": bounded(data.get("run_id"), 80),
                "session_id": bounded(data.get("session_id"), 80),
                "sink_chunks": data["sink_chunks"],
                "protocols": {
                    name: bounded(v)
                    for name, v in data.get("protocols", {}).items()
                    if name in ("asr", "tts")
                },
                "limitations": bounded(data.get("limitations"), 1000),
            },
        )
    if (
        "positive_trials_detected" in data
        and "negative_trials_triggered" in data
        and data.get("provider") == "cpu"
    ):
        return (
            "wake_trial",
            "HPC-live",
            {
                name: number(data.get(name))
                for name in (
                    "positive_trials",
                    "positive_trials_detected",
                    "negative_trials",
                    "negative_trials_triggered",
                )
            }
            | {
                "limitations": "Synthetic development cases; correlated variants, not real-speaker accuracy."
            },
        )
    return None


def snapshot(root):
    root = root.resolve()
    reports, warnings = [], []
    results = root / "var/results"
    candidates = sorted(results.glob("*.json"))[:200] if results.is_dir() else []
    current = results / "ui-transcript.json"
    if current.is_file():
        candidates = [current] + [p for p in candidates if p != current][:199]
    candidates.extend(sorted(results.glob("*.conf"))[:20])
    trial = root / "var/experiments/002-hey-swift-keyword-spotting/results.json"
    if trial.is_file():
        candidates.append(trial)
    consumed = 0
    for path in candidates:
        try:
            resolved = project_file(root, path)
            if consumed + resolved.stat().st_size > MAX_TOTAL:
                raise ValueError("snapshot read budget exceeded")
            raw = resolved.read_bytes()
            consumed += len(raw)
            if path.suffix == ".conf":
                text = raw.decode()
                if not text.startswith("# REVIEW ARTIFACT ONLY:"):
                    continue
                normalized = (
                    "configuration",
                    "review",
                    {
                        "status": "Review artifact only; not installed or applied",
                        "targets": re.findall(r'target.object = "([A-Za-z0-9_.:-]+)"', text)[:2],
                        "aec": "not_run",
                    },
                )
            else:
                normalized = normalize(json.loads(raw), include_transcript=path == current)
            if normalized is None:
                continue
            kind, evidence, payload = normalized
            reports.append(
                {
                    "id": hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:16],
                    "source": str(path.relative_to(root)),
                    "kind": kind,
                    "evidence": evidence,
                    "modified_utc": datetime.fromtimestamp(
                        resolved.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "data": payload,
                }
            )
        except (ValueError, OSError, TypeError, AttributeError, RecursionError):
            warnings.append(
                {
                    "source": str(path.relative_to(root)),
                    "reason": "Unreadable, invalid or unsupported report",
                }
            )
    reports.sort(key=lambda r: r["modified_utc"], reverse=True)
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reports": reports,
        "warnings": warnings,
        "status": {
            "milestone": "M1 · Audio readiness",
            "pi_verification": "Not established",
            "runtime": "Report viewer",
            "controls": "Read only",
        },
    }


def render(data, *, live=False):
    nonce = secrets.token_urlsafe(24)
    policy = (
        f"default-src 'none'; script-src 'nonce-{nonce}' https://sdk.scdn.co; "
        f"style-src 'nonce-{nonce}'; connect-src 'self' https://api.spotify.com "
        "https://accounts.spotify.com https://spclient.wg.spotify.com wss://*.spotify.com; "
        "img-src data: https://i.scdn.co; media-src blob: https://*.scdn.co "
        "https://*.spotifycdn.com https://*.spotify.com; worker-src blob:; "
        "frame-src https://sdk.scdn.co; base-uri 'none'; form-action 'none'"
        if live
        else f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
        "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'"
    )
    encoded = (
        json.dumps({**data, "live": live}, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    return (
        (ASSETS / "dashboard.html")
        .read_text()
        .replace("__NONCE__", nonce)
        .replace("__CSP__", policy)
        .replace("__STYLE__", (ASSETS / "dashboard.css").read_text())
        .replace(
            "__SCRIPT__",
            (ASSETS / "audio_prebuffer.js").read_text()
            + "\n"
            + (ASSETS / "dashboard.js").read_text(),
        )
        .replace("__DATA__", encoded)
    )


def render_audio_probe():
    """Render the local, no-upload laptop AEC recorder with a page-local nonce."""
    nonce = secrets.token_urlsafe(24)
    policy = (
        f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
        "media-src blob:; connect-src 'none'; img-src data:; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    )
    body = (ASSETS / "audio_probe.html").read_text().replace("__NONCE__", nonce)
    return body, policy


def stream_records(root, *, include_ui=True):
    """One JSON record per line for a terminal client; never open a listener."""
    data = snapshot(root)
    serialized = json.dumps(data, allow_nan=False)
    if include_ui:
        yield {
            "type": "bootstrap",
            "html": render(data),
            "snapshot": data,
            "snapshot_json": serialized,
        }
    else:
        yield {"type": "snapshot", "snapshot": data, "snapshot_json": serialized}
    while True:
        data = snapshot(root)
        yield {
            "type": "snapshot",
            "snapshot": data,
            "snapshot_json": json.dumps(data, allow_nan=False),
        }


def stream(root, interval):
    try:
        for record in stream_records(root):
            print(json.dumps(record, allow_nan=False), flush=True)
            time.sleep(interval)
    except (BrokenPipeError, KeyboardInterrupt):
        # Closing SSH/the terminal ends the feed, without installing a service.
        pass


def handler(root, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Avoid writing request URLs or user-controlled content into logs.

        def do_GET(self):
            # SSH/VS Code may forward a different client port. Keep the hostname
            # strictly loopback while accepting that client's valid port.
            hosts = self.headers.get_all("Host") or []
            authority = (
                re.fullmatch(r"(?:127\.0\.0\.1|localhost):([0-9]{1,5})", hosts[0])
                if len(hosts) == 1
                else None
            )
            if not authority or not 1 <= int(authority.group(1)) <= 65535:
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            probe_policy = None
            if path == "/":
                content = render(snapshot(root), live=True).encode()
                content_type = "text/html; charset=utf-8"
            elif path == "/audio-probe":
                probe, probe_policy = render_audio_probe()
                content = probe.encode()
                content_type = "text/html; charset=utf-8"
            elif path == "/api/snapshot":
                content = json.dumps(snapshot(root), allow_nan=False).encode()
                content_type = "application/json; charset=utf-8"
            elif path == "/api/voice/capabilities":
                content = b'{"available":false}'
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            if probe_policy:
                self.send_header("Content-Security-Policy", probe_policy)
                self.send_header("Permissions-Policy", "microphone=(self)")
            self.end_headers()
            self.wfile.write(content)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="self-contained offline HTML snapshot")
    export.add_argument("--output", type=output_path, default="var/results/dashboard.html")
    serve = commands.add_parser("serve", help="loopback-only read-only viewer")
    serve.add_argument("--port", type=int, default=8765)
    feed = commands.add_parser("stream", help="sanitized JSONL report feed over stdout/SSH")
    feed.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "export":
        args.output.write_text(render(snapshot(root)))
        print(f"Offline dashboard: {args.output}")
    elif args.command == "stream":
        if not math.isfinite(args.interval) or not 1 <= args.interval <= 60:
            parser.error("stream interval must be 1..60 seconds")
        stream(root, args.interval)
    else:
        if not 1024 <= args.port <= 65535:
            parser.error("port must be 1024..65535")
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler(root, args.port)) as server:
            print(f"ME3 dashboard: http://127.0.0.1:{args.port} (read only)", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
