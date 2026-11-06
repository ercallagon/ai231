import json


from me3_voice.dashboard import normalize, render, snapshot, stream_records


def save(root, name, data):
    path = root / "var/results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def test_dashboard_allowlist_preserves_missing_values_and_excludes_payloads(tmp_path):
    save(
        tmp_path,
        "replay.json",
        {
            "mode": "HPC-live-file-inference",
            "input_rate": 48000,
            "model_rate": 16000,
            "wake_backend": "sherpa-onnx-kws",
            "wake_peak_score": None,
            "input_stats": {"clipped_fraction": 0, "peak": float("nan")},
            "wake_events": [
                {
                    "triggered": True,
                    "phrase": "HEY_SWIFT",
                    "score": None,
                    "audio_seconds": 1.76,
                    "secret": "private",
                }
            ],
            "credentials": "private",
            "transcript": "private",
            "aec": "not_run",
        },
    )
    save(tmp_path, "demo.json", {"mode": "mock", "sink_chunks": 8, "history": ["private"]})
    save(tmp_path, "unrelated.json", {"credentials": "private"})
    result = snapshot(tmp_path)
    assert len(result["reports"]) == 2
    encoded = json.dumps(result, allow_nan=False)
    assert "private" not in encoded and "credentials" not in encoded
    replay = next(r for r in result["reports"] if r["kind"] == "replay")["data"]
    assert replay["input_rate"] == 48000 and replay["peak"] is None
    assert replay["wake_peak_score"] is None and replay["wake_events"][0]["score"] is None
    assert replay["clipped_fraction"] == 0
    assert result["status"]["pi_verification"] == "Not established"


def test_bad_or_external_report_does_not_break_other_evidence(tmp_path):
    save(
        tmp_path,
        "inventory.json",
        {
            "mode": "read-only-inventory",
            "pi_detected": True,
            "audio_device_directory": True,
            "verdict": "qualified",
        },
    )
    (tmp_path / "var/results/broken.json").write_text("{broken")
    outside = tmp_path / "external.json"
    outside.write_text('{"secret":"never serve this"}')
    (tmp_path / "var/results/linked.json").symlink_to(outside)
    result = snapshot(tmp_path)
    assert len(result["reports"]) == 1 and len(result["warnings"]) == 2
    # Host inventory and an arbitrary verdict string never promote acoustic qualification.
    assert result["status"]["pi_verification"] == "Not established"
    assert "never serve this" not in json.dumps(result)


def test_unknown_trace_mode_is_not_silently_labelled_pi_verified():
    kind, evidence, result = normalize(
        {
            "event_counts": {"x": 1},
            "turns": [
                {"mode": "Pi-verified", "turn_id": 1, "durations_ms": {"asr_request_ms": [2]}}
            ],
        }
    )
    assert kind == "trace" and evidence == "reported"
    assert result["turns"][0]["durations_ms"]["asr_request_ms"] == [2]


def test_export_is_self_contained_and_script_payload_is_escaped(tmp_path):
    save(
        tmp_path,
        "replay.json",
        {
            "mode": "HPC-live-file-inference",
            "wake_backend": "</script><script>window.pwned=true</script>",
        },
    )
    html = render(snapshot(tmp_path))
    assert "window.pwned=true</script>" not in html
    assert "\\u003c/script>" in html
    assert "__SCRIPT__" not in html and "__STYLE__" not in html
    assert "Content-Security-Policy" in html and "nonce-" in html
    assert "connect-src 'self'; img-src data:" in html
    assert (
        "https://sdk.scdn.co"
        not in html.split('Content-Security-Policy" content="', 1)[1].split('"', 1)[0]
    )


def test_live_ui_policy_allows_spotify_player_only_for_live_dashboard(tmp_path):
    html = render(snapshot(tmp_path), live=True)
    policy = html.split('Content-Security-Policy" content="', 1)[1].split('"', 1)[0]
    assert "https://sdk.scdn.co" in policy
    assert "frame-src https://sdk.scdn.co;" in policy
    assert "https://api.spotify.com" in policy
    assert "wss://*.spotify.com" in policy


def test_empty_workspace_has_no_fabricated_results(tmp_path):
    result = snapshot(tmp_path)
    assert not result["reports"] and not result["warnings"]
    assert result["status"]["controls"] == "Read only"
    assert "pi_verified" not in result


def test_aec_review_artifact_does_not_count_as_executed_aec(tmp_path):
    path = tmp_path / "var/results/pipewire-echo-cancel-review.conf"
    path.parent.mkdir(parents=True)
    path.write_text('# REVIEW ARTIFACT ONLY: not installed\ntarget.object = "physical-node"')
    result = snapshot(tmp_path)
    report = result["reports"][0]
    assert report["evidence"] == "review"
    assert report["data"]["aec"] == "not_run"
    assert report["data"]["targets"] == ["physical-node"]


def test_terminal_stream_bootstraps_same_ui_then_updates_without_payload_leaks(tmp_path):
    save(tmp_path, "demo.json", {"mode": "mock", "sink_chunks": 8, "history": ["private"]})
    records = stream_records(tmp_path)
    first = next(records)
    assert first["type"] == "bootstrap" and "Connect report feed" in first["html"]
    assert json.loads(first["snapshot_json"]) == first["snapshot"]
    # The embedded app includes a static Spotify scope name containing the
    # word "private"; inspect the report payload rather than application code.
    assert "private" not in first["snapshot_json"]
    assert "credentials" not in first["snapshot_json"]
    save(tmp_path, "demo.json", {"mode": "mock", "sink_chunks": 12, "history": ["private"]})
    second = next(records)
    assert second["type"] == "snapshot" and "html" not in second
    assert second["snapshot"]["reports"][0]["data"]["sink_chunks"] == 12
    assert json.loads(second["snapshot_json"]) == second["snapshot"]
    records.close()


def test_terminal_stream_can_be_consumed_as_one_json_record_per_line(tmp_path):
    import os
    import subprocess
    import sys

    # A finite consumer verifies framing without introducing a persistent service.
    script = (
        "import json; from pathlib import Path; "
        "from me3_voice.dashboard import stream_records; "
        "r=stream_records(Path.cwd()); "
        "print(json.dumps(next(r)),flush=True); print(json.dumps(next(r)),flush=True)"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["type"] for r in rows] == ["bootstrap", "snapshot"]
    assert rows[1]["snapshot"]["reports"] == []


def test_loopback_http_read_only_contract(tmp_path):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    from me3_voice.dashboard import handler

    # Construct with the actual ephemeral port so Host validation is exercised.
    server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *args: None)
    server.RequestHandlerClass = handler(tmp_path, server.server_port)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", "/api/snapshot")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["reports"] == []
        connection.request("GET", "/api/voice/capabilities")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"available": False}
        for host in ("127.0.0.1:18765", "localhost:18765"):
            connection.request("GET", "/api/snapshot", headers={"Host": host})
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["reports"] == []
        for host in (
            "127.0.0.1:0",
            "localhost:99999",
            "evil@localhost:18765",
            "127.0.0.1.evil:18765",
        ):
            connection.request("GET", "/", headers={"Host": host})
            response = connection.getresponse()
            assert response.status == 403
            response.read()
        connection.request("GET", "/var/results/secret.json")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        connection.request("GET", "/", headers={"Host": "external.example"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.request("POST", "/api/snapshot", body="run hardware")
        response = connection.getresponse()
        assert response.status == 501
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_audio_presence_and_execution_labels_do_not_invent_measurements():
    _, evidence, view = normalize({"mode": "Pi-local-inference", "wake_backend": "not_configured"})
    assert evidence == "reported" and view["execution_mode"] == "Pi-local-inference"
    assert view["wake_events_available"] is False
    assert view["activation_count"] is None and view["voiced_frames"] is None
    assert view["clipped_fraction"] is None
    _, evidence, explicit = normalize({"mode": "HPC-live-file-inference", "wake_events": []})
    assert evidence == "HPC-live" and explicit["wake_events_available"] is True


def test_provider_and_connection_claims_are_not_admitted_from_report_extras(tmp_path):
    save(
        tmp_path,
        "replay.json",
        {
            "mode": "HPC-live-file-inference",
            "api_key": "private-value",
            "runtime_connected": True,
            "pi_verified": True,
            "provider": {"endpoint": "https://private-value", "token": "private-value"},
        },
    )
    data = snapshot(tmp_path)
    encoded = json.dumps(data)
    assert "private-value" not in encoded and "runtime_connected" not in encoded
    assert data["status"]["pi_verification"] == "Not established"


def test_mixed_application_and_scenario_reports_preserve_evidence_without_payloads():
    report = {
        "kind": "application-replay",
        "schema_version": 1,
        "agent_mode": "mock",
        "frontend_mode": "HPC-live CPU file inference",
        "run_id": "r",
        "session_ids": ["s"],
        "input_rate": 48000,
        "model_rate": 16000,
        "frontend": {"wake_events": [], "vad_frames": 10},
        "backends": {"asr": "mock", "llm": "mock", "tts": "mock", "api_key": "private-value"},
        "backend_contracts": {"asr": "vllm-qwen3-asr-v1", "llm": "private-value"},
        "diagnostics": {
            "event_counts": {"turn_completed": 1},
            "turns": [
                {
                    "turn_id": 1,
                    "session_id": "s",
                    "run_id": "r",
                    "mode": "mock",
                    "outcome": "completed",
                    "durations_ms": {"asr_request_ms": [12]},
                }
            ],
        },
        "history": ["private-value"],
        "credentials": "private-value",
    }
    kind, evidence, data = normalize(report)
    assert kind == "replay" and evidence == "reported"
    assert data["execution_mode"] == "HPC-live-file-inference" and data["agent_mode"] == "mock"
    assert data["turns"][0]["run_id"] == "r" and data["turns"][0]["session_id"] == "s"
    assert "private-value" not in json.dumps(data)
    assert data["backend_contracts"] == {"asr": "vllm-qwen3-asr-v1"}


def test_timing_detail_projection_keeps_only_bounded_diagnostic_fields():
    kind, evidence, data = normalize(
        {
            "event_counts": {"turn_completed": 1},
            "turns": [
                {
                    "run_id": "r",
                    "session_id": "s",
                    "turn_id": 1,
                    "mode": "mock",
                    "outcome": "completed",
                    "asr_input_duration_ms": 1000,
                    "asr_input_incremental": True,
                    "asr_partial_observed": True,
                    "durations_ms": {
                        "tts_render_total_ms": [250],
                        "llm_request_total_ms": [400],
                        "secret_duration": [123],
                    },
                    "transcript": "private-value",
                }
            ],
        }
    )
    assert kind == "trace" and evidence == "mock"
    row = data["turns"][0]
    assert row["asr_input_duration_ms"] == 1000
    assert row["asr_input_incremental"] and row["asr_partial_observed"]
    assert row["durations_ms"] == {
        "tts_render_total_ms": [250],
        "llm_request_total_ms": [400],
    }
    assert "private-value" not in json.dumps(data)
    scenario = {
        "kind": "scenario-suite",
        "schema_version": 1,
        "mode": "mock",
        "scenarios": [
            {
                "name": "test",
                "passed": True,
                "elapsed_seconds": None,
                "controlled": {"sink_delay_seconds": 60, "api_key": "private-value"},
            }
        ],
    }
    kind, evidence, data = normalize(scenario)
    assert kind == "scenarios" and evidence == "mock"
    assert data["scenarios"][0]["elapsed_seconds"] is None
    assert "private-value" not in json.dumps(data)


def test_dgx3_inspection_is_read_only_and_never_exports_credentials_or_schema():
    report = {
        "kind": "dgx3-discovery",
        "mode": "read-only-service-inspection",
        "host": "10.153.60.233",
        "inference_submitted": False,
        "api_key": "private-value",
        "openapi": {"secret": "private-value"},
        "checks": [
            {
                "port": 8000,
                "path": "/v1/models",
                "status": 401,
                "headers": {"Authorization": "private-value"},
            },
            {"port": 8000, "path": "/load", "status": 200},
        ],
    }
    kind, evidence, payload = normalize(report)
    assert kind == "services" and evidence == "reported"
    assert len(payload["checks"]) == 1 and payload["checks"][0]["status"] == 401
    assert "private-value" not in json.dumps(payload)
    report["inference_submitted"] = True
    assert normalize(report) is None
