import json

import pytest

from me3_voice.telemetry import read_trace, summarize, write_trace


def event(name, time, **kwargs):
    return {
        "schema_version": 1,
        "event": name,
        "monotonic": time,
        "run_id": "run",
        "session_id": "session",
        "turn_id": 1,
        "clock_id": "clock",
        "mode": "mock",
        **kwargs,
    }


def test_overlapping_stages_have_distinct_boundaries_and_no_invented_acoustics():
    report = summarize(
        [
            event("endpoint_submitted", 1),
            event("asr_started", 1),
            event("asr_final", 1.2),
            event("llm_started", 1.2),
            event("llm_first_segment", 1.4),
            event("tts_started", 1.4, segment_id=1),
            event("tts_first_chunk", 1.5, segment_id=1),
            event("sink_write_started_mock", 1.6),
            event("tts_started", 1.7, segment_id=2),
            event("tts_first_chunk", 1.8, segment_id=2),
            event("turn_completed", 2),
        ]
    )
    row = report["turns"][0]
    assert row["outcome"] == "completed"
    assert row["durations_ms"]["endpoint_to_sink_write_mock_ms"] == pytest.approx([600])
    assert row["durations_ms"]["tts_first_chunk_ms"] == pytest.approx([100, 100])
    assert "interrupt_to_sink_stop_mock_ms" in row["unavailable_metrics"]
    assert report["acoustic_response_ms"] is None and report["acoustic_barge_in_ms"] is None


def test_last_voiced_to_endpoint_is_paired_per_turn_and_not_inferred_when_missing():
    report = summarize(
        [
            event("last_voiced_frame_consumed_file", 1),
            event("endpoint_submitted", 2.024),
            event("endpoint_submitted", 3, turn_id=2),
        ]
    )
    assert report["turns"][0]["durations_ms"]["last_voiced_frame_to_endpoint_ms"] == pytest.approx(
        [1024]
    )
    assert "last_voiced_frame_to_endpoint_ms" in report["turns"][1]["unavailable_metrics"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "other"),
        ("run_id", "other"),
        ("turn_id", 2),
        ("clock_id", "other"),
        ("mode", "HPC-live"),
        ("segment_id", 2),
    ],
)
def test_cannot_pair_stages_across_correlation_boundaries(field, value):
    report = summarize(
        [
            event("tts_started", 1, segment_id=1),
            event("tts_first_chunk", 2, **{"segment_id": 1, field: value}),
        ]
    )
    assert not any(row["durations_ms"] for row in report["turns"])


def test_incomplete_and_failed_turns_remain_visible():
    report = summarize(
        [
            event("asr_started", 1),
            event("interrupt_requested", 1.1),
            event("playback_stopped_mock", 1.2),
            event("asr_started", 2, turn_id=2),
            event("turn_error", 2.5, turn_id=2),
        ]
    )
    assert [r["outcome"] for r in report["turns"]] == ["interrupted", "error"]
    assert "asr_request_ms" in report["turns"][0]["unavailable_metrics"]
    assert report["turns"][0]["durations_ms"]["interrupt_to_sink_stop_mock_ms"] == pytest.approx(
        [100]
    )


def test_stage_detail_uses_observed_boundaries_and_preserves_missing_completion():
    report = summarize(
        [
            event("asr_started", 1),
            event("asr_stream_opened", 1.01),
            event("asr_partial", 1.2),
            event("endpoint_submitted", 1.5, audio_samples=16000),
            event("asr_final", 1.8),
            event("llm_request_started", 1.8),
            event("llm_first_delta", 2),
            event("llm_request_finished", 2.3),
            event("tts_started", 2.3, segment_id=1),
            event("tts_first_chunk", 2.5, segment_id=1),
            event("tts_finished", 2.9, segment_id=1),
            event("tts_started", 3, segment_id=2),
            event("interrupt_requested", 3.1),
        ]
    )
    row = report["turns"][0]
    assert row["asr_input_duration_ms"] == 1000
    assert row["asr_input_incremental"] is True
    assert row["asr_partial_observed"] is True
    assert row["durations_ms"]["llm_request_to_first_delta_ms"] == pytest.approx([200])
    assert row["durations_ms"]["llm_request_total_ms"] == pytest.approx([500])
    assert row["durations_ms"]["tts_render_total_ms"] == pytest.approx([600])
    assert row["durations_ms"]["tts_first_chunk_ms"] == pytest.approx([200])
    assert row["outcome"] == "interrupted"
    assert "tts_render_total_ms" not in row["unavailable_metrics"]
    with pytest.raises(ValueError, match="submitted audio"):
        summarize([event("endpoint_submitted", 1, audio_samples=480001)])


def test_reasoning_queue_parse_action_and_post_tool_spans_are_separate():
    report = summarize(
        [
            event("llm_request_submission_started", 1),
            event("llm_response_headers_received", 1.1),
            event("llm_first_sse_event", 1.5),
            event("llm_first_tool_delta", 1.7),
            event("llm_tool_parse_started", 2),
            event("llm_tool_parse_finished", 2.01),
            event("llm_tool_call_ready", 2.01),
            event("tool_started", 2.02),
            event("tool_completed", 2.22),
            event("post_tool_reasoning_started", 2.23),
            event("post_tool_reasoning_finished", 2.8),
        ]
    )
    durations = report["turns"][0]["durations_ms"]
    assert durations["llm_submit_to_response_headers_ms"] == pytest.approx([100])
    assert durations["llm_headers_to_first_sse_event_ms"] == pytest.approx([400])
    assert durations["llm_tool_parse_ms"] == pytest.approx([10])
    assert durations["tool_duration_ms"] == pytest.approx([200])
    assert durations["post_tool_reasoning_ms"] == pytest.approx([570])


def test_trace_rejects_payloads_bad_clocks_and_overwriting_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace = tmp_path / "var/results/trace.jsonl"
    rows = [event("asr_started", 1)]
    write_trace(trace, rows)
    assert read_trace(trace) == rows
    with pytest.raises(FileExistsError):
        write_trace(trace, rows)
    for bad in [
        event("x", float("nan")),
        event("x", 1, text="secret"),
        event("x", 1, turn_id=True),
    ]:
        with pytest.raises(ValueError):
            summarize([bad])
    with pytest.raises(ValueError, match="backwards"):
        summarize([event("asr_started", 2), event("asr_final", 1)])
    assert json.loads(trace.read_text()) == rows[0]


def test_music_button_diagnostics_and_provider_roles_remain_bounded(tmp_path):
    from me3_voice.telemetry import validate

    accepted = event(
        "music_ui_control", 1, action="next", outcome="accepted", audibility_verified=False
    )
    path = tmp_path / "var/results/controls.jsonl"
    path.parent.mkdir(parents=True)
    write_trace(
        path,
        [
            accepted,
            event(
                "turn_error",
                2,
                error="ProviderError",
                error_code="request_limit_reached",
                error_role="asr",
            ),
        ],
    )
    assert summarize(read_trace(path))["event_counts"]["music_ui_control"] == 1
    for bad in (
        dict(accepted, audibility_verified=True),
        dict(accepted, action="play"),
        dict(accepted, error_role="asr"),
    ):
        with pytest.raises(ValueError):
            validate(bad)


def test_music_playback_observations_are_payload_free_and_traceable(tmp_path):
    from me3_voice.telemetry import validate

    observations = [
        event(
            "music_playback_observed",
            1,
            status="playing",
            source="web_playback_sdk",
            track_changed=False,
        ),
        event(
            "music_playback_observed",
            2,
            status="playing",
            source="web_playback_sdk",
            track_changed=True,
        ),
        event(
            "music_playback_observed",
            3,
            status="paused",
            source="web_playback_sdk",
            track_changed=False,
        ),
    ]
    path = tmp_path / "var/results/music-observed.jsonl"
    path.parent.mkdir(parents=True)
    write_trace(path, observations)
    assert summarize(read_trace(path))["event_counts"]["music_playback_observed"] == 3
    for bad in (
        dict(observations[0], status="connected"),
        dict(observations[0], source="browser", track_id="private"),
        dict(observations[0], track_changed=1),
    ):
        with pytest.raises(ValueError):
            validate(bad)


def test_provider_timeout_boundary_survives_trace_and_is_allowlisted():
    report = summarize(
        [
            event(
                "turn_error",
                1,
                error="ProviderError",
                error_code="provider_timeout",
                error_role="llm",
                error_stage="reasoning",
                error_boundary="first_text_delta",
            )
        ]
    )
    assert report["turns"][0]["error_boundary"] == "first_text_delta"
    with pytest.raises(ValueError, match="timeout boundary"):
        summarize([event("turn_error", 1, error_boundary="private-header-value")])


def test_reasoning_parse_detail_is_symbolic_and_survives_report_summary():
    report = summarize(
        [
            event(
                "turn_error",
                1,
                error="ProviderError",
                error_code="provider_invalid_response",
                error_role="llm",
                error_stage="response_parse",
                error_detail="invalid_tool_arguments",
            )
        ]
    )
    assert report["turns"][0]["error_detail"] == "invalid_tool_arguments"
    with pytest.raises(ValueError, match="reasoning response detail"):
        summarize([event("turn_error", 1, error_detail="secret-response-body")])


def test_empty_model_response_is_not_misreported_as_a_completed_answer():
    report = summarize(
        [
            event("llm_empty_response", 1),
            event("turn_completed", 2),
        ]
    )
    assert report["turns"][0]["outcome"] == "empty_model_response"


def test_trace_journal_retains_checkpoints_without_duplicate_events(tmp_path):
    from me3_voice.telemetry import TraceJournal
    from me3_voice.artifacts import output_path

    # Use the project's allowed var tree rather than an external export path.
    from pathlib import Path
    import uuid

    path = output_path(str(Path("var/scratch") / ("journal-" + uuid.uuid4().hex + ".jsonl")))
    journal = TraceJournal(path)
    first = event("endpoint_submitted", 1, audio_samples=64000)
    journal.flush([first])
    assert len(path.read_text().splitlines()) == 1
    second = event("interrupt_requested", 2)
    journal.flush([first, second])
    journal.flush([first, second])
    assert len(path.read_text().splitlines()) == 2
    # A new owner cannot replace this run's evidence.
    with pytest.raises(FileExistsError):
        TraceJournal(path).flush([first])


def test_real_frontend_confirmation_and_local_light_events_are_exportable():
    from me3_voice.telemetry import validate

    assert validate(
        event(
            "vad_candidate_confirmed",
            1,
            voiced_frames=8,
            window_frames=12,
            required_voiced_frames=8,
        )
    )
    assert validate(event("barge_candidate_rejected", 2, voiced_frames=1, window_frames=12))
    assert validate(event("local_light_command_routed", 3, action="light_on"))
    with pytest.raises(ValueError):
        validate(event("vad_candidate_confirmed", 4, voiced_frames=33))
