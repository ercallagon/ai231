"""Local diagnostic traces, without audio/text payloads or acoustic inference.

Only matching run/session/turn/clock/segment IDs can form a duration. Missing
boundaries stay missing; streaming durations must never be added into E2E.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from threading import Lock

from .artifacts import output_path

FIELDS = {
    "event",
    "monotonic",
    "session_id",
    "turn_id",
    "mode",
    "run_id",
    "clock_id",
    "schema_version",
    "state",
    "segment_id",
    "error",
    "error_code",
    "error_role",
    "error_stage",
    "error_boundary",
    "error_detail",
    "action",
    "outcome",
    "reason",
    "input_following",
    "audibility_verified",
    "status",
    "source",
    "track_changed",
    "tool_call_id",
    "audio_samples",
    "voiced_frames",
    "window_frames",
    "required_voiced_frames",
    "llm_requests_used",
    "llm_request_limit",
    "onit_requests",
    "onit_worker_ms",
    "onit_model_ms",
    "onit_ttft_ms",
    "onit_completion_tokens",
}
MODES = {"mock", "HPC-live", "Pi-verified"}
PAIRS = {
    "vad_candidate_confirmation_ms": (
        "vad_candidate_started",
        "vad_candidate_confirmed",
    ),
    "llm_request_to_first_delta_ms": ("llm_request_started", "llm_first_delta"),
    "llm_request_total_ms": ("llm_request_started", "llm_request_finished"),
    "llm_submit_to_response_headers_ms": (
        "llm_request_submission_started",
        "llm_response_headers_received",
    ),
    "llm_headers_to_first_sse_event_ms": (
        "llm_response_headers_received",
        "llm_first_sse_event",
    ),
    "llm_request_to_first_tool_delta_ms": (
        "llm_request_started",
        "llm_first_tool_delta",
    ),
    "llm_request_to_valid_tool_call_ms": (
        "llm_request_started",
        "llm_tool_call_ready",
    ),
    "llm_first_sse_to_valid_tool_call_ms": (
        "llm_first_sse_event",
        "llm_tool_call_ready",
    ),
    "llm_tool_parse_ms": ("llm_tool_parse_started", "llm_tool_parse_finished"),
    "post_tool_reasoning_ms": (
        "post_tool_reasoning_started",
        "post_tool_reasoning_finished",
    ),
    "turn_analysis_ms": ("turn_analysis_started", "turn_analysis_finished"),
    "last_voiced_frame_to_sink_mock_ms": (
        "last_voiced_frame_consumed_file",
        "sink_write_started_mock",
    ),
    "last_voiced_frame_to_verified_mock_task_ms": (
        "last_voiced_frame_consumed_file",
        "task_verified_mock",
    ),
    "tts_first_chunk_to_sink_mock_ms": ("tts_first_chunk", "sink_write_started_mock"),
    "last_voiced_frame_to_endpoint_ms": (
        "last_voiced_frame_consumed_file",
        "endpoint_submitted",
    ),
    "endpoint_to_asr_final_ms": ("endpoint_submitted", "asr_final"),
    "tool_duration_ms": ("tool_started", "tool_completed"),
    "asr_request_ms": ("asr_started", "asr_final"),
    "llm_first_segment_ms": ("llm_started", "llm_first_segment"),
    "tts_first_chunk_ms": ("tts_started", "tts_first_chunk"),
    "tts_render_total_ms": ("tts_started", "tts_finished"),
    "endpoint_to_sink_write_mock_ms": ("endpoint_submitted", "sink_write_started_mock"),
    "interrupt_to_sink_stop_mock_ms": ("interrupt_requested", "playback_stopped_mock"),
    "endpoint_to_browser_send_ms": ("endpoint_submitted", "sink_write_started_browser"),
    "last_voiced_frame_to_browser_send_ms": (
        "last_voiced_frame_consumed_file",
        "sink_write_started_browser",
    ),
    "interrupt_to_browser_stop_sent_ms": ("interrupt_requested", "playback_stopped_browser"),
}


def validate(event):
    if not isinstance(event, dict) or set(event) - FIELDS:
        raise ValueError("trace accepts diagnostic fields only; no text/audio payloads")
    for name in ("event", "session_id", "run_id", "clock_id", "mode"):
        value = event.get(name)
        if not isinstance(value, str) or not 1 <= len(value) <= 100:
            raise ValueError(f"invalid trace {name}")
        if any(not (c.isalnum() or c in "_-.:") for c in value):
            raise ValueError(f"invalid trace {name}")
    if event["mode"] not in MODES or event.get("schema_version") != 1:
        raise ValueError("unsupported trace mode/schema")
    if type(event.get("turn_id")) is not int or event["turn_id"] < 0:
        raise ValueError("invalid trace turn ID")
    timestamp = event.get("monotonic")
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("invalid monotonic timestamp")
    for field in ("voiced_frames", "window_frames", "required_voiced_frames"):
        if field in event and (
            event["event"] not in {"vad_candidate_confirmed", "barge_candidate_rejected"}
            or type(event[field]) is not int
            or not 0 <= event[field] <= 32
        ):
            raise ValueError("invalid VAD confirmation count")
    if "segment_id" in event and (type(event["segment_id"]) is not int or event["segment_id"] < 1):
        raise ValueError("invalid segment ID")
    if "audio_samples" in event and (
        event["event"] != "endpoint_submitted"
        or type(event["audio_samples"]) is not int
        or not 0 <= event["audio_samples"] <= 16000 * 30
    ):
        raise ValueError("invalid submitted audio samples")
    if any(
        name in event
        for name in (
            "onit_requests",
            "onit_worker_ms",
            "onit_model_ms",
            "onit_ttft_ms",
            "onit_completion_tokens",
        )
    ):
        if event["event"] != "onit_model_completed":
            raise ValueError("Onit metrics must belong to an Onit completion event")
        if "onit_requests" in event and (
            type(event["onit_requests"]) is not int or not 0 <= event["onit_requests"] <= 100
        ):
            raise ValueError("invalid Onit request count")
        if "onit_completion_tokens" in event and (
            type(event["onit_completion_tokens"]) is not int
            or not 0 <= event["onit_completion_tokens"] <= 100000
        ):
            raise ValueError("invalid Onit token count")
        if any(
            name in event
            and (
                type(event[name]) not in (int, float)
                or not math.isfinite(event[name])
                or not 0 <= event[name] <= 600000
            )
            for name in ("onit_worker_ms", "onit_model_ms", "onit_ttft_ms")
        ):
            raise ValueError("invalid Onit timing")
    for field in ("state", "error", "tool_call_id"):
        if field in event and (
            not isinstance(event[field], str)
            or not 1 <= len(event[field]) <= 100
            or not all(c.isalnum() or c in "_-" for c in event[field])
        ):
            raise ValueError(f"invalid symbolic {field}")
    if "error_code" in event and (
        not isinstance(event["error_code"], str)
        or not 1 <= len(event["error_code"]) <= 48
        or not re.fullmatch(r"[a-z0-9_]+", event["error_code"])
    ):
        raise ValueError("invalid symbolic error code")
    if "error_role" in event and (
        event["event"] != "turn_error" or event["error_role"] not in {"asr", "tts", "llm"}
    ):
        raise ValueError("invalid provider failure role")
    if "error_stage" in event and (
        event["event"] != "turn_error"
        or event["error_stage"]
        not in {"asr", "reasoning", "response_parse", "tool", "tts", "playback", "deadline"}
    ):
        raise ValueError("invalid failure stage")
    if "error_boundary" in event and (
        event["event"] != "turn_error"
        or event["error_boundary"]
        not in {"response_headers", "first_sse_event", "first_text_delta", "sse_stream"}
    ):
        raise ValueError("invalid provider timeout boundary")
    if "error_detail" in event and (
        event["event"] != "turn_error"
        or event["error_detail"]
        not in {
            "invalid_sse_json",
            "incomplete_sse",
            "invalid_event_shape",
            "event_not_object",
            "provider_error_event",
            "missing_choices",
            "choices_not_array",
            "unexpected_choice_count",
            "choice_not_object",
            "invalid_choice_index",
            "invalid_delta_shape",
            "invalid_tool_call_shape",
            "invalid_tool_arguments",
            "invalid_tool_identity",
            "unadvertised_tool_name",
            "duplicate_tool_identity",
            "invalid_finish_reason",
        }
    ):
        raise ValueError("invalid reasoning response detail")
    if event["event"] == "local_light_command_routed":
        if event.get("action") not in {"light_on", "light_off", "set_color", "set_brightness"}:
            raise ValueError("invalid local light route")
    elif (
        any(key in event for key in ("action", "audibility_verified"))
        or event["event"] == "music_ui_control"
    ):
        if (
            event["event"] != "music_ui_control"
            or event.get("action")
            not in {
                "pause",
                "resume",
                "stop",
                "next",
                "previous",
                "seek",
                "volume_up",
                "volume_down",
                "volume_set",
            }
            or event.get("outcome") not in {"accepted", "failed"}
            or event.get("audibility_verified") is not False
        ):
            raise ValueError("invalid music control diagnostic")
    if "outcome" in event and event["event"] == "music_duck":
        if event.get("outcome") not in {
            "confirmed",
            "restored",
            "uncertain",
            "unsupported",
            "not_playing",
            "already_ducked",
            "unchanged",
            "preserved_external_change",
            "failed",
        } or event.get("reason") not in {
            None,
            "target_not_active",
            "volume_control_unavailable",
            "volume_unavailable",
        }:
            raise ValueError("invalid music ducking diagnostic")
    if "input_following" in event and (
        event["event"] not in {"playback_stopped_browser", "playback_stopped_mock"}
        or type(event["input_following"]) is not bool
    ):
        raise ValueError("invalid interruption continuation")
    if any(key in event for key in ("status", "source", "track_changed")):
        if (
            event["event"] != "music_playback_observed"
            or event.get("status") not in {"playing", "paused", "not_playing", "buffering"}
            or event.get("source") not in {"web_playback_sdk", "spotify_api"}
            or type(event.get("track_changed")) is not bool
        ):
            raise ValueError("invalid music playback observation")
    if "llm_requests_used" in event or "llm_request_limit" in event:
        limit = event.get("llm_request_limit")
        if (
            event["event"] not in {"llm_request_finished", "turn_error"}
            or type(event.get("llm_requests_used")) is not int
            or (limit is not None and type(limit) is not int)
            or event["llm_requests_used"] < 0
            or (limit is not None and not event["llm_requests_used"] <= limit <= 100000)
        ):
            raise ValueError("invalid reasoning request accounting")
    return event


def write_trace(path: Path, events):
    """Explicit export after a run; never perform disk I/O in audio callbacks."""
    path = output_path(str(path))
    checked = [validate(event) for event in events]
    with path.open("x") as stream:
        for event in checked:
            stream.write(json.dumps(event, allow_nan=False) + "\n")


class TraceJournal:
    """Append checkpoints from a control task, including abnormal teardown."""

    def __init__(self, path):
        self.path = output_path(str(path))
        self.written = 0
        self.created = False
        self.lock = Lock()

    def flush(self, events):
        with self.lock:
            pending = [validate(event) for event in events[self.written :]]
            if not pending and self.created:
                return
            with self.path.open("a" if self.created else "x") as stream:
                for event in pending:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
            self.created = True
            self.written += len(pending)


def read_trace(path):
    # Bound memory and input sizes for a local developer report.
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("trace exceeds 16 MiB; split runs")
    events = []
    with path.open() as stream:
        for line in stream:
            if len(line) > 4096 or len(events) >= 100000:
                raise ValueError("trace exceeds event bounds")
            events.append(validate(json.loads(line)))
    return events


def summarize(events):
    groups = defaultdict(list)
    counts = Counter()
    for event in events:
        validate(event)
        key = tuple(event[k] for k in ("run_id", "session_id", "turn_id", "clock_id", "mode"))
        groups[key].append(event)
        counts[event["event"]] += 1
    rows = []
    for key, group in groups.items():
        if any(b["monotonic"] < a["monotonic"] for a, b in zip(group, group[1:])):
            raise ValueError("clock moved backwards within a turn")
        timings = defaultdict(list)
        pending = {}
        for event in group:
            for metric, (start, end) in PAIRS.items():
                # TTS segments are separate requests; no cross-segment pairing.
                segment = (
                    event.get("segment_id")
                    if metric in {"tts_first_chunk_ms", "tts_render_total_ms"}
                    else None
                )
                if metric == "tool_duration_ms":
                    segment = event.get("tool_call_id")
                pair_key = (metric, segment)
                if event["event"] == start:
                    pending[pair_key] = event["monotonic"]
                elif event["event"] == end and pair_key in pending:
                    timings[metric].append((event["monotonic"] - pending.pop(pair_key)) * 1000)
        rows.append(
            {
                **dict(zip(("run_id", "session_id", "turn_id", "clock_id", "mode"), key)),
                "durations_ms": dict(timings),
                "asr_input_duration_ms": next(
                    (
                        e["audio_samples"] / 16
                        for e in group
                        if e["event"] == "endpoint_submitted" and "audio_samples" in e
                    ),
                    None,
                ),
                "asr_input_incremental": any(e["event"] == "asr_stream_opened" for e in group),
                "asr_partial_observed": any(e["event"] == "asr_partial" for e in group),
                "outcome": "error"
                if any(e["event"] == "turn_error" for e in group)
                else "empty_model_response"
                if any(e["event"] == "llm_empty_response" for e in group)
                else "completed"
                if any(e["event"] == "turn_completed" for e in group)
                else "interrupted"
                if any(e["event"] == "interrupt_requested" for e in group)
                else "not_completed_or_control_only",
                "unavailable_metrics": sorted(set(PAIRS) - set(timings)),
                "reasoning_request_budget": next(
                    (
                        {
                            "used": e["llm_requests_used"],
                            "limit": e["llm_request_limit"],
                            "remaining": e["llm_request_limit"] - e["llm_requests_used"]
                            if e["llm_request_limit"] is not None
                            else None,
                        }
                        for e in reversed(group)
                        if "llm_requests_used" in e
                    ),
                    None,
                ),
                "error_code": next(
                    (e["error_code"] for e in reversed(group) if "error_code" in e), None
                ),
                "error_boundary": next(
                    (e["error_boundary"] for e in reversed(group) if "error_boundary" in e),
                    None,
                ),
                "error_detail": next(
                    (e["error_detail"] for e in reversed(group) if "error_detail" in e),
                    None,
                ),
            }
        )
    import numpy as np

    aggregates = defaultdict(list)
    for row in rows:
        for metric, values in row["durations_ms"].items():
            aggregates[(row["mode"], row["clock_id"], metric)].extend(values)
    return {
        "software_aggregates": [
            {
                "mode": mode,
                "clock_id": clock,
                "metric": metric,
                "count": len(values),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "max_ms": max(values),
            }
            for (mode, clock, metric), values in sorted(aggregates.items())
        ],
        "schema_version": 1,
        "event_counts": dict(counts),
        "turns": rows,
        "speech_end_to_first_audible_response_ms": None,
        "interruption_to_audio_stop_ms": None,
        "verified_physical_task_completion_ms": None,
        "acoustic_response_ms": None,
        "acoustic_barge_in_ms": None,
        "limitations": "Local diagnostic boundaries only. A sink write is not audible playback. "
        "LLM first text delta is not a token count; tokens/second is unavailable. "
        "ASR input duration includes any pre-roll and is not speech duration. "
        "TTS first chunk is decoded PCM, not first network byte. "
        "Missing metrics are unavailable, not zero. "
        "No aggregation across modes/clocks and no sum of overlapping stages.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=output_path, required=True)
    args = parser.parse_args()
    if args.output == args.trace.resolve():
        parser.error("summary cannot overwrite its input trace")
    report = summarize(read_trace(args.trace))
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(f"Diagnostic trace: {sum(report['event_counts'].values())} events; {args.output}")


if __name__ == "__main__":
    main()
