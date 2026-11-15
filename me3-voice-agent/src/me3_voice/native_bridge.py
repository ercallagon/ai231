"""Opt-in native engine behind the existing loopback browser entrypoint."""

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from aiohttp import web, WSMsgType

from .native_app import NativeApplication, NativeState
from .readiness import build_wake
from .dashboard import render, render_audio_probe, snapshot
from .artifacts import output_path


_SAFE_EVENT_TEXT = {
    "event",
    "run_id",
    "session_id",
    "error_code",
    "error_role",
    "connection_state",
    "failure_code",
    "error_stage",
    "error_boundary",
    "exception_type",
    "provider_exception_type",
    "remote_cause_type",
    "cause_type",
    "state",
    "status",
    "role",
    "tool_name",
    "action",
    "mode",
    "failure_stage",
    "projection",
    "provider",
    "backend",
    "model",
    "stream_id",
    "result_status",
    "outcome",
    "reason",
    "route",
    "request_id",
    "llm_request_id",
    "response_id",
    "operation_id",
    "tool_call_id",
    "call_id",
    "owner_id",
    "audio_context_state",
}


def _safe_event(event):
    """Retain timing and symbolic diagnostics, never speech, prompts or credentials."""
    result = {}
    for key, value in event.items():
        if not isinstance(key, str):
            continue
        if (
            type(value) in (int, float, bool)
            and not key.endswith(("key", "token"))
            and (type(value) is not float or math.isfinite(value))
        ):
            result[key] = value
        elif (
            key in _SAFE_EVENT_TEXT
            and isinstance(value, str)
            and re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", value)
        ):
            result[key] = value
    if result.get("event") == "asr_provider_transport":
        stages = event.get("connection_stages")
        if isinstance(stages, dict):
            for name in (
                "dns_started",
                "dns_completed",
                "connection_establishment_started",
                "connection_established",
                "websocket_upgrade_request_sent",
                "websocket_http_response_received",
                "websocket_http_status",
                "connection_failed_monotonic",
                "connection_cancelled_monotonic",
            ):
                value = stages.get(name)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    result[name] = value
            value = stages.get("connection_terminal_exception")
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value):
                result["connection_terminal_exception"] = value
    if result.get("event") == "tts_segment_summary":
        diagnostics = event.get("diagnostics")
        if isinstance(diagnostics, dict):
            for section, source_name, fields in (
                (
                    "provider",
                    "provider",
                    (
                        "provider_audio_seconds",
                        "provider_receive_wait_seconds",
                        "consumer_pause_seconds",
                        "synthesis_wall_seconds",
                        "provider_pcm_bytes",
                        "first_pcm_monotonic",
                        "last_pcm_monotonic",
                        "provider_flushed_monotonic",
                        "connection_setup_seconds",
                        "connection_ready_monotonic",
                        "connection_reused",
                        "connection_attempt_monotonic",
                        "first_text_submitted_monotonic",
                        "flush_submitted_monotonic",
                    ),
                ),
                ("client_mcp", "client_mcp", ("tts_next_count", "pcm_bytes", "audio_seconds")),
                (
                    "mcp_server",
                    "mcp_server",
                    ("tts_next_count", "pcm_chunks", "pcm_bytes", "audio_seconds"),
                ),
            ):
                values = diagnostics.get(source_name)
                if section == "mcp_server" and isinstance(values, dict):
                    values = values.get("mcp")
                if not isinstance(values, dict):
                    continue
                for name in fields:
                    value = values.get(name)
                    if name == "connection_reused" and type(value) is bool:
                        result[f"{section}_{name}"] = value
                    elif type(value) in (int, float) and math.isfinite(value) and value >= 0:
                        result[f"{section}_{name}"] = value
                if section == "client_mcp":
                    latency = values.get("tts_next_latency_seconds")
                    if isinstance(latency, dict):
                        value = latency.get("p95_seconds")
                        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                            result["client_mcp_next_p95_seconds"] = value
                elif section == "mcp_server":
                    latency = (values.get("tts_next_seconds") or {}).get("p95_seconds")
                    if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
                        result["mcp_server_next_p95_seconds"] = latency
            provider = diagnostics.get("provider")
            if isinstance(provider, dict):
                connection_trace = provider.get("connection_trace")
                if isinstance(connection_trace, dict):
                    for name in (
                        "dns_started",
                        "dns_completed",
                        "dns_cache_hit",
                        "dns_cache_miss",
                        "connection_pool_wait_started",
                        "connection_pool_wait_completed",
                        "connection_establishment_started",
                        "connection_established",
                        "websocket_upgrade_request_sent",
                        "websocket_http_response_received",
                        "connection_failed_monotonic",
                        "connection_cancelled_monotonic",
                    ):
                        value = connection_trace.get(name)
                        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                            result[f"tts_connection_{name}"] = value
                for key in ("status", "error_code", "failure_type"):
                    value = provider.get(key)
                    pattern = (
                        r"[a-z0-9_]{1,48}"
                        if key == "error_code"
                        else r"[A-Za-z][A-Za-z0-9_.-]{0,79}"
                    )
                    if isinstance(value, str) and re.fullmatch(pattern, value):
                        result[f"provider_{key}"] = value
            mcp_server = diagnostics.get("mcp_server")
            if isinstance(mcp_server, dict):
                value = mcp_server.get("status")
                if isinstance(value, str) and value in {
                    "running",
                    "completed",
                    "cancelled",
                    "failed",
                    "unknown",
                }:
                    result["mcp_server_status"] = value
    if result.get("event") == "tool_completed":
        diagnostics = event.get("diagnostics")
        if isinstance(diagnostics, dict):
            # Retrieved evidence diagnostics are deliberately projected field by
            # field. Never serialize the nested result/evidence object itself.
            source_id = diagnostics.get("source_id")
            if isinstance(source_id, str) and re.fullmatch(r"S[0-9]{1,6}", source_id):
                result["research_source_id"] = source_id
            source_url_hash = diagnostics.get("source_url_hash")
            if isinstance(source_url_hash, str) and re.fullmatch(r"[a-f0-9]{16}", source_url_hash):
                result["research_source_url_hash"] = source_url_hash
            for key in ("extraction_method", "evidence_status"):
                value = diagnostics.get(key)
                if isinstance(value, str) and re.fullmatch(r"[a-z0-9_]{1,48}", value):
                    result[f"research_{key}"] = value
            for key in ("evidence_characters", "evidence_chunks"):
                value = diagnostics.get(key)
                if type(value) is int and 0 <= value <= 10_000_000:
                    result[f"research_{key}"] = value
            for key in ("http_status", "redirect_count"):
                value = diagnostics.get(key)
                if type(value) is int and 0 <= value <= 10000:
                    result[f"research_{key}"] = value
            for key in ("stage", "failure_class", "fallback_status"):
                value = diagnostics.get(key)
                if isinstance(value, str) and re.fullmatch(r"[a-z0-9_]{1,48}", value):
                    result[f"research_{key}"] = value
            host_hash = diagnostics.get("host_hash")
            if isinstance(host_hash, str) and re.fullmatch(r"[a-f0-9]{12}", host_hash):
                result["research_host_hash"] = host_hash
            if type(diagnostics.get("reused_existing_leads")) is bool:
                result["research_reused_existing_leads"] = diagnostics["reused_existing_leads"]
            fallback = diagnostics.get("fallback_error")
            if isinstance(fallback, dict):
                for key in ("error", "stage", "failure_class"):
                    value = fallback.get(key)
                    if isinstance(value, str) and re.fullmatch(r"[a-z0-9_]{1,48}", value):
                        result[f"research_fallback_{key}"] = value
                status = fallback.get("http_status")
                if type(status) is int and 0 <= status <= 10000:
                    result["research_fallback_http_status"] = status
                fallback_host = fallback.get("host_hash")
                if isinstance(fallback_host, str) and re.fullmatch(r"[a-f0-9]{12}", fallback_host):
                    result["research_fallback_host_hash"] = fallback_host
    if result.get("event") == "llm_citation_check":
        for key in ("citation_ids", "retained_source_ids"):
            values = event.get(key)
            if (
                isinstance(values, list)
                and len(values) <= 256
                and all(
                    isinstance(value, str) and re.fullmatch(r"S[0-9]{1,6}", value)
                    for value in values
                )
            ):
                result[key] = values
    if result.get("event") == "llm_claim_support_diagnostic":
        for key, pattern in (
            ("source_id", r"S[0-9]{1,6}"),
            ("passage_id", r"S[0-9]{1,6}-P[0-9]{1,6}"),
        ):
            value = event.get(key)
            if isinstance(value, str) and re.fullmatch(pattern, value):
                result[key] = value
        codes = event.get("contradiction_codes")
        allowed = {"entity_anchor_conflict", "numeric_anchor_conflict", "polarity_conflict"}
        if (
            isinstance(codes, list)
            and len(codes) <= 3
            and all(isinstance(code, str) and code in allowed for code in codes)
        ):
            result["contradiction_codes"] = sorted(set(codes))
    return result


class NativeEventJournal:
    def __init__(self, trace_dir, run_id):
        path = output_path(str(Path(trace_dir) / f"native-{run_id}.jsonl"))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(fd, "w", encoding="utf-8")
        self.last_sequence = 0
        self.lock = threading.Lock()

    def write_new(self, events):
        with self.lock:
            rows = [event for event in events if event["sequence"] > self.last_sequence]
            if rows:
                self.stream.write(
                    "".join(json.dumps(_safe_event(row), allow_nan=False) + "\n" for row in rows)
                )
                self.stream.flush()
                self.last_sequence = rows[-1]["sequence"]

    def close(self):
        with self.lock:
            self.stream.close()


def context_view(state, application, *, archived=False):
    # Projection only: never used as another LLM history or persisted state owner.
    rows = {}
    turn = 0
    for message in state.context.get_messages():
        role = message.get("role")
        text = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(text, str) or not text:
            continue
        if role == "user":
            turn += 1
        if not turn:
            continue
        key = f"{turn}:{role}"
        if key in rows:
            rows[key]["text"] += " " + text
        else:
            rows[key] = {
                "id": key,
                "turn_id": turn,
                "role": role,
                "text": text,
                "confirmed_text": "",
                "status": "final",
                "delivery": "asr_final" if role == "user" else "sink_confirmed",
            }
        if role == "assistant":
            response_id = getattr(application, "response_by_conversation_turn", {}).get(turn)
            response = getattr(application, "response_records", {}).get(response_id)
            if response is None and not hasattr(application, "response_records"):
                # Compatibility for legacy transcript projections without native
                # lifecycle telemetry. Native sessions never infer delivery here.
                rows[key]["confirmed_text"] = rows[key]["text"]
                rows[key]["generation_complete"] = True
                rows[key]["status"] = "final"
            elif response is not None:
                generated = response.get("generated_state") == "completed"
                completed = (
                    response.get("response_state") == "completed"
                    and response.get("tts_state") == "completed"
                    and response.get("playback_state") == "browser_acknowledged"
                    and response.get("playback_packets", 0) == response.get("playback_acks", 0)
                )
                interrupted = response.get("response_state") in {"interrupted", "superseded"}
                failed = response.get("tts_state") == "failed"
                rows[key]["generation_complete"] = generated
                rows[key]["confirmed_text"] = rows[key]["text"] if completed else ""
                rows[key]["delivery"] = "sink_confirmed" if completed else "generated"
                rows[key]["status"] = (
                    "final"
                    if completed
                    else "interrupted"
                    if interrupted
                    else "error"
                    if failed
                    else "partial"
                )
                rows[key]["response"] = {
                    "response_id": response.get("response_id"),
                    "state": response.get("response_state", "pending"),
                    "generation": (
                        "interrupted"
                        if response.get("generated_state") == "partial"
                        else response.get("generated_state", "unknown")
                    ),
                    "tts": response.get("tts_state", "unknown"),
                    "playback": response.get("playback_state", "unknown"),
                    "archive": "included" if archived else "pending",
                    "segments": response.get("tts_segments", [])[-50:],
                    "packets": response.get("playback_packets", 0),
                    "acks": response.get("playback_acks", 0),
                    "superseded_by_turn_id": response.get("superseded_by_turn_id"),
                    "interruption_reason": response.get("interruption_reason"),
                }
    partial = getattr(application, "partial_transcript", None)
    if isinstance(partial, str) and partial.strip():
        partial_turn = turn + 1
        rows[f"{partial_turn}:user"] = {
            "id": f"{partial_turn}:user",
            "turn_id": partial_turn,
            "role": "user",
            "text": partial[:8000],
            "confirmed_text": "",
            "status": "partial",
            "delivery": "asr_partial",
        }
    # A failed/interrupted response may not have been committed into the
    # canonical LLM context. Project its already captured text as partial,
    # retaining the stable response identity and lifecycle states.
    for response_id in getattr(application, "response_order", ()):
        response = getattr(application, "response_records", {}).get(response_id)
        if not isinstance(response, dict):
            continue
        text = response.get("generated_text")
        response_turn = response.get("turn_id")
        if (
            not isinstance(text, str)
            or not text
            or type(response_turn) is not int
            or response_turn < 1
        ):
            continue
        key = f"{response_turn}:assistant"
        if key in rows:
            continue
        completed = (
            response.get("response_state") == "completed"
            and response.get("tts_state") == "completed"
            and response.get("playback_state") == "browser_acknowledged"
            and response.get("playback_packets", 0) == response.get("playback_acks", 0)
        )
        state_name = response.get("response_state", "partial")
        rows[key] = {
            "id": f"{response_turn}:assistant:{response_id}",
            "turn_id": response_turn,
            "role": "assistant",
            "text": text[:8000],
            "confirmed_text": text[:8000] if completed else "",
            "status": "final" if completed else state_name,
            "delivery": "sink_confirmed" if completed else "generated",
            "generation_complete": response.get("generated_state") == "completed",
            "response": {
                "response_id": response_id,
                "state": state_name,
                "generation": response.get("generated_state", "unknown"),
                "tts": response.get("tts_state", "unknown"),
                "playback": response.get("playback_state", "unknown"),
                "archive": "included" if archived else "pending",
                "segments": response.get("tts_segments", [])[-50:],
                "packets": response.get("playback_packets", 0),
                "acks": response.get("playback_acks", 0),
                "superseded_by_turn_id": response.get("superseded_by_turn_id"),
                "interruption_reason": response.get("interruption_reason"),
            },
        }
    visible = []
    remaining = 90000
    for row in reversed(list(rows.values())):
        row["text"] = row["text"][:8000]
        if row["role"] == "assistant" and "response" not in row:
            row["confirmed_text"] = row["text"]
        size = len(row["text"]) + len(row["confirmed_text"])
        if size > remaining or len(visible) >= 100:
            break
        visible.append(row)
        remaining -= size
    task_store = state.task_runtime.store if state.task_runtime else None
    task_revision = task_store.revision if task_store else 0
    action_revision = getattr(application, "action_revision", 0)
    tasks = (task_store.snapshot() if task_store else []) + list(
        getattr(application, "fast_action_rows", ())
    )
    tasks.sort(
        key=lambda row: (
            row.get("turn_id", 0),
            row.get("started_utc", ""),
            row.get("call_id", ""),
        )
    )
    return {
        "schema_version": 1,
        "kind": "conversation-transcript",
        "display_authorized": True,
        "session_id": state.session_id,
        # The browser rejects non-advancing transcript revisions. Include the
        # task/action stores because their snapshots can change independently of
        # conversation-event delivery (for example, background task progress).
        "revision": application.event_count + task_revision + action_revision,
        "mode": "mock" if all(s.backend == "mock" for s in state.backends.values()) else "HPC-live",
        "backend_modes": {role: spec.backend for role, spec in state.backends.items()},
        "state": "idle" if application.projection == "wake_listening" else application.projection,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "messages": list(reversed(visible)),
        "tasks": tasks[-50:],
    }


def build_native_app(root, backends, credentials, wake_args, options, *, trace_dir=None):
    from .voice_bridge import (
        BrowserSink,
        local_request,
        music_ui_control_fields,
        music_playback_observed_fields,
    )

    if getattr(wake_args, "onit_python", None):
        raise ValueError(
            "native runtime uses LangGraph; Onit is available only in the legacy rollback"
        )
    state = NativeState(
        backends,
        credentials,
        device_config=getattr(wake_args, "device_config", None),
        receipt_store=getattr(wake_args, "receipt_store", None),
    )
    app = web.Application(client_max_size=16384)
    busy = False

    async def close(_app):
        await state.close()

    app.on_cleanup.append(close)

    async def page(request):
        local_request(request)
        return web.Response(
            text=await asyncio.to_thread(lambda: render(snapshot(root), live=True)),
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
            },
        )

    async def audio_probe(request):
        local_request(request)
        body, policy = render_audio_probe()
        return web.Response(
            text=body,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": policy,
                "Permissions-Policy": "microphone=(self)",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
            },
        )

    async def health(request):
        local_request(request)
        return web.json_response(
            {
                "bridge": "ready",
                "engine": "native",
                "voice_session": "active" if busy else "idle",
                "speech_services": "failed"
                if state.owner and state.owner.done()
                else "initialized"
                if state.speech
                else "lazy",
                "session_resume": True,
                "resume_scope": "SQLite LangGraph checkpoint; task resume is explicit",
                "task_runtime": "ready" if state.task_runtime else "lazy",
            }
        )

    devices = state.tasks.actions.capabilities()
    music = state.tasks.music.capabilities()

    async def capability(request):
        local_request(request)
        from .capability_report import build_capability_report

        capability_report = build_capability_report(
            backends=backends,
            options=options,
            credentials=state.credentials,
            device_capabilities=devices,
            device_details={
                "weather_enabled": getattr(state.tasks.actions, "weather", None) is not None,
                "communications_enabled": bool(
                    getattr(getattr(state.tasks.actions, "communications", None), "enabled", False)
                ),
            },
            music_capabilities=music,
            credential_presence={
                "ME3_SPOTIFY_ACCESS_TOKEN": bool(getattr(state.tasks.music, "token", None)),
                "ME3_SPOTIFY_REFRESH_TOKEN": bool(
                    getattr(state.tasks.music, "refresh_token", None)
                ),
            },
            task_runtime=state.task_runtime,
            speech_client=state.speech,
            wake_configured=bool(getattr(wake_args, "kws_assets", None)),
        )
        return web.json_response(
            {
                "available": True,
                "engine": "native",
                "capture": "browser-pcm16",
                "playback": "browser-acknowledged-pcm16",
                "asr_partial": backends["asr"].streaming,
                "aec": "browser-requested-unverified",
                "wake": "Hey Swift"
                if getattr(wake_args, "kws_assets", None)
                else "configured model",
                "task_artifact_access": bool(state.task_runtime),
                "research": {
                    "allow_live": options.research.allow_live,
                    "request_limit": options.research.request_limit,
                },
                "devices": devices,
                "music": music,
                "capability_report": capability_report,
            },
            headers={"Cache-Control": "no-store"},
        )

    async def reports(request):
        local_request(request)
        return web.json_response(
            await asyncio.to_thread(snapshot, root), headers={"Cache-Control": "no-store"}
        )

    async def unsupported(request):
        local_request(request)
        return web.json_response(
            {
                "available": False,
                "engine": "native",
                "message": "This legacy integration is not yet bound to the native engine.",
            },
            status=503,
        )

    async def voice(request):
        nonlocal busy
        local_request(request)
        if busy:
            raise web.HTTPConflict(text="voice session already active")
        busy = True
        ws = web.WebSocketResponse(max_msg_size=16384, heartbeat=10)
        publisher = None
        event_publisher = None
        sink = None
        application = None
        journal = None
        diagnostic = None
        end_requested = False
        try:
            await ws.prepare(request)
            sink = BrowserSink(ws)
            message = await asyncio.wait_for(ws.receive(), 10)
            if message.type != WSMsgType.TEXT:
                raise ValueError("expected hello")
            hello = json.loads(message.data)
            if (
                not isinstance(hello, dict)
                or hello.get("type") != "hello"
                or type(hello.get("rate")) is not int
                or hello["rate"] not in (16000, 44100, 48000)
                or hello.get("channels") != 1
                or set(hello)
                - {
                    "type",
                    "rate",
                    "channels",
                    "aec_reported",
                    "conversation_mode",
                    "capture_source",
                }
                or hello.get("conversation_mode", "followup")
                not in {"single", "followup", "extended"}
                or hello.get("capture_source", "browser") not in {"browser", "pipewire"}
            ):
                raise ValueError("unsupported audio format")
            wake = build_wake(wake_args)
            if wake is None:
                raise ValueError("real wake model required for browser capture")
            application = NativeApplication(
                state,
                sink,
                options=options,
                rate=hello["rate"],
                wake=wake,
                conversation_mode=hello.get("conversation_mode", "followup"),
                followup_seconds=getattr(wake_args, "followup_seconds", 20),
                extended_seconds=getattr(wake_args, "extended_seconds", 20),
            )
            from .transcript import TranscriptPublisher

            publisher = TranscriptPublisher(
                NativeTranscriptView(state, application),
                archive_dir=Path("var/results/voice-sessions"),
            )
            await publisher.__aenter__()
            if getattr(wake_args, "diagnostic_audio", False):
                from .diagnostics import DiagnosticCapture

                diagnostic = DiagnosticCapture(
                    root,
                    application.run_id,
                    hello["rate"],
                    max_seconds=getattr(wake_args, "diagnostic_max_seconds", 60),
                )
            if trace_dir is not None:
                journal = NativeEventJournal(trace_dir, application.run_id)
            async with application:
                await sink.send(
                    {
                        "type": "ready",
                        "engine": "native",
                        "run_id": application.run_id,
                        "session_id": state.session_id,
                        "mode": "mock"
                        if all(p.backend == "mock" for p in backends.values())
                        else "HPC-live",
                        "provider_roles": {role: p.model or "Mock" for role, p in backends.items()},
                        "task_artifact_access": bool(state.task_runtime),
                        "devices": devices,
                        "session_resume": True,
                        "music": music,
                        "diagnostic_audio": bool(diagnostic),
                        "limitations": "Native voice plus durable LangGraph tasks. Pi acoustics and live task-model quality remain unqualified; GPIO output is accepted by the adapter without physical sensor readback.",
                    }
                )

                async def publish():
                    revision = None
                    transcript_revision = None
                    transcript_sent_at = 0.0
                    last_event = 0
                    status_sequence = 0
                    try:
                        while not ws.closed:
                            messages = state.context.get_messages()
                            task_revision = (
                                state.task_runtime.store.revision
                                if state.task_runtime and state.task_runtime.store
                                else 0
                            )
                            identity = (
                                len(messages),
                                application.projection,
                                task_revision,
                                application.turn_id,
                                application.parts.stt.speaking,
                                application.action_revision,
                            )
                            if identity != revision:
                                revision = identity
                                status_sequence += 1
                                await sink.send(
                                    {
                                        "type": "status",
                                        "engine": "native",
                                        "state": application.projection,
                                        "session_id": state.session_id,
                                        "run_id": application.run_id,
                                        "capture_active": True,
                                        "turn_id": application.turn_id,
                                        "status_sequence": status_sequence,
                                        "event_sequence": application.event_count,
                                        "input_open": application.parts.stt.speaking,
                                    }
                                )
                            if application.event_count > last_event:
                                events = tuple(application.events)
                                for event in events:
                                    if event["sequence"] > last_event:
                                        await sink.send(
                                            {
                                                "type": "runtime_event",
                                                "sequence": event["sequence"],
                                                "data": event,
                                            }
                                        )
                                        last_event = event["sequence"]
                                if journal:
                                    await asyncio.to_thread(journal.write_new, events)
                            last_content = messages[-1].get("content") if messages else None
                            visible_revision = (
                                len(messages),
                                last_content if isinstance(last_content, str) else None,
                                application.partial_transcript,
                                task_revision,
                                application.action_revision,
                                application.projection,
                            )
                            now = asyncio.get_running_loop().time()
                            if visible_revision != transcript_revision and (
                                transcript_revision is None
                                or application.projection != transcript_revision[-1]
                                or now - transcript_sent_at >= 0.25
                            ):
                                await sink.send(
                                    {"type": "transcript", "data": context_view(state, application)}
                                )
                                transcript_revision = visible_revision
                                transcript_sent_at = now
                            await asyncio.sleep(0.02)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logging.error("Native event delivery failed: %s", type(exc).__name__)
                        application.event("bridge_error", exception_type=type(exc).__name__)
                        await ws.close()

                event_publisher = asyncio.create_task(publish())
                sequence = 0
                async for message in ws:
                    if message.type == WSMsgType.BINARY:
                        if not message.data or len(message.data) > 8192 or len(message.data) % 2:
                            raise ValueError("invalid PCM")
                        if diagnostic:
                            diagnostic.received(message.data)
                        await application.feed(message.data, sequence)
                        sequence += 1
                    elif message.type == WSMsgType.TEXT:
                        value = json.loads(message.data)
                        if (
                            isinstance(value, dict)
                            and value.get("type") == "audio_ack"
                            and set(value).issubset(
                                {
                                    "type",
                                    "id",
                                    "schedule_gap_seconds",
                                    "queued_audio_seconds",
                                    "audio_context_state",
                                    "arrival_ms",
                                }
                            )
                            and set(value).issuperset({"type", "id"})
                        ):
                            schedule_gap = value.get("schedule_gap_seconds")
                            queued_audio = value.get("queued_audio_seconds")
                            arrival_ms = value.get("arrival_ms")
                            context_state = value.get("audio_context_state")
                            if (
                                type(value["id"]) is not int
                                or any(
                                    item is not None
                                    and (
                                        type(item) not in (int, float)
                                        or not math.isfinite(item)
                                        or item < 0
                                        or item > 120
                                    )
                                    for item in (schedule_gap, queued_audio)
                                )
                                or (
                                    arrival_ms is not None
                                    and (
                                        type(arrival_ms) not in (int, float)
                                        or not math.isfinite(arrival_ms)
                                        or arrival_ms < 0
                                        or arrival_ms > 86_400_000
                                    )
                                )
                                or (
                                    context_state is not None
                                    and context_state not in ("running", "suspended", "closed")
                                )
                            ):
                                continue
                            ack = sink.acknowledge_with_response(value["id"])
                            if ack is not None:
                                ack_seconds, response_id = ack
                                application.event(
                                    "playback_ack",
                                    packet_id=value["id"],
                                    response_id=response_id,
                                    latency_seconds=ack_seconds,
                                    pending_packets=len(sink.pending),
                                    schedule_gap_seconds=schedule_gap,
                                    queued_audio_seconds=queued_audio,
                                    audio_context_state=context_state,
                                    browser_arrival_ms=arrival_ms,
                                )
                        elif isinstance(value, dict) and value.get("type") in {
                            "music_ui_control",
                            "music_playback_observed",
                        }:
                            validator = (
                                music_ui_control_fields
                                if value["type"] == "music_ui_control"
                                else music_playback_observed_fields
                            )
                            application.event(value["type"], **validator(value))
                        elif (
                            isinstance(value, dict)
                            and set(value) == {"type", "command", "payload"}
                            and value.get("type") == "task_control"
                        ):
                            try:
                                result = await state.task_runtime.control(
                                    value["command"], value["payload"]
                                )
                                await sink.send(
                                    {
                                        "type": "task_control_result",
                                        "command": value["command"],
                                        "result": result,
                                    }
                                )
                            except Exception as exc:
                                code = getattr(exc, "code", "invalid_task_control")
                                await sink.send(
                                    {
                                        "type": "task_control_result",
                                        "command": value.get("command"),
                                        "error": code,
                                    }
                                )
                        elif isinstance(value, dict) and value.get("type") == "device_ui_action":
                            import re

                            if (
                                set(value) - {"type", "command_id", "action", "value"}
                                or not isinstance(value.get("command_id"), str)
                                or not re.fullmatch(r"ui-[a-f0-9]{32}", value["command_id"])
                                or not isinstance(value.get("action"), str)
                                or value.get("action")
                                not in {"light_on", "light_off", "set_color", "set_brightness"}
                                or ("value" in value and not isinstance(value["value"], str))
                            ):
                                raise ValueError("invalid device UI control")
                            command_id = value["command_id"]
                            arguments = {
                                key: value[key] for key in ("action", "value") if key in value
                            }
                            prior = state.ui_device_actions.get(command_id)
                            if prior is not None:
                                if prior[0] != arguments:
                                    raise ValueError(
                                        "device command ID reused with different arguments"
                                    )
                                result = dict(prior[1])
                                status = "duplicate"
                            else:
                                from .contracts import TurnKey
                                from .providers import ToolCall

                                call = ToolCall(command_id, "run_device_action", arguments)
                                result = await state.tasks.execute(
                                    TurnKey(
                                        session_id=state.session_id,
                                        turn_id=max(1, application.turn_id),
                                    ),
                                    call,
                                )
                                state.ui_device_actions[command_id] = (arguments, dict(result))
                                status = (
                                    "failed"
                                    if result.get("result_status") in {"unknown", "unavailable"}
                                    else "accepted"
                                )
                            await sink.send(
                                {
                                    "type": "device_action_ack",
                                    "status": status,
                                    "command_id": command_id,
                                    "message": result.get(
                                        "summary", "Device action outcome unavailable."
                                    ),
                                }
                            )
                        elif value == {"type": "interrupt"}:
                            await application.interrupt()
                        elif value == {"type": "arm"}:
                            if (
                                application.projection != "wake_listening"
                                or application.gate.active
                            ):
                                raise ValueError("manual arm requires wake standby")
                            application.gate.active = True
                            application.projection = "listening"
                            application.event("wake_manual_armed")
                            application._arm_followup_timer()
                        elif value == {"type": "stop"}:
                            await application.interrupt()
                            end_requested = True
                            break
                        else:
                            raise ValueError("unsupported native control")
                    elif message.type in {WSMsgType.ERROR, WSMsgType.CLOSED, WSMsgType.CLOSE}:
                        break
        except Exception as exc:
            logging.error(
                "Native bridge session error run=%s type=%s",
                application.run_id if application else "pre-session",
                type(exc).__name__,
            )
            if application:
                application.event("bridge_error", exception_type=type(exc).__name__)
            if not ws.closed:
                with suppress(ConnectionError, RuntimeError):
                    await ws.send_json(
                        {
                            "type": "error",
                            "error": type(exc).__name__,
                            "message": "Native voice failed; inspect local diagnostic and reconnect.",
                        }
                    )
        finally:
            if event_publisher:
                event_publisher.cancel()
                await asyncio.gather(event_publisher, return_exceptions=True)
            if publisher:
                try:
                    await publisher.__aexit__(None, None, None)
                except Exception as exc:
                    logging.exception(
                        "Native transcript archive failed run=%s type=%s",
                        application.run_id if application else "pre-session",
                        type(exc).__name__,
                    )
            if journal:
                try:
                    if application:
                        await asyncio.to_thread(journal.write_new, tuple(application.events))
                    journal.close()
                except (OSError, ValueError) as exc:
                    logging.error("Native event journal failed: %s", type(exc).__name__)
            if diagnostic:
                try:
                    await asyncio.to_thread(diagnostic.save)
                except (OSError, ValueError) as exc:
                    logging.error("Native audio diagnostic save failed: %s", type(exc).__name__)
            if sink:
                await sink.close()
            await ws.close()
            if end_requested:
                import uuid

                state.context.set_messages([])
                state.session_id = uuid.uuid4().hex
            busy = False
        return ws

    app.router.add_get("/", page)
    app.router.add_get("/audio-probe", audio_probe)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/snapshot", reports)
    app.router.add_get("/api/voice/capabilities", capability)
    app.router.add_get("/api/voice/ws", voice)
    app.router.add_route("*", "/api/voice/{tail:.*}", unsupported)
    return app


class NativeTranscriptView:
    """Existing publisher reads a projection; no second conversation history."""

    def __init__(self, state, application):
        self.resources, self.application = state, application
        self.pending_archives = []

    @property
    def revision(self):
        return self.application.event_count

    def snapshot(self):
        from .transcript import normalize_transcript

        return normalize_transcript(context_view(self.resources, self.application, archived=True))
