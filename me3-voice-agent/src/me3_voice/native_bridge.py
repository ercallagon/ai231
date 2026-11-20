"""Opt-in native engine behind the existing loopback browser entrypoint."""

import asyncio
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import re
import threading
import traceback
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit
import aiohttp
from aiohttp import ClientSession, ClientTimeout, web, WSMsgType

from .native_app import NativeApplication, NativeState
from .readiness import build_wake
from .dashboard import render, render_audio_probe, snapshot
from .artifacts import output_path, public_task_artifact_name
from .research import public_url
from .music import MusicUnavailable


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
    "rejection_rule",
    "route",
    "request_id",
    "llm_request_id",
    "response_id",
    "operation_id",
    "tool_call_id",
    "call_id",
    "owner_id",
    "audio_context_state",
    "start_evidence",
}


def _voice_readiness(speech_owner_live, services):
    required = ("asr", "gpt_oss", "tts")
    configured = bool(speech_owner_live) and all(
        isinstance(services.get(key), dict)
        and services[key].get("client_state") == "READY"
        and services[key].get("status") != "UNCONFIGURED"
        for key in required
    )
    ready = configured and all(services[key].get("status") != "DISCONNECTED" for key in required)
    return configured, ready


def _parse_wpctl_volume(output):
    """Return PipeWire's reported default-sink level, or None for unknown output."""
    match = re.search(r"(?:^|\s)Volume:\s*(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", output)
    if not match:
        return None
    value = float(match.group(1))
    return value if 0.0 <= value <= 1.0 else None


async def _pipewire_output_volume(level=None):
    """Read or adjust only the local PipeWire default sink via fixed wpctl args."""
    if level is not None:
        if type(level) not in (int, float) or not 0 <= level <= 1:
            return None
    try:
        if level is not None:
            process = await asyncio.create_subprocess_exec(
                "wpctl",
                "set-volume",
                "@DEFAULT_AUDIO_SINK@",
                f"{level:.2f}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                "wpctl",
                "get-volume",
                "@DEFAULT_AUDIO_SINK@",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
    except OSError:
        return None
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=2)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return None
    if process.returncode != 0:
        return None
    if level is not None:
        return await _pipewire_output_volume()
    return _parse_wpctl_volume(stdout.decode("utf-8", errors="replace"))


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
    if result.get("event") == "tavily_quick_research_finished":
        source_id = event.get("source_id")
        if isinstance(source_id, str) and re.fullmatch(r"S[0-9]{1,6}", source_id):
            result["research_source_id"] = source_id
        source_url = event.get("source_url")
        if isinstance(source_url, str):
            try:
                safe_url = public_url(source_url)
                if not any(
                    re.search(r"(?:key|token|secret|auth|signature|session|password|credential)", key, re.I)
                    for key, _ in parse_qsl(urlsplit(safe_url).query, max_num_fields=50)
                ):
                    result["research_source_url"] = safe_url
            except (ValueError, UnicodeError):
                pass
    if result.get("event") == "llm_citation_check":
        for key, pattern in (
            ("citation_ids", r"S[0-9]{1,6}"),
            ("retained_source_ids", r"S[0-9]{1,6}"),
            ("draft_source_ids", r"S[0-9]{1,6}"),
            ("draft_passage_ids", r"S[0-9]{1,6}-P[0-9]{1,6}"),
            ("retained_passage_ids", r"S[0-9]{1,6}-P[0-9]{1,6}"),
        ):
            values = event.get(key)
            if (
                isinstance(values, list)
                and len(values) <= 256
                and all(
                    isinstance(value, str) and re.fullmatch(pattern, value)
                    for value in values
                )
            ):
                result[key] = values
        citation_ids = result.get("citation_ids")
        references = event.get("citation_sources")
        if event.get("evidence_valid") is True and isinstance(citation_ids, list) and isinstance(references, list):
            safe_references = []
            seen = set()
            for reference in references[:16]:
                if not isinstance(reference, dict):
                    continue
                source_id = reference.get("source_id")
                try:
                    source_url = public_url(reference.get("url"))
                except (ValueError, UnicodeError):
                    continue
                try:
                    credential_query = any(
                        re.search(
                            r"(?:key|token|secret|auth|signature|session|password|credential)",
                            key,
                            re.I,
                        )
                        for key, _ in parse_qsl(urlsplit(source_url).query, max_num_fields=50)
                    )
                except ValueError:
                    continue
                if (
                    source_id not in citation_ids
                    or source_id in seen
                    or credential_query
                ):
                    continue
                seen.add(source_id)
                item = {"source_id": source_id, "url": source_url}
                title = reference.get("title")
                if isinstance(title, str):
                    item["title"] = " ".join(title.split())[:250]
                for key in ("evidence_status", "source_type"):
                    value = reference.get(key)
                    if isinstance(value, str) and re.fullmatch(r"[a-z0-9_]{1,48}", value):
                        item[key] = value
                for key in ("fetched", "truncated"):
                    if type(reference.get(key)) is bool:
                        item[key] = reference[key]
                safe_references.append(item)
            result["citation_sources"] = safe_references
            result["citation_source_mapping_complete"] = set(citation_ids) == seen
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


def _load_private_demo_owner_profile(path):
    """Load approved owner facts from a private, non-secret profile file."""
    path = Path(path)
    try:
        metadata = path.lstat()
        if (
            not path.is_file()
            or path.is_symlink()
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or metadata.st_size > 1024
        ):
            return {}
        profile = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(profile, dict) or "display_name" not in profile:
        return {}
    name = profile["display_name"]
    if (
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 100
        or any(ord(character) < 32 for character in name)
    ):
        return {}
    result = dict(profile)
    result["display_name"] = name.strip()
    if "current_role" in profile:
        role = profile["current_role"]
        if (
            not isinstance(role, str)
            or not role.strip()
            or len(role) > 100
            or any(ord(character) < 32 for character in role)
        ):
            return {}
        result["current_role"] = role.strip()
    return result


def _load_private_demo_owner_name(path):
    """Backward-compatible display-name accessor for the private profile."""
    return _load_private_demo_owner_profile(path).get("display_name", "")


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
    def config_digest(path):
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
        except OSError:
            return None

    configuration_profile = {
        "id": os.environ.get("ME3_PROFILE_ID") or None,
        "backend_sha256": config_digest(getattr(wake_args, "backend_config", None)),
        "runtime_sha256": config_digest(getattr(wake_args, "runtime_config", None)),
    }
    state = NativeState(
        backends,
        credentials,
        device_config=getattr(wake_args, "device_config", None),
        receipt_store=getattr(wake_args, "receipt_store", None),
    )
    state.owner_profile = _load_private_demo_owner_profile(
        os.environ.get("ME3_DEMO_OWNER_CONFIG", "var/cache/demo-owner.json")
    )
    state.owner_name = state.owner_profile.get("display_name", "")
    app = web.Application(client_max_size=16384)
    busy = False
    wake_configured = any(
        getattr(wake_args, name, None)
        for name in ("wake_model", "mel_model", "embedding_model", "kws_assets", "keywords_file")
    )
    wake_runtime = {"instance": None, "error": None}

    async def close(_app):
        await state.close()

    app.on_cleanup.append(close)

    async def initialize_services(_app):
        try:
            wake_runtime["instance"] = await asyncio.to_thread(build_wake, wake_args)
        except Exception as exc:
            wake_runtime["error"] = type(exc).__name__
        warm_runtime = getattr(state, "warm_runtime", None)
        if warm_runtime:
            await warm_runtime(options)

    app.on_startup.append(initialize_services)

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

    def health_snapshot():
        from .remote import estimated_provider_cost_usd, request_ledger_summary

        def service(
            status,
            detail,
            model=None,
            *,
            client_state=None,
            provider_state=None,
            inference_state=None,
            capability_state=None,
            checked_at=None,
        ):
            row = {"status": status, "detail": detail}
            if model:
                row["model"] = model
            if client_state:
                row["client_state"] = client_state
            if provider_state:
                row["provider_state"] = provider_state
            if inference_state:
                row["inference_state"] = inference_state
            if capability_state:
                row["capability_state"] = capability_state
            if checked_at:
                row["checked_at_utc"] = checked_at
            return row

        def credential_present(name):
            return bool(state.credentials.get(name) or os.environ.get(name))

        def provider_status(role):
            spec = backends[role]
            vendor = "deepgram" if role in {"asr", "tts"} else "openrouter"
            probe = state.provider_probes.get(vendor)
            if role in {"asr", "tts"}:
                client_state = "READY" if state.speech is not None else "UNINITIALIZED"
            elif role == "llm":
                client_state = (
                    "READY"
                    if getattr(getattr(state, "budget", None), "client", None) is not None
                    else "UNINITIALIZED"
                )
            else:
                client_state = "UNINITIALIZED"
            if spec.backend != "external":
                return service(
                    "DEGRADED", "Mock adapter selected; no external provider call.",
                    spec.model or "Mock", client_state=client_state,
                    provider_state="NOT_APPLICABLE",
                )
            if not spec.allow_live or spec.request_limit == 0:
                return service(
                    "UNCONFIGURED", "Live requests are disabled by the selected profile.",
                    spec.model, client_state=client_state, provider_state="NOT_ENABLED",
                )
            if not credential_present(spec.auth_env):
                return service(
                    "UNCONFIGURED", "The selected provider credential is not loaded.",
                    spec.model, client_state=client_state, provider_state="NOT_CONFIGURED",
                )
            if client_state != "READY":
                if probe and probe.get("provider_state") == "AUTHENTICATED_REACHABLE":
                    return service(
                        "DEGRADED",
                        "Provider account endpoint is authenticated/reachable, but its local client did not initialize and inference is untested.",
                        spec.model,
                        client_state=client_state,
                        provider_state="AUTHENTICATED_REACHABLE",
                        inference_state="NOT_PROBED",
                        checked_at=probe.get("checked_at_utc"),
                    )
                return service(
                    "DISCONNECTED",
                    "Local provider client did not initialize; the remote provider was not contacted.",
                    spec.model,
                    client_state=client_state,
                    provider_state="NOT_PROBED",
                )
            if probe:
                provider_state = probe.get("provider_state", "UNAVAILABLE")
                return service(
                    "DEGRADED" if provider_state == "AUTHENTICATED_REACHABLE" else "DISCONNECTED",
                    "Provider account endpoint is authenticated/reachable; selected model inference remains untested."
                    if provider_state == "AUTHENTICATED_REACHABLE"
                    else "Provider connectivity check failed; inspect the bounded error/status and retry when network access is restored.",
                    spec.model,
                    client_state=client_state,
                    provider_state=provider_state,
                    inference_state="NOT_PROBED",
                    checked_at=probe.get("checked_at_utc"),
                )
            return service(
                "DEGRADED",
                "Local client is initialized; remote provider has not been contacted by a health check.",
                spec.model,
                client_state=client_state,
                provider_state="NOT_PROBED",
            )

        haiku_client_state = (
            getattr(state.task_runtime, "provider_client_state", "UNINITIALIZED")
            if state.task_runtime is not None
            else "UNINITIALIZED"
        )
        openrouter_probe = state.provider_probes.get("openrouter")
        speech_owner_live = bool(
            state.speech is not None
            and state.owner is not None
            and not state.owner.done()
        )
        voice_detail = (
            "Local speech pipeline is initialized and its owner task is active."
            if speech_owner_live
            else "Speech pipeline is warming or failed; start will retry local initialization once."
        )
        services = {
            "bridge": service("READY", "Native loopback HTTP and WebSocket routes are serving."),
            "session": service(
                "READY" if speech_owner_live else "DEGRADED",
                "Voice session is active." if busy else voice_detail,
                client_state="READY" if speech_owner_live else "UNINITIALIZED",
                provider_state="LOCAL_ONLY",
            ),
            "me2_wake": service(
                "READY"
                if wake_runtime["instance"] is not None
                else "DISCONNECTED"
                if wake_runtime["error"]
                else "DEGRADED"
                if wake_configured
                else "UNCONFIGURED",
                "Local wake model loaded before bridge readiness; acoustic accuracy is not qualified."
                if wake_runtime["instance"] is not None
                else "Wake model initialization failed; inspect local startup diagnostics."
                if wake_runtime["error"]
                else "Wake assets are configured but not loaded."
                if wake_configured
                else "Wake assets are not configured.",
                client_state="READY" if wake_runtime["instance"] is not None else "UNINITIALIZED",
                provider_state="LOCAL_ONLY",
            ),
            "asr": provider_status("asr"),
            "gpt_oss": provider_status("llm"),
            "haiku": service(
                "DISCONNECTED"
                if options.task_model
                and credential_present(backends["llm"].auth_env)
                and haiku_client_state == "UNINITIALIZED"
                else "DEGRADED"
                if options.task_model and credential_present(backends["llm"].auth_env)
                else "UNCONFIGURED",
                "Local task model client is initialized; remote provider is not probed by health checks."
                if options.task_model and credential_present(backends["llm"].auth_env)
                else "No separate Haiku task model is configured or its credential is missing.",
                options.task_model or None,
                client_state=haiku_client_state,
                provider_state=(
                    openrouter_probe.get("provider_state")
                    if openrouter_probe
                    else "NOT_PROBED"
                ),
                inference_state="NOT_PROBED",
                checked_at=openrouter_probe.get("checked_at_utc") if openrouter_probe else None,
            ),
            "tts": provider_status("tts"),
            "tavily": service(
                "READY"
                if options.research.allow_live
                and credential_present("ME3_RESEARCH_TAVILY_API_KEY")
                and getattr(state.research, "operation_state", "NOT_PROBED") == "SEARCH_READY"
                else "DISCONNECTED"
                if options.research.allow_live
                and getattr(state.research, "provider_state", "") == "UNAVAILABLE"
                else "DEGRADED"
                if options.research.allow_live
                and (options.research.request_limit is None or options.research.request_limit > 0)
                and credential_present("ME3_RESEARCH_TAVILY_API_KEY")
                else "UNCONFIGURED",
                "A production basic search succeeded in this bridge process."
                if getattr(state.research, "operation_state", "NOT_PROBED") == "SEARCH_READY"
                else "Configured; search readiness is checked on demand. No paid health search is sent."
                if options.research.allow_live
                and (options.research.request_limit is None or options.research.request_limit > 0)
                and credential_present("ME3_RESEARCH_TAVILY_API_KEY")
                else "Research is disabled or its credential is not loaded.",
                client_state=(
                    "READY"
                    if getattr(getattr(state, "research", None), "client", None) is not None
                    else "UNINITIALIZED"
                ),
                provider_state=getattr(state.research, "provider_state", "NOT_PROBED"),
                capability_state=getattr(state.research, "operation_state", "NOT_PROBED"),
                checked_at=getattr(state.research, "last_checked_at", None),
            ),
            "tasks": service(
                "DISCONNECTED"
                if state.task_runtime is not None
                and getattr(state.task_runtime, "closed", False)
                else "READY"
                if state.task_runtime is not None and state.task_runtime.store is not None
                else "DEGRADED",
                "LangGraph checkpoint/workspace is initialized."
                if state.task_runtime is not None and state.task_runtime.store is not None
                else "Task store initializes during bridge startup."
                if not getattr(state, "startup_errors", {}).get("tasks")
                else "Task runtime startup failed; reconnect to retry and inspect local diagnostics.",
            ),
            "spotify": service(
                "DEGRADED" if music.get("configured") else "UNCONFIGURED",
                "Spotify is configured; an online receiver/player is not confirmed by this local check."
                if music.get("configured")
                else "Spotify OAuth/config is not loaded.",
            ),
            "calls_messages": service(
                "DEGRADED"
                if bool(getattr(getattr(state.tasks.actions, "communications", None), "enabled", False))
                else "UNCONFIGURED",
                "Phone adapter is configured; paired-device delivery is not checked here."
                if bool(getattr(getattr(state.tasks.actions, "communications", None), "enabled", False))
                else "No phone/SMS adapter is configured for this laptop profile.",
            ),
            "github": service(
                "DEGRADED"
                if getattr(state.task_runtime, "github_writer", None) is not None
                else "UNCONFIGURED",
                "GitHub writer is configured; authentication and repository visibility are checked before asking for approval."
                if getattr(state.task_runtime, "github_writer", None) is not None
                else "GitHub write token/repository configuration is absent; no write tool is advertised.",
                client_state=(
                    getattr(state.task_runtime.github_writer, "client_state", "INITIALIZED_UNVERIFIED")
                    if getattr(state.task_runtime, "github_writer", None) is not None
                    else "UNCONFIGURED"
                ),
                provider_state="NOT_PROBED",
            ),
            "gpio": service(
                "DEGRADED",
                "Laptop device actions are simulated; no GPIO hardware is opened."
                if devices.get("target", "mock") != "pi"
                else "Pi adapter is selected; physical output/readback is not verified by this check.",
            ),
        }
        if state.owner is not None and state.owner.done() and state.speech is not None:
            services["asr"] = service("DISCONNECTED", "Local speech MCP owner has stopped.", backends["asr"].model)
            services["tts"] = service("DISCONNECTED", "Local speech MCP owner has stopped.", backends["tts"].model)

        run_id = os.environ.get("ME3_REQUEST_RUN_ID", "")
        ledger = request_ledger_summary(run_id, os.environ.get("ME3_REQUEST_LEDGER"))
        rows = ledger.get("requests", [])

        def apply_inference_observation(service_name, spec):
            if not spec or not spec.model:
                return
            provider = urlsplit(spec.endpoint).hostname or ""
            observed = [
                row
                for row in rows
                if row.get("provider") == provider and row.get("model") == spec.model
            ]
            if not observed:
                return
            latest = max(observed, key=lambda row: row.get("updated_at") or 0)
            checked_at = services[service_name].get("checked_at_utc")
            if checked_at:
                try:
                    probe_epoch = datetime.fromisoformat(checked_at).timestamp()
                except ValueError:
                    probe_epoch = 0
                if (latest.get("updated_at") or 0) <= probe_epoch:
                    return
            outcome = latest.get("status")
            if outcome == "success":
                services[service_name].update(
                    status="READY",
                    provider_state="INFERENCE_READY",
                    inference_state="READY",
                    checked_at_utc=datetime.fromtimestamp(
                        latest["updated_at"], timezone.utc
                    ).isoformat()
                    if isinstance(latest.get("updated_at"), (int, float))
                    else None,
                    detail="Latest production inference request completed successfully.",
                )
            elif outcome == "failure":
                error = latest.get("error_code")
                provider_state = (
                    "AUTH_REJECTED"
                    if error == "provider_rejected"
                    else "REACHABLE_RATE_LIMITED"
                    if error == "provider_rate_limited"
                    else "UNAVAILABLE"
                )
                services[service_name].update(
                    status="DEGRADED",
                    provider_state=provider_state,
                    inference_state="FAILED",
                    detail="Latest production inference request failed; inspect the bounded error code.",
                )
            elif outcome in {"unknown", "dispatched_unknown"}:
                services[service_name].update(
                    status="DEGRADED",
                    provider_state="UNKNOWN_OUTCOME",
                    inference_state="UNKNOWN",
                    detail="Latest provider request outcome is unresolved; check the run ledger before retrying.",
                )

        for service_name, role in (("gpt_oss", "llm"), ("asr", "asr"), ("tts", "tts")):
            apply_inference_observation(service_name, backends[role])
        if options.task_model:
            task_spec = replace(backends["llm"], model=options.task_model)
            apply_inference_observation("haiku", task_spec)

        voice_configured, voice_ready = _voice_readiness(speech_owner_live, services)
        services["session"].update(
            status="READY" if voice_ready else "DEGRADED",
            detail=(
                "Voice services are locally initialized and required providers are configured."
                if voice_ready
                else "Start is unavailable until speech and the selected ASR, GPT-OSS and TTS services are ready."
            ),
        )

        def quota(model, limit):
            used = sum(row.get("model") == model for row in rows)
            return {
                "limit": limit,
                "used": used,
                "remaining": max(0, limit - used) if type(limit) is int else None,
                "limit_source": "application" if limit is not None else "provider/account",
            }

        actual_cost = 0.0
        cost_known = False
        for row in rows:
            cost = estimated_provider_cost_usd(
                row.get("model"), row.get("actual_input_tokens"), row.get("actual_output_tokens")
            )
            if cost is not None:
                actual_cost += cost
                cost_known = True
        research_used = (
            state.research.usage_count()
            if state.research and hasattr(state.research, "usage_count")
            else getattr(state.research, "requests", 0)
            if state.research
            else 0
        )
        budget = {
            "run_id": ledger.get("run_id"),
            "gpt_oss_requests": quota(backends["llm"].model, backends["llm"].request_limit),
            "haiku_requests": quota(options.task_model, options.task_request_limit) if options.task_model else None,
            "asr_requests": quota(backends["asr"].model, backends["asr"].request_limit),
            "tts_requests": quota(backends["tts"].model, backends["tts"].request_limit),
            "tavily_requests": {
                "limit": options.research.request_limit,
                "used": research_used,
                "remaining": (
                    max(0, options.research.request_limit - research_used)
                    if options.research.request_limit is not None
                    else None
                ),
                "limit_source": (
                    "application" if options.research.request_limit is not None else "provider/account"
                ),
            },
            "token_usage": ledger.get("usage", {}),
            "listed_price_estimate_from_actual_model_tokens_usd": round(actual_cost, 8)
            if cost_known
            else None,
            "vendor_billed_cost_usd": None,
            "cost_alert": {
                "threshold_usd": options.cost_alert_usd,
                "observed_openrouter_list_price_usd": round(actual_cost, 8)
                if cost_known
                else None,
                "state": "ALERT"
                if cost_known and actual_cost >= options.cost_alert_usd
                else "CLEAR"
                if cost_known
                else "NO_USAGE_DATA",
                "coverage": "OpenRouter actual-token list-price estimate only; excludes ASR, TTS and Tavily vendor billing.",
            },
        }
        return {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_profile": configuration_profile,
            "services": services,
            "budget": budget,
            "voice_configured": voice_configured,
            "voice_ready": voice_ready,
        }

    async def health(request):
        local_request(request)
        snapshot = health_snapshot()
        speech_ready = bool(
            state.speech is not None and state.owner is not None and not state.owner.done()
        )
        return web.json_response(
            {
                "bridge": "ready",
                "engine": "native",
                "voice_session": "active" if busy else "idle",
                "speech_services": "initialized" if speech_ready else "failed"
                if state.startup_errors.get("speech") or (state.owner and state.owner.done())
                else "lazy",
                "voice_ready": snapshot["voice_ready"],
                "voice_configured": snapshot["voice_configured"],
                "voice_startup_error": state.startup_errors.get("speech"),
                "session_resume": True,
                "resume_scope": "SQLite LangGraph checkpoint; task resume is explicit",
                "task_runtime": "ready" if state.task_runtime else "lazy",
                **snapshot,
            }
        )

    async def prepare_session(request):
        """Retry local service initialization once when the user starts a conversation."""
        local_request(request)
        warm_runtime = getattr(state, "warm_runtime", None)
        if warm_runtime:
            try:
                await asyncio.wait_for(warm_runtime(options), timeout=12)
            except asyncio.TimeoutError:
                state.startup_errors["speech"] = "startup_timeout"
        local_ready = bool(
            state.speech is not None
            and state.owner is not None
            and not state.owner.done()
            and not state.startup_errors.get("speech")
        )
        result = health_snapshot()
        ready = local_ready and result["voice_configured"]
        result.update(
            {
                "bridge": "ready",
                "engine": "native",
                "voice_ready": result["voice_ready"],
                "voice_configured": ready,
                "speech_services": "initialized" if ready else "failed",
                "voice_startup_error": state.startup_errors.get("speech"),
                "voice_session": "active" if busy else "idle",
                "session_resume": True,
                "resume_scope": "SQLite LangGraph checkpoint; task resume is explicit",
                "task_runtime": "ready" if state.task_runtime else "lazy",
            }
        )
        return web.json_response(result, status=200 if ready else 503)

    provider_probe_lock = asyncio.Lock()

    async def check_provider_connectivity(request):
        """Check account endpoints only; never invoke paid model/speech inference."""
        local_request(request)
        async with provider_probe_lock:
            now = time.monotonic()
            prior = state.provider_probes
            if prior and all(now - item.get("monotonic", 0) < 30 for item in prior.values()):
                return web.json_response(
                    {
                        "checks": {
                            key: {k: v for k, v in item.items() if k != "monotonic"}
                            for key, item in prior.items()
                        },
                        "tavily": "NOT_PROBED_COSTED_SEARCH_SKIPPED",
                        "inference": "NOT_PROBED",
                        "cached": True,
                    },
                    headers={"Cache-Control": "no-store"},
                )

            targets = (
                (
                    "openrouter",
                    "https://openrouter.ai/api/v1/key",
                    "Bearer",
                    state.credentials.get(backends["llm"].auth_env)
                    or os.environ.get(backends["llm"].auth_env, ""),
                ),
                (
                    "deepgram",
                    "https://api.deepgram.com/v1/projects",
                    "Token",
                    state.credentials.get(backends["asr"].auth_env)
                    or os.environ.get(backends["asr"].auth_env, ""),
                ),
            )
            timeout = ClientTimeout(total=5, connect=3)
            async with ClientSession(timeout=timeout, trust_env=False) as client:
                async def check(name, url, scheme, key):
                    checked_at = datetime.now(timezone.utc).isoformat()
                    if not key:
                        result = {
                            "provider_state": "NOT_CONFIGURED",
                            "inference_state": "NOT_PROBED",
                            "checked_at_utc": checked_at,
                            "monotonic": time.monotonic(),
                        }
                        state.provider_probes[name] = result
                        return name, result
                    try:
                        async with client.get(
                            url,
                            headers={"Authorization": f"{scheme} {key}"},
                            allow_redirects=False,
                        ) as response:
                            status = response.status
                        provider_state = (
                            "AUTHENTICATED_REACHABLE"
                            if 200 <= status < 300
                            else "AUTH_REJECTED"
                            if status in {401, 403}
                            else "REACHABLE_RATE_LIMITED"
                            if status == 429
                            else "UNAVAILABLE"
                        )
                        result = {
                            "provider_state": provider_state,
                            "inference_state": "NOT_PROBED",
                            "http_status": status,
                            "checked_at_utc": checked_at,
                            "monotonic": time.monotonic(),
                        }
                    except asyncio.CancelledError:
                        raise
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        result = {
                            "provider_state": "UNAVAILABLE",
                            "inference_state": "NOT_PROBED",
                            "error": "network_or_timeout",
                            "checked_at_utc": checked_at,
                            "monotonic": time.monotonic(),
                        }
                    state.provider_probes[name] = result
                    return name, result

                checks = dict(await asyncio.gather(*(check(*target) for target in targets)))
            return web.json_response(
                {
                    "checks": {
                        key: {k: v for k, v in item.items() if k != "monotonic"}
                        for key, item in checks.items()
                    },
                    "tavily": "NOT_PROBED_COSTED_SEARCH_SKIPPED",
                    "inference": "NOT_PROBED",
                    "cached": False,
                },
                headers={"Cache-Control": "no-store"},
            )

    devices = state.tasks.actions.capabilities()
    music = state.tasks.music.capabilities()
    last_music_command = 0.0

    output_volume_lock = asyncio.Lock()
    last_output_volume_update = 0.0

    async def output_volume(request):
        nonlocal last_output_volume_update
        local_request(request)
        if request.method == "POST":
            if request.headers.get("Origin") != "http://" + request.headers.get("Host", ""):
                raise web.HTTPForbidden()
            if request.content_type != "application/json":
                raise web.HTTPUnsupportedMediaType()
            try:
                value = await request.json()
            except (ValueError, json.JSONDecodeError):
                raise web.HTTPBadRequest(text="Invalid speaker volume") from None
            level = value.get("volume") if isinstance(value, dict) else None
            if (
                not isinstance(value, dict)
                or set(value) != {"volume"}
                or type(level) not in (int, float)
                or not 0.0 <= level <= 1.0
            ):
                raise web.HTTPBadRequest(text="Speaker volume must be between 0 and 1")
            now = time.monotonic()
            if now - last_output_volume_update < 0.15:
                raise web.HTTPTooManyRequests(text="Wait before changing speaker volume again")
            last_output_volume_update = now
            async with output_volume_lock:
                current = await _pipewire_output_volume(float(level))
        else:
            current = await _pipewire_output_volume()
        if current is None:
            return web.json_response(
                {"available": False, "scope": "local_default_output_sink", "volume": None},
                status=503,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(
            {"available": True, "scope": "local_default_output_sink", "volume": current},
            headers={"Cache-Control": "no-store"},
        )

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

    async def task_artifact(request):
        local_request(request)
        artifact_id = request.match_info["artifact_id"]
        name = request.match_info["name"]
        if not re.fullmatch(r"[a-f0-9]{24}", artifact_id) or not public_task_artifact_name(name):
            raise web.HTTPNotFound(text="Task artifact unavailable")
        task_runtime = state.task_runtime
        store = task_runtime.store if task_runtime else None
        task_id = "task_" + artifact_id
        try:
            metadata = await store.artifact_metadata(task_id, name) if store else None
            if not metadata:
                raise web.HTTPNotFound(text="Task artifact unavailable")
            artifact = await store.read_artifact(task_id, name)
            if (
                artifact["sha256"] != metadata["sha256"]
                or len(artifact["content"].encode("utf-8")) != metadata["size"]
            ):
                raise web.HTTPNotFound(text="Task artifact unavailable")
        except web.HTTPException:
            raise
        except Exception:
            raise web.HTTPNotFound(text="Task artifact unavailable") from None
        return web.Response(
            text=artifact["content"],
            content_type="text/plain",
            charset="utf-8",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": f'attachment; filename="{Path(name).name}"',
                "Content-Security-Policy": "default-src 'none'",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
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

    async def music_token(request):
        local_request(request)
        origin = request.headers.get("Origin")
        if origin and origin != "http://" + request.headers.get("Host", ""):
            raise web.HTTPForbidden()
        if request.headers.get("Sec-Fetch-Site") not in {None, "same-origin"}:
            raise web.HTTPForbidden()
        try:
            token = await state.tasks.music.browser_access_token()
        except MusicUnavailable as exc:
            return web.json_response(
                {"available": False, "code": exc.code, "message": str(exc)[:160]},
                status=409 if exc.code == "music_streaming_scope_required" else 503,
                headers={"Cache-Control": "no-store, private", "Pragma": "no-cache"},
            )
        return web.json_response(
            {"available": True, "access_token": token},
            headers={
                "Cache-Control": "no-store, private",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cross-Origin-Resource-Policy": "same-origin",
            },
        )

    async def music_status(request):
        local_request(request)
        try:
            result = await state.tasks.music.current_state()
        except MusicUnavailable as exc:
            return web.json_response(
                {
                    "configured": bool(getattr(state.tasks.music, "token", None)),
                    "status": "unavailable",
                    "code": exc.code,
                    "message": str(exc)[:160],
                    "audible": None,
                },
                status=503,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def music_device(request):
        local_request(request)
        if request.headers.get("Origin") != "http://" + request.headers.get("Host", ""):
            raise web.HTTPForbidden()
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            value = await request.json()
        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="Invalid player registration") from None
        if (
            not isinstance(value, dict)
            or set(value) != {"type", "device_id"}
            or value.get("type") not in {"ready", "not_ready"}
            or not isinstance(value.get("device_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value["device_id"])
        ):
            raise web.HTTPBadRequest(text="Invalid player registration")
        if value["type"] == "ready":
            try:
                state.tasks.music.register_browser_device(value["device_id"])
            except (ValueError, MusicUnavailable) as exc:
                raise web.HTTPConflict(text=str(exc)[:160]) from None
        else:
            state.tasks.music.unregister_browser_device(value["device_id"])
        return web.json_response(
            state.tasks.music.capabilities(),
            headers={"Cache-Control": "no-store, private"},
        )

    async def music_control(request):
        nonlocal last_music_command
        import time

        local_request(request)
        if request.headers.get("Origin") != "http://" + request.headers.get("Host", ""):
            raise web.HTTPForbidden()
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            value = await request.json()
        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="Invalid Spotify command") from None
        action = value.get("action") if isinstance(value, dict) else None
        fields = (
            {"command_id", "action", "position_ms"}
            if action == "seek"
            else {"command_id", "action", "volume_percent"}
            if action == "volume_set"
            else {"command_id", "action"}
        )
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or not isinstance(value.get("command_id"), str)
            or not re.fullmatch(r"ui-[a-f0-9]{32}", value["command_id"])
            or action
            not in {
                "pause", "resume", "stop", "next", "previous", "seek",
                "volume_up", "volume_down", "volume_set",
            }
            or action == "seek"
            and (type(value.get("position_ms")) is not int or not 0 <= value["position_ms"] <= 7_200_000)
            or action == "volume_set"
            and (type(value.get("volume_percent")) is not int or not 0 <= value["volume_percent"] <= 100)
        ):
            raise web.HTTPBadRequest(text="Invalid Spotify command")
        now = time.monotonic()
        if now - last_music_command < 0.2:
            raise web.HTTPTooManyRequests(text="Wait for the current Spotify command")
        last_music_command = now
        try:
            receipt = await state.tasks.music.ui_control(
                value["command_id"], action, value.get("position_ms"), value.get("volume_percent")
            )
            result = await state.tasks.music.current_state(force=True)
        except (MusicUnavailable, ValueError) as exc:
            return web.json_response(
                {
                    "available": False,
                    "code": getattr(exc, "code", "music_invalid_command"),
                    "message": str(exc)[:160],
                },
                status=503,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(
            {"available": True, "receipt": receipt, "state": result},
            headers={"Cache-Control": "no-store"},
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
        session_stage = "websocket_prepare"
        try:
            await ws.prepare(request)
            sink = BrowserSink(ws, track_audio_stops=True)
            session_stage = "websocket_hello"
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
                or hello.get("conversation_mode", "continuous")
                not in {"single", "followup", "extended", "continuous"}
                or hello.get("capture_source", "browser") not in {"browser", "pipewire"}
            ):
                raise ValueError("unsupported audio format")
            session_stage = "wake_initialization"
            wake = wake_runtime["instance"]
            if wake is not None and hasattr(wake, "reset"):
                wake.reset()
            if wake_runtime["error"]:
                raise ValueError("wake model initialization failed")
            if wake is None:
                raise ValueError("real wake model required for browser capture")
            session_stage = "native_application_startup"
            application = NativeApplication(
                state,
                sink,
                options=options,
                rate=hello["rate"],
                wake=wake,
                conversation_mode=hello.get("conversation_mode", "continuous"),
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

                    async def send_transcript(visible_revision, *, force=False):
                        nonlocal transcript_revision, transcript_sent_at
                        now = asyncio.get_running_loop().time()
                        if visible_revision != transcript_revision and (
                            force
                            or transcript_revision is None
                            or application.projection != transcript_revision[-1]
                            or now - transcript_sent_at >= 0.25
                        ):
                            await sink.send(
                                {"type": "transcript", "data": context_view(state, application)}
                            )
                            transcript_revision = visible_revision
                            transcript_sent_at = asyncio.get_running_loop().time()

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
                            last_content = messages[-1].get("content") if messages else None
                            visible_revision = (
                                len(messages),
                                last_content if isinstance(last_content, str) else None,
                                application.partial_transcript,
                                task_revision,
                                application.action_revision,
                                application.projection,
                            )
                            if application.event_count > last_event:
                                events = tuple(application.events)
                                for event in events:
                                    if event["sequence"] > last_event:
                                        if (
                                            event.get("event")
                                            in {
                                                "turn_completed",
                                                "turn_failed",
                                                "turn_error",
                                                "turn_interrupted",
                                            }
                                            and visible_revision != transcript_revision
                                        ):
                                            # The transcript send is awaited before the
                                            # terminal event enters the WebSocket stream.
                                            await send_transcript(visible_revision, force=True)
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
                            await send_transcript(visible_revision)
                            await asyncio.sleep(0.02)
                    except asyncio.CancelledError:
                        raise
                    except ConnectionError:
                        # A peer closing during an awaited WebSocket write is
                        # normal transport shutdown, not an application failure.
                        with suppress(ConnectionError, RuntimeError):
                            await ws.close()
                        return
                    except RuntimeError as exc:
                        if ws.closed:
                            return
                        logging.error("Native event delivery failed: %s", type(exc).__name__)
                        application.event("bridge_error", exception_type=type(exc).__name__)
                        await ws.close()
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
                            and set(value) == {"type", "id"}
                            and value["type"] == "audio_stop_ack"
                        ):
                            sent_at = sink.acknowledge_stop(value["id"])
                            if sent_at is not None:
                                application.event(
                                    "audio_queue_cleared_ack",
                                    stop_id=value["id"],
                                    audio_stop_send_started_at=sent_at,
                                )
                        elif (
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
                        elif (
                            isinstance(value, dict)
                            and set(value) == {"type", "route"}
                            and value.get("type") == "research_route"
                        ):
                            if value["route"] != "tavily_quick_research":
                                await sink.send(
                                    {
                                        "type": "research_route_result",
                                        "ok": False,
                                        "error": "unsupported_research_route",
                                    }
                                )
                                continue
                            try:
                                application.designate_next_tavily_quick_research()
                            except ValueError as exc:
                                await sink.send(
                                    {
                                        "type": "research_route_result",
                                        "ok": False,
                                        "error": str(exc),
                                    }
                                )
                            else:
                                await sink.send(
                                    {
                                        "type": "research_route_result",
                                        "ok": True,
                                        "route": "tavily_quick_research",
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
                        elif (
                            isinstance(value, dict)
                            and set(value) == {"type", "text"}
                            and value.get("type") == "test_text"
                        ):
                            await application.inject_test_text(value["text"])
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
            failure_stage = None
            worker_state = getattr(exc, "worker_state", None)
            cause = exc.__cause__ or exc.__context__
            cause_type = type(cause).__name__ if cause else None
            if application and application.startup_stage != "ready":
                failure_stage = (
                    getattr(exc, "error_stage", None) or application.startup_stage
                )
                job = application.job
                if worker_state is None and job is not None:
                    worker_state = "cancelled" if job.cancelled() else (
                        "stopped" if job.done() else "running"
                    )
            if failure_stage is None:
                failure_stage = session_stage
            logging.error(
                "Native bridge session error run=%s type=%s stage=%s cause=%s worker=%s frames=%s",
                application.run_id if application else "pre-session",
                type(exc).__name__,
                failure_stage,
                cause_type,
                worker_state,
                [
                    f"{Path(frame.filename).name}:{frame.name}:{frame.lineno}"
                    for frame in traceback.extract_tb(exc.__traceback__)[-8:]
                ],
            )
            if application:
                application.event(
                    "bridge_error",
                    exception_type=type(exc).__name__,
                    failure_stage=failure_stage,
                    cause_type=cause_type,
                    worker_state=worker_state,
                )
            if not ws.closed:
                with suppress(ConnectionError, RuntimeError):
                    await ws.send_json(
                        {
                            "type": "error",
                            "error": type(exc).__name__,
                            "failure_stage": failure_stage,
                            "cause_type": cause_type,
                            "worker_state": worker_state,
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
    app.router.add_post("/api/session/prepare", prepare_session)
    app.router.add_post("/api/health/provider-check", check_provider_connectivity)
    app.router.add_get("/api/audio/output-volume", output_volume)
    app.router.add_post("/api/audio/output-volume", output_volume)
    app.router.add_get("/api/snapshot", reports)
    app.router.add_get("/api/voice/tasks/{artifact_id}/artifacts/{name:.*}", task_artifact)
    app.router.add_get("/api/voice/capabilities", capability)
    app.router.add_get("/api/voice/music/token", music_token)
    app.router.add_get("/api/voice/music/status", music_status)
    app.router.add_post("/api/voice/music/device", music_device)
    app.router.add_post("/api/voice/music/control", music_control)
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
