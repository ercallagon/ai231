"""Sanitized, profile-aware projection of the live ME3 capability registrations.

This module only projects existing registries. It never dispatches a tool, starts
a provider, or treats credential presence as proof that a credential works.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def _provider(spec):
    if spec is None:
        return "local"
    if spec.backend == "mock":
        return "mock"
    try:
        host = urlsplit(spec.endpoint).hostname
    except ValueError:
        host = None
    return host or "external provider"


def _mode(spec):
    return "mock" if spec.backend == "mock" else "real_provider_selected"


def _credential_status(name, credentials, presence_overrides=None, *, required=True):
    if not name:
        return None
    override = (presence_overrides or {}).get(name)
    present = (
        bool(override)
        if type(override) is bool
        else bool(credentials.get(name) or os.environ.get(name))
    )
    return {
        "name": name,
        "required": required,
        "presence": "present_unvalidated" if present else "missing",
    }


def _availability(*, registered, configured, mode, operationally_verified):
    if not registered or not configured:
        return "unavailable"
    if operationally_verified:
        return "operationally_verified"
    if mode == "mock":
        return "mock_configured_unverified"
    return "configured_unverified"


def build_capability_report(
    *,
    backends,
    options,
    credentials=None,
    device_capabilities=None,
    device_details=None,
    music_capabilities=None,
    credential_presence=None,
    task_runtime=None,
    speech_client=None,
    wake_configured=False,
):
    """Build a serializable description from the registrations used at runtime."""
    from .conversation_tools import DEFINITIONS, schemas
    from .native_app import NativeApplication
    from .servers import speech_tool_names
    from .task_runtime import TaskRuntime, conversation_schemas

    credentials = credentials or {}
    credential_presence = credential_presence or {}
    device_capabilities = device_capabilities or {"target": "mock", "status": "simulated"}
    device_details = device_details or {}
    music_capabilities = music_capabilities or {"configured": False, "output_backend": "mock"}
    research = getattr(task_runtime, "research", None)
    if research is None:
        # Research options are the same profile gates used by NativeApplication.
        research_options = options.research
        research_requests = 0
        research_sources = ()
    else:
        research_options = research.options
        research_requests = research.requests
        research_sources = research.sources

    pipecat_registered_names = {
        schema.name for schema in NativeApplication.registered_tool_schemas()
    }
    pipecat_advertised_names = {
        schema.name
        for schema in NativeApplication.available_tool_schemas(
            research_options, research_requests, research_sources
        )
    }
    if task_runtime is None:
        # Construction is side-effect free; initialization opens SQLite and is
        # intentionally not part of a capability read.
        task_runtime = TaskRuntime(backends["llm"], credentials, research=None)
    task_tools = task_runtime.registered_tools()
    task_names = {tool.name for tool in task_tools}

    speech_tools = {}
    for role in ("asr", "tts"):
        speech_tools[role] = (
            set(speech_client.tools.get(role, ()))
            if speech_client is not None
            else set(speech_tool_names(role))
        )

    def provider_credential(role):
        spec = backends.get(role)
        if not spec or spec.backend != "external":
            return None
        return _credential_status(spec.auth_env, credentials, credential_presence)

    def provider_configured(role):
        spec = backends[role]
        if spec.backend == "mock":
            return True
        credential = provider_credential(role)
        return bool(
            spec.allow_live
            and spec.request_limit != 0
            and credential
            and credential["presence"] == "present_unvalidated"
        )

    research_credential = _credential_status(
        "ME3_RESEARCH_TAVILY_API_KEY", credentials, credential_presence
    )
    research_limit = research_options.request_limit
    research_enabled = bool(
        research_options.allow_live and (research_limit is None or research_limit > 0)
    )
    research_available = bool(
        research_enabled and (research_limit is None or research_requests < research_limit)
    )
    research_credential_present = research_credential["presence"] == "present_unvalidated"
    device_mode = device_capabilities.get("target", "mock")
    device_available = device_capabilities.get("available") is not False
    music_mode = "real_provider_selected" if music_capabilities.get("configured") else "mock"
    music_credentials = [
        _credential_status("ME3_SPOTIFY_ACCESS_TOKEN", credentials, credential_presence),
        _credential_status(
            "ME3_SPOTIFY_REFRESH_TOKEN", credentials, credential_presence, required=False
        ),
    ]

    rows = []

    def add(
        name,
        *,
        surface,
        implementation,
        provider,
        registered,
        configured,
        credentials_required=(),
        dependencies=(),
        permission_gate="none",
        mode="local",
        operationally_verified=False,
        verification_evidence="not_probed",
        advertised=None,
        selected_by=None,
    ):
        required = [entry for entry in credentials_required if entry is not None]
        row = {
            "id": f"{surface}:{name}",
            "name": name,
            "surface": surface,
            "implementation": implementation,
            "provider": provider,
            "registered": bool(registered),
            "advertised": advertised,
            "configured": bool(configured),
            "credential_requirements": required,
            "dependencies": list(dependencies),
            "permission_gate": permission_gate,
            "mode": mode,
            "selected_by": selected_by,
            "operationally_verified": bool(operationally_verified),
            "verification_evidence": verification_evidence,
        }
        row["availability"] = _availability(
            registered=row["registered"],
            configured=row["configured"],
            mode=mode,
            operationally_verified=row["operationally_verified"],
        )
        rows.append(row)

    for role, names in (("asr", speech_tools["asr"]), ("tts", speech_tools["tts"])):
        spec = backends[role]
        inference_tool_names = (
            {"asr_open", "asr_append", "asr_finish", "asr_append_batch", "asr_events"}
            if role == "asr"
            else {"tts_open", "tts_next"}
        )
        credential = provider_credential(role)
        service_operational = bool(speech_client and role in speech_client.tools)
        for name in sorted(names):
            requires_inference = name in inference_tool_names
            configured = (
                (spec.backend == "mock" or bool(spec.allow_live and spec.request_limit != 0))
                if requires_inference
                else True
            )
            if requires_inference and credential and credential["presence"] == "missing":
                configured = False
            add(
                name,
                surface=f"project_owned_{role}_mcp",
                implementation="SpeechService / MCP stdio",
                provider=_provider(spec) if requires_inference else "local MCP control",
                registered=True,
                configured=configured,
                credentials_required=(credential,) if requires_inference else (),
                dependencies=(
                    f"{spec.protocol or 'speech MCP contract'}",
                    "local stdio subprocess",
                ),
                permission_gate=(
                    "allow_live and request_limit is not 0"
                    if spec.backend == "external"
                    else "none"
                ),
                mode=_mode(spec) if requires_inference else "local",
                operationally_verified=False,
                advertised=name in speech_tools[role],
                verification_evidence="provider inference has not been exercised by this report",
            )
            rows[-1]["service_operationally_verified"] = service_operational
            rows[-1]["service_verification_evidence"] = (
                "MCP initialize, tool discovery and speech_capabilities completed; "
                "provider inference is not implied"
                if service_operational
                else "MCP subprocess has not been initialized in this process"
            )

    add(
        "fast_command_router",
        surface="native_fast_route",
        implementation="NativeApplication.fast_action; shared device/music handlers",
        provider="local parser and configured adapters",
        registered=callable(getattr(NativeApplication, "fast_action", None)),
        configured=bool(device_available or music_capabilities.get("configured")),
        dependencies=(
            f"device adapter: {device_mode}",
            f"music output: {music_capabilities.get('output_backend', 'mock')}",
        ),
        permission_gate=(
            "device configuration; consequential actions retain existing approval gates"
        ),
        mode=(
            "hardware_configured_unverified"
            if device_mode == "pi"
            else "real_provider_selected"
            if music_capabilities.get("configured")
            else "mock"
        ),
        operationally_verified=False,
        verification_evidence=(
            "route is registered; runtime action receipts determine actual outcome"
        ),
    )

    add(
        "browser_microphone_capture",
        surface="browser_ui",
        implementation="existing browser WebRTC/audio worklet input",
        provider="browser and selected microphone",
        registered=True,
        configured=True,
        dependencies=("browser microphone permission", "secure context or localhost"),
        permission_gate="user grants browser microphone permission",
        mode="real browser device",
        verification_evidence=(
            "browser permission/device behavior is not probed by this server report"
        ),
    )
    add(
        "browser_audio_playback",
        surface="browser_ui",
        implementation="existing acknowledged PCM16 playback sink",
        provider="browser audio output device",
        registered=True,
        configured=True,
        dependencies=("browser audio context", "selected speaker/output device"),
        permission_gate="browser playback activation and output-device selection",
        mode="real browser device",
        verification_evidence=(
            "packet acknowledgments are software evidence; audible output is not verified"
        ),
    )
    add(
        "wakeword_gate",
        surface="native_pipecat",
        implementation="existing configured KWS/WakeGate",
        provider="local wake model" if wake_configured else "none",
        registered=True,
        configured=bool(wake_configured),
        dependencies=("configured wake assets and keywords",)
        if wake_configured
        else ("wake assets not configured",),
        permission_gate="wake admission gate; user Arm remains an existing fallback",
        mode="local_model" if wake_configured else "unavailable",
        verification_evidence=(
            "asset configuration is presence only; wake accuracy requires acoustic qualification"
        ),
    )

    for schema in (*schemas(), *conversation_schemas()):
        name = schema.name
        role = "llm"
        provider = "local Pipecat handler"
        required_credentials = (provider_credential(role),)
        configured = provider_configured(role)
        dependencies = ["Pipecat function calling"]
        permission = "native tool validation and existing action receipt/idempotency checks"
        mode = "local"
        if name == "get_runtime_time":
            provider = "local runtime clock"
        if name in {"web_search", "retrieve_source", "verify_evidence"}:
            role = "research"
            provider = (
                "Tavily"
                if name == "web_search"
                else "public HTTP source"
                if name == "retrieve_source"
                else "local retained evidence"
            )
            required_credentials = (
                (provider_credential("llm"), research_credential)
                if name == "web_search"
                else (provider_credential("llm"),)
            )
            configured = (
                provider_configured("llm")
                and research_available
                and (research_credential_present if name == "web_search" else True)
            )
            dependencies = ["research.allow_live", "provider/account quota"]
            permission = (
                "research provider/account quota"
                if name == "web_search"
                else "public URL validation and retained-source contract"
            )
            mode = (
                "real_provider_selected"
                if name == "web_search" and configured
                else "network_source_retrieval"
                if name == "retrieve_source" and configured
                else "local_evidence_check"
                if name == "verify_evidence" and configured
                else "unavailable"
            )
        elif name in {"run_device_action", "set_mock_light", "read_mock_light"}:
            provider = (
                "ME3 simulated device adapter"
                if name in {"set_mock_light", "read_mock_light"}
                else f"ME3 {device_mode} device adapter"
            )
            configured = provider_configured("llm") and (
                True if name in {"set_mock_light", "read_mock_light"} else device_available
            )
            mode = (
                "mock"
                if name in {"set_mock_light", "read_mock_light"}
                else "hardware_configured_unverified"
                if device_mode == "pi" and device_available
                else "unavailable"
                if device_mode == "pi"
                else "mock"
            )
            dependencies = (
                ["in-memory simulation"]
                if name in {"set_mock_light", "read_mock_light"}
                else [
                    "device configuration",
                    "GPIO hardware" if device_mode == "pi" else "none; in-memory simulation",
                ]
            )
            permission = (
                "simulation only"
                if name in {"set_mock_light", "read_mock_light"}
                else "explicit device profile; physical output has no sensor readback"
            )
        elif name == "control_music":
            provider = "Spotify" if music_capabilities.get("configured") else "ME3 simulated music"
            configured = provider_configured("llm")
            dependencies = [
                "Spotify OAuth/configuration and output device"
                if music_capabilities.get("configured")
                else "mock adapter"
            ]
            permission = "existing music configuration and OAuth scopes"
            mode = music_mode
            required_credentials = (
                provider_credential("llm"),
                _credential_status(
                    "ME3_SPOTIFY_ACCESS_TOKEN",
                    credentials,
                    credential_presence,
                    required=bool(music_capabilities.get("configured")),
                ),
                music_credentials[1],
            )
        elif name in {"start_task", "task_status", "steer_task"}:
            provider = "LangGraph task runtime"
            configured = provider_configured("llm")
            mode = "local"
            dependencies = ["local SQLite checkpoint", "confined task workspace"]
            permission = "task ID and revision checks; external effects require explicit approval"
        registered = name in pipecat_registered_names
        add(
            name,
            surface="native_pipecat",
            implementation="NativeApplication.register_tools",
            provider=provider,
            registered=registered,
            configured=configured,
            credentials_required=required_credentials,
            dependencies=dependencies,
            permission_gate=permission,
            mode=mode,
            operationally_verified=False,
            advertised=name in pipecat_advertised_names,
            selected_by={
                "provider": _provider(backends["llm"]),
                "model": backends["llm"].model or "mock",
                "mode": _mode(backends["llm"]),
            },
            verification_evidence=(
                "registered in runtime schema; this report does not execute the capability"
            ),
        )
        if name == "run_device_action":
            rows[-1]["supported_actions"] = list(DEFINITIONS[name][1]["action"].get("enum", ()))
            rows[-1]["action_modes"] = {
                action: (
                    "configured_unverified"
                    if action == "get_weather" and device_details.get("weather_enabled")
                    else "unavailable"
                    if action == "get_weather"
                    else "configured_unverified"
                    if action in {"call_demo", "message_demo"}
                    and device_details.get("communications_enabled")
                    else "unavailable"
                    if action in {"call_demo", "message_demo"}
                    else "local_runtime"
                    if action == "get_time"
                    else "simulated"
                    if action == "set_temperature"
                    else "adapter_configured_unverified"
                    if device_mode == "pi" and device_available
                    else "unavailable"
                    if device_mode == "pi"
                    else "simulated"
                )
                for action in rows[-1]["supported_actions"]
            }
            rows[-1]["action_dependencies"] = {
                "get_weather": (
                    "enabled Open-Meteo config and network access"
                    if device_details.get("weather_enabled")
                    else "Open-Meteo is not configured"
                ),
                "call_demo": (
                    "Pi mode, owner-only communications config, paired phone and "
                    "OS telephony support"
                ),
                "message_demo": (
                    "Pi mode, owner-only communications config, paired phone and "
                    "OS messaging support"
                ),
                "set_temperature": "display-only simulated value; no thermostat adapter",
            }
        elif name == "control_music":
            rows[-1]["supported_actions"] = list(DEFINITIONS[name][1]["action"].get("enum", ()))

    for tool in task_tools:
        name = tool.name
        role = "llm"
        provider = "LangGraph local task/tool runtime"
        required_credentials = (provider_credential(role),)
        configured = provider_configured("llm")
        dependencies = ["durable SQLite checkpoint", "task-confined workspace"]
        permission = "task state/revision guard"
        mode = "local"
        if name in {"web_search", "retrieve_source", "verify_evidence"}:
            provider = (
                "Tavily"
                if name == "web_search"
                else "public HTTP source"
                if name == "retrieve_source"
                else "local retained evidence"
            )
            required_credentials = (
                (provider_credential("llm"), research_credential)
                if name == "web_search"
                else (provider_credential("llm"),)
            )
            configured = (
                provider_configured("llm")
                and research_available
                and (research_credential_present if name == "web_search" else True)
            )
            mode = (
                "real_provider_selected"
                if name == "web_search" and configured
                else "network_source_retrieval"
                if name == "retrieve_source" and configured
                else "local_evidence_check"
                if name == "verify_evidence" and configured
                else "unavailable"
            )
            dependencies = ["research allow_live", "provider/account quota"]
            permission = "provider/account quota and URL/source verification contract"
        elif name == "request_external_action":
            provider = "mock approval-gated dispatcher"
            mode = "mock"
            dependencies = ["LangGraph interrupt/resume", "mock dispatcher"]
            permission = "explicit task approval; runtime is mock-only"
        add(
            name,
            surface="langgraph_task",
            implementation="TaskRuntime._build_tools / compiled ToolNode",
            provider=provider,
            registered=name in task_names,
            configured=configured,
            credentials_required=required_credentials,
            dependencies=dependencies,
            permission_gate=permission,
            mode=mode,
            operationally_verified=False,
            advertised=True,
            selected_by={
                "provider": _provider(backends["llm"]),
                "model": backends["llm"].model or "mock",
                "mode": _mode(backends["llm"]),
            },
            verification_evidence=(
                "task store initialized; tool execution is not exercised by this report"
                if task_runtime.store
                else "task runtime is lazy; no database was opened and no tool was executed"
            ),
        )

    report = {
        "schema_version": 1,
        "kind": "swift-me3-capability-report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "native",
        "profile": {
            "provider_modes": {role: spec.backend for role, spec in backends.items()},
            "provider_models": {role: spec.model or "mock" for role, spec in backends.items()},
            "provider_limits": {
                role: {
                    "allow_live": spec.allow_live,
                    "request_limit": spec.request_limit,
                    "timeout_seconds": spec.timeout_seconds,
                    "protocol": spec.protocol or "mock",
                    "max_audio_seconds": spec.max_audio_seconds,
                    "max_input_tokens_per_request": spec.max_input_tokens or None,
                    "max_input_tokens_aggregate": spec.max_input_tokens_total or None,
                }
                for role, spec in backends.items()
            },
            "research_enabled": research_enabled,
            "research_request_limit": research_options.request_limit,
            "research_requests_used": research_requests,
            "research_requests_remaining": (
                max(0, research_limit - research_requests) if research_limit is not None else None
            ),
            "device_mode": device_mode,
            "weather_enabled": bool(device_details.get("weather_enabled")),
            "communications_enabled": bool(device_details.get("communications_enabled")),
            "music_mode": music_mode,
            "task_storage": {
                "database": "ME3_TASK_DATABASE configured"
                if os.environ.get("ME3_TASK_DATABASE")
                else "project-local default",
                "workspace": "ME3_TASK_WORKSPACE configured"
                if os.environ.get("ME3_TASK_WORKSPACE")
                else "project-local default",
            },
            "wake_configured": bool(wake_configured),
            "credential_values_included": False,
        },
        "runtime_checks": {
            "asr_mcp_initialized": bool(speech_client and "asr" in speech_client.tools),
            "tts_mcp_initialized": bool(speech_client and "tts" in speech_client.tools),
            "task_store_initialized": bool(task_runtime.store),
            "provider_inference_performed": False,
            "hardware_effects_performed": False,
        },
        "capabilities": rows,
    }
    return report


def render_capability_markdown(report):
    """Deck-friendly compact summary generated from the same JSON projection."""
    lines = [
        "# Swift ME3 Capability Profile",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "Credential presence means only that a value was loaded; it is not provider validation.",
        "real_provider_selected is profile intent only. allow_live, credentials, and "
        "successful requests are separate checks.",
        "Operational verification is reported only when the current process has direct evidence.",
        "",
        "## Active model profile",
        "",
        "| Role | Mode | Model | Request cap | Live enabled |",
        "|---|---|---|---:|---:|",
    ]
    for role in ("asr", "llm", "tts"):
        limits = report["profile"]["provider_limits"][role]
        lines.append(
            f"| {role} | {report['profile']['provider_modes'][role]} | "
            f"{report['profile']['provider_models'][role]} | "
            f"{limits['request_limit']} | {limits['allow_live']} |"
        )
    lines.extend(
        (
            "",
            "| Capability | Surface | Provider | Mode | Registered | Advertised | "
            "Configured | Verified | Service check | Availability |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        )
    )
    for row in report["capabilities"]:
        values = (
            row["name"],
            row["surface"],
            row["provider"],
            row["mode"],
            str(row["registered"]),
            str(row["advertised"]),
            str(row["configured"]),
            str(row["operationally_verified"]),
            str(row.get("service_operationally_verified", "—")),
            row["availability"],
        )
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    lines.extend(("", "## Dependencies and gates", ""))
    for row in report["capabilities"]:
        deps = ", ".join(row["dependencies"]) or "none"
        gates = row["permission_gate"] or "none"
        creds = (
            ", ".join(
                f"{item['name']} ({item['presence']})" for item in row["credential_requirements"]
            )
            or "none"
        )
        lines.append(
            f"- **{row['name']}** — credentials: {creds}; dependencies: {deps}; gate: {gates}."
        )
    return "\n".join(lines) + "\n"


def export_capability_report(report, output_stem):
    """Write exclusive, private JSON and Markdown artifacts under project var/."""
    stem = Path(output_stem).resolve()
    var_root = (Path.cwd() / "var").resolve()
    if not stem.is_relative_to(var_root):
        raise ValueError("capability report output must stay under var/")
    paths = (stem.with_suffix(".json"), stem.with_suffix(".md"))
    if any(path.exists() for path in paths):
        raise FileExistsError("preserve existing capability report; choose a fresh output stem")
    payloads = (
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        render_capability_markdown(report),
    )
    for path, payload in zip(paths, payloads, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
    return paths
