"""Harness-side MCP speech client; provider code is never called here."""

import asyncio
from collections import deque
import math
import sys
import json
import os
import re
import tempfile
import time
from .remote import ProviderError
from dataclasses import asdict
from .providers import ASRResult
from contextlib import AsyncExitStack

from mcp import Client, StdioServerParameters
from mcp.client import stdio as mcp_stdio

from .contracts import FRAME_BYTES, TurnKey, decode_pcm, encode_pcm


def _safe_tts_diagnostics(value):
    """Copy only bounded timing/count fields returned by the owned speech MCP."""
    if not isinstance(value, dict) or value.get("status") not in {
        "running",
        "completed",
        "cancelled",
        "failed",
        "unknown",
    }:
        return None

    def numbers(source, fields, *, integers=()):
        if not isinstance(source, dict):
            return {}
        result = {}
        for name in fields:
            item = source.get(name)
            if name in integers:
                if type(item) is int and 0 <= item <= 10_000_000:
                    result[name] = item
            elif type(item) in (int, float) and math.isfinite(item) and item >= 0:
                result[name] = float(item)
        return result

    server = numbers(
        value.get("mcp"),
        {
            "open_monotonic",
            "first_tts_next_monotonic",
            "last_tts_next_monotonic",
            "first_pcm_monotonic",
            "last_pcm_monotonic",
            "audio_seconds",
        },
        integers={"tts_next_count", "pcm_chunks", "pcm_bytes"},
    )
    raw_mcp = value.get("mcp")
    raw_distribution = raw_mcp.get("tts_next_seconds", {}) if isinstance(raw_mcp, dict) else {}
    distribution = numbers(
        raw_distribution,
        {"p50_seconds", "p95_seconds", "max_seconds", "total_seconds"},
        integers={"sample_count", "retained_sample_count"},
    )
    if distribution:
        server["tts_next_seconds"] = distribution

    provider_source = value.get("provider")
    provider = {}
    if isinstance(provider_source, dict):
        status = provider_source.get("status")
        if status in {"running", "completed", "cancelled", "failed", "unknown"}:
            provider["status"] = status
        provider.update(
            numbers(
                provider_source,
                {
                    "request_started_monotonic",
                    "socket_ready_monotonic",
                    "first_pcm_monotonic",
                    "last_pcm_monotonic",
                    "provider_flushed_monotonic",
                    "completed_monotonic",
                    "synthesis_wall_seconds",
                    "provider_audio_seconds",
                },
                integers={
                    "submitted_characters",
                    "provider_pcm_packets",
                    "provider_pcm_bytes",
                },
            )
        )
        for name in ("failure_type", "error_code"):
            item = provider_source.get(name)
            if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", item):
                provider[name] = item
    return {"status": value["status"], "mcp": server, "provider": provider or None}


def _timing_summary(samples):
    values = sorted(samples)
    if not values:
        return {"sample_count": 0, "retained_sample_count": 0}

    def percentile(fraction):
        index = int(len(values) * fraction + 0.999999) - 1
        return values[max(0, min(len(values) - 1, index))]

    return {
        "sample_count": len(values),
        "retained_sample_count": len(values),
        "p50_seconds": percentile(0.50),
        "p95_seconds": percentile(0.95),
        "max_seconds": values[-1],
        "total_seconds": sum(values),
    }


def _server_params(kind, spec, transcript, delay=0, tts_delay=0, credentials=None):
    environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    for name in ("ME3_REQUEST_RUN_ID", "ME3_REQUEST_LEDGER"):
        if os.environ.get(name):
            environment[name] = os.environ[name]
    args = [
        "-B",
        "-m",
        "me3_voice.servers",
        kind,
        "--mock-transcripts-json",
        json.dumps(transcript if isinstance(transcript, (list, tuple)) else [transcript]),
        "--mock-delay",
        str(delay),
        "--mock-tts-delay",
        str(tts_delay),
    ]
    if spec:
        args += ["--backend-spec-json", json.dumps(asdict(spec))]
        if spec.backend == "external":
            value = (credentials or {}).get(spec.auth_env) or os.environ.get(spec.auth_env)
            if value:
                environment[spec.auth_env] = value
    return StdioServerParameters(command=sys.executable, args=args, env=environment)


async def preflight_mock_speech(backends, *, timeout=15, stderr_limit=4096):
    """Start each real MCP child independently; never calls ASR/TTS inference."""
    if any(backends[kind].backend != "mock" for kind in ("asr", "tts")):
        raise ValueError("speech MCP preflight must use mocked inference")
    results = []
    for kind in ("asr", "tts"):
        started = time.perf_counter()
        process_box = []
        original_spawn = mcp_stdio._create_platform_compatible_process

        async def capture_spawn(*args, **kwargs):
            process = await original_spawn(*args, **kwargs)
            process_box.append(process)
            return process

        mcp_stdio._create_platform_compatible_process = capture_spawn
        stderr = tempfile.TemporaryFile(mode="w+b", dir="var/scratch")
        error = None
        tools = []
        protocol = None
        try:
            spec = backends[kind]
            params = _server_params(kind, spec, "phase4 startup preflight")
            async with asyncio.timeout(timeout):
                async with Client(
                    mcp_stdio.stdio_client(params, errlog=stderr),
                    mode="legacy",
                    read_timeout_seconds=timeout,
                ) as client:
                    protocol = client.session.protocol_version
                    tools = sorted(tool.name for tool in (await client.list_tools()).tools)
                    expected = (
                        {"cancel", "asr_open", "asr_append", "asr_finish"}
                        if kind == "asr"
                        else {"cancel", "tts_open", "tts_next"}
                    )
                    if not expected <= set(tools):
                        raise RuntimeError(
                            f"missing required tools: {sorted(expected - set(tools))}"
                        )
                    if "speech_capabilities" in tools:
                        await client.call_tool("speech_capabilities", {})
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
        finally:
            mcp_stdio._create_platform_compatible_process = original_spawn
            process = process_box[-1] if process_box else None
            exit_code = process.returncode if process else None
            stderr.flush()
            stderr.seek(0)
            raw = stderr.read(stderr_limit).decode("utf-8", errors="replace")
            stderr.close()
        raw = re.sub(
            r"(?i)(api[_ -]?key|authorization|bearer|token)(\s*[:=]\s*)\S+",
            r"\1\2[REDACTED]",
            raw,
        )
        results.append(
            {
                "service": kind,
                "ok": error is None and exit_code == 0,
                "startup_seconds": time.perf_counter() - started,
                "exit_code": exit_code,
                "protocol_version": protocol,
                "tools": tools,
                "error": error,
                "stderr": raw[-stderr_limit:],
            }
        )
    return results


class SpeechClient:
    def __init__(
        self,
        timeout: float = 5,
        transcript="Mock question",
        delay: float = 0,
        tts_delay=0,
        backends=None,
        credentials=None,
    ):
        self.timeout, self.transcript, self.delay = timeout, transcript, delay
        self.tts_delay = tts_delay
        self.backends = backends
        self.credentials = credentials or {}
        self.startup_diagnostics = []
        self.last_tts_diagnostics = None

    async def __aenter__(self):
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        try:
            self.clients = {}
            self.capabilities = {}
            self.tools = {}
            for kind in ("asr", "tts"):
                service_started = time.monotonic()
                spec = self.backends[kind] if self.backends else None
                params = _server_params(
                    kind, spec, self.transcript, self.delay, self.tts_delay, self.credentials
                )
                self.clients[kind] = await self.stack.enter_async_context(
                    Client(
                        params,
                        mode="legacy",
                        read_timeout_seconds=max(
                            15,
                            min(self.timeout, spec.timeout_seconds if spec else self.timeout) + 2,
                        ),
                    )
                )
                initialized_at = time.monotonic()
                if not hasattr(self, "protocols"):
                    self.protocols = {}
                self.protocols[kind] = self.clients[kind].session.protocol_version
                listing = await self.clients[kind].list_tools()
                tools_ready_at = time.monotonic()
                expected = (
                    {"cancel", "asr_open", "asr_append", "asr_finish"}
                    if kind == "asr"
                    else {"cancel", "tts_open", "tts_next"}
                )
                self.tools[kind] = {tool.name for tool in listing.tools}
                if not expected <= self.tools[kind]:
                    raise ValueError(f"{kind} MCP discovery is missing required tools")
                self.capabilities[kind] = (
                    await self.call(kind, "speech_capabilities", {})
                    if "speech_capabilities" in self.tools[kind]
                    else {"contract_version": 1}
                )
                ready_at = time.monotonic()
                self.startup_diagnostics.append(
                    {
                        "service": kind,
                        "protocol_version": self.protocols[kind],
                        "tool_count": len(self.tools[kind]),
                        "mcp_initialize_seconds": initialized_at - service_started,
                        "tool_discovery_seconds": tools_ready_at - initialized_at,
                        "capability_read_seconds": ready_at - tools_ready_at,
                        "ready_monotonic": ready_at,
                    }
                )
            return self
        except BaseException:
            await self.stack.aclose()
            raise

    async def __aexit__(self, *exc):
        return await self.stack.__aexit__(*exc)

    async def call(self, kind: str, name: str, arguments: dict) -> dict:
        deadline = (
            min(self.timeout, self.backends[kind].timeout_seconds)
            if self.backends
            else self.timeout
        )
        try:
            async with asyncio.timeout(deadline):
                result = await self.clients[kind].call_tool(name, arguments)
        except TimeoutError:
            error = ProviderError(
                f"{kind} MCP call exceeded its configured deadline",
                code="provider_timeout",
                error_stage=name if re.fullmatch(r"[a-z0-9_]{1,48}", name) else "mcp_call",
                error_boundary="speech_mcp_client",
                error_detail="TimeoutError",
            )
            error.role = kind
            raise error from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = ProviderError(
                f"{kind} MCP transport failed; payload excluded",
                code="mcp_transport_error",
                role=kind,
                error_stage=name if re.fullmatch(r"[a-z0-9_]{1,48}", name) else "mcp_call",
                error_boundary="speech_mcp_client",
                error_detail=type(exc).__name__,
                cause_type=(
                    type(exc.__cause__ or exc.__context__).__name__
                    if exc.__cause__ or exc.__context__
                    else None
                ),
            )
            raise error from None
        if result.is_error:
            error = ProviderError(
                f"{kind} MCP tool failed; provider payload excluded",
                code="mcp_tool_error",
                error_stage=name if re.fullmatch(r"[a-z0-9_]{1,48}", name) else "mcp_call",
                error_boundary="speech_mcp_client",
                error_detail="MCPToolError",
            )
            error.role = kind
            raise error
        data = result.structured_content
        if not isinstance(data, dict):
            raise ValueError("missing structured speech result")
        failure = data.get("speech_error")
        if isinstance(failure, dict):
            code = failure.get("code")
            if (
                failure.get("role") != kind
                or not isinstance(code, str)
                or not re.fullmatch(r"[a-z0-9_]{1,48}", code)
            ):
                raise ValueError("invalid speech failure contract")
            exception_type = failure.get("exception_type")
            cause_type = failure.get("cause_type")
            if exception_type is not None and not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_.]{0,80}", exception_type
            ):
                raise ValueError("invalid speech failure diagnostic")
            if cause_type is not None and not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_.]{0,80}", cause_type
            ):
                raise ValueError("invalid speech failure diagnostic")
            stage = failure.get("stage")
            boundary = failure.get("boundary")
            if stage is not None and not re.fullmatch(r"[a-z0-9_]{1,48}", stage):
                raise ValueError("invalid speech failure diagnostic")
            if boundary is not None and not re.fullmatch(r"[a-z0-9_]{1,48}", boundary):
                raise ValueError("invalid speech failure diagnostic")
            error = ProviderError(
                f"{kind} provider failed; payload excluded",
                code=code,
                error_stage=stage,
                error_boundary=boundary or "speech_mcp",
                error_detail=exception_type,
                cause_type=cause_type,
            )
            error.role = kind
            error.remote_cause_type = cause_type
            error.tts_diagnostics = _safe_tts_diagnostics(failure.get("tts_diagnostics"))
            raise error
        return data

    async def open_operation(self, kind, name, arguments):
        # Shield the cheap control request so cancellation cannot orphan a slot
        # after the server creates it but before the client receives its identity.
        pending = asyncio.create_task(self.call(kind, name, arguments))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            try:
                opened = await pending
                if isinstance(opened.get("operation_id"), str):
                    await self.release(kind, opened["operation_id"])
            except Exception:
                pass
            raise

    async def release(self, kind, identifier):
        # A bounded, best-effort remote release; subprocess exit is the final cleanup.
        try:
            return await self.call(kind, "cancel", {"operation_id": identifier})
        except Exception:
            return None

    @staticmethod
    def check_key(key: TurnKey, result: dict):
        if any(result.get(name) != value for name, value in key.model_dump().items()):
            raise ValueError("stale or mismatched speech result")

    async def transcribe(self, key: TurnKey, pcm: bytes) -> str:
        async def chunks():
            for start in range(0, len(pcm), FRAME_BYTES):
                yield pcm[start : start + FRAME_BYTES]

        final = None
        async for result in self.transcribe_stream(key, chunks()):
            if result.final:
                final = result.text
        if final is None:
            raise ValueError("ASR stream ended without final result")
        return final

    async def transcribe_stream(self, key, chunks):
        result = await self.open_operation("asr", "asr_open", key.model_dump())
        identifier = result["operation_id"]
        revision = 0
        try:
            self.check_key(key, result)
            sequence = 0
            async for pcm in chunks:
                result = await self.call(
                    "asr",
                    "asr_append",
                    {"operation_id": identifier, "sequence": sequence, "pcm_b64": encode_pcm(pcm)},
                )
                self.check_key(key, result)
                sequence += 1
                if result.get("next_sequence") != sequence:
                    raise ValueError("ASR append sequence mismatch")
                partial = result.get("partial")
                if partial is not None:
                    if (
                        partial.get("final") is not False
                        or type(partial.get("revision")) is not int
                        or partial["revision"] <= revision
                        or not isinstance(partial.get("text"), str)
                    ):
                        raise ValueError("invalid ASR partial result")
                    revision = partial["revision"]
                    yield ASRResult(partial["text"])
            result = await self.call("asr", "asr_finish", {"operation_id": identifier})
            self.check_key(key, result)
            if (
                result.get("final") is not True
                or type(result.get("revision")) is not int
                or result["revision"] <= revision
                or not isinstance(result.get("text"), str)
            ):
                raise ValueError("invalid ASR final result")
            raw = result.get("provider_text")
            if raw is not None and (not isinstance(raw, str) or len(raw) > 8000):
                raise ValueError("invalid provider text boundary")
            yield ASRResult(result["text"], final=True, provider_text=raw)
        finally:
            await self.release("asr", identifier)

    async def synthesize(self, key: TurnKey, text: str):
        started = time.monotonic()
        open_started = started
        next_latencies = deque(maxlen=1024)
        pcm_chunks = pcm_bytes = next_count = 0
        first_pcm_at = last_pcm_at = None
        server_diagnostics = None
        identifier = None
        status = "failed"
        try:
            result = await self.open_operation(
                "tts", "tts_open", {**key.model_dump(), "text": text}
            )
            opened_at = time.monotonic()
            identifier = result["operation_id"]
            self.check_key(key, result)
            sequence = 0
            while True:
                next_started = time.monotonic()
                result = await self.call("tts", "tts_next", {"operation_id": identifier})
                next_latencies.append(time.monotonic() - next_started)
                next_count += 1
                self.check_key(key, result)
                diagnostic = _safe_tts_diagnostics(result.get("tts_diagnostics"))
                if diagnostic is not None:
                    server_diagnostics = diagnostic
                if result["done"]:
                    status = "completed"
                    return
                if result["sequence"] != sequence:
                    raise ValueError("out-of-order TTS chunk")
                sequence += 1
                pcm = decode_pcm(result["pcm_b64"])
                now = time.monotonic()
                if first_pcm_at is None:
                    first_pcm_at = now
                last_pcm_at = now
                pcm_chunks += 1
                pcm_bytes += len(pcm)
                yield pcm
        except (asyncio.CancelledError, GeneratorExit):
            status = "cancelled"
            raise
        except Exception as exc:
            status = "failed"
            error_diagnostics = _safe_tts_diagnostics(getattr(exc, "tts_diagnostics", None))
            if error_diagnostics is not None:
                server_diagnostics = error_diagnostics
            raise
        finally:
            release_diagnostics = None
            if identifier:
                released = await self.release("tts", identifier)
                if isinstance(released, dict):
                    release_diagnostics = _safe_tts_diagnostics(released.get("tts_diagnostics"))
            if server_diagnostics is None:
                server_diagnostics = release_diagnostics
            finished = time.monotonic()
            server = server_diagnostics["mcp"] if server_diagnostics is not None else {}
            provider = server_diagnostics["provider"] if server_diagnostics is not None else None
            self.last_tts_diagnostics = {
                "status": status,
                "client_mcp": {
                    "tts_open_started_monotonic": open_started,
                    "tts_open_completed_monotonic": locals().get("opened_at"),
                    "tts_open_seconds": (
                        opened_at - open_started if "opened_at" in locals() else None
                    ),
                    "tts_next_count": next_count,
                    "tts_next_latency_seconds": _timing_summary(next_latencies),
                    "first_native_client_pcm_monotonic": first_pcm_at,
                    "last_native_client_pcm_monotonic": last_pcm_at,
                    "pcm_chunks": pcm_chunks,
                    "pcm_bytes": pcm_bytes,
                    "audio_seconds": pcm_bytes / 32000,
                    "synthesis_wall_seconds": finished - started,
                },
                "mcp_server": server,
                "provider": provider,
            }
