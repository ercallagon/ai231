"""Harness-side MCP speech client; provider code is never called here."""

import asyncio
import sys
import json
import os
import re
from .remote import ProviderError
from dataclasses import asdict
from .providers import ASRResult
from contextlib import AsyncExitStack

from mcp import Client, StdioServerParameters

from .contracts import FRAME_BYTES, TurnKey, decode_pcm, encode_pcm


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

    async def __aenter__(self):
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        try:
            self.clients = {}
            for kind in ("asr", "tts"):
                spec = self.backends[kind] if self.backends else None
                environment = {"PYTHONDONTWRITEBYTECODE": "1"}
                if spec and spec.backend == "external":
                    value = self.credentials.get(spec.auth_env) or os.environ.get(spec.auth_env)
                    if value:
                        environment[spec.auth_env] = value
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-B",
                        "-m",
                        "me3_voice.servers",
                        kind,
                        "--mock-transcripts-json",
                        json.dumps(
                            self.transcript
                            if isinstance(self.transcript, (list, tuple))
                            else [self.transcript]
                        ),
                        "--mock-delay",
                        str(self.delay),
                        "--mock-tts-delay",
                        str(self.tts_delay),
                    ]
                    + (["--backend-spec-json", json.dumps(asdict(spec))] if spec else []),
                    env=environment,
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
                if not hasattr(self, "protocols"):
                    self.protocols = {}
                self.protocols[kind] = self.clients[kind].session.protocol_version
                listing = await self.clients[kind].list_tools()
                expected = (
                    {"cancel", "asr_open", "asr_append", "asr_finish"}
                    if kind == "asr"
                    else {"cancel", "tts_open", "tts_next"}
                )
                if not expected <= {tool.name for tool in listing.tools}:
                    raise ValueError(f"{kind} MCP discovery is missing required tools")
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
                f"{kind} MCP call exceeded its configured deadline", code="provider_timeout"
            )
            error.role = kind
            raise error from None
        if result.is_error:
            error = ProviderError(f"{kind} MCP tool failed; provider payload excluded")
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
            error = ProviderError(f"{kind} provider failed; payload excluded", code=code)
            error.role = kind
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
            await self.call(kind, "cancel", {"operation_id": identifier})
        except Exception:
            pass

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
        result = await self.open_operation("tts", "tts_open", {**key.model_dump(), "text": text})
        identifier = result["operation_id"]
        try:
            self.check_key(key, result)
            sequence = 0
            while True:
                result = await self.call("tts", "tts_next", {"operation_id": identifier})
                self.check_key(key, result)
                if result["done"]:
                    return
                if result["sequence"] != sequence:
                    raise ValueError("out-of-order TTS chunk")
                sequence += 1
                yield decode_pcm(result["pcm_b64"])
        finally:
            await self.release("tts", identifier)
