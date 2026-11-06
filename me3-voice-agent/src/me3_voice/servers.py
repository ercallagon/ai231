"""Our speech MCP servers: stdio only, pull-based TTS, bounded utterance ASR."""

import argparse
import asyncio
import secrets
import time
import json
from dataclasses import dataclass, field
from typing import Any
from functools import wraps
import re

from .remote import ProviderError
from .english import EnglishTextRejected

from mcp.server import MCPServer

from .contracts import MAX_AUDIO_BYTES, TurnKey, decode_pcm, encode_pcm
from .providers import ASRProvider, ASRResult, BufferedASRStream, MockASR, MockTTS, TTSProvider


@dataclass
class Operation:
    key: TurnKey
    touched: float
    audio: bytearray = field(default_factory=bytearray)
    iterator: object = None
    sequence: int = 0
    busy: bool = False
    asr_stream: object = None
    worker: object = None
    revision: int = 0
    closed: bool = False
    output_bytes: int = 0


class SpeechService:
    """Single-client server; one operation, 30 s input, 60 s lazy lease expiry.

    No background producer: each TTS next pulls exactly one chunk. Closing or
    cancelling always releases the generator; process shutdown is the final fence.
    """

    def __init__(self, asr: ASRProvider, tts: TTSProvider, clock=time.monotonic):
        self.asr, self.tts, self.clock = asr, tts, clock
        self.operations: dict[str, Operation] = {}

    async def dispose(self, op):
        if op.closed:
            return
        op.closed = True
        if op.worker and not op.worker.done():
            op.worker.cancel()
            await asyncio.gather(op.worker, return_exceptions=True)
        if op.iterator:
            await op.iterator.aclose()
        if op.asr_stream:
            await op.asr_stream.aclose()

    async def cancel(self, operation_id: str) -> dict[str, Any]:
        op = self.operations.pop(operation_id, None)
        if op:
            await self.dispose(op)
        return {"cancelled": op is not None}

    async def close(self):
        for identifier in list(self.operations):
            await self.cancel(identifier)
        for provider in (self.asr, self.tts):
            if hasattr(provider, "aclose"):
                await provider.aclose()

    async def work(self, identifier, op, coroutine):
        op.busy = True
        op.worker = asyncio.ensure_future(coroutine)
        try:
            result = await op.worker
            if self.operations.get(identifier) is not op:
                raise ValueError("operation cancelled")
            return result
        except asyncio.CancelledError:
            if self.operations.get(identifier) is not op:
                raise ValueError("operation cancelled") from None
            raise
        finally:
            op.busy = False

    def recognition(self, op, result, final=False):
        if result is None and not final:
            return None
        if (
            not isinstance(result, ASRResult)
            or type(result.text) is not str
            or len(result.text) > 8000
            or result.final is not final
        ):
            raise ValueError("invalid ASR provider result")
        op.revision += 1
        row = {"text": result.text, "final": result.final, "revision": op.revision}
        if result.provider_text is not None:
            if not isinstance(result.provider_text, str) or len(result.provider_text) > 8000:
                raise ValueError("invalid raw ASR text")
            row["provider_text"] = result.provider_text
        return row

    async def open(self, session_id: str, turn_id: int, text: str | None = None) -> dict[str, Any]:
        key = TurnKey(session_id=session_id, turn_id=turn_id)
        for identifier, op in list(self.operations.items()):
            if self.clock() - op.touched >= 60:
                await self.cancel(identifier)
        if self.operations:
            raise ValueError("operation already active; cancel or finish first")
        if text is not None and (not text.strip() or len(text) > 500):
            raise ValueError("TTS text must contain 1..500 characters")
        identifier = secrets.token_hex(16)
        self.operations[identifier] = Operation(
            key,
            self.clock(),
            iterator=self.tts.synthesize(text) if text is not None else None,
            asr_stream=(
                self.asr.open_stream()
                if hasattr(self.asr, "open_stream")
                else BufferedASRStream(self.asr)
            )
            if text is None
            else None,
        )
        return {"operation_id": identifier, **key.model_dump()}

    def get(self, identifier: str) -> Operation:
        op = self.operations.get(identifier)
        if op is None or self.clock() - op.touched >= 60:
            raise ValueError("unknown or expired operation")
        if op.busy:
            raise ValueError("concurrent operation request")
        op.touched = self.clock()
        return op

    async def append(self, operation_id: str, sequence: int, pcm_b64: str) -> dict[str, Any]:
        op = self.get(operation_id)
        if op.iterator is not None:
            raise ValueError("not an ASR operation")
        if sequence != op.sequence:
            raise ValueError("out-of-order or duplicate audio chunk")
        pcm = decode_pcm(pcm_b64)
        if len(op.audio) + len(pcm) > MAX_AUDIO_BYTES:
            raise ValueError("utterance exceeds 30 seconds")
        try:
            partial = await self.work(operation_id, op, op.asr_stream.append(pcm))
            partial = self.recognition(op, partial)
            op.audio.extend(pcm)
            op.sequence += 1
            return {"next_sequence": op.sequence, **op.key.model_dump(), "partial": partial}
        except BaseException:
            self.operations.pop(operation_id, None)
            await self.dispose(op)
            raise

    async def finish(self, operation_id: str) -> dict[str, Any]:
        op = self.get(operation_id)
        if op.iterator is not None or not op.audio:
            raise ValueError("expected nonempty ASR audio")
        try:
            result = await self.work(operation_id, op, op.asr_stream.finish())
            final = self.recognition(op, result, final=True)
            return {**op.key.model_dump(), **final}
        finally:
            self.operations.pop(operation_id, None)
            await self.dispose(op)

    async def next(self, operation_id: str) -> dict[str, Any]:
        op = self.get(operation_id)
        if op.iterator is None:
            raise ValueError("not a TTS operation")
        try:
            pcm = await self.work(operation_id, op, anext(op.iterator, None))
            if self.operations.get(operation_id) is not op:
                raise ValueError("operation cancelled")
            if pcm is None:
                await self.cancel(operation_id)
                return {**op.key.model_dump(), "done": True}
            # Apply the same bound to provider output as to MCP input.
            encoded = encode_pcm(pcm)
            decode_pcm(encoded)
            op.sequence += 1
            op.output_bytes += len(pcm)
            if op.output_bytes > MAX_AUDIO_BYTES:
                raise ValueError("TTS exceeds 30 seconds")
            return {
                **op.key.model_dump(),
                "done": False,
                "pcm_b64": encoded,
                "sequence": op.sequence - 1,
            }
        except BaseException:
            self.operations.pop(operation_id, None)
            await op.iterator.aclose()
            raise
        finally:
            op.busy = False
            if operation_id not in self.operations:
                await op.iterator.aclose()


def speech_tool(handler, kind):
    """Preserve allowlisted failure codes without returning provider payloads."""

    @wraps(handler)
    async def wrapped(*args, **kwargs):
        try:
            return await handler(*args, **kwargs)
        except (ProviderError, EnglishTextRejected) as exc:
            code = exc.code if re.fullmatch(r"[a-z0-9_]{1,48}", exc.code) else "provider_error"
            return {"speech_error": {"code": code, "role": kind}}

    return wrapped


def make_server(kind: str, service: SpeechService) -> MCPServer:
    server = MCPServer(f"me3-{kind}")
    server.tool(name="cancel")(service.cancel)
    if kind == "asr":

        @server.tool()
        async def asr_open(session_id: str, turn_id: int) -> dict[str, Any]:
            """Start bounded PCM16/16kHz mono transcription with the configured provider."""
            return await service.open(session_id, turn_id)

        server.tool(name="asr_append")(speech_tool(service.append, kind))
        server.tool(name="asr_finish")(speech_tool(service.finish, kind))
    elif kind == "tts":

        @server.tool()
        async def tts_open(session_id: str, turn_id: int, text: str) -> dict[str, Any]:
            """Start synthesis through the configured provider; defaults to mock tones."""
            return await service.open(session_id, turn_id, text)

        server.tool(name="tts_next")(speech_tool(service.next, kind))
    else:
        raise ValueError("unsupported server kind")
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["asr", "tts"])
    parser.add_argument("--mock-transcript", default="Mock question")
    parser.add_argument("--mock-delay", type=float, default=0)
    parser.add_argument("--mock-tts-delay", type=float, default=0)
    parser.add_argument("--mock-transcripts-json", help="test-only deterministic ASR script")
    parser.add_argument(
        "--backend-spec-json", help="selected server-side backend descriptor; no credentials"
    )
    args = parser.parse_args()
    transcript = args.mock_transcript
    if args.mock_transcripts_json:
        transcript = json.loads(args.mock_transcripts_json)
        if (
            not isinstance(transcript, list)
            or not 1 <= len(transcript) <= 100
            or any(not isinstance(t, str) or len(t) > 500 for t in transcript)
        ):
            parser.error("mock script requires 1..100 bounded strings")
    if not 0 <= args.mock_delay <= 30 or not 0 <= args.mock_tts_delay <= 30:
        parser.error("mock delay must be 0..30 seconds")
    asr, tts = MockASR(transcript, args.mock_delay), MockTTS(args.mock_tts_delay)
    if args.backend_spec_json:
        from .configuration import BackendSpec, ConfigurationError
        from .remote import make_provider

        try:
            spec = BackendSpec(**json.loads(args.backend_spec_json)).validate(
                args.kind, runnable=True
            )
            provider = make_provider(
                args.kind,
                spec,
                transcript=transcript,
                delay=args.mock_delay if args.kind == "asr" else args.mock_tts_delay,
            )
        except (ConfigurationError, ValueError, TypeError):
            parser.error(
                "invalid backend descriptor; check the documented role/protocol and local configuration"
            )
        if args.kind == "asr":
            asr = provider
        else:
            tts = provider
    service = SpeechService(asr, tts)
    server = make_server(args.kind, service)

    async def run():
        try:
            await server.run_stdio_async()
        finally:
            await service.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
