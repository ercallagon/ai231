"""Our speech MCP servers: stdio only, pull-based TTS, bounded utterance ASR."""

import argparse
import asyncio
import secrets
import time
import json
from collections import deque
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
    tts_opened_at: float | None = None
    tts_first_next_at: float | None = None
    tts_last_next_at: float | None = None
    tts_next_count: int = 0
    tts_next_seconds: deque = field(default_factory=lambda: deque(maxlen=1024))
    tts_first_pcm_at: float | None = None
    tts_last_pcm_at: float | None = None
    tts_pcm_chunks: int = 0
    tts_pcm_bytes: int = 0
    tts_status: str = "running"


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
        result = {"cancelled": op is not None}
        if op:
            diagnostics = self.provider_diagnostics(op)
            if diagnostics is not None:
                result["provider_diagnostics"] = diagnostics
            if op.iterator is not None:
                op.tts_status = "cancelled"
                result["tts_diagnostics"] = self.tts_diagnostics(op)
        return result

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

    @staticmethod
    def provider_diagnostics(op):
        snapshot = getattr(op.asr_stream, "diagnostics", None)
        return snapshot() if callable(snapshot) else None

    @staticmethod
    def duration_summary(values):
        samples = sorted(values)
        if not samples:
            return {"sample_count": 0, "retained_sample_count": 0}

        def percentile(fraction):
            # Nearest-rank; the bounded deque retains at most 1024 MCP calls.
            index = int((len(samples) * fraction) + 0.999999) - 1
            return samples[max(0, min(len(samples) - 1, index))]

        return {
            "sample_count": len(samples),
            "retained_sample_count": len(samples),
            "p50_seconds": percentile(0.50),
            "p95_seconds": percentile(0.95),
            "max_seconds": samples[-1],
            "total_seconds": sum(samples),
        }

    def tts_diagnostics(self, op):
        provider_snapshot = getattr(self.tts, "diagnostics", None)
        provider = provider_snapshot() if callable(provider_snapshot) else None
        return {
            "status": op.tts_status,
            "mcp": {
                "open_monotonic": op.tts_opened_at,
                "first_tts_next_monotonic": op.tts_first_next_at,
                "last_tts_next_monotonic": op.tts_last_next_at,
                "tts_next_count": op.tts_next_count,
                "tts_next_seconds": self.duration_summary(op.tts_next_seconds),
                "first_pcm_monotonic": op.tts_first_pcm_at,
                "last_pcm_monotonic": op.tts_last_pcm_at,
                "pcm_chunks": op.tts_pcm_chunks,
                "pcm_bytes": op.tts_pcm_bytes,
                "audio_seconds": op.tts_pcm_bytes / 32000,
            },
            "provider": provider if isinstance(provider, dict) else None,
        }

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
            tts_opened_at=self.clock() if text is not None else None,
            asr_stream=(
                self.asr.open_stream()
                if hasattr(self.asr, "open_stream")
                else BufferedASRStream(self.asr)
            )
            if text is None
            else None,
        )
        stream = self.operations[identifier].asr_stream
        # Deepgram's websocket is lazy at the SDK boundary. Start its single
        # authorized connection attempt as soon as the MCP ASR operation opens;
        # run_stt does not wait for this network handshake before admitting audio.
        if stream is not None and hasattr(stream, "start_background"):
            stream.start_background()
        result = {"operation_id": identifier, **key.model_dump()}
        diagnostics = self.provider_diagnostics(self.operations[identifier])
        if diagnostics is not None:
            result["provider_diagnostics"] = diagnostics
        return result

    def get(self, identifier: str) -> Operation:
        op = self.operations.get(identifier)
        if op is None or self.clock() - op.touched >= 60:
            raise ValueError("unknown or expired operation")
        if op.busy:
            raise ValueError("concurrent operation request")
        op.touched = self.clock()
        return op

    async def capabilities(self) -> dict[str, Any]:
        return {
            "contract_version": 2,
            "max_asr_batch_bytes": 3 * 1024,
            "asr_events": True,
            "endpoint_owner": "client",
        }

    async def append(self, operation_id: str, sequence: int, pcm_b64: str) -> dict[str, Any]:
        return await self._append(operation_id, sequence, decode_pcm(pcm_b64))

    async def append_batch(
        self, operation_id: str, sequence: int, pcm_b64: str, contract_version: int = 2
    ) -> dict[str, Any]:
        if contract_version != 2:
            raise ValueError("unsupported speech contract version")
        # Additive bounded API; legacy per-32ms and TTS limits are unchanged.
        import base64
        import binascii

        if len(pcm_b64) > 4096:
            raise ValueError("batch exceeds 96ms")
        try:
            pcm = base64.b64decode(pcm_b64, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("invalid base64 PCM") from None
        if not pcm or len(pcm) > 3072 or len(pcm) % 2:
            raise ValueError("expected 1..1536 mono PCM16 samples")
        return {**await self._append(operation_id, sequence, pcm), "contract_version": 2}

    async def events(self, operation_id: str) -> dict[str, Any]:
        op = self.operations.get(operation_id)
        if op is None:
            raise ValueError("unknown operation")
        if self.clock() - op.touched >= 60:
            raise ValueError("expired operation")
        # Never serialize event observation behind provider append/finalization.
        # Partial hypotheses are latest-value/coalesced, not an unbounded queue.
        result = None
        if not op.busy and hasattr(op.asr_stream, "poll"):
            result = await op.asr_stream.poll()
        if self.operations.get(operation_id) is not op:
            raise ValueError("operation cancelled")
        op.touched = self.clock()
        response = {**op.key.model_dump(), "partial": self.recognition(op, result)}
        diagnostics = self.provider_diagnostics(op)
        if diagnostics is not None:
            response["provider_diagnostics"] = diagnostics
        return response

    async def _append(self, operation_id, sequence, pcm):
        op = self.get(operation_id)
        if op.iterator is not None:
            raise ValueError("not an ASR operation")
        if sequence != op.sequence:
            raise ValueError("out-of-order or duplicate audio chunk")
        if len(op.audio) + len(pcm) > MAX_AUDIO_BYTES:
            raise ValueError("utterance exceeds 30 seconds")
        try:
            partial = await self.work(operation_id, op, op.asr_stream.append(pcm))
            partial = self.recognition(op, partial)
            op.audio.extend(pcm)
            op.sequence += 1
            response = {"next_sequence": op.sequence, **op.key.model_dump(), "partial": partial}
            diagnostics = self.provider_diagnostics(op)
            if diagnostics is not None:
                response["provider_diagnostics"] = diagnostics
            return response
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
        except BaseException:
            self.operations.pop(operation_id, None)
            await self.dispose(op)
            raise
        self.operations.pop(operation_id, None)
        await self.dispose(op)
        response = {**op.key.model_dump(), **final}
        diagnostics = self.provider_diagnostics(op)
        if diagnostics is not None:
            response["provider_diagnostics"] = diagnostics
        return response

    async def next(self, operation_id: str) -> dict[str, Any]:
        op = self.get(operation_id)
        if op.iterator is None:
            raise ValueError("not a TTS operation")
        request_started = self.clock()
        op.tts_next_count += 1
        if op.tts_first_next_at is None:
            op.tts_first_next_at = request_started
        try:
            pcm = await self.work(operation_id, op, anext(op.iterator, None))
            op.tts_next_seconds.append(self.clock() - request_started)
            op.tts_last_next_at = self.clock()
            if self.operations.get(operation_id) is not op:
                raise ValueError("operation cancelled")
            if pcm is None:
                op.tts_status = "completed"
                diagnostics = self.tts_diagnostics(op)
                await self.cancel(operation_id)
                return {
                    **op.key.model_dump(),
                    "done": True,
                    "tts_diagnostics": diagnostics,
                }
            # Apply the same bound to provider output as to MCP input.
            encoded = encode_pcm(pcm)
            decode_pcm(encoded)
            op.sequence += 1
            op.output_bytes += len(pcm)
            if op.output_bytes > MAX_AUDIO_BYTES:
                raise ValueError("TTS exceeds 30 seconds")
            now = self.clock()
            if op.tts_first_pcm_at is None:
                op.tts_first_pcm_at = now
            op.tts_last_pcm_at = now
            op.tts_pcm_chunks += 1
            op.tts_pcm_bytes += len(pcm)
            return {
                **op.key.model_dump(),
                "done": False,
                "pcm_b64": encoded,
                "sequence": op.sequence - 1,
            }
        except BaseException as exc:
            op.tts_status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            if isinstance(exc, Exception):
                try:
                    exc.tts_diagnostics = self.tts_diagnostics(op)
                except Exception:
                    pass
            self.operations.pop(operation_id, None)
            await op.iterator.aclose()
            raise
        finally:
            op.busy = False
            if operation_id not in self.operations:
                await op.iterator.aclose()


def speech_tool(handler, kind):
    """Preserve allowlisted failure codes without returning provider payloads."""

    def safe_type(value):
        return value if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]{0,80}", value) else "Exception"

    @wraps(handler)
    async def wrapped(*args, **kwargs):
        try:
            return await handler(*args, **kwargs)
        except (ProviderError, EnglishTextRejected) as exc:
            code = exc.code if re.fullmatch(r"[a-z0-9_]{1,48}", exc.code) else "provider_error"
            cause = exc.__cause__ or exc.__context__
            cause_type = getattr(exc, "cause_type", None) or (
                type(cause).__name__ if cause else None
            )
            error = {
                "code": code,
                "role": kind,
                "stage": getattr(exc, "error_stage", None),
                "boundary": getattr(exc, "error_boundary", None),
                "exception_type": safe_type(
                    getattr(exc, "error_detail", None) or type(exc).__name__
                ),
                "cause_type": safe_type(cause_type) if cause_type else None,
            }
            diagnostics = getattr(exc, "tts_diagnostics", None)
            if kind == "tts" and isinstance(diagnostics, dict):
                error["tts_diagnostics"] = diagnostics
            return {"speech_error": error}
        except Exception as exc:
            # Avoid MCP's generic is_error text path, which loses both the
            # failure stage and exception class while potentially exposing an
            # implementation message. Never include str(exc) or provider data.
            cause = exc.__cause__ or exc.__context__
            error = {
                "code": "mcp_tool_error",
                "role": kind,
                "stage": "mcp_tool",
                "boundary": "speech_mcp",
                "exception_type": safe_type(type(exc).__name__),
                "cause_type": safe_type(type(cause).__name__) if cause else None,
            }
            diagnostics = getattr(exc, "tts_diagnostics", None)
            if kind == "tts" and isinstance(diagnostics, dict):
                error["tts_diagnostics"] = diagnostics
            return {"speech_error": error}

    return wrapped


def make_server(kind: str, service: SpeechService) -> MCPServer:
    server = MCPServer(f"me3-{kind}")
    server.tool(name="cancel")(service.cancel)
    server.tool(name="speech_capabilities")(service.capabilities)
    if kind == "asr":

        @server.tool()
        async def asr_open(session_id: str, turn_id: int) -> dict[str, Any]:
            """Start bounded PCM16/16kHz mono transcription with the configured provider."""
            return await service.open(session_id, turn_id)

        server.tool(name="asr_append")(speech_tool(service.append, kind))
        server.tool(name="asr_finish")(speech_tool(service.finish, kind))
        server.tool(name="asr_append_batch")(speech_tool(service.append_batch, kind))
        server.tool(name="asr_events")(speech_tool(service.events, kind))
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
