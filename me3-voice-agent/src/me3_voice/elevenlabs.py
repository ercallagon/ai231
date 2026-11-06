"""Offline-testable ElevenLabs adapters; no live HTTP transport is supplied.

The injected transport owns authentication, timeouts and response cleanup. Adding
a live transport is gated on explicit account/model access and spending limits.
ASR is utterance-based; TTS reblocks a byte stream into bounded PCM chunks.
"""

import re
from collections.abc import AsyncIterator
from typing import Protocol

from .contracts import FRAME_BYTES, MAX_AUDIO_BYTES


class ElevenTransport(Protocol):
    async def transcript(self, path: str, fields: dict, pcm: bytes) -> dict: ...

    def audio(self, path: str, query: dict, body: dict) -> AsyncIterator[bytes]: ...


class ElevenASR:
    def __init__(self, transport: ElevenTransport, model: str):
        self.transport, self.model = transport, model

    async def transcribe(self, pcm: bytes) -> str:
        if not 3200 <= len(pcm) <= MAX_AUDIO_BYTES or len(pcm) % 2:
            raise ValueError("ASR expects 100 ms..30 s PCM16")
        result = await self.transport.transcript(
            "/v1/speech-to-text", {"model_id": self.model, "file_format": "pcm_s16le_16"}, pcm
        )
        if not isinstance(result.get("text"), str):
            raise ValueError("provider returned no transcript")
        return result["text"]


class ElevenTTS:
    def __init__(self, transport: ElevenTransport, model: str, voice_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", voice_id):
            raise ValueError("invalid voice identifier")
        self.transport, self.model, self.voice_id = transport, model, voice_id

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not text.strip() or len(text) > 500:
            raise ValueError("TTS text must contain 1..500 characters")
        stream = self.transport.audio(
            f"/v1/text-to-speech/{self.voice_id}/stream",
            {"output_format": "pcm_16000"},
            {"text": text, "model_id": self.model},
        )
        pending = bytearray()
        total = 0
        try:
            async for block in stream:
                total += len(block)
                if total > MAX_AUDIO_BYTES:
                    raise ValueError("provider audio exceeds 30 seconds")
                # Never retain a whole HTTP response; transport blocks are capped too.
                if len(block) > 65536:
                    raise ValueError("transport block exceeds 64 KiB")
                pending.extend(block)
                while len(pending) >= FRAME_BYTES:
                    chunk = bytes(pending[:FRAME_BYTES])
                    del pending[:FRAME_BYTES]
                    yield chunk
            if len(pending) % 2:
                raise ValueError("provider returned truncated PCM sample")
            if pending:
                yield bytes(pending)
        finally:
            await stream.aclose()
