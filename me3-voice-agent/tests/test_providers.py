import asyncio

import pytest

from me3_voice.elevenlabs import ElevenASR, ElevenTTS


class FakeTransport:
    def __init__(self, blocks=()):
        self.blocks = blocks
        self.closed = False
        self.requests = []

    async def transcript(self, path, fields, pcm):
        self.requests.append((path, fields, pcm))
        return {"text": "offline transcript"}

    async def audio(self, path, query, body):
        self.requests.append((path, query, body))
        try:
            for block in self.blocks:
                await asyncio.sleep(0)
                yield block
        finally:
            self.closed = True


async def test_asr_adapter_raw_pcm_request():
    transport = FakeTransport()
    assert (
        await ElevenASR(transport, "selected-model").transcribe(b"\1\0" * 1600)
        == "offline transcript"
    )
    assert transport.requests[0][0] == "/v1/speech-to-text"
    assert transport.requests[0][1]["file_format"] == "pcm_s16le_16"


async def test_tts_byte_boundaries_and_early_close():
    transport = FakeTransport([b"\1" * 3, b"\2" * 2047])
    chunks = [chunk async for chunk in ElevenTTS(transport, "model", "voice").synthesize("hello")]
    assert list(map(len, chunks)) == [1024, 1024, 2]
    assert b"".join(chunks) == b"\1" * 3 + b"\2" * 2047
    assert transport.closed
    assert transport.requests[0][1]["output_format"] == "pcm_16000"
    transport = FakeTransport([bytes(2048), bytes(2048)])
    stream = ElevenTTS(transport, "model", "voice").synthesize("hello")
    await anext(stream)
    await stream.aclose()
    assert transport.closed


async def test_tts_truncation_and_oversize_rejected():
    for block in [b"\1", bytes(65537)]:
        transport = FakeTransport([block])
        with pytest.raises(ValueError):
            _ = [chunk async for chunk in ElevenTTS(transport, "model", "voice").synthesize("x")]
        assert transport.closed
