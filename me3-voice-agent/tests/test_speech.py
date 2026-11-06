import asyncio

import pytest

from me3_voice.contracts import FRAME_BYTES, MAX_AUDIO_BYTES, TurnKey, encode_pcm
from me3_voice.providers import MockASR, MockTTS
from me3_voice.servers import SpeechService
from me3_voice.speech import SpeechClient


@pytest.mark.asyncio
async def test_real_stdio_mcp_contract_and_cleanup():
    key = TurnKey(session_id="integration", turn_id=1)
    async with SpeechClient(transcript="hello") as speech:
        listing = await speech.clients["asr"].list_tools()
        assert {tool.name for tool in listing.tools} == {
            "asr_open",
            "asr_append",
            "asr_finish",
            "cancel",
        }
        assert await speech.transcribe(key, b"\x01\x00" * 700) == "hello"
        chunks = [chunk async for chunk in speech.synthesize(key, "hello")]
        assert len(chunks) == 4
        assert all(len(chunk) == FRAME_BYTES for chunk in chunks)
        stream = speech.synthesize(key, "cancel this")
        await anext(stream)
        await stream.aclose()
        assert len([c async for c in speech.synthesize(key, "next")]) == 4


async def test_sequence_limits_expiry_and_slot_reuse():
    now = [0.0]
    service = SpeechService(MockASR(), MockTTS(), clock=lambda: now[0])
    opened = await service.open("test", 1)
    identifier = opened["operation_id"]
    with pytest.raises(ValueError, match="active"):
        await service.open("test", 2)
    with pytest.raises(ValueError, match="duplicate"):
        await service.append(identifier, 1, encode_pcm(b"\0\0"))
    with pytest.raises(ValueError, match="base64"):
        await service.append(identifier, 0, "!!!!")
    service.operations[identifier].audio = bytearray(MAX_AUDIO_BYTES)
    with pytest.raises(ValueError, match="30 seconds"):
        await service.append(identifier, 0, encode_pcm(b"\0\0"))
    now[0] = 60
    with pytest.raises(ValueError, match="expired"):
        service.get(identifier)
    await service.open("test", 2)
    assert len(service.operations) == 1


async def test_cancel_provider_in_flight_rejects_late_result():
    started, release = asyncio.Event(), asyncio.Event()

    class LateASR:
        async def transcribe(self, pcm):
            started.set()
            await release.wait()
            return "obsolete"

    service = SpeechService(LateASR(), MockTTS())
    identifier = (await service.open("test", 1))["operation_id"]
    await service.append(identifier, 0, encode_pcm(b"\x01\0"))
    task = asyncio.create_task(service.finish(identifier))
    await started.wait()
    await service.cancel(identifier)
    release.set()
    with pytest.raises(ValueError, match="cancelled"):
        await task
    assert not service.operations


async def test_stdio_timeout_recovery():
    from me3_voice.remote import ProviderError

    async with SpeechClient(timeout=0.25, delay=0.6) as speech:
        with pytest.raises(ProviderError) as error:
            await speech.transcribe(TurnKey(session_id="timeout", turn_id=1), b"\x01\0" * 10)
        assert error.value.code == "provider_timeout"
        speech.timeout = 3
        assert await speech.transcribe(TurnKey(session_id="timeout", turn_id=2), b"\x01\0" * 10)


async def test_stdio_cancel_inflight_then_reuse():
    async with SpeechClient(delay=0.6) as speech:
        key = TurnKey(session_id="cancel", turn_id=1)
        opened = await speech.call("asr", "asr_open", key.model_dump())
        identifier = opened["operation_id"]
        await speech.call(
            "asr",
            "asr_append",
            {
                "operation_id": identifier,
                "sequence": 0,
                "pcm_b64": encode_pcm(b"\1\0"),
            },
        )
        pending = asyncio.create_task(
            speech.call("asr", "asr_finish", {"operation_id": identifier})
        )
        await asyncio.sleep(0.05)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await speech.release("asr", identifier)
        assert await speech.transcribe(TurnKey(session_id="cancel", turn_id=2), b"\1\0")


async def test_provider_budget_code_survives_speech_contract_and_releases_slot():
    from types import SimpleNamespace
    from me3_voice.remote import ProviderError
    from me3_voice.servers import speech_tool

    class BudgetASR:
        async def transcribe(self, pcm):
            raise ProviderError("private provider body", code="request_limit_reached")

    service = SpeechService(BudgetASR(), MockTTS())
    op = await service.open("budget", 1)
    await service.append(op["operation_id"], 0, encode_pcm(b"\1\0"))
    finish = speech_tool(service.finish, "asr")

    class Client:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(is_error=False, structured_content=await finish(**arguments))

    client = SpeechClient()
    client.clients = {"asr": Client()}
    with pytest.raises(ProviderError) as error:
        await client.call("asr", "asr_finish", {"operation_id": op["operation_id"]})
    assert error.value.code == "request_limit_reached"
    assert error.value.role == "asr"
    assert "private provider body" not in str(error.value)
    assert not service.operations
    await service.close()


async def test_english_rejection_preserves_speech_error_code_and_next_turn_recovers():
    from me3_voice.english import EnglishTextRejected
    from me3_voice.servers import speech_tool

    class RejectOnce:
        calls = 0

        async def transcribe(self, pcm):
            self.calls += 1
            if self.calls == 1:
                raise EnglishTextRejected("asr")
            return "play music by maroon five"

    service = SpeechService(RejectOnce(), MockTTS())
    finish = speech_tool(service.finish, "asr")
    first = (await service.open("recovery", 1))["operation_id"]
    await service.append(first, 0, encode_pcm(b"\1\0" * 512))
    result = await finish(first)
    assert result == {"speech_error": {"code": "english_only_asr", "role": "asr"}}
    assert not service.operations
    second = (await service.open("recovery", 2))["operation_id"]
    await service.append(second, 0, encode_pcm(b"\1\0" * 512))
    assert (await finish(second))["text"] == "play music by maroon five"
    assert not service.operations
    await service.close()
