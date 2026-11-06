import asyncio
import json
import wave

import pytest

from me3_voice.contracts import TurnKey
from me3_voice.diagnostics import DiagnosticCapture
from me3_voice.providers import ASRResult


async def test_capture_preserves_submitted_bytes_results_and_cancelled_turn(tmp_path):
    gate = asyncio.Event()

    class Speech:
        async def transcribe_stream(self, key, chunks):
            async for _ in chunks:
                pass
            if key.turn_id == 2:
                gate.set()
                await asyncio.Event().wait()
            yield ASRResult("literal question?", final=True)

    capture = DiagnosticCapture(tmp_path, "run1", 48000, max_seconds=1)
    raw = b"\x01\x00" * 48001
    capture.received(raw)
    speech = capture.wrap(Speech())

    async def chunks():
        yield b"\x02\x00" * 512

    key = TurnKey(session_id="s", turn_id=1)
    results = [r async for r in speech.transcribe_stream(key, chunks())]
    assert results[0].text == "literal question?"

    async def cancelled():
        async for _ in speech.transcribe_stream(TurnKey(session_id="s", turn_id=2), chunks()):
            pass

    task = asyncio.create_task(cancelled())
    await gate.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    capture.save()
    data = json.loads((capture.path / "manifest.json").read_text())
    assert data["raw"]["truncated"]
    assert data["raw"]["saved_bytes"] == 96000
    assert data["turns"][0]["results"][0]["text"] == "literal question?"
    assert data["turns"][1]["outcome"] == "cancelled_or_abandoned"
    assert not data["turns"][1]["truncated"]
    with wave.open(str(capture.path / "asr-2.wav")) as audio:
        assert audio.getframerate() == 16000
        assert audio.readframes(512) == b"\x02\x00" * 512
    assert (capture.path / "manifest.json").stat().st_mode & 0o777 == 0o600
    capture.save()  # no duplicate file on cleanup


def test_capture_rejects_reused_or_escaping_output(tmp_path):
    with pytest.raises(ValueError):
        DiagnosticCapture(tmp_path, "../outside", 16000)
    capture = DiagnosticCapture(tmp_path, "run2", 16000)
    capture.save()
    with pytest.raises(ValueError):
        DiagnosticCapture(tmp_path, "run2", 16000)


async def test_asr_retention_limit_never_truncates_provider_input(tmp_path):
    received = []

    class Speech:
        async def transcribe_stream(self, key, chunks):
            received.append(b"".join([pcm async for pcm in chunks]))
            yield ASRResult("unchanged", final=True)

    capture = DiagnosticCapture(tmp_path, "bounded", 16000, max_seconds=1)
    speech = capture.wrap(Speech())
    pcm = b"\1\0" * 12000

    async def chunks():
        yield pcm

    for turn in (1, 2):
        async for _ in speech.transcribe_stream(TurnKey(session_id="s", turn_id=turn), chunks()):
            pass
    capture.save()
    report = json.loads((capture.path / "manifest.json").read_text())
    assert received == [pcm, pcm]
    assert sum(r["saved_bytes"] for r in report["turns"]) == 32000
    assert report["turns"][1]["truncated"] is True
