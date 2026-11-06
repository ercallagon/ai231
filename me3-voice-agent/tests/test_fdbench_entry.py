"""Offline compatibility checks for the class benchmark's actual adapter boundary."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "me3_fdbench", Path(__file__).parents[1] / "scripts/fdbench_entry.py"
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.mark.asyncio
async def test_transport_ack_stop_and_closed_sink():
    from pipecat.frames.frames import InterruptionFrame, OutputAudioRawFrame

    sent, marks = [], []

    class Output:
        async def queue_frame(self, frame, callback):
            sent.append(frame)
            assert not marks or marks[-1] == "audio_out"
            await callback(self, frame, None)

    sink = entry.BenchmarkSink(Output(), lambda stage: marks.append(stage))
    await sink(b"\x00\x00" * 16)
    assert isinstance(sent[0], OutputAudioRawFrame)
    assert sent[0].sample_rate == 16000 and marks == ["audio_out"]
    await sink.stop()
    assert isinstance(sent[-1], InterruptionFrame) and sink.stopped
    await sink.close()
    await sink(b"\x00\x00")
    assert len(sent) == 2


def test_stage_marks_preserve_identity_and_do_not_invent_first_token():
    marks = []
    row = {"event": "endpoint_submitted", "monotonic": 12.5, "session_id": "s", "turn_id": 2}
    entry.report_stage(row, lambda stage, **fields: marks.append((stage, fields)))
    assert marks[0][1] == {"t": 12.5, "session_id": "s", "turn_id": 2}
    entry.report_stage(
        {**row, "event": "llm_first_segment"}, lambda *args, **kw: marks.append(args)
    )
    assert len(marks) == 1


def test_transcript_marks_require_opt_in_and_matching_final_turn(monkeypatch):
    row = {"event": "asr_final", "monotonic": 2.0, "session_id": "new", "turn_id": 3}
    messages = [
        {"role": "user", "turn_id": 2, "status": "final", "text": "old text"},
        {"role": "user", "turn_id": 3, "status": "partial", "text": "draft"},
    ]
    marks = []

    def emit(stage, **fields):
        marks.append(fields)

    monkeypatch.setenv("ME3_FDBENCH_EXPORT_TEXT", "1")
    entry.report_stage(row, emit, messages=messages)
    assert "text" not in marks[-1]
    messages[-1].update(status="final", text="current final")
    entry.report_stage(row, emit, messages=messages)
    assert marks[-1]["text"] == "current final"
    messages[-1]["text"] = "x" * 501
    entry.report_stage(row, emit, messages=messages)
    assert "text" not in marks[-1]  # Do not send a truncated transcript as full WER input.
    monkeypatch.delenv("ME3_FDBENCH_EXPORT_TEXT")
    entry.report_stage(row, emit, messages=messages)
    assert "text" not in marks[-1]


@pytest.mark.asyncio
async def test_partial_build_failure_closes_owned_session_and_speech(monkeypatch, tmp_path):
    import asyncio
    import me3_voice.audio
    import me3_voice.speech

    closed = []

    class Speech:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append("speech")

    class Output:
        async def queue_frame(self, frame, callback):
            await callback(self, frame, None)

    class Transport:
        def output(self):
            return Output()

    def fail_vad():
        raise RuntimeError("controlled VAD startup failure")

    monkeypatch.setenv("ME3_FDBENCH_MOCK", "1")
    monkeypatch.setattr(me3_voice.speech, "SpeechClient", Speech)
    monkeypatch.setattr(me3_voice.audio, "SileroVAD", fail_vad)
    before = asyncio.all_tasks()
    with pytest.raises(RuntimeError, match="controlled VAD"):
        await entry.build(Transport(), tmp_path / "failed-start")
    await asyncio.sleep(0)
    assert closed == ["speech"]
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]
