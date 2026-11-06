import wave

import pytest

from me3_voice.audio import ConversationInput, ReplayInput, SileroVAD, read_wav
from me3_voice.contracts import FRAME_BYTES


class MarkedVAD:
    def voiced(self, pcm):
        return pcm[0] != 0


def frame(value):
    return bytes([value, 0]) * (FRAME_BYTES // 2)


def test_activation_preroll_pause_followup_and_exit():
    source = ReplayInput(MarkedVAD(), pre_roll_frames=2, silence_frames=3)
    for value in [0, 0, 1]:
        assert source.feed(frame(value)) is None
    source.activate()
    source.activate()  # Does not reset pre-roll on duplicate wake.
    assert source.feed(frame(2)) is None
    for value in [0, 0, 3, 0, 0]:
        assert source.feed(frame(value)) is None
    first = source.feed(frame(0))
    assert first == b"".join(frame(value) for value in [0, 1, 2, 0, 0, 3, 0, 0, 0])
    for value in [4, 0, 0]:
        assert source.feed(frame(value)) is None
    assert source.feed(frame(0)) == b"".join(frame(value) for value in [4, 0, 0, 0])
    source.reset()
    assert source.feed(frame(9)) is None


def test_endpoint_cap_and_input_validation():
    source = ReplayInput(MarkedVAD(), pre_roll_frames=1, max_frames=3)
    source.activate()
    assert source.feed(frame(1)) is None
    assert source.feed(frame(1)) is None
    assert len(source.feed(frame(1))) == FRAME_BYTES * 3
    assert source.events[-1]["reason"] == "cap"
    with pytest.raises(ValueError):
        source.feed(b"bad")


def test_actual_local_silero_silence():
    vad = SileroVAD()
    assert not any(vad.voiced(frame(0)) for _ in range(40))
    assert vad.analyzer._model.session.get_providers() == ["CPUExecutionProvider"]


def test_wav_format_gate(tmp_path):
    path = tmp_path / "wrong.wav"
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, 16000, 0, "NONE", ""))
        output.writeframes(b"\0" * 8)
    with pytest.raises(ValueError, match="mono"):
        read_wav(path)


async def test_input_onset_interrupts_while_preserving_new_audio():
    from me3_voice.providers import MockLLM
    from me3_voice.session import Session

    import asyncio

    started = asyncio.Event()

    class Speech:
        def __init__(self):
            self.inputs = []

        async def transcribe(self, key, pcm):
            self.inputs.append(pcm)
            return "test"

        async def synthesize(self, key, text):
            yield frame(9)

    async def blocked_sink(pcm):
        started.set()
        await asyncio.Event().wait()

    speech = Speech()
    async with Session(speech, MockLLM(), blocked_sink) as session:
        controller = ConversationInput(ReplayInput(MarkedVAD(), silence_frames=2), session)
        controller.activate()
        for value in [1, 0, 0]:
            await controller.feed(frame(value))
        await asyncio.wait_for(started.wait(), 1)
        await controller.feed(frame(2))
        assert session.playback.queue.empty()
        for value in [0, 0]:
            await controller.feed(frame(value))
        await asyncio.sleep(0)
        assert speech.inputs == [frame(1) + frame(0) * 2, frame(2) + frame(0) * 2]


async def test_browser_barge_confirmation_ignores_short_blip_and_keeps_real_onset():
    import asyncio

    from me3_voice.providers import MockLLM
    from me3_voice.session import Session, State

    playing = asyncio.Event()

    class Speech:
        def __init__(self):
            self.inputs = []

        async def transcribe(self, key, pcm):
            self.inputs.append(pcm)
            return "test"

        async def synthesize(self, key, text):
            yield frame(9)

    async def blocked_sink(pcm):
        playing.set()
        await asyncio.Event().wait()

    speech = Speech()
    async with Session(speech, MockLLM(), blocked_sink) as session:
        controller = ConversationInput(
            ReplayInput(MarkedVAD(), silence_frames=2),
            session,
            barge_min_voiced_frames=5,
        )
        controller.activate()
        for value in [1, 0, 0]:
            await controller.feed(frame(value))
        await asyncio.wait_for(playing.wait(), 1)
        for value in [2, 2, 0, 0]:
            await controller.feed(frame(value))
        assert session.state == State.SPEAKING
        assert len(speech.inputs) == 1
        assert not any(e["event"] == "interrupt_requested" for e in session.events)
        assert any(e["event"] == "barge_candidate_rejected" for e in session.events)
        for value in [3] * 5 + [0, 0]:
            await controller.feed(frame(value))
        await asyncio.sleep(0)
        assert any(e["event"] == "interrupt_requested" for e in session.events)
        assert len(speech.inputs) == 2
        assert frame(3) * 5 in speech.inputs[-1]


async def test_live_barge_gate_waits_for_sustained_vad_and_preserves_candidate_audio():
    import asyncio

    from me3_voice.providers import MockLLM
    from me3_voice.session import Session, State

    playing = asyncio.Event()

    class Speech:
        def __init__(self):
            self.inputs = []

        async def transcribe(self, key, pcm):
            self.inputs.append(pcm)
            return "a sustained user request"

        async def synthesize(self, key, text):
            yield frame(9)

    async def blocked_sink(pcm):
        playing.set()
        await asyncio.Event().wait()

    speech = Speech()
    async with Session(speech, MockLLM(), blocked_sink) as session:
        controller = ConversationInput(
            ReplayInput(MarkedVAD(), silence_frames=2),
            session,
            barge_min_voiced_frames=8,
            barge_window_frames=12,
        )
        controller.activate()
        for value in [1, 0, 0]:
            await controller.feed(frame(value))
        await asyncio.wait_for(playing.wait(), 1)

        for value in [2] * 7:
            await controller.feed(frame(value))
        assert session.state == State.SPEAKING
        assert len(speech.inputs) == 1
        assert not any(row["event"] == "interrupt_requested" for row in session.events)

        await controller.feed(frame(2))
        assert session.input_open
        assert any(
            row["event"] == "vad_candidate_confirmed"
            and row.get("voiced_frames") == 8
            and row.get("window_frames") == 12
            for row in session.events
        )
        assert any(row["event"] == "interrupt_requested" for row in session.events)
        for value in [0, 0]:
            await controller.feed(frame(value))
        await asyncio.sleep(0)
        assert len(speech.inputs) == 2
        assert frame(2) * 8 in speech.inputs[-1]


async def test_sustained_music_playback_requires_confirmed_vad_before_opening_user_turn():
    import asyncio
    from me3_voice.providers import MockLLM
    from me3_voice.session import Session

    class Speech:
        inputs = []

        async def transcribe(self, key, pcm):
            self.inputs.append(pcm)
            return "play Maroon Five"

        async def synthesize(self, key, text):
            yield frame(0)

    speech = Speech()

    async def sink(_pcm):
        pass

    async with Session(speech, MockLLM(), sink) as session:
        controller = ConversationInput(
            ReplayInput(MarkedVAD(), silence_frames=2),
            session,
            barge_min_voiced_frames=5,
        )
        controller.activate("wake_detected_file")
        session.observe_music_playback("playing", source="web_playback_sdk", track_changed=True)
        for value in [2, 2, 0, 0]:
            await controller.feed(frame(value))
        assert not session.input_open
        assert not speech.inputs
        assert any(event["event"] == "barge_candidate_rejected" for event in session.events)
        for value in [3] * 5:
            await controller.feed(frame(value))
        assert session.input_open
        assert any(event["event"] == "vad_candidate_confirmed" for event in session.events)
        for value in [0, 0]:
            await controller.feed(frame(value))
        if session.generation:
            await asyncio.wait_for(session.generation, 1)
        assert len(speech.inputs) == 1
        assert any(event["event"] == "music_playback_observed" for event in session.events)


async def test_brief_noise_during_reasoning_does_not_cancel_pending_correction():
    import asyncio
    from me3_voice.session import Session, State

    started = asyncio.Event()

    class Speech:
        def __init__(self):
            self.inputs = []

        async def transcribe(self, key, pcm):
            self.inputs.append(pcm)
            return "play maroon five instead"

        async def synthesize(self, key, text):
            yield frame(9)

    class SlowReasoning:
        async def respond(self, history):
            started.set()
            await asyncio.Event().wait()
            yield "unreachable"

    async def sink(pcm):
        pass

    speech = Speech()
    async with Session(speech, SlowReasoning(), sink) as session:
        controller = ConversationInput(
            ReplayInput(MarkedVAD(), silence_frames=2), session, barge_min_voiced_frames=5
        )
        controller.activate()
        for value in [1, 0, 0]:
            await controller.feed(frame(value))
        await asyncio.wait_for(started.wait(), 1)
        pending = session.generation
        assert session.state == State.THINKING
        for value in [2, 2, 0, 0]:
            await controller.feed(frame(value))
        assert not pending.done()
        assert len(speech.inputs) == 1
        for value in [3] * 5 + [0, 0]:
            await controller.feed(frame(value))
        await asyncio.sleep(0)
        assert pending.cancelled()
        assert len(speech.inputs) == 2
        assert frame(3) * 5 in speech.inputs[-1]


def test_endpoint_diagnostics_bound_retention_without_changing_utterances():
    source = ReplayInput(None, silence_frames=1)
    source.activate()
    frame = b"\1\0" * 512
    for _ in range(2500):
        assert source.feed_scored(frame, True) is None
        result = source.feed_scored(frame, False)
        assert result.endswith(frame * 2)
    assert source.event_count == 5001
    assert len(source.events) == 1000
    assert source.events[-1]["event"] == "endpoint"
    assert source.events[-1]["audio_seconds"] == 5000 * 0.032
    assert source.active and not source.has_speech
