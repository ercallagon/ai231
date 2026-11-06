import pytest

from me3_voice.providers import MockLLM
from me3_voice.session import Session, State
from test_session import Speech, sink


@pytest.mark.parametrize("mode,seconds", [("followup", 7), ("extended", 20)])
async def test_deadline_starts_after_output_and_empty_input_cannot_extend_it(mode, seconds):
    now = [100.0]
    async with Session(
        Speech(), MockLLM(), sink, clock=lambda: now[0], conversation_mode=mode
    ) as s:
        s.activate()
        await (await s.submit(b"hello"))
        assert s.engagement_deadline == now[0] + seconds
        deadline = s.engagement_deadline
        now[0] += 2
        await (await s.submit(b""))
        assert s.engagement_deadline == deadline
        now[0] = deadline
        await s.tick()
        assert s.state == State.IDLE
        s.activate()
        assert s.engagement_deadline > deadline
        await s.tick()
        assert s.state == State.LISTENING


async def test_speech_before_expiry_can_finish_and_single_mode_returns_to_standby():
    now = [0.0]
    async with Session(
        Speech(), MockLLM(), sink, clock=lambda: now[0], conversation_mode="single"
    ) as s:
        s.activate()
        now[0] = 6.9
        s.candidate_started_at = now[0]
        now[0] = 7.1
        await s.tick()
        assert s.state == State.LISTENING
        await s.begin_input(started_at=6.9)
        now[0] = 10
        await s.tick()
        assert s.input_open
        await s.push_input(b"hello")
        await (await s.commit_input())
        assert s.state == State.IDLE
        assert s.playback.queue.empty()


async def test_explicit_clarification_uses_answer_window_and_exact_exit_preserves_history():
    async with Session(Speech(), MockLLM(), sink, conversation_mode="single") as s:
        s.activate()
        await (await s.submit(b"create a reminder"))
        assert s.awaiting_answer
        assert s.state == State.LISTENING
        assert 17 < s.engagement_snapshot()["remaining_seconds"] <= 18
        history = list(s.history)
        await (await s.submit(b"That's all."))
        assert s.state == State.IDLE
        assert s.history == history


async def test_playback_time_is_excluded_and_late_input_cannot_execute():
    now = [0.0]

    async def delayed_sink(_pcm):
        assert session.engagement_deadline is None
        now[0] += 1
        await session.tick()
        assert session.state == State.SPEAKING

    async with Session(
        Speech(), MockLLM(), delayed_sink, clock=lambda: now[0], conversation_mode="followup"
    ) as session:
        session.activate()
        await (await session.submit(b"hello"))
        assert session.engagement_deadline == now[0] + 7
        now[0] = session.engagement_deadline
        with pytest.raises(ValueError, match="engagement expired"):
            await session.begin_input()
        assert session.state == State.IDLE
