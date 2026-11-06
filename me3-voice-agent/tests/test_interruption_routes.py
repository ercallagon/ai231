"""Transcript injection into Session; actual device executor and bounded playback."""

import asyncio
import pytest
from me3_voice.session import Session, State
from me3_voice.providers import MockLLM
from me3_voice.music import MusicController
from me3_voice.tools import MockTaskExecutor
from me3_voice.transcript import Transcript


class Speech:
    async def transcribe(self, key, pcm):
        return pcm.decode()

    async def synthesize(self, key, text):
        yield b"\1\0" * 512
        yield b"\2\0" * 512


class NoModel:
    async def respond(self, history):
        raise AssertionError("unambiguous control unexpectedly reached reasoning")
        yield ""


@pytest.mark.parametrize(
    "next_text,brightness,on",
    [("Actually, 30 percent.", 30, True), ("Turn it off again.", 70, False)],
)
async def test_completed_light_correction_during_ack_is_a_new_single_action(
    next_text, brightness, on
):
    entered = asyncio.Event()
    release = asyncio.Event()
    display = Transcript()
    tasks = MockTaskExecutor(music=MusicController())

    async def sink(pcm):
        entered.set()
        await release.wait()

    async with Session(Speech(), NoModel(), sink, tasks=tasks, transcript=display) as s:
        s.activate()
        old = await s.submit(b"Set brightness to 70 percent.")
        await asyncio.wait_for(entered.wait(), 1)
        assert tasks.actions.effects == 1
        await s.interrupt()
        release.set()
        await asyncio.gather(old, return_exceptions=True)
        await (await s.submit(next_text.encode()))
        assert (
            tasks.actions.effects == 2
            and tasks.actions.brightness == brightness
            and tasks.actions.light is on
        )
        assert len(display.snapshot()["tasks"]) == 2
        assert s.playback.queue.empty() and s.state == State.LISTENING
        assert not any(
            m.get("delivery")
            for m in s.history
            if m.get("turn_id") == 1 and m["role"] == "assistant"
        )


@pytest.mark.parametrize("control", ["Stop speaking.", "Cancel that.", "End conversation."])
async def test_controls_do_not_invent_actions(control):
    display = Transcript()
    chunks = []
    tasks = MockTaskExecutor(music=MusicController())

    async def sink(pcm):
        chunks.append(pcm)

    async with Session(Speech(), NoModel(), sink, tasks=tasks, transcript=display) as s:
        s.activate()
        await (await s.submit(b"Turn on the lights."))
        before = len(chunks)
        await (await s.submit(control.encode()))
        assert tasks.actions.effects == 1 and tasks.actions.light
        if control.startswith("Stop"):
            assert len(chunks) == before and s.state == State.LISTENING
        elif control.startswith("Cancel"):
            assert "Which task" in s.history[-1]["content"]
        else:
            assert s.state == State.IDLE
        assert s.playback.queue.empty()


@pytest.mark.parametrize("phase", ["before_effect", "after_effect"])
async def test_pending_effect_is_unknown_on_speech_cancel_not_falsely_cancelled(phase):
    gate = asyncio.Event()
    submitted = asyncio.Event()
    display = Transcript()

    class Pending(MockTaskExecutor):
        async def execute(self, key, call):
            async def operation():
                submitted.set()
                if phase == "before_effect":
                    await gate.wait()
                result = await super(Pending, self).execute(key, call)
                if phase == "after_effect":
                    await gate.wait()
                return result

            self.job = asyncio.create_task(operation())
            return await asyncio.shield(self.job)

        async def close(self):
            gate.set()
            if hasattr(self, "job"):
                await self.job
            await super().close()

    tasks = Pending(music=MusicController())

    async def sink(pcm):
        pass

    async with Session(Speech(), NoModel(), sink, tasks=tasks, transcript=display) as s:
        s.activate()
        generation = await s.submit(b"Turn on the lights.")
        await submitted.wait()
        if phase == "after_effect":
            async with asyncio.timeout(1):
                while tasks.actions.effects != 1:
                    await asyncio.sleep(0)
        await s.interrupt()
        await asyncio.gather(generation, return_exceptions=True)
        row = display.snapshot()["tasks"][0]
        expected = "simulated" if phase == "after_effect" else "unknown"
        assert row["status"] == expected
        gate.set()
        await tasks.job
        assert tasks.actions.effects == 1 and tasks.actions.light
        await (await s.submit(b"Stop speaking."))
        assert tasks.actions.effects == 1 and display.snapshot()["tasks"][0]["status"] == expected
        assert s.playback.queue.empty()


async def test_abandoned_interruption_does_not_respeak_or_repeat_effect():
    entered = asyncio.Event()
    display = Transcript()
    tasks = MockTaskExecutor(music=MusicController())

    async def sink(pcm):
        entered.set()
        await asyncio.Event().wait()

    async with Session(Speech(), MockLLM(), sink, tasks=tasks, transcript=display) as s:
        s.activate()
        old = await s.submit(b"Turn on the lights.")
        await entered.wait()
        await s.interrupt(for_new_input=True)
        await asyncio.gather(old, return_exceptions=True)
        await (await s.submit(b""))
        assert tasks.actions.effects == 1 and tasks.actions.light
        assert s.state == State.LISTENING and s.playback.queue.empty()
        assert display.snapshot()["tasks"][0]["status"] == "simulated"


async def test_bounded_mixed_turns_and_reconnects_release_transient_work():
    baseline = set(asyncio.all_tasks())
    tasks = MockTaskExecutor(music=MusicController())

    async def sink(pcm):
        await asyncio.sleep(0)

    try:
        for _ in range(4):
            async with Session(
                Speech(),
                NoModel(),
                sink,
                tasks=tasks,
                close_tasks=False,
                conversation_mode="followup",
            ) as s:
                s.activate()
                for text in (
                    "Set brightness to 70 percent.",
                    "Actually, 30 percent.",
                    "Stop speaking.",
                    "Turn off the lights.",
                    "Cancel that.",
                    "",
                ):
                    await (await s.submit(text.encode()))
                    assert s.state == State.LISTENING and s.playback.queue.empty()
                await s.interrupt()
                await (await s.submit(b"End conversation."))
                assert s.state == State.IDLE
            assert not s.playback.queue.qsize()
        assert tasks.actions.effects == 12 and not tasks.actions.light
    finally:
        await tasks.close()
    await asyncio.sleep(0)
    assert not [t for t in asyncio.all_tasks() - baseline if not t.done()]


@pytest.mark.parametrize("late_tool", [False, True])
async def test_unrelated_command_fences_old_reasoning_even_if_provider_returns_after_cancel(
    late_tool,
):
    from me3_voice.providers import ToolCall

    started = asyncio.Event()

    class LateModel:
        async def respond(self, history):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                yield (
                    ToolCall("obsolete-effect", "run_device_action", {"action": "light_off"})
                    if late_tool
                    else "obsolete reply"
                )

    tasks = MockTaskExecutor(music=MusicController())
    display = Transcript()

    async def sink(pcm):
        pass

    async with Session(Speech(), LateModel(), sink, tasks=tasks, transcript=display) as s:
        s.activate()
        old = await s.submit(b"Tell me one short joke.")
        await started.wait()
        await (await s.submit(b"Turn on the lights."))
        await asyncio.gather(old, return_exceptions=True)
        assert tasks.actions.effects == 1 and tasks.actions.light
        assert len(display.snapshot()["tasks"]) == 1
        assert not any("obsolete" in row.get("content", "") for row in s.history)
        assert s.state == State.LISTENING and s.playback.queue.empty()


async def test_failed_playback_does_not_poison_next_turn_or_repeat_completed_action():
    fail = [True]
    chunks = []

    async def sink(pcm):
        if fail[0]:
            fail[0] = False
            raise ConnectionError("controlled sink failure")
        chunks.append(pcm)

    tasks = MockTaskExecutor(music=MusicController())
    display = Transcript()
    async with Session(Speech(), MockLLM(), sink, tasks=tasks, transcript=display) as s:
        s.activate()
        await (await s.submit(b"Turn on the lights."))
        assert tasks.actions.effects == 1
        assert any(e["event"] == "turn_error" for e in s.events)
        assert display.snapshot()["tasks"][0]["status"] == "simulated"
        before = len(chunks)
        await (await s.submit(b"Hello again."))
        assert len(chunks) > before and tasks.actions.effects == 1
        assert s.history[-1]["role"] == "assistant"
        assert s.state == State.LISTENING and s.playback.queue.empty()


async def test_ambiguous_cancel_preserves_light_and_timer_until_explicit_target():
    display = Transcript()
    tasks = MockTaskExecutor(music=MusicController())

    async def sink(pcm):
        pass

    async with Session(Speech(), MockLLM(), sink, tasks=tasks, transcript=display) as s:
        s.activate()
        await (await s.submit(b"simulate timer 10 seconds"))
        timer = tasks.actions.schedule["id"]
        await (await s.submit(b"Turn on the lights."))
        await (await s.submit(b"Cancel that."))
        assert tasks.actions.effects == 2 and tasks.actions.light
        assert tasks.actions.schedule["id"] == timer and tasks.actions.schedule["active"]
        assert "Which task" in s.history[-1]["content"] and s.awaiting_answer
        await (await s.submit(b"simulate cancel schedule"))
        assert tasks.actions.schedule is None and tasks.actions.light
        assert (
            tasks.actions.effects == 2
        )  # cancellation is tracked by its operation, not creation count
        assert len(tasks.actions.operations) == 3
        assert tasks.actions.output.display_state[0] == "    "
        assert display.snapshot()["tasks"][0]["status"] == "cancelled"
