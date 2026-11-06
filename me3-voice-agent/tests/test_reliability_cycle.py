import asyncio

import pytest

from me3_voice.scenarios import run_text_injected_stress
from me3_voice.playback import MemorySink
from me3_voice.providers import MockLLM
from me3_voice.session import Session
from me3_voice.transcript import Transcript


@pytest.mark.parametrize("turns,concurrency", [(3, 2), (10, 3)])
async def test_text_injected_concurrent_turn_cycle_is_bounded_and_recovers(turns, concurrency):
    report = await run_text_injected_stress(turns=turns, concurrency=concurrency)

    assert report["passed"] is True
    assert report["completed_turns"] == turns * concurrency
    assert report["mock_task_effects"] == ((turns + 9) // 10 + (turns + 8) // 10) * concurrency
    assert report["playback_queue_capacity"] == 8
    assert report["playback_queue_drained_each_turn"] is True
    assert report["leaked_me3_tasks"] == 0
    assert report["latency_aggregates"]
    assert all(row["mode"] == "mock" for row in report["latency_aggregates"])


async def test_repeated_post_asr_interruption_keeps_task_receipts_and_corrects_only_target():
    """Repeat action-complete → speech-interrupt → unrelated task → correction."""

    pcm = b"\1\0" * 512

    class InjectedSpeech:
        def __init__(self):
            self.lines = [
                line
                for _ in range(8)
                for line in ("simulate light on", "simulate color red", "simulate light off")
            ]

        async def transcribe(self, key, _pcm):
            return self.lines[key.turn_id - 1]

        async def synthesize(self, _key, _text):
            for _ in range(4):
                await asyncio.sleep(0)
                yield pcm

    sink = MemorySink()
    transcript = Transcript(
        mode="mock",
        backend_modes={"asr": "mock", "llm": "mock", "tts": "mock"},
        playback="memory",
    )
    async with Session(InjectedSpeech(), MockLLM(), sink, transcript=transcript) as session:
        session.activate(activation_event="text_injected_activation")
        completed_targets = 0
        interrupted_outputs = 0
        for cycle in range(8):
            # The tool receipt is committed before the deliberately blocked speech sink.
            sink.delay = 30
            sink.entered.clear()
            old = await session.submit(pcm)
            await asyncio.wait_for(sink.entered.wait(), 2)
            assert session.tasks.actions.light is True
            assert session.tasks.actions.effects == completed_targets + 1
            await session.interrupt()
            assert old.done() and old.cancelled()
            interrupted_outputs += 1
            completed_targets += 1
            assert session.playback.queue.empty()
            assert not any(
                row["role"] == "assistant" and row["turn_id"] == cycle * 3 + 1
                for row in session.history
            )

            # An unrelated completed color task remains applied after later light correction.
            sink.delay = 0
            await asyncio.wait_for(await session.submit(pcm), 2)
            assert session.tasks.actions.color == "red"
            completed_targets += 1
            await asyncio.wait_for(await session.submit(pcm), 2)
            assert session.tasks.actions.light is False
            completed_targets += 1
            assert session.tasks.actions.effects == completed_targets
            assert session.state.value == "listening"
            assert session.playback.queue.empty()

        receipts = transcript.snapshot()["tasks"]
        assert len(receipts) == 24
        assert all(row["status"] in {"simulated", "completed"} for row in receipts)
        assert len([row for row in session.history if row["role"] == "user"]) == 24
        assert interrupted_outputs == 8
        assert session.tasks.actions.color == "red"
        assert session.tasks.actions.light is False

    this = asyncio.current_task()
    assert not [task for task in asyncio.all_tasks() if task is not this and not task.done()]
