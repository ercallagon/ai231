import asyncio
import json

import pytest

from me3_voice.contracts import TurnKey
from me3_voice.providers import ToolCall
from me3_voice.scenarios import MockBackgroundTaskExecutor
from me3_voice.session import Session, State
from me3_voice.transcript import Transcript


class TextSpeech:
    async def transcribe(self, key, pcm):
        return pcm.decode()

    async def synthesize(self, key, text):
        for _ in range(8):
            await asyncio.sleep(0.001)
            yield b"mock pcm"


class ScenarioLLM:
    def __init__(self, executor):
        self.executor = executor
        self.progress_answer = ""

    async def respond(self, history):
        latest = next(row for row in reversed(history) if row.get("role") == "user")
        text = latest["content"].casefold()
        call_id = f"test-{latest['turn_id']}"
        if text == "start project":
            yield ToolCall(call_id, "run_agent_task", {"task": "Create the test project outline."})
        elif text == "set a light":
            yield ToolCall(call_id, "run_device_action", {"action": "light_on"})
        elif text == "change light color":
            yield ToolCall(call_id, "run_device_action", {"action": "set_color", "value": "blue"})
        elif text == "how is the project going?":
            self.progress_answer = self.executor.current_phase
            yield f"The mock task is working on: {self.progress_answer}."
        elif history[-1].get("role") == "tool":
            receipt = json.loads(history[-1]["content"])
            yield receipt.get("summary", "Mock action finished.")
        else:
            yield "Ready."


async def sink(_pcm):
    await asyncio.sleep(0.04)


async def finish_turn(session, text):
    generation = await session.submit(text.encode())
    await generation


@pytest.mark.asyncio
async def test_background_task_ack_interleaves_with_turns_and_survives_speech_interrupt(tmp_path):
    worker = MockBackgroundTaskExecutor(tmp_path / "output", duration_seconds=3)
    transcript = Transcript(mode="mock")
    llm = ScenarioLLM(worker)
    async with Session(
        TextSpeech(), llm, sink, transcript=transcript, tasks=worker, turn_timeout=None
    ) as session:
        session.activate()
        await finish_turn(session, "start project")
        task = transcript.snapshot()["tasks"][0]
        assert task["status"] in {"accepted", "running"}
        assert "few minutes" in session.history[-1]["content"]
        task_id = task["task_id"]
        original_key = TurnKey(session_id=session.session_id, turn_id=1)
        duplicate = await worker.execute_background(
            original_key,
            ToolCall("test-1", "run_agent_task", {"task": "Create the test project outline."}),
            on_update=lambda **update: session.background_task_update(
                original_key, "test-1", **update
            ),
        )
        assert duplicate["task_id"] == task_id
        assert worker.started_count == 1

        await finish_turn(session, "set a light")
        assert worker.started_count == 1
        assert worker.actions.light is True
        assert transcript.snapshot()["tasks"][0]["status"] in {"accepted", "running"}

        generation = await session.submit(b"how is the project going?")
        async with asyncio.timeout(2):
            while session.state is not State.SPEAKING:
                await asyncio.sleep(0.001)
        assert llm.progress_answer
        await session.interrupt()
        await asyncio.gather(generation, return_exceptions=True)
        assert transcript.snapshot()["tasks"][0]["status"] == "running"
        assert session.state is State.LISTENING
        assert session.playback.queue.empty()

        # A barge-in with no committed follow-up cancels only the spoken status.
        # The accepted project task must continue advancing under its stable ID.
        phase_before = worker.current_phase
        async with asyncio.timeout(1.5):
            while worker.current_phase == phase_before:
                await asyncio.sleep(0.005)
        assert transcript.snapshot()["tasks"][0]["status"] == "running"
        assert worker.started_count == 1

        # An unrelated device action remains available while that task runs.
        await finish_turn(session, "change light color")
        assert worker.actions.color == "blue"
        assert transcript.snapshot()["tasks"][0]["status"] == "running"
        assert worker.started_count == 1

        jobs = [job for _, job in worker.operations.values()]
        await asyncio.gather(*jobs)
        final = transcript.snapshot()["tasks"][0]
        assert final["task_id"] == task_id
        assert final["status"] == "completed"
        assert final["absolute_path"].endswith("/output/project-plan.md")
        assert worker.started_count == 1
        assert worker.actions.effects == 2
        progress_events = [
            event for event in session.events if event["event"] == "background_task_update"
        ]
        assert any(event.get("task_phase") for event in progress_events)
        assert any(event.get("task_heartbeat") is True for event in progress_events)
        assert all(event.get("task_id") == task_id for event in progress_events)


@pytest.mark.asyncio
@pytest.mark.parametrize(("outcome", "expected"), [("failure", "failed"), ("success", "cancelled")])
async def test_mock_background_failure_and_explicit_cancellation_are_truthful(
    tmp_path, outcome, expected
):
    worker = MockBackgroundTaskExecutor(tmp_path / outcome, duration_seconds=2, outcome=outcome)
    transcript = Transcript(mode="mock")
    llm = ScenarioLLM(worker)
    async with Session(
        TextSpeech(), llm, sink, transcript=transcript, tasks=worker, turn_timeout=None
    ) as session:
        session.activate()
        await finish_turn(session, "start project")
        task_id = transcript.snapshot()["tasks"][0]["task_id"]
        if outcome == "success":
            assert await worker.cancel_task(task_id)
        else:
            await asyncio.gather(*(job for _, job in worker.operations.values()))
        assert transcript.snapshot()["tasks"][0]["status"] == expected
