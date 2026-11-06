"""Context contracts without a live model; no claims of semantic reasoning quality."""

import asyncio
import json
import pytest
from me3_voice.onit_tasks import OnitPrimaryReasoning, OnitTaskUnavailable
from me3_voice.session import Session, State
from me3_voice.providers import MockLLM, ToolCall
from me3_voice.tools import MockTaskExecutor
from me3_voice.music import MusicController
from me3_voice.transcript import Transcript


class Speech:
    async def transcribe(self, key, pcm):
        return pcm.decode()

    async def synthesize(self, key, text):
        yield b"\0\0"


async def sink(pcm):
    pass


async def payload(history):
    return (await anext(OnitPrimaryReasoning().respond(history))).arguments["task"]


def tool(turn, status="completed", summary="ACTION-COMPLETE"):
    return {
        "role": "tool",
        "turn_id": turn,
        "tool_call_id": f"c{turn}",
        "tool_request": {
            "name": "run_device_action",
            "arguments": json.dumps({"action": "set_brightness", "value": "30"}),
        },
        "content": json.dumps({"result_status": status, "summary": summary}),
    }


async def test_onit_preserves_receipts_corrections_and_latest_tail_without_recursive_context():
    rows = [
        {"role": "user", "content": "My location is Marilao, Bulacan.", "turn_id": 1},
        tool(1),
        {"role": "user", "content": "Correction: use Manila instead.", "turn_id": 2},
        tool(2, "unknown", "NOT-CONFIRMED"),
        {"role": "user", "content": "x" * 1700 + " KEEP-THE-LAST-CORRECTION", "turn_id": 3},
    ]
    value = await payload(rows)
    assert all(
        x in value
        for x in [
            "Marilao",
            "use Manila instead",
            "ACTION-COMPLETE",
            "NOT-CONFIRMED",
            "KEEP-THE-LAST-CORRECTION",
            '"value":"30"',
        ]
    )
    rows[1]["tool_request"] = {
        "name": "run_agent_task",
        "arguments": json.dumps({"task": "RECURSIVE-OLD-PROMPT"}),
    }
    assert "RECURSIVE-OLD-PROMPT" not in await payload(rows)
    assert len(value) <= 8000


async def test_context_capacity_is_explicit_without_silent_forgetting():
    with pytest.raises(OnitTaskUnavailable) as e:
        await payload([{"role": "user", "content": "x" * 7900, "turn_id": 1}])
    assert e.value.code == "context_capacity"


async def test_failed_action_is_present_as_failure_in_next_actual_context():
    class Failure(RuntimeError):
        task_status = "failed"
        user_summary = "Mock target rejected this request before execution."

    class Failing(MockTaskExecutor):
        async def execute(self, key, call):
            raise Failure()

    class Peer:
        histories = []

        async def respond(self, history):
            self.histories.append(history)
            yield "Context inspected."

    tasks = Failing(music=MusicController())
    peer = Peer()
    display = Transcript()
    async with Session(Speech(), peer, sink, tasks=tasks, transcript=display) as s:
        s.activate()
        await (await s.submit(b"Turn on the lights."))
        assert tasks.actions.effects == 0 and display.snapshot()["tasks"][0]["status"] == "failed"
        await (await s.submit(b"Was that successful?"))
        receipts = [r for r in peer.histories[-1] if r["role"] == "tool"]
        assert (
            len(receipts) == 1 and json.loads(receipts[0]["content"])["result_status"] == "failed"
        )
        assert "rejected" in await payload(peer.histories[-1])


async def test_reference_after_intervening_question_reaches_model_without_guessing_old_parameters():
    class Peer:
        histories = []

        async def respond(self, history):
            self.histories.append(history)
            yield "Please specify which setting you mean."

    peer = Peer()
    tasks = MockTaskExecutor(music=MusicController())
    async with Session(Speech(), peer, sink, tasks=tasks) as s:
        s.activate()
        await (await s.submit(b"Set brightness to 70 percent."))
        await (await s.submit(b"Actually, 30 percent."))
        assert tasks.actions.brightness == 30 and tasks.actions.effects == 2
        await (await s.submit(b"What is a timer?"))
        await (await s.submit(b"Actually, 20 percent."))
        assert tasks.actions.effects == 2  # no guessed cross-topic direct action
        context = await payload(peer.histories[-1])
        assert (
            "70 percent" in context
            and "30 percent" in context
            and "What is a timer?" in context
            and "20 percent" in context
        )
        assert s.state == State.LISTENING


async def test_end_expiry_reactivation_new_sessions_and_shared_receipts_do_not_inherit_chat():
    clock = [0.0]
    tasks = MockTaskExecutor(music=MusicController())
    display = Transcript()
    async with Session(
        Speech(),
        MockLLM(),
        sink,
        tasks=tasks,
        close_tasks=False,
        transcript=display,
        conversation_mode="followup",
        clock=lambda: clock[0],
    ) as one:
        one.activate()
        await (await one.submit(b"My project name is UniqueProject."))
        history = list(one.history)
        await one.exit()
        assert one.history == history  # retained for ended-session evidence only
        one.activate()
        assert one.history == [] and display.snapshot()["messages"] == []
        await (await one.submit(b"Hello."))
        clock[0] = one.engagement_deadline
        await one.tick()
        assert one.state == State.IDLE
        one.activate()
        assert one.history == []
    async with Session(Speech(), MockLLM(), sink, tasks=tasks, close_tasks=False) as two:
        two.activate()
        await (await two.submit(b"Hello."))
        assert "UniqueProject" not in await payload(two.history)
    await tasks.close()


async def test_recovered_receipt_does_not_restore_context_speak_or_reexecute(tmp_path, monkeypatch):
    from pathlib import Path
    from me3_voice.receipts import ReceiptStore
    from me3_voice.contracts import TurnKey

    monkeypatch.chdir(tmp_path)
    call = ToolCall("ui-" + "3" * 32, "run_device_action", {"action": "light_on"})
    first = MockTaskExecutor(music=MusicController())
    first.receipt_store = ReceiptStore(Path("var/cache/recovered.db"))
    await first.execute(TurnKey(session_id="old", turn_id=1), call)
    await first.close()
    tasks = MockTaskExecutor(music=MusicController())
    tasks.receipt_store = ReceiptStore(Path("var/cache/recovered.db"))
    display = Transcript()
    audio = []

    async def capture(pcm):
        audio.append(pcm)

    async with Session(Speech(), MockLLM(), capture, tasks=tasks, transcript=display) as s:
        s.activate()
        await s.submit_ui_device_action(call)
        await s.generation
        assert tasks.actions.effects == 0 and audio == []
        row = display.snapshot()["tasks"][0]
        assert row["status"] == "reported" and "Recorded" in row["summary"] and "widget" not in row
        assert not any(r["role"] == "assistant" for r in s.history)
        await (await s.submit(b"Hello."))
        assert audio and tasks.actions.effects == 0


async def test_interrupted_ack_keeps_completed_action_in_actual_onit_payload():
    started = asyncio.Event()
    display = Transcript()
    tasks = MockTaskExecutor(music=MusicController())

    async def blocked(pcm):
        started.set()
        await asyncio.Event().wait()

    async with Session(Speech(), MockLLM(), blocked, tasks=tasks, transcript=display) as s:
        s.activate()
        old = await s.submit(b"Set brightness to 70 percent.")
        await started.wait()
        await s.interrupt()
        await asyncio.gather(old, return_exceptions=True)
        assert tasks.actions.effects == 1 and tasks.actions.brightness == 70
        context = await payload(s.history)
        assert "set_brightness" in context and "70" in context and "simulated" in context
        assert not any(row["role"] == "assistant" for row in s.history)
        assert display.snapshot()["tasks"][0]["status"] == "simulated"


async def test_context_capacity_is_actionable_session_error_and_fresh_activation_recovers():
    tasks = MockTaskExecutor(music=MusicController())
    display = Transcript()
    async with Session(Speech(), MockLLM(), sink, tasks=tasks, transcript=display) as s:
        s.activate()
        await (await s.submit(b"Turn on the lights."))
        assert tasks.actions.effects == 1
        prior_receipt = next(row for row in s.history if row["role"] == "tool")
        s.llm = OnitPrimaryReasoning()
        await (await s.submit(b"x" * 7900))
        errors = [e for e in s.events if e["event"] == "turn_error"]
        assert errors[-1]["error_code"] == "context_capacity"
        assert tasks.actions.effects == 1 and s.playback.queue.empty()
        assert prior_receipt in s.history
        assert display.snapshot()["tasks"][0]["status"] == "simulated"
        await s.exit()
        s.activate()
        assert s.history == []
        assert len(await payload([{"role": "user", "content": "Hello", "turn_id": 1}])) < 8000


def test_direct_reasoning_payload_keeps_corrections_and_failed_receipts():
    from me3_voice.remote import ChatReasoning

    history = [
        {"role": "user", "content": "My location is Marilao.", "turn_id": 1},
        tool(1, "failed", "Not executed."),
        {"role": "user", "content": "Correction: use Manila.", "turn_id": 2},
    ]
    messages = ChatReasoning.messages(history)
    encoded = json.dumps(messages)
    assert "Marilao" in encoded and "use Manila" in encoded
    assert "Not executed" in encoded and "failed" in encoded
    assert messages[-1]["role"] == "user" and messages[-1]["content"] == "Correction: use Manila."


def test_actual_process_restart_does_not_load_saved_transcript_as_chat(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    source = str(Path(__file__).resolve().parents[1] / "src")
    script = tmp_path / "chat_process.py"
    script.write_text("""
import asyncio,json,sys
from pathlib import Path
from me3_voice.session import Session
from me3_voice.providers import MockLLM
from me3_voice.tools import MockTaskExecutor
from me3_voice.music import MusicController
from me3_voice.transcript import Transcript
class Speech:
 async def transcribe(self,key,pcm):return pcm.decode()
 async def synthesize(self,key,text):yield b"\\0\\0"
class Peer:
 seen=None
 async def respond(self,history):self.seen=history;yield "A deterministic reply."
async def main():
 peer=Peer();display=Transcript()
 async def sink(pcm):pass
 async with Session(Speech(),peer,sink,tasks=MockTaskExecutor(music=MusicController()),transcript=display) as s:
  s.activate();await(await s.submit(b"My codeword is SyntheticOrchid." if sys.argv[1]=="first" else b"Hello."))
  Path(sys.argv[1]+"-context.json").write_text(json.dumps(peer.seen))
  if sys.argv[1]=="first":
   p=Path("var/results/ui-transcript.json");p.parent.mkdir(parents=True);p.write_text(json.dumps(display.snapshot()))
asyncio.run(main())
""")
    env = {**os.environ, "PYTHONPATH": source, "PYTHONDONTWRITEBYTECODE": "1"}
    for stage in ("first", "restarted"):
        run = subprocess.run(
            [sys.executable, "-B", str(script), stage],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=15,
        )
        assert run.returncode == 0, run.stderr.decode()
    assert "SyntheticOrchid" in (tmp_path / "var/results/ui-transcript.json").read_text()
    assert "SyntheticOrchid" not in (tmp_path / "restarted-context.json").read_text()
    assert json.loads((tmp_path / "restarted-context.json").read_text()) == [
        {"role": "user", "content": "Hello.", "turn_id": 1}
    ]


async def test_two_similar_mock_background_tasks_keep_ids_and_ambiguous_cancel_clarifies(
    tmp_path, monkeypatch
):
    from me3_voice.scenarios import MockBackgroundTaskExecutor

    monkeypatch.delenv("ME3_MUSIC_CONFIG", raising=False)
    worker = MockBackgroundTaskExecutor(tmp_path / "drafts", duration_seconds=60)
    display = Transcript()
    async with Session(Speech(), MockLLM(), sink, tasks=worker, transcript=display) as s:
        s.activate()
        await (await s.submit(b"agent task: Create project Alpha draft."))
        await (await s.submit(b"agent task: Create project Beta draft."))
        rows = display.snapshot()["tasks"]
        ids = [r["task_id"] for r in rows]
        assert len(set(ids)) == 2 and worker.started_count == 2
        assert all(r["status"] in {"accepted", "running"} for r in rows)
        await (await s.submit(b"Cancel that."))
        assert worker.started_count == 2 and all(
            not job.done() for _, job in worker.operations.values()
        )
        assert "Which task" in s.history[-1]["content"]
        context = await payload(s.history)
        assert all(identity in context for identity in ids) and "accepted" in context
    assert not list((tmp_path / "drafts").glob("*"))  # shut down before any fixture output phase


async def test_late_outcome_updates_reasoning_history_even_without_dashboard():
    tasks = MockTaskExecutor(music=MusicController())
    gate = asyncio.Event()
    started = asyncio.Event()
    original = tasks.actions._run

    async def delayed(*args):
        started.set()
        await gate.wait()
        return await original(*args)

    tasks.actions._run = delayed
    async with Session(Speech(), MockLLM(), sink, tasks=tasks) as s:
        s.activate()
        job = await s.submit(b"Turn on the lights.")
        await started.wait()
        await s.interrupt()
        await asyncio.gather(job, return_exceptions=True)
        assert json.loads(s.history[-1]["content"])["result_status"] == "unknown"
        gate.set()
        async with asyncio.timeout(1):
            while json.loads(s.history[-1]["content"])["result_status"] == "unknown":
                await asyncio.sleep(0)
        assert "simulated" in await payload(s.history)
        assert tasks.actions.effects == 1 and s.playback.queue.empty()


@pytest.mark.parametrize("tool_request", [None, [], {"arguments": "not-json"}])
async def test_malformed_tool_context_fails_explicitly_without_model_call(tool_request):
    with pytest.raises(OnitTaskUnavailable) as error:
        await payload(
            [
                {"role": "user", "content": "What happened?", "turn_id": 2},
                {"role": "tool", "tool_request": tool_request, "content": "{}", "turn_id": 1},
            ]
        )
    assert error.value.code == "invalid_context"


async def test_lossless_context_compaction_preserves_every_turn_and_correction():
    records = [
        {
            "role": "user" if n % 2 == 0 else "assistant",
            "turn_id": n // 2 + 1,
            "content": ("Correction applies here. " if n % 7 == 0 else "Conversation note. ")
            + str(n)
            + "x" * 73,
        }
        for n in range(60)
    ]
    encoded = await payload(records)
    assert "Lossless row schema" in encoded and len(encoded) <= 8000
    compact = json.loads(encoded.split("\n")[-1])
    assert compact == [[r["role"], r["turn_id"], r["content"]] for r in records]
    assert len(records) == 60


async def test_compacted_context_reaches_actual_session_executor_boundary_without_loss():
    seen = []

    class InspectingExecutor(MockTaskExecutor):
        async def execute(self, key, call):
            assert call.name == "run_agent_task"
            seen.append(call.arguments["task"])
            return {
                "summary": "Context accepted by deterministic peer.",
                "mode": "local-answer",
                "answer": "Context accepted by deterministic peer.",
            }

    tasks = InspectingExecutor(music=MusicController())
    rows = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "turn_id": i // 2 + 1,
            "content": f"Record {i}: " + "context " * 11,
        }
        for i in range(60)
    ]
    # Include uncertainty and the latest corrected parameter, not just dialogue.
    rows.insert(3, tool(2, "unknown", "Brightness outcome is not confirmed."))
    display = Transcript()
    async with Session(
        Speech(), OnitPrimaryReasoning(), sink, tasks=tasks, transcript=display
    ) as s:
        s.activate()
        s.history.extend(rows)
        await (await s.submit(b"Use 30 percent instead; keep the earlier outcome uncertain."))
        assert len(seen) == 1 and len(seen[0]) <= 8000
        assert "Lossless row schema" in seen[0]
        decoded = json.loads(seen[0].split("\n")[-1])
        dialogue = [r for r in decoded if r[0] != "task_receipt"]
        assert dialogue[:-1] == [
            [r["role"], r["turn_id"], r["content"]] for r in rows if r["role"] != "tool"
        ]
        assert dialogue[-1][2].startswith("Use 30 percent")
        receipt = next(r for r in decoded if r[0] == "task_receipt")
        assert receipt[-1]["result_status"] == "unknown"
        assert receipt[4]["value"] == "30"
        assert not any(e["event"] == "turn_error" for e in s.events)
        assert tasks.actions.effects == 0 and s.playback.queue.empty()
