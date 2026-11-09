"""Equivalent legacy Session fixture; same real project MCP boundaries, no APIs."""

import asyncio, json, sys
from pathlib import Path
from me3_voice.session import Session
from me3_voice.providers import ToolCall
from me3_voice.speech import SpeechClient
from me3_voice.playback import MemorySink


class Tasks:
    task_mode = "mock"
    communication_enabled = False

    def __init__(self):
        self.calls = []

    async def execute(self, key, call):
        assert call.arguments == {"word": "amber"}
        self.calls.append(call.call_id)
        return {"result_status": "completed", "summary": "Stored amber"}

    def cancel_pending(self):
        pass

    async def close(self):
        pass


class Model:
    def __init__(self):
        self.calls = 0

    async def respond(self, history):
        self.calls += 1
        if self.calls == 1:
            yield ToolCall("proof-tool-1", "fixture_memory", {"word": "amber"})
        else:
            text = (
                "Saved amber. " + "pending " * 30 + "Ready."
                if sum(x["role"] == "user" for x in history) == 1
                else "The word is amber."
            )
            yield text


async def case(interrupt):
    tasks = Tasks()
    model = Model()
    sink = MemorySink(delay=0.02)
    async with SpeechClient(
        transcript=["Save amber in memory.", "Which word?"], tts_delay=0.02, timeout=10
    ) as speech:
        async with Session(
            speech, model, sink, tasks=tasks, llm_timeout=10, turn_timeout=15
        ) as session:
            session.activate()
            first = await session.submit(b"\1\0" * 512)
            if interrupt:
                await asyncio.wait_for(sink.entered.wait(), 8)
                await session.interrupt()
            else:
                await first
            await (await session.submit(b"\1\0" * 512))
            assert tasks.calls == ["proof-tool-1"]
            assert "amber" in session.history[-1]["content"]
            assert session.playback.queue.empty()
            result = {
                "case": "interruption" if interrupt else "followup",
                "tool_calls": tasks.calls,
                "history": session.history,
                "events": session.events,
                "sink_chunks": len(sink.chunks),
                "llm_requests": model.calls,
                "queued_output": session.playback.queue.qsize(),
                "speech_protocols": speech.protocols,
                "external_requests": 0,
                "mode": "mock / real MCP / memory sink",
            }
        return result


async def main():
    data = [await case(False), await case(True)]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("var/results/phase2-session.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(data, f, indent=2)
    print("Equivalent Session cases passed:", len(data))


if __name__ == "__main__":
    asyncio.run(main())
