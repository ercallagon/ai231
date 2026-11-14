"""Offline native Pipecat proof. Real project speech MCP subprocesses; no model API."""

import asyncio
import json
import time
import os
import sys
from pathlib import Path

from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    InterruptionFrame,
    LLMTextFrame,
    ErrorFrame,
)
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from me3_voice.contracts import TurnKey
from me3_voice.speech import SpeechClient
from me3_voice.native_pipeline import make_native_pipeline


class OfflineGroq(GroqLLMService):
    """Only replace network completion generation; native parser/tools/context run."""

    def __init__(self, events, background=False):
        self.background = background
        super().__init__(api_key="offline-not-a-provider-key")
        self.events = events
        self.contexts = []
        self.stream_closed = 0

    async def get_chat_completions(self, context):
        messages = json.loads(json.dumps(context.get_messages()))
        self.contexts.append(messages)
        self.events.append({"event": "llm_request", "at": time.monotonic()})
        tool_done = len(self.contexts) > 1
        user_count = sum(m.get("role") == "user" for m in messages)

        async def chunks():
            try:
                if not tool_done:
                    deltas = [
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "proof-tool-1",
                                    "type": "function",
                                    "function": {"name": "remember", "arguments": '{"word":'},
                                }
                            ]
                        },
                        {"tool_calls": [{"index": 0, "function": {"arguments": '"amber"}'}}]},
                    ]
                else:
                    text = (
                        ("Saved amber. " + "pending " * 30 + "Ready.")
                        if user_count == 1
                        else "The word is amber."
                    )
                    deltas = [{"content": word + " "} for word in text.split()]
                if not tool_done and self.background:
                    deltas.insert(0, {"content": "Working now. "})
                for delta in deltas:
                    await asyncio.sleep(0.005)
                    yield ChatCompletionChunk(
                        id="offline",
                        object="chat.completion.chunk",
                        created=0,
                        model="offline-fixture",
                        choices=[{"index": 0, "delta": delta, "finish_reason": None}],
                    )
            finally:
                self.stream_closed += 1
                self.events.append({"event": "llm_stream_closed", "at": time.monotonic()})

        return chunks()


class Recorder(FrameProcessor):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.errors = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            self.errors.append(str(frame))
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (InterruptionFrame, LLMTextFrame, TTSStartedFrame, TTSStoppedFrame)
        ):
            self.events.append({"event": type(frame).__name__, "at": time.monotonic()})
        await self.push_frame(frame, direction)


class MemoryOutput(BaseOutputTransport):
    def __init__(self, events):
        super().__init__(
            TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=16000,
                audio_out_channels=1,
                audio_out_10ms_chunks=2,
            )
        )
        self.events = events
        self.writes = []
        self.first_write = asyncio.Event()

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        self.writes.append((time.monotonic(), len(frame.audio)))
        self.first_write.set()
        await asyncio.sleep(len(frame.audio) / 32000)
        return True


async def until(predicate, timeout=8):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def distribution(values):
    values = sorted(values)
    return {
        "n": len(values),
        "p50": values[len(values) // 2] if values else None,
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))] if values else None,
        "max": max(values) if values else None,
    }


async def run_case(interrupt=False, background=False):
    events = []
    cpu_started = time.process_time()
    tool_calls = []
    background_release = asyncio.Event()
    handler_cancelled = []
    lag = []
    pipeline_ready_at = None

    async def heartbeat():
        while True:
            expected = time.monotonic() + 0.01
            await asyncio.sleep(0.01)
            lag.append((time.monotonic(), max(0, time.monotonic() - expected) * 1000))

    heartbeat_task = asyncio.create_task(heartbeat())
    async with SpeechClient(
        transcript=["Save amber in memory.", "Which word?"], tts_delay=0.02, timeout=10
    ) as speech:
        context = LLMContext(
            [{"role": "system", "content": "Offline lifecycle fixture."}],
            tools=ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        "remember", "Store one fixture word", {"word": {"type": "string"}}, ["word"]
                    )
                ]
            ),
        )
        llm = OfflineGroq(events, background=background)
        rec = Recorder(events)
        output = MemoryOutput(events)

        async def remember(params):
            assert params.arguments == {"word": "amber"}
            try:
                if background:
                    await background_release.wait()
            except asyncio.CancelledError:
                handler_cancelled.append(True)
                raise
            tool_calls.append(params.tool_call_id)
            await params.result_callback({"stored": "amber"})

        llm.register_function("remember", remember, cancel_on_interruption=not background)
        parts = make_native_pipeline(
            speech,
            llm,
            output,
            context=context,
            after_tts=[rec],
            session_id="native-proof",
            silence_seconds=0.05,
            on_event=events.append,
        )
        worker = PipelineWorker(
            parts.pipeline,
            params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
            enable_rtvi=False,
            enable_turn_tracking=False,
            idle_timeout_secs=None,
        )
        ready = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def started(*_):
            nonlocal pipeline_ready_at
            pipeline_ready_at = time.monotonic()
            ready.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        task = asyncio.create_task(runner.run())

        async def utterance():
            before = sum(e["event"] == "asr_append" for e in events)
            await worker.queue_frame(VADUserStartedSpeakingFrame())
            await worker.queue_frame(InputAudioRawFrame(b"\x01\0" * 512, 16000, 1))
            await until(lambda: sum(e["event"] == "asr_append" for e in events) > before)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())

        try:
            await asyncio.wait_for(ready.wait(), 8)
            await utterance()
            await asyncio.wait_for(output.first_write.wait(), 8)
            if interrupt:
                requested = time.monotonic()
                events.append({"event": "interrupt_fixture_onset", "at": requested})
                await worker.queue_frame(VADUserStartedSpeakingFrame())
                await until(
                    lambda: any(
                        x["event"] == "InterruptionFrame" and x["at"] >= requested for x in events
                    )
                )
                await asyncio.sleep(0.25)
                n = len(output.writes)
                await asyncio.sleep(0.15)
                assert len(output.writes) == n, "old output continued after drain"
                if background:
                    assert not handler_cancelled and not tool_calls
                    background_release.set()
                    await until(lambda: len(tool_calls) == 1)
                    await until(
                        lambda: any(
                            m.get("role") in {"tool", "developer"}
                            and "stored" in str(m.get("content"))
                            for m in context.get_messages()
                        )
                    )
                else:
                    assert not any(
                        "Ready." in str(m.get("content")) for m in context.get_messages()
                    ), "unspoken tail leaked into history"
            else:
                await until(
                    lambda: any(
                        m.get("role") == "assistant"
                        and m.get("content")
                        and "Ready." in str(m["content"])
                        for m in context.get_messages()
                    )
                )
            await utterance()
            await until(
                lambda: any(
                    m.get("role") == "assistant" and "The word is amber." in str(m.get("content"))
                    for m in context.get_messages()
                )
            )
            assert tool_calls == ["proof-tool-1"], tool_calls
            assert len([e for e in events if e["event"] == "asr_final"]) == 2
            assert len([e for e in events if e["event"] == "asr_partial"]) == 2
            assert not rec.errors, rec.errors
            result = {
                "case": "background-survives-interruption"
                if background
                else "interruption"
                if interrupt
                else "followup",
                "mode": "mock real stdio MCP, native Pipecat, memory output",
                "tool_calls": tool_calls,
                "messages": context.get_messages(),
                "events": events,
                "writes": len(output.writes),
                "output_bytes": sum(b for _, b in output.writes),
                "llm_requests": len(llm.contexts),
                "llm_streams_closed": llm.stream_closed,
                "speech_protocols": speech.protocols,
                "external_requests": 0,
                "background_handler_cancelled": handler_cancelled,
                "event_loop_lag_ms": distribution([value for _, value in lag]),
                "loop_lag_startup_ms": distribution(
                    [value for at, value in lag if at < pipeline_ready_at + 0.2]
                ),
                "loop_lag_steady_ms": distribution(
                    [value for at, value in lag if at >= pipeline_ready_at + 0.2]
                ),
                "lag_window": "startup includes process/pipeline startup and first200ms after ready; steady follows",
                "cpu_process_seconds": time.process_time() - cpu_started,
                "resource_snapshot": resources(),
                "limitations": "HPC software fixture; no acoustic or provider performance inference",
            }
        finally:
            await runner.cancel()
            await asyncio.wait_for(task, 10)
            await llm._client.close()
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        # Real child service operations must be reusable after pipeline cancellation.
        assert await speech.transcribe(
            TurnKey(session_id="post-cleanup", turn_id=1), b"\x01\0" * 512
        )
        assert (
            len(
                [
                    x
                    async for x in speech.synthesize(
                        TurnKey(session_id="post-cleanup", turn_id=2), "Reuse."
                    )
                ]
            )
            == 4
        )
        result["mcp_reuse_after_pipeline_cleanup"] = True
        return result


def resources():
    result = []
    pids = [os.getpid()]
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    if children.exists():
        pids += [int(x) for x in children.read_text().split()]
    for pid in pids:
        values = {"pid": pid, "role": "proof" if pid == os.getpid() else "owned child"}
        try:
            for row in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if row.startswith(("Rss:", "Pss:", "Swap:")):
                    key, value = row.split(":", 1)
                    values[key + "_kib"] = int(value.split()[0])
        except (PermissionError, FileNotFoundError):
            values["unavailable"] = True
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            values["threads"] = int(
                next(line.split()[1] for line in status.splitlines() if line.startswith("Threads:"))
            )
        except (OSError, StopIteration, ValueError):
            values["threads"] = None
        result.append(values)
    return result


async def main():
    data = [
        await run_case(),
        await run_case(interrupt=True),
        await run_case(interrupt=True, background=True),
    ]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("var/results/phase2-native.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print("Native contract cases passed:", len(data))


if __name__ == "__main__":
    asyncio.run(main())
