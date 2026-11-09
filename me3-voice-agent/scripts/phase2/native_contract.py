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
    TranscriptionFrame,
    InterimTranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    InterruptionFrame,
    LLMTextFrame,
    ErrorFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from me3_voice.contracts import TurnKey, encode_pcm
from me3_voice.speech import SpeechClient


class McpSTT(STTService):
    """Native adapter: append while speaking, finalize on fixture VAD stop.

    VAD frames are injected as annotated fixture events; not acoustic inference.
    This proves current 32ms contract only, not the proposed batched/events API.
    """

    def __init__(self, speech, events):
        super().__init__(
            sample_rate=16000, settings=STTSettings(model="project-mock", language=None)
        )
        self.speech, self.events = speech, events
        self.operation = None
        self.turn = 0
        self.appended = asyncio.Event()

    async def run_stt(self, audio):
        if self.operation is None:
            self.turn += 1
            self.key = TurnKey(session_id="native-proof", turn_id=self.turn)
            opened = await self.speech.open_operation("asr", "asr_open", self.key.model_dump())
            self.operation, self.sequence = opened["operation_id"], 0
        result = await self.speech.call(
            "asr",
            "asr_append",
            {
                "operation_id": self.operation,
                "sequence": self.sequence,
                "pcm_b64": encode_pcm(audio),
            },
        )
        self.speech.check_key(self.key, result)
        self.sequence += 1
        if result.get("partial"):
            self.events.append({"event": "asr_partial", "at": time.monotonic()})
            yield InterimTranscriptionFrame(result["partial"]["text"], "user", "offline")
        self.appended.set()

    async def process_frame(self, frame, direction):
        # Operation cancellation is resource cleanup, not another turn controller.
        if isinstance(frame, VADUserStoppedSpeakingFrame) and self.operation:
            identifier, self.operation = self.operation, None
            try:
                result = await self.speech.call("asr", "asr_finish", {"operation_id": identifier})
                self.speech.check_key(self.key, result)
                self.events.append(
                    {"event": "asr_final", "text": result["text"], "at": time.monotonic()}
                )
                await self.push_frame(
                    TranscriptionFrame(result["text"], "user", "offline", finalized=True)
                )
            finally:
                await self.speech.release("asr", identifier)
        await super().process_frame(frame, direction)

    async def cleanup(self):
        if self.operation:
            await self.speech.release("asr", self.operation)
            self.operation = None
        await super().cleanup()


class McpTTS(TTSService):
    def __init__(self, speech, events):
        super().__init__(
            sample_rate=16000, settings=TTSSettings(model="project-mock", voice=None, language=None)
        )
        self.speech, self.events, self.operation = speech, events, 0

    async def run_tts(self, text, context_id):
        self.operation += 1
        key = TurnKey(session_id="native-proof", turn_id=self.operation)
        self.events.append({"event": "tts_start", "text": text, "at": time.monotonic()})
        yield TTSStartedFrame()
        stream = self.speech.synthesize(key, text)
        try:
            async for pcm in stream:
                yield TTSAudioRawFrame(pcm, 16000, 1)
        finally:
            await stream.aclose()
            self.events.append({"event": "tts_released", "at": time.monotonic()})
        yield TTSStoppedFrame()


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


async def run_case(interrupt=False, background=False):
    events = []
    cpu_started = time.process_time()
    tool_calls = []
    background_release = asyncio.Event()
    handler_cancelled = []
    lag = []

    async def heartbeat():
        while True:
            expected = time.monotonic() + 0.01
            await asyncio.sleep(0.01)
            lag.append(max(0, time.monotonic() - expected) * 1000)

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
        pair = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[VADUserTurnStartStrategy()],
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.05)],
                )
            ),
        )
        stt = McpSTT(speech, events)
        llm = OfflineGroq(events, background=background)
        tts = McpTTS(speech, events)
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
        worker = PipelineWorker(
            Pipeline([stt, pair.user(), llm, tts, rec, output, pair.assistant()]),
            params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
            enable_rtvi=False,
            enable_turn_tracking=False,
            idle_timeout_secs=None,
        )
        ready = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def started(*_):
            ready.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        task = asyncio.create_task(runner.run())

        async def utterance():
            stt.appended.clear()
            await worker.queue_frame(VADUserStartedSpeakingFrame())
            await worker.queue_frame(InputAudioRawFrame(b"\x01\0" * 512, 16000, 1))
            await asyncio.wait_for(stt.appended.wait(), 8)
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
                "event_loop_lag_ms": {
                    "n": len(lag),
                    "p50": sorted(lag)[len(lag) // 2] if lag else None,
                    "p95": sorted(lag)[min(len(lag) - 1, int(len(lag) * 0.95))] if lag else None,
                    "max": max(lag) if lag else None,
                },
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
