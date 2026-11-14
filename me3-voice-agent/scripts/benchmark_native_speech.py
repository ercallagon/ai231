"""Offline matched Session/native speech benchmark; real MCP, fixture inference.

No providers, hardware or model assets are contacted. Optional WAV is replayed
with mock transcription, not used for accuracy claims. Outputs stay under var/.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import wave
from datetime import datetime, timezone

from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.workers.runner import WorkerRunner
from pipecat.services.groq.llm import GroqLLMService

from me3_voice.contracts import FRAME_BYTES
from me3_voice.speech import SpeechClient
from me3_voice.native_pipeline import make_native_pipeline
from me3_voice.session import Session

sys.path.insert(0, str(Path(__file__).parent / "phase2"))
from native_contract import MemoryOutput, distribution, resources, until

REPLY = "A short reply."


class TracedSpeech(SpeechClient):
    def __init__(self):
        super().__init__(transcript="Benchmark input.", delay=0.003, tts_delay=0.003, timeout=10)
        self.calls = []

    async def call(self, kind, name, arguments):
        start = time.monotonic()
        result = await super().call(kind, name, arguments)
        self.calls.append(
            {
                "kind": kind,
                "name": name,
                "start": start,
                "end": time.monotonic(),
                "payload_json_bytes": len(json.dumps(arguments)),
                "done": result.get("done"),
                "has_partial": isinstance(result.get("partial"), dict),
            }
        )
        return result


class FixtureModel:
    def __init__(self):
        self.timings = []

    async def respond(self, _history):
        start = time.monotonic()
        await asyncio.sleep(0.003)
        self.timings.append((start, time.monotonic()))
        yield REPLY


class FixtureNative(GroqLLMService):
    def __init__(self):
        super().__init__(api_key="offline-not-a-provider-key")
        self.timings = []

    async def get_chat_completions(self, _context):
        start = time.monotonic()

        async def chunks():
            await asyncio.sleep(0.003)
            self.timings.append((start, time.monotonic()))
            for delta, finish in [({"content": REPLY}, None), ({}, "stop")]:
                yield ChatCompletionChunk(
                    id="offline",
                    object="chat.completion.chunk",
                    created=0,
                    model="fixture",
                    choices=[{"index": 0, "delta": delta, "finish_reason": finish}],
                )

        return chunks()


class PacedSink:
    def __init__(self):
        self.writes = []

    async def __call__(self, pcm):
        self.writes.append((time.monotonic(), len(pcm)))
        await asyncio.sleep(len(pcm) / 32000)

    async def stop(self):
        pass

    async def close(self):
        pass


def row(calls, endpoint, first_write, llm_timing):
    final = next(c["end"] for c in calls if c["name"] == "asr_finish")
    first = next(c for c in calls if c["name"] == "tts_next" and c["done"] is False)
    opened = next(c["start"] for c in calls if c["name"] == "tts_open")
    return {
        "speech_end_to_asr_final_ms": (final - endpoint) * 1000,
        "mock_llm_first_token_ms": (llm_timing[1] - llm_timing[0]) * 1000,
        "asr_first_partial_ipc_ms": next(
            (c["end"] - c["start"]) * 1000 for c in calls if c["has_partial"]
        ),
        "speech_end_to_first_memory_write_ms": (first_write - endpoint) * 1000,
        "tts_open_to_first_pcm_ms": (first["end"] - opened) * 1000,
        "final_to_tts_open_ms": (opened - final) * 1000,
        "tool_round_trips": len(calls),
        "ipc_ms": [(c["end"] - c["start"]) * 1000 for c in calls],
        "append_ipc_ms": [
            (c["end"] - c["start"]) * 1000
            for c in calls
            if c["name"] in ("asr_append", "asr_append_batch")
        ],
        "request_json_bytes": sum(c["payload_json_bytes"] for c in calls),
    }


async def compare_profile(kind, frames, samples, paced):
    lag = []
    queue_peaks = {"legacy_input": 0, "legacy_playback": 0, "native_pending_bytes": 0}
    startup = time.monotonic()
    cpu_start = time.process_time()
    ready_at = None
    source = None

    async def monitor():
        while True:
            expected = time.monotonic() + 0.01
            await asyncio.sleep(0.01)
            lag.append((time.monotonic(), max(0, time.monotonic() - expected) * 1000))
            if isinstance(source, Session):
                queue_peaks["legacy_input"] = max(
                    queue_peaks["legacy_input"],
                    getattr(source, "input_queue", asyncio.Queue()).qsize(),
                )
                queue_peaks["legacy_playback"] = max(
                    queue_peaks["legacy_playback"], source.playback.queue.qsize()
                )
            elif source is not None:
                queue_peaks["native_pending_bytes"] = max(
                    queue_peaks["native_pending_bytes"], len(source.stt.pending)
                )
                # Diagnostic introspection scoped to the installed Pipecat1.13
                # API; does not add queues or affect production control.
                for processor in source.pipeline.processors:
                    for field in ("_FrameProcessor__input_queue", "_FrameProcessor__process_queue"):
                        queue = getattr(processor, field, None)
                        if queue is not None:
                            name = processor.name + "." + field.rsplit("__", 1)[-1]
                            queue_peaks[name] = max(queue_peaks.get(name, 0), queue.qsize())

    monitoring = asyncio.create_task(monitor())
    results = []
    try:
        async with TracedSpeech() as speech:
            if kind == "session":
                output = PacedSink()
                model = FixtureModel()
                async with Session(
                    speech, model, output, llm_timeout=10, turn_timeout=15
                ) as session:
                    source = session
                    session.activate()
                    ready_at = time.monotonic()
                    for index in range(samples + 1):
                        speech.calls.clear()
                        before = len(output.writes)
                        await session.begin_input()
                        for pcm in frames:
                            await session.push_input(pcm)
                            if paced:
                                await asyncio.sleep(len(pcm) / 32000)
                        endpoint = time.monotonic()
                        await (await session.commit_input())
                        assert session.history[-1]["content"] == REPLY
                        assert len(output.writes) > before
                        if index:
                            results.append(
                                row(
                                    list(speech.calls),
                                    endpoint,
                                    output.writes[before][0],
                                    model.timings[-1],
                                )
                            )
            else:
                events = []
                llm = FixtureNative()
                output = MemoryOutput(events)
                parts = make_native_pipeline(
                    speech,
                    llm,
                    output,
                    batch_ms=int(kind),
                    silence_seconds=0.05,
                    on_event=events.append,
                )
                source = parts
                worker = PipelineWorker(
                    parts.pipeline,
                    params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
                    enable_rtvi=False,
                    enable_turn_tracking=False,
                    idle_timeout_secs=None,
                )
                ready, done = asyncio.Event(), asyncio.Event()

                @worker.event_handler("on_pipeline_started")
                async def start(*_):
                    ready.set()

                @parts.aggregators.assistant().event_handler("on_assistant_turn_stopped")
                async def stopped(*_):
                    done.set()

                runner = WorkerRunner(handle_sigint=False)
                await runner.add_workers(worker)
                task = asyncio.create_task(runner.run())
                try:
                    await asyncio.wait_for(ready.wait(), 10)
                    ready_at = time.monotonic()
                    for index in range(samples + 1):
                        speech.calls.clear()
                        done.clear()
                        before = len(output.writes)
                        appended = sum(e["event"] == "asr_append" for e in events)
                        await worker.queue_frame(VADUserStartedSpeakingFrame())
                        for pcm in frames:
                            await worker.queue_frame(InputAudioRawFrame(pcm, 16000, 1))
                            if paced:
                                await asyncio.sleep(len(pcm) / 32000)
                        await until(
                            lambda: sum(e["event"] == "asr_append" for e in events) > appended
                        )
                        endpoint = time.monotonic()
                        await worker.queue_frame(VADUserStoppedSpeakingFrame())
                        await asyncio.wait_for(done.wait(), 10)
                        assert (
                            parts.aggregators.assistant().context.get_messages()[-1]["content"]
                            == REPLY
                        )
                        assert len(output.writes) > before
                        if index:
                            results.append(
                                row(
                                    list(speech.calls),
                                    endpoint,
                                    output.writes[before][0],
                                    llm.timings[-1],
                                )
                            )
                finally:
                    await runner.cancel()
                    await asyncio.wait_for(task, 10)
                    await llm._client.close()
            snapshot = resources()
    finally:
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
    return {
        "profile": kind,
        "samples": samples,
        "paced": paced,
        "warmup_turns_excluded": 1,
        "stages": {
            k: distribution([r[k] for r in results])
            for k in results[0]
            if not isinstance(results[0][k], list)
        },
        "ipc_ms": distribution([value for r in results for value in r["ipc_ms"]]),
        "append_ipc_ms": distribution([value for r in results for value in r["append_ipc_ms"]]),
        "loop_lag_startup_ms": distribution(
            [value for at, value in lag if ready_at is None or at < ready_at + 0.2]
        ),
        "loop_lag_steady_ms": distribution(
            [value for at, value in lag if ready_at is not None and at >= ready_at + 0.2]
        ),
        "queue_peaks_sampled_10ms": queue_peaks,
        "resource_snapshot": snapshot,
        "elapsed_seconds": time.monotonic() - startup,
        "main_cpu_seconds": time.process_time() - cpu_start,
        "raw_trials": results,
    }


def serialization_profile():
    from me3_voice.contracts import encode_pcm

    result = {}
    for size in (1024, 2048, 3072):
        values = []
        pcm = b"\1\0" * (size // 2)
        for _ in range(1000):
            start = time.perf_counter_ns()
            json.dumps({"operation_id": "a" * 32, "sequence": 1, "pcm_b64": encode_pcm(pcm)})
            values.append((time.perf_counter_ns() - start) / 1_000_000)
        result[str(size)] = distribution(values)
    return result


async def main(args):
    if args.wav:
        with wave.open(str(args.wav), "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("WAV must be mono PCM16/16k")
            audio = wav.readframes(args.frames * 512)
        if len(audio) < args.frames * FRAME_BYTES:
            raise ValueError("WAV too short")
    else:
        audio = b"\1\0" * (args.frames * 512)
    frames = [audio[i : i + FRAME_BYTES] for i in range(0, len(audio), FRAME_BYTES)]
    results = [
        await compare_profile(kind, frames, args.samples, args.paced)
        for kind in ["session", "32", "64", "96"]
    ]
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": "replay/mock upstream, actual MCP, paced memory output; no physical/inference claims",
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "dependencies": {
                k: importlib.metadata.version(k) for k in ["pipecat-ai", "mcp", "aiohttp", "soxr"]
            },
        },
        "fixture": {
            "wav": str(args.wav) if args.wav else None,
            "submitted_pcm_sha256": hashlib.sha256(audio).hexdigest(),
            "frames": len(frames),
            "audio_seconds": len(audio) / 32000,
            "mock_asr": "Benchmark input.",
            "mock_reply": REPLY,
        },
        "serialization_cpu_ms": serialization_profile(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("src/me3_voice/native_speech.py"),
                Path("src/me3_voice/native_pipeline.py"),
                Path("src/me3_voice/speech.py"),
                Path("src/me3_voice/servers.py"),
                Path("src/me3_voice/session.py"),
                Path("src/me3_voice/remote.py"),
                Path("scripts/phase2/native_contract.py"),
                Path("pyproject.toml"),
                Path("uv.lock"),
            ]
        },
        "other_me3_tests_during_run": False,
        "results": results,
        "limitations": [
            "Endpoint is explicit fixture submission, not acoustic speech end",
            "First memory write is not audible output",
            "Native and legacy output packet sizes differ",
            "No physical Pi, provider TTFT or accuracy measured",
            "Resource snapshots are not peak PSS; private framework queues are not fully inventoried",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(data, handle, indent=2)
    print(
        json.dumps(
            [
                {
                    "profile": r["profile"],
                    "samples": r["samples"],
                    "first_write_ms": r["stages"]["speech_end_to_first_memory_write_ms"],
                }
                for r in results
            ]
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--paced", action="store_true")
    p.add_argument("--wav", type=Path)
    a = p.parse_args()
    if (
        not a.output.resolve().is_relative_to(Path("var").resolve())
        or not 1 <= a.samples <= 1000
        or not 1 <= a.frames <= 90
    ):
        p.error("output under var/; samples1..1000; frames1..90")
    asyncio.run(main(a))
