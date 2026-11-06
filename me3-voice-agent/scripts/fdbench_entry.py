"""ME3 direct-feed adapter for the course full-duplex benchmark.

This explicit benchmark-only entry injects a live Session activation at the
start of the socket run because benchmark clips intentionally omit "Hey Swift".
It keeps ME3 ASR/TTS MCP, Session, Onit, task validation, and cancellation in
the media path; it does not open a microphone or speaker.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

PROJECT_MARKER = "src/me3_voice/voice_bridge.py"
SAMPLE_RATE = 16000
ROOT = Path(__file__).resolve().parents[1]


def audio_rates() -> tuple[int, int]:
    return SAMPLE_RATE, SAMPLE_RATE


def selected_backends():
    from me3_voice.configuration import load_backends

    if os.environ.get("ME3_FDBENCH_MOCK") == "1":
        return load_backends(None)
    return load_backends(
        Path(os.environ.get("ME3_BACKEND_CONFIG", ROOT / "var/cache/voice-backends.json"))
    )


def describe():
    specs = selected_backends()
    return {
        **{
            role.upper(): spec.model if spec.backend == "external" else "mock"
            for role, spec in specs.items()
        },
        "VAD": "Silero (Pipecat)",
        "Harness": "ME3 Session / mock reasoning"
        if specs["llm"].backend == "mock"
        else "Onit / ME3 Session",
    }


def report_stage(row, mark, *, messages=()):
    stages = {
        "endpoint_submitted": "vad_end",
        "asr_final": "asr_final",
        "asr_partial": "asr_partial",
        "llm_request_started": "llm_request",
        "tts_started": "tts_request",
        "tts_first_chunk": "tts_first_audio",
        "tool_started": "tool_start",
        "tool_completed": "tool_end",
    }
    stage = stages.get(row.get("event"))
    if stage:
        fields = {"t": row["monotonic"], "session_id": row["session_id"], "turn_id": row["turn_id"]}
        if stage == "asr_final" and os.environ.get("ME3_FDBENCH_EXPORT_TEXT") == "1":
            message = next(
                (
                    m
                    for m in reversed(messages)
                    if m["turn_id"] == row["turn_id"]
                    and m["role"] == "user"
                    and m["status"] == "final"
                ),
                None,
            )
            if message and len(message["text"]) <= 500:
                fields["text"] = message["text"]
        mark(stage, **fields)
    # First speakable segment is NOT necessarily first model token. No false TTFT.


class BenchmarkSink:
    """Send ME3's 16 kHz PCM through fdbench and await its playout-time proxy."""

    delivery = "benchmark_transport_acknowledged"

    def __init__(self, output, mark=lambda *args, **kwargs: None):
        self.mark = mark
        self.output = output
        self.playout_until = time.monotonic()
        self.stopped = False
        self.closed = False

    async def __call__(self, pcm: bytes):
        from pipecat.frames.frames import OutputAudioRawFrame

        if self.closed:
            return
        self.stopped = False
        duration = len(pcm) / (SAMPLE_RATE * 2)
        self.playout_until = max(self.playout_until, time.monotonic()) + duration
        completed = asyncio.get_running_loop().create_future()

        async def on_sent(_processor, _frame, _direction):
            if not completed.done():
                completed.set_result(None)

        await self.output.queue_frame(
            OutputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1),
            callback=on_sent,
        )
        await completed
        self.mark("audio_out")

    async def flush(self):
        remaining = self.playout_until - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def stop(self):
        from pipecat.frames.frames import InterruptionFrame

        self.stopped = True
        self.playout_until = time.monotonic()
        completed = asyncio.get_running_loop().create_future()

        async def on_cleared(_processor, _frame, _direction):
            if not completed.done():
                completed.set_result(None)

        await self.output.queue_frame(InterruptionFrame(), callback=on_cleared)
        await completed

    async def close(self):
        self.closed = True
        self.stopped = True


class ME3InputProcessor:
    """Lazy import factory is used below to keep module checks inexpensive."""


async def build(transport, log_dir: Path):
    stack = AsyncExitStack()
    try:
        return await _build(transport, log_dir, stack)
    except BaseException:
        await stack.aclose()
        raise


async def _build(transport, log_dir: Path, stack):
    from pipecat.frames.frames import InputAudioRawFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import WorkerRunner
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    from me3_voice.audio import SileroVAD
    from me3_voice.credentials import load_secrets
    from me3_voice.onit_tasks import OnitPrimaryReasoning, OnitTaskExecutor
    from me3_voice.remote import LiveTransport
    from me3_voice.runtime import AudioApplication
    from me3_voice.providers import MockLLM
    from me3_voice.tools import MockTaskExecutor
    from me3_voice.session import Session
    from me3_voice.speech import SpeechClient
    from me3_voice.transcript import Transcript

    log_dir.mkdir(parents=True, exist_ok=True)
    startup_log = log_dir / "fdbench-startup.jsonl"

    def startup_stage(name):
        with startup_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": name, "monotonic": time.monotonic()}) + "\n")

    activation = os.environ.get("ME3_FDBENCH_ACTIVATION", "active")
    if activation not in {"active", "inactive"}:
        raise ValueError("benchmark activation must be active or inactive")
    startup_stage("entry_build_started")
    secret_path = Path(os.environ.get("ME3_SECRETS_FILE", ROOT / "var/cache/voice-backends.env"))
    llm_secret_path = Path(os.environ.get("ME3_LLM_SECRETS_FILE", secret_path))
    backends = selected_backends()
    credentials = (
        {}
        if os.environ.get("ME3_FDBENCH_MOCK") == "1"
        else load_secrets(
            secret_path,
            {
                spec.auth_env
                for role, spec in backends.items()
                if role != "llm" and spec.backend == "external"
            },
        )
    )
    llm = backends["llm"]
    if llm.backend == "external":
        credentials.update(load_secrets(llm_secret_path, {llm.auth_env}))
    startup_stage("configuration_loaded")
    for spec in backends.values():
        if spec.backend == "external":
            LiveTransport(spec, credentials).authorize()
    startup_stage("providers_authorized")

    speech = SpeechClient(
        backends=backends,
        credentials=credentials,
        timeout=max(spec.timeout_seconds for spec in backends.values()),
    )
    await stack.enter_async_context(speech)
    startup_stage("speech_mcp_ready")
    if backends["llm"].backend == "mock":
        task_executor = MockTaskExecutor()
        stack.push_async_callback(task_executor.close)
        reasoning = MockLLM()
    else:
        task_executor = OnitTaskExecutor(
            Path(os.environ.get("ME3_ONIT_PYTHON", str(ROOT / "var/cache/onit-venv/bin/python"))),
            backends["llm"],
            credentials,
            root=ROOT / "var/scratch/fdbench-onit-tasks",
            max_requests=int(os.environ["ME3_FDBENCH_REQUEST_LIMIT"])
            if os.environ.get("ME3_FDBENCH_REQUEST_LIMIT")
            else None,
        )
        stack.push_async_callback(task_executor.close)
        task_executor.prewarm()
        await task_executor.startup
        startup_stage("onit_worker_ready")
        reasoning = OnitPrimaryReasoning()
    transcript = Transcript(
        mode="mock" if os.environ.get("ME3_FDBENCH_MOCK") == "1" else "HPC-live",
        backend_modes={role: spec.backend for role, spec in backends.items()},
        playback="benchmark",
    )
    startup_stage("transcript_ready")
    output = transport.output()
    startup_stage("transport_output_ready")
    try:
        from fdbench_marks import mark
    except ImportError:

        def mark(*args, **kwargs):
            pass

    sink = BenchmarkSink(output, mark)
    session = Session(
        speech,
        reasoning,
        sink,
        tasks=task_executor,
        close_tasks=False,
        idle_seconds=120,
        turn_timeout=None,
        llm_timeout=None,
        conversation_mode="extended",
        extended_seconds=120,
        transcript=transcript,
        execution_mode="mock" if os.environ.get("ME3_FDBENCH_MOCK") == "1" else "HPC-live",
    )
    session.on_event = lambda row: report_stage(row, mark, messages=transcript.messages)
    startup_stage("session_constructed")
    await stack.enter_async_context(session)
    startup_stage("session_ready")
    startup_stage("vad_initialization_started")
    app = AudioApplication(
        session,
        wake=None,
        vad=SileroVAD(),
        rate=SAMPLE_RATE,
        channels=1,
        wake_frame=None,
        silence_frames=32,
    )
    startup_stage("audio_application_ready")

    class Input(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.frames = 0

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, InputAudioRawFrame):
                if frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1:
                    raise ValueError("fdbench input must remain 16 kHz mono")
                if self.frames == 0 and activation == "active":
                    # Course clips intentionally omit a wake word. Begin one
                    # isolated benchmark session before forwarding its first
                    # frame; normal browser/Pi activation is untouched.
                    app.input.activate("benchmark_direct_feed_activation")
                await app.feed(frame.audio, app.pre.sequence)
                self.frames += 1
                return
            await self.push_frame(frame, direction)

    processor = Input()
    worker = PipelineWorker(
        Pipeline([transport.input(), processor, output]),
        params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE),
        enable_rtvi=False,
        enable_turn_tracking=False,
        idle_timeout_secs=None,
    )
    runner = WorkerRunner(handle_sigint=False)
    startup_stage("pipeline_constructed")
    cleaned = False

    def write_summary(stage):
        # Payload-free evidence only: never persist transcript, audio, prompts,
        # credentials, tool arguments, or model reasoning.
        (log_dir / "me3-benchmark-summary.json").write_text(
            json.dumps(
                {
                    "evidence": "configured providers; 16 kHz mono direct feed; no microphone/speaker",
                    "providers": describe(),
                    "pi_hardware_verified": False,
                    "mock_providers": os.environ.get("ME3_FDBENCH_MOCK") == "1",
                    "stage": stage,
                    "activation": activation,
                    "input_frames": processor.frames,
                    "session_state": session.state.value,
                    "session_turn": session.turn,
                    "session_event_names": [row.get("event") for row in session.events],
                    "task_count": len(getattr(task_executor, "operations", {})),
                    "playback_queue_size": session.playback.queue.qsize(),
                    "playout_completion": "queued PCM-duration software proxy; not acoustic playback acknowledgement",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @transport.event_handler("on_connected")
    async def on_benchmark_connected(_transport, _websocket):
        startup_stage("websocket_connected")

    @transport.event_handler("on_disconnected")
    async def on_benchmark_disconnected(_transport, _websocket):
        write_summary("transport_disconnected")

    async def cleanup():
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        try:
            write_summary("cleanup_started")
        finally:
            await stack.aclose()
        write_summary("cleanup_complete")

    async def run():
        startup_stage("pipeline_run_started")
        try:
            await runner.add_workers(worker)
            startup_stage("pipeline_worker_added")
            await runner.run()
        finally:
            await cleanup()

    async def cancel():
        try:
            await worker.cancel()
        finally:
            await cleanup()

    return run(), cancel
