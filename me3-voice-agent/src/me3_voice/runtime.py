"""Native file PCM -> Pipecat ingress -> frontend -> Session -> MCP -> memory sink.

The frontend is actual CPU inference when configured; speech/reasoning providers
are configurable. Playback remains an in-memory sink. No audio device is opened.
"""

import asyncio
import hashlib
import time
import wave
from contextlib import AsyncExitStack

from .audio import ConversationInput, ReplayInput, SileroVAD
from .configuration import load_backends
from .frontend import FrontendMonitor
from .harness import PipecatAudioInput
from .playback import MemorySink
from .preprocessing import PCMPreprocessor
from .remote import LiveTransport, make_provider
from .readiness import build_wake, inventory
from .session import Session, State
from .speech import SpeechClient
from .telemetry import summarize, write_trace


class AudioApplication:
    def __init__(
        self,
        session,
        wake,
        vad,
        rate=16000,
        channels=1,
        *,
        wake_frame=None,
        silence_frames=20,
        turn_analyzer=None,
        barge_min_voiced_frames=1,
        barge_window_frames=8,
    ):
        self.session, self.wake = session, wake
        self.pre = PCMPreprocessor(rate, channels)
        self.monitor = FrontendMonitor(wake, vad, session_policy=False)
        self.source = ReplayInput(vad, silence_frames=silence_frames)
        self.input = ConversationInput(
            self.source,
            session,
            turn_analyzer,
            barge_min_voiced_frames=barge_min_voiced_frames,
            barge_window_frames=barge_window_frames,
        )
        self.wake_frame = wake_frame
        self.last_activity_sample = 0
        self.utterance_count = 0
        self.session_ids = []
        self.initialized = False

    def sync_idle(self):
        if self.session.state == State.IDLE and self.source.active:
            self.source.reset()
            if self.input.turn_analyzer:
                self.input.turn_analyzer.clear()
            if self.wake:
                self.wake.end_session()

    async def normalized(self, pcm):
        # Bound controller scheduling quantum; do not duplicate model inference.
        for offset in range(0, len(pcm), 128):
            if self.session.conversation_mode and not self.source.has_speech:
                await self.session.tick()
            self.sync_idle()
            for signal in self.monitor.push(pcm[offset : offset + 128]):
                if signal["kind"] == "wake":
                    if signal["triggered"] and self.session.state == State.IDLE:
                        self.input.activate("wake_detected_file", seed_speech=True)
                        self.session_ids.append(self.session.session_id)
                        self.last_activity_sample = signal["samples"]
                else:
                    if self.wake_frame is not None and self.source.frames == self.wake_frame:
                        self.input.activate("wake_injected_file", seed_speech=True)
                        self.session_ids.append(self.session.session_id)
                        self.last_activity_sample = signal["samples"]
                        self.wake_frame = None
                    if self.session.state != State.IDLE and (
                        signal["voiced"] or self.session.state in {State.THINKING, State.SPEAKING}
                    ):
                        self.last_activity_sample = signal["samples"]
                    if await self.input.feed(signal["pcm"], signal["voiced"]):
                        self.utterance_count += 1
                    if (
                        not self.session.conversation_mode
                        and self.session.state == State.LISTENING
                        and not self.source.has_speech
                        and not self.session.input_open
                        and signal["samples"] - self.last_activity_sample
                        >= self.session.idle_seconds * 16000
                    ):
                        self.session.event("inactivity_file_timeout")
                        await self.session.exit()
                        self.sync_idle()
            await asyncio.sleep(0)

    async def feed(self, pcm, sequence):
        await self.normalized(self.pre.push(pcm, sequence))

    async def finish(self):
        await self.normalized(self.pre.finish())
        self.monitor.finish()
        unfinished = self.source.has_speech
        if unfinished:
            self.session.event("unfinished_input_file")
            await self.session.interrupt()
        elif self.session.generation:
            await self.session.generation
        return unfinished


async def replay_application(args):
    backends = load_backends(getattr(args, "backend_config", None))
    from .credentials import load_secrets

    credentials = (
        load_secrets(
            args.secrets_file,
            {spec.auth_env for spec in backends.values() if spec.backend == "external"},
        )
        if getattr(args, "secrets_file", None)
        else {}
    )
    if getattr(args, "llm_secrets_file", None):
        credentials.update(load_secrets(args.llm_secrets_file, {backends["llm"].auth_env}))
    external = any(spec.backend == "external" for spec in backends.values())
    for spec in backends.values():
        if spec.backend == "external":
            LiveTransport(
                spec, credentials
            ).authorize()  # Fail before model loading/subprocesses; no request.
    host = inventory()
    wake = build_wake(args)
    if wake and args.wake_frame is not None:
        raise ValueError("choose actual wake detection or an explicit injected wake frame")
    if args.processed_wav:
        raise ValueError(
            "application replay does not export audio; use readiness replay for processed WAV"
        )
    if args.output.resolve() == args.wav.resolve():
        raise ValueError("report cannot overwrite recording")
    if args.trace and (
        args.trace.exists() or args.trace.resolve() in {args.output.resolve(), args.wav.resolve()}
    ):
        raise ValueError("trace needs a fresh path separate from report/input")
    playback_wav = getattr(args, "playback_wav", None)
    if playback_wav and (
        playback_wav.exists()
        or playback_wav.resolve()
        in {args.output.resolve(), args.wav.resolve(), args.trace.resolve() if args.trace else None}
    ):
        raise ValueError("playback WAV needs a fresh path separate from report/trace/input")
    if args.transcript and args.output.resolve().name == "ui-transcript.json":
        raise ValueError("application report must be separate from UI transcript")
    sink = MemorySink()
    started = time.monotonic()
    async with AsyncExitStack() as stack:
        display = None
        if args.transcript:
            from .transcript import Transcript, TranscriptPublisher

            display = Transcript(
                mode="HPC-live" if external else "mock",
                backend_modes={role: spec.backend for role, spec in backends.items()},
            )
            await stack.enter_async_context(TranscriptPublisher(display))
        speech = await stack.enter_async_context(
            SpeechClient(
                transcript=args.mock_transcripts or "Mock question",
                backends=backends,
                credentials=credentials,
                timeout=max(spec.timeout_seconds for spec in backends.values()),
            )
        )
        onit_python = getattr(args, "onit_python", None)
        if onit_python:
            from .onit_tasks import OnitPrimaryReasoning, OnitTaskExecutor

            tasks = OnitTaskExecutor(
                onit_python,
                backends["llm"],
                credentials,
                max_requests=getattr(args, "onit_max_requests", None),
            )
            await tasks.prepare()
            reasoning = OnitPrimaryReasoning()
        else:
            tasks = None
            reasoning = make_provider("llm", backends["llm"], credentials=credentials)
            if hasattr(reasoning, "aclose"):
                stack.push_async_callback(reasoning.aclose)
        session = await stack.enter_async_context(
            Session(
                speech,
                reasoning,
                sink,
                transcript=display,
                idle_seconds=60,
                turn_timeout=None,
                llm_timeout=(backends["llm"].timeout_seconds * 8 + 15) if onit_python else None,
                execution_mode="HPC-live" if external else "mock",
                tasks=tasks,
                direct_clock_enabled=getattr(args, "direct_clock", False),
            )
        )
        if tasks is not None:
            tasks.on_event = session.event
        if hasattr(reasoning, "on_event"):
            reasoning.on_event = session.event
        with wave.open(str(args.wav), "rb") as wav:
            rate, channels = wav.getframerate(), wav.getnchannels()
            if (
                wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
                or not 0 < wav.getnframes() <= rate * 60
            ):
                raise ValueError("application replay requires PCM16 WAV, 0..60 seconds")
            turn_analyzer = None
            if getattr(args, "turn_policy", "silence") == "smart-turn":
                from pipecat.utils.env import env_truthy

                if env_truthy("PIPECAT_SMART_TURN_LOG_DATA", default=False):
                    raise ValueError(
                        "disable PIPECAT_SMART_TURN_LOG_DATA; raw model logging outside var/ is forbidden"
                    )
                from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
                    LocalSmartTurnAnalyzerV3,
                )

                turn_analyzer = LocalSmartTurnAnalyzerV3(sample_rate=16000, cpu_count=1)
                turn_analyzer.set_sample_rate(16000)
                stack.push_async_callback(turn_analyzer.cleanup)
            app = AudioApplication(
                session,
                wake,
                SileroVAD(args.vad_threshold),
                rate,
                channels,
                wake_frame=args.wake_frame if wake else (args.wake_frame or 0),
                silence_frames=args.endpoint_ms // 32,
                turn_analyzer=turn_analyzer,
            )
            async with PipecatAudioInput(app) as ingress:
                sequence = 0
                consumed_frames = 0
                source_epoch = time.monotonic()
                max_schedule_lag_ms = 0.0
                while pcm := wav.readframes(max(1, rate // 100)):
                    await ingress.feed(pcm, sequence)
                    sequence += 1
                    consumed_frames += len(pcm) // (channels * 2)
                    if args.pace:
                        target = source_epoch + args.pace * consumed_frames / rate
                        remaining = target - time.monotonic()
                        max_schedule_lag_ms = max(max_schedule_lag_ms, -remaining * 1000)
                        if remaining > 0:
                            await asyncio.sleep(remaining)
                unfinished = await ingress.finish()
        frontend_report = app.monitor.report()
        frontend_report["endpoint_event_count"] = app.source.event_count
        frontend_report["endpoint_events_truncated"] = app.source.event_count > len(
            app.source.events
        )
        task_backend = session.tasks
        mock_action_owner = getattr(task_backend, "actions", None) or getattr(
            getattr(task_backend, "mock", None), "actions", None
        )
        frontend_report["frontend_events"] = sorted(
            list(app.monitor.events)
            + [event for event in app.source.events if event["event"] in {"vad_onset", "endpoint"}],
            key=lambda event: event["audio_seconds"],
        )[-1000:]
        frontend_report["incomplete_utterance"] = unfinished
        report = {
            "kind": "application-replay",
            "schema_version": 1,
            "mode": "mixed",
            "run_id": session.run_id,
            "session_ids": app.session_ids,
            "audio_origin": args.audio_origin,
            "input_sha256": hashlib.sha256(args.wav.read_bytes()).hexdigest(),
            "hash_scope": "whole WAV file",
            "input_rate": rate,
            "input_channels": channels,
            "model_rate": 16000,
            "input_stats": app.pre.input_stats.report(),
            "output_stats": app.pre.output_stats.report(),
            "frontend_mode": "Pi-local CPU file inference"
            if host["pi_detected"]
            else "HPC-live CPU file inference",
            "agent_mode": "external-services" if external else "mock",
            "reasoning_harness": "Onit primary / DGX Qwen"
            if onit_python
            else "ME3 provider adapter",
            "onit_worker_startup_ms": tasks.startup_ms if tasks is not None else None,
            "input_pipeline": "pipecat-streaming-pcm",
            "input_pipeline_frames": ingress.processor.frames,
            "wake_backend": wake.detector.backend if wake else "injected",
            "wake_provenance": wake.detector.provenance if wake else None,
            "frontend": frontend_report,
            "utterances": app.utterance_count,
            "unfinished_input": unfinished,
            "sink_chunks": len(sink.chunks),
            "protocols": speech.protocols,
            "backends": {name: spec.backend for name, spec in backends.items()},
            "backend_contracts": {
                name: spec.protocol if spec.backend == "external" else "mock"
                for name, spec in backends.items()
            },
            "mock_effects": getattr(mock_action_owner, "effects", None),
            "elapsed_seconds": time.monotonic() - started,
            "pace": args.pace,
            "endpoint_silence_ms": args.endpoint_ms,
            "turn_policy": getattr(args, "turn_policy", "silence"),
            "max_replay_schedule_lag_ms": max_schedule_lag_ms if args.pace else None,
            "limitations": "Actual CPU frontend on supplied file; injected wake if no model supplied. Provider choices are explicit in backends; external-service success must be checked against turn outcomes. Tasks and playback remain in memory. No device/AEC/acoustic latency or Pi qualification. Pace 0 is scheduling stress, not conversational timing.",
        }
    if args.trace:
        write_trace(args.trace, session.events)
    if playback_wav:
        with playback_wav.open("xb") as stream:
            with wave.open(stream, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"".join(sink.chunks))
    report["diagnostics"] = summarize(session.events)
    return report
