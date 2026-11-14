"""Native speech contracts through actual project stdio MCP servers, mock inference."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.frame_processor import FrameProcessor
from me3_voice.native_speech import McpSTTService, McpTTSService, spoken_text
from me3_voice.speech import SpeechClient
from me3_voice.contracts import TurnKey, decode_pcm, encode_pcm


async def until(predicate, timeout=8):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class Collect(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


@asynccontextmanager
async def running(speech, batch_ms=32):
    events = []
    stt = McpSTTService(speech, batch_ms=batch_ms, on_event=events.append)
    sink = Collect()
    worker = PipelineWorker(
        Pipeline([stt, sink]),
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
        enable_rtvi=False,
        enable_turn_tracking=False,
        idle_timeout_secs=None,
    )
    ready = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def start(*_):
        ready.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(ready.wait(), 8)
        yield worker, stt, sink, events
    finally:
        await runner.cancel()
        await asyncio.wait_for(task, 8)
        assert stt.operation is None


async def feed(worker, events, chunks=1):
    before = sum(e["event"] == "asr_append" for e in events)
    await worker.queue_frame(VADUserStartedSpeakingFrame())
    for _ in range(chunks):
        await worker.queue_frame(InputAudioRawFrame(b"\1\0" * 512, 16000, 1))
    await until(lambda: sum(e["event"] == "asr_append" for e in events) > before)


@pytest.mark.parametrize("batch_ms", [32, 64, 96])
async def test_real_mcp_native_partials_final_tail_and_operation_reuse(batch_ms):
    async with SpeechClient(transcript="native speech", timeout=5) as speech:
        async with running(speech, batch_ms) as (worker, stt, sink, events):
            partials = []
            stt.on_partial = partials.append
            await feed(worker, events, chunks=4)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(isinstance(f, TranscriptionFrame) for f in sink.frames))
            finals = [f for f in sink.frames if isinstance(f, TranscriptionFrame)]
            assert [f.text for f in finals] == ["native speech"]
            assert finals[0].finalized
            assert any(isinstance(text, str) and text for text in partials)
            assert partials[-1] is None
            assert sum(e.get("bytes", 0) for e in events if e["event"] == "asr_append") == 4096
            assert not stt.pending
        assert (
            await speech.transcribe(TurnKey(session_id="reuse", turn_id=1), b"\1\0")
            == "native speech"
        )


async def test_vad_confirmation_replays_bounded_pre_onset_audio_in_order():
    class CapturePCM(SpeechClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.accepted_pcm = []

        async def call(self, kind, name, arguments):
            if name in {"asr_append", "asr_append_batch"}:
                self.accepted_pcm.append(decode_pcm(arguments["pcm_b64"]))
            return await super().call(kind, name, arguments)

    async with CapturePCM(transcript="onset restored", timeout=5) as speech:
        async with running(speech) as (worker, stt, sink, events):
            assert stt.onset_buffer_frames == 8
            for index in range(10):
                await worker.queue_frame(InputAudioRawFrame(bytes([index + 1]) * 1024, 16000, 1))
            await until(lambda: len(stt.onset_buffer) == 8)
            assert stt.operation is None and len(stt.onset_buffer) == 8

            await worker.queue_frame(VADUserStartedSpeakingFrame())
            await worker.queue_frame(InputAudioRawFrame(b"\x0b\0" * 512, 16000, 1))
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(isinstance(f, TranscriptionFrame) for f in sink.frames))

            expected = [bytes([i]) * 1024 for i in range(3, 11)] + [b"\x0b\0" * 512]
            assert b"".join(speech.accepted_pcm) == b"".join(expected)
            assert not stt.onset_buffer
            onset = next(e for e in events if e["event"] == "asr_onset_audio_submitted")
            assert onset["frames"] == 8 and onset["bytes"] == 8 * 1024
            assert not any(e["event"] == "asr_open_started" for e in events[: events.index(onset)])


async def test_native_trace_relays_only_validated_provider_connection_diagnostics():
    import time

    class DiagnosticSpeech(SpeechClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.connection_started = time.monotonic()

        def snapshot(self, state, accepted=0, sent=0, queued=0, **extra):
            return {
                "connection_state": state,
                "connection_attempt_monotonic": self.connection_started,
                "websocket_handshake_monotonic": self.connection_started + 0.001,
                "connection_ready_monotonic": self.connection_started + 0.02,
                "connection_failed_monotonic": None,
                "connection_closed_monotonic": None,
                "first_append_monotonic": self.connection_started + 0.021,
                "first_send_started_monotonic": self.connection_started + 0.022,
                "first_send_completed_monotonic": self.connection_started + 0.023,
                "accepted_audio_bytes": accepted,
                "sent_audio_bytes": sent,
                "queued_audio_bytes": queued,
                "peak_queued_audio_bytes": max(accepted - sent, 0),
                "discarded_audio_bytes": 0,
                "queue_limit_bytes": 64000,
                "failure_code": None,
                "connection_stages": {
                    "dns_started": self.connection_started + 0.001,
                    "connection_established": self.connection_started + 0.002,
                    "websocket_http_response_received": self.connection_started + 0.003,
                    "websocket_http_status": 101,
                    "connection_terminal_exception": "TimeoutError",
                    "request_outcome": "unknown",
                    "request_error_code": "provider_timeout",
                    "authorization_header": "must_not_be_forwarded",
                },
                **extra,
            }

        async def call(self, kind, name, arguments):
            row = await super().call(kind, name, arguments)
            if kind == "asr" and name == "asr_open":
                row["provider_diagnostics"] = self.snapshot("connecting")
            elif kind == "asr" and name in {"asr_append", "asr_append_batch"}:
                row["provider_diagnostics"] = self.snapshot("ready", accepted=1024, sent=1024)
            elif kind == "asr" and name == "cancel":
                row["provider_diagnostics"] = self.snapshot(
                    "closed",
                    accepted=1024,
                    sent=1024,
                    **{"connection_closed_monotonic": self.connection_started + 0.04},
                )
            return row

    async with DiagnosticSpeech(transcript="diagnostic turn", timeout=5) as speech:
        async with running(speech) as (worker, stt, sink, events):
            await feed(worker, events)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(isinstance(f, TranscriptionFrame) for f in sink.frames))
            await until(
                lambda: any(
                    e.get("event") == "asr_provider_transport"
                    and e.get("connection_state") == "closed"
                    for e in events
                )
            )
            snapshots = [e for e in events if e["event"] == "asr_provider_transport"]
            assert {row["connection_state"] for row in snapshots} >= {"connecting", "ready"}
            assert all(row["queue_limit_bytes"] == 64000 for row in snapshots)
            final_stages = snapshots[-1]["connection_stages"]
            assert (
                final_stages["dns_started"] == snapshots[-1]["connection_attempt_monotonic"] + 0.001
            )
            assert final_stages["websocket_http_status"] == 101
            assert final_stages["request_error_code"] == "provider_timeout"
            assert "authorization_header" not in final_stages
            assert not any("private" in repr(row) for row in snapshots)


async def test_interruption_discards_old_pre_onset_audio_before_next_vad_turn():
    class CapturePCM(SpeechClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.accepted_pcm = []

        async def call(self, kind, name, arguments):
            if name in {"asr_append", "asr_append_batch"}:
                self.accepted_pcm.append(decode_pcm(arguments["pcm_b64"]))
            return await super().call(kind, name, arguments)

    async with CapturePCM(transcript="fresh turn", timeout=5) as speech:
        async with running(speech) as (worker, stt, _sink, events):
            for _ in range(3):
                await worker.queue_frame(InputAudioRawFrame(b"\x01\0" * 512, 16000, 1))
            await until(lambda: len(stt.onset_buffer) == 3)
            assert stt.operation is None

            await worker.queue_frame(InterruptionFrame())
            await until(lambda: not stt.onset_buffer)
            assert stt.operation is None

            fresh = [bytes([i]) * 1024 for i in (2, 3)]
            for pcm in fresh:
                await worker.queue_frame(InputAudioRawFrame(pcm, 16000, 1))
            await until(lambda: len(stt.onset_buffer) == 2)
            await worker.queue_frame(VADUserStartedSpeakingFrame())
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(e["event"] == "asr_final" for e in events))
            assert b"".join(speech.accepted_pcm) == b"".join(fresh)
            assert not stt.onset_buffer
            assert sum(e["event"] == "asr_open_started" for e in events) == 1


async def test_delayed_append_is_decoupled_ordered_and_flushed_before_final():
    import time

    class DelayedAppend(SpeechClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.started = []
            self.completed = []

        async def call(self, kind, name, arguments):
            if name in {"asr_append", "asr_append_batch"}:
                self.started.append(arguments["sequence"])
                await asyncio.sleep(0.2)
                result = await super().call(kind, name, arguments)
                self.completed.append(arguments["sequence"])
                return result
            if name == "asr_finish":
                assert self.completed == list(range(len(self.started)))
            return await super().call(kind, name, arguments)

    async with DelayedAppend(transcript="ordered final", timeout=5) as speech:
        async with running(speech) as (worker, stt, sink, events):
            started = time.perf_counter()
            await worker.queue_frame(VADUserStartedSpeakingFrame())
            for _ in range(6):
                await worker.queue_frame(InputAudioRawFrame(b"\1\0" * 512, 16000, 1))
            accepted_seconds = time.perf_counter() - started
            assert accepted_seconds < 0.15
            await until(lambda: stt.append_queue is not None)
            assert stt.append_queue.qsize() >= 4
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(isinstance(f, TranscriptionFrame) for f in sink.frames))
            assert speech.started == list(range(len(speech.started)))
            assert speech.completed == speech.started
            assert [f.text for f in sink.frames if isinstance(f, TranscriptionFrame)] == [
                "ordered final"
            ]
            assert any(e["event"] == "asr_finish_requested" for e in events)
            assert events.index(
                next(e for e in events if e["event"] == "asr_final")
            ) > events.index(next(e for e in events if e["event"] == "asr_finish_requested"))


async def test_audio_frame_admission_does_not_wait_for_active_append_io():
    class DelayedAppend(SpeechClient):
        async def call(self, kind, name, arguments):
            if name in {"asr_append", "asr_append_batch"}:
                self.append_started.set()
                await asyncio.sleep(0.22)
            return await super().call(kind, name, arguments)

    async with DelayedAppend(timeout=5) as speech:
        speech.append_started = asyncio.Event()
        async with running(speech) as (worker, stt, _sink, events):
            await worker.queue_frame(VADUserStartedSpeakingFrame())
            await worker.queue_frame(InputAudioRawFrame(b"\1\0" * 512, 16000, 1))
            await asyncio.wait_for(speech.append_started.wait(), 2)
            queued_before = sum(event["event"] == "asr_append_queued" for event in events)
            started = asyncio.get_running_loop().time()
            await worker.queue_frame(InputAudioRawFrame(b"\2\0" * 512, 16000, 1))
            await until(
                lambda: (
                    sum(event["event"] == "asr_append_queued" for event in events) > queued_before
                )
            )
            admitted = asyncio.get_running_loop().time() - started
            assert admitted < 0.1
            assert stt.append_queue is not None and stt.append_queue.qsize() >= 1
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(
                lambda: any(isinstance(frame, TranscriptionFrame) for frame in _sink.frames)
            )


async def test_append_queue_saturation_fails_explicitly_without_unbounded_growth():
    class SlowAppend(SpeechClient):
        async def call(self, kind, name, arguments):
            if name in {"asr_append", "asr_append_batch"}:
                await asyncio.sleep(0.25)
            return await super().call(kind, name, arguments)

    async with SlowAppend(timeout=5) as speech:
        async with running(speech) as (worker, stt, sink, events):
            stt.append_queue_limit = 1
            await worker.queue_frame(VADUserStartedSpeakingFrame())
            for _ in range(12):
                await worker.queue_frame(InputAudioRawFrame(b"\1\0" * 512, 16000, 1))
            await until(lambda: any(e["event"] == "asr_append_queue_saturated" for e in events))
            assert stt.append_queue.maxsize == 1
            assert stt.append_queue.qsize() <= 1
            # Overload is reported at the VAD turn boundary. It must not
            # abort the frame processor before the real endpoint can arrive.
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(e["event"] == "asr_turn_aborted" for e in events))
            names = [e["event"] for e in events]
            assert names.index("vad_speech_stopped") < names.index("asr_turn_aborted")
            assert any(e.get("error_code") == "queue_saturated" for e in events)
            assert not [f for f in sink.frames if isinstance(f, TranscriptionFrame)]


async def test_delayed_real_asr_finish_is_cancellable_and_does_not_emit_old_final():
    async with SpeechClient(transcript=["obsolete", "current"], delay=0.25, timeout=5) as speech:
        async with running(speech) as (worker, stt, sink, events):
            await feed(worker, events)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(e["event"] == "asr_finish_requested" for e in events))
            assert stt.finalizer and not stt.finalizer.done()
            # Next onset is processed while the real server's finish is asleep.
            await feed(worker, events)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(isinstance(f, TranscriptionFrame) for f in sink.frames))
            assert [f.text for f in sink.frames if isinstance(f, TranscriptionFrame)] == ["current"]


async def test_already_returned_result_delivered_after_cancel_is_fenced():
    class DelayedDelivery(SpeechClient):
        async def call(self, kind, name, arguments):
            row = await super().call(kind, name, arguments)
            if name == "asr_finish" and not hasattr(self, "delayed_once"):
                self.delayed_once = True
                self.waiting.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return row  # Deliberate adversarial delivery, real MCP result.
            return row

    async with DelayedDelivery(transcript=["obsolete", "current"]) as speech:
        speech.waiting = asyncio.Event()
        async with running(speech) as (worker, stt, sink, events):
            await feed(worker, events)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await asyncio.wait_for(speech.waiting.wait(), 8)
            await worker.queue_frame(InterruptionFrame())
            await until(lambda: any(e["event"] == "asr_stale_rejected" for e in events))
            assert not [f for f in sink.frames if isinstance(f, TranscriptionFrame)]
            await feed(worker, events)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(isinstance(f, TranscriptionFrame) for f in sink.frames))
            assert [f.text for f in sink.frames if isinstance(f, TranscriptionFrame)] == ["current"]


async def test_tts_ignores_punctuation_only_stream_fragments():
    events = []
    tts = McpTTSService(object(), on_event=events.append)
    assert [frame async for frame in tts.run_tts(" … -- * ", "context-a")] == []
    assert tts.turn == 0
    assert [event["event"] for event in events] == ["tts_fragment_skipped"]


async def test_spoken_tts_removes_markdown_without_changing_display_context():
    class Speech:
        last_tts_diagnostics = None

        def __init__(self):
            self.text = []

        def synthesize(self, _key, text):
            self.text.append(text)

            async def stream():
                yield b"\0\0" * 160

            return stream()

    source = "The lamp is now **red** with 30 % brightness.\n```python\nprint('hi')\n```"
    assert spoken_text(source) == (
        "The lamp is now red with 30 percent brightness. Code is shown on screen."
    )
    speech, events = Speech(), []
    tts = McpTTSService(speech, on_event=events.append)
    await tts.on_turn_context_created("response-context")
    frames = [frame async for frame in tts.run_tts(source, "response-context")]
    assert speech.text == [spoken_text(source)]
    assert all("**" not in text and "```" not in text for text in speech.text)
    assert all(frame.context_id == "response-context" for frame in frames)
    assert events[0]["response_id"] == "response-context"


async def test_tts_stream_cancellation_and_large_text_reuse_real_mcp():
    async with SpeechClient(tts_delay=0.01) as speech:
        events = []
        tts = McpTTSService(speech, on_event=events.append)
        stream = tts.run_tts("cancel this", "context-a")
        first = await anext(stream)
        assert isinstance(first, TTSAudioRawFrame) and first.context_id == "context-a"
        await stream.aclose()
        chunks = [f async for f in tts.run_tts("word " * 160, "context-b")]
        pcm = [f for f in chunks if isinstance(f, TTSAudioRawFrame)]
        assert len(pcm) == 8 and all(f.context_id == "context-b" for f in pcm)
        assert not any(isinstance(frame, TTSStoppedFrame) for frame in chunks)
        assert tts.turn == 3
        summaries = [event for event in events if event["event"] == "tts_segment_summary"]
        assert [event["status"] for event in summaries] == [
            "cancelled",
            "completed",
            "completed",
        ]
        assert not any(event["event"] == "tts_response_stop_queued" for event in events)


async def test_v2_batch_validation_preserves_legacy_chunk_limit():
    async with SpeechClient() as speech:
        opened = await speech.open_operation(
            "asr", "asr_open", TurnKey(session_id="batch", turn_id=1).model_dump()
        )
        identifier = opened["operation_id"]
        with pytest.raises(Exception):
            await speech.call(
                "asr",
                "asr_append",
                {"operation_id": identifier, "sequence": 0, "pcm_b64": encode_pcm(b"\1\0" * 1024)},
            )
        result = await speech.call(
            "asr",
            "asr_append_batch",
            {
                "operation_id": identifier,
                "sequence": 0,
                "pcm_b64": encode_pcm(b"\1\0" * 1536),
                "contract_version": 2,
            },
        )
        assert result["next_sequence"] == 1 and result["contract_version"] == 2
        with pytest.raises(Exception):
            await speech.call(
                "asr",
                "asr_append_batch",
                {"operation_id": identifier, "sequence": 1, "pcm_b64": encode_pcm(b"\1\0" * 1537)},
            )
        await speech.release("asr", identifier)


async def test_event_read_and_tts_do_not_wait_for_asr_finalization():
    async with SpeechClient(delay=0.5, tts_delay=0.01, timeout=5) as speech:
        identifier = (
            await speech.open_operation(
                "asr", "asr_open", TurnKey(session_id="duplex", turn_id=1).model_dump()
            )
        )["operation_id"]
        await speech.call(
            "asr",
            "asr_append",
            {"operation_id": identifier, "sequence": 0, "pcm_b64": encode_pcm(b"\1\0" * 512)},
        )
        finish = asyncio.create_task(speech.call("asr", "asr_finish", {"operation_id": identifier}))
        await asyncio.sleep(0.03)
        # Event observation stays nonblocking on the SAME MCP connection.
        row = await asyncio.wait_for(
            speech.call("asr", "asr_events", {"operation_id": identifier}), 0.3
        )
        assert row["partial"] is None
        stream = speech.synthesize(TurnKey(session_id="duplex", turn_id=2), "Independent output.")
        assert await asyncio.wait_for(anext(stream), 0.3)
        assert not finish.done()
        await stream.aclose()
        await speech.release("asr", identifier)
        with pytest.raises(Exception):
            await finish


async def test_native_release_timeout_is_bounded_and_observable():
    class Unresponsive:
        async def release(self, *_):
            await asyncio.sleep(10)

    events = []
    stt = McpSTTService(Unresponsive(), on_event=events.append)
    await asyncio.wait_for(stt.release("fixture"), 0.5)
    assert [e["event"] for e in events] == ["asr_release_timeout"]


async def test_new_onset_is_forwarded_before_old_resource_cleanup():
    class SlowRelease(SpeechClient):
        async def release(self, kind, identifier):
            if self.hold:
                self.entered.set()
                await self.proceed.wait()
            await super().release(kind, identifier)

    async with SlowRelease(delay=0.5) as speech:
        speech.hold = False
        speech.entered = asyncio.Event()
        speech.proceed = asyncio.Event()
        async with running(speech) as (worker, stt, sink, events):
            await feed(worker, events)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(e["event"] == "asr_finish_requested" for e in events))
            speech.hold = True
            await worker.queue_frame(VADUserStartedSpeakingFrame())
            await asyncio.wait_for(speech.entered.wait(), 0.5)
            # Remote cleanup is blocked, but native turn control was forwarded.
            assert len([f for f in sink.frames if isinstance(f, VADUserStartedSpeakingFrame)]) == 2
            speech.hold = False
            speech.proceed.set()


async def test_delayed_asr_cancel_cannot_revive_native_context_or_model_request():
    from me3_voice.native_pipeline import make_native_pipeline
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.aggregators.llm_context import LLMContext

    class Probe(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.requests = []

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame):
                self.requests.append(
                    [m["content"] for m in frame.context.get_messages() if m["role"] == "user"]
                )
            else:
                await self.push_frame(frame, direction)

    async with SpeechClient(transcript=["obsolete input", "current input"], delay=0.3) as speech:
        events = []
        context = LLMContext()
        probe = Probe()
        parts = make_native_pipeline(
            speech, probe, Collect(), context=context, silence_seconds=0.05, on_event=events.append
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
        async def start(*_):
            ready.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        task = asyncio.create_task(runner.run())
        try:
            await asyncio.wait_for(ready.wait(), 8)
            await feed(worker, events)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: any(e["event"] == "asr_finish_requested" for e in events))
            assert not probe.requests
            await feed(worker, events)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await until(lambda: bool(probe.requests))
            await asyncio.sleep(0.4)
            assert probe.requests == [["current input"]]
            assert [m["content"] for m in context.get_messages() if m["role"] == "user"] == [
                "current input"
            ]
        finally:
            await runner.cancel()
            await asyncio.wait_for(task, 8)
