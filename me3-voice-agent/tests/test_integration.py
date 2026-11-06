import asyncio
import json
from dataclasses import replace

import pytest

from me3_voice.configuration import BackendSpec, ConfigurationError, load_backends
from me3_voice.contracts import FRAME_BYTES, TurnKey
from me3_voice.harness import PipecatAudioInput
from me3_voice.playback import MemorySink
from me3_voice.providers import MockLLM, ToolCall
from me3_voice.runtime import AudioApplication
from me3_voice.session import Session, State
from me3_voice.speech import SpeechClient
from me3_voice.transcript import Transcript
from me3_voice.wake import WakeDetection, WakeGate


async def test_stdio_partial_before_input_end_and_cancel_reuse():
    release = asyncio.Event()

    async def chunks():
        yield b"\1\0" * 512
        await release.wait()
        yield b"\1\0" * 100

    async with SpeechClient(transcript="streamed final text") as speech:
        stream = speech.transcribe_stream(TurnKey(session_id="stream", turn_id=1), chunks())
        partial = await anext(stream)
        assert partial.text == "streamed" and not partial.final
        await stream.aclose()
        assert (
            await speech.transcribe(TurnKey(session_id="stream", turn_id=2), b"\1\0")
            == "streamed final text"
        )
        stream = speech.transcribe_stream(TurnKey(session_id="stream", turn_id=3), chunks())
        await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.01)
        assert not pending.done()
        release.set()
        result = await pending
        assert result.final and result.text == "streamed final text"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)


class Speech:
    def __init__(self):
        self.inputs = []

    async def transcribe(self, key, pcm):
        self.inputs.append(pcm)
        return "hello"

    async def synthesize(self, key, text):
        yield b"\2\0" * 512


class Detector:
    frame_samples = 320
    backend = "mock-keyword-events"
    provenance = {}

    def __init__(self):
        self.samples = 0

    def detect(self, pcm):
        self.samples += 320
        return WakeDetection("HEY_SWIFT") if self.samples in (640, 7680) else None


class MarkedVAD:
    def voiced(self, pcm):
        return any(pcm)


@pytest.mark.parametrize("pipeline", [False, True], ids=["direct", "pipecat"])
async def test_connected_wake_preroll_followup_input_timeout_and_rewake(pipeline):
    speech = Speech()
    voiced, quiet = b"\1\0" * 512, bytes(FRAME_BYTES)
    sink = MemorySink()
    async with Session(speech, MockLLM(), sink, idle_seconds=0.128) as session:
        app = AudioApplication(session, WakeGate(Detector(), cooldown_samples=0), MarkedVAD())
        app.source.silence_frames = 2

        async def check(feed):
            for _ in range(2):
                await feed(voiced * 3 + quiet * 2)
                await session.generation
            assert speech.inputs == [voiced * 3 + quiet * 2, voiced * 3 + quiet * 2]
            assert len(app.session_ids) == 1 and app.utterance_count == 2
            await feed(quiet * 4)
            assert session.state == State.IDLE and not app.wake.active
            await feed(voiced * 4 + quiet * 2)
            await session.generation
            assert len(app.session_ids) == 2 and speech.inputs[-1].endswith(voiced * 4 + quiet * 2)
            assert app.monitor.activations == 2

        if pipeline:
            async with PipecatAudioInput(app) as ingress:

                async def feed(pcm):
                    await ingress.feed(pcm, app.pre.sequence)

                await check(feed)
                assert not await ingress.finish()
            assert ingress.task.done()
        else:
            await check(app.normalized)


async def test_pipecat_streaming_partial_and_barge_in_preserve_task_correction():
    display, sink = Transcript(), MemorySink(delay=60)
    voiced, quiet = b"\1\0" * 512, bytes(FRAME_BYTES)
    async with SpeechClient(transcript=["set mock light on", "set mock light off"]) as speech:
        async with Session(speech, MockLLM(), sink, transcript=display) as session:
            app = AudioApplication(session, None, MarkedVAD(), wake_frame=0, silence_frames=2)
            async with PipecatAudioInput(app) as ingress:
                await ingress.feed(voiced * 3, 0)
                async with asyncio.timeout(3):
                    while not display.messages:
                        await asyncio.sleep(0.001)
                assert display.snapshot()["messages"][0]["status"] == "partial"
                assert not sink.chunks and app.utterance_count == 0
                await ingress.feed(quiet * 2, 1)
                await asyncio.wait_for(sink.entered.wait(), 3)
                old = session.generation
                assert not old.done() and session.tasks.actions.effects == 1
                assert session.input_bytes == FRAME_BYTES * 5
                sink.delay = 0
                await asyncio.wait_for(ingress.feed(voiced * 3 + quiet * 2, 2), 3)
                await session.generation
                assert old.done() and session.tasks.actions.effects == 2
                assert not session.tasks.actions.light and sink.stops >= 2
                assert not any(
                    m["role"] == "assistant" and m["turn_id"] == 1 for m in session.history
                )
                assert not await ingress.finish()
                with pytest.raises(RuntimeError, match="not accepting"):
                    await ingress.feed(quiet, 3)
            assert ingress.task.done() and session.playback.queue.empty()


@pytest.mark.parametrize("failure", ["cancel", "deadline"])
async def test_pipecat_ingress_failure_and_cancel_release_worker_and_session(failure):
    class BlockedInput(AudioApplication):
        def __init__(self, session):
            super().__init__(session, None, MarkedVAD(), wake_frame=0)
            self.entered = asyncio.Event()
            self.released = asyncio.Event()

        async def feed(self, pcm, sequence):
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.released.set()

    async with Session(Speech(), MockLLM(), MemorySink()) as session:
        app = BlockedInput(session)
        async with PipecatAudioInput(app) as ingress:
            if failure == "deadline":
                ingress.timeout = 0.1
            pending = asyncio.create_task(ingress.feed(b"\1\0" * 512, 0))
            await asyncio.wait_for(app.entered.wait(), 3)
            if failure == "cancel":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                with pytest.raises(TimeoutError):
                    await pending
            with pytest.raises(RuntimeError, match="not accepting"):
                await ingress.feed(b"\1\0" * 512, 0)
        assert app.released.is_set() and ingress.task.done()

    class BrokenInput(AudioApplication):
        async def feed(self, pcm, sequence):
            raise ValueError("controlled ingestion failure")

    async with Session(Speech(), MockLLM(), MemorySink()) as session:
        app = BrokenInput(session, None, MarkedVAD(), wake_frame=0)
        async with PipecatAudioInput(app) as ingress:
            with pytest.raises(ValueError, match="ordered complete"):
                await ingress.feed(b"\1", 0)
            assert not app.pre.sequence
            with pytest.raises(ValueError, match="controlled ingestion"):
                await ingress.feed(b"\1\0" * 512, 0)
            with pytest.raises(RuntimeError, match="not accepting"):
                await ingress.finish()
        assert ingress.task.done() and not session.input_open


async def test_partial_projection_and_incomplete_input_cleanup_through_stdio():
    display = Transcript()
    async with SpeechClient(transcript="partial final") as speech:
        async with Session(speech, MockLLM(), MemorySink(), transcript=display) as session:
            session.activate()
            await session.begin_input()
            await session.push_input(b"\1\0" * 512)
            async with asyncio.timeout(3):
                while not display.messages:
                    await asyncio.sleep(0.001)
            assert display.snapshot()["messages"][0]["status"] == "partial"
            await session.interrupt()
            assert session.asr_task is None and not session.input_open
            await (await session.submit(b"\1\0" * 512))
            assert display.snapshot()["messages"][-1]["status"] == "final"


def test_configuration_requires_real_contract_and_never_echoes_secret(tmp_path):
    assert all(s.backend == "mock" for s in load_backends().values())
    spec = BackendSpec(
        backend="external",
        endpoint="https://example.invalid/asr",
        protocol="documented-but-not-implemented",
        auth_env="ME3_ASR_API_KEY",
        model="chosen",
    )
    assert spec.validate("asr", runnable=False) is spec
    with pytest.raises(ConfigurationError, match="not implemented"):
        spec.validate("asr", runnable=True)
    for endpoint in (
        "https://user:private-secret@example.invalid",
        "https://example.invalid?token=private-secret",
    ):
        with pytest.raises(ConfigurationError) as result:
            replace(spec, endpoint=endpoint).validate("asr")
        assert "private-secret" not in str(result.value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"asr": {"api_key": "private-secret"}, "tts": {}, "llm": {}}))
    with pytest.raises(ConfigurationError) as result:
        load_backends(path)
    assert "private-secret" not in str(result.value)


async def test_task_result_history_idempotency_and_correction():
    class Commands(Speech):
        async def transcribe(self, key, pcm):
            return ["set mock light on", "mock light status", "set mock light off"][key.turn_id - 1]

    async with Session(Commands(), MockLLM(), MemorySink()) as session:
        session.activate()
        await (await session.submit(b"\1\0"))
        assert session.tasks.actions.effects == 1 and session.tasks.actions.light
        await session.tasks.execute(
            session.key, ToolCall("light-change", "set_mock_light", {"enabled": True})
        )
        assert session.tasks.actions.effects == 1
        await (await session.submit(b"\1\0"))
        assert session.tasks.actions.effects == 1
        await (await session.submit(b"\1\0"))
        assert session.tasks.actions.effects == 2 and not session.tasks.actions.light
        assert len([m for m in session.history if m["role"] == "tool"]) == 3
        assert any(e["event"] == "tool_completed" for e in session.events)


async def test_cancelled_open_cannot_orphan_remote_operation_slot():
    opened, release = asyncio.Event(), asyncio.Event()
    client = SpeechClient()
    calls = []

    async def call(kind, name, arguments):
        if name == "asr_open":
            opened.set()
            await release.wait()
            return {"operation_id": "created-slot"}
        calls.append((name, arguments))
        return {"cancelled": True}

    client.call = call
    pending = asyncio.create_task(client.open_operation("asr", "asr_open", {}))
    await opened.wait()
    pending.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert calls == [("cancel", {"operation_id": "created-slot"})]


async def test_invalid_stream_contract_closes_slot_and_allows_next_operation():
    from me3_voice.providers import MockASR, MockTTS, ASRResult
    from me3_voice.servers import SpeechService
    from me3_voice.contracts import encode_pcm

    provider = MockASR()
    service = SpeechService(provider, MockTTS())
    opened = await service.open("bad-result", 1)
    stream = service.operations[opened["operation_id"]].asr_stream

    async def invalid(pcm):
        return ASRResult("premature final", final=True)

    stream.append = invalid
    with pytest.raises(ValueError, match="provider result"):
        await service.append(opened["operation_id"], 0, encode_pcm(b"\1\0"))
    assert stream.closed and not service.operations
    await service.open("bad-result", 2)
    await service.close()


def test_tool_timing_and_percentiles_keep_identity_and_missing_boundaries():
    from me3_voice.telemetry import summarize

    def event(name, when, call):
        return {
            "event": name,
            "monotonic": when,
            "session_id": "s",
            "turn_id": 1,
            "mode": "mock",
            "schema_version": 1,
            "run_id": "r",
            "clock_id": "local",
            "tool_call_id": call,
        }

    result = summarize(
        [
            event("tool_started", 1, "a"),
            event("tool_started", 2, "b"),
            event("tool_completed", 3, "a"),
            event("tool_completed", 6, "b"),
        ]
    )
    assert result["turns"][0]["durations_ms"]["tool_duration_ms"] == [2000, 4000]
    aggregate = result["software_aggregates"][0]
    assert aggregate["p50_ms"] == 3000 and aggregate["p95_ms"] == 3900
    assert "asr_request_ms" in result["turns"][0]["unavailable_metrics"]


def test_cli_diagnostic_replay_does_not_start_speech_providers(tmp_path, monkeypatch):
    from argparse import Namespace
    from me3_voice.cli import replay
    import me3_voice.readiness
    import me3_voice.runtime

    monkeypatch.setattr(
        me3_voice.readiness, "replay", lambda _: {"mode": "HPC-live-file-inference"}
    )

    async def forbidden(_):
        raise AssertionError("provider path must be explicitly requested")

    monkeypatch.setattr(me3_voice.runtime, "replay_application", forbidden)
    output = tmp_path / "report.json"
    replay(
        Namespace(
            mock_conversation=False,
            trace=None,
            transcript=False,
            wake_frame=None,
            backend_config=None,
            output=output,
        )
    )
    assert json.loads(output.read_text())["mode"] == "HPC-live-file-inference"


async def test_tts_duration_bound_counts_samples_not_number_of_partial_frames():
    from me3_voice.servers import SpeechService
    from me3_voice.providers import MockASR
    from me3_voice.contracts import MAX_AUDIO_BYTES

    closed = asyncio.Event()

    class SmallPackets:
        async def synthesize(self, text):
            try:
                for _ in range(1200):
                    yield b"\1\0"
            finally:
                closed.set()

    service = SpeechService(MockASR(), SmallPackets())
    identifier = (await service.open("small-packets", 1, "hello"))["operation_id"]
    for _ in range(1000):
        assert not (await service.next(identifier))["done"]
    service.operations[identifier].output_bytes = MAX_AUDIO_BYTES
    with pytest.raises(ValueError, match="30 seconds"):
        await service.next(identifier)
    assert closed.is_set() and not service.operations


async def test_first_tts_packet_is_played_before_second_packet_is_available():
    first_packet = asyncio.Event()
    second_packet = asyncio.Event()

    class StreamingSpeech(Speech):
        async def synthesize(self, key, text):
            await first_packet.wait()
            yield b"\1\0" * 512
            await second_packet.wait()
            yield b"\2\0" * 512

    sink = MemorySink()
    display = Transcript()
    async with Session(StreamingSpeech(), MockLLM(), sink, transcript=display) as session:
        session.activate()
        task = await session.submit(b"\1\0")
        async with asyncio.timeout(2):
            while not any(event["event"] == "tts_started" for event in session.events):
                await asyncio.sleep(0.001)
        assert session.state == State.THINKING and display.state == "thinking"
        first_packet.set()
        await asyncio.wait_for(sink.entered.wait(), 1)
        assert session.state == State.SPEAKING and display.state == "speaking"
        assert not second_packet.is_set()
        assert not any(m["role"] == "assistant" for m in session.history)
        second_packet.set()
        await task
        assert len(sink.chunks) == 2
        assert session.history[-1]["delivery"] == "mock_sink_confirmed"


async def test_playback_stops_locally_before_remote_cleanup_completes():
    remote_cleanup = asyncio.Event()
    remote_entered = asyncio.Event()
    stopped = asyncio.Event()

    class SlowCleanup(Speech):
        async def synthesize(self, key, text):
            try:
                yield b"\1\0" * 512
                await asyncio.Event().wait()
            finally:
                remote_entered.set()
                await remote_cleanup.wait()

    class LocalSink(MemorySink):
        async def stop(self):
            await super().stop()
            stopped.set()

    sink = LocalSink(delay=60)
    async with Session(SlowCleanup(), MockLLM(), sink) as session:
        session.activate()
        await session.submit(b"\1\0")
        await asyncio.wait_for(sink.entered.wait(), 1)
        stopped.clear()
        interrupt = asyncio.create_task(session.interrupt())
        await asyncio.wait_for(stopped.wait(), 1)
        await asyncio.wait_for(remote_entered.wait(), 1)
        assert not interrupt.done() and session.playback.queue.empty() and not sink.chunks
        remote_cleanup.set()
        await asyncio.wait_for(interrupt, 1)
        assert not any(m["role"] == "assistant" for m in session.history)


async def test_pipecat_native_stereo_rate_matches_startup_and_single_resampling():
    async with Session(Speech(), MockLLM(), MemorySink()) as session:
        app = AudioApplication(session, None, MarkedVAD(), rate=48000, channels=2)
        ingress = PipecatAudioInput(app)
        advertised = []

        @ingress.worker.event_handler("on_pipeline_started")
        async def started(worker, frame):
            advertised.append(ingress.processor.processor_setup.audio_in_sample_rate)

        async with ingress:
            await ingress.feed(bytes(4800 * 4), 0)
            await ingress.finish()
        assert advertised == [48000]
        assert ingress.processor.frames == 1 and ingress.task.done()
        assert app.pre.input_stats.samples == 9600
        assert app.pre.output_stats.samples == 1600
        assert not session.history and not app.utterance_count
