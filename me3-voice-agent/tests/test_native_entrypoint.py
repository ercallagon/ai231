"""Real application CLI and loopback WebSocket; speech MCP children are actual."""

import asyncio
from argparse import Namespace
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from uuid import uuid4

from aiohttp import ClientSession, WSMsgType
from aiohttp.test_utils import TestServer
import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

from me3_voice.configuration import load_backends, load_runtime, ConfigurationError
from me3_voice.voice_bridge import build_app, BrowserSink


class Wake:
    frame_samples = 1280

    def feed(self, _pcm):
        return {"triggered": True}

    def end_session(self):
        pass


class FixtureVAD(FrameProcessor):
    """Annotated detector boundary, never an acoustic reliability measurement."""

    def __init__(self):
        super().__init__()
        self.speaking = False
        self.quiet = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            voiced = any(frame.audio)
            if voiced and not self.speaking:
                self.speaking = True
                await self.broadcast_frame(VADUserStartedSpeakingFrame)
            self.quiet = 0 if voiced else self.quiet + 1
            if self.speaking and self.quiet >= 2:
                self.speaking = False
                await self.broadcast_frame(VADUserStoppedSpeakingFrame)
        await self.push_frame(frame, direction)


@pytest.fixture
def native_args():
    return Namespace(engine="native", runtime_config=None, followup_seconds=20, extended_seconds=20)


async def next_json(ws, kind, *, acknowledge=True, timeout=12):
    async with asyncio.timeout(timeout):
        while True:
            message = await ws.receive()
            if message.type == WSMsgType.BINARY:
                if acknowledge:
                    await ws.send_json(
                        {"type": "audio_ack", "id": struct.unpack("<I", message.data[:4])[0]}
                    )
            elif message.type == WSMsgType.TEXT:
                data = message.json()
                if data.get("type") == "error":
                    raise AssertionError(data)
                if data.get("type") == kind:
                    return data
            else:
                raise AssertionError("connection closed before " + kind)


async def send_turn(ws):
    for _ in range(5):
        await ws.send_bytes(b"\1\0" * 512)
    for _ in range(3):
        await ws.send_bytes(b"\0\0" * 512)


async def wait_answer(ws, count=1):
    async with asyncio.timeout(15):
        while True:
            data = await next_json(ws, "transcript")
            answers = [m for m in data["data"]["messages"] if m["role"] == "assistant"]
            if len(answers) >= count:
                return data["data"]


def test_runtime_selection_is_explicit_and_legacy_default(tmp_path):
    assert load_runtime().engine == "legacy"
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"engine": "native", "silence_seconds": 0}))
    assert load_runtime(path).engine == "native"
    assert load_runtime(path, engine="legacy").engine == "legacy"
    path.write_text('{"engine":"nativ","extra":true}')
    with pytest.raises(ConfigurationError):
        load_runtime(path)
    with pytest.raises(ConfigurationError):
        load_runtime(tmp_path / "missing.json")


@pytest.mark.parametrize("engine", ["legacy", "native"])
def test_real_cli_entrypoint_with_mock_upstream_tool_events(tmp_path, engine):
    # Generated application artifacts must live under the repository's var/ boundary.
    output = Path.cwd() / "var" / "scratch" / f"pytest-{engine}-{uuid4().hex}.json"
    command = [
        sys.executable,
        "-B",
        "-m",
        "me3_voice.cli",
        "demo",
        "--engine",
        engine,
        "--mock-transcripts-json",
        json.dumps(["set mock light on", "mock light status"]),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-4000:]
    value = json.loads(output.read_text())
    assert value["engine"] == engine and set(value["protocols"]) == {"asr", "tts"}
    assert [m["content"] for m in value["history"] if m["role"] == "user"] == [
        "set mock light on",
        "mock light status",
    ]
    assert value["sink_chunks"] > 0
    if engine == "native":
        assert not value["legacy_session_imported"]
        assert len([e for e in value["events"] if e["event"] == "tool_completed"]) == 2


async def test_actual_native_browser_route_continuity_and_reconnect(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge
    import me3_voice.session as legacy

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    monkeypatch.setattr(
        legacy.Session,
        "__init__",
        lambda *_a, **_k: pytest.fail("Session instantiated in native path"),
    )
    app = build_app(tmp_path, load_backends(), {}, native_args)
    async with TestServer(app) as server, ClientSession() as client:
        async with client.get(server.make_url("/")) as response:
            page = await response.text()
            assert "Last source retrieval" in page
        session_id = None
        for expected in (1, 2):
            async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
                await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
                ready = await next_json(ws, "ready")
                assert ready["engine"] == "native"
                if session_id:
                    assert ready["session_id"] == session_id
                session_id = ready["session_id"]
                await send_turn(ws)
                view = await wait_answer(ws, expected)
                assert len([m for m in view["messages"] if m["role"] == "user"]) == expected
                await ws.close()
            await asyncio.sleep(0.1)
        async with client.get(server.make_url("/api/health")) as response:
            assert (await response.json())["session_resume"]


async def test_native_browser_writes_private_sanitized_session_events(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    scratch = Path.cwd() / "var" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-journal-", dir=scratch) as folder:
        async with (
            TestServer(
                build_app(tmp_path, load_backends(), {}, native_args, trace_dir=Path(folder))
            ) as server,
            ClientSession() as client,
        ):
            async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
                await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
                ready = await next_json(ws, "ready")
                await send_turn(ws)
                await wait_answer(ws)
                await ws.close()
            await asyncio.sleep(0.05)
        journal = Path(folder) / f"native-{ready['run_id']}.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        assert journal.stat().st_mode & 0o777 == 0o600
        assert rows and all(row["run_id"] == ready["run_id"] for row in rows)
        assert all(row["session_id"] == ready["session_id"] for row in rows)
        assert sum(row["event"] == "wake_detected" for row in rows) == 1
        assert any(row["event"] == "turn_completed" for row in rows)
        assert all("text" not in row and "content" not in row for row in rows)


async def test_native_browser_manual_arm_accepts_one_turn_without_wake_detection(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge

    class NoWake(Wake):
        def feed(self, _pcm):
            return {"triggered": False}

    monkeypatch.setattr(bridge, "build_wake", lambda _: NoWake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            await next_json(ws, "ready")
            await ws.send_json({"type": "arm"})
            await send_turn(ws)
            result = await wait_answer(ws)
            assert any(row["role"] == "assistant" for row in result["messages"])


def test_native_event_projection_excludes_secrets_and_free_text():
    from me3_voice.native_bridge import _safe_event

    row = _safe_event(
        {
            "event": "asr_final",
            "at": 123.5,
            "turn_id": 2,
            "text": "private speech",
            "content": "private answer",
            "api_key": "secret",
            "auth_token": "secret",
            "error_code": "asr_timeout",
            "error_message": "private provider payload",
        }
    )
    assert row == {"event": "asr_final", "at": 123.5, "turn_id": 2, "error_code": "asr_timeout"}
    tts = _safe_event(
        {
            "event": "tts_segment_summary",
            "diagnostics": {
                "provider": {
                    "provider_audio_seconds": 2.4,
                    "synthesis_wall_seconds": 5.6,
                    "text": "private text",
                    "authorization": "secret",
                },
                "client_mcp": {
                    "tts_next_count": 8,
                    "tts_next_latency_seconds": {"p95_seconds": 0.02, "payload": "secret"},
                },
            },
        }
    )
    assert tts == {
        "event": "tts_segment_summary",
        "provider_provider_audio_seconds": 2.4,
        "provider_synthesis_wall_seconds": 5.6,
        "client_mcp_tts_next_count": 8,
        "client_mcp_next_p95_seconds": 0.02,
    }


async def test_native_bridge_serves_the_existing_local_aec_probe(tmp_path, native_args):
    app = build_app(tmp_path, load_backends(), {}, native_args)
    async with TestServer(app, host="127.0.0.1") as server, ClientSession() as client:
        async with client.get(
            server.make_url("/audio-probe"), headers={"Host": "example.com"}
        ) as response:
            assert response.status == 403
        async with client.get(server.make_url("/audio-probe")) as response:
            assert response.status == 200
            body = await response.text()
            policy = response.headers["Content-Security-Policy"]
            nonce = re.search(r"script-src 'nonce-([^']+)'", policy).group(1)
            assert f'<script nonce="{nonce}">' in body
            assert "connect-src 'none'" in policy
            assert "Nothing is uploaded" in body
            assert "echoCancellation:{ideal:requested}" in body
            assert response.headers["Permissions-Policy"] == "microphone=(self)"


async def test_native_browser_opt_in_saves_bounded_received_audio(
    tmp_path, monkeypatch, native_args
):
    import wave

    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    native_args.diagnostic_audio = True
    native_args.diagnostic_max_seconds = 1
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            ready = await next_json(ws, "ready")
            assert ready["diagnostic_audio"] is True
            await ws.send_bytes(b"\1\0" * 512)
            await ws.send_json({"type": "stop"})
            while (await ws.receive()).type not in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                pass
    path = tmp_path / "var/recordings/diagnostics" / ready["run_id"]
    with wave.open(str(path / "browser-received.wav"), "rb") as captured:
        assert captured.getparams().nchannels == 1
        assert captured.readframes(512) == b"\1\0" * 512
    assert json.loads((path / "manifest.json").read_text())["raw"]["received_bytes"] == 1024


async def test_browser_stop_fences_a_writer_waiting_for_capacity():
    class Socket:
        closed = False

        def __init__(self):
            self.packets = []

        async def send_bytes(self, b):
            self.packets.append(b)

        async def send_json(self, _):
            pass

    socket = Socket()
    sink = BrowserSink(socket)
    for _ in range(16):
        await sink(b"\1\0")
    delayed = asyncio.create_task(sink(b"\2\0"))
    await asyncio.sleep(0)
    await sink.stop()
    assert await delayed is False
    assert len(socket.packets) == 16 and not sink.pending


def test_native_replay_event_ring_retains_turn_lifecycle_through_streamed_playback():
    from types import SimpleNamespace

    from me3_voice.native_app import NATIVE_REPLAY_EVENT_HISTORY_LIMIT, NativeApplication

    app = NativeApplication(SimpleNamespace(session_id="event-retention"), object(), annotated=True)
    app.event("asr_final")
    app.event("llm_usage", prompt_tokens=123, completion_tokens=45)
    app.event("tts_start")
    for _ in range(2890):
        app.event("playback_packet", bytes=1280)

    names = [event["event"] for event in app.events]
    assert names[:3] == ["asr_final", "llm_usage", "tts_start"]
    assert len(app.events) == 2893 < NATIVE_REPLAY_EVENT_HISTORY_LIMIT


def test_native_transcript_projection_shows_live_asr_partial_without_context_insertion():
    from types import SimpleNamespace

    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_bridge import context_view
    from me3_voice.configuration import load_backends

    state = SimpleNamespace(
        context=LLMContext([{"role": "user", "content": "previous finalized question"}]),
        session_id="partial-session",
        backends=load_backends(),
        task_runtime=None,
    )
    application = SimpleNamespace(
        partial_transcript="current words so far",
        projection="listening",
        event_count=4,
    )
    view = context_view(state, application)
    assert view["messages"][-1] == {
        "id": "2:user",
        "turn_id": 2,
        "role": "user",
        "text": "current words so far",
        "confirmed_text": "",
        "status": "partial",
        "delivery": "asr_partial",
    }
    assert state.context.get_messages() == [
        {"role": "user", "content": "previous finalized question"}
    ]


def test_native_transcript_revision_advances_for_task_and_action_updates():
    from types import SimpleNamespace

    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_bridge import context_view
    from me3_voice.configuration import load_backends

    task_store = SimpleNamespace(revision=2, snapshot=lambda: [])
    state = SimpleNamespace(
        context=LLMContext(),
        session_id="revision-session",
        backends=load_backends(),
        task_runtime=SimpleNamespace(store=task_store),
    )
    application = SimpleNamespace(
        partial_transcript=None,
        projection="listening",
        event_count=4,
        action_revision=0,
        fast_action_rows=[],
    )

    first = context_view(state, application)["revision"]
    application.action_revision += 1
    second = context_view(state, application)["revision"]
    task_store.revision += 1
    third = context_view(state, application)["revision"]

    assert second > first
    assert third > second


async def test_native_transcript_view_archives_session_with_existing_publisher(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from pipecat.processors.aggregators.llm_context import LLMContext
    import me3_voice.transcript as transcript_module
    from me3_voice.native_bridge import NativeTranscriptView
    from me3_voice.transcript import TranscriptPublisher
    from me3_voice.configuration import load_backends

    monkeypatch.setattr(transcript_module, "output_path", lambda value: Path(value))
    archive_dir = tmp_path / "voice-sessions"
    archive_dir.mkdir()
    state = SimpleNamespace(
        context=LLMContext(
            [
                {"role": "user", "content": "Turn the simulated lamp on"},
                {"role": "assistant", "content": "The simulated lamp is on."},
            ]
        ),
        session_id="native-session-archive",
        backends=load_backends(),
        task_runtime=None,
    )
    application = SimpleNamespace(
        partial_transcript=None,
        projection="listening",
        event_count=3,
        fast_action_rows=[],
    )
    publisher = TranscriptPublisher(
        NativeTranscriptView(state, application),
        path=tmp_path / "current.json",
        archive_dir=archive_dir,
    )
    await publisher.__aenter__()
    await publisher.__aexit__(None, None, None)

    archived = json.loads((archive_dir / "native-session-archive.json").read_text())
    assert [row["text"] for row in archived["messages"]] == [
        "Turn the simulated lamp on",
        "The simulated lamp is on.",
    ]
    assert archived["mode"] == "mock"


def test_native_transcript_response_projection_distinguishes_interrupted_rain_answer():
    from types import SimpleNamespace
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_bridge import context_view
    from me3_voice.configuration import load_backends

    response_id = "response-123"
    state = SimpleNamespace(
        context=LLMContext(
            [
                {"role": "user", "content": "Explain rain in three sentences."},
                {
                    "role": "assistant",
                    "content": "First sentence. Second sentence. Third sentence.",
                },
            ]
        ),
        session_id="rain-turn-session",
        backends=load_backends(),
        task_runtime=None,
    )
    application = SimpleNamespace(
        partial_transcript=None,
        projection="listening",
        event_count=8,
        fast_action_rows=[],
        response_by_conversation_turn={1: response_id},
        response_records={
            response_id: {
                "response_id": response_id,
                "response_state": "interrupted",
                "generated_state": "completed",
                "tts_state": "interrupted",
                "playback_state": "partially_acknowledged",
                "tts_segments": [
                    {
                        "segment_id": 1,
                        "characters": 48,
                        "text_sha256": "a" * 16,
                        "state": "cancelled",
                    }
                ],
                "playback_packets": 12,
                "playback_acks": 7,
                "superseded_by_turn_id": 2,
                "interruption_reason": "vad_barge_in",
            }
        },
    )
    row = context_view(state, application, archived=True)["messages"][1]
    assert row["status"] == "interrupted"
    assert row["delivery"] == "generated"
    assert row["generation_complete"] is True
    assert row["confirmed_text"] == ""
    assert row["response"]["state"] == "interrupted"
    assert row["response"]["archive"] == "included"
    assert row["response"]["superseded_by_turn_id"] == 2


def test_native_tts_submission_event_keeps_only_sanitized_identity_and_hash():
    from types import SimpleNamespace
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_app import NativeApplication

    app = NativeApplication(
        SimpleNamespace(session_id="tts-event-session", context=LLMContext()), object()
    )
    app.event("tts_response_started", response_id="response-456")
    event = app.event(
        "tts_start",
        response_id="response-456",
        segment_id=1,
        characters=12,
        submitted_text="The lamp is red.",
    )
    assert "submitted_text" not in event
    assert "The lamp is red." not in repr(event)
    segment = app.response_records["response-456"]["tts_segments"][0]
    assert segment["characters"] == len("The lamp is red.")
    assert len(segment["text_sha256"]) == 16


async def test_native_followup_window_starts_after_response_and_resets_on_followup():
    from types import SimpleNamespace

    from me3_voice.native_app import NativeApplication

    app = NativeApplication(
        SimpleNamespace(session_id="followup-test"),
        object(),
        wake=SimpleNamespace(frame_samples=320),
        followup_seconds=0.06,
    )
    app.parts = SimpleNamespace(stt=SimpleNamespace(speaking=False))
    app.gate.active = True
    app.projection = "speaking"
    app.settled.clear()
    app._arm_followup_timer()
    await asyncio.sleep(0.08)
    assert app.gate.active  # An in-progress answer never expires.

    app.projection = "listening"
    app.settled.set()
    app._arm_followup_timer()
    await asyncio.sleep(0.03)
    app._cancel_followup_timer()  # The next valid user turn resets the window.
    app._arm_followup_timer()
    await asyncio.sleep(0.04)
    assert app.gate.active
    await asyncio.sleep(0.04)
    assert not app.gate.active
    assert any(row["event"] == "engagement_expired" for row in app.events)


async def test_delayed_asr_interrupt_through_websocket_never_publishes_old_turn(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge
    from me3_voice.speech import SpeechClient

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    monkeypatch.setattr(
        module,
        "SpeechClient",
        lambda **kw: SpeechClient(**kw, transcript=["obsolete", "current"], delay=0.3),
    )
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            await next_json(ws, "ready")
            await send_turn(ws)
            async with asyncio.timeout(10):
                while (await next_json(ws, "runtime_event"))["data"][
                    "event"
                ] != "asr_finish_requested":
                    pass
            await ws.send_json({"type": "interrupt"})
            await next_json(ws, "audio_stop")
            await send_turn(ws)
            view = await wait_answer(ws)
            assert [m["text"] for m in view["messages"] if m["role"] == "user"] == ["current"]
            assert "obsolete" not in str(view["messages"])


async def test_barge_in_before_ack_rejects_old_text_and_late_ack(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge
    import me3_voice.native_llm as model_module

    class Stream:
        calls = 0

        async def respond(self, _):
            self.calls += 1
            if self.calls == 1:
                yield "First sentence. "
                await asyncio.sleep(0.5)
                yield "Obsolete tail."
            else:
                yield "Current reply."

    upstream = Stream()
    monkeypatch.setattr(model_module, "MockLLM", lambda: upstream)
    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            await next_json(ws, "ready")
            await send_turn(ws)
            async with asyncio.timeout(10):
                while True:
                    message = await ws.receive()
                    if message.type == WSMsgType.BINARY:
                        old_id = struct.unpack("<I", message.data[:4])[0]
                        break
            await ws.send_json({"type": "interrupt"})
            await next_json(ws, "audio_stop", acknowledge=False)
            await ws.send_json({"type": "audio_ack", "id": old_id})
            await send_turn(ws)
            view = await wait_answer(ws)
            assistants = [m["text"] for m in view["messages"] if m["role"] == "assistant"]
            assert assistants == ["Current reply."]
            assert upstream.calls == 2


async def test_same_connection_asr_failure_can_be_interrupted_and_recovered(
    tmp_path, monkeypatch, native_args
):
    from me3_voice.native_app import NativeState, NativeApplication
    from me3_voice.playback import MemorySink
    from me3_voice.speech import SpeechClient

    state = NativeState(
        speech_factory=lambda: SpeechClient(
            transcript=["failed", "recovered"], delay=0.2, timeout=5
        )
    )
    try:
        async with NativeApplication(state, MemorySink(), annotated=True) as app:
            state.speech.timeout = 0.03
            with pytest.raises(RuntimeError):
                await app.utterance(b"\1\0" * 512)
            assert not [m for m in state.context.get_messages() if m.get("role") == "assistant"]
            await app.interrupt()
            state.speech.timeout = 5
            await app.utterance(b"\1\0" * 512)
            assert [m["content"] for m in state.context.get_messages() if m["role"] == "user"] == [
                "recovered"
            ]
    finally:
        await state.close()


async def test_native_feed_recovers_after_completed_turn_error_without_reconnecting():
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.playback import MemorySink

    state = NativeState()
    try:
        async with NativeApplication(state, MemorySink(), annotated=True) as app:
            app.error = RuntimeError("simulated completed turn failure")
            await app.feed(b"\1\0" * 512, 0)
            await app.drain()
            assert app.error is None
            recovered = [event for event in app.events if event["event"] == "turn_recovered"]
            assert len(recovered) == 1
            assert recovered[0]["error_code"] == "pipeline_error"
    finally:
        await state.close()


async def test_native_known_device_command_uses_shared_fast_route_without_llm():
    from me3_voice.native_app import NativeState, NativeApplication
    from me3_voice.native_llm import NativeLLM
    from pipecat.processors.aggregators.llm_context import LLMContext

    class NoLLM:
        async def respond(self, _history):
            raise AssertionError("known device commands must not call the LLM")
            yield "unreachable"

    state = NativeState()
    llm = None
    try:
        app = NativeApplication(state, object(), annotated=True)
        app.turn_id = 1
        context = LLMContext()
        context.add_message({"role": "user", "content": "Can you turn on the light?"})
        llm = NativeLLM(state.backends["llm"], upstream=NoLLM(), fast_action=app.fast_action)
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        spoken = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert state.tasks.actions.light is True
        assert "lamp" in spoken
        assert llm.calls == 0
        assert any(e["event"] == "fast_action_completed" for e in app.events)
        route = next(e for e in app.events if e["event"] == "route_selected")
        assert route["route"] == "fast_command"
        assert isinstance(route["at"], (int, float))
        assert app.fast_action_rows[-1]["kind"] == "light_on"
        assert app.fast_action_rows[-1]["status"] == "simulated"
        from me3_voice.native_bridge import context_view

        projected = context_view(state, app)["tasks"]
        assert projected[-1]["kind"] == "light_on"
        assert projected[-1]["widget"]["type"] == "lamp"
        assert projected[-1]["summary"].startswith("Simulated")
        context.set_messages([])
        context.add_message({"role": "user", "content": "Pause the music"})
        app.turn_id = 2
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        assert (
            "pause" in "".join(chunk.choices[0].delta.content or "" for chunk in chunks).casefold()
        )
        assert llm.calls == 0
        assert len(state.tasks.music.operations) == 1
        context.set_messages([])
        context.add_message({"role": "user", "content": "Set the music volume to 30 percent"})
        app.turn_id = 3
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        spoken = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert "30%" in spoken
        assert state.tasks.music.mock_volume == 30
        assert llm.calls == 0
        assert len(state.tasks.music.operations) == 2

        context.set_messages([])
        context.add_message({"role": "user", "content": "time is it in Manila"})
        app.turn_id = 4
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        time_answer = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert time_answer
        assert llm.calls == 0
        assert app.fast_action_rows[-1]["kind"] == "get_time"

        # A contextual percentage is only routed directly when the immediately
        # preceding turn actually completed a light action. This exercises the
        # same OpenAI-shaped tool history Pipecat supplies to the native LLM.
        context.set_messages(
            [
                {"role": "user", "content": "Turn on the lights"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "light-call",
                            "type": "function",
                            "function": {
                                "name": "run_device_action",
                                "arguments": '{"action":"light_on"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "light-call",
                    "content": '{"result_status":"verified","summary":"Lamp is on."}',
                },
                {"role": "assistant", "content": "The lamp is on."},
                {"role": "user", "content": "Dim it to thirty percent"},
            ]
        )
        app.turn_id = 2
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        assert "30" in "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert state.tasks.actions.brightness == 30
        assert llm.calls == 0
    finally:
        if llm is not None:
            await llm._client.close()
        await state.close()


async def test_fast_music_receipt_maps_provider_mode_to_safe_ui_evidence():
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.native_bridge import context_view
    from me3_voice.providers import ToolCall

    state = NativeState()
    try:
        app = NativeApplication(state, object(), annotated=True)
        call = ToolCall("music-ui-1", "control_music", {"action": "play", "query": "artist"})
        row = app._fast_action_row(call, 1)
        app._finish_fast_action(
            row,
            result={
                "mode": "spotify",
                "result_status": "reported",
                "summary": "Spotify reported playback started.",
                "display": "playing",
            },
        )

        receipt = context_view(state, app)["tasks"][-1]
        assert receipt["kind"] == "music_play"
        assert receipt["status"] == "reported"
        assert receipt["mode"] == "reported"
        assert receipt["summary"] == "Spotify reported playback started."
    finally:
        await state.close()


async def test_interrupted_fast_action_is_recorded_before_later_unrelated_turn():
    from me3_voice.native_app import NativeApplication, NativeState
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    try:
        app = NativeApplication(state, object(), annotated=True)
        app.turn_id = 1
        context = LLMContext()
        context.add_message({"role": "user", "content": "Turn on the lights"})

        answer = await app.fast_action(context)
        assert "lamp" in answer.casefold()
        assert state.tasks.actions.effects == 1

        messages = context.get_messages()
        assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
        call = messages[-2]["tool_calls"][0]
        result = json.loads(messages[-1]["content"])
        assert call["function"]["name"] == "run_device_action"
        assert result["ok"] is True
        assert result["action"] == "light_on"

        # The spoken confirmation may be interrupted before Pipecat commits its
        # assistant text. The next unrelated request must still see the tool
        # result and must not execute the previous command a second time.
        context.add_message({"role": "user", "content": "Explain how rain forms"})
        app.turn_id = 2
        assert await app.fast_action(context) is None
        assert state.tasks.actions.effects == 1
    finally:
        await state.close()


async def test_native_fast_music_route_clarifies_missing_track_without_llm_or_tool_call():
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.native_llm import NativeLLM
    from pipecat.processors.aggregators.llm_context import LLMContext

    class NoLLM:
        async def respond(self, _history):
            raise AssertionError("incomplete music requests must clarify before calling tools")
            yield "unreachable"

    state = NativeState()
    llm = None
    try:
        app = NativeApplication(state, object(), annotated=True)
        app.turn_id = 1
        context = LLMContext()
        context.add_message({"role": "user", "content": "Can you turn on the music?"})
        llm = NativeLLM(state.backends["llm"], upstream=NoLLM(), fast_action=app.fast_action)
        llm.new_turn()

        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        answer = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert answer == "What song, artist, or playlist would you like me to play?"
        assert llm.calls == 0
        assert state.tasks.music.operations == {}
        assert any(event["event"] == "clarification_requested" for event in app.events)
    finally:
        if llm is not None:
            await llm._client.close()
        await state.close()


async def test_native_fast_route_does_not_intercept_research_or_delegated_work():
    from me3_voice.native_app import NativeState, NativeApplication
    from me3_voice.native_llm import NativeLLM
    from pipecat.processors.aggregators.llm_context import LLMContext

    class CountingLLM:
        calls = 0

        async def respond(self, _history):
            self.calls += 1
            yield "Handled by the general conversation model."

    state = NativeState()
    upstream = CountingLLM()
    llm = None
    try:
        app = NativeApplication(state, object(), annotated=True)
        llm = NativeLLM(state.backends["llm"], upstream=upstream, fast_action=app.fast_action)
        for turn, text in enumerate(
            (
                "Research the latest Bank of England rate and summarize the source.",
                "Start a task to create a small website about tide pools.",
                "Could you explain how echo cancellation works?",
            ),
            start=1,
        ):
            app.turn_id = turn
            llm.new_turn()
            context = LLMContext()
            context.add_message({"role": "user", "content": text})
            chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
            spoken = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
            assert "general conversation model" in spoken
            assert llm.calls == 1
        assert upstream.calls == 3
        assert not any(event["event"] == "fast_action_started" for event in app.events)
    finally:
        if llm is not None:
            await llm._client.close()
        await state.close()


def test_native_file_replay_uses_actual_cli_dispatch(tmp_path, monkeypatch):
    import wave
    import me3_voice.native_app as module
    from me3_voice.cli import main

    path = tmp_path / "input.wav"
    # Input fixtures may live in pytest's tmp_path; generated outputs may not.
    output = Path.cwd() / "var" / "scratch" / f"pytest-replay-{uuid4().hex}.json"
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"engine":"native","silence_seconds":0}')
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((b"\1\0" * 512) * 5 + (b"\0\0" * 512) * 4)
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "me3",
            "replay",
            str(path),
            "--conversation",
            "--runtime-config",
            str(runtime),
            "--wake-frame",
            "0",
            "--output",
            str(output),
        ],
    )
    main()
    value = json.loads(output.read_text())
    assert (
        value["engine"] == "native" and value["utterances"] == 1 and not value["unfinished_input"]
    )
    assert any(e["event"] == "turn_completed" for e in value["events"])


async def test_native_startup_failure_closes_partial_model_and_can_retry(monkeypatch):
    from me3_voice.native_app import NativeState, NativeApplication
    from me3_voice.playback import MemorySink

    original = NativeApplication.register_tools

    def broken(self):
        raise ValueError("fixture registration failure")

    state = NativeState()
    try:
        monkeypatch.setattr(NativeApplication, "register_tools", broken)
        app = NativeApplication(state, MemorySink(), annotated=True)
        with pytest.raises(ValueError):
            await app.__aenter__()
        assert app.llm._client.is_closed()
        monkeypatch.setattr(NativeApplication, "register_tools", original)
        async with NativeApplication(state, MemorySink(), annotated=True) as retry:
            await retry.utterance(b"\1\0" * 512)
            assert any(e["event"] == "turn_completed" for e in retry.events)
    finally:
        await state.close()


@pytest.mark.parametrize(
    (
        "answer_sentences",
        "inter_sentence_delay",
        "second_tts_startup_delay",
        "first_tts_startup_delay",
    ),
    [
        (("At sea level, water boils at 100 degrees Celsius.",), 0, 0, 0),
        (
            (
                "At sea level, water boils at 100 degrees Celsius.",
                "That is 212 degrees Fahrenheit.",
            ),
            0,
            0,
            0,
        ),
        (
            (
                "At sea level, water boils at 100 degrees Celsius.",
                "That is 212 degrees Fahrenheit.",
                "The boiling point changes when atmospheric pressure changes.",
            ),
            0,
            0,
            0,
        ),
        (
            (
                "At sea level, water boils at 100 degrees Celsius.",
                "That is 212 degrees Fahrenheit.",
            ),
            3.1,
            0,
            0,
        ),
        (
            (
                "At sea level, water boils at 100 degrees Celsius.",
                "That is 212 degrees Fahrenheit.",
            ),
            0,
            3.2,
            0,
        ),
        (
            ("At sea level, water boils at 100 degrees Celsius.",),
            0,
            0,
            3.2,
        ),
    ],
    ids=(
        "one-sentence",
        "two-sentences",
        "three-sentences",
        "llm-idle-gap",
        "delayed-second-tts-pcm",
        "delayed-first-tts-pcm",
    ),
)
async def test_native_wav_replay_writes_sanitized_ledger_and_audio_summary(
    tmp_path,
    monkeypatch,
    answer_sentences,
    inter_sentence_delay,
    second_tts_startup_delay,
    first_tts_startup_delay,
):
    import sqlite3
    import wave
    from argparse import Namespace
    import me3_voice.native_app as native
    import me3_voice.readiness as readiness
    import me3_voice.task_runtime as task_runtime
    from me3_voice.configuration import RuntimeOptions
    from me3_voice.research import ResearchOptions
    from me3_voice.speech import SpeechClient

    speech_timeouts = []
    settle_timeouts = []
    original_wait_for = native.asyncio.wait_for

    async def capture_wait_for(awaitable, timeout, *args, **kwargs):
        settle_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout, *args, **kwargs)

    class CapturingSpeechClient(SpeechClient):
        def __init__(self, *args, **kwargs):
            speech_timeouts.append(kwargs.get("timeout", 5))
            self.synthesis_count = 0
            super().__init__(*args, **kwargs)

        async def synthesize(self, key, text):
            self.synthesis_count += 1
            if self.synthesis_count == 1 and first_tts_startup_delay:
                await asyncio.sleep(first_tts_startup_delay)
            if self.synthesis_count == 2 and second_tts_startup_delay:
                await asyncio.sleep(second_tts_startup_delay)
            async for pcm in super().synthesize(key, text):
                yield pcm

    class ScriptedAnswer:
        def __init__(self, sentences, delay):
            self.sentences, self.delay = sentences, delay

        async def respond(self, _history):
            for index, sentence in enumerate(self.sentences):
                yield sentence
                if index + 1 < len(self.sentences) and self.delay:
                    await asyncio.sleep(self.delay)

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\1\0" * 5_120 + b"\0\0" * 12_800)
    monkeypatch.setattr(native, "native_vad", FixtureVAD)
    monkeypatch.setattr(readiness, "build_wake", lambda _args: None)
    monkeypatch.setattr(native, "SpeechClient", CapturingSpeechClient)
    monkeypatch.setattr(native.asyncio, "wait_for", capture_wait_for)
    original_task_runtime = task_runtime.TaskRuntime

    def isolated_task_runtime(*args, **kwargs):
        return original_task_runtime(
            *args,
            database=tmp_path / "task-runtime" / "tasks.sqlite",
            artifact_root=tmp_path / "task-artifacts",
            **kwargs,
        )

    monkeypatch.setattr(task_runtime, "TaskRuntime", isolated_task_runtime)
    ledger = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "native-replay-test-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    backend_config = tmp_path / "backends.json"
    backend_config.write_text(
        json.dumps(
            {
                "asr": {"timeout_seconds": 35},
                "llm": {"timeout_seconds": 45},
                "tts": {"timeout_seconds": 5 if first_tts_startup_delay else 60},
            }
        )
    )
    args = Namespace(
        onit_python=None,
        direct_clock=False,
        turn_policy="silence",
        processed_wav=None,
        wav=source,
        output=tmp_path / "result.json",
        trace=None,
        backend_config=backend_config,
        secrets_file=None,
        llm_secrets_file=None,
        wake_frame=0,
        transcript=True,
        mock_transcripts=["What is the boiling point of water at sea level?"],
        pace=0,
        playback_wav=tmp_path / "answer.wav",
    )
    result = await native.replay_native_application(
        args,
        RuntimeOptions(engine="native", research=ResearchOptions(False, 0)),
        upstream=ScriptedAnswer(answer_sentences, inter_sentence_delay),
    )
    assert native.NATIVE_REPLAY_SETTLE_TIMEOUT_SECONDS == 60
    assert 60 in settle_timeouts
    assert speech_timeouts == [45 if first_tts_startup_delay else 60]
    assert result["asr_transcription"] == "What is the boiling point of water at sea level?"
    assistant_text = " ".join(
        row["content"]
        for row in result["history"]
        if row.get("role") == "assistant" and isinstance(row.get("content"), str)
    )
    assert all(sentence in assistant_text for sentence in answer_sentences)
    assert "turn_completed" in result["completion_events"]
    assert sum(event["event"] == "tts_start" for event in result["events"]) == len(answer_sentences)
    assert sum(event["event"] == "tts_released" for event in result["events"]) == len(
        answer_sentences
    )
    summaries = [e for e in result["events"] if e["event"] == "tts_segment_summary"]
    assert len(summaries) == len(answer_sentences)
    assert all(e["status"] == "completed" for e in summaries)
    assert all(e["native_audio_seconds"] == pytest.approx(0.128) for e in summaries)
    assert all(
        e["diagnostics"]["status"] == "completed"
        and e["diagnostics"]["client_mcp"]["tts_next_count"] == 5
        for e in summaries
    )
    assert sum(e["event"] == "tts_response_started" for e in result["events"]) == 1
    assert sum(e["event"] == "tts_response_stop_queued" for e in result["events"]) == 1
    output_speaking_transitions = 2 if second_tts_startup_delay >= 3 else 1
    assert (
        sum(e["event"] == "output_speaking_started" for e in result["events"])
        == output_speaking_transitions
    )
    assert (
        sum(e["event"] == "output_speaking_stopped" for e in result["events"])
        == output_speaking_transitions
    )
    assert sum(e["event"] == "assistant_turn_started" for e in result["events"]) == 1
    assert sum(e["event"] == "assistant_turn_stopped" for e in result["events"]) == 1
    if second_tts_startup_delay >= 3:
        segment_two_start = next(
            e for e in result["events"] if e["event"] == "tts_start" and e.get("segment_id") == 2
        )
        segment_two_pcm = next(
            e
            for e in result["events"]
            if e["event"] == "tts_pcm" and e["sequence"] > segment_two_start["sequence"]
        )
        output_stops = [e for e in result["events"] if e["event"] == "output_speaking_stopped"]
        output_starts = [e for e in result["events"] if e["event"] == "output_speaking_started"]
        response_stop = next(
            e for e in result["events"] if e["event"] == "tts_response_stop_queued"
        )
        assert segment_two_start["sequence"] < output_stops[0]["sequence"]
        assert output_stops[0]["sequence"] < segment_two_pcm["sequence"]
        assert segment_two_pcm["sequence"] < output_starts[1]["sequence"]
        assert output_starts[1]["sequence"] < response_stop["sequence"]
    assert (
        next(e for e in result["events"] if e["event"] == "assistant_turn_stopped")["projection"]
        == "listening"
    )
    assert not any(
        e["event"] == "assistant_turn_stopped" and e["interrupted"] for e in result["events"]
    )
    event_names = [event["event"] for event in result["events"]]
    assert event_names.index("turn_completed") > max(
        index for index, name in enumerate(event_names) if name == "tts_pcm"
    )
    assert (
        event_names.index("tts_response_stop_queued")
        < max(index for index, name in enumerate(event_names) if name == "output_speaking_stopped")
        < event_names.index("turn_completed")
    )
    assert result["tts_audio"]["pcm16_valid"] and result["tts_audio"]["wav_valid"]
    assert result["request_ledger"] == {
        "run_id": "native-replay-test-01",
        "requests": [],
        "counts": {},
    }
    assert set(result["provider_budget"]) == {"asr", "llm", "tts"}
    assert all(not spec["allow_live"] for spec in result["provider_budget"].values())
    with wave.open(str(args.playback_wav), "rb") as rendered:
        assert rendered.getnchannels() == 1 and rendered.getsampwidth() == 2
        assert rendered.getframerate() == 16000 and rendered.getnframes() > 0
    assert not any(
        event["event"] == "tool_completed"
        and event.get("tool_name") in {"web_search", "retrieve_source"}
        for event in result["events"]
    )
    assert result["latency_seconds"]["turn_started_to_completed"] is not None
    assert "offline-test-secret" not in repr(result)
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []


@pytest.mark.parametrize(
    "delay_mode",
    [
        "baseline",
        "jitter",
        "append_ack_200",
        "append_ack_600",
        "append_ack_900",
        "append_ack_1200",
        "append_ack_1500",
    ],
)
async def test_labeled_nine_second_native_replay_with_delayed_mcp_appends(
    tmp_path, monkeypatch, delay_mode
):
    import wave
    import time
    import me3_voice.native_app as native
    from me3_voice.configuration import RuntimeOptions
    from me3_voice.research import ResearchOptions
    from me3_voice.speech import SpeechClient

    source = Path("var/results/native-live-smoke/boiling-question-16k.wav")
    assert source.is_file()
    with wave.open(str(source), "rb") as wav:
        assert wav.getnframes() / wav.getframerate() == 9.0

    class JitterSpeechClient(SpeechClient):
        sequences = []

        async def call(self, kind, name, arguments):
            if kind == "asr" and name in {"asr_append", "asr_append_batch"}:
                sequence = arguments["sequence"]
                self.sequences.append(sequence)
                if delay_mode == "jitter":
                    delay = {7: 0.18, 31: 0.24, 68: 0.16}.get(
                        sequence, 0.012 + (sequence % 3) * 0.004
                    )
                    await asyncio.sleep(delay)
                elif delay_mode.startswith("append_ack_") and sequence == 0:
                    await asyncio.sleep(int(delay_mode.rsplit("_", 1)[1]) / 1000)
            return await super().call(kind, name, arguments)

    monkeypatch.setattr(native, "SpeechClient", JitterSpeechClient)
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "offline-labeled-replay-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "requests.sqlite"))

    class Answer:
        async def respond(self, _history):
            yield "At sea level, water boils at 100 degrees Celsius."
            yield "That is 212 degrees Fahrenheit."

    args = Namespace(
        onit_python=None,
        direct_clock=False,
        turn_policy="silence",
        processed_wav=None,
        wav=source,
        output=tmp_path / f"{delay_mode}.json",
        trace=tmp_path / f"{delay_mode}.jsonl",
        backend_config=None,
        secrets_file=None,
        llm_secrets_file=None,
        wake_frame=0,
        transcript=True,
        mock_transcripts=["What is the boiling point of water at sea level?"],
        wake_model=None,
        mel_model=None,
        embedding_model=None,
        pace=1,
        playback_wav=tmp_path / f"{delay_mode}.wav",
    )
    started = time.perf_counter()
    try:
        result = await native.replay_native_application(
            args,
            RuntimeOptions(engine="native", research=ResearchOptions(False, 0)),
            upstream=Answer(),
        )
    except RuntimeError:
        if not delay_mode.startswith(("append_ack_1200", "append_ack_1500")):
            raise
        trace_rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
        saturation = next(e for e in trace_rows if e["event"] == "asr_append_queue_saturated")
        assert saturation["queue_limit"] == 40
        assert saturation["depth"] == 40
        speech_start = next(e for e in trace_rows if e["event"] == "vad_speech_started")
        assert speech_start["onset_buffer_frames"] == 8
        assert sum(e["event"] == "vad_speech_started" for e in trace_rows) == 1
        speech_stop = next(e for e in trace_rows if e["event"] == "vad_speech_stopped")
        aborted = next(e for e in trace_rows if e["event"] == "asr_turn_aborted")
        failed = next(e for e in trace_rows if e["event"] == "replay_failed")
        assert speech_start["utterance_id"] == speech_stop["utterance_id"] == 1
        assert speech_stop["at"] > saturation["at"]
        assert aborted["error_code"] == failed["error_code"] == "queue_saturated"
        assert failed["error_stage"] == "asr_append"
        assert failed["failure_origin_event"] == "asr_turn_aborted"
        # Eight preserved onset frames leave about one second of post-VAD
        # headroom; a sustained 1.2s stall still fails explicitly and bounded.
        assert 0.95 <= saturation["at"] - speech_start["at"] <= 1.3
        return
    if delay_mode in {"append_ack_1200", "append_ack_1500"}:
        pytest.fail("append acknowledgement delay unexpectedly exceeded the bounded queue")
    elapsed = time.perf_counter() - started
    assert result["asr_transcription"] == "What is the boiling point of water at sea level?"
    assert "100 degrees Celsius" in result["llm_response"]
    assert "turn_completed" in result["completion_events"]
    assert result["tts_audio"]["wav_valid"] and result["tts_audio"]["pcm16_valid"]
    assert result["request_ledger"]["requests"] == []
    if delay_mode in {"jitter", "append_ack_200", "append_ack_600", "append_ack_900"}:
        assert JitterSpeechClient.sequences == list(range(len(JitterSpeechClient.sequences)))
        assert len(JitterSpeechClient.sequences) > 1
        assert not any(e.get("event") == "asr_append_queue_saturated" for e in result["events"])
        assert sum(e["event"] == "asr_opened" for e in result["events"]) == 1
        assert sum(e["event"] == "vad_speech_started" for e in result["events"]) == 1
        assert not any(
            e.get("idle_watchdog_triggered")
            for e in result["events"]
            if e["event"] == "vad_speech_stopped"
        )
        assert all(
            e.get("queue_depth", 0) <= e.get("queue_limit", 32)
            for e in result["events"]
            if e["event"] in {"asr_append_queued", "asr_append_queue_saturated"}
        )
        assert any(e["event"] == "asr_flush_completed" for e in result["events"])
        assert (
            max(
                e.get("queue_depth", 0)
                for e in result["events"]
                if e["event"] in {"asr_append_queued", "asr_append_started"}
            )
            <= 40
        )
    assert elapsed >= 9.0


async def test_failed_native_replay_persists_sanitized_partial_trace(tmp_path, monkeypatch):
    import wave
    import me3_voice.native_app as native
    from me3_voice.configuration import RuntimeOptions
    from me3_voice.remote import ProviderError
    from me3_voice.research import ResearchOptions
    from me3_voice.speech import SpeechClient

    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        # Two quiet frames make the test exercise the same VAD-stop error
        # boundary used by a completed spoken turn.
        writer.writeframes(b"\1\0" * 2048 + b"\0\0" * 1024)

    class FailedAppend(SpeechClient):
        async def call(self, kind, name, arguments):
            if kind == "asr" and name in {"asr_append", "asr_append_batch"}:
                raise ProviderError(
                    "provider diagnostic contains private-value",
                    code="provider_timeout",
                    role="asr",
                    error_stage="asr_append",
                    error_boundary="deepgram_websocket",
                    error_detail="TimeoutError",
                )
            return await super().call(kind, name, arguments)

    monkeypatch.setattr(native, "SpeechClient", FailedAppend)
    monkeypatch.setattr(native, "native_vad", FixtureVAD)
    monkeypatch.setattr("me3_voice.readiness.build_wake", lambda _args: None)
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "partial-trace-test-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "ledger.sqlite"))
    trace = tmp_path / "failed.jsonl"
    args = Namespace(
        onit_python=None,
        direct_clock=False,
        turn_policy="silence",
        processed_wav=None,
        wav=source,
        output=tmp_path / "result.json",
        trace=trace,
        backend_config=None,
        secrets_file=None,
        llm_secrets_file=None,
        wake_frame=0,
        transcript=False,
        mock_transcripts=["offline question"],
        wake_model=None,
        mel_model=None,
        embedding_model=None,
        pace=0,
        playback_wav=None,
    )
    with pytest.raises(RuntimeError):
        await native.replay_native_application(
            args,
            RuntimeOptions(engine="native", research=ResearchOptions(False, 0)),
        )
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(row["event"] == "asr_append_failed" for row in rows)
    failure = next(row for row in rows if row["event"] == "replay_failed")
    assert failure["error_code"] == "provider_timeout"
    assert failure["error_stage"] == "asr_append"
    assert failure["error_boundary"] == "deepgram_websocket"
    assert failure["provider_exception_type"] == "TimeoutError"
    assert failure["audio_frames_delivered"] >= 1
    assert "private-value" not in trace.read_text()


async def test_research_schemas_follow_live_permission_and_remaining_budget():
    from me3_voice.configuration import RuntimeOptions
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.playback import MemorySink
    from me3_voice.research import ResearchClient, ResearchOptions

    research = ResearchClient(ResearchOptions(False, 0))
    state = NativeState(research=research)
    try:
        async with NativeApplication(
            state,
            MemorySink(),
            annotated=True,
            options=RuntimeOptions(engine="native"),
        ) as app:

            def names():
                return {schema.name for schema in state.context.tools.standard_tools}

            assert not names() & {"web_search", "retrieve_source", "verify_evidence"}

            research.options = ResearchOptions(True, 1)
            app.llm.refresh_tools(state.context)
            assert {"web_search", "retrieve_source"} <= names()
            assert "verify_evidence" not in names()  # No retained source yet.

            research.sources["S1"] = {"text": "fixture evidence"}
            research.requests = 1
            app.llm.refresh_tools(state.context)
            assert not names() & {"web_search", "retrieve_source"}
            assert "verify_evidence" in names()  # Local check remains usable.
    finally:
        await state.close()


async def test_native_provider_budget_survives_reconnect_without_sdk_retries(monkeypatch):
    import tempfile
    from pathlib import Path
    from me3_voice.native_llm import NativeLLM
    from me3_voice.remote import LiveTransport, ProviderError
    from me3_voice.configuration import BackendSpec
    from pipecat.processors.aggregators.llm_context import LLMContext

    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        model="openai/gpt-oss-120b",
        auth_env="ME3_TEST_ONLY",
        allow_live=True,
        request_limit=1,
        max_input_tokens=32000,
        max_input_tokens_total=64000,
    )
    credentials = {"ME3_TEST_ONLY": "offline-fixture-not-a-real-key"}
    ledger = Path(tempfile.mkdtemp()) / "provider.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "native-test-run-01")
    budget = LiveTransport(spec, credentials)
    calls = []

    from types import SimpleNamespace

    for index in (0, 1):
        llm = NativeLLM(spec, credentials=credentials, budget=budget)

        async def create(**_params):
            calls.append(1)
            from openai.types.chat import ChatCompletionChunk

            async def chunks():
                yield ChatCompletionChunk(
                    id="fixture",
                    object="chat.completion.chunk",
                    created=0,
                    model="fixture",
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": "fixture"},
                            "finish_reason": "stop",
                        }
                    ],
                )

            return chunks()

        async def close():
            return None

        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=close,
            max_retries=0,
        )
        try:
            assert llm._client.max_retries == 0 and not llm._retry_on_timeout
            assert llm._settings.max_completion_tokens == 2000
            if index:
                with pytest.raises(ProviderError):
                    await llm.get_chat_completions(LLMContext())
            else:
                stream = await llm.get_chat_completions(LLMContext())
                assert [c.choices[0].delta.content async for c in stream] == ["fixture"]
        finally:
            await llm._client.close()
    assert calls == [1] and budget.requests == 1
    import sqlite3

    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT status FROM provider_requests").fetchone() == ("success",)


async def test_native_stream_failure_and_cancellation_update_ledger_conservatively(
    tmp_path, monkeypatch
):
    import asyncio
    import sqlite3
    from me3_voice.native_llm import NativeLLM
    from me3_voice.remote import LiveTransport, ProviderError
    from me3_voice.configuration import BackendSpec

    ledger = tmp_path / "provider.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "stream-test-run-01")
    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://llm.test/v1/chat/completions",
        model="test-model",
        auth_env="TEST_KEY",
        allow_live=True,
        request_limit=3,
        max_input_tokens=1000,
        max_input_tokens_total=3000,
    )
    budget = LiveTransport(spec, {"TEST_KEY": "offline-test-key"})

    class Tracker:
        safe_error = staticmethod(NativeLLM.safe_error)

    tracker = Tracker()
    tracker.budget = budget
    rows = []

    async def broken():
        yield "partial"
        raise ProviderError("offline failure", code="provider_timeout")

    _, request_id = budget.authorize(
        input_tokens=1, input_token_limit=1000, input_tokens_total_limit=3000
    )
    with pytest.raises(ProviderError):
        async for _ in NativeLLM._track_ledger_stream(tracker, broken(), request_id):
            pass
    with sqlite3.connect(ledger) as db:
        rows.append(
            db.execute(
                "SELECT status,error_code FROM provider_requests ORDER BY created"
            ).fetchall()
        )

    async def delayed():
        yield "partial"
        await asyncio.sleep(30)

    _, request_id = budget.authorize(
        input_tokens=1, input_token_limit=1000, input_tokens_total_limit=3000
    )

    async def consume():
        async for _ in NativeLLM._track_ledger_stream(tracker, delayed(), request_id):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with sqlite3.connect(ledger) as db:
        rows.append(
            db.execute(
                "SELECT status,error_code FROM provider_requests ORDER BY created"
            ).fetchall()
        )
    assert rows[0] == [("failure", "provider_timeout")]
    assert rows[1] == [("failure", "provider_timeout"), ("dispatched_unknown", None)]


async def test_native_llm_serialized_input_budget_rejects_before_reservation_or_sdk_dispatch(
    tmp_path, monkeypatch
):
    import sqlite3
    from types import SimpleNamespace
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.configuration import BackendSpec
    from me3_voice.native_llm import NativeLLM
    from me3_voice.remote import LiveTransport

    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "native-budget-test-01")
    ledger = tmp_path / "provider-requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    base = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://llm.example.test/v1/chat/completions",
        model="test-model",
        auth_env="ME3_TEST_BUDGET_KEY",
        allow_live=True,
        request_limit=5,
        max_input_tokens=129,
        max_input_tokens_total=512,
    )
    credentials = {"ME3_TEST_BUDGET_KEY": "offline-only-test-key"}
    budget = LiveTransport(base, credentials)
    llm = NativeLLM(base, credentials=credentials, budget=budget)
    calls = []

    async def forbidden_dispatch(**_params):
        calls.append("dispatched")
        raise AssertionError("a request over the input budget must not reach the SDK")

    async def close():
        return None

    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden_dispatch)),
        close=close,
        max_retries=0,
    )
    events = []
    llm.on_event = lambda name, **fields: events.append((name, fields))
    try:
        stream = await llm.get_chat_completions(LLMContext())
        # Consume the framework's explicit safe failure, not a provider stream.
        _ = [chunk async for chunk in stream]
    finally:
        await llm._client.close()
    assert calls == [] and budget.requests == 0
    assert any(name == "llm_input_budget_rejected" for name, _ in events)
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT COUNT(*) FROM provider_requests").fetchone()[0] == 0
        assert db.execute(
            "SELECT input_tokens_per_request_limit,input_tokens_limit FROM provider_campaign_caps"
        ).fetchone() == (129, 512)


async def test_native_wake_standby_does_not_submit_background_audio_to_asr():
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.playback import MemorySink
    from me3_voice.configuration import RuntimeOptions

    class NeverWake:
        frame_samples = 1280

        def feed(self, _pcm):
            return {"triggered": False}

        def end_session(self):
            pass

    state = NativeState()
    try:
        async with NativeApplication(
            state, MemorySink(), options=RuntimeOptions(engine="native"), wake=NeverWake()
        ) as app:
            # A bounded interval of audible but unrecognized background audio.
            for sequence in range(40):
                await app.feed(b"\x00\x20" * 512, sequence)
                await app.drain()
            assert not app.gate.active
            assert app.parts.stt.operation is None
            assert not app.parts.stt.onset_buffer
            assert not any(event["event"] == "asr_open_started" for event in app.events)
    finally:
        await state.close()


async def test_explicit_stop_ends_context_while_disconnect_preserves_it(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            first = (await next_json(ws, "ready"))["session_id"]
            await send_turn(ws)
            await wait_answer(ws)
            await ws.send_json({"type": "stop"})
            while (await ws.receive()).type not in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                pass
        await asyncio.sleep(0.05)
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            assert (await next_json(ws, "ready"))["session_id"] != first
            view = (await next_json(ws, "transcript"))["data"]
            assert not view["messages"]


async def test_dead_speech_owner_fails_closed_without_budget_reset():
    from me3_voice.native_app import NativeState

    state = NativeState()
    try:
        await state.start()
        state.owner.cancel()
        await asyncio.gather(state.owner, return_exceptions=True)
        with pytest.raises(RuntimeError, match="restart bridge"):
            await state.start()
    finally:
        await state.close()


async def test_native_dashboard_protocol_and_optional_telemetry(tmp_path, monkeypatch, native_args):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        response = await client.get(server.make_url("/api/voice/capabilities"))
        capabilities = await response.json()
        assert capabilities["available"] and capabilities["engine"] == "native"
        assert capabilities["devices"]["target"] == "mock"
        assert capabilities["devices"]["status"] == "simulated"
        assert not capabilities["music"]["configured"]
        assert (await client.get(server.make_url("/api/snapshot"))).status == 200
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            ready = await next_json(ws, "ready")
            status = await next_json(ws, "status")
            assert status["run_id"] == ready["run_id"]
            assert isinstance(status["turn_id"], int) and status["status_sequence"] > 0
            assert status["event_sequence"] >= 0
            await ws.send_json({"type": "music_ui_control", "action": "pause", "outcome": "failed"})
            await ws.send_json(
                {
                    "type": "music_playback_observed",
                    "status": "not_playing",
                    "source": "spotify_api",
                    "track_changed": False,
                }
            )
            event = await next_json(ws, "runtime_event")
            assert event["sequence"] == event["data"]["sequence"]
            assert event["data"]["session_id"] == ready["session_id"]
            action = {
                "type": "device_ui_action",
                "command_id": "ui-0123456789abcdef0123456789abcdef",
                "action": "light_on",
            }
            await ws.send_json(action)
            first = await next_json(ws, "device_action_ack")
            assert first["status"] == "accepted"
            await ws.send_json(action)
            assert (await next_json(ws, "device_action_ack"))["status"] == "duplicate"
            await send_turn(ws)
            assert (await wait_answer(ws))["messages"]


async def test_native_capability_reports_engine_and_shared_contract_fields(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        response = await client.get(server.make_url("/api/voice/capabilities"))
        assert response.status == 200
        capability = await response.json()
        assert capability["engine"] == "native"
        assert capability["available"] is True
        assert capability["wake"] == "configured model"
        assert capability["capture"] == "browser-pcm16"
        assert capability["playback"] == "browser-acknowledged-pcm16"
        assert capability["aec"] == "browser-requested-unverified"
        assert "asr_partial" in capability
        assert capability["task_artifact_access"] is False
        assert capability["devices"]["target"] == "mock"
        assert capability["devices"]["status"] == "simulated"
        assert capability["music"]["configured"] is False
        assert capability["music"]["output_backend"] == "unavailable"
